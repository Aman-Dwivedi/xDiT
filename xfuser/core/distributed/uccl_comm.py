# Copyright 2024 xDiT team.
"""
UCCL P2P Communication Wrapper for xDiT

This module provides a drop-in replacement for torch.distributed send/recv operations
using UCCL P2P's low-level API (from uccl import p2p). It automatically handles:
- Connection management between ranks
- Memory registration for RDMA operations  
- Metadata exchange via torch.distributed
- Synchronous and asynchronous transfers
- Seamless fallback to torch.distributed when UCCL is not available

Environment Variables:
    USE_UCCL_P2P: Set to "1" to enable UCCL P2P communication (default: "0")
    UCCL_NUM_CPUS: Number of CPU threads for RDMA operations (default: 4)
"""

from __future__ import annotations
import os
import warnings
from typing import Optional, Dict, Any
import torch
import torch.distributed as dist

# Try to import UCCL P2P API
try:
    from uccl import p2p
    UCCL_AVAILABLE = True
except ImportError as e:
    UCCL_AVAILABLE = False
    _import_error = str(e)


class UCCLCommWrapper:
    """
    Wrapper class for UCCL P2P communication using low-level p2p.Endpoint API.
    
    This class manages UCCL p2p endpoints and connections, providing methods compatible with
    torch.distributed's send/recv API. It automatically handles memory registration
    and connection establishment between all ranks.
    """
    
    def __init__(self):
        self._enabled = False
        self._initialized = False
        self._endpoint: Optional[Any] = None  # p2p.Endpoint
        self._connections: Dict[int, int] = {}  # rank -> conn_id
        self._memory_regions: Dict[int, int] = {}  # tensor ptr -> mr_id
        self._use_uccl = os.environ.get("USE_UCCL_P2P", "0") == "1"
        self._num_cpus = int(os.environ.get("UCCL_NUM_CPUS", "4"))
        self._local_gpu_idx: Optional[int] = None
        self._rank = -1
        self._world_size = -1
        
        if self._use_uccl and not UCCL_AVAILABLE:
            warnings.warn(
                f"USE_UCCL_P2P is set but UCCL is not available. "
                f"Import error: {_import_error}. "
                f"Falling back to torch.distributed."
            )
            self._use_uccl = False
    
    def is_enabled(self) -> bool:
        """Check if UCCL P2P is enabled and available."""
        return self._use_uccl and UCCL_AVAILABLE
    
    def initialize(self, local_gpu_idx: Optional[int] = None):
        """
        Initialize UCCL p2p endpoint and establish connections with all ranks.
        
        Args:
            local_gpu_idx: Optional GPU index. If None, derived from torch.distributed.
        """
        if not self.is_enabled():
            return
        
        if self._initialized:
            warnings.warn("UCCL communication already initialized")
            return
        
        if not dist.is_initialized():
            raise RuntimeError(
                "torch.distributed must be initialized before UCCL communication"
            )
        
        # UCCL can work with any backend, but we need gloo for CPU group operations
        # We'll create a separate gloo group for metadata exchange if needed
        
        try:
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()
            
            # Determine local GPU index
            if local_gpu_idx is not None:
                self._local_gpu_idx = local_gpu_idx
            else:
                # Try to get from environment
                self._local_gpu_idx = int(os.environ.get("LOCAL_RANK", self._rank))
            
            # Create UCCL endpoint
            self._endpoint = p2p.Endpoint(self._local_gpu_idx, self._num_cpus)
            local_metadata = self._endpoint.get_metadata()
            
            # Exchange metadata with all ranks via torch.distributed
            # If using NCCL backend, we need to move tensors to GPU for collective ops
            backend = dist.get_backend()
            all_metadata = [None] * self._world_size
            
            if backend == "nccl":
                # NCCL requires GPU tensors
                metadata_tensor = torch.ByteTensor(list(local_metadata)).cuda()
                gathered_tensors = [
                    torch.zeros_like(metadata_tensor) for _ in range(self._world_size)
                ]
                dist.all_gather(gathered_tensors, metadata_tensor)
                # Move back to CPU and convert
                for i, tensor in enumerate(gathered_tensors):
                    all_metadata[i] = bytes(tensor.cpu().tolist())
            else:
                # Gloo and other backends work with CPU tensors
                metadata_tensor = torch.ByteTensor(list(local_metadata))
                gathered_tensors = [
                    torch.zeros_like(metadata_tensor) for _ in range(self._world_size)
                ]
                dist.all_gather(gathered_tensors, metadata_tensor)
                for i, tensor in enumerate(gathered_tensors):
                    all_metadata[i] = bytes(tensor.tolist())
            
            # Establish connections with other ranks
            # Strategy: Build full mesh by having each rank connect to higher ranks
            # and accept from lower ranks. This ensures no deadlocks.
            # For world_size=2: rank 0 connects to rank 1, rank 1 accepts from rank 0
            
            # First, count how many connections we need
            num_connects = self._world_size - self._rank - 1  # connect to higher ranks
            num_accepts = self._rank  # accept from lower ranks
            
            print(f"[Rank {self._rank}] Establishing connections: will connect to {num_connects} higher ranks, accept from {num_accepts} lower ranks")
            
            # Phase 1: Connect to all higher-ranked peers (non-blocking initiates)
            pending_connects = []
            for peer_rank in range(self._rank + 1, self._world_size):
                ip, port, gpu_idx = p2p.Endpoint.parse_metadata(all_metadata[peer_rank])
                ok, conn_id = self._endpoint.connect(ip, gpu_idx, remote_port=port)
                if not ok:
                    raise RuntimeError(f"[Rank {self._rank}] Failed to connect to rank {peer_rank}")
                self._connections[peer_rank] = conn_id
                pending_connects.append(peer_rank)
                print(f"[Rank {self._rank}] Connected to rank {peer_rank} at {ip}:{port} (conn_id={conn_id})")
            
            # Phase 2: Accept connections from all lower-ranked peers
            for _ in range(num_accepts):
                ok, r_ip, r_gpu, conn_id = self._endpoint.accept()
                if not ok:
                    raise RuntimeError(f"[Rank {self._rank}] Failed to accept connection")
                
                # Find which rank this connection is from by matching IP/GPU
                peer_rank = None
                for candidate_rank in range(self._rank):
                    candidate_ip, _, candidate_gpu = p2p.Endpoint.parse_metadata(all_metadata[candidate_rank])
                    if candidate_ip == r_ip and candidate_gpu == r_gpu:
                        peer_rank = candidate_rank
                        break
                
                if peer_rank is None:
                    raise RuntimeError(f"[Rank {self._rank}] Accepted connection from unknown peer {r_ip}:{r_gpu}")
                
                self._connections[peer_rank] = conn_id
                print(f"[Rank {self._rank}] Accepted connection from rank {peer_rank} (conn_id={conn_id})")
            
            # Verify we have the right number of connections
            expected_connections = self._world_size - 1
            if len(self._connections) != expected_connections:
                raise RuntimeError(
                    f"[Rank {self._rank}] Expected {expected_connections} connections but got {len(self._connections)}"
                )
            
            self._initialized = True
            self._enabled = True
            print(f"[Rank {self._rank}] UCCL P2P communication initialized with {len(self._connections)} connections")
            
        except Exception as e:
            warnings.warn(f"Failed to initialize UCCL P2P: {e}. Falling back to torch.distributed.")
            self._use_uccl = False
            self._enabled = False
    
    def _register_tensor(self, tensor: torch.Tensor) -> int:
        """
        Register a tensor for UCCL communication and return memory region ID.
        
        Args:
            tensor: Tensor to register
            
        Returns:
            Memory region ID
        """
        ptr = tensor.data_ptr()
        if ptr in self._memory_regions:
            return self._memory_regions[ptr]
        
        size = tensor.numel() * tensor.element_size()
        ok, mr_id = self._endpoint.reg(ptr, size)
        if not ok:
            raise RuntimeError(f"Failed to register memory at {ptr} with size {size}")
        
        self._memory_regions[ptr] = mr_id
        return mr_id
    
    def _get_or_register_tensor(self, tensor: torch.Tensor) -> int:
        """Get memory region ID for tensor, registering if necessary."""
        ptr = tensor.data_ptr()
        if ptr in self._memory_regions:
            return self._memory_regions[ptr]
        return self._register_tensor(tensor)
    
    def send(self, tensor: torch.Tensor, dst: int, group=None):
        """
        Send tensor to destination rank (synchronous).
        
        Args:
            tensor: Tensor to send
            dst: Destination rank (global rank)
            group: Process group (ignored for UCCL)
        """
        if not self._enabled:
            return dist.send(tensor, dst=dst, group=group)
        
        if dst not in self._connections:
            raise RuntimeError(f"No connection to rank {dst}")
        
        conn_id = self._connections[dst]
        mr_id = self._get_or_register_tensor(tensor)
        ptr = tensor.data_ptr()
        size = tensor.numel() * tensor.element_size()
        
        ok = self._endpoint.send(conn_id, mr_id, ptr, size)
        if not ok:
            raise RuntimeError(f"Failed to send tensor to rank {dst}")
    
    def recv(self, tensor: torch.Tensor, src: int, group=None):
        """
        Receive tensor from source rank (synchronous).
        
        Args:
            tensor: Tensor to receive into
            src: Source rank (global rank)
            group: Process group (ignored for UCCL)
        """
        if not self._enabled:
            return dist.recv(tensor, src=src, group=group)
        
        if src not in self._connections:
            raise RuntimeError(f"No connection from rank {src}")
        
        conn_id = self._connections[src]
        mr_id = self._get_or_register_tensor(tensor)
        ptr = tensor.data_ptr()
        size = tensor.numel() * tensor.element_size()
        
        ok = self._endpoint.recv(conn_id, mr_id, ptr, size)
        if not ok:
            raise RuntimeError(f"Failed to receive tensor from rank {src}")
    
    def isend(self, tensor: torch.Tensor, dst: int, group=None) -> Any:
        """
        Initiate asynchronous send (non-blocking).
        
        Args:
            tensor: Tensor to send
            dst: Destination rank (global rank)
            group: Process group (ignored for UCCL)
        
        Returns:
            Handle that can be waited on
        """
        if not self._enabled:
            return dist.isend(tensor, dst=dst, group=group)
        
        if dst not in self._connections:
            raise RuntimeError(f"No connection to rank {dst}")
        
        conn_id = self._connections[dst]
        mr_id = self._get_or_register_tensor(tensor)
        ptr = tensor.data_ptr()
        size = tensor.numel() * tensor.element_size()
        
        ok, transfer_id = self._endpoint.send_async(conn_id, mr_id, ptr, size)
        if not ok:
            raise RuntimeError(f"Failed to initiate async send to rank {dst}")
        
        return UCCLTransferHandle(transfer_id, self)
    
    def irecv(self, tensor: torch.Tensor, src: int, group=None) -> Any:
        """
        Initiate asynchronous receive (non-blocking).
        
        Args:
            tensor: Tensor to receive into
            src: Source rank (global rank)
            group: Process group (ignored for UCCL)
        
        Returns:
            Handle that can be waited on
        """
        if not self._enabled:
            return dist.irecv(tensor, src=src, group=group)
        
        if src not in self._connections:
            raise RuntimeError(f"No connection from rank {src}")
        
        conn_id = self._connections[src]
        mr_id = self._get_or_register_tensor(tensor)
        ptr = tensor.data_ptr()
        size = tensor.numel() * tensor.element_size()
        
        ok, transfer_id = self._endpoint.recv_async(conn_id, mr_id, ptr, size)
        if not ok:
            raise RuntimeError(f"Failed to initiate async recv from rank {src}")
        
        return UCCLTransferHandle(transfer_id, self)
    
    def wait(self, transfer_id: int):
        """
        Wait for asynchronous operation to complete.
        
        Args:
            transfer_id: Transfer ID returned by isend/irecv
        """
        if not self._enabled:
            return
        
        while True:
            ok, is_done = self._endpoint.poll_async(transfer_id)
            if not ok:
                raise RuntimeError(f"Error polling transfer {transfer_id}")
            if is_done:
                break
    
    def finalize(self):
        """Clean up UCCL resources."""
        if self._enabled and self._initialized:
            # Deregister all memory regions
            for mr_id in self._memory_regions.values():
                self._endpoint.dereg(mr_id)
            self._memory_regions.clear()
            
            # Clear connections
            self._connections.clear()
            
            # Endpoint will be cleaned up automatically
            self._endpoint = None
            self._initialized = False
            self._enabled = False
            print(f"[Rank {self._rank}] UCCL P2P communication finalized")


class UCCLTransferHandle:
    """Handle for UCCL asynchronous transfers, compatible with torch.distributed.Work API."""
    
    def __init__(self, transfer_id: int, comm_wrapper: UCCLCommWrapper):
        self._transfer_id = transfer_id
        self._comm_wrapper = comm_wrapper
    
    def wait(self):
        """Wait for the transfer to complete."""
        self._comm_wrapper.wait(self._transfer_id)
    
    def is_completed(self) -> bool:
        """Check if the transfer is completed."""
        # UCCL doesn't provide a non-blocking check, so we return False
        return False


# Global UCCL communication wrapper instance
_uccl_comm: Optional[UCCLCommWrapper] = None


def get_uccl_comm() -> UCCLCommWrapper:
    """Get the global UCCL communication wrapper instance."""
    global _uccl_comm
    if _uccl_comm is None:
        _uccl_comm = UCCLCommWrapper()
    return _uccl_comm


def init_uccl_comm(local_gpu_idx: Optional[int] = None):
    """
    Initialize UCCL communication.
    
    Args:
        local_gpu_idx: Optional GPU index. If None, derived from torch.distributed.
    """
    comm = get_uccl_comm()
    comm.initialize(local_gpu_idx)


def finalize_uccl_comm():
    """Finalize UCCL communication and clean up resources."""
    global _uccl_comm
    if _uccl_comm is not None:
        _uccl_comm.finalize()
        _uccl_comm = None


def is_uccl_enabled() -> bool:
    """Check if UCCL communication is enabled."""
    comm = get_uccl_comm()
    return comm.is_enabled()

