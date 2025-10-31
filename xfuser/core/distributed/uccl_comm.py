# Copyright 2024 xDiT team.
"""
UCCL P2P Communication Wrapper for xDiT

This module provides a drop-in replacement for torch.distributed send/recv operations
using UCCL Collective API (from uccl import collective). It automatically handles:
- Connection management between ranks (local IPC and remote RDMA)
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
from typing import Optional, Any
import torch
import torch.distributed as dist

# Try to import UCCL Collective API
try:
    from uccl import collective
    UCCL_AVAILABLE = True
except ImportError as e:
    UCCL_AVAILABLE = False
    _import_error = str(e)


class UCCLCommWrapper:
    """
    Wrapper class for UCCL P2P communication using the collective API.
    
    This class wraps the UCCL collective API, providing methods compatible with
    torch.distributed's send/recv API. The collective API automatically handles:
    - Local IPC connections for same-node transfers
    - Remote RDMA connections for cross-node transfers
    - Memory registration only when needed (for remote connections)
    """
    
    def __init__(self):
        self._enabled = False
        self._initialized = False
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
        Initialize UCCL collective context and establish connections with all ranks.
        
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
        
        try:
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()
            
            # Determine local GPU index
            if local_gpu_idx is not None:
                self._local_gpu_idx = local_gpu_idx
            else:
                # Try to get from environment
                self._local_gpu_idx = int(os.environ.get("LOCAL_RANK", self._rank))
            
            # Initialize UCCL collective context
            # Use disable_uccl_intra=True to use NCCL for intra-node, UCCL only for inter-node RDMA
            collective.init_collective(
                num_cpus=self._num_cpus, 
                local_gpu_idx=self._local_gpu_idx,
                disable_uccl_intra=True
            )
            
            self._initialized = True
            self._enabled = True
            print(f"[Rank {self._rank}] UCCL collective communication initialized")
            
        except Exception as e:
            warnings.warn(f"Failed to initialize UCCL collective: {e}. Falling back to torch.distributed.")
            self._use_uccl = False
            self._enabled = False
    
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
        
        collective.send(tensor, dst)
    
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
        
        collective.recv(tensor, src)
    
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
        
        # Ensure tensor is contiguous and synchronized
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        
        # Synchronize CUDA stream to ensure tensor data is ready
        if tensor.is_cuda:
            torch.cuda.synchronize(tensor.device)
        
        # Register tensor for RDMA if needed (collective API handles this intelligently)
        collective.register_tensor(tensor)
        transfer_id_or_p2pop = collective.isend(tensor, dst)
        
        # When disable_uccl_intra=True, local transfers return P2POp, remote return int
        if isinstance(transfer_id_or_p2pop, int):
            print(f"[Rank {self._rank}] UCCL RDMA SEND: {tensor.numel()} elements ({tensor.element_size() * tensor.numel() / 1024 / 1024:.2f} MB) to rank {dst}")
            return UCCLTransferHandle(transfer_id_or_p2pop, tensor)
        else:
            # P2POp needs to be batched and executed to get a Work handle
            # Execute it immediately as a single-op batch
            # batch_isend_irecv returns a list of Work objects
            print(f"[Rank {self._rank}] NCCL LOCAL SEND: {tensor.numel()} elements to rank {dst}")
            work_list = dist.batch_isend_irecv([transfer_id_or_p2pop])
            return work_list[0]
    
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
        
        # Ensure tensor is contiguous
        if not tensor.is_contiguous():
            raise RuntimeError("Receive tensor must be contiguous")
        
        # Register tensor for RDMA if needed (collective API handles this intelligently)
        collective.register_tensor(tensor)
        transfer_id_or_p2pop = collective.irecv(tensor, src)
        
        # When disable_uccl_intra=True, local transfers return P2POp, remote return int
        if isinstance(transfer_id_or_p2pop, int):
            print(f"[Rank {self._rank}] UCCL RDMA RECV: {tensor.numel()} elements ({tensor.element_size() * tensor.numel() / 1024 / 1024:.2f} MB) from rank {src}")
            return UCCLTransferHandle(transfer_id_or_p2pop, tensor)
        else:
            # P2POp needs to be batched and executed to get a Work handle
            # Execute it immediately as a single-op batch
            # batch_isend_irecv returns a list of Work objects
            print(f"[Rank {self._rank}] NCCL LOCAL RECV: {tensor.numel()} elements from rank {src}")
            work_list = dist.batch_isend_irecv([transfer_id_or_p2pop])
            return work_list[0]
    
    def finalize(self):
        """Clean up UCCL resources."""
        if self._enabled and self._initialized:
            collective.finalize_collective()
            self._initialized = False
            self._enabled = False
            print(f"[Rank {self._rank}] UCCL collective communication finalized")


class UCCLTransferHandle:
    """Handle for UCCL transfers that mimics torch.distributed.Work interface."""
    
    def __init__(self, transfer_id: int, tensor: torch.Tensor = None):
        self.transfer_id = transfer_id
        self._tensor = tensor  # Keep reference to prevent GC during async transfer
    
    def wait(self):
        """Wait for the transfer to complete."""
        collective.wait(self.transfer_id)
        # Ensure GPU sees the completed data (especially important for receives)
        if self._tensor is not None and self._tensor.is_cuda:
            torch.cuda.synchronize(self._tensor.device)
    
    def is_completed(self):
        """Check if the transfer has completed."""
        return collective.test(self.transfer_id)


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

