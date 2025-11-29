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
    ENABLE_P2P_PROFILING: Set to "1" to enable P2P communication profiling (default: "0")
    P2P_PROFILING_OUTPUT_DIR: Directory for profiling output (default: "./profiling_data")
"""

from __future__ import annotations
import os
import sys
import warnings
from typing import Optional, Any
import torch
import torch.distributed as dist
import threading

# Import profiling library (now inside xDiT)
try:
    # Get xDiT root directory (parent of xfuser)
    _xdit_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    _profiling_lib_path = os.path.join(_xdit_root, 'p2p_profiling_lib')
    
    # Add to path if not already there
    if _profiling_lib_path not in sys.path:
        sys.path.insert(0, _xdit_root)
    
    from p2p_profiling_lib import get_profiler
    P2P_PROFILING_AVAILABLE = True
except ImportError as e:
    P2P_PROFILING_AVAILABLE = False
    _profiling_import_error = str(e)

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
        profiler = get_profiler() if P2P_PROFILING_AVAILABLE else None
        comm_type = 'uccl' if self._enabled else 'nccl'
        
        if profiler:
            with profiler.profile_send(tensor, dst, comm_type=comm_type, sync_mode='sync'):
                if not self._enabled:
                    return dist.send(tensor, dst=dst, group=group)
                collective.send(tensor, dst)
        else:
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
        profiler = get_profiler() if P2P_PROFILING_AVAILABLE else None
        comm_type = 'uccl' if self._enabled else 'nccl'
        
        if profiler:
            with profiler.profile_recv(tensor, src, comm_type=comm_type, sync_mode='sync'):
                if not self._enabled:
                    return dist.recv(tensor, src=src, group=group)
                collective.recv(tensor, src)
        else:
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
        profiler = get_profiler() if P2P_PROFILING_AVAILABLE else None
        comm_type = 'uccl' if self._enabled else 'nccl'
        profile_ctx = None
        
        if profiler:
            profile_ctx = profiler.profile_send(tensor, dst, comm_type=comm_type, sync_mode='async')
            profile_ctx.__enter__()
        
        try:
            if not self._enabled:
                # Ensure tensor is contiguous and synchronized (matching UCCL behavior)
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()
                
                # Synchronize CUDA stream to ensure tensor data is ready
                if tensor.is_cuda:
                    torch.cuda.synchronize(tensor.device)
                
                handle = dist.isend(tensor, dst=dst, group=group)
                wrapped = ProfiledWorkHandle(handle, profile_ctx, tensor) if profile_ctx else handle
            else:
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
                    wrapped = UCCLTransferHandle(transfer_id_or_p2pop, tensor, profile_ctx)
                else:
                    # P2POp needs to be batched and executed to get a Work handle
                    # Execute it immediately as a single-op batch
                    # batch_isend_irecv returns a list of Work objects
                    print(f"[Rank {self._rank}] UCCL LOCAL SEND: {tensor.numel()} elements to rank {dst}")
                    if transfer_id_or_p2pop.op == dist.isend:
                       temp = dist.isend(transfer_id_or_p2pop.tensor, transfer_id_or_p2pop.peer, transfer_id_or_p2pop.group, transfer_id_or_p2pop.tag)
                    else:
                      raise RuntimeError("Dikkat")
                      work_list = dist.batch_isend_irecv([transfer_id_or_p2pop])
                    wrapped = ProfiledWorkHandle(temp, profile_ctx, tensor) if profile_ctx else temp

            # Auto-wait in background to record end-to-end profiling for fire-and-forget sends
            if profile_ctx is not None:
                auto_wait_sends = os.environ.get("P2P_PROFILE_AUTOWAIT_SEND", "1") == "1"
                if auto_wait_sends:
                    def _bg_wait(h):
                        try:
                            h.wait()
                        except Exception:
                            pass
                    threading.Thread(target=_bg_wait, args=(wrapped,), daemon=True).start()
            return wrapped
        except Exception as e:
            if profile_ctx:
                profile_ctx.__exit__(type(e), e, e.__traceback__)
            raise
    
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
        profiler = get_profiler() if P2P_PROFILING_AVAILABLE else None
        comm_type = 'uccl' if self._enabled else 'nccl'
        profile_ctx = None
        
        if profiler:
            profile_ctx = profiler.profile_recv(tensor, src, comm_type=comm_type, sync_mode='async')
            profile_ctx.__enter__()
        
        try:
            if not self._enabled:
                # Ensure tensor is contiguous (matching UCCL behavior)
                if not tensor.is_contiguous():
                    raise RuntimeError("Receive tensor must be contiguous")
                
                # Synchronize CUDA stream before receiving
                if tensor.is_cuda:
                    torch.cuda.synchronize(tensor.device)
                
                handle = dist.irecv(tensor, src=src, group=group)
                return ProfiledWorkHandle(handle, profile_ctx, tensor) if profile_ctx else handle
            
            # Ensure tensor is contiguous
            if not tensor.is_contiguous():
                raise RuntimeError("Receive tensor must be contiguous")
            
            # Register tensor for RDMA if needed (collective API handles this intelligently)
            collective.register_tensor(tensor)
            transfer_id_or_p2pop = collective.irecv(tensor, src)
            
            # When disable_uccl_intra=True, local transfers return P2POp, remote return int
            if isinstance(transfer_id_or_p2pop, int):
                print(f"[Rank {self._rank}] UCCL RDMA RECV: {tensor.numel()} elements ({tensor.element_size() * tensor.numel() / 1024 / 1024:.2f} MB) from rank {src}")
                return UCCLTransferHandle(transfer_id_or_p2pop, tensor, profile_ctx)
            else:
                # P2POp needs to be batched and executed to get a Work handle
                # Execute it immediately as a single-op batch
                # batch_isend_irecv returns a list of Work objects
                print(f"[Rank {self._rank}] UCCL LOCAL RECV: {tensor.numel()} elements from rank {src}")
                if transfer_id_or_p2pop.op == dist.irecv:
                  temp = dist.irecv(transfer_id_or_p2pop.tensor, transfer_id_or_p2pop.peer, transfer_id_or_p2pop.group, transfer_id_or_p2pop.tag)
                else:
                  raise RuntimeError("Wapisi Dikkat")
                  work_list = dist.batch_isend_irecv([transfer_id_or_p2pop])
                return ProfiledWorkHandle(temp, profile_ctx, tensor) if profile_ctx else temp
        except Exception as e:
            if profile_ctx:
                profile_ctx.__exit__(type(e), e, e.__traceback__)
            raise
    
    def finalize(self):
        """Clean up UCCL resources."""
        if self._enabled and self._initialized:
            collective.finalize_collective()
            self._initialized = False
            self._enabled = False
            print(f"[Rank {self._rank}] UCCL collective communication finalized")


class UCCLTransferHandle:
    """Handle for UCCL transfers that mimics torch.distributed.Work interface."""
    
    def __init__(self, transfer_id: int, tensor: torch.Tensor = None, profile_ctx=None):
        self.transfer_id = transfer_id
        self._tensor = tensor  # Keep reference to prevent GC during async transfer
        self._profile_ctx = profile_ctx  # Keep reference to profiling context
        self._waited = False
        self._wait_lock = threading.Lock()
    
    def wait(self):
        """Wait for the transfer to complete (idempotent)."""
        with self._wait_lock:
            if self._waited:
                return
            collective.wait(self.transfer_id)
            self._waited = True
        
        # Ensure GPU sees the completed data (especially important for receives)
        if self._tensor is not None and self._tensor.is_cuda:
            torch.cuda.synchronize(self._tensor.device)
        # Deregister the tensor for UCCL RDMA transfers after sync and before profiling completion
        if self._tensor is not None:
            try:
                collective.deregister_tensor(self._tensor)
            except Exception:
                pass
        # Record completion in profiler
        if self._profile_ctx is not None:
            self._profile_ctx.complete()
    
    def is_completed(self):
        """Check if the transfer has completed."""
        return collective.test(self.transfer_id)


class ProfiledWorkHandle:
    """Wrapper for torch.distributed.Work handle that records profiling completion."""
    
    def __init__(self, work, profile_ctx=None, tensor=None):
        self._work = work
        self._profile_ctx = profile_ctx
        self._tensor = tensor  # Keep reference to tensor for post-wait synchronization
        self._waited = False
        self._wait_lock = threading.Lock()
    
    def wait(self):
        """Wait for the work to complete (idempotent)."""
        with self._wait_lock:
            if self._waited:
                return
            self._work.wait()
            self._waited = True
        # Ensure GPU sees the completed data (matching UCCL behavior for fair comparison)
        if self._tensor is not None and self._tensor.is_cuda:
            torch.cuda.synchronize(self._tensor.device)
        # Record completion in profiler
        if self._profile_ctx is not None:
            self._profile_ctx.complete()
    
    def is_completed(self):
        """Check if the work has completed."""
        return self._work.is_completed()
    
    def __getattr__(self, name):
        """Forward other attributes to the wrapped work object."""
        return getattr(self._work, name)


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
    
    # Initialize profiling if enabled
    if P2P_PROFILING_AVAILABLE and os.environ.get("ENABLE_P2P_PROFILING", "0") == "1":
        from p2p_profiling_lib import enable_profiling
        node_rank = int(os.environ.get("NODE_RANK", "0"))
        profiler = enable_profiling(node_rank=node_rank)
        if profiler:
            profiler.set_rank(dist.get_rank() if dist.is_initialized() else 0)
            print(f"[P2P Profiling] Enabled for node {node_rank}, output directory: ./profiling_results")


def finalize_uccl_comm():
    """Finalize UCCL communication and clean up resources."""
    global _uccl_comm
    
    # Finalize profiling if enabled
    if P2P_PROFILING_AVAILABLE:
        profiler = get_profiler()
        if profiler:
            profiler.finalize()
            print("[P2P Profiling] Finalized and statistics saved")
    
    if _uccl_comm is not None:
        _uccl_comm.finalize()
        _uccl_comm = None


def is_uccl_enabled() -> bool:
    """Check if UCCL communication is enabled."""
    comm = get_uccl_comm()
    return comm.is_enabled()

