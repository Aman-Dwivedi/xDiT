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
import torch
import torch.distributed as dist
import os
import warnings
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
    torch.distributed's send/recv API. The collective API automatically handles
    UCCL or NCCL communication based on availability.
    """

    def __init__(self):
        self._enabled = False
        self._initialized = False
        self._use_uccl = os.environ.get("USE_UCCL_P2P", "0") == "1"
        self._num_cpus = int(os.environ.get("UCCL_NUM_CPUS", "4"))
        self.send_chunks_tensor = None
        self.recv_chunks_tensor = None
        # self._local_gpu_idx: Optional[int] = None
        # self._rank = -1
        # self._world_size = -1

        if self._use_uccl and not UCCL_AVAILABLE:
            warnings.warn(
                f"USE_UCCL_P2P is set but UCCL is not available. "
                f"Import error: {_import_error}. "
                f"Falling back to torch.distributed."
            )
            self._use_uccl = False
    
    def is_enabled(self):
        """Check if UCCL P2P is enabled and available."""
        return self._use_uccl

    def initialize(self):
        """
        Initialize UCCL collective context and establish connections with all ranks.
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
        
        device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)
        # self._rank = dist.get_rank()
        # self._world_size = dist.get_world_size()

        # # Determine local GPU index
        # if local_gpu_idx is not None:
        #     self._local_gpu_idx = local_gpu_idx
        # else:
        #     # Try to get from environment
        #     self._local_gpu_idx = int(os.environ.get("LOCAL_RANK", self._rank))

        # Initialize UCCL collective context
        collective.init_collective(num_cpus=self._num_cpus, disable_uccl_intra=True)

        self._initialized = True
        self._enabled = True
        # print(f"[Rank {self._rank}] UCCL collective communication initialized") 

    def isend(self, tensor: torch.Tensor, dst: int, group=None):
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

        # Register tensor for UCCL
        # Default Values from benchmark_uccl_alltoall
        block_size = 4 * 1024
        num_kv_heads = 32 // 4
        head_dim = 128
        dtype = torch.float16
        device = torch.cuda.current_device()
        global_rank = dist.get_rank()

        send_tensor_uccl = torch.randn(
            block_size,
            2,
            num_kv_heads,
            head_dim,
            dtype=dtype,
            device=device,
        )

        collective.register_tensor(send_tensor_uccl)
        send_req = collective.isend(send_tensor_uccl, dst)
        collective.deregister_tensor(send_tensor_uccl)
        return send_req

    def irecv(self, tensor: torch.Tensor, src: int, group=None):
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

        # Register tensor for UCCL
        # Default Values from benchmark_uccl_alltoall
        block_size = 4 * 1024
        num_kv_heads = 32 // 4
        head_dim = 128
        dtype = torch.float16
        device = torch.cuda.current_device()
        global_rank = dist.get_rank()

        recv_tensor_uccl = torch.empty(
            block_size,
            2,
            num_kv_heads,
            head_dim,
            dtype=dtype,
            device=device,
        )

        collective.register_tensor(recv_tensor_uccl)
        recv_req = collective.isend(recv_tensor_uccl, src)
        collective.deregister_tensor(recv_tensor_uccl)
        return recv_req

    def finalize(self):
        """Clean up UCCL resources."""
        if self._enabled and self._initialized:
            collective.finalize_collective()
            dist.destroy_process_group()
            self._initialized = False
            self._enabled = False
            # print(f"[Rank {self._rank}] UCCL collective communication finalized")
