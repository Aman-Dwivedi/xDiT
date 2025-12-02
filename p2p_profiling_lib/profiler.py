"""
Core profiling functionality for P2P communications.
"""

import time
import csv
import os
import shutil
import glob
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
import torch

@dataclass
class TransferRecord:
    """Record of a single P2P transfer operation."""
    timestamp: float
    operation: str  # 'send' or 'recv'
    comm_type: str  # 'uccl' or 'nccl'
    sync_mode: str  # 'sync' or 'async'
    data_size_bytes: int
    duration_ms: float
    src_rank: int
    dst_rank: int
    tensor_shape: str
    dtype: str
    success: bool
    error_msg: str = ""

class P2PProfiler:
    """
    Thread-safe profiler for P2P communication operations.
    
    Usage:
        profiler = P2PProfiler(output_dir="./profiling_data", enabled=True)
        
        # For synchronous operations (automatic completion tracking):
        with profiler.profile_send(tensor, dst_rank, comm_type='uccl', sync_mode='sync'):
            # perform send operation
            pass
        
        # For asynchronous operations (manual completion tracking):
        with profiler.profile_send(tensor, dst_rank, comm_type='uccl', sync_mode='async') as ctx:
            handle = perform_async_send(tensor, dst_rank)
            # ... other work ...
            handle.wait()  # Wait for completion
            ctx.complete()  # Record actual completion time
    """
    
    def __init__(self, output_dir: str = "./profiling_results", enabled: bool = True, node_rank: int = 0):
        """
        Initialize the profiler.
        
        Args:
            output_dir: Directory to store profiling data (default: ./profiling_results)
            enabled: Whether profiling is enabled
            node_rank: Node rank (only node 0 will profile by default)
        """
        self.enabled = enabled and (node_rank == 0)
        self.output_dir = output_dir
        self.records: List[TransferRecord] = []
        self.lock = threading.Lock()
        self.current_rank = None
        
        if self.enabled:
            # Create timestamped subdirectory for this run
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = os.path.join(output_dir, timestamp)
            os.makedirs(self.run_dir, exist_ok=True)
            
            self.raw_csv_path = os.path.join(self.run_dir, "p2p_profiling_raw.csv")
            self.summary_csv_path = os.path.join(self.run_dir, "p2p_profiling_summary.csv")
            self._init_csv()
    


    
    def _init_csv(self):
        """Initialize the raw CSV file with headers."""
        with open(self.raw_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'timestamp', 'operation', 'comm_type', 'sync_mode',
                'data_size_bytes', 'duration_ms', 'src_rank', 'dst_rank',
                'tensor_shape', 'dtype', 'success', 'error_msg'
            ])
            writer.writeheader()
    
    def set_rank(self, rank: int):
        """Set the current process rank."""
        self.current_rank = rank
    
    class ProfileContext:
        """Context manager for profiling a single operation."""
        
        def __init__(self, profiler: 'P2PProfiler', operation: str, comm_type: str,
                     sync_mode: str, tensor: Optional[torch.Tensor], src_rank: int, dst_rank: int):
            self.profiler = profiler
            self.operation = operation
            self.comm_type = comm_type
            self.sync_mode = sync_mode
            self.tensor = tensor
            self.src_rank = src_rank
            self.dst_rank = dst_rank
            self.start_time = None
            self.record = None
            self.completed = False
        
        def __enter__(self):
            if self.profiler.enabled:
                # Only record start time, don't synchronize here
                # For async operations, we don't want to block at the start
                self.start_time = time.perf_counter()
            return self
        
        def complete(self):
            """
            Manually mark the operation as complete (for async operations).
            Call this after waiting on the async handle.
            Note: The caller should ensure the operation is actually complete
            (e.g., by calling handle.wait()) before calling this method.
            """
            if not self.profiler.enabled or self.completed:
                return
            
            self._record_completion(None, None, None)
            self.completed = True
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.profiler.enabled and not self.completed:
                # For sync operations or if complete() wasn't called
                # Only synchronize for sync operations
                if self.sync_mode == 'sync' and self.tensor is not None and self.tensor.is_cuda:
                    torch.cuda.synchronize(self.tensor.device)
                
                self._record_completion(exc_type, exc_val, exc_tb)
                self.completed = True
            
            return False  # Don't suppress exceptions
        
        def _record_completion(self, exc_type, exc_val, exc_tb):
            """Internal method to record the completion of the operation."""
            duration_ms = (time.perf_counter() - self.start_time) * 1000
            
            # Get tensor info
            if self.tensor is not None:
                data_size = self.tensor.numel() * self.tensor.element_size()
                tensor_shape = str(tuple(self.tensor.shape))
                dtype = str(self.tensor.dtype)
            else:
                data_size = 0
                tensor_shape = "unknown"
                dtype = "unknown"
            
            # Create record
            record = TransferRecord(
                timestamp=self.start_time,
                operation=self.operation,
                comm_type=self.comm_type,
                sync_mode=self.sync_mode,
                data_size_bytes=data_size,
                duration_ms=duration_ms,
                src_rank=self.src_rank,
                dst_rank=self.dst_rank,
                tensor_shape=tensor_shape,
                dtype=dtype,
                success=(exc_type is None),
                error_msg=str(exc_val) if exc_val else ""
            )
            
            self.profiler._record_transfer(record)
    
    def profile_send(self, tensor: torch.Tensor, dst_rank: int, 
                     comm_type: str, sync_mode: str) -> ProfileContext:
        """
        Profile a send operation.
        
        Args:
            tensor: Tensor being sent
            dst_rank: Destination rank
            comm_type: 'uccl' or 'nccl'
            sync_mode: 'sync' or 'async'
        """
        src_rank = self.current_rank if self.current_rank is not None else -1
        return self.ProfileContext(self, 'send', comm_type, sync_mode, tensor, src_rank, dst_rank)
    
    def profile_recv(self, tensor: Optional[torch.Tensor], src_rank: int,
                     comm_type: str, sync_mode: str) -> ProfileContext:
        """
        Profile a recv operation.
        
        Args:
            tensor: Tensor buffer for receiving (may be None before allocation)
            src_rank: Source rank
            comm_type: 'uccl' or 'nccl'
            sync_mode: 'sync' or 'async'
        """
        dst_rank = self.current_rank if self.current_rank is not None else -1
        return self.ProfileContext(self, 'recv', comm_type, sync_mode, tensor, src_rank, dst_rank)
    
    def _record_transfer(self, record: TransferRecord):
        """Record a transfer (thread-safe)."""
        with self.lock:
            self.records.append(record)
            self._append_to_csv(record)
    
    def _append_to_csv(self, record: TransferRecord):
        """Append a record to the raw CSV file."""
        with open(self.raw_csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'timestamp', 'operation', 'comm_type', 'sync_mode',
                'data_size_bytes', 'duration_ms', 'src_rank', 'dst_rank',
                'tensor_shape', 'dtype', 'success', 'error_msg'
            ])
            writer.writerow(asdict(record))
    
    def finalize(self):
        """
        Finalize profiling and generate summary statistics.
        Should be called at the end of the run.
        """
        if not self.enabled:
            return
        
        from .stats import compile_stats
        compile_stats(self.raw_csv_path, self.summary_csv_path)
        print(f"\n[P2P Profiler] Raw data saved to: {self.raw_csv_path}")
        print(f"[P2P Profiler] Summary saved to: {self.summary_csv_path}")
    
    def get_records(self) -> List[TransferRecord]:
        """Get all recorded transfers."""
        with self.lock:
            return self.records.copy()


# Global profiler instance
_global_profiler: Optional[P2PProfiler] = None
_profiler_lock = threading.Lock()


def get_profiler() -> Optional[P2PProfiler]:
    """Get the global profiler instance."""
    return _global_profiler


def enable_profiling(output_dir: str = "./profiling_results", node_rank: int = 0) -> P2PProfiler:
    """
    Enable global profiling.
    
    Args:
        output_dir: Directory to store profiling data (default: ./profiling_results)
        node_rank: Node rank (only node 0 will profile)
    
    Returns:
        The global profiler instance
    """
    global _global_profiler
    with _profiler_lock:
        if _global_profiler is None:
            _global_profiler = P2PProfiler(output_dir=output_dir, enabled=True, node_rank=node_rank)
        return _global_profiler


def disable_profiling():
    """Disable global profiling."""
    global _global_profiler
    with _profiler_lock:
        if _global_profiler is not None:
            _global_profiler.finalize()
            _global_profiler = None
