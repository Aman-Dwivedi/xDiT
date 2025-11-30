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
    duration_ms: float  # Transfer time only (excludes registration)
    registration_ms: float  # Memory registration time (0.0 for NCCL or when not applicable)
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
        
        # Profile a send operation
        with profiler.profile_send(tensor, dst_rank, comm_type='uccl', sync_mode='async') as ctx:
            # perform send operation
            pass
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
            # Archive old results before starting new run
            self._archive_old_results()
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.raw_csv_path = os.path.join(output_dir, f"p2p_profiling_raw_{timestamp}.csv")
            self.summary_csv_path = os.path.join(output_dir, f"p2p_profiling_summary_{timestamp}.csv")
            self._init_csv()
    
    def _archive_old_results(self):
        """Archive old profiling results to past_results directory."""
        if not os.path.exists(self.output_dir):
            return
        
        # Find all CSV files in output directory
        csv_files = glob.glob(os.path.join(self.output_dir, "*.csv"))
        
        if not csv_files:
            return  # No old results to archive
        
        # Create archive directory with timestamp
        archive_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parent_dir = os.path.dirname(os.path.abspath(self.output_dir))
        archive_dir = os.path.join(parent_dir, "past_profiling_results", f"run_{archive_timestamp}")
        os.makedirs(archive_dir, exist_ok=True)
        
        # Move all CSV files to archive
        moved_count = 0
        for csv_file in csv_files:
            try:
                filename = os.path.basename(csv_file)
                dest_path = os.path.join(archive_dir, filename)
                shutil.move(csv_file, dest_path)
                moved_count += 1
            except Exception as e:
                print(f"[P2P Profiler] Warning: Failed to move {csv_file}: {e}")
        
        if moved_count > 0:
            print(f"[P2P Profiler] Archived {moved_count} old result file(s) to: {archive_dir}")

    
    def _init_csv(self):
        """Initialize the raw CSV file with headers."""
        with open(self.raw_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'timestamp', 'operation', 'comm_type', 'sync_mode',
                'data_size_bytes', 'duration_ms', 'registration_ms', 'src_rank', 'dst_rank',
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
            self._defer_recording = False  # For async operations, defer recording until wait()
            self.registration_ms = 0.0  # Registration time, set externally before __exit__ or record_completion
        
        def __enter__(self):
            if self.profiler.enabled:
                self.start_time = time.perf_counter()
                # For async operations, defer recording until wait() is called
                if self.sync_mode == 'async':
                    self._defer_recording = True
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            # For async operations, don't record here - will be recorded in wait()
            if self.profiler.enabled and not self._defer_recording:
                total_duration_ms = (time.perf_counter() - self.start_time) * 1000
                # Subtract registration time to get transfer time
                transfer_duration_ms = total_duration_ms - self.registration_ms
                
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
                    duration_ms=transfer_duration_ms,
                    registration_ms=self.registration_ms,
                    src_rank=self.src_rank,
                    dst_rank=self.dst_rank,
                    tensor_shape=tensor_shape,
                    dtype=dtype,
                    success=(exc_type is None),
                    error_msg=str(exc_val) if exc_val else ""
                )
                
                self.profiler._record_transfer(record)
            
            return False  # Don't suppress exceptions
        
        def set_registration_time(self, registration_ms: float):
            """Set the registration time measured externally."""
            self.registration_ms = registration_ms
        
        def get_profiling_info(self):
            """Get profiling information for deferred recording (used for async operations)."""
            if not self.profiler.enabled or self.start_time is None:
                return None
            
            # Get tensor info
            if self.tensor is not None:
                data_size = self.tensor.numel() * self.tensor.element_size()
                tensor_shape = str(tuple(self.tensor.shape))
                dtype = str(self.tensor.dtype)
            else:
                data_size = 0
                tensor_shape = "unknown"
                dtype = "unknown"
            
            return {
                'start_time': self.start_time,
                'operation': self.operation,
                'comm_type': self.comm_type,
                'sync_mode': self.sync_mode,
                'data_size_bytes': data_size,
                'src_rank': self.src_rank,
                'dst_rank': self.dst_rank,
                'tensor_shape': tensor_shape,
                'dtype': dtype,
                'registration_ms': self.registration_ms,
                'profiler': self.profiler
            }
        
        def record_completion(self, success: bool = True, error_msg: str = ""):
            """Record the completion of an async operation (called from wait())."""
            if not self.profiler.enabled or not self._defer_recording or self.start_time is None:
                return
            
            end_time = time.perf_counter()
            total_duration_ms = (end_time - self.start_time) * 1000
            # Subtract registration time to get transfer time
            transfer_duration_ms = total_duration_ms - self.registration_ms
            
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
                duration_ms=transfer_duration_ms,
                registration_ms=self.registration_ms,
                src_rank=self.src_rank,
                dst_rank=self.dst_rank,
                tensor_shape=tensor_shape,
                dtype=dtype,
                success=success,
                error_msg=error_msg
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
                'data_size_bytes', 'duration_ms', 'registration_ms', 'src_rank', 'dst_rank',
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
