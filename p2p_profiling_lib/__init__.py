"""
P2P Communication Profiling Library for xDiT
This library provides utilities to profile and benchmark UCCL/NCCL P2P communications.
It tracks transfer metrics including data size, timing, communication type, and sync mode.
"""

from .profiler import P2PProfiler, get_profiler, enable_profiling, disable_profiling
from .stats import compile_stats

__all__ = [
    'P2PProfiler',
    'get_profiler', 
    'enable_profiling',
    'disable_profiling',
    'compile_stats'
]
