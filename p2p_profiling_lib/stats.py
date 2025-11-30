"""
Statistics compilation for P2P profiling data.
"""

import csv
from typing import Dict, List, Any
from collections import defaultdict


def compile_stats(raw_csv_path: str, summary_csv_path: str):
    """
    Compile raw profiling data into a human-readable summary.
    
    Args:
        raw_csv_path: Path to the raw profiling CSV
        summary_csv_path: Path to write the summary CSV
    """
    # Read raw data
    records = []
    with open(raw_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    
    if not records:
        print("[P2P Profiler] No records to compile")
        return
    
    # Compute statistics
    stats = {
        'total_operations': len(records),
        'total_send': 0,
        'total_recv': 0,
        'uccl_send_sync': 0,
        'uccl_send_async': 0,
        'uccl_recv_sync': 0,
        'uccl_recv_async': 0,
        'nccl_send_sync': 0,
        'nccl_send_async': 0,
        'nccl_recv_sync': 0,
        'nccl_recv_async': 0,
        'total_data_bytes': 0,
        'total_duration_ms': 0,
        'total_registration_ms': 0,
        'failed_operations': 0,
        'avg_duration_ms': 0,
        'avg_registration_ms': 0,
        'avg_data_size_mb': 0,
    }
    
    # Per-category stats
    category_stats = defaultdict(lambda: {
        'count': 0,
        'total_bytes': 0,
        'total_duration_ms': 0,
        'total_registration_ms': 0,
        'avg_duration_ms': 0,
        'avg_registration_ms': 0,
        'avg_size_mb': 0
    })
    
    for record in records:
        operation = record['operation']
        comm_type = record['comm_type']
        sync_mode = record['sync_mode']
        data_size = int(record['data_size_bytes'])
        duration = float(record['duration_ms'])
        registration = float(record.get('registration_ms', 0))
        success = record['success'].lower() == 'true'
        
        # Update totals
        stats['total_data_bytes'] += data_size
        stats['total_duration_ms'] += duration
        stats['total_registration_ms'] += registration
        
        if operation == 'send':
            stats['total_send'] += 1
        else:
            stats['total_recv'] += 1
        
        if not success:
            stats['failed_operations'] += 1
        
        # Update specific category
        key = f"{comm_type}_{operation}_{sync_mode}"
        stats[key] = stats.get(key, 0) + 1
        
        # Update category stats
        category = f"{comm_type.upper()}_{operation.upper()}_{sync_mode.upper()}"
        category_stats[category]['count'] += 1
        category_stats[category]['total_bytes'] += data_size
        category_stats[category]['total_duration_ms'] += duration
        category_stats[category]['total_registration_ms'] += registration
    
    # Compute averages
    if stats['total_operations'] > 0:
        stats['avg_duration_ms'] = stats['total_duration_ms'] / stats['total_operations']
        stats['avg_registration_ms'] = stats['total_registration_ms'] / stats['total_operations']
        stats['avg_data_size_mb'] = (stats['total_data_bytes'] / stats['total_operations']) / (1024 * 1024)
    
    for category, cat_stats in category_stats.items():
        if cat_stats['count'] > 0:
            cat_stats['avg_duration_ms'] = cat_stats['total_duration_ms'] / cat_stats['count']
            cat_stats['avg_registration_ms'] = cat_stats['total_registration_ms'] / cat_stats['count']
            cat_stats['avg_size_mb'] = (cat_stats['total_bytes'] / cat_stats['count']) / (1024 * 1024)
    
    # Write summary
    with open(summary_csv_path, 'w', newline='') as f:
        # Overall summary
        f.write("=== OVERALL SUMMARY ===\n")
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Total Operations', stats['total_operations']])
        writer.writerow(['Total Send Operations', stats['total_send']])
        writer.writerow(['Total Recv Operations', stats['total_recv']])
        writer.writerow(['Failed Operations', stats['failed_operations']])
        writer.writerow(['Total Data Transferred (MB)', f"{stats['total_data_bytes'] / (1024 * 1024):.2f}"])
        writer.writerow(['Total Transfer Duration (ms)', f"{stats['total_duration_ms']:.2f}"])
        writer.writerow(['Total Registration Duration (ms)', f"{stats['total_registration_ms']:.2f}"])
        writer.writerow(['Average Transfer Duration per Op (ms)', f"{stats['avg_duration_ms']:.2f}"])
        writer.writerow(['Average Registration Duration per Op (ms)', f"{stats['avg_registration_ms']:.2f}"])
        writer.writerow(['Average Data Size per Op (MB)', f"{stats['avg_data_size_mb']:.2f}"])
        writer.writerow([])
        
        # Breakdown by communication type and mode
        f.write("=== BREAKDOWN BY TYPE AND MODE ===\n")
        writer.writerow(['Category', 'Count', 'Percentage'])
        
        categories = [
            ('UCCL Send Sync', stats.get('uccl_send_sync', 0)),
            ('UCCL Send Async', stats.get('uccl_send_async', 0)),
            ('UCCL Recv Sync', stats.get('uccl_recv_sync', 0)),
            ('UCCL Recv Async', stats.get('uccl_recv_async', 0)),
            ('NCCL Send Sync', stats.get('nccl_send_sync', 0)),
            ('NCCL Send Async', stats.get('nccl_send_async', 0)),
            ('NCCL Recv Sync', stats.get('nccl_recv_sync', 0)),
            ('NCCL Recv Async', stats.get('nccl_recv_async', 0)),
        ]
        
        for name, count in categories:
            percentage = (count / stats['total_operations'] * 100) if stats['total_operations'] > 0 else 0
            writer.writerow([name, count, f"{percentage:.1f}%"])
        
        writer.writerow([])
        
        # Detailed category statistics
        f.write("=== DETAILED CATEGORY STATISTICS ===\n")
        writer.writerow(['Category', 'Count', 'Total Data (MB)', 'Avg Data (MB)', 'Total Transfer (ms)', 'Avg Transfer (ms)', 'Total Registration (ms)', 'Avg Registration (ms)'])
        
        for category in sorted(category_stats.keys()):
            cat_stats = category_stats[category]
            writer.writerow([
                category,
                cat_stats['count'],
                f"{cat_stats['total_bytes'] / (1024 * 1024):.2f}",
                f"{cat_stats['avg_size_mb']:.2f}",
                f"{cat_stats['total_duration_ms']:.2f}",
                f"{cat_stats['avg_duration_ms']:.2f}",
                f"{cat_stats['total_registration_ms']:.2f}",
                f"{cat_stats['avg_registration_ms']:.2f}"
            ])
        
        writer.writerow([])
        
        # UCCL vs NCCL comparison
        f.write("=== UCCL vs NCCL COMPARISON ===\n")
        writer.writerow(['Metric', 'UCCL', 'NCCL'])
        
        uccl_total = sum([stats.get(f'uccl_{op}_{mode}', 0) 
                          for op in ['send', 'recv'] 
                          for mode in ['sync', 'async']])
        nccl_total = sum([stats.get(f'nccl_{op}_{mode}', 0) 
                          for op in ['send', 'recv'] 
                          for mode in ['sync', 'async']])
        
        writer.writerow(['Total Operations', uccl_total, nccl_total])
        
        uccl_sync = stats.get('uccl_send_sync', 0) + stats.get('uccl_recv_sync', 0)
        uccl_async = stats.get('uccl_send_async', 0) + stats.get('uccl_recv_async', 0)
        nccl_sync = stats.get('nccl_send_sync', 0) + stats.get('nccl_recv_sync', 0)
        nccl_async = stats.get('nccl_send_async', 0) + stats.get('nccl_recv_async', 0)
        
        writer.writerow(['Sync Operations', uccl_sync, nccl_sync])
        writer.writerow(['Async Operations', uccl_async, nccl_async])
        
        # Calculate data and duration for UCCL vs NCCL
        uccl_data = sum([category_stats[cat]['total_bytes'] 
                        for cat in category_stats.keys() if 'UCCL' in cat])
        nccl_data = sum([category_stats[cat]['total_bytes'] 
                        for cat in category_stats.keys() if 'NCCL' in cat])
        
        uccl_duration = sum([category_stats[cat]['total_duration_ms'] 
                            for cat in category_stats.keys() if 'UCCL' in cat])
        nccl_duration = sum([category_stats[cat]['total_duration_ms'] 
                            for cat in category_stats.keys() if 'NCCL' in cat])
        
        uccl_registration = sum([category_stats[cat]['total_registration_ms'] 
                                for cat in category_stats.keys() if 'UCCL' in cat])
        nccl_registration = sum([category_stats[cat]['total_registration_ms'] 
                                for cat in category_stats.keys() if 'NCCL' in cat])
        
        writer.writerow(['Total Data (MB)', 
                        f"{uccl_data / (1024 * 1024):.2f}", 
                        f"{nccl_data / (1024 * 1024):.2f}"])
        writer.writerow(['Total Transfer Duration (ms)', 
                        f"{uccl_duration:.2f}", 
                        f"{nccl_duration:.2f}"])
        writer.writerow(['Total Registration Duration (ms)', 
                        f"{uccl_registration:.2f}", 
                        f"{nccl_registration:.2f}"])
        
        if uccl_total > 0:
            writer.writerow(['Avg Transfer per Op (ms)', 
                            f"{uccl_duration / uccl_total:.2f}", 
                            f"{nccl_duration / nccl_total:.2f}" if nccl_total > 0 else "N/A"])
            writer.writerow(['Avg Registration per Op (ms)', 
                            f"{uccl_registration / uccl_total:.2f}", 
                            f"{nccl_registration / nccl_total:.2f}" if nccl_total > 0 else "N/A"])
    
    print(f"[P2P Profiler] Summary statistics compiled successfully")
    _print_summary(stats, uccl_total, nccl_total, uccl_sync, uccl_async, nccl_sync, nccl_async)


def _print_summary(stats: Dict, uccl_total: int, nccl_total: int, 
                   uccl_sync: int, uccl_async: int, nccl_sync: int, nccl_async: int):
    """Print a brief summary to console."""
    print("\n" + "="*60)
    print("P2P PROFILING SUMMARY")
    print("="*60)
    print(f"Total Operations: {stats['total_operations']}")
    print(f"  Send: {stats['total_send']}, Recv: {stats['total_recv']}")
    print(f"  Failed: {stats['failed_operations']}")
    print(f"\nUCCL Operations: {uccl_total}")
    print(f"  Sync: {uccl_sync}, Async: {uccl_async}")
    print(f"\nNCCL Operations: {nccl_total}")
    print(f"  Sync: {nccl_sync}, Async: {nccl_async}")
    print(f"\nTotal Data: {stats['total_data_bytes'] / (1024 * 1024):.2f} MB")
    print(f"Total Transfer Duration: {stats['total_duration_ms']:.2f} ms")
    print(f"Total Registration Duration: {stats['total_registration_ms']:.2f} ms")
    print(f"Avg Transfer Duration: {stats['avg_duration_ms']:.2f} ms")
    print(f"Avg Registration Duration: {stats['avg_registration_ms']:.2f} ms")
    print("="*60 + "\n")
