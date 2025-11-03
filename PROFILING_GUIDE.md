# P2P Profiling Guide

Profiling is already enabled in `setup_uccl_p2p_env.sh` via `ENABLE_P2P_PROFILING=1`.

Run normal command - profiling happens automatically. Results are saved to:
- Raw data: `./profiling_results/p2p_profiling_raw_YYYYMMDD_HHMMSS.csv`
- Summary: `./profiling_results/p2p_profiling_summary_YYYYMMDD_HHMMSS.csv` (generated on cleanup/shutdown)

## Output Format

Raw CSV columns: timestamp, operation, comm_type, sync_mode, data_size_bytes, duration_ms, src_rank, dst_rank, tensor_shape, dtype, success, error_msg

Summary includes: overall stats, breakdown by UCCL/NCCL and sync/async, detailed category stats, UCCL vs NCCL comparison.

## Configuration

Only node 0 collects data by default (set `NODE_RANK=0` on master). Old results auto-archived to `past_profiling_results/`.

## Benchmarking UCCL vs NCCL

Run twice:
1. With `USE_UCCL_P2P=1`
2. With `USE_UCCL_P2P=0`

Compare the summary CSVs for throughput and duration differences.

## Analyzing Results

```python
import pandas as pd
df = pd.read_csv('./profiling_results/p2p_profiling_raw_YYYYMMDD_HHMMSS.csv')
df['throughput_mbps'] = (df['data_size_bytes'] / (1024*1024)) / (df['duration_ms'] / 1000)
print(df.groupby('comm_type')['throughput_mbps'].mean())
```

## Troubleshooting

If no data appears:
- Check `ENABLE_P2P_PROFILING=1` is set
- Verify `NODE_RANK=0` on master node
- Look for `[P2P Profiling] Enabled` in logs
- Check UCCL is actually being used (should see "UCCL P2P communication enabled")
