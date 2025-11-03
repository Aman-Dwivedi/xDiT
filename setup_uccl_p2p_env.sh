#!/bin/bash
# =============================================================================
# UCCL P2P Environment Variables Setup Script
# =============================================================================
# Usage: source setup_uccl_p2p_env.sh

# ========== REQUIRED: Enable UCCL P2P ==========
export USE_UCCL_P2P=1                    
export UCCL_NUM_CPUS=4                   # Number of CPU threads for RDMA ops (default: 4)

# ========== OPTIONAL: Profiling ==========
export ENABLE_P2P_PROFILING=1            # Enable P2P profiling (default: "0")
export NODE_RANK=0                       # Node rank

# ========== OPTIONAL: UCCL Configuration ==========
export UCCL_ENTROPY=2                   
export UCCL_CHUNK_SIZE_KB=64            

# ========== NCCL Network Configuration (for inter-node) ==========
export NCCL_SOCKET_IFNAME=ens51f1np1     # Network interface for NCCL
export GLOO_SOCKET_IFNAME=ens51f1np1     # Network interface for Gloo
export NCCL_IB_DISABLE=1                 # Disable InfiniBand (if not using IB)
export NCCL_IB_GID_INDEX=3               # InfiniBand GID index (if using IB)
export NCCL_IB_HCA=mlx5_0:1              # InfiniBand HCA (if using IB)
export UCX_NET_DEVICES=mlx5_0:1          # UCX network device

# ========== Python Path ==========
export PYTHONPATH=/home/yangzhou/shreyas/uccl:$PYTHONPATH

# ========== Display Status ==========
echo "=========================================="
echo "UCCL P2P Environment Variables Loaded"
echo "=========================================="
echo "  USE_UCCL_P2P=$USE_UCCL_P2P"
echo "  UCCL_NUM_CPUS=$UCCL_NUM_CPUS"
echo "  ENABLE_P2P_PROFILING=$ENABLE_P2P_PROFILING"
echo "  NODE_RANK=$NODE_RANK"
echo "  UCCL_ENTROPY=$UCCL_ENTROPY"
echo "  UCCL_CHUNK_SIZE_KB=$UCCL_CHUNK_SIZE_KB"
echo "  NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
echo "=========================================="

