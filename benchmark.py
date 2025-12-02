import subprocess
import time
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--node_rank", type=int, required=True,
                        help="Node rank for torchrun")
    args = parser.parse_args()

    node_rank = args.node_rank

    command = [
        "torchrun",
        "--nproc_per_node=4",
        "--nnodes=2",
        f"--node_rank={node_rank}",
        "--master_addr=10.162.224.130",
        "--master_port=12355",
        "examples/pixartalpha_example.py",
        "--model", "PixArt-alpha/PixArt-XL-2-1024-MS",
        "--pipefusion_parallel_degree", "8",
        "--num_inference_steps", "20",
        "--warmup_steps", "1",
        "--prompt", "A small dog",
    ]

    for i in range(5):
        print(f"\n===== Run {i+1} / 5 =====\n")
        subprocess.run(command, check=True)
        if i < 4:
            time.sleep(5)

if __name__ == "__main__":
    main()
