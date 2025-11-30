import subprocess
import time
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--node_rank", type=int, required=True,
                        help="Rank of this node (0 or 1)")
    args = parser.parse_args()

    base_cmd = [
        "torchrun",
        "--nproc_per_node=1",
        "--nnodes=2",
        f"--node_rank={args.node_rank}",
        "--master_addr=10.162.224.130",
        "--master_port=12355",
        "examples/pixartalpha_example.py",
        "--model", "PixArt-alpha/PixArt-XL-2-1024-MS",
        "--pipefusion_parallel_degree", "2",
        "--num_inference_steps", "20",
        "--warmup_steps", "1",
        "--prompt", "A small dog"
    ]

    for i in range(5):
        print(f"\n==== Run {i+1} / 5 ====\n")
        subprocess.run(base_cmd)
        if i < 4:
            print("Sleeping 5 seconds...\n")
            time.sleep(5)

if __name__ == "__main__":
    main()
