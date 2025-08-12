#!/usr/bin/env python3
"""
Benchmark script for torchvision transforms performance.

This script benchmarks the performance of torchvision.transforms.v2 transforms
in various configurations and will be extended to compare against other libraries
like OpenCV.

The pipeline tested: uint8 image -> resize -> normalize (to [0,1] float)
"""

import argparse
import torch
import random
from time import perf_counter_ns
from typing import Callable, List, Tuple, Dict, Any
import torchvision.transforms.v2.functional as F


def bench(f: Callable, data_generator: Callable, num_exp: int, warmup: int) -> torch.Tensor:
    """
    Benchmark function execution time with fresh data for each experiment.

    Args:
        f: Function to benchmark
        data_generator: Callable that returns fresh data for each experiment
        num_exp: Number of experiments to run
        warmup: Number of warmup runs

    Returns:
        Tensor of execution times in nanoseconds
    """
    for _ in range(warmup):
        data = data_generator()
        f(data)

    times = []
    for _ in range(num_exp):
        data = data_generator()
        start = perf_counter_ns()
        result = f(data)
        end = perf_counter_ns()
        times.append(end - start)
        del result

    return torch.tensor(times, dtype=torch.float32)


def report_stats(times: torch.Tensor, unit: str) -> float:
    mul = {
        "ns": 1,
        "µs": 1e-3,
        "ms": 1e-6,
        "s": 1e-9,
    }[unit]

    times = times * mul
    std = times.std().item()
    med = times.median().item()
    mean = times.mean().item()
    min_time = times.min().item()
    max_time = times.max().item()

    print(f"  Median: {med:.2f}{unit} ± {std:.2f}{unit}")
    print(f"  Mean: {mean:.2f}{unit}, Min: {min_time:.2f}{unit}, Max: {max_time:.2f}{unit}")

    return med


def transform_pipeline(images: torch.Tensor, target_size: int) -> torch.Tensor:
    images = F.resize(images, size=target_size, antialias=True)
    images = F.to_dtype(images, dtype=torch.float32, scale=True)
    images = F.normalize(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return images


def run_benchmark(args) -> Dict[str, float]:
    memory_format = torch.channels_last if args.contiguity == "CL" else torch.contiguous_format
    print(f"\n=== TorchVision Transform Benchmark ===")
    print(f"Threads: {args.num_threads}, Batch size: {args.batch_size}")
    print(f"Memory format: {'channels_last' if memory_format == torch.channels_last else 'channels_first'}")

    torch.set_num_threads(args.num_threads)

    def generate_test_images():
        height = random.randint(args.min_size, args.max_size)
        width = random.randint(args.min_size, args.max_size)

        images = torch.randint(0, 256, (args.batch_size, 3, height, width), dtype=torch.uint8)

        if memory_format == torch.channels_last:
            images = images.to(memory_format=torch.channels_last)

        return images

    times = bench(
        lambda images: transform_pipeline(images, args.target_size),
        generate_test_images,
        args.num_exp,
        args.warmup,
    )

    median_time = report_stats(times, "ms")

    return {"median_time_ms": median_time}


def main():
    """Main benchmark runner."""
    parser = argparse.ArgumentParser(description="Benchmark torchvision transforms")
    parser.add_argument("--num-exp", type=int, default=100, help="Number of experiments we average over")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup runs before running the num-exp experiments")
    parser.add_argument("--target-size", type=int, default=224, help="Resize target size")
    parser.add_argument("--min-size", type=int, default=128, help="Minimum input image size for random generation")
    parser.add_argument("--max-size", type=int, default=512, help="Maximum input image size for random generation")
    parser.add_argument("--num-threads", type=int, default=1, help="Number of intra-op threads as set with torch.set_num_threads()")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size. 1 means single image processing without a batch dimension")
    parser.add_argument("--contiguity", choices=["CL", "CF"], default="CF", help="Memory format: CL (channels_last) or CF (channels_first, i.e. contiguous)")

    args = parser.parse_args()

    try:
        result = run_benchmark(args)
    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
