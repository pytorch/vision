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
import numpy as np

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


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


def torchvision_pipeline(images: torch.Tensor, target_size: int) -> torch.Tensor:
    images = F.resize(images, size=(target_size, target_size), interpolation=F.InterpolationMode.BILINEAR, antialias=True)
    images = F.to_dtype(images, dtype=torch.float32, scale=True)
    images = F.normalize(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return images


def opencv_pipeline(images: np.ndarray, target_size: int) -> np.ndarray:
    img = cv2.resize(images, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    return img


def run_benchmark(args) -> Dict[str, float]:
    if args.backend == "opencv" and not HAS_OPENCV:
        raise RuntimeError("OpenCV not available. Install with: pip install opencv-python")
    
    backend_name = args.backend.upper()
    print(f"\n=== {backend_name} ===")
    print(f"Threads: {args.num_threads}, Batch size: {args.batch_size}")

    memory_format = torch.channels_last if args.contiguity == "CL" else torch.contiguous_format
    print(f"Memory format: {'channels_last' if memory_format == torch.channels_last else 'channels_first'}")
    
    if args.backend == "torchvision":
        torch.set_num_threads(args.num_threads)
        pipeline = torchvision_pipeline
    elif args.backend == "opencv":
        cv2.setNumThreads(args.num_threads)
        pipeline = opencv_pipeline

    
    def generate_test_images():
        height = random.randint(args.min_size, args.max_size)
        width = random.randint(args.min_size, args.max_size)
        images = torch.randint(0, 256, (args.batch_size, 3, height, width), dtype=torch.uint8)
        
        memory_format = torch.channels_last if args.contiguity == "CL" else torch.contiguous_format
        if memory_format == torch.channels_last:
            images = images.to(memory_format=torch.channels_last)
        
        if args.batch_size == 1:
            images = images[0]
        
        if args.backend == "opencv":
            if args.batch_size > 1:
                raise ValueError("Batches not supported in OpenCV pipeline (yet??)")
            # TODO double check that contiguity requirement is respected for numpy array
            images = images.transpose(2, 0).numpy()

        return images
    
    times = bench(
        lambda images: pipeline(images, args.target_size),
        data_generator=generate_test_images,
        num_exp=args.num_exp,
        warmup=args.warmup,
    )
    
    report_stats(times, "ms")


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
    all_backends = ["torchvision", "opencv"]
    parser.add_argument("--backend", choices=all_backends + ["all"], default="all", help="Backend to use for transforms")

    args = parser.parse_args()
    
    print(f"Averaging over {args.num_exp} runs, {args.warmup} warmup runs")

    backends_to_run = all_backends if args.backend == "all" else args.backend
    for backend in backends_to_run:
        args.backend = backend
        try:
            result = run_benchmark(args)
        except Exception as e:
            print(f"ERROR with {backend}: {e}")


if __name__ == "__main__":
    main()
