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

try:
    import albumentations as A
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False

from PIL import Image
from tabulate import tabulate

# ImageNet normalization constants
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


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


def report_stats(times: torch.Tensor, unit: str, verbose: bool = True) -> Dict[str, float]:
    mul = {
        "ns": 1,
        "µs": 1e-3,
        "ms": 1e-6,
        "s": 1e-9,
    }[unit]

    times = times * mul
    stats = {
        "std": times.std().item(),
        "median": times.median().item(),
        "mean": times.mean().item(),
        "min": times.min().item(),
        "max": times.max().item(),
    }
    
    if verbose:
        print(f"  Median: {stats['median']:.2f}{unit} ± {stats['std']:.2f}{unit}")
        print(f"  Mean: {stats['mean']:.2f}{unit}, Min: {stats['min']:.2f}{unit}, Max: {stats['max']:.2f}{unit}")
    
    return stats


def torchvision_pipeline(images: torch.Tensor, target_size: int) -> torch.Tensor:
    images = F.resize(images, size=(target_size, target_size), interpolation=F.InterpolationMode.BILINEAR, antialias=True)
    images = F.to_dtype(images, dtype=torch.float32, scale=True)
    images = F.normalize(images, mean=NORM_MEAN, std=NORM_STD)
    return images


def opencv_pipeline(image: np.ndarray, target_size: int) -> torch.Tensor:
    img = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)  # no antialias in OpenCV
    img = img.astype(np.float32) / 255.0
    img = (img - np.array(NORM_MEAN)) / np.array(NORM_STD)
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    return torch.from_numpy(img)


def pil_pipeline(image: Image.Image, target_size: int) -> torch.Tensor:
    img = image.resize((target_size, target_size), Image.BILINEAR)  # PIL forces antialias
    img = F.pil_to_tensor(img)
    img = F.to_dtype(img, dtype=torch.float32, scale=True)
    img = F.normalize(img, mean=NORM_MEAN, std=NORM_STD)
    return img


def albumentations_pipeline(image: np.ndarray, target_size: int) -> torch.Tensor:
    transform = A.Compose([
        A.Resize(target_size, target_size, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=NORM_MEAN, std=NORM_STD, max_pixel_value=255.0)
    ])
    img = transform(image=image)["image"]
    img = torch.from_numpy(img).permute(2, 0, 1)
    return img


# TODO double check that this works as expected: no graph break, and no issues with dynamic shapes
compiled_torchvision_pipeline = torch.compile(torchvision_pipeline, mode="default", fullgraph=True, dynamic=True)


def run_benchmark(args) -> Dict[str, Any]:
    backend = args.backend.lower()
    
    if backend == "opencv" and not HAS_OPENCV:
        raise RuntimeError("OpenCV not available. Install with: pip install opencv-python")
    if backend == "albumentations" and not HAS_ALBUMENTATIONS:
        raise RuntimeError("Albumentations not available. Install with: pip install albumentations")
    
    if args.verbose:
        backend_display = args.backend.upper()
        print(f"\n=== {backend_display} ===")
        print(f"Threads: {args.num_threads}, Batch size: {args.batch_size}")

        memory_format = torch.channels_last if args.contiguity == "CL" else torch.contiguous_format
        print(f"Memory format: {'channels_last' if memory_format == torch.channels_last else 'channels_first'}")
    
    if backend == "tv":
        torch.set_num_threads(args.num_threads)
        pipeline = torchvision_pipeline
    elif backend == "tv-compiled":
        torch.set_num_threads(args.num_threads)
        pipeline = compiled_torchvision_pipeline
    elif backend == "opencv":
        cv2.setNumThreads(args.num_threads)
        pipeline = opencv_pipeline
    elif backend == "pil":
        torch.set_num_threads(args.num_threads)
        pipeline = pil_pipeline
    elif backend == "albumentations":
        cv2.setNumThreads(args.num_threads)
        pipeline = albumentations_pipeline
    
    def generate_test_images():
        height = random.randint(args.min_size, args.max_size)
        width = random.randint(args.min_size, args.max_size)
        images = torch.randint(0, 256, (args.batch_size, 3, height, width), dtype=torch.uint8)
        
        memory_format = torch.channels_last if args.contiguity == "CL" else torch.contiguous_format
        if memory_format == torch.channels_last:
            images = images.to(memory_format=torch.channels_last)
        
        if args.batch_size == 1:
            images = images[0]
        
        if backend == "opencv":
            if args.batch_size > 1:
                raise ValueError("Batches not supported in OpenCV pipeline")
            # TODO double check that contiguity requirement is respected for numpy array
            images = images.numpy().transpose(1, 2, 0)
        elif backend == "pil":
            if args.batch_size > 1:
                raise ValueError("Batches not supported in PIL pipeline")
            # Convert to PIL Image (CHW -> HWC)
            images = images.numpy().transpose(1, 2, 0)
            images = Image.fromarray(images)
        elif backend == "albumentations":
            if args.batch_size > 1:
                # TODO is that true????
                raise ValueError("Batches not supported in Albumentations pipeline")
            images = images.numpy().transpose(1, 2, 0)

        return images
    
    times = bench(
        lambda images: pipeline(images, args.target_size),
        data_generator=generate_test_images,
        num_exp=args.num_exp,
        warmup=args.warmup,
    )
    
    stats = report_stats(times, "ms", args.verbose)
    return {"backend": args.backend, "stats": stats}


def print_comparison_table(results: List[Dict[str, Any]]) -> None:
    torchvision_median = next((r["stats"]["median"] for r in results if r["backend"].lower() == "tv"), None)
    
    table_data = []
    for result in results:
        stats = result["stats"]
        relative = f"{stats['median'] / torchvision_median:.2f}x" if torchvision_median else "N/A"
        
        table_data.append({
            "Backend": result["backend"],
            "Median (ms)": f"{stats['median']:.2f}",
            "Std (ms)": f"{stats['std']:.2f}",
            "Mean (ms)": f"{stats['mean']:.2f}",
            "Min (ms)": f"{stats['min']:.2f}",
            "Max (ms)": f"{stats['max']:.2f}",
            "Relative": relative
        })
    
    print(tabulate(table_data, headers="keys", tablefmt="grid"))


def main():
    parser = argparse.ArgumentParser(description="Benchmark torchvision transforms")
    parser.add_argument("--num-exp", type=int, default=100, help="Number of experiments we average over")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup runs before running the num-exp experiments")
    parser.add_argument("--target-size", type=int, default=224, help="Resize target size")
    parser.add_argument("--min-size", type=int, default=128, help="Minimum input image size for random generation")
    parser.add_argument("--max-size", type=int, default=512, help="Maximum input image size for random generation")
    parser.add_argument("--num-threads", type=int, default=1, help="Number of intra-op threads as set with torch.set_num_threads()")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size. 1 means single image processing without a batch dimension")
    parser.add_argument("--contiguity", choices=["CL", "CF"], default="CF", help="Memory format: CL (channels_last) or CF (channels_first, i.e. contiguous)")
    all_backends = ["tv", "tv-compiled", "opencv", "pil", "albumentations"]
    parser.add_argument("--backend", type=str.lower, choices=all_backends + ["all"], default="all", help="Backend to use for transforms")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()
    
    print(f"Averaging over {args.num_exp} runs, {args.warmup} warmup runs")

    backends_to_run = all_backends if args.backend.lower() == "all" else [args.backend]
    results = []
    
    for backend in backends_to_run:
        args.backend = backend
        try:
            result = run_benchmark(args)
            results.append(result)
        except Exception as e:
            print(f"ERROR with {backend}: {e}")
    
    if len(results) > 1:
        print_comparison_table(results)


if __name__ == "__main__":
    main()
