"""
Utility functions for benchmarking transforms.
"""

from time import perf_counter_ns
from typing import Any, Callable, Dict, List

import torch
import torchvision
from tabulate import tabulate

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

try:
    import kornia as K

    HAS_KORNIA = True
except ImportError:
    HAS_KORNIA = False

from PIL import Image


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


def print_comparison_table(results: List[Dict[str, Any]]) -> None:
    # Use first backend as reference for relative comparison
    reference_median = results[0]["stats"]["median"] if results else None

    table_data = []
    for result in results:
        stats = result["stats"]
        speed_up = f"{reference_median / stats['median']:.2f}x" if reference_median else "N/A"

        table_data.append(
            {
                "Backend": result["backend"],
                "Median (ms)": f"{stats['median']:.2f}",
                "Std (ms)": f"{stats['std']:.2f}",
                "Mean (ms)": f"{stats['mean']:.2f}",
                "Min (ms)": f"{stats['min']:.2f}",
                "Max (ms)": f"{stats['max']:.2f}",
                "Speed-up": speed_up,
            }
        )

    print(tabulate(table_data, headers="keys", tablefmt="grid"))


def print_benchmark_info(args):
    """Print benchmark configuration and library versions."""
    device = args.device.lower()

    memory_format = "channels_last" if args.contiguity == "CL" else "channels_first"

    # Collect configuration info
    config = [
        ["Device", device],
        ["Threads", args.num_threads],
        ["Batch size", args.batch_size],
        ["Memory format", memory_format],
        ["Experiments", f"{args.num_exp} (+ {args.warmup} warmup)"],
        ["Input → output size", f"{args.min_size}-{args.max_size} → {args.target_size}×{args.target_size}"],
    ]

    print(tabulate(config, headers=["Parameter", "Value"], tablefmt="simple"))
    print()

    # Collect library versions
    versions = [
        ["PyTorch", torch.__version__],
        ["TorchVision", torchvision.__version__],
        ["OpenCV", cv2.__version__ if HAS_OPENCV else "Not available"],
        ["PIL/Pillow", getattr(Image, "__version__", "Version unavailable")],
        ["Albumentations", A.__version__ if HAS_ALBUMENTATIONS else "Not available"],
        ["Kornia", K.__version__ if HAS_KORNIA else "Not available"],
    ]

    print(tabulate(versions, headers=["Library", "Version"], tablefmt="simple"))
