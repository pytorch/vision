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
import warnings
from typing import Dict, Any
import torchvision.transforms.v2.functional as F
import numpy as np
from utils import bench, report_stats, print_comparison_table, print_benchmark_info

# Filter out the specific TF32 warning
warnings.filterwarnings(
    "ignore",
    message="Please use the new API settings to control TF32 behavior.*",
    category=UserWarning,
    module="torch.backends.cuda",
)

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
    import kornia.augmentation as KA
    HAS_KORNIA = True
except ImportError:
    HAS_KORNIA = False

from PIL import Image

# ImageNet normalization constants
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]




def torchvision_pipeline(images: torch.Tensor, target_size: int) -> torch.Tensor:
    images = F.resize(
        images, size=(target_size, target_size), interpolation=F.InterpolationMode.BILINEAR, antialias=True
    )
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
    transform = A.Compose(
        [
            A.Resize(target_size, target_size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=NORM_MEAN, std=NORM_STD, max_pixel_value=255.0),
        ]
    )
    img = transform(image=image)["image"]
    img = torch.from_numpy(img).permute(2, 0, 1)
    return img


def kornia_pipeline(image: torch.Tensor, target_size: int) -> torch.Tensor:
    # Kornia expects float tensors in [0, 1] range
    # TODO check that this is needed?
    img = image.float() / 255.0
    img = img.unsqueeze(0)  # Add batch dimension for kornia

    img = K.geometry.transform.resize(img, (target_size, target_size), interpolation="bilinear")

    img = K.enhance.normalize(img, mean=torch.tensor(NORM_MEAN), std=torch.tensor(NORM_STD))

    return img.squeeze(0)  # Remove batch dimension


# TODO double check that this works as expected: no graph break, and no issues with dynamic shapes
compiled_torchvision_pipeline = torch.compile(torchvision_pipeline, mode="default", fullgraph=True, dynamic=True)


def run_benchmark(args) -> Dict[str, Any]:
    backend = args.backend.lower()
    
    device = args.device.lower()
    # Check device compatibility
    if device == 'cuda' and backend not in ['tv', 'tv-compiled']:
        raise RuntimeError(f"CUDA device not supported for {backend} backend. Only 'tv' and 'tv-compiled' support CUDA.")
    
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Install cuda-enabled torch and torchvision, or use 'cpu' device.")

    if backend == "opencv" and not HAS_OPENCV:
        raise RuntimeError("OpenCV not available. Install with: pip install opencv-python")
    if backend == "albumentations" and not HAS_ALBUMENTATIONS:
        raise RuntimeError("Albumentations not available. Install with: pip install albumentations")
    if backend == "kornia" and not HAS_KORNIA:
        raise RuntimeError("Kornia not available. Install with: pip install kornia")

    if args.verbose:
        backend_display = args.backend.upper()
        print(f"\n=== {backend_display} ===")
        print(f"Device: {device}, Threads: {args.num_threads}, Batch size: {args.batch_size}")

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
    elif backend == "kornia":
        torch.set_num_threads(args.num_threads)
        pipeline = kornia_pipeline

    def generate_test_images():
        height = random.randint(args.min_size, args.max_size)
        width = random.randint(args.min_size, args.max_size)
        images = torch.randint(0, 256, (args.batch_size, 3, height, width), dtype=torch.uint8)

        memory_format = torch.channels_last if args.contiguity == "CL" else torch.contiguous_format
        if memory_format == torch.channels_last:
            images = images.to(memory_format=torch.channels_last)
        
        # Move to device for torchvision backends
        if backend in ['tv', 'tv-compiled']:
            images = images.to(device)

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
        elif backend == "kornia":
            if args.batch_size > 1:
                # TODO is that true????
                raise ValueError("Batches not supported in Kornia pipeline")

        return images

    times = bench(
        lambda images: pipeline(images, args.target_size),
        data_generator=generate_test_images,
        num_exp=args.num_exp,
        warmup=args.warmup,
    )

    stats = report_stats(times, "ms", args.verbose)
    return {"backend": args.backend, "stats": stats}




def main():
    parser = argparse.ArgumentParser(description="Benchmark torchvision transforms")
    parser.add_argument("--num-exp", type=int, default=100, help="Number of experiments we average over")
    parser.add_argument(
        "--warmup", type=int, default=10, help="Number of warmup runs before running the num-exp experiments"
    )
    parser.add_argument(
        "--target-size", type=int, default=224, help="size parameter of the Resize step, for both H and W."
    )
    parser.add_argument("--min-size", type=int, default=128, help="Minimum input image size for random generation")
    parser.add_argument("--max-size", type=int, default=512, help="Maximum input image size for random generation")
    parser.add_argument(
        "--num-threads", type=int, default=1, help="Number of intra-op threads as set with torch.set_num_threads() & Co"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size. 1 means single 3D image without a batch dimension"
    )
    parser.add_argument(
        "--contiguity",
        choices=["CL", "CF"],
        default="CF",
        help="Memory format: CL (channels_last) or CF (channels_first, i.e. contiguous)",
    )
    all_backends = ["tv", "tv-compiled", "opencv", "pil", "albumentations", "kornia"]
    parser.add_argument(
        "--backend", type=str.lower, choices=all_backends + ["all"], default="all", help="Backend to benchmark"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use: cpu, cuda, or gpu (default: cpu)")

    args = parser.parse_args()

    print_benchmark_info(args)

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
