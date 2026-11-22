"""Compare Gaussian blur with the original dense convolution.

Run from a torchvision development installation, for example::

    python benchmarks/gaussian_blur.py --device cpu --dtype float64 --size 512 --kernels 31x33 31x35 65x65
    python benchmarks/gaussian_blur.py --device cuda --size 1024 --kernels 3x3 33x33 65x65 131x131

JSON lines report median runtime, error against the dense implementation, and
memory. CPU memory is the largest individual allocation recorded by the PyTorch
profiler, not process RSS. CUDA memory is the peak allocated memory above the
live input, including convolution workspace and output. The dense CPU float64
path can require H * W * kx * ky * element_size bytes: use --max-workspace-gib
to skip configurations that would require excessive memory.

Separable filtering is mathematically equivalent, but floating point summation
order differs. Compare errors as well as runtime; speedups depend on the image,
kernel, dtype, device, memory format, and convolution backend.

Dispatch keeps kernels with at most 1024 taps, and one-dimensional kernels, on
the dense path. CPU inputs other than float64 also retain that path below
2**28 input-element/kernel-tap products, to avoid two-convolution overhead on
small workloads. These are conservative heuristics, not universal crossover
points. Use --batch-size, --channels, and --channels-last to measure other
workloads. Each path is warmed up and measured in alternating order over
--rounds repetitions; the raw round medians are included in the output.
"""

import argparse
import inspect
import json
import platform
import statistics

import torch
import torch.nn.functional as nnf
from torch.utils.benchmark import Timer
from torchvision.transforms.v2.functional._misc import _get_gaussian_kernel2d, gaussian_blur_image


def dense_gaussian_blur(image, kernel_size, sigma):
    fp = image.is_floating_point()
    kernel = _get_gaussian_kernel2d(kernel_size, sigma, dtype=image.dtype if fp else torch.float32, device=image.device)
    channels = image.shape[-3]
    kernel = kernel.expand(channels, 1, *kernel.shape)
    output = image if fp else image.float()
    output = nnf.pad(output, [kernel_size[0] // 2] * 2 + [kernel_size[1] // 2] * 2, mode="reflect")
    output = nnf.conv2d(output, kernel, groups=channels)
    return output if fp else output.round().to(image.dtype)


def measure_memory(fn, image, kernel_size, sigma):
    if image.is_cuda:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        _ = fn(image, kernel_size, sigma)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - baseline
        del _
        return peak
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True, acc_events=True
    ) as prof:
        fn(image, kernel_size, sigma)
    return max(event.self_cpu_memory_usage for event in prof.events())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float64", "float16", "bfloat16", "uint8"], default="float32")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--kernels", nargs="+", default=["3x3", "15x15", "31x31", "33x33", "65x65"])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--min-run-time", type=float, default=0.5)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-workspace-gib", type=float, default=4.0)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    dtype = getattr(torch, args.dtype)
    image = torch.randint(0, 256, (args.batch_size, args.channels, args.size, args.size), device=args.device).to(dtype)
    if args.channels_last:
        image = image.contiguous(memory_format=torch.channels_last)
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "source": inspect.getfile(gaussian_blur_image),
                "platform": platform.platform(),
                "gpu": torch.cuda.get_device_name() if image.is_cuda else None,
                **vars(args),
            }
        ),
        flush=True,
    )
    for kernel in args.kernels:
        kernel_size = [int(size) for size in kernel.split("x")]
        sigma = [size / 6 for size in kernel_size]
        estimated_workspace = image.numel() * kernel_size[0] * kernel_size[1] * max(4, image.element_size())
        if estimated_workspace > args.max_workspace_gib * 1024**3:
            print(json.dumps({"kernel": kernel, "skipped": "dense workspace estimate exceeds limit"}), flush=True)
            continue
        row = {"kernel": kernel}
        with torch.no_grad():
            functions = [("dense", dense_gaussian_blur), ("separable_dispatch", gaussian_blur_image)]
            # Warm both paths, then alternate their order to reduce cache,
            # device clock and scheduling bias. Retain the per-round medians.
            for _, fn in functions:
                for _ in range(2):
                    fn(image, kernel_size, sigma)
            timings = {name: [] for name, _ in functions}
            for repeat in range(args.rounds):
                for name, fn in functions[:: 1 if repeat % 2 == 0 else -1]:
                    measurement = Timer(
                        "fn(image, kernel_size, sigma)",
                        globals={"fn": fn, "image": image, "kernel_size": kernel_size, "sigma": sigma},
                        num_threads=args.threads,
                    ).blocked_autorange(min_run_time=args.min_run_time)
                    timings[name].append(measurement.median * 1000)
            for name, fn in functions:
                row[name + "_ms"] = statistics.median(timings[name])
                row[name + "_rounds_ms"] = timings[name]
                row[name + "_memory_bytes"] = measure_memory(fn, image, kernel_size, sigma)
            expected = dense_gaussian_blur(image, kernel_size, sigma)
            actual = gaussian_blur_image(image, kernel_size, sigma)
            row["max_abs_error"] = (actual.double() - expected.double()).abs().max().item()
            row["speedup"] = row["dense_ms"] / row["separable_dispatch_ms"]
            del expected, actual
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
