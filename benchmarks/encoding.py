import os
import platform
import statistics

import torch
import torch.utils.benchmark as benchmark
import torchvision


def print_machine_specs():
    print("Processor:", platform.processor())
    print("Platform:", platform.platform())
    print("Logical CPUs:", os.cpu_count())
    print(f"\nCUDA device: {torch.cuda.get_device_name()}")
    print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


def get_data():
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.PILToTensor(),
        ]
    )
    path = os.path.join(os.getcwd(), "data")
    testset = torchvision.datasets.Places365(
        root="./data", download=not os.path.exists(path), transform=transform, split="val"
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1000, shuffle=False, num_workers=1, collate_fn=lambda batch: [r[0] for r in batch]
    )
    return next(iter(testloader))


def run_benchmark(batch):
    results = []
    for device in ["cpu", "cuda"]:
        batch_device = [t.to(device=device) for t in batch]
        for size in [1, 100, 1000]:
            for num_threads in [1, 12, 24]:
                for stmt, strat in zip(
                    [
                        "[torchvision.io.encode_jpeg(img) for img in batch_input]",
                        "torchvision.io.encode_jpeg(batch_input)",
                    ],
                    ["unfused", "fused"],
                ):
                    batch_input = batch_device[:size]
                    t = benchmark.Timer(
                        stmt=stmt,
                        setup="import torchvision",
                        globals={"batch_input": batch_input},
                        label="Image Encoding",
                        sub_label=f"{device.upper()} ({strat}): {stmt}",
                        description=f"{size} images",
                        num_threads=num_threads,
                    )
                    results.append(t.blocked_autorange())
    compare = benchmark.Compare(results)
    compare.print()


if __name__ == "__main__":
    print_machine_specs()
    batch = get_data()
    mean_h, mean_w = statistics.mean(t.shape[-2] for t in batch), statistics.mean(t.shape[-1] for t in batch)
    print(f"\nMean image size: {int(mean_h)}x{int(mean_w)}")
    run_benchmark(batch)
