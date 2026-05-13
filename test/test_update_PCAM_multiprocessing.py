# test/test_update_PCAM_multiprocessing.py
import os
import socket
import tempfile
from contextlib import closing

import pytest
import torch
from torch.utils.data import DataLoader, distributed
from torchvision import datasets
from torchvision.transforms import v2


def _find_free_port() -> int:
    """Pick a free TCP port for DDP init."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _ddp_worker(rank: int, world_size: int, port: int, root: str, backend: str):
    """Single DDP worker."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)

    torch.distributed.init_process_group(backend, rank=rank, world_size=world_size)

    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        pin_memory = True
    else:
        device = torch.device("cpu")
        pin_memory = False

    # ---- dataset ----
    ds = datasets.PCAM(root=root, split="train", download=True, transform=v2.ToTensor())
    sampler = distributed.DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(
        ds,
        batch_size=16,
        sampler=sampler,
        num_workers=2,
        pin_memory=pin_memory,
        persistent_workers=True,
    )

    # ---- iterate few batches ----
    local_seen = 0
    for i, (x, y) in enumerate(loader):
        assert x.ndim == 4 and y.ndim == 1
        if backend == "nccl":
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        local_seen += x.size(0)
        if i >= 3:
            break

    # ---- allreduce sanity check ----
    t = torch.tensor([local_seen], dtype=torch.int64, device=device)
    torch.distributed.all_reduce(t)
    assert t.item() > 0

    torch.distributed.destroy_process_group()


@pytest.mark.parametrize("world_size", [2])
@pytest.mark.parametrize("backend", ["gloo"])  # add "nccl" if you want GPU test too
def test_pcam_ddp(world_size, backend):
    """Smoke test PCAM with DDP + multiprocessing DataLoader."""
    if backend == "nccl" and not torch.cuda.is_available():
        pytest.skip("CUDA not available for NCCL backend")

    with tempfile.TemporaryDirectory() as tmp:
        root  = os.path.join(tmp, "pcam_data")
        os.makedirs(root, exist_ok=True)
        port = _find_free_port()

        # The simple spawn call you wanted
        torch.multiprocessing.spawn(
            _ddp_worker,
            args=(world_size, port, root, backend),  # passed to worker after rank
            nprocs=world_size,
            join=True,
        )


if __name__ == "__main__":
    # Allow running standalone: python test/test_update_PCAM_multiprocessing.py
    pytest.main([__file__, "-vvv", "-k", "test_pcam_ddp"])
