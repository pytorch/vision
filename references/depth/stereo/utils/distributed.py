import os

import torch
import torch.distributed as dist


def _redefine_print(is_main):
    """disables printing when not in main process"""
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_main or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def setup_ddp(args):
    # Set the local_rank, rank, and world_size values as args fields
    # This is done differently depending on how we're running the script. We
    # currently support either torchrun or the custom run_with_submitit.py
    # If you're confused (like I was), this might help a bit
    # https://discuss.pytorch.org/t/what-is-the-difference-between-rank-and-local-rank/61940/2

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print("Not using distributed mode")
        args.distributed = False
        args.world_size = 1
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    dist.init_process_group(
        backend="nccl",
        rank=args.rank,
        world_size=args.world_size,
        init_method=args.dist_url,
    )
    torch.distributed.barrier()
    _redefine_print(is_main=(args.rank == 0))


def reduce_across_processes(val):
    t = torch.tensor(val, device="cuda")
    dist.barrier()
    dist.all_reduce(t)
    return t
