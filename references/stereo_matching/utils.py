import datetime
import math
import os
import time
from collections import defaultdict
from collections import deque

import torch
import torch.distributed as dist
import torch.nn.functional as F

from typing import Callable


from torchvision.datasets import (
    InStereo2k,
    CREStereo,
    SintelStereo,
    SceneFlowStereo,
    FallingThingsStereo,
    Middlebury2014Stereo,
    ETH3DStereo,
    Kitti2012Stereo,
    Kitti2015Stereo,
    CarlaStereo,
)


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt="{median:.4f} ({global_avg:.4f})"):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        t = reduce_across_processes([self.count, self.total])
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )


class MetricLogger:
    def __init__(self, delimiter="\t", log_dir=None, log_name=None):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        
        if log_dir is None and log_name is not None:
            raise ValueError("both `log_dir` and `log_name` must be specified for logging to a file")
        
        if log_name is None and log_dir is not None:
            raise ValueError("both `log_dir` and `log_name` must be specified for logging to a file")
        
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            self.log_name = os.path.join(log_dir, log_name)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if not isinstance(v, (float, int)):
                raise TypeError(
                    f"This method expects the value of the input arguments to be of type float or int, instead  got {type(v)}"
                )
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, **kwargs):
        self.meters[name] = SmoothedValue(**kwargs)

    def log_every(self, iterable, print_freq=5, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if print_freq is not None and i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    log_msg_prt = log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    
                else:
                    log_msg_prt = log_msg.format(
                            i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)
                        )
                print(log_msg_prt)
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))            
        header_msg = f"{header} Total time: {total_time_str}"
        print(header_msg)
            
            
def compute_metrics(flow_pred, flow_gt, valid_flow_mask=None):

    epe = ((flow_pred - flow_gt) ** 2).sum(dim=1).sqrt()
    flow_norm = (flow_gt ** 2).sum(dim=1).sqrt()

    if valid_flow_mask is not None:
        epe = epe[valid_flow_mask]
        flow_norm = flow_norm[valid_flow_mask]
    
    relative_epe = epe / flow_norm

    metrics = {
        "epe": epe.mean().item(),
        "1px": (epe < 1).float().mean().item(),
        "3px": (epe < 3).float().mean().item(),
        "5px": (epe < 5).float().mean().item(),
        "f1": ((epe > 3) & (relative_epe > 0.05)).float().mean().item() * 100,
    }
    return metrics, epe.numel()

def sequence_loss(flow_preds, flow_gt, valid_flow_mask, gamma=0.8, max_flow=256, weights=None):
    """Loss function defined over sequence of flow predictions"""
    
    if gamma > 1:
        raise ValueError(f"Gamma should be < 1, got {gamma}.")

    # exlude invalid pixels and extremely large diplacements
    flow_norm = torch.sum(flow_gt ** 2, dim=1).sqrt()
    valid_flow_mask = valid_flow_mask & (flow_norm < max_flow)

    valid_flow_mask = valid_flow_mask[:, None, :, :]

    flow_preds = torch.stack(flow_preds)  # shape = (num_flow_updates, batch_size, 2, H, W)

    abs_diff = (flow_preds - flow_gt).abs()
    abs_diff = (abs_diff * valid_flow_mask).mean(axis=(1, 2, 3, 4))

    num_predictions = flow_preds.shape[0]
    if weights is None or len(weights) != num_predictions:
        weights = torch.tensor([gamma], device=flow_gt.device, dtype=flow_gt.dtype) ** torch.arange(num_predictions - 1, -1, -1, device=flow_gt.device, dtype=flow_gt.dtype)
    flow_loss = (abs_diff * weights).sum()

    return flow_loss, weights

def sequence_consistency_loss(flow_preds, gamma=0.8, rescale_factor: float = 0.25):
    """Loss function defined over sequence of flow predictions"""

    if gamma > 1:
        raise ValueError(f"Gamma should be < 1, got {gamma}.")
    
    if rescale_factor > 1:
        raise ValueError(f"Rescale factor should be < 1, got {rescale_factor}.")

    flow_preds = torch.stack(flow_preds)  # shape = (num_flow_updates, batch_size, 2, H, W)
    N, B, _, H, W = flow_preds.shape
    
    # rescale flow predictions to account for bilinear upsampling artifacts
    if rescale_factor:
        flow_preds = F.interpolate(
            flow_preds.view(N * B, 2, H, W),
            scale_factor=rescale_factor,
            mode="bilinear",
            align_corners=True
        ) * rescale_factor
        flow_preds = torch.stack(torch.chunk(flow_preds, N, dim=0), dim=0)

    abs_diff = (flow_preds[1:] - flow_preds[:-1]).square()
    abs_diff = abs_diff.mean(axis=(1, 2, 3, 4))

    num_predictions = flow_preds.shape[0] - 1 # because we are comparing differences
    weights = torch.tensor([gamma], device=flow_preds.device) ** torch.arange(num_predictions - 1, -1, -1, device=flow_preds.device)
    flow_loss = (abs_diff * weights).sum()

    return flow_loss


class InputPadder:
    """Pads images such that dimensions are divisible by 8"""

    # TODO: Ideally, this should be part of the eval transforms preset, instead
    # of being part of the validation code. It's not obvious what a good
    # solution would be, because we need to unpad the predicted flows according
    # to the input images' size, and in some datasets (Kitti) images can have
    # variable sizes.

    def __init__(self, dims, mode="sintel", multiplier=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // multiplier) + 1) * multiplier - self.ht) % multiplier
        pad_wd = (((self.wd // multiplier) + 1) * multiplier - self.wd) % multiplier
        if mode == "sintel":
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode="replicate") for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0] : c[1], c[2] : c[3]]


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

    if all(key in os.environ for key in ("LOCAL_RANK", "RANK", "WORLD_SIZE")):
        # if we're here, the script was called with torchrun. Otherwise
        # these args will be set already by the run_with_submitit script
        args.local_rank = int(os.environ["LOCAL_RANK"])
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])

    elif "gpu" in args:
        # if we're here, the script was called by run_with_submitit.py
        args.local_rank = args.gpu
    else:
        print("Not using distributed mode!")
        args.distributed = False
        args.world_size = 1
        return

    args.distributed = True

    _redefine_print(is_main=(args.rank == 0))

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(
        backend="nccl",
        rank=args.rank,
        world_size=args.world_size,
        init_method=args.dist_url,
    )
    torch.distributed.barrier()


def reduce_across_processes(val):
    t = torch.tensor(val, device="cuda")
    dist.barrier()
    dist.all_reduce(t)
    return t


def freeze_batch_norm(model):
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()
            
            
def get_dataset_by_name(name: str, root: str, transforms: Callable):
    """Helper function to return a speciffic dataset configuration given it's name"""
    if name == "crestereo":
        return CREStereo(root=root, transforms=transforms)
    elif name == "carla-highres":
        return CarlaStereo(root=root, transforms=transforms)
    elif name == "instereo":
        return InStereo2k(root=root, transforms=transforms)
    elif name == "sintel":
        return SintelStereo(root=root, transforms=transforms)
    elif name == "sceneflow-monkaa":
        return SceneFlowStereo(root=root, transforms=transforms, split="Monkaa", pass_name="both")
    elif name == "sceneflow-flyingthings":
        return SceneFlowStereo(root=root, transforms=transforms, split="FlyingThings3D", pass_name="both")
    elif name == "sceneflow-driving":
        return SceneFlowStereo(root=root, transforms=transforms, split="Driving", pass_name="both")
    elif name == "fallingthings":
        return FallingThingsStereo(root=root, transforms=transforms, split="both")
    elif name == "eth3d-train":
        return ETH3DStereo(root=root, transforms=transforms, split="train")
    elif name == "instereo-2k":
        return InStereo2k(root=root, transforms=transforms, split="train")
    elif name == "middlebury2014-train":
        return Middlebury2014Stereo(root=root, transforms=transforms, split="train", calibration="perfect")
    elif name == "kitti2012-train":
        return Kitti2012Stereo(root=root, transforms=transforms, split="train")
    elif name == "kitti2015-train":
        return Kitti2015Stereo(root=root, transforms=transforms, split="train")
    elif name == "eth3d-train":
        return ETH3DStereo(root=root, transforms=transforms, split="train")
    else:
        raise ValueError(f"Unknown dataset {name}")
    
class ConsinAnnealingWarmupRestarts(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_0: int,
        T_mult: float = 1.,
        eta_min: float = 0.0001,
        T_warmup: int = 0,
        gamma: float = 1.,
        last_epoch: int = -1,
    ):
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("{} is not an Optimizer".format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_warmup = T_warmup
        self.gamma = gamma

        # iterations in current cycle
        self.T_cur = 0
        self._last_lr = 0
        self.N_cycle = 0
        self._C_cycle = 0

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs

        elif self.T_cur < self.T_warmup:
            return [
                (base_lr - self.eta_min) * self.T_cur /
                self.T_warmup + self.eta_min
                for base_lr in self.base_lrs
            ]

        else:
            return [
                self.eta_min + (base_lr - self.eta_min)
                * (1 + math.cos(math.pi * (self.T_cur - self.T_warmup) / (self.T_i - self.T_warmup))) / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.N_cycle = self.N_cycle + 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError(
                    "Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    # reset current epoch and compute current cycle
                    self.T_cur = epoch % self.T_0
                    self.N_cycle = epoch // self.T_0
                else:
                    # compute through how many cycles we have exponentiated the cycle size
                    n = int(
                        math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.N_cycle = n
                    self.T_cur = epoch - self.T_0 * \
                        (self.T_mult ** n - 1) / \
                        (self.T_mult - 1) - self.T_warmup
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.last_epoch = math.floor(epoch)

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
                param_group, lr = data
                param_group['lr'] = lr
                self.print_lr(self.verbose, i, lr, epoch)

        self.base_lrs = [group['initial_lr'] *
                         (self.gamma ** self.N_cycle) for group in self.optimizer.param_groups]
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
