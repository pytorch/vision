import argparse
import warnings
from math import ceil
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torchvision.models.optical_flow
import utils
from presets import StereoMatchingPresetEval, StereoMatchingPresetCRETrain
from torch.optim.lr_scheduler import _LRScheduler
from torchvision.datasets import (
    InStereo2k,
    CREStereo,
    SintelStereo,
    SceneFlowStereo,
    FallingThingsStereo,
    Middlebury2014Stereo,
)


class InfiniteWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset) -> None:
        super().__init__()
        self.dataset = dataset
        self.dataset_size = len(dataset)
        # arbitrarily large value
        self.max_size = int(1e18)

    def __getitem__(self, idx: int):
        sample_idx = idx % self.dataset_size
        return self.dataset[sample_idx]


class ArrayScheduler(_LRScheduler):
    def __init__(self, optimizer, schedule: np.ndarray, last_epoch: int = -1, verbose=False) -> None:
        super().__init__(optimizer, last_epoch)
        self.schedule = schedule.to_list()
        self._step_count = 0

    def step(self, epoch=None):
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = self.schedule[self._step_count]
            self._step_count += 1

    def get_lr(self):
        return self.schedule[self._step_count]


def get_dataset_by_name(name: str, root: str, transforms: Callable):
    """Helper function to return a speciffic dataset configuration given it's name"""
    if name == "cre-synthetic":
        return CREStereo(root=root, transforms=transforms)
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
        return FallingThingsStereo(root=root, transforms=transforms, split=["both"])


def shuffle_dataset(dataset):
    """Shuffle the dataset"""
    perm = torch.randperm(len(dataset))
    return torch.utils.data.Subset(dataset, perm)


def get_train_dataset(dataset_root: str, args: argparse.Namespace):
    datasets = []
    for dataset_name in args.train_dataset:
        datasets.append(
            get_dataset_by_name(dataset_name, dataset_root, StereoMatchingPresetCRETrain(crop_size=(384, 512)))
        )

    if len(datasets) == 0:
        raise ValueError("No datasets specified for training")

    samples_per_step = args.world_size * args.batch_size

    for idx, (dataset, steps_per_dataset) in enumerate(zip(datasets, args.dataset_steps)):
        # compute how much we have to expand the dataset to make it fit the desired number of steps
        dataset_size = len(dataset)
        # compute how many steps we can make by default with the dataset as is
        steps_in_dataset = ceil(dataset_size / samples_per_step)
        # see how how much we have to enlarge the dataset by
        dataset_multiplier = steps_per_dataset / steps_in_dataset
        # expand the dataset by the desired amount
        datasets[idx] = dataset * dataset_multiplier
        datasets[idx] = shuffle_dataset(datasets[idx])

    dataset = torch.utils.data.ConcatDataset(datasets)


@torch.no_grad()
def _evaluate(model, args, val_dataset, *, padder_mode, num_flow_updates=None, batch_size=None, header=None):
    """Helper function to compute various metrics (epe, etc.) for a model on a given dataset.

    We process as many samples as possible with ddp, and process the rest on a single worker.
    """
    batch_size = batch_size or args.batch_size
    device = torch.device(args.device)

    model.eval()

    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        sampler = torch.utils.data.SequentialSampler(val_dataset)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        sampler=sampler,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=args.workers,
    )

    num_flow_updates = num_flow_updates or args.num_flow_updates

    def inner_loop(blob):
        if blob[0].dim() == 3:
            # input is not batched so we add an extra dim for consistency
            blob = [x[None, :, :, :] if x is not None else None for x in blob]

        image1, image2, flow_gt = blob[:3]
        valid_flow_mask = None if len(blob) == 3 else blob[-1]

        image1, image2 = image1.to(device), image2.to(device)

        padder = utils.InputPadder(image1.shape, mode=padder_mode)
        image1, image2 = padder.pad(image1, image2)

        flow_predictions = model(image1, image2, num_flow_updates=num_flow_updates)
        flow_pred = flow_predictions[-1]
        flow_pred = padder.unpad(flow_pred).cpu()

        metrics, num_pixels_tot = utils.compute_metrics(flow_pred, flow_gt, valid_flow_mask)

        # We compute per-pixel epe (epe) and per-image epe (called f1-epe in RAFT paper).
        # per-pixel epe: average epe of all pixels of all images
        # per-image epe: average epe on each image independently, then average over images
        for name in ("epe", "1px", "3px", "5px", "f1"):  # f1 is called f1-all in paper
            logger.meters[name].update(metrics[name], n=num_pixels_tot)
        logger.meters["per_image_epe"].update(metrics["epe"], n=batch_size)

    logger = utils.MetricLogger()
    for meter_name in ("epe", "1px", "3px", "5px", "per_image_epe", "f1"):
        logger.add_meter(meter_name, fmt="{global_avg:.4f}")

    num_processed_samples = 0
    for blob in logger.log_every(val_loader, header=header, print_freq=None):
        inner_loop(blob)
        num_processed_samples += blob[0].shape[0]  # batch size

    if args.distributed:
        num_processed_samples = utils.reduce_across_processes(num_processed_samples)
        print(
            f"Batch-processed {num_processed_samples} / {len(val_dataset)} samples. "
            "Going to process the remaining samples individually, if any."
        )
        if args.rank == 0:  # we only need to process the rest on a single worker
            for i in range(num_processed_samples, len(val_dataset)):
                inner_loop(val_dataset[i])

        logger.synchronize_between_processes()

    print(header, logger)


def evaluate(model, args):
    val_datasets = args.val_dataset or []

    if args.weights and args.test_only:
        weights = torchvision.models.get_weight(args.weights)
        trans = weights.transforms()

        def preprocessing(img1, img2, flow, valid_flow_mask):
            img1, img2 = trans(img1, img2)
            if flow is not None and not isinstance(flow, torch.Tensor):
                flow = torch.from_numpy(flow)
            if valid_flow_mask is not None and not isinstance(valid_flow_mask, torch.Tensor):
                valid_flow_mask = torch.from_numpy(valid_flow_mask)
            return img1, img2, flow, valid_flow_mask

    else:
        preprocessing = StereoMatchingPresetEval()

    for name in val_datasets:
        if name == "Middlebury":
            if args.batch_size != 1 and (not args.distributed or args.rank == 0):
                warnings.warn(
                    f"Batch-size={args.batch_size} was passed. For technical reasons, evaluating on Middlebury can only be done with a batch-size of 1."
                )

            val_dataset = Middlebury2014Stereo(root=args.dataset_root, split="train", transforms=preprocessing)
            _evaluate(
                model, args, val_dataset, num_flow_updates=24, padder_mode="kitti", header="Kitti val", batch_size=1
            )
        else:
            warnings.warn(f"Can't validate on {val_dataset}, skipping.")


def train(model, optimizer, scheduler, train_loader, logger, args):
    device = torch.device(args.device)
    for data_blob in logger.log_every(train_loader):

        optimizer.zero_grad()

        image1, image2, flow_gt, valid_flow_mask = (x.to(device) for x in data_blob)
        flow_predictions = model(image1, image2, num_flow_updates=args.num_flow_updates)

        loss = utils.sequence_loss(flow_predictions, flow_gt, valid_flow_mask, args.gamma)
        metrics, _ = utils.compute_metrics(flow_predictions[-1], flow_gt, valid_flow_mask)

        metrics.pop("f1")
        logger.update(loss=loss, **metrics)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        optimizer.step()
        scheduler.step()

        if not args.distributed or args.rank == 0:
            model_without_ddp = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
                "args": args,
            }
            torch.save(checkpoint, Path(args.output_dir) / f"{args.name}_{step}.pth")
            torch.save(checkpoint, Path(args.output_dir) / f"{args.name}.pth")


def main(args):
    utils.setup_ddp(args)
    args.test_only = args.train_dataset is None

    if args.distributed and args.device == "cpu":
        raise ValueError("The device must be cuda if we want to run in distributed mode using torchrun")
    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    model = torchvision.prototype.model.depth.stereo.__dict__[args.model](weights=args.weights)

    if args.distributed:
        model = model.to(args.local_rank)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
        model_without_ddp = model.module
    else:
        model.to(device)
        model_without_ddp = model

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])

    if args.test_only:
        # Set deterministic CUDNN algorithms, since they can affect epe a fair bit.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        evaluate(model, args)
        return

    print(f"Parameter Count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    train_dataset = get_train_dataset(args.train_dataset, args.dataset_root)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=args.adamw_eps)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=ceil(len(train_dataset) / (args.world_size * args.batch_size)),
        pct_start=0.05,
        cycle_momentum=False,
        anneal_strategy="linear",
    )

    if args.resume is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
    else:
        args.start_epoch = 0

    torch.backends.cudnn.benchmark = True

    model.train()
    if args.freeze_batch_norm:
        utils.freeze_batch_norm(model.module)

    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, drop_last=True)
    else:
        sampler = torch.utils.data.RandomSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.workers,
    )

    logger = utils.MetricLogger()

    done = False
    for epoch in range(args.start_epoch, args.epochs):
        train(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            logger=logger,
            args=args,
        )

    if epoch % args.val_freq == 0 or done:
        evaluate(model, args)
        model.train()
        if args.freeze_batch_norm:
            utils.freeze_batch_norm(model.module)


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(add_help=add_help, description="Train or evaluate an optical-flow model.")
    # saving and loading parameters
    parser.add_argument(
        "--name",
        default="raft",
        type=str,
        help="The name of the experiment - determines the name of the files where weights are saved.",
    )
    parser.add_argument("--output-dir", default=".", type=str, help="Output dir where checkpoints will be stored.")
    parser.add_argument(
        "--resume",
        type=str,
        help="A path to previously saved weights. Used to re-start training from, or evaluate a pre-saved model.",
    )

    # dataloader parameters
    parser.add_argument("--workers", type=int, default=12, help="Number of workers for the data loading part.")
    parser.add_argument("--shuffle", type=bool, default=False, help="Wether or not to shuffle dataset indices.")

    # train dataset configuration parameters
    parser.add_argument(
        "--train-dataset",
        type=str,
        nargs="+",
        help="The dataset(s) to use for training. If not passed, only validation is performed (and you probably want to pass --resume).",
    )
    parser.add_argument(
        "--dataset-steps",
        type=Optional[int],
        nargs="+",
        help="The number of steps in a dataset chain schedule. If not provided will use the same dataset throughout all the provided number of optimization steps",
    )
    parser.add_argument("--iterations", type=int, default=350_000, help="The total number steps to train.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per GPU.")

    # validation dataset parameters
    parser.add_argument("--val-dataset", type=str, nargs="+", help="The dataset(s) to use for validation.")
    parser.add_argument("--val-freq", type=int, default=10_000, help="Validate every X optimization steps")

    parser.add_argument("--lr", type=float, default=0.00002, help="Learning rate for AdamW optimizer")
    parser.add_argument("--weight-decay", type=float, default=0.00005, help="Weight decay for AdamW optimizer")
    parser.add_argument("--adamw-eps", type=float, default=1e-8, help="eps value for AdamW optimizer")

    parser.add_argument(
        "--freeze-batch-norm", action="store_true", help="Set BatchNorm modules of the model in eval mode."
    )

    parser.add_argument(
        "--model", type=str, default="raft_large", help="The name of the model to use - either raft_large or raft_small"
    )
    # TODO: resume and weights should be in an exclusive arg group

    parser.add_argument(
        "--num_flow_updates",
        type=int,
        default=12,
        help="number of updates (or 'iters') in the update operator of the model.",
    )

    parser.add_argument("--gamma", type=float, default=0.8, help="exponential weighting for loss. Must be < 1.")

    parser.add_argument("--dist-url", default="env://", help="URL used to set up distributed training")

    parser.add_argument(
        "--dataset-root",
        help="Root folder where the datasets are stored. Will be passed as the 'root' parameter of the datasets.",
        required=True,
    )

    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load.")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu, Default: cuda)")
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(exist_ok=True)
    main(args)
