import argparse
import os
import warnings
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
import torch.distributed as dist
import torchvision.models.optical_flow
import torchvision.prototype.models.depth.stereo
import utils
import visualization

from parsing import make_dataset, make_eval_transform, make_train_transform, VALID_DATASETS
from torch import nn
from torchvision.transforms.functional import get_dimensions, InterpolationMode, resize
from utils.metrics import AVAILABLE_METRICS
from utils.norm import freeze_batch_norm


def make_stereo_flow(flow: Union[torch.Tensor, List[torch.Tensor]], model_out_channels: int) -> torch.Tensor:
    """Helper function to make stereo flow from a given model output"""
    if isinstance(flow, list):
        return [make_stereo_flow(flow_i, model_out_channels) for flow_i in flow]

    B, C, H, W = flow.shape
    # we need to add zero flow if the model outputs 2 channels
    if C == 1 and model_out_channels == 2:
        zero_flow = torch.zeros_like(flow)
        # by convention the flow is X-Y axis, so we need the Y flow last
        flow = torch.cat([flow, zero_flow], dim=1)
    return flow


def make_lr_schedule(args: argparse.Namespace, optimizer: torch.optim.Optimizer) -> np.ndarray:
    """Helper function to return a learning rate scheduler for CRE-stereo"""
    if args.decay_after_steps < args.warmup_steps:
        raise ValueError(f"decay_after_steps: {args.function} must be greater than warmup_steps: {args.warmup_steps}")

    warmup_steps = args.warmup_steps if args.warmup_steps else 0
    flat_lr_steps = args.decay_after_steps - warmup_steps if args.decay_after_steps else 0
    decay_lr_steps = args.total_iterations - flat_lr_steps

    max_lr = args.lr
    min_lr = args.min_lr

    schedulers = []
    milestones = []

    if warmup_steps > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_factor, total_iters=warmup_steps
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_factor, total_iters=warmup_steps
            )
        else:
            raise ValueError(f"Unknown lr warmup method {args.lr_warmup_method}")
        schedulers.append(warmup_lr_scheduler)
        milestones.append(warmup_steps)

    if flat_lr_steps > 0:
        flat_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=max_lr, total_iters=flat_lr_steps)
        schedulers.append(flat_lr_scheduler)
        milestones.append(flat_lr_steps + warmup_steps)

    if decay_lr_steps > 0:
        if args.lr_decay_method == "cosine":
            decay_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=decay_lr_steps, eta_min=min_lr
            )
        elif args.lr_decay_method == "linear":
            decay_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=max_lr, end_factor=min_lr, total_iters=decay_lr_steps
            )
        elif args.lr_decay_method == "exponential":
            decay_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=args.lr_decay_gamma, last_epoch=-1
            )
        else:
            raise ValueError(f"Unknown lr decay method {args.lr_decay_method}")
        schedulers.append(decay_lr_scheduler)

    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers, milestones=milestones)
    return scheduler


def shuffle_dataset(dataset):
    """Shuffle the dataset"""
    perm = torch.randperm(len(dataset))
    return torch.utils.data.Subset(dataset, perm)


def resize_dataset_to_n_steps(
    dataset: torch.utils.data.Dataset, dataset_steps: int, samples_per_step: int, args: argparse.Namespace
) -> torch.utils.data.Dataset:
    original_size = len(dataset)
    if args.steps_is_epochs:
        samples_per_step = original_size
    target_size = dataset_steps * samples_per_step

    dataset_copies = []
    n_expands, remainder = divmod(target_size, original_size)
    for idx in range(n_expands):
        dataset_copies.append(dataset)

    if remainder > 0:
        dataset_copies.append(torch.utils.data.Subset(dataset, list(range(remainder))))

    if args.dataset_shuffle:
        dataset_copies = [shuffle_dataset(dataset_copy) for dataset_copy in dataset_copies]

    dataset = torch.utils.data.ConcatDataset(dataset_copies)
    return dataset


def get_train_dataset(dataset_root: str, args: argparse.Namespace) -> torch.utils.data.Dataset:
    datasets = []
    for dataset_name in args.train_datasets:
        transform = make_train_transform(args)
        dataset = make_dataset(dataset_name, dataset_root, transform)
        datasets.append(dataset)

    if len(datasets) == 0:
        raise ValueError("No datasets specified for training")

    samples_per_step = args.world_size * args.batch_size

    for idx, (dataset, steps_per_dataset) in enumerate(zip(datasets, args.dataset_steps)):
        datasets[idx] = resize_dataset_to_n_steps(dataset, steps_per_dataset, samples_per_step, args)

    dataset = torch.utils.data.ConcatDataset(datasets)
    if args.dataset_order_shuffle:
        dataset = shuffle_dataset(dataset)

    print(f"Training dataset: {len(dataset)} samples")
    return dataset


@torch.inference_mode()
def _evaluate(
    model,
    args,
    val_loader,
    *,
    padder_mode,
    print_freq=10,
    writer=None,
    step=None,
    iterations=None,
    batch_size=None,
    header=None,
):
    """Helper function to compute various metrics (epe, etc.) for a model on a given dataset."""
    model.eval()
    header = header or "Test:"
    device = torch.device(args.device)
    metric_logger = utils.MetricLogger(delimiter="  ")

    iterations = iterations or args.recurrent_updates

    logger = utils.MetricLogger()
    for meter_name in args.metrics:
        logger.add_meter(meter_name, fmt="{global_avg:.4f}")
    if "fl-all" not in args.metrics:
        logger.add_meter("fl-all", fmt="{global_avg:.4f}")

    num_processed_samples = 0
    with torch.cuda.amp.autocast(enabled=args.mixed_precision, dtype=torch.float16):
        for blob in metric_logger.log_every(val_loader, print_freq, header):
            image_left, image_right, disp_gt, valid_disp_mask = (x.to(device) for x in blob)
            padder = utils.InputPadder(image_left.shape, mode=padder_mode)
            image_left, image_right = padder.pad(image_left, image_right)

            disp_predictions = model(image_left, image_right, flow_init=None, num_iters=iterations)
            disp_pred = disp_predictions[-1][:, :1, :, :]
            disp_pred = padder.unpad(disp_pred)

            metrics, _ = utils.compute_metrics(disp_pred, disp_gt, valid_disp_mask, metrics=logger.meters.keys())
            num_processed_samples += image_left.shape[0]
            for name in metrics:
                logger.meters[name].update(metrics[name], n=1)

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)

    print("Num_processed_samples: ", num_processed_samples)
    if (
        hasattr(val_loader.dataset, "__len__")
        and len(val_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        warnings.warn(
            f"Number of processed samples {num_processed_samples} is different"
            f"from the dataset size {len(val_loader.dataset)}. This may happen if"
            "the dataset is not divisible by the batch size. Try lowering the batch size or GPU number for more accurate results."
        )

    if writer is not None and args.rank == 0:
        for meter_name, meter_value in logger.meters.items():
            scalar_name = f"{meter_name} {header}"
            writer.add_scalar(scalar_name, meter_value.avg, step)

    logger.synchronize_between_processes()
    print(header, logger)


def make_eval_loader(dataset_name: str, args: argparse.Namespace) -> torch.utils.data.DataLoader:
    if args.weights:
        weights = torchvision.models.get_weight(args.weights)
        trans = weights.transforms()

        def preprocessing(image_left, image_right, disp, valid_disp_mask):
            C_o, H_o, W_o = get_dimensions(image_left)
            image_left, image_right = trans(image_left, image_right)

            C_t, H_t, W_t = get_dimensions(image_left)
            scale_factor = W_t / W_o

            if disp is not None and not isinstance(disp, torch.Tensor):
                disp = torch.from_numpy(disp)
                if W_t != W_o:
                    disp = resize(disp, (H_t, W_t), mode=InterpolationMode.BILINEAR) * scale_factor
            if valid_disp_mask is not None and not isinstance(valid_disp_mask, torch.Tensor):
                valid_disp_mask = torch.from_numpy(valid_disp_mask)
                if W_t != W_o:
                    valid_disp_mask = resize(valid_disp_mask, (H_t, W_t), mode=InterpolationMode.NEAREST)
            return image_left, image_right, disp, valid_disp_mask

    else:
        preprocessing = make_eval_transform(args)

    val_dataset = make_dataset(dataset_name, args.dataset_root, transforms=preprocessing)
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=False)
    else:
        sampler = torch.utils.data.SequentialSampler(val_dataset)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.workers,
    )

    return val_loader


def evaluate(model, loaders, args, writer=None, step=None):
    for loader_name, loader in loaders.items():
        _evaluate(
            model,
            args,
            loader,
            iterations=args.recurrent_updates,
            padder_mode=args.padder_type,
            header=f"{loader_name} evaluation",
            batch_size=args.batch_size,
            writer=writer,
            step=step,
        )


def run(model, optimizer, scheduler, train_loader, val_loaders, logger, writer, scaler, args):
    device = torch.device(args.device)
    # wrap the loader in a logger
    loader = iter(logger.log_every(train_loader))
    # output channels
    model_out_channels = model.module.output_channels if args.distributed else model.output_channels

    torch.set_num_threads(args.threads)

    sequence_criterion = utils.SequenceLoss(
        gamma=args.gamma,
        max_flow=args.max_disparity,
        exclude_large_flows=args.flow_loss_exclude_large,
    ).to(device)

    if args.consistency_weight:
        consistency_criterion = utils.FlowSequenceConsistencyLoss(
            args.gamma,
            resize_factor=0.25,
            rescale_factor=0.25,
            rescale_mode="bilinear",
        ).to(device)
    else:
        consistency_criterion = None

    if args.psnr_weight:
        psnr_criterion = utils.PSNRLoss().to(device)
    else:
        psnr_criterion = None

    if args.smoothness_weight:
        smoothness_criterion = utils.SmoothnessLoss().to(device)
    else:
        smoothness_criterion = None

    if args.photometric_weight:
        photometric_criterion = utils.FlowPhotoMetricLoss(
            ssim_weight=args.photometric_ssim_weight,
            max_displacement_ratio=args.photometric_max_displacement_ratio,
            ssim_use_padding=False,
        ).to(device)
    else:
        photometric_criterion = None

    for step in range(args.start_step + 1, args.total_iterations + 1):
        data_blob = next(loader)
        optimizer.zero_grad()

        # unpack the data blob
        image_left, image_right, disp_mask, valid_disp_mask = (x.to(device) for x in data_blob)
        with torch.cuda.amp.autocast(enabled=args.mixed_precision, dtype=torch.float16):
            disp_predictions = model(image_left, image_right, flow_init=None, num_iters=args.recurrent_updates)
            # different models have different outputs, make sure we get the right ones for this task
            disp_predictions = make_stereo_flow(disp_predictions, model_out_channels)
            # should the architecture or training loop require it, we have to adjust the disparity mask
            # target to possibly look like an optical flow mask
            disp_mask = make_stereo_flow(disp_mask, model_out_channels)
            # sequence loss on top of the model outputs

        loss = sequence_criterion(disp_predictions, disp_mask, valid_disp_mask) * args.flow_loss_weight

        if args.consistency_weight > 0:
            loss_consistency = consistency_criterion(disp_predictions)
            loss += loss_consistency * args.consistency_weight

        if args.psnr_weight > 0:
            loss_psnr = 0.0
            for pred in disp_predictions:
                # predictions might have 2 channels
                loss_psnr += psnr_criterion(
                    pred * valid_disp_mask.unsqueeze(1),
                    disp_mask * valid_disp_mask.unsqueeze(1),
                ).mean()  # mean the psnr loss over the batch
            loss += loss_psnr / len(disp_predictions) * args.psnr_weight

        if args.photometric_weight > 0:
            loss_photometric = 0.0
            for pred in disp_predictions:
                # predictions might have 1 channel, therefore we need to inpute 0s for the second channel
                if model_out_channels == 1:
                    pred = torch.cat([pred, torch.zeros_like(pred)], dim=1)

                loss_photometric += photometric_criterion(
                    image_left, image_right, pred, valid_disp_mask
                )  # photometric loss already comes out meaned over the batch
            loss += loss_photometric / len(disp_predictions) * args.photometric_weight

        if args.smoothness_weight > 0:
            loss_smoothness = 0.0
            for pred in disp_predictions:
                # predictions might have 2 channels
                loss_smoothness += smoothness_criterion(
                    image_left, pred[:, :1, :, :]
                ).mean()  # mean the smoothness loss over the batch
            loss += loss_smoothness / len(disp_predictions) * args.smoothness_weight

        with torch.no_grad():
            metrics, _ = utils.compute_metrics(
                disp_predictions[-1][:, :1, :, :],  # predictions might have 2 channels
                disp_mask[:, :1, :, :],  # so does the ground truth
                valid_disp_mask,
                args.metrics,
            )

        metrics.pop("fl-all", None)
        logger.update(loss=loss, **metrics)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)
            optimizer.step()

        scheduler.step()

        if not dist.is_initialized() or dist.get_rank() == 0:
            if writer is not None and step % args.tensorboard_log_frequency == 0:
                # log the loss and metrics to tensorboard

                writer.add_scalar("loss", loss, step)
                for name, value in logger.meters.items():
                    writer.add_scalar(name, value.avg, step)
                # log the images to tensorboard
                pred_grid = visualization.make_training_sample_grid(
                    image_left, image_right, disp_mask, valid_disp_mask, disp_predictions
                )
                writer.add_image("predictions", pred_grid, step, dataformats="HWC")

                # second thing we want to see is how relevant the iterative refinement is
                pred_sequence_grid = visualization.make_disparity_sequence_grid(disp_predictions, disp_mask)
                writer.add_image("sequence", pred_sequence_grid, step, dataformats="HWC")

        if step % args.save_frequency == 0:
            if not args.distributed or args.rank == 0:
                model_without_ddp = (
                    model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
                )
                checkpoint = {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": step,
                    "args": args,
                }
                os.makedirs(args.checkpoint_dir, exist_ok=True)
                torch.save(checkpoint, Path(args.checkpoint_dir) / f"{args.name}_{step}.pth")
                torch.save(checkpoint, Path(args.checkpoint_dir) / f"{args.name}.pth")

        if step % args.valid_frequency == 0:
            evaluate(model, val_loaders, args, writer, step)
            model.train()
            if args.freeze_batch_norm:
                if isinstance(model, nn.parallel.DistributedDataParallel):
                    freeze_batch_norm(model.module)
                else:
                    freeze_batch_norm(model)

    # one final save at the end
    if not args.distributed or args.rank == 0:
        model_without_ddp = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        checkpoint = {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "args": args,
        }
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        torch.save(checkpoint, Path(args.checkpoint_dir) / f"{args.name}_{step}.pth")
        torch.save(checkpoint, Path(args.checkpoint_dir) / f"{args.name}.pth")


def main(args):
    args.total_iterations = sum(args.dataset_steps)

    # initialize DDP setting
    utils.setup_ddp(args)
    print(args)

    args.test_only = args.train_datasets is None

    # set the appropriate devices
    if args.distributed and args.device == "cpu":
        raise ValueError("The device must be cuda if we want to run in distributed mode using torchrun")
    device = torch.device(args.device)

    # select model architecture
    model = torchvision.prototype.models.depth.stereo.__dict__[args.model](weights=args.weights)

    # convert to DDP if need be
    if args.distributed:
        model = model.to(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    else:
        model.to(device)
        model_without_ddp = model

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    val_loaders = {name: make_eval_loader(name, args) for name in args.test_datasets}

    # EVAL ONLY configurations
    if args.test_only:
        evaluate(model, val_loaders, args)
        return

    # Sanity check for the parameter count
    print(f"Parameter Count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Compose the training dataset
    train_dataset = get_train_dataset(args.dataset_root, args)

    # initialize the optimizer
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer {args.optimizer}. Please choose between adam and sgd")

    # initialize the learning rate schedule
    scheduler = make_lr_schedule(args, optimizer)

    # load them from checkpoint if needed
    args.start_step = 0
    if args.resume_path is not None:
        checkpoint = torch.load(args.resume_path, map_location="cpu")
        if "model" in checkpoint:
            # this means the user requested to resume from a training checkpoint
            model_without_ddp.load_state_dict(checkpoint["model"])
            # this means the user wants to continue training from where it was left off
            if args.resume_schedule:
                optimizer.load_state_dict(checkpoint["optimizer"])
                scheduler.load_state_dict(checkpoint["scheduler"])
                args.start_step = checkpoint["step"] + 1
                # modify starting point of the dat
                sample_start_step = args.start_step * args.batch_size * args.world_size
                train_dataset = train_dataset[sample_start_step:]

        else:
            # this means the user wants to finetune on top of a model state dict
            # and that no other changes are required
            model_without_ddp.load_state_dict(checkpoint)

    torch.backends.cudnn.benchmark = True

    # enable training mode
    model.train()
    if args.freeze_batch_norm:
        freeze_batch_norm(model_without_ddp)

    # put dataloader on top of the dataset
    # make sure to disable shuffling since the dataset is already shuffled
    # in order to guarantee quasi randomness whilst retaining a deterministic
    # dataset consumption order
    if args.distributed:
        # the train dataset is preshuffled in order to respect the iteration order
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=False, drop_last=True)
    else:
        # the train dataset is already shuffled, so we can use a simple SequentialSampler
        sampler = torch.utils.data.SequentialSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.workers,
    )

    # initialize the logger
    if args.tensorboard_summaries:
        from torch.utils.tensorboard import SummaryWriter

        tensorboard_path = Path(args.checkpoint_dir) / "tensorboard"
        os.makedirs(tensorboard_path, exist_ok=True)

        tensorboard_run = tensorboard_path / f"{args.name}"
        writer = SummaryWriter(tensorboard_run)
    else:
        writer = None

    logger = utils.MetricLogger(delimiter="  ")

    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None
    # run the training loop
    # this will perform optimization, respectively logging and saving checkpoints
    # when need be
    run(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loaders=val_loaders,
        logger=logger,
        writer=writer,
        scaler=scaler,
        args=args,
    )


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Stereo Matching Training", add_help=add_help)
    # checkpointing
    parser.add_argument("--name", default="crestereo", help="name of the experiment")
    parser.add_argument("--resume", type=str, default=None, help="from which checkpoint to resume")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="path to the checkpoint directory")

    # dataset
    parser.add_argument("--dataset-root", type=str, default="", help="path to the dataset root directory")
    parser.add_argument(
        "--train-datasets",
        type=str,
        nargs="+",
        default=["crestereo"],
        help="dataset(s) to train on",
        choices=list(VALID_DATASETS.keys()),
    )
    parser.add_argument(
        "--dataset-steps", type=int, nargs="+", default=[300_000], help="number of steps for each dataset"
    )
    parser.add_argument(
        "--steps-is-epochs", action="store_true", help="if set, dataset-steps are interpreted as epochs"
    )
    parser.add_argument(
        "--test-datasets",
        type=str,
        nargs="+",
        default=["middlebury2014-train"],
        help="dataset(s) to test on",
        choices=["middlebury2014-train"],
    )
    parser.add_argument("--dataset-shuffle", type=bool, help="shuffle the dataset", default=True)
    parser.add_argument("--dataset-order-shuffle", type=bool, help="shuffle the dataset order", default=True)
    parser.add_argument("--batch-size", type=int, default=2, help="batch size per GPU")
    parser.add_argument("--workers", type=int, default=4, help="number of workers per GPU")
    parser.add_argument(
        "--threads",
        type=int,
        default=16,
        help="number of CPU threads per GPU. This can be changed around to speed-up transforms if needed. This can lead to worker thread contention so use with care.",
    )

    # model architecture
    parser.add_argument(
        "--model",
        type=str,
        default="crestereo_base",
        help="model architecture",
        choices=["crestereo_base", "raft_stereo"],
    )
    parser.add_argument("--recurrent-updates", type=int, default=10, help="number of recurrent updates")
    parser.add_argument("--freeze-batch-norm", action="store_true", help="freeze batch norm parameters")

    # loss parameters
    parser.add_argument("--gamma", type=float, default=0.8, help="gamma parameter for the flow sequence loss")
    parser.add_argument("--flow-loss-weight", type=float, default=1.0, help="weight for the flow loss")
    parser.add_argument(
        "--flow-loss-exclude-large",
        action="store_true",
        help="exclude large flow values from the loss. A large value is defined as a value greater than the ground truth flow norm",
        default=False,
    )
    parser.add_argument("--consistency-weight", type=float, default=0.0, help="consistency loss weight")
    parser.add_argument(
        "--consistency-resize-factor",
        type=float,
        default=0.25,
        help="consistency loss resize factor to account for the fact that the flow is computed on a downsampled image",
    )
    parser.add_argument("--psnr-weight", type=float, default=0.0, help="psnr loss weight")
    parser.add_argument("--smoothness-weight", type=float, default=0.0, help="smoothness loss weight")
    parser.add_argument("--photometric-weight", type=float, default=0.0, help="photometric loss weight")
    parser.add_argument(
        "--photometric-max-displacement-ratio",
        type=float,
        default=0.15,
        help="Only pixels with a displacement smaller than this ratio of the image width will be considered for the photometric loss",
    )
    parser.add_argument("--photometric-ssim-weight", type=float, default=0.85, help="photometric ssim loss weight")

    # transforms parameters
    parser.add_argument("--gpu-transforms", action="store_true", help="use GPU transforms")
    parser.add_argument(
        "--eval-size", type=int, nargs="+", default=[384, 512], help="size of the images for evaluation"
    )
    parser.add_argument("--resize-size", type=int, nargs=2, default=None, help="resize size")
    parser.add_argument("--crop-size", type=int, nargs=2, default=[384, 512], help="crop size")
    parser.add_argument("--scale-range", type=float, nargs=2, default=[0.6, 1.0], help="random scale range")
    parser.add_argument("--rescale-prob", type=float, default=1.0, help="probability of resizing the image")
    parser.add_argument(
        "--scaling-type", type=str, default="linear", help="scaling type", choices=["exponential", "linear"]
    )
    parser.add_argument("--flip-prob", type=float, default=0.5, help="probability of flipping the image")
    parser.add_argument(
        "--norm-mean", type=float, nargs="+", default=[0.5, 0.5, 0.5], help="mean for image normalization"
    )
    parser.add_argument(
        "--norm-std", type=float, nargs="+", default=[0.5, 0.5, 0.5], help="std for image normalization"
    )
    parser.add_argument(
        "--use-grayscale", action="store_true", help="use grayscale images instead of RGB", default=False
    )
    parser.add_argument("--max-disparity", type=float, default=None, help="maximum disparity")
    parser.add_argument(
        "--interpolation-strategy",
        type=str,
        default="bilinear",
        help="interpolation strategy",
        choices=["bilinear", "bicubic", "mixed"],
    )
    parser.add_argument("--spatial-shift-prob", type=float, default=1.0, help="probability of shifting the image")
    parser.add_argument(
        "--spatial-shift-max-angle", type=float, default=0.1, help="maximum angle for the spatial shift"
    )
    parser.add_argument(
        "--spatial-shift-max-displacement", type=float, default=2.0, help="maximum displacement for the spatial shift"
    )
    parser.add_argument("--gamma-range", type=float, nargs="+", default=[0.8, 1.2], help="range for gamma correction")
    parser.add_argument(
        "--brightness-range", type=float, nargs="+", default=[0.8, 1.2], help="range for brightness correction"
    )
    parser.add_argument(
        "--contrast-range", type=float, nargs="+", default=[0.8, 1.2], help="range for contrast correction"
    )
    parser.add_argument(
        "--saturation-range", type=float, nargs="+", default=0.0, help="range for saturation correction"
    )
    parser.add_argument("--hue-range", type=float, nargs="+", default=0.0, help="range for hue correction")
    parser.add_argument(
        "--asymmetric-jitter-prob",
        type=float,
        default=1.0,
        help="probability of using asymmetric jitter instead of symmetric jitter",
    )
    parser.add_argument("--occlusion-prob", type=float, default=0.5, help="probability of occluding the rightimage")
    parser.add_argument(
        "--occlusion-px-range", type=int, nargs="+", default=[50, 100], help="range for the number of occluded pixels"
    )
    parser.add_argument("--erase-prob", type=float, default=0.0, help="probability of erasing in both images")
    parser.add_argument(
        "--erase-px-range", type=int, nargs="+", default=[50, 100], help="range for the number of erased pixels"
    )
    parser.add_argument(
        "--erase-num-repeats", type=int, default=1, help="number of times to repeat the erase operation"
    )

    # optimizer parameters
    parser.add_argument("--optimizer", type=str, default="adam", help="optimizer", choices=["adam", "sgd"])
    parser.add_argument("--lr", type=float, default=4e-4, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay")
    parser.add_argument("--clip-grad-norm", type=float, default=0.0, help="clip grad norm")

    # lr_scheduler parameters
    parser.add_argument("--min-lr", type=float, default=2e-5, help="minimum learning rate")
    parser.add_argument("--warmup-steps", type=int, default=6_000, help="number of warmup steps")
    parser.add_argument(
        "--decay-after-steps", type=int, default=180_000, help="number of steps after which to start decay the lr"
    )
    parser.add_argument(
        "--lr-warmup-method", type=str, default="linear", help="warmup method", choices=["linear", "cosine"]
    )
    parser.add_argument("--lr-warmup-factor", type=float, default=0.02, help="warmup factor for the learning rate")
    parser.add_argument(
        "--lr-decay-method",
        type=str,
        default="linear",
        help="decay method",
        choices=["linear", "cosine", "exponential"],
    )
    parser.add_argument("--lr-decay-gamma", type=float, default=0.8, help="decay factor for the learning rate")

    # deterministic behaviour
    parser.add_argument("--seed", type=int, default=42, help="seed for random number generators")

    # mixed precision training
    parser.add_argument("--mixed-precision", action="store_true", help="use mixed precision training")

    # logging
    parser.add_argument("--tensorboard-summaries", action="store_true", help="log to tensorboard")
    parser.add_argument("--tensorboard-log-frequency", type=int, default=100, help="log frequency")
    parser.add_argument("--save-frequency", type=int, default=1_000, help="save frequency")
    parser.add_argument("--valid-frequency", type=int, default=1_000, help="validation frequency")
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["mae", "rmse", "1px", "3px", "5px", "relepe"],
        help="metrics to log",
        choices=AVAILABLE_METRICS,
    )

    # distributed parameters
    parser.add_argument("--world-size", type=int, default=8, help="number of distributed processes")
    parser.add_argument("--dist-url", type=str, default="env://", help="url used to set up distributed training")
    parser.add_argument("--device", type=str, default="cuda", help="device to use for training")

    # weights API
    parser.add_argument("--weights", type=str, default=None, help="weights API url")
    parser.add_argument(
        "--resume-path", type=str, default=None, help="a path from which to resume or start fine-tuning"
    )
    parser.add_argument("--resume-schedule", action="store_true", help="resume optimizer state")

    # padder parameters
    parser.add_argument("--padder-type", type=str, default="kitti", help="padder type", choices=["kitti", "sintel"])
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
