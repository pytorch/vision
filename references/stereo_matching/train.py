import argparse
import os
import random
import uuid
import warnings
from math import ceil
from pathlib import Path
from typing import Callable, List, Union
from torch import nn
import numpy as np
import torch
import torch.distributed as dist
import torchvision.models.optical_flow
import torchvision.prototype.models.depth.stereo
import utils
import yaml
from presets import HighScaleStereoMatchingPresetCRETrain, LowScaleStereoMatchingPresetCRETrain, MidScaleStereoMatchingPresetCRETrain, MiddleBurryEvalPreset, StereoMatchingPresetEval, StereoMatchingPresetCRETrain, SuperHighScaleStereoMatchingPresetCRETrain, SuperLowScaleStereoMatchingPresetCRETrain, SuperWideScaleStereoMatchingPresetCRETrain
from utils import get_dataset_by_name


from visualization import *

from torch.utils.tensorboard import SummaryWriter

def construct_experiment_name(args: dict):
    # return a random uuid if no experiment name is given
    return str(uuid.uuid4())

def make_stereo_flow(flow: torch.Tensor) -> torch.Tensor:
    """Helper function to make stereo flow from a given model output"""
    B, C, H, W = flow.shape
    # we need to add zero flow
    if C == 1:
        zero_flow = torch.zeros_like(flow)
        # by convention the flow is X-Y axis, so we need the Y flow last
        flow = torch.cat([flow, zero_flow], dim=1)
    return flow

def view_flow_as_stereo_target(
    model_ouputs: Union[torch.Tensor, List[torch.Tensor]], args: argparse.Namespace
) -> torch.Tensor: 
    """Helper function to construct the model outputs for a given architecture"""
    if args.should_learn_y_flow:
        cut_off_index = 2
    else:
        cut_off_index = 1
        
    if isinstance(model_ouputs, list) and args.should_learn_y_flow:
        for idx in range(len(model_ouputs)):
            model_ouputs[idx] = make_stereo_flow(model_ouputs[idx])
    else:
        model_ouputs = make_stereo_flow(model_ouputs)
                            
    if "raft" in args.model:
        return model_ouputs
    
    elif "crestereo" in args.model:
        # CRE-stereo like models, return the entire flow-map
        # we are interested only on the X-axis flow for stereo matching
        if isinstance(model_ouputs, list):
            outs = list(map(lambda x: x[:, :cut_off_index, :, :], model_ouputs))
            return outs
        else:
            return model_ouputs[:, :cut_off_index, :, :]
    
    else:
        raise ValueError(f"Unknown model {args.model}")


def make_cre_stereo_schedule(args: argparse.Namespace, optimizer: torch.optim.Optimizer) -> np.ndarray:
    """Helper function to return a learning rate scheduler for CRE-stereo"""
    warmup_steps = args.warmup_steps if args.warmup_steps else 0
    flat_lr_steps = args.flat_lr_steps - warmup_steps if args.flat_lr_steps else 0
    decay_lr_steps = args.total_iterations - flat_lr_steps 

    max_lr = args.lr
    min_lr = args.min_lr
    
    schedulers = []
    
    if args.lr_decay_method == "cosine-restart":
        scheduler = utils.ConsinAnnealingWarmupRestarts(
            optimizer=optimizer,
            T_0=args.period_lr_steps,
            T_mult=args.period_lr_mult,
            T_warmup=warmup_steps,
            eta_min=min_lr,
            gamma=args.lr_decay_gamma,
        )
        
        return scheduler
    
    if warmup_steps > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=warmup_steps
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=warmup_steps
            )
        else:
            raise ValueError(f"Unknown lr warmup method {args.lr_warmup_method}")
        schedulers.append(warmup_lr_scheduler)
        
    if flat_lr_steps > 0:
        flat_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=max_lr, total_iters=flat_lr_steps)
        schedulers.append(flat_lr_scheduler)
        
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
        
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers, milestones=[warmup_steps, flat_lr_steps + warmup_steps])
    return scheduler


def get_transforms(args: argparse.Namespace):
    if args.transforms == "cre-low-scale":
        return LowScaleStereoMatchingPresetCRETrain
    elif args.transforms == "cre-high-scale":
        return HighScaleStereoMatchingPresetCRETrain
    elif args.transforms == "cre-super-wide-scale":
        return SuperWideScaleStereoMatchingPresetCRETrain
    elif args.transforms == "cre-super-high-scale":
        return SuperHighScaleStereoMatchingPresetCRETrain
    elif args.transforms == "cre-mid-scale":
        return MidScaleStereoMatchingPresetCRETrain
    elif args.transforms == "cre-super-low-scale":
        return SuperLowScaleStereoMatchingPresetCRETrain
    else:
        raise ValueError(f"Unknown transforms {args.transforms}")

def shuffle_dataset(dataset):
    """Shuffle the dataset"""
    perm = torch.randperm(len(dataset))
    return torch.utils.data.Subset(dataset, perm)


def get_train_dataset(dataset_root: str, args: argparse.Namespace):
    datasets = []
    for dataset_name in args.train_dataset:
        transform = get_transforms(args)
        datasets.append(
            get_dataset_by_name(dataset_name, dataset_root, transform)
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
        # we shuffle the dataset "domain-wise" dataset in order to
        # avoid undesirable learning dynamics from consecutive frames
        # whilst still preserving the training schedule speciffied by the user
        if args.dataset_shuffle:
            datasets[idx] = shuffle_dataset(datasets[idx])

    dataset = torch.utils.data.ConcatDataset(datasets)
    if args.schedule_shuffle:
        dataset = shuffle_dataset(dataset)

    print(f"Training dataset: {len(dataset)} samples")
    return dataset    

@torch.no_grad()
def _evaluate(model, args, val_dataset, *, padder_mode, writter=None, step=None, iterations=None, batch_size=None, header=None):
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

    iterations = iterations or args.recurrent_updates

    def inner_loop(blob):
        if blob[0].dim() == 3:
            # input is not batched so we add an extra dim for consistency
            blob = [x.unsqueeze(0) if x is not None else None for x in blob]

        image_left, image_right, disp_gt = blob[:3]
        valid_disp_mask = None if len(blob) == 3 else blob[-1]

        image_left, image_right = image_left.to(device), image_right.to(device)

        padder = utils.InputPadder(image_left.shape, mode=padder_mode)
        image_left, image_right = padder.pad(image_left, image_right)

        if "crestereo" in args.model:
            # TODO: this needs to be abstracted somehow
            disp_predictions = model(image_left, image_right, flow_init=None, iterations=args.recurrent_updates)
        elif "raft" in args.model:
            disp_predictions = model(image_left, image_right, args.recurrent_updates)
        else:
            raise ValueError(f"Unknown model {args.model}")
        # different models have different outputs, make sure we get the right ones for this task
        disp_predictions = view_flow_as_stereo_target(disp_predictions, args)
        disp_pred = disp_predictions[-1]
        disp_pred = padder.unpad(disp_pred).cpu()

        metrics, num_pixels_tot = utils.compute_metrics(disp_pred, disp_gt, valid_disp_mask)

        # We compute per-pixel epe (epe) and per-image epe (called f1-epe in RAFT paper).
        # per-pixel epe: average epe of all pixels of all images
        # per-image epe: average epe on each image independently, then average over images
        for name in ("epe", "1px", "3px", "5px", "f1"):  # f1 is called f1-all in paper
            logger.meters[name].update(metrics[name], n=num_pixels_tot)
        logger.meters["per_image_epe"].update(metrics["epe"], n=batch_size)

    logger = utils.MetricLogger(log_dir=args.checkpoint_dir, log_name=f"{args.experiment_name}_val.log")
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

    if writter is not None:
        for meter_name, meter_value in logger.meters.items():
            scalar_name = f"{meter_name} {header}"
            writter.add_scalar(scalar_name, meter_value.avg, step)
    print(header, logger)


def evaluate(model, args, writter=None, step=None):
    val_datasets = args.valid_dataset or []

    if args.weights and args.test_only:
        weights = torchvision.models.get_weight(args.weights)
        trans = weights.transforms()

        def preprocessing(image_left, image_right, disp, valid_disp_mask):
            image_left, image_right = trans(image_left, image_right)
            if disp is not None and not isinstance(disp, torch.Tensor):
                disp = torch.from_numpy(disp)
            if valid_disp_mask is not None and not isinstance(valid_disp_mask, torch.Tensor):
                valid_disp_mask = torch.from_numpy(valid_disp_mask)
            return image_left, image_right, disp, valid_disp_mask

    else:
        preprocessing = MiddleBurryEvalPreset

    for name in val_datasets:
        if args.batch_size != 1 and (not args.distributed or args.rank == 0):
            warnings.warn(
                f"Batch-size={args.batch_size} was passed. For technical reasons, evaluating on {name} can only be done with a batch-size of 1."
            )

        val_dataset = get_dataset_by_name(root=args.dataset_root, name=name, transforms=preprocessing)
        _evaluate(
            model,
            args,
            val_dataset,
            iterations=args.recurrent_updates * 2,
            padder_mode="kitti",
            header=f"{name} evaluation",
            batch_size=1,
            writter=writter,
            step=step,
        )
        


def run(model, optimizer, scheduler, train_loader, logger, writer, scaler, args):
    device = torch.device(args.device)
    # wrap the loader in a logger
    loader = iter(logger.log_every(train_loader))
    # buffered weights for the loss
    loss_weights = None
    # uncomment to profile

    """
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=5, active=3, repeat=2, skip_first=100),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./crestereo_log'),
        profile_memory=True,
        with_stack=True,
        record_shapes=True,
        with_flops=True,
    ) as prof:
    """
    torch.set_num_threads(16)
    for step in range(args.start_step + 1, args.total_iterations + 1):
        data_blob = next(loader)
        optimizer.zero_grad()

        # unpack the data blob
        image_left, image_right, disp_mask, valid_disp_mask = (x.to(device) for x in data_blob)
        # some models might be doing recurrent updates or refinements
        # therefore we make a simple check for compatibility purposes
        # if you are reusing or modiffying this piece of code, make sure to
        # adjust it to your architecture naming scheme
        with torch.cuda.amp.autocast(enabled=args.mixed_precision, dtype=torch.float16):
            if "crestereo" in args.model:
                # TODO: this needs to be abstracted somehow
                disp_predictions = model(image_left, image_right, flow_init=None, iterations=args.recurrent_updates)
            elif "raft" in args.model:
                disp_predictions = model(image_left, image_right, args.recurrent_updates)
            else:
                raise ValueError(f"Unknown model {args.model}")
            # different models have different outputs, make sure we get the right ones for this task
            disp_predictions = view_flow_as_stereo_target(disp_predictions, args)
            # should the architecture or training loop require it, we have to adjust the disparity mask
            # target to possibly look like an optical flow mask
            disp_mask = view_flow_as_stereo_target(disp_mask, args)
            # sequence loss on top of the model outputs
        
        loss, loss_weights = utils.sequence_loss(disp_predictions, disp_mask, valid_disp_mask, args.gamma, weights=loss_weights)
        if args.consistency_alpha > 0:
            loss += utils.sequence_consistency_loss(disp_predictions, args.gamma) * args.consistency_alpha
            
        metrics, _ = utils.compute_metrics(disp_predictions[-1], disp_mask, valid_disp_mask)

        metrics.pop("f1")
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
            if writer is not None and step % args.write_summary_freq == 0:
                # log the loss and metrics to tensorboard
                writer.add_scalar("loss", loss, step)
                for name, value in logger.meters.items():
                    writer.add_scalar(name, value.avg, step)
                # log the images to tensorboard
                
                # first thing we want is to a vertical alignment of final_pred, input, valid_mask
                images_left = image_left.detach().cpu() * 0.5 + 0.5
                images_right = image_right.detach().cpu() * 0.5 + 0.5
                # we might have to reshape the disparities
                targets = disp_mask.detach().cpu()
                # convert this to float
                masks = valid_disp_mask.float().detach().cpu()
                preds = disp_predictions[-1].detach().cpu()
                # if we have 2 channels, we need to keep only the first channel
                if preds.shape[1] == 2:
                    preds = preds[:, :1, ...]
                if targets.shape[1] == 2:
                    targets = targets[:, :1, ...]
                # we then have to repeat the images on the 3 channels
                # in order to have gray scale images for the input and the target
                preds = preds.repeat(1, 3, 1, 1)
                targets = targets.repeat(1, 3, 1, 1)
                # masks are 2D so we have to unsqueeze and repeat
                masks = masks.unsqueeze(1).repeat(1, 3, 1, 1)
                
                # the grid is going to self-normalize in 0-1 range                                
                pred_grid = make_pair_grid(images_left, images_right, masks, targets, preds, orientation="horizontal")
                # we multiply by 255 to get the correct range for the tensorboard
                pred_grid = pred_grid.detach().cpu().numpy() * 255.0
                pred_grid = np.transpose(pred_grid, (1, 2, 0)).astype(np.uint8)
                
                writer.add_image("predictions", pred_grid, step, dataformats="HWC")
                
                # second thing we want to see is how relevant the iterative refinement is
                seq_len = len(disp_predictions) + 1
                disp_predictions = list(map(lambda x: x[:, :1, :, :], disp_predictions + [disp_mask]))                 
                sequence = make_disparity_sequence(disp_predictions)
                # swap axes to have the sequence in the correct order for each batch sample
                sequence = torch.swapaxes(sequence, 0, 1).contiguous().view(-1, 1, image_left.shape[2], image_left.shape[3])
                sequence = make_grid(sequence, nrow=seq_len)
                sequence = sequence.detach().cpu().numpy() * 255.0
                sequence = np.transpose(sequence, (1, 2, 0)).astype(np.uint8)
                writer.add_image("sequence", sequence, step, dataformats="HWC")

        if step % args.save_freq == 0:
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
                torch.save(checkpoint, Path(args.checkpoint_dir) / f"{args.experiment_name}_{step}.pth")
                torch.save(checkpoint, Path(args.checkpoint_dir) / f"{args.experiment_name}.pth")

        if step % args.valid_freq == 0:
            evaluate(model, args, writer, step)
            model.train()
            if args.freeze_batch_norm:
                if isinstance(model, nn.parallel.DistributedDataParallel):
                    utils.freeze_batch_norm(model.module)
                else:
                    utils.freeze_batch_norm(model)

                # prof.step()
    
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def main(args):
    args.total_iterations = sum(args.dataset_steps)

    # intialize DDP setting
    utils.setup_ddp(args)
    args.test_only = args.train_dataset is None
    
    # set random seed
    set_seed(args.seed)

    # set the appropiate devices
    if args.distributed and args.device == "cpu":
        raise ValueError("The device must be cuda if we want to run in distributed mode using torchrun")
    device = torch.device(args.device)

    # enable deterministic algorithms
    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    # select model architecture
    model = torchvision.prototype.models.depth.stereo.__dict__[args.model](weights=args.weights)

    # convert to DDP if need be
    if args.distributed:
        model = model.to(args.local_rank)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
        model_without_ddp = model.module
    else:
        model.to(device)
        model_without_ddp = model

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    # load checkpoints if needed
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])

    # EVAL ONLY configurations
    if args.test_only:
        # Set deterministic CUDNN algorithms, since they can affect epe a fair bit.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        evaluate(model, args)
        return

    # Sanity check for the parameter count
    print(f"Parameter Count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Compose the training dataset
    train_dataset = get_train_dataset(args.dataset_root, args)

    # initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # initialize the learning rate schedule
    scheduler = make_cre_stereo_schedule(args, optimizer)

    # load them from checkpoint if need
    if args.resume is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        args.start_step = checkpoint["step"] + 1
    else:
        args.start_step = 0

    torch.backends.cudnn.benchmark = True

    # enable training mode
    model.train()
    if args.freeze_batch_norm:
        utils.freeze_batch_norm(model_without_ddp)

    # put dataloader on top of the dataset
    # make sure to disable shuffling since the dataset is already shuffled
    # in order to guarantee quasi randomness whilst retaining a deterministic
    # dataset consumption order
    if args.distributed:
        # the train dataset is preshuffled in order to respect the iteration order
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=False, drop_last=True)
    else:
        # the train dataset is already shuffled so we can use a simple SequentialSampler
        sampler = torch.utils.data.SequentialSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.workers,
    )

    # intialize the logger
    if args.tensorboard_summaries:
        tensorboard_path = Path(args.checkpoint_dir) / "tensorboard"
        os.makedirs(tensorboard_path, exist_ok=True)
        
        tensorboard_run = tensorboard_path / f"{args.experiment_name}"
        writer = SummaryWriter(tensorboard_run)
    else:
        writer = None
    
    logger = utils.MetricLogger(log_dir=args.checkpoint_dir, log_name=f"{args.experiment_name}_train.log")

    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None
    # run the training loop
    # this will perform optimization, respectively logging and saving checkpoints
    # when need be
    run(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        logger=logger,
        writer=writer,
        scaler=scaler,
        args=args,        
    )


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Stereo Matching Training", add_help=add_help)
    parser.add_argument("--run-config", type=str, default="train_configs/default.yml", help="path to the config file")
    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    experiment_name = args.run_config.split(os.sep)[-1].replace(".yml", "")
    with open(args.run_config) as f:
        config = yaml.safe_load(f)
    config["experiment_name"] = experiment_name
    # print each config key and value
    print("Experiment Configurations:")
    for k, v in config.items():
        print(f"{k}: {v}")
    config = argparse.Namespace(**config)
    main(config)
