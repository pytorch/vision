import argparse
import os
import uuid
import warnings
from math import ceil
from pathlib import Path
from typing import Callable, List, Union
from torch import nn
import numpy as np
import torch
import torchvision.models.optical_flow
import torchvision.prototype.models.depth.stereo
import utils
import yaml
from presets import StereoMatchingPresetEval, StereoMatchingPresetCRETrain
from torch.optim.lr_scheduler import _LRScheduler
from torchvision.datasets import (
    InStereo2k,
    CREStereo,
    SintelStereo,
    SceneFlowStereo,
    FallingThingsStereo,
    Middlebury2014Stereo,
    ETH3DStereo,
)


def construct_experiment_name(args: dict):
    # return a random uuid if no experiment name is given
    return str(uuid.uuid4())


def extract_model_outputs(
    model_ouputs: Union[torch.Tensor, List[torch.Tensor]], args: argparse.Namespace
) -> torch.Tensor:
    """Helper function to extract the model outputs from a given architecture"""
    if "raft" in args.model:
        return model_ouputs
    elif "crestereo" in args.model:
        # CRE-stereo like models, return the entire flow-map
        # we are interested only on the X-axis flow for stereo matching
        if isinstance(model_ouputs, list):
            outs = list(map(lambda x: x[:, :1, :, :], model_ouputs))
            return outs
        else:
            return model_ouputs[:, :1, :, :]
    else:
        raise ValueError(f"Unknown model {args.model}")


def make_cre_stereo_schedule(args: argparse.Namespace) -> np.ndarray:
    """Helper function to return a learning rate scheduler for CRE-stereo"""
    warm_up_steps = args.warm_up_steps
    flat_lr_steps = args.flat_lr_steps - warm_up_steps
    decay_lr_steps = args.total_iterations - flat_lr_steps

    max_lr = args.lr
    # make linear warm-up
    warm_up_lr = np.linspace(0.05 * max_lr, max_lr, warm_up_steps)
    # make flat learning rate schedule
    flat_lr = np.linspace(max_lr, max_lr, flat_lr_steps)
    # make linear decay
    decay_lr = np.linspace(max_lr, 0.05 * max_lr, decay_lr_steps)
    schedule = np.concatenate([warm_up_lr, flat_lr, decay_lr])
    return schedule


class ArrayScheduler(_LRScheduler):
    """
    Simple _LRScheduler wrapper that consumes a numpy array as a schedule.
    """

    def __init__(self, optimizer, schedule: np.ndarray, last_epoch: int = -1, verbose=False) -> None:
        self.schedule = schedule.tolist()
        self._step_count = 0

        super().__init__(optimizer, last_epoch)

    def step(self, epoch=None):
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = self.schedule[self._step_count]
            self._step_count += 1

    def get_lr(self):
        return self.schedule[self._step_count]


def get_dataset_by_name(name: str, root: str, transforms: Callable):
    """Helper function to return a speciffic dataset configuration given it's name"""
    if name == "crestereo":
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
    elif name == "eth3d-train":
        return ETH3DStereo(root=root, transforms=transforms, split="train")
    elif name == "instereo-2k":
        return InStereo2k(root=root, transforms=transforms, split="train")
    elif name == "middlebury2014-train":
        return Middlebury2014Stereo(root=root, transforms=transforms, split="train", calibration="perfect")
    else:
        raise ValueError(f"Unknown dataset {name}")


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
def _evaluate(model, args, val_dataset, *, padder_mode, iterations=None, batch_size=None, header=None):
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
        disp_predictions = extract_model_outputs(disp_predictions, args)
        disp_pred = disp_predictions[-1]
        disp_pred = padder.unpad(disp_pred).cpu()

        metrics, num_pixels_tot = utils.compute_metrics(disp_pred, disp_gt, valid_disp_mask)

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
        preprocessing = StereoMatchingPresetEval()

    for name in val_datasets:
        if args.batch_size != 1 and (not args.distributed or args.rank == 0):
            warnings.warn(
                f"Batch-size={args.batch_size} was passed. For technical reasons, evaluating on Middlebury can only be done with a batch-size of 1."
            )

        val_dataset = get_dataset_by_name(root=args.dataset_root, name=name, transforms=preprocessing)
        _evaluate(
            model,
            args,
            val_dataset,
            iterations=args.recurrent_updates * 2,
            padder_mode="kitti",
            header="Kitti val",
            batch_size=1,
        )
        


def run(model, optimizer, scheduler, train_loader, logger, args):
    device = torch.device(args.device)
    # wrap the loader in a logger
    loader = iter(logger.log_every(train_loader))

    # consume until we reach the end of the iterations
    for step in range(args.start_step, args.total_iterations):
        data_blob = next(loader)
        optimizer.zero_grad()

        # unpack the data blob
        image_left, image_right, disp_mask, valid_disp_mask = (x.to(device) for x in data_blob)
        # some models might be doing recurrent updates or refinements
        # therefore we make a simple check for compatibility purposes
        # if you are reusing or modiffying this piece of code, make sure to
        # adjust it to your architecture naming scheme
        if "crestereo" in args.model:
            # TODO: this needs to be abstracted somehow
            disp_predictions = model(image_left, image_right, flow_init=None, iterations=args.recurrent_updates)
        elif "raft" in args.model:
            disp_predictions = model(image_left, image_right, args.recurrent_updates)
        else:
            raise ValueError(f"Unknown model {args.model}")
        # different models have different outputs, make sure we get the right ones for this task
        disp_predictions = extract_model_outputs(disp_predictions, args)
        # sequence loss on top of the model outputs
        loss = utils.sequence_loss(disp_predictions, disp_mask, valid_disp_mask, args.gamma)
        metrics, _ = utils.compute_metrics(disp_predictions[-1], disp_mask, valid_disp_mask)

        metrics.pop("f1")
        logger.update(loss=loss, **metrics)

        loss.backward()

        if args.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)

        optimizer.step()
        scheduler.step()

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
            evaluate(model, args)
            model.train()
            if args.freeze_batch_norm:
                if isinstance(model, nn.parallel.DistributedDataParallel):
                    utils.freeze_batch_norm(model.module)
                else:
                    utils.freeze_batch_norm(model)


def main(args):
    args.total_iterations = sum(args.dataset_steps)

    # intialize DDP setting
    utils.setup_ddp(args)
    args.test_only = args.train_dataset is None

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
    scheduler = ArrayScheduler(optimizer, schedule=make_cre_stereo_schedule(args))

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
    logger = utils.MetricLogger()

    # run the training loop
    # this will perform optimization, respectively logging and saving checkpoints
    # when need be
    run(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        logger=logger,
        args=args,
    )


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Stereo Matching Training", add_help=add_help)
    parser.add_argument("--config", type=str, default="train_configs/default.yml", help="path to the config file")
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    config_name = construct_experiment_name(config)
    config = argparse.Namespace(**config)
    config.experiment_name = args.config.split(os.sep)[-1].replace(".yml", "")
    print(config)
    main(config)
