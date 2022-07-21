import datetime
import io
import itertools
import os
import random
import time
import warnings

import helpers
import presets
import torch
import torch.utils.data
import torchvision
import utils
from torch import nn
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService, adapter
from torchvision.transforms.functional import InterpolationMode


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema=None, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        if args.data_loading_only:
            continue
        start_time = time.time()
        if args.data_loader != "ffcv":
            image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))


def evaluate(model, criterion, data_loader, device, args, print_freq=100, log_suffix=""):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    metric_logger.add_meter("acc1", utils.SmoothedValue())
    metric_logger.add_meter("acc5", utils.SmoothedValue())

    num_processed_samples = 0
    cm = torch.no_grad() if args.data_loader.lower() == "ffcv" else torch.inference_mode()
    with cm:
        print("in cm")
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            print("evaluate batch")
            if args.data_loading_only:
                continue
            if args.data_loader != "ffcv":
                image = image.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size
    # gather the stats from all processes

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader, "dataset")
        and hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
    return metric_logger.acc1.global_avg


def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def load_data(traindir, valdir, args):
    # Data loading code
    print("Loading data")
    val_resize_size, val_crop_size, train_crop_size = args.val_resize_size, args.val_crop_size, args.train_crop_size
    interpolation = InterpolationMode(args.interpolation)

    print("Loading training data")
    st = time.time()
    auto_augment_policy = getattr(args, "auto_augment", None)
    random_erase_prob = getattr(args, "random_erase", 0.0)

    if args.no_transforms:
        preset = val_preset = helpers.no_transforms
    else:
        preset = presets.ClassificationPresetTrain(
            crop_size=train_crop_size,
            interpolation=interpolation,
            auto_augment_policy=auto_augment_policy,
            random_erase_prob=random_erase_prob,
        )
        val_preset = presets.ClassificationPresetEval(
            crop_size=val_crop_size, resize_size=val_resize_size, interpolation=interpolation
        )
    if args.ds_type == "dp":
        builder = helpers.make_pre_loaded_dp if args.preload_ds else helpers.make_dp
        dataset = builder(traindir, transforms=preset)
        dataset_test = builder(valdir, transforms=val_preset)

        train_sampler = test_sampler = None
        shuffle_train = True
    elif args.ds_type == "iterable":
        dataset = torchvision.datasets.ImageFolder(traindir, transform=preset)
        dataset = helpers.MapStyleToIterable(dataset, shuffle=True)

        dataset_test = torchvision.datasets.ImageFolder(valdir, transform=val_preset)
        dataset_test = helpers.MapStyleToIterable(dataset_test, shuffle=False)

        train_sampler = test_sampler = None
        shuffle_train = None  # but actually True
    elif args.ds_type == "mapstyle":
        builder = helpers.PreLoadedMapStyle if args.preload_ds else torchvision.datasets.ImageFolder
        dataset = builder(traindir, transform=preset)
        dataset_test = builder(valdir, transform=val_preset)

        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
        shuffle_train = None  # but actually True
    else:
        raise ValueError(f"Invalid value for args.ds_type ({args.ds_type})")
    print("Took", time.time() - st)

    data_loader_arg = args.data_loader.lower()
    if data_loader_arg not in ("v1", "v2", "ffcv"):
        raise ValueError(f"invalid data-loader param. Got {args.data_loader}")
    if data_loader_arg == "v1":
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle_train,
            sampler=train_sampler,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
        )
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=args.batch_size,
            sampler=test_sampler,
            num_workers=args.workers,
            pin_memory=True,
        )
    elif data_loader_arg == "v2":
        if args.ds_type != "dp":
            raise ValueError("DataLoader2 only works with datapipes.")

        # Note: we are batching and collating here *after the transforms*, which is consistent with DLV1.
        # But maybe it would be more efficient to do that before, so that the transforms can work on batches??

        dataset = dataset.batch(args.batch_size, drop_last=True).collate()
        data_loader = DataLoader2(
            dataset,
            datapipe_adapter_fn=adapter.Shuffle(),
            reading_service=MultiProcessingReadingService(num_workers=args.workers),
        )

        dataset_test = dataset_test.batch(args.batch_size, drop_last=True).collate()  # TODO: Do we need drop_last here?
        data_loader_test = DataLoader2(
            dataset_test,
            reading_service=MultiProcessingReadingService(num_workers=args.workers),
        )
    else:  # ffcv
        import numpy as np
        from ffcv.fields.basics import IntDecoder
        from ffcv.fields.decoders import RandomResizedCropRGBImageDecoder, CenterCropRGBImageDecoder
        from ffcv.loader import Loader, OrderOption
        from ffcv.transforms import (
            ToTensor,
            ToDevice,
            Squeeze,
            NormalizeImage,
            RandomHorizontalFlip,
            ToTorchImage,
        )

        train_sampler = test_sampler = None

        IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
        IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
        image_pipeline_train = [
            RandomResizedCropRGBImageDecoder(
                scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), output_size=(train_crop_size, train_crop_size),
            ),
            RandomHorizontalFlip(0.5),
            ToTensor(),
            ToDevice(torch.device(args.gpu), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32),  # Note: this is done on GPU
        ]
        image_pipeline_test = [
            CenterCropRGBImageDecoder(
                output_size=(val_crop_size, val_crop_size), ratio=val_crop_size / val_resize_size
            ),  # See https://github.com/libffcv/ffcv-imagenet/blob/f134cbfff7f590954edc5c24275444b7dd2f57f6/train_imagenet.py#L265
            ToTensor(),
            ToDevice(torch.device(args.gpu), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32),
        ]
        label_pipeline = [IntDecoder(), ToTensor(), Squeeze(), ToDevice(torch.device(args.gpu), non_blocking=True)]
        data_loader = Loader(
            "/data/home/nicolashug/cluster/work/downloads/imagenet_train_jpg.beton",
            # "/data/home/nicolashug/cluster/work/downloads/imagenet_val_jpg.beton",
            batch_size=args.batch_size,
            drop_last=True,
            num_workers=args.workers,
            order=OrderOption[args.order.upper()],
            os_cache=True,
            pipelines={
                "img": image_pipeline_train,
                "label": label_pipeline,
            },
            distributed=True,
            seed=0,
            batches_ahead=2,  # Same default as prefetch_factor from DataLoader
        )
        data_loader_test = Loader(
            "/data/home/nicolashug/cluster/work/downloads/imagenet_val_jpg.beton",
            batch_size=args.batch_size,
            num_workers=args.workers,
            order=OrderOption.SEQUENTIAL,
            os_cache=True,
            pipelines={
                "img": image_pipeline_test,
                "label": label_pipeline,
            },
            distributed=True,
            batches_ahead=2,  # Same default as prefetch_factor from DataLoader
        )

    return data_loader, data_loader_test, train_sampler


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    print(f"file-system = {args.fs}")
    if args.fs == "fsx":
        dataset_dir = "/datasets01"
    elif args.fs == "fsx_isolated":
        dataset_dir = "/fsx_isolated"
    elif args.fs == "ontap":
        dataset_dir = "/datasets01_ontap"
    elif args.fs == "ontap_isolated":
        dataset_dir = "/ontap_isolated"
    else:
        raise ValueError(f"bad args.fs, got {args.fs}")

    dataset_dir += "/imagenet_full_size/061417/"
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")
    data_loader, data_loader_test, train_sampler = load_data(train_dir, val_dir, args)

    num_classes = 1000  # I'm lazy

    print("Creating model")
    model = torchvision.models.__dict__[args.model](weights=args.weights, num_classes=num_classes)
    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))
    parameters = utils.set_weight_decay(
        model,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    model_ema = None
    if args.model_ema:
        # Decay adjustment that aims to keep the decay independent from other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and ommit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if model_ema:
            evaluate(model_ema, criterion, data_loader_test, device=device, args=args, log_suffix="EMA")
        else:
            evaluate(model, criterion, data_loader_test, device=device, args=args)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema, scaler)
        print("train_on_epoch done")
        lr_scheduler.step()
        print("scheduler step done")
        evaluate(model, criterion, data_loader_test, device=device, args=args)
        print("evaluate done")
        if model_ema:
            evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA", args=args)
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            if model_ema:
                checkpoint["model_ema"] = model_ema.state_dict()
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))
        print("checkpointing done")

        if epoch == 0:
            first_epoch_time = time.time() - start_time

    total_time = time.time() - start_time
    print(f"Training time: {datetime.timedelta(seconds=int(total_time))}")
    print(f"Training time (w/o 1st epoch): {datetime.timedelta(seconds=int(total_time - first_epoch_time))}")


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    parser.add_argument("--data-path", default="/datasets01/imagenet_full_size/061417/", type=str, help="dataset path")
    parser.add_argument("--fs", default="fsx", type=str)
    parser.add_argument("--model", default="resnet18", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=90, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--bias-weight-decay",
        default=None,
        type=float,
        help="weight decay for bias parameters of all layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--transformer-embedding-decay",
        default=None,
        type=float,
        help="weight decay for embedding parameters for vision transformer models (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing"
    )
    parser.add_argument("--mixup-alpha", default=0.0, type=float, help="mixup alpha (default: 0.0)")
    parser.add_argument("--cutmix-alpha", default=0.0, type=float, help="cutmix alpha (default: 0.0)")
    parser.add_argument("--lr-scheduler", default="steplr", type=str, help="the lr scheduler (default: steplr)")
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument(
        "--lr-warmup-method", default="constant", type=str, help="the warmup method (default: constant)"
    )
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
    parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument(
        "--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters"
    )
    parser.add_argument(
        "--model-ema-steps",
        type=int,
        default=32,
        help="the number of iterations that controls how often to update the EMA model (default: 32)",
    )
    parser.add_argument(
        "--model-ema-decay",
        type=float,
        default=0.99998,
        help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",
    )
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )
    parser.add_argument(
        "--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)"
    )
    parser.add_argument(
        "--val-resize-size", default=256, type=int, help="the resize size used for validation (default: 256)"
    )
    parser.add_argument(
        "--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)"
    )
    parser.add_argument(
        "--train-crop-size", default=224, type=int, help="the random crop size used for training (default: 224)"
    )
    parser.add_argument("--clip-grad-norm", default=None, type=float, help="the maximum gradient norm (default None)")
    parser.add_argument("--ra-sampler", action="store_true", help="whether to use Repeated Augmentation in training")
    parser.add_argument(
        "--ra-reps", default=3, type=int, help="number of repetitions for Repeated Augmentation (default: 3)"
    )
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")

    parser.add_argument(
        "--ds-type",
        default="mapstyle",
        type=str,
        help="'dp' or 'iterable' or 'mapstyle' (for regular indexable datasets)",
    )
    parser.add_argument(
        "--preload-ds",
        action="store_true",
        help="whether to use a fake dataset where all images are pre-loaded and tranformed",
    )
    parser.add_argument(
        "--data-loading-only",
        action="store_true",
        help="When on, we bypass the model's forward and backward passes. So mostly only the dataloading happens",
    )
    parser.add_argument(
        "--no-transforms",
        action="store_true",
        help="Whether to apply transforms to the images. No transforms means we "
        "load and decode PIL images as usual, but we don't transform them. Instead we discard them "
        "and the dataset will produce random tensors instead, so Acc resuts will be garbage.",
    )
    parser.add_argument(
        "--data-loader",
        default="V1",
        type=str,
        help="'V1' or 'V2' or 'FFCV'. V2 only works for datapipes",
    )
    parser.add_argument(
        "--order",
        default="RANDOM",
        type=str,
        help="'RANDOM' or 'QUASI_RANDOM' or 'SEQUENTIAL'. Only relevant for FFCV dataloader. QUASI_RANDOM doesn't work in distributed mode.",
    )

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
