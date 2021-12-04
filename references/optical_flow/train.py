import argparse
import warnings
from pathlib import Path

import torch
import utils
from presets import OpticalFlowPresetTrain, OpticalFlowPresetEval
from torchvision.datasets import KittiFlow, FlyingChairs, FlyingThings3D, Sintel, HD1K
from torchvision.models.optical_flow import raft_large, raft_small


def get_train_dataset(stage, dataset_root):
    if stage == "chairs":
        transforms = OpticalFlowPresetTrain(crop_size=(368, 496), min_scale=0.1, max_scale=1.0, do_flip=True)
        return FlyingChairs(root=dataset_root, split="train", transforms=transforms)
    elif stage == "things":
        transforms = OpticalFlowPresetTrain(crop_size=(400, 720), min_scale=-0.4, max_scale=0.8, do_flip=True)
        return FlyingThings3D(root=dataset_root, split="train", pass_name="both", transforms=transforms)
    elif stage == "sintel_SKH":  # S + K + H as from paper
        crop_size = (368, 768)
        transforms = OpticalFlowPresetTrain(crop_size=crop_size, min_scale=-0.2, max_scale=0.6, do_flip=True)

        things_clean = FlyingThings3D(root=dataset_root, split="train", pass_name="clean", transforms=transforms)
        sintel = Sintel(root=dataset_root, split="train", pass_name="both", transforms=transforms)

        kitti_transforms = OpticalFlowPresetTrain(crop_size=crop_size, min_scale=-0.3, max_scale=0.5, do_flip=True)
        kitti = KittiFlow(root=dataset_root, split="train", transforms=kitti_transforms)

        hd1k_transforms = OpticalFlowPresetTrain(crop_size=crop_size, min_scale=-0.5, max_scale=0.2, do_flip=True)
        hd1k = HD1K(root=dataset_root, split="train", transforms=hd1k_transforms)

        # As future improvement, we could probably be using a distributed sampler here
        # The distribution is S(.71), T(.135), K(.135), H(.02)
        return 100 * sintel + 200 * kitti + 5 * hd1k + things_clean
    elif stage == "kitti":
        transforms = OpticalFlowPresetTrain(
            # resize and crop params
            crop_size=(288, 960),
            min_scale=-0.2,
            max_scale=0.4,
            stretch_prob=0,
            # flip params
            do_flip=False,
            # jitter params
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.3 / 3.14,
            asymmetric_jitter_prob=0,
        )
        return KittiFlow(root=dataset_root, split="train", transforms=transforms)
    else:
        raise ValueError(f"Unknown stage {stage}")


@torch.no_grad()
def _validate(model, args, val_dataset, *, padder_mode, num_flow_updates=None, batch_size=None, header=None):
    """Helper function to compute various metrics (epe, etc.) for a model on a given dataset.

    We process as many samples as possible with ddp, and process the rest on a single worker.
    """
    model.eval()

    sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        sampler=sampler,
        batch_size=batch_size or args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    num_flow_updates = num_flow_updates or args.num_flow_updates

    def inner_loop(blob):
        if blob[0].dim() == 3:
            # input is not batched so we add an extra dim for consistency
            blob = [x[None, :, :, :] if x is not None else None for x in blob]

        image1, image2, flow_gt = blob[:3]
        valid_flow_mask = None if len(blob) == 3 else blob[-1]

        image1, image2 = image1.cuda(), image2.cuda()

        padder = utils.InputPadder(image1.shape, mode=padder_mode)
        image1, image2 = padder.pad(image1, image2)

        flow_predictions = model(image1, image2, num_flow_updates=num_flow_updates)
        flow_pred = flow_predictions[-1]

        flow_pred = padder.unpad(flow_pred).cpu()

        epe = ((flow_pred - flow_gt) ** 2).sum(dim=1).sqrt()

        logger.meters["epe"].update(epe.mean().item(), n=epe.numel())
        for distance in (1, 3, 5):
            logger.meters[f"{distance}px"].update((epe < distance).float().mean().item(), n=epe.numel())

        relative_epe = epe / (flow_gt ** 2).sum(dim=1).sqrt()
        if valid_flow_mask is not None:
            epe, relative_epe = epe[valid_flow_mask], relative_epe[valid_flow_mask]
        bad_predictions = ((epe > 3) & (relative_epe > 0.05)).float()

        # note the n=1 for per_image_epe: we compute an average over averages. We first average within each image and
        # then average over the images. This is in contrast with the other epe computation, where we
        # average only once over all the pixels of all images.
        logger.meters["per_image_epe"].update(epe.mean().item(), n=1)  # f1-epe in paper
        logger.meters["f1"].update(bad_predictions.mean().item() * 100, n=bad_predictions.numel())  # f1-all in paper

    logger = utils.MetricLogger()
    for meter_name in ("epe", "1px", "3px", "5px", "per_image_epe", "f1"):
        logger.add_meter(meter_name, fmt="{global_avg:.4f}")

    num_processed_samples = 0
    for blob in logger.log_every(val_loader, header=header, print_freq=None):
        inner_loop(blob)
        num_processed_samples += blob[0].shape[0]  # batch size

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


def validate(model, args):
    val_datasets = args.val_dataset or []
    for name in val_datasets:
        if name == "kitti":
            # Kitti has different image sizes so we need to individually pad them, we can't batch.
            # see comment in InputPadder
            if args.batch_size != 1 and args.rank == 0:
                warnings.warn(
                    f"Batch-size={args.batch_size} was passed. For technical reasons, evaluating on Kitti can only be done with a batch-size of 1."
                )

            val_dataset = KittiFlow(root=args.dataset_root, split="train", transforms=OpticalFlowPresetEval())
            _validate(
                model, args, val_dataset, num_flow_updates=24, padder_mode="kitti", header="Kitti val", batch_size=1
            )
        elif name == "sintel":
            for pass_name in ("clean", "final"):
                val_dataset = Sintel(
                    root=args.dataset_root, split="train", pass_name=pass_name, transforms=OpticalFlowPresetEval()
                )
                _validate(
                    model,
                    args,
                    val_dataset,
                    num_flow_updates=32,
                    padder_mode="sintel",
                    header=f"Sintel val {pass_name}",
                )
        else:
            warnings.warn(f"Can't validate on {val_dataset}, skipping.")


def main(args):
    utils.setup_ddp(args)

    model = raft_small() if args.small else raft_large()
    model = model.to(args.local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    if args.resume is not None:
        d = torch.load(args.resume, map_location="cpu")
        if args.map_orig_to_ours:
            d = utils.map_orig_to_ours(d)
        model.load_state_dict(d, strict=True)

    if args.train_dataset is None:
        # Set deterministic CUDNN algorithms, since they can affect epe a fair bit.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        validate(model, args)
        return

    print(f"Parameter Count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    torch.backends.cudnn.benchmark = True

    model.train()
    if args.freeze_batch_norm:
        utils.freeze_batch_norm(model.module)

    train_dataset = get_train_dataset(args.train_dataset, args.dataset_root)

    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, drop_last=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=args.adamw_eps)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=args.lr,
        total_steps=args.num_steps + 100,
        pct_start=0.05,
        cycle_momentum=False,
        anneal_strategy="linear",
    )

    logger = utils.MetricLogger()

    done = False
    current_epoch = current_step = 0
    while not done:
        sampler.set_epoch(current_epoch)  # needed, otherwise the data loading order would be the same for all epochs
        print(f"EPOCH {current_epoch}")

        for data_blob in logger.log_every(train_loader):

            optimizer.zero_grad()

            image1, image2, flow, valid_flow_mask = (x.cuda() for x in data_blob)
            flow_predictions = model(image1, image2, num_flow_updates=args.num_flow_updates)

            loss, metrics = utils.sequence_loss(flow_predictions, flow, valid_flow_mask, args.gamma)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            optimizer.step()
            scheduler.step()

            logger.update(**metrics)

            current_step += 1

            if current_step == args.num_steps:
                done = True
                break

        # Note: we don't sync the SmoothedValues across processes, so the printed metrics are just those of rank 0
        print(f"Epoch {current_epoch} done. ", logger)

        current_epoch += 1

        if args.rank == 0:
            torch.save(model.state_dict(), Path(args.output_dir) / f"{args.name}_{current_epoch}.pth")
            torch.save(model.state_dict(), Path(args.output_dir) / f"{args.name}.pth")

        if current_epoch % args.val_freq == 0 or done:
            validate(model, args)
            model.train()
            if args.freeze_batch_norm:
                utils.freeze_batch_norm(model.module)


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(add_help=add_help, description="Train or evaluate an optical-flow model.")
    parser.add_argument(
        "--name",
        default="raft",
        type=str,
        help="The name of the experiment - determines the name of the files where weights are saved.",
    )
    parser.add_argument(
        "--output-dir", default="checkpoints", type=str, help="Output dir where checkpoints will be stored."
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="A path to previously saved weights. Used to re-start training from, or evaluate a pre-saved model.",
    )

    parser.add_argument("--num-workers", type=int, default=12, help="Number of workers for the data loading part.")

    parser.add_argument(
        "--train-dataset",
        type=str,
        help="The dataset to use for training. If not passed, only validation is performed (and you probably want to pass --resume).",
    )
    parser.add_argument("--val-dataset", type=str, nargs="+", help="The dataset(s) to use for validation.")
    parser.add_argument("--val-freq", type=int, default=2, help="Validate every X epochs")
    # TODO: eventually, it might be preferable to support epochs instead of num_steps.
    # Keeping it this way for now to reproduce results more easily.
    parser.add_argument("--num-steps", type=int, default=100000, help="The total number of steps (updates) to train.")
    parser.add_argument("--batch-size", type=int, default=6)

    parser.add_argument("--lr", type=float, default=0.00002, help="Learning rate for AdamW optimizer")
    parser.add_argument("--weight-decay", type=float, default=0.00005, help="Weight decay for AdamW optimizer")
    parser.add_argument("--adamw-eps", type=float, default=1e-8, help="eps value for AdamW optimizer")

    parser.add_argument(
        "--freeze-batch-norm", action="store_true", help="Set BatchNorm modules of the model in eval mode."
    )

    parser.add_argument("--small", action="store_true", help="Use the 'small' RAFT architecture.")

    parser.add_argument(
        "--num_flow_updates",
        type=int,
        default=12,
        help="number of updates (or 'iters') in the update operator of the model.",
    )

    parser.add_argument("--gamma", type=float, default=0.8, help="exponential weighting for loss. Must be < 1.")

    parser.add_argument("--dist-url", default="env://", help="URL used to set up distributed training")

    # TODO: remove
    parser.add_argument("--map-orig-to-ours", action="store_true")

    # TODO: remove the default
    _DATASET_ROOT = "/data/home/nicolashug/cluster/work/downloads"
    parser.add_argument(
        "--dataset-root",
        default=_DATASET_ROOT,
        help="Root folder where the datasets are stored. Will be passed as the 'root' parameter of the datasets.",
    )

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(exist_ok=True)
    main(args)
