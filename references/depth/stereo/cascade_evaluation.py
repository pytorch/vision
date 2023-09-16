import os
import warnings

import torch
import torchvision
import torchvision.prototype.models.depth.stereo
import utils
from torch.nn import functional as F
from train import make_eval_loader

from utils.metrics import AVAILABLE_METRICS
from visualization import make_prediction_image_side_to_side


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Stereo Matching Evaluation", add_help=add_help)
    parser.add_argument("--dataset", type=str, default="middlebury2014-train", help="dataset to use")
    parser.add_argument("--dataset-root", type=str, default="", help="root of the dataset")

    parser.add_argument("--checkpoint", type=str, default="", help="path to weights")
    parser.add_argument("--weights", type=str, default=None, help="torchvision API weight")
    parser.add_argument(
        "--model",
        type=str,
        default="crestereo_base",
        help="which model to use if not speciffying a training checkpoint",
    )
    parser.add_argument("--img-folder", type=str, default="images")

    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--workers", type=int, default=0, help="number of workers")

    parser.add_argument("--eval-size", type=int, nargs="+", default=[384, 512], help="resize size")
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

    parser.add_argument("--n_iterations", nargs="+", type=int, default=[10], help="number of recurent iterations")
    parser.add_argument("--n_cascades", nargs="+", type=int, default=[1], help="number of cascades")
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["mae", "rmse", "1px", "3px", "5px", "relepe"],
        help="metrics to log",
        choices=AVAILABLE_METRICS,
    )
    parser.add_argument("--mixed-precision", action="store_true", help="use mixed precision training")

    parser.add_argument("--world-size", type=int, default=1, help="number of distributed processes")
    parser.add_argument("--dist-url", type=str, default="env://", help="url used to set up distributed training")
    parser.add_argument("--device", type=str, default="cuda", help="device to use for training")

    parser.add_argument("--save-images", action="store_true", help="save images of the predictions")
    parser.add_argument("--padder-type", type=str, default="kitti", help="padder type", choices=["kitti", "sintel"])

    return parser


def cascade_inference(model, image_left, image_right, iterations, cascades):
    # check that image size is divisible by 16 * (2 ** (cascades - 1))
    for image in [image_left, image_right]:
        if image.shape[-2] % ((2 ** (cascades - 1))) != 0:
            raise ValueError(
                f"image height is not divisible by {16 * (2 ** (cascades - 1))}. Image shape: {image.shape[-2]}"
            )

        if image.shape[-1] % ((2 ** (cascades - 1))) != 0:
            raise ValueError(
                f"image width is not divisible by {16 * (2 ** (cascades - 1))}. Image shape: {image.shape[-2]}"
            )

    left_image_pyramid = [image_left]
    right_image_pyramid = [image_right]
    for idx in range(0, cascades - 1):
        ds_factor = int(2 ** (idx + 1))
        ds_shape = (image_left.shape[-2] // ds_factor, image_left.shape[-1] // ds_factor)
        left_image_pyramid += F.interpolate(image_left, size=ds_shape, mode="bilinear", align_corners=True).unsqueeze(0)
        right_image_pyramid += F.interpolate(image_right, size=ds_shape, mode="bilinear", align_corners=True).unsqueeze(
            0
        )

    flow_init = None
    for left_image, right_image in zip(reversed(left_image_pyramid), reversed(right_image_pyramid)):
        flow_pred = model(left_image, right_image, flow_init, num_iters=iterations)
        # flow pred is a list
        flow_init = flow_pred[-1]

    return flow_init


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
    iterations=10,
    cascades=1,
    batch_size=None,
    header=None,
    save_images=False,
    save_path="",
):
    """Helper function to compute various metrics (epe, etc.) for a model on a given dataset.
    We process as many samples as possible with ddp.
    """
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
        batch_idx = 0
        for blob in metric_logger.log_every(val_loader, print_freq, header):
            image_left, image_right, disp_gt, valid_disp_mask = (x.to(device) for x in blob)
            padder = utils.InputPadder(image_left.shape, mode=padder_mode)
            image_left, image_right = padder.pad(image_left, image_right)

            disp_pred = cascade_inference(model, image_left, image_right, iterations, cascades)
            disp_pred = disp_pred[:, :1, :, :]
            disp_pred = padder.unpad(disp_pred)

            if save_images:
                if args.distributed:
                    rank_prefix = args.rank
                else:
                    rank_prefix = 0
                make_prediction_image_side_to_side(
                    disp_pred, disp_gt, valid_disp_mask, save_path, prefix=f"batch_{rank_prefix}_{batch_idx}"
                )

            metrics, _ = utils.compute_metrics(disp_pred, disp_gt, valid_disp_mask, metrics=logger.meters.keys())
            num_processed_samples += image_left.shape[0]
            for name in metrics:
                logger.meters[name].update(metrics[name], n=1)

            batch_idx += 1

    num_processed_samples = utils.reduce_across_processes(num_processed_samples) / args.world_size

    print("Num_processed_samples: ", num_processed_samples)
    if (
        hasattr(val_loader.dataset, "__len__")
        and len(val_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        warnings.warn(
            f"Number of processed samples {num_processed_samples} is different"
            f"from the dataset size {len(val_loader.dataset)}. This may happen if"
            "the dataset is not divisible by the batch size. Try lowering the batch size for more accurate results."
        )

    if writer is not None and args.rank == 0:
        for meter_name, meter_value in logger.meters.items():
            scalar_name = f"{meter_name} {header}"
            writer.add_scalar(scalar_name, meter_value.avg, step)

    logger.synchronize_between_processes()
    print(header, logger)

    logger_metrics = {k: v.global_avg for k, v in logger.meters.items()}
    return logger_metrics


def evaluate(model, loader, args, writer=None, step=None):
    os.makedirs(args.img_folder, exist_ok=True)
    checkpoint_name = os.path.basename(args.checkpoint) or args.weights
    image_checkpoint_folder = os.path.join(args.img_folder, checkpoint_name)

    metrics = {}
    base_image_folder = os.path.join(image_checkpoint_folder, args.dataset)
    os.makedirs(base_image_folder, exist_ok=True)

    for n_cascades in args.n_cascades:
        for n_iters in args.n_iterations:

            config = f"{n_cascades}c_{n_iters}i"
            config_image_folder = os.path.join(base_image_folder, config)
            os.makedirs(config_image_folder, exist_ok=True)

            metrics[config] = _evaluate(
                model,
                args,
                loader,
                padder_mode=args.padder_type,
                header=f"{args.dataset} evaluation@ size:{args.eval_size} n_cascades:{n_cascades} n_iters:{n_iters}",
                batch_size=args.batch_size,
                writer=writer,
                step=step,
                iterations=n_iters,
                cascades=n_cascades,
                save_path=config_image_folder,
                save_images=args.save_images,
            )

    metric_log = []
    metric_log_dict = {}
    # print the final results
    for config in metrics:
        config_tokens = config.split("_")
        config_iters = config_tokens[1][:-1]
        config_cascades = config_tokens[0][:-1]

        metric_log_dict[config_cascades] = metric_log_dict.get(config_cascades, {})
        metric_log_dict[config_cascades][config_iters] = metrics[config]

        evaluation_str = f"{args.dataset} evaluation@ size:{args.eval_size} n_cascades:{config_cascades} recurrent_updates:{config_iters}"
        metrics_str = f"Metrics: {metrics[config]}"
        metric_log.extend([evaluation_str, metrics_str])

        print(evaluation_str)
        print(metrics_str)

    eval_log_name = f"{checkpoint_name.replace('.pth', '')}_eval.log"
    print("Saving eval log to: ", eval_log_name)
    with open(eval_log_name, "w") as f:
        f.write(f"Dataset: {args.dataset} @size: {args.eval_size}:\n")
        # write the dict line by line for each key, and each value in the keys
        for config_cascades in metric_log_dict:
            f.write("{\n")
            f.write(f"\t{config_cascades}: {{\n")
            for config_iters in metric_log_dict[config_cascades]:
                # convert every metric to 4 decimal places
                metrics = metric_log_dict[config_cascades][config_iters]
                metrics = {k: float(f"{v:.3f}") for k, v in metrics.items()}
                f.write(f"\t\t{config_iters}: {metrics}\n")
            f.write("\t},\n")
            f.write("}\n")


def load_checkpoint(args):
    utils.setup_ddp(args)

    if not args.weights:
        checkpoint = torch.load(args.checkpoint, map_location=torch.device("cpu"))
        if "model" in checkpoint:
            experiment_args = checkpoint["args"]
            model = torchvision.prototype.models.depth.stereo.__dict__[experiment_args.model](weights=None)
            model.load_state_dict(checkpoint["model"])
        else:
            model = torchvision.prototype.models.depth.stereo.__dict__[args.model](weights=None)
            model.load_state_dict(checkpoint)

        # set the appropriate devices
        if args.distributed and args.device == "cpu":
            raise ValueError("The device must be cuda if we want to run in distributed mode using torchrun")
        device = torch.device(args.device)
    else:
        model = torchvision.prototype.models.depth.stereo.__dict__[args.model](weights=args.weights)

    # convert to DDP if need be
    if args.distributed:
        model = model.to(args.device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    else:
        model.to(device)

    return model


def main(args):
    model = load_checkpoint(args)
    loader = make_eval_loader(args.dataset, args)
    evaluate(model, loader, args)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
