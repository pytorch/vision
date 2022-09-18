import argparse
from collections import defaultdict
from distutils.command.config import config
from functools import partial
import os
import pathlib
import time
from typing import List, Union
from parsing import make_dataset, make_eval_transform
from torch.nn import functional as F
import torch
import utils
import torchvision
import numpy as np
import torchvision.prototype.models.depth.stereo
import warnings
from vizualization import make_prediction_image_side_to_side

def get_args_parser(add_help=True):
    import argparse
    
    parser = argparse.ArgumentParser(description="PyTorch Stereo Matching Evaluation", add_help=add_help)
    parser.add_argument("--dataset", type=str, default="middlebury2014-train", help="dataset to use")
    parser.add_argument("--dataset-root", type=str, default="/fsx/users/teodorponcu/datasets", help="root of the dataset")
    
    parser.add_argument("--checkpoint", type=str, default="", help="path to weights")
    parser.add_argument("--img-folder", type=str, default="images")
    
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="number of workers")
    
    parser.add_argument("--resize-size", type=int, n_args='+', default=[384, 512], help="resize size")
    parser.add_argument("--n_iterations", nargs='+', type=int, default=[10], help="number of recurent iterations")
    parser.add_argument("--n_cascades", nargs='+', type=int, default=[1], help="number of cascades")
    parser.add_argument("--device", type=str, default="cuda", help="device to use")
    
    parser.add_argument("--save-predictions", action="store_true", help="save predictions")
    
    return parser

def cascade_inference(model, image_left, image_right, iterations, cascades):
    # check that image size is divisible by 16 * (2 ** (cascades - 1))
    for image in [image_left, image_right]:
        if image.shape[-2] % ((2 ** (cascades - 1))) != 0:
            raise ValueError(f"image height is not divisible by {16 * (2 ** (cascades - 1))}. Image shape: {image.shape[-2]}")
    
        if image.shape[-1] % ((2 ** (cascades - 1))) != 0:
            raise ValueError(f"image width is not divisible by {16 * (2 ** (cascades - 1))}. Image shape: {image.shape[-2]}")
    
    left_image_pyramid = [image_left]
    right_image_pyramid = [image_right]
    for idx in range(0, cascades - 1):
        ds_factor = int(2 ** (idx + 1))
        ds_shape = (image_left.shape[-2] // ds_factor, image_left.shape[-1] // ds_factor)
        left_image_pyramid += F.interpolate(image_left, size=ds_shape, mode="bilinear", align_corners=True).unsqueeze(0)
        right_image_pyramid += F.interpolate(image_right, size=ds_shape, mode="bilinear", align_corners=True).unsqueeze(0)
        
    flow_init = None
    for left_image, right_image in zip(reversed(left_image_pyramid), reversed(right_image_pyramid)):
        flow_pred = model(left_image, right_image, flow_init, iterations=iterations)
        # flow pred is a list
        flow_init = flow_pred[-1]
        
    return flow_init

@torch.inference_mode()
def _evaluate(
    model, args, val_dataset, *, padder_mode, print_freq=10, writter=None, step=None, iterations=10, cascades=1, batch_size=None, header=None, save_images=False, save_path="",
):
    """Helper function to compute various metrics (epe, etc.) for a model on a given dataset.
    We process as many samples as possible with ddp.
    """
    model.eval()
    header = header or "Test:"
    device = torch.device(args.device)
    metric_logger = utils.MetricLogger(delimiter="  ")
    
    effective_batch_size = batch_size * args.world_size
    can_do_distributed_test = effective_batch_size < len(val_dataset)
    
    if args.distributed and can_do_distributed_test:
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

    logger = utils.MetricLogger()
    for meter_name in args.metrics:
        logger.add_meter(meter_name, fmt="{global_avg:.4f}")
    if "f1" not in args.metrics:
        logger.add_meter("f1", fmt="{global_avg:.4f}")
        
    if can_do_distributed_test:
        should_forward = True
        module = model
    else:
        should_forward = args.rank == 0
        # doing forward on rank 0 will trigger a synchronization
        # which will not be fulfilled by non rank 0 processes
        module = model.module

    num_processed_samples = 0
    with torch.cuda.amp.autocast(enabled=args.mixed_precision, dtype=torch.float16):
        if should_forward:
            batch_idx = 0
            for blob in metric_logger.log_every(val_loader, print_freq, header):
                image_left, image_right, disp_gt, valid_disp_mask = (x.to(device) for x in blob)
                padder = utils.InputPadder(image_left.shape, mode=padder_mode)
                image_left, image_right = padder.pad(image_left, image_right)

                disp_pred = cascade_inference(module, image_left, image_right, iterations, cascades)
                disp_pred = disp_pred[:, :1, :, :]
                disp_pred = padder.unpad(disp_pred)

                if save_images:
                    make_prediction_image_side_to_side(disp_pred, disp_gt, valid_disp_mask, args.save_path, prefix=f"batch_{batch_idx}")
                    
                metrics, _ = utils.compute_metrics(disp_pred, disp_gt, valid_disp_mask, metrics=logger.meters.keys())
                num_processed_samples += image_left.shape[0]
                for name in metrics:
                    logger.meters[name].update(metrics[name], n=1)
                    
                batch_idx += 1
    
    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    
    print("Num_processed_samples: ", num_processed_samples)
    if num_processed_samples != len(val_dataset):
        warnings.warn(
            f"Number of processed samples {num_processed_samples} is different"
            f"from the dataset size {len(val_dataset)}. This may happen if"
            "the dataset is not divisible by the batch size. Try lowering the batch size for more accurate results."
        )

    if writter is not None and args.rank == 0:
        for meter_name, meter_value in logger.meters.items():
            scalar_name = f"{meter_name} {header}"
            writter.add_scalar(scalar_name, meter_value.avg, step)
    
    logger.synchronize_between_processes()
    print(header, logger)
    
    logger_metrics = {k: v.global_avg for k, v in logger.meters.items()}
    return logger_metrics


def evaluate(model, args, writter=None, step=None):
    val_datasets = args.test_datasets or []
    os.makedirs(args.img_folder, exist_ok=True)
    
    checkpoint_name = os.path.basename(args.checkpoint)
    image_checkpoint_folder = os.path.join(args.img_folder, checkpoint_name)

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
        preprocessing = make_eval_transform(args)

    metrics = {}
    for name in val_datasets:
        base_image_folder = os.path.join(image_checkpoint_folder, name)
        os.makedirs(base_image_folder, exist_ok=True)
        
        metrics[name] = {}
        val_dataset = make_dataset(name, args.dataset_root, transforms=preprocessing)
        for n_cascades in args.num_cascades:
            for n_iters in args.num_iters:
                
                config = f"{n_cascades}c_{n_iters}i"
                config_image_folder = os.path.join(base_image_folder, config)
                os.makedirs(config_image_folder, exist_ok=True)
                
                metrics[name][config] = _evaluate(
                    model,
                    args,
                    val_dataset,
                    iterations=args.recurrent_updates,
                    padder_mode="kitti",
                    header=f"{name} evaluation",
                    batch_size=args.batch_size,
                    writter=writter,
                    step=step,
                    iterations=n_iters,
                    cascades=n_cascades,
                )
    
    metric_log = []
    # print the final results
    for name in val_datasets:
        for config in metrics[name]:
            config_tokens = config.split("_")
            config_iters = config_tokens[1][:-1]
            config_cascades = config_tokens[0][:-1]
            
            evaluation_str = f"{name} evaluation@ size:{args.resize_size} n_cascades:{config_cascades} recurrent_updates:{config_iters}"
            metrics_str = f"Metrics: {metrics[name][config]}"
            metric_log.extend([evaluation_str, metrics_str])
            
            print(evaluation_str)
            print(metrics_str)        

    eval_log_name = f"{checkpoint_name.replace('.pth', '')}_eval.log"
    print("Saving eval log to: ", eval_log_name)
    with open(eval_log_name, "w") as f:
        f.write("\n".join(metric_log))
            

def load_checkpoint(args):
    checkpoint = torch.load(args.checkpoint, map_location=torch.device("cpu"))
    experiment_args = checkpoint["args"]
    model = torchvision.prototype.models.depth.stereo.__dict__[experiment_args.model](weights=None)
    model.load_state_dict(checkpoint["model"])
    return model

    
def main(args):
    model = load_checkpoint(args)
    evaluate(model, args, padder_mode="kitti")

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)

