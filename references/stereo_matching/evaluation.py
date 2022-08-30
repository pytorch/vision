import argparse
from collections import defaultdict
from distutils.command.config import config
from functools import partial
import os
import pathlib
import time
from typing import List, Union
from presets import MiddleBurryEvalPreset, StereoMatchingPresetEval
from utils import get_dataset_by_name
from torch.nn import functional as F
import torch
import utils
import torchvision
import numpy as np
import torchvision.prototype.models.depth.stereo

def get_args_parser(add_help=True):
    import argparse
    
    parser = argparse.ArgumentParser(description="PyTorch Stereo Matching Evaluation", add_help=add_help)
    parser.add_argument("--dataset", type=str, default="middlebury2014-train", help="dataset to use")
    parser.add_argument("--dataset-root", type=str, default="/fsx/users/teodorponcu/datasets", help="root of the dataset")
    
    parser.add_argument("--checkpoint", type=str, default="", help="path to weights")
    
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="number of workers")
    
    parser.add_argument("--n_iterations", nargs='+', type=int, default=[10], help="number of recurent iterations")
    parser.add_argument("--n_cascades", nargs='+', type=int, default=[1], help="number of cascades")
    parser.add_argument("--device", type=str, default="cuda", help="device to use")
    
    parser.add_argument("--make-report", action="store_true", help="make a report")
    parser.add_argument("--report-dir", type=str, default="/fsx/users/teodorponcu/reports/crestereo", help="path to report")
    
    return parser

def cascade_inference(model, image_left, image_right, iterations, cascades):
    # check that image size is divisible by 16 * (2 ** (cascades - 1))
    for image in [image_left, image_right]:
        if image.shape[-2] % (16 * (2 ** (cascades - 1))) != 0:
            raise ValueError(f"image height is not divisible by {16 * (2 ** (cascades - 1))}. Image shape: {image.shape[-2]}")
    
        if image.shape[-1] % (16 * (2 ** (cascades - 1))) != 0:
            raise ValueError(f"image width is not divisible by {16 * (2 ** (cascades - 1))}. Image shape: {image.shape[-2]}")
    
    left_image_pyramid = [image_left]
    right_image_pyramid = [image_right]
    for idx in range(0, cascades - 1):
        ds_factor = int(2 ** (idx + 1))
        ds_shape = (image_left.shape[-2] // ds_factor, image_left.shape[-1] // ds_factor)
        left_image_pyramid += F.interpolate(image_left, size=ds_shape, mode="bilinear", align_corners=True).unsqueeze(0)
        right_image_pyramid += F.interpolate(image_right, size=ds_shape, mode="bilinear", align_corners=True).unsqueeze(0)
        
    flow_init = [None]
    for left_image, right_image in zip(reversed(left_image_pyramid), reversed(right_image_pyramid)):
        print(left_image.shape, right_image.shape)
        flow_pred = model(left_image, right_image, flow_init[-1], iterations=iterations)
        # flow pred is a list
        flow_init = flow_init + [flow_pred[-1]]
        
    return flow_init[-1]

@torch.inference_mode()
def evaluate(model, args, dataset, *, padder_mode, header = None):
    batch_size = 1
    device = torch.device(args.device)
    
    model.eval()
    model.to(device)
    
    sampler = torch.utils.data.SequentialSampler(dataset)
    loader  = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=args.num_workers
    )
    
    iterations = args.n_iterations
    cascades   = args.n_cascades
    
    def inner_loop(blob):
        if blob[0].dim() == 3:
            # input is not batched so we add an extra dim for consistency
            blob = [x.unsqueeze(0) if x is not None else None for x in blob]

        image_left, image_right, disp_gt = blob[:3]
        valid_disp_mask = None if len(blob) == 3 else blob[-1]

        image_left, image_right = image_left.to(device), image_right.to(device)

        padder = utils.InputPadder(image_left.shape, mode=padder_mode, multiplier=16 * (2 ** (n_cascades - 1)))
        image_left, image_right = padder.pad(image_left, image_right)

        iter_start_time = time.time()
        disp_pred = cascade_inference(model, image_left, image_right, n_iter, n_cascades)
        disp_pred = disp_pred[:, :1, :, :]
        iter_run_time = time.time() - iter_start_time
        
        
        # prevent memory blowup
        image_left, image_right = image_left.cpu(), image_right.cpu()
        disp_pred = disp_pred.cpu()
        
        # unpad
        image_left, image_right = padder.unpad(image_left), padder.unpad(image_right)
        disp_pred = padder.unpad(disp_pred)

        if disp_gt is not None:
            metrics, num_pixels_tot = utils.compute_metrics(disp_pred, disp_gt, valid_disp_mask)
            # We compute per-pixel epe (epe) and per-image epe (called f1-epe in RAFT paper).
            # per-pixel epe: average epe of all pixels of all images
            # per-image epe: average epe on each image independently, then average over images
            for name in ("epe", "1px", "3px", "5px", "f1"):  # f1 is called f1-all in paper
                logger.meters[name].update(metrics[name], n=num_pixels_tot)
            logger.meters["per_image_epe"].update(metrics["epe"], n=batch_size)
            
        return disp_pred, disp_gt, valid_disp_mask, iter_run_time
     
    experiment_checkpoint = os.path.basename(args.checkpoint).split(".")[0]      
    logger = utils.MetricLogger()
    
    for meter_name in ("epe", "1px", "3px", "5px", "per_image_epe", "f1"):
        logger.add_meter(meter_name, fmt="{global_avg:.4f}")

    num_processed_samples = 0
    preds = defaultdict(list)
    # masks prevents inf / nan values for datasets with very large disparity values
    masks = defaultdict(list)
    targets = defaultdict(list)
    run_times = defaultdict(list)
    
    default_partial = partial(defaultdict, list)
    # if we don't do this we get TypeError: first argument must be callable or None
    eval_metrics = defaultdict(default_partial)
    
    for n_iter in iterations:
        for n_cascades in cascades:
            config_key = (n_iter, n_cascades)
            for blob in logger.log_every(loader, header=header, print_freq=None):
                model_preds, target_preds, valid_disp_mask, run_time = inner_loop(blob)
                preds[config_key].append(model_preds.cpu())
                masks[config_key].append(valid_disp_mask.cpu())
                targets[config_key].append(target_preds.cpu() if target_preds is not None else None)
                run_times[config_key].append(run_time)
                
                num_processed_samples += blob[0].shape[0]  # batch size
            print(logger)
                
    if args.make_report:
        report_dir = pathlib.Path(args.report_dir) / experiment_checkpoint
        os.makedirs(report_dir, exist_ok=True)
        experiment_dir = report_dir / args.dataset
        os.makedirs(experiment_dir, exist_ok=True)
        
        import matplotlib.pyplot as plt
        # save predictions as individual images
        for config_key in preds:
            config_report_dir = experiment_dir / str(config_key)
            os.makedirs(config_report_dir, exist_ok=True)
            comparison_img_dir = config_report_dir / "side_by_side"
            os.makedirs(comparison_img_dir, exist_ok=True)
            prediction_save_dir = config_report_dir / "predictions"
            os.makedirs(prediction_save_dir, exist_ok=True)
            
            for idx, (model_pred, target_pred, mask) in enumerate(zip(preds[config_key], targets[config_key], masks[config_key])):
                # save predictions as an individual numpy array
                np.save(
                    os.path.join(
                        prediction_save_dir,
                        f"{config_key}_{idx}.npy"
                    ),
                    model_pred.squeeze(0).cpu().numpy()
                )
                
                if target_pred is None:
                    continue

                model_pred_v = model_pred.clone().squeeze(0).cpu().permute(1, 2, 0).numpy()
                target_pred_v = target_pred.clone().squeeze(0).cpu().permute(1, 2, 0).numpy()
                mask = mask.clone().squeeze(0).cpu().numpy()
                # add masks to prevent nan / inf in images for malformed GT
                model_pred_v[~mask] = 0
                target_pred_v[~mask] = 0
                print(target_pred_v.max(), target_pred_v.min())
                target_pred_v[target_pred_v > 512] = 0
                print(model_pred_v.min(), model_pred_v.max(), target_pred_v.min(), target_pred_v.max())
                # normalize predictions to [0, 1]
                model_pred_disp = (model_pred_v - model_pred_v.min()) / (model_pred_v.max() - model_pred_v.min())
                target_pred_disp = (target_pred_v - target_pred_v.min()) / (target_pred_v.max() - target_pred_v.min())
                
                fig, ax = plt.subplots(1, 2, figsize=(16, 8))
                ax[0].imshow(model_pred_disp, cmap="magma")
                ax[0].set_title("Model prediction")
                ax[1].imshow(target_pred_disp, cmap="magma")
                ax[1].set_title("Ground truth")
                fig.suptitle(f"{config_key} - {idx}")
                
                fig.savefig(os.path.join(config_report_dir, f"{config_key}_{idx}.png"))
                plt.close(fig)
                
                # compute MAE and MSE
                eval_metrics["MAE"][config_key].append(np.abs(model_pred_v - target_pred_v).mean())
                eval_metrics["MSE"][config_key].append(np.square(model_pred_v - target_pred_v).mean())
                eval_metrics["run_time"][config_key].append(run_times[config_key][idx])

        # make bar plots of metrics, grouped metric type
        n_metrics = len(eval_metrics)
        fig, ax = plt.subplots(1, n_metrics, figsize=(n_metrics * 5, 8))
        
        for metric_type in eval_metrics:
            keys = list(sorted(eval_metrics[metric_type].keys()))
            # make bar plots for each metric type, grouped by config
            for idx, metric in enumerate(eval_metrics.keys()):
                # run_time[0] is a special case, because the code path is not warmed up
                if metric == "run_time": start_element = 0
                else: start_element = 1
                values = [eval_metrics[metric][key][start_element:] for key in keys]
                mean_values = list(
                    np.mean(collected_values[start_element:])
                    for collected_values in values
                )
                
                ax[idx].bar(list(str(k) for k in keys), mean_values)
                ax[idx].set_title(metric)
                ax[idx].grid()
                ax[idx].legend()
                # make tight layout
        
        plt.tight_layout()
        fig.savefig(os.path.join(experiment_dir, f"{metric_type}.png"))
            

def load_checkpoint(args):
    checkpoint = torch.load(args.checkpoint)
    experiment_args = checkpoint["args"]
    model = torchvision.prototype.models.depth.stereo.__dict__[experiment_args.model](weights=None)
    model.load_state_dict(checkpoint["model"])
    return model
    
def get_eval_preset(dataset_name: str):
    if "middlebury" in dataset_name:
        return MiddleBurryEvalPreset
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    
def main(args):
    model = load_checkpoint(args)
    
    transforms = get_eval_preset(args.dataset)
    dataset = get_dataset_by_name(args.dataset, args.dataset_root, transforms)
    evaluate(model, args, dataset, padder_mode="kitti")

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)

