import os
from typing import List

import numpy as np
import torch
from torch import Tensor
from torchvision.utils import make_grid


@torch.no_grad()
def make_disparity_image(disparity: Tensor):
    # normalize image to [0, 1]
    disparity = disparity.detach().cpu()
    disparity = (disparity - disparity.min()) / (disparity.max() - disparity.min())
    return disparity


@torch.no_grad()
def make_disparity_image_pairs(disparity: Tensor, image: Tensor):
    disparity = make_disparity_image(disparity)
    # image is in [-1, 1], bring it to [0, 1]
    image = image.detach().cpu()
    image = image * 0.5 + 0.5
    return disparity, image


@torch.no_grad()
def make_disparity_sequence(disparities: List[Tensor]):
    # convert each disparity to [0, 1]
    for idx, disparity_batch in enumerate(disparities):
        disparities[idx] = torch.stack(list(map(make_disparity_image, disparity_batch)))
    # make the list into a batch
    disparity_sequences = torch.stack(disparities)
    return disparity_sequences


@torch.no_grad()
def make_pair_grid(*inputs, orientation="horizontal"):
    # make a grid of images with the outputs and references side by side
    if orientation == "horizontal":
        # interleave the outputs and references
        canvas = torch.zeros_like(inputs[0])
        canvas = torch.cat([canvas] * len(inputs), dim=0)
        size = len(inputs)
        for idx, inp in enumerate(inputs):
            canvas[idx::size, ...] = inp
        grid = make_grid(canvas, nrow=len(inputs), padding=16, normalize=True, scale_each=True)
    elif orientation == "vertical":
        # interleave the outputs and references
        canvas = torch.cat(inputs, dim=0)
        size = len(inputs)
        for idx, inp in enumerate(inputs):
            canvas[idx::size, ...] = inp
        grid = make_grid(canvas, nrow=len(inputs[0]), padding=16, normalize=True, scale_each=True)
    else:
        raise ValueError("Unknown orientation: {}".format(orientation))
    return grid


@torch.no_grad()
def make_training_sample_grid(
    left_images: Tensor,
    right_images: Tensor,
    disparities: Tensor,
    masks: Tensor,
    predictions: List[Tensor],
) -> np.ndarray:
    # detach images and renormalize to [0, 1]
    images_left = left_images.detach().cpu() * 0.5 + 0.5
    images_right = right_images.detach().cpu() * 0.5 + 0.5
    # detach the disparties and predictions
    disparities = disparities.detach().cpu()
    predictions = predictions[-1].detach().cpu()
    # keep only the first channel of pixels, and repeat it 3 times
    disparities = disparities[:, :1, ...].repeat(1, 3, 1, 1)
    predictions = predictions[:, :1, ...].repeat(1, 3, 1, 1)
    # unsqueeze and repeat the masks
    masks = masks.detach().cpu().unsqueeze(1).repeat(1, 3, 1, 1)
    # make a grid that will self normalize across the batch
    pred_grid = make_pair_grid(images_left, images_right, masks, disparities, predictions, orientation="horizontal")
    pred_grid = pred_grid.permute(1, 2, 0).numpy()
    pred_grid = (pred_grid * 255).astype(np.uint8)
    return pred_grid


@torch.no_grad()
def make_disparity_sequence_grid(predictions: List[Tensor], disparities: Tensor) -> np.ndarray:
    # right most we will be adding the ground truth
    seq_len = len(predictions) + 1
    predictions = list(map(lambda x: x[:, :1, :, :].detach().cpu(), predictions + [disparities]))
    sequence = make_disparity_sequence(predictions)
    # swap axes to have the in the correct order for each batch sample
    sequence = torch.swapaxes(sequence, 0, 1).contiguous().reshape(-1, 1, disparities.shape[-2], disparities.shape[-1])
    sequence = make_grid(sequence, nrow=seq_len, padding=16, normalize=True, scale_each=True)
    sequence = sequence.permute(1, 2, 0).numpy()
    sequence = (sequence * 255).astype(np.uint8)
    return sequence


@torch.no_grad()
def make_prediction_image_side_to_side(
    predictions: Tensor, disparities: Tensor, valid_mask: Tensor, save_path: str, prefix: str
) -> None:
    import matplotlib.pyplot as plt

    # normalize the predictions and disparities in [0, 1]
    predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min())
    disparities = (disparities - disparities.min()) / (disparities.max() - disparities.min())
    predictions = predictions * valid_mask
    disparities = disparities * valid_mask

    predictions = predictions.detach().cpu()
    disparities = disparities.detach().cpu()

    for idx, (pred, gt) in enumerate(zip(predictions, disparities)):
        pred = pred.permute(1, 2, 0).numpy()
        gt = gt.permute(1, 2, 0).numpy()
        # plot pred and gt side by side
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(pred)
        ax[0].set_title("Prediction")
        ax[1].imshow(gt)
        ax[1].set_title("Ground Truth")
        save_name = os.path.join(save_path, "{}_{}.png".format(prefix, idx))
        plt.savefig(save_name)
        plt.close()
