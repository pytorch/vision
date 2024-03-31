import random
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torch import Tensor

T_FLOW = Union[Tensor, np.ndarray, None]
T_MASK = Union[Tensor, np.ndarray, None]
T_STEREO_TENSOR = Tuple[Tensor, Tensor]
T_COLOR_AUG_PARAM = Union[float, Tuple[float, float]]


def rand_float_range(size: Sequence[int], low: float, high: float) -> Tensor:
    return (low - high) * torch.rand(size) + high


class InterpolationStrategy:

    _valid_modes: List[str] = ["mixed", "bicubic", "bilinear"]

    def __init__(self, mode: str = "mixed") -> None:
        if mode not in self._valid_modes:
            raise ValueError(f"Invalid interpolation mode: {mode}. Valid modes are: {self._valid_modes}")

        if mode == "mixed":
            self.strategies = [F.InterpolationMode.BILINEAR, F.InterpolationMode.BICUBIC]
        elif mode == "bicubic":
            self.strategies = [F.InterpolationMode.BICUBIC]
        elif mode == "bilinear":
            self.strategies = [F.InterpolationMode.BILINEAR]

    def __call__(self) -> F.InterpolationMode:
        return random.choice(self.strategies)

    @classmethod
    def is_valid(mode: str) -> bool:
        return mode in InterpolationStrategy._valid_modes

    @property
    def valid_modes() -> List[str]:
        return InterpolationStrategy._valid_modes


class ValidateModelInput(torch.nn.Module):
    # Pass-through transform that checks the shape and dtypes to make sure the model gets what it expects
    def forward(self, images: T_STEREO_TENSOR, disparities: T_FLOW, masks: T_MASK):
        if images[0].shape != images[1].shape:
            raise ValueError("img1 and img2 should have the same shape.")
        h, w = images[0].shape[-2:]
        if disparities[0] is not None and disparities[0].shape != (1, h, w):
            raise ValueError(f"disparities[0].shape should be (1, {h}, {w}) instead of {disparities[0].shape}")
        if masks[0] is not None:
            if masks[0].shape != (h, w):
                raise ValueError(f"masks[0].shape should be ({h}, {w}) instead of {masks[0].shape}")
            if masks[0].dtype != torch.bool:
                raise TypeError(f"masks[0] should be of dtype torch.bool instead of {masks[0].dtype}")

        return images, disparities, masks


class ConvertToGrayscale(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        images: Tuple[PIL.Image.Image, PIL.Image.Image],
        disparities: Tuple[T_FLOW, T_FLOW],
        masks: Tuple[T_MASK, T_MASK],
    ) -> Tuple[T_STEREO_TENSOR, Tuple[T_FLOW, T_FLOW], Tuple[T_MASK, T_MASK]]:
        img_left = F.rgb_to_grayscale(images[0], num_output_channels=3)
        img_right = F.rgb_to_grayscale(images[1], num_output_channels=3)

        return (img_left, img_right), disparities, masks


class MakeValidDisparityMask(torch.nn.Module):
    def __init__(self, max_disparity: Optional[int] = 256) -> None:
        super().__init__()
        self.max_disparity = max_disparity

    def forward(
        self,
        images: T_STEREO_TENSOR,
        disparities: Tuple[T_FLOW, T_FLOW],
        masks: Tuple[T_MASK, T_MASK],
    ) -> Tuple[T_STEREO_TENSOR, Tuple[T_FLOW, T_FLOW], Tuple[T_MASK, T_MASK]]:
        valid_masks = tuple(
            torch.ones(images[idx].shape[-2:], dtype=torch.bool, device=images[idx].device) if mask is None else mask
            for idx, mask in enumerate(masks)
        )

        valid_masks = tuple(
            torch.logical_and(mask, disparity > 0).squeeze(0) if disparity is not None else mask
            for mask, disparity in zip(valid_masks, disparities)
        )

        if self.max_disparity is not None:
            valid_masks = tuple(
                torch.logical_and(mask, disparity < self.max_disparity).squeeze(0) if disparity is not None else mask
                for mask, disparity in zip(valid_masks, disparities)
            )

        return images, disparities, valid_masks


class ToGPU(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        images: T_STEREO_TENSOR,
        disparities: Tuple[T_FLOW, T_FLOW],
        masks: Tuple[T_MASK, T_MASK],
    ) -> Tuple[T_STEREO_TENSOR, Tuple[T_FLOW, T_FLOW], Tuple[T_MASK, T_MASK]]:
        dev_images = tuple(image.cuda() for image in images)
        dev_disparities = tuple(map(lambda x: x.cuda() if x is not None else None, disparities))
        dev_masks = tuple(map(lambda x: x.cuda() if x is not None else None, masks))
        return dev_images, dev_disparities, dev_masks


class ConvertImageDtype(torch.nn.Module):
    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.dtype = dtype

    def forward(
        self,
        images: T_STEREO_TENSOR,
        disparities: Tuple[T_FLOW, T_FLOW],
        masks: Tuple[T_MASK, T_MASK],
    ) -> Tuple[T_STEREO_TENSOR, Tuple[T_FLOW, T_FLOW], Tuple[T_MASK, T_MASK]]:
        img_left = F.convert_image_dtype(images[0], dtype=self.dtype)
        img_right = F.convert_image_dtype(images[1], dtype=self.dtype)

        img_left = img_left.contiguous()
        img_right = img_right.contiguous()

        return (img_left, img_right), disparities, masks


class Normalize(torch.nn.Module):
    def __init__(self, mean: List[float], std: List[float]) -> None:
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(
        self,
        images: T_STEREO_TENSOR,
        disparities: Tuple[T_FLOW, T_FLOW],
        masks: Tuple[T_MASK, T_MASK],
    ) -> Tuple[T_STEREO_TENSOR, Tuple[T_FLOW, T_FLOW], Tuple[T_MASK, T_MASK]]:

        img_left = F.normalize(images[0], mean=self.mean, std=self.std)
        img_right = F.normalize(images[1], mean=self.mean, std=self.std)

        img_left = img_left.contiguous()
        img_right = img_right.contiguous()

        return (img_left, img_right), disparities, masks


class ToTensor(torch.nn.Module):
    def forward(
        self,
        images: Tuple[PIL.Image.Image, PIL.Image.Image],
        disparities: Tuple[T_FLOW, T_FLOW],
        masks: Tuple[T_MASK, T_MASK],
    ) -> Tuple[T_STEREO_TENSOR, Tuple[T_FLOW, T_FLOW], Tuple[T_MASK, T_MASK]]:
        if images[0] is None:
            raise ValueError("img_left is None")
        if images[1] is None:
            raise ValueError("img_right is None")

        img_left = F.pil_to_tensor(images[0])
        img_right = F.pil_to_tensor(images[1])
        disparity_tensors = ()
        mask_tensors = ()

        for idx in range(2):
            disparity_tensors += (torch.from_numpy(disparities[idx]),) if disparities[idx] is not None else (None,)
            mask_tensors += (torch.from_numpy(masks[idx]),) if masks[idx] is not None else (None,)

        return (img_left, img_right), disparity_tensors, mask_tensors


class AsymmetricColorJitter(T.ColorJitter):
    # p determines the probability of doing asymmetric vs symmetric color jittering
    def __init__(
        self,
        brightness: T_COLOR_AUG_PARAM = 0,
        contrast: T_COLOR_AUG_PARAM = 0,
        saturation: T_COLOR_AUG_PARAM = 0,
        hue: T_COLOR_AUG_PARAM = 0,
        p: float = 0.2,
    ):
        super().__init__(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        self.p = p

    def forward(
        self,
        images: T_STEREO_TENSOR,
        disparities: Tuple[T_FLOW, T_FLOW],
        masks: Tuple[T_MASK, T_MASK],
    ) -> Tuple[T_STEREO_TENSOR, Tuple[T_FLOW, T_FLOW], Tuple[T_MASK, T_MASK]]:

        if torch.rand(1) < self.p:
            # asymmetric: different transform for img1 and img2
            img_left = super().forward(images[0])
            img_right = super().forward(images[1])
        else:
            # symmetric: same transform for img1 and img2
            batch = torch.stack(images)
            batch = super().forward(batch)
            img_left, img_right = batch[0], batch[1]

        return (img_left, img_right), disparities, masks


class AsymetricGammaAdjust(torch.nn.Module):
    def __init__(self, p: float, gamma_range: Tuple[float, float], gain: float = 1) -> None:
        super().__init__()
        self.gamma_range = gamma_range
        self.gain = gain
        self.p = p

    def forward(
        self,
        images: T_STEREO_TENSOR,
        disparities: Tuple[T_FLOW, T_FLOW],
        masks: Tuple[T_MASK, T_MASK],
    ) -> Tuple[T_STEREO_TENSOR, Tuple[T_FLOW, T_FLOW], Tuple[T_MASK, T_MASK]]:

        gamma = rand_float_range((1,), low=self.gamma_range[0], high=self.gamma_range[1]).item()

        if torch.rand(1) < self.p:
            # asymmetric: different transform for img1 and img2
            img_left = F.adjust_gamma(images[0], gamma, gain=self.gain)
            img_right = F.adjust_gamma(images[1], gamma, gain=self.gain)
        else:
            # symmetric: same transform for img1 and img2
            batch = torch.stack(images)
            batch = F.adjust_gamma(batch, gamma, gain=self.gain)
            img_left, img_right = batch[0], batch[1]

        return (img_left, img_right), disparities, masks


class RandomErase(torch.nn.Module):
    # Produces multiple symmetric random erasures
    # these can be viewed as occlusions present in both camera views.
    # Similarly to Optical Flow occlusion prediction tasks, we mask these pixels in the disparity map
    def __init__(
        self,
        p: float = 0.5,
        erase_px_range: Tuple[int, int] = (50, 100),
        value: Union[Tensor, float] = 0,
        inplace: bool = False,
        max_erase: int = 2,
    ):
        super().__init__()
        self.min_px_erase = erase_px_range[0]
        self.max_px_erase = erase_px_range[1]
        if self.max_px_erase < 0:
            raise ValueError("erase_px_range[1] should be equal or greater than 0")
        if self.min_px_erase < 0:
            raise ValueError("erase_px_range[0] should be equal or greater than 0")
        if self.min_px_erase > self.max_px_erase:
            raise ValueError("erase_prx_range[0] should be equal or lower than erase_px_range[1]")

        self.p = p
        self.value = value
        self.inplace = inplace
        self.max_erase = max_erase

    def forward(
        self,
        images: T_STEREO_TENSOR,
        disparities: T_STEREO_TENSOR,
        masks: T_STEREO_TENSOR,
    ) -> Tuple[T_STEREO_TENSOR, Tuple[T_FLOW, T_FLOW], Tuple[T_MASK, T_MASK]]:

        if torch.rand(1) < self.p:
            return images, disparities, masks

        image_left, image_right = images
        mask_left, mask_right = masks
        for _ in range(torch.randint(self.max_erase, size=(1,)).item()):
            y, x, h, w, v = self._get_params(image_left)
            image_right = F.erase(image_right, y, x, h, w, v, self.inplace)
            image_left = F.erase(image_left, y, x, h, w, v, self.inplace)
            # similarly to optical flow occlusion prediction, we consider
            # any erasure pixels that are in both images to be occluded therefore
            # we mark them as invalid
            if mask_left is not None:
                mask_left = F.erase(mask_left, y, x, h, w, False, self.inplace)
            if mask_right is not None:
                mask_right = F.erase(mask_right, y, x, h, w, False, self.inplace)

        return (image_left, image_right), disparities, (mask_left, mask_right)

    def _get_params(self, img: torch.Tensor) -> Tuple[int, int, int, int, float]:
        img_h, img_w = img.shape[-2:]
        crop_h, crop_w = (
            random.randint(self.min_px_erase, self.max_px_erase),
            random.randint(self.min_px_erase, self.max_px_erase),
        )
        crop_x, crop_y = (random.randint(0, img_w - crop_w), random.randint(0, img_h - crop_h))

        return crop_y, crop_x, crop_h, crop_w, self.value


class RandomOcclusion(torch.nn.Module):
    # This adds an occlusion in the right image
    # the occluded patch works as a patch erase where the erase value is the mean
    # of the pixels from the selected zone
    def __init__(self, p: float = 0.5, occlusion_px_range: Tuple[int, int] = (50, 100), inplace: bool = False):
        super().__init__()

        self.min_px_occlusion = occlusion_px_range[0]
        self.max_px_occlusion = occlusion_px_range[1]

        if self.max_px_occlusion < 0:
            raise ValueError("occlusion_px_range[1] should be greater or equal than 0")
        if self.min_px_occlusion < 0:
            raise ValueError("occlusion_px_range[0] should be greater or equal than 0")
        if self.min_px_occlusion > self.max_px_occlusion:
            raise ValueError("occlusion_px_range[0] should be lower than occlusion_px_range[1]")

        self.p = p
        self.inplace = inplace

    def forward(
        self,
        images: T_STEREO_TENSOR,
        disparities: T_STEREO_TENSOR,
        masks: T_STEREO_TENSOR,
    ) -> Tuple[T_STEREO_TENSOR, Tuple[T_FLOW, T_FLOW], Tuple[T_MASK, T_MASK]]:

        left_image, right_image = images

        if torch.rand(1) < self.p:
            return images, disparities, masks

        y, x, h, w, v = self._get_params(right_image)
        right_image = F.erase(right_image, y, x, h, w, v, self.inplace)

        return ((left_image, right_image), disparities, masks)

    def _get_params(self, img: torch.Tensor) -> Tuple[int, int, int, int, float]:
        img_h, img_w = img.shape[-2:]
        crop_h, crop_w = (
            random.randint(self.min_px_occlusion, self.max_px_occlusion),
            random.randint(self.min_px_occlusion, self.max_px_occlusion),
        )

        crop_x, crop_y = (random.randint(0, img_w - crop_w), random.randint(0, img_h - crop_h))
        occlusion_value = img[..., crop_y : crop_y + crop_h, crop_x : crop_x + crop_w].mean(dim=(-2, -1), keepdim=True)

        return (crop_y, crop_x, crop_h, crop_w, occlusion_value)


class RandomSpatialShift(torch.nn.Module):
    # This transform applies a vertical shift and a slight angle rotation and the same time
    def __init__(
        self, p: float = 0.5, max_angle: float = 0.1, max_px_shift: int = 2, interpolation_type: str = "bilinear"
    ) -> None:
        super().__init__()
        self.p = p
        self.max_angle = max_angle
        self.max_px_shift = max_px_shift
        self._interpolation_mode_strategy = InterpolationStrategy(interpolation_type)

    def forward(
        self,
        images: T_STEREO_TENSOR,
        disparities: T_STEREO_TENSOR,
        masks: T_STEREO_TENSOR,
    ) -> Tuple[T_STEREO_TENSOR, Tuple[T_FLOW, T_FLOW], Tuple[T_MASK, T_MASK]]:
        # the transform is applied only on the right image
        # in order to mimic slight calibration issues
        img_left, img_right = images

        INTERP_MODE = self._interpolation_mode_strategy()

        if torch.rand(1) < self.p:
            # [0, 1] -> [-a, a]
            shift = rand_float_range((1,), low=-self.max_px_shift, high=self.max_px_shift).item()
            angle = rand_float_range((1,), low=-self.max_angle, high=self.max_angle).item()
            # sample center point for the rotation matrix
            y = torch.randint(size=(1,), low=0, high=img_right.shape[-2]).item()
            x = torch.randint(size=(1,), low=0, high=img_right.shape[-1]).item()
            # apply affine transformations
            img_right = F.affine(
                img_right,
                angle=angle,
                translate=[0, shift],  # translation only on the y-axis
                center=[x, y],
                scale=1.0,
                shear=0.0,
                interpolation=INTERP_MODE,
            )

        return ((img_left, img_right), disparities, masks)


class RandomHorizontalFlip(torch.nn.Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(
        self,
        images: T_STEREO_TENSOR,
        disparities: Tuple[T_FLOW, T_FLOW],
        masks: Tuple[T_MASK, T_MASK],
    ) -> Tuple[T_STEREO_TENSOR, Tuple[T_FLOW, T_FLOW], Tuple[T_MASK, T_MASK]]:

        img_left, img_right = images
        dsp_left, dsp_right = disparities
        mask_left, mask_right = masks

        if dsp_right is not None and torch.rand(1) < self.p:
            img_left, img_right = F.hflip(img_left), F.hflip(img_right)
            dsp_left, dsp_right = F.hflip(dsp_left), F.hflip(dsp_right)
            if mask_left is not None and mask_right is not None:
                mask_left, mask_right = F.hflip(mask_left), F.hflip(mask_right)
            return ((img_right, img_left), (dsp_right, dsp_left), (mask_right, mask_left))

        return images, disparities, masks


class Resize(torch.nn.Module):
    def __init__(self, resize_size: Tuple[int, ...], interpolation_type: str = "bilinear") -> None:
        super().__init__()
        self.resize_size = list(resize_size)  # doing this to keep mypy happy
        self._interpolation_mode_strategy = InterpolationStrategy(interpolation_type)

    def forward(
        self,
        images: T_STEREO_TENSOR,
        disparities: Tuple[T_FLOW, T_FLOW],
        masks: Tuple[T_MASK, T_MASK],
    ) -> Tuple[T_STEREO_TENSOR, Tuple[T_FLOW, T_FLOW], Tuple[T_MASK, T_MASK]]:
        resized_images = ()
        resized_disparities = ()
        resized_masks = ()

        INTERP_MODE = self._interpolation_mode_strategy()

        for img in images:
            # We hard-code antialias=False to preserve results after we changed
            # its default from None to True (see
            # https://github.com/pytorch/vision/pull/7160)
            # TODO: we could re-train the stereo models with antialias=True?
            resized_images += (F.resize(img, self.resize_size, interpolation=INTERP_MODE, antialias=False),)

        for dsp in disparities:
            if dsp is not None:
                # rescale disparity to match the new image size
                scale_x = self.resize_size[1] / dsp.shape[-1]
                resized_disparities += (F.resize(dsp, self.resize_size, interpolation=INTERP_MODE) * scale_x,)
            else:
                resized_disparities += (None,)

        for mask in masks:
            if mask is not None:
                resized_masks += (
                    # we squeeze and unsqueeze because the API requires > 3D tensors
                    F.resize(
                        mask.unsqueeze(0),
                        self.resize_size,
                        interpolation=F.InterpolationMode.NEAREST,
                    ).squeeze(0),
                )
            else:
                resized_masks += (None,)

        return resized_images, resized_disparities, resized_masks


class RandomRescaleAndCrop(torch.nn.Module):
    # This transform will resize the input with a given proba, and then crop it.
    # These are the reversed operations of the built-in RandomResizedCrop,
    # although the order of the operations doesn't matter too much: resizing a
    # crop would give the same result as cropping a resized image, up to
    # interpolation artifact at the borders of the output.
    #
    # The reason we don't rely on RandomResizedCrop is because of a significant
    # difference in the parametrization of both transforms, in particular,
    # because of the way the random parameters are sampled in both transforms,
    # which leads to fairly different results (and different epe). For more details see
    # https://github.com/pytorch/vision/pull/5026/files#r762932579
    def __init__(
        self,
        crop_size: Tuple[int, int],
        scale_range: Tuple[float, float] = (-0.2, 0.5),
        rescale_prob: float = 0.8,
        scaling_type: str = "exponential",
        interpolation_type: str = "bilinear",
    ) -> None:
        super().__init__()
        self.crop_size = crop_size
        self.min_scale = scale_range[0]
        self.max_scale = scale_range[1]
        self.rescale_prob = rescale_prob
        self.scaling_type = scaling_type
        self._interpolation_mode_strategy = InterpolationStrategy(interpolation_type)

        if self.scaling_type == "linear" and self.min_scale < 0:
            raise ValueError("min_scale must be >= 0 for linear scaling")

    def forward(
        self,
        images: T_STEREO_TENSOR,
        disparities: Tuple[T_FLOW, T_FLOW],
        masks: Tuple[T_MASK, T_MASK],
    ) -> Tuple[T_STEREO_TENSOR, Tuple[T_FLOW, T_FLOW], Tuple[T_MASK, T_MASK]]:

        img_left, img_right = images
        dsp_left, dsp_right = disparities
        mask_left, mask_right = masks
        INTERP_MODE = self._interpolation_mode_strategy()

        # randomly sample scale
        h, w = img_left.shape[-2:]
        # Note: in original code, they use + 1 instead of + 8 for sparse datasets (e.g. Kitti)
        # It shouldn't matter much
        min_scale = max((self.crop_size[0] + 8) / h, (self.crop_size[1] + 8) / w)

        # exponential scaling will draw a random scale in (min_scale, max_scale) and then raise
        # 2 to the power of that random value. This final scale distribution will have a different
        # mean and variance than a uniform distribution. Note that a scale of 1 will result in
        # a rescaling of 2X the original size, whereas a scale of -1 will result in a rescaling
        # of 0.5X the original size.
        if self.scaling_type == "exponential":
            scale = 2 ** torch.empty(1, dtype=torch.float32).uniform_(self.min_scale, self.max_scale).item()
        # linear scaling will draw a random scale in (min_scale, max_scale)
        elif self.scaling_type == "linear":
            scale = torch.empty(1, dtype=torch.float32).uniform_(self.min_scale, self.max_scale).item()

        scale = max(scale, min_scale)

        new_h, new_w = round(h * scale), round(w * scale)

        if torch.rand(1).item() < self.rescale_prob:
            # rescale the images
            img_left = F.resize(img_left, size=(new_h, new_w), interpolation=INTERP_MODE)
            img_right = F.resize(img_right, size=(new_h, new_w), interpolation=INTERP_MODE)

            resized_masks, resized_disparities = (), ()

            for disparity, mask in zip(disparities, masks):
                if disparity is not None:
                    if mask is None:
                        resized_disparity = F.resize(disparity, size=(new_h, new_w), interpolation=INTERP_MODE)
                        # rescale the disparity
                        resized_disparity = (
                            resized_disparity * torch.tensor([scale], device=resized_disparity.device)[:, None, None]
                        )
                        resized_mask = None
                    else:
                        resized_disparity, resized_mask = _resize_sparse_flow(
                            disparity, mask, scale_x=scale, scale_y=scale
                        )
                resized_masks += (resized_mask,)
                resized_disparities += (resized_disparity,)

        else:
            resized_disparities = disparities
            resized_masks = masks

        disparities = resized_disparities
        masks = resized_masks

        # Note: For sparse datasets (Kitti), the original code uses a "margin"
        # See e.g. https://github.com/princeton-vl/RAFT/blob/master/core/utils/augmentor.py#L220:L220
        # We don't, not sure if it matters much
        y0 = torch.randint(0, img_left.shape[1] - self.crop_size[0], size=(1,)).item()
        x0 = torch.randint(0, img_right.shape[2] - self.crop_size[1], size=(1,)).item()

        img_left = F.crop(img_left, y0, x0, self.crop_size[0], self.crop_size[1])
        img_right = F.crop(img_right, y0, x0, self.crop_size[0], self.crop_size[1])
        if dsp_left is not None:
            dsp_left = F.crop(disparities[0], y0, x0, self.crop_size[0], self.crop_size[1])
        if dsp_right is not None:
            dsp_right = F.crop(disparities[1], y0, x0, self.crop_size[0], self.crop_size[1])

        cropped_masks = ()
        for mask in masks:
            if mask is not None:
                mask = F.crop(mask, y0, x0, self.crop_size[0], self.crop_size[1])
            cropped_masks += (mask,)

        return ((img_left, img_right), (dsp_left, dsp_right), cropped_masks)


def _resize_sparse_flow(
    flow: Tensor, valid_flow_mask: Tensor, scale_x: float = 1.0, scale_y: float = 0.0
) -> Tuple[Tensor, Tensor]:
    # This resizes both the flow and the valid_flow_mask mask (which is assumed to be reasonably sparse)
    # There are as-many non-zero values in the original flow as in the resized flow (up to OOB)
    # So for example if scale_x = scale_y = 2, the sparsity of the output flow is multiplied by 4

    h, w = flow.shape[-2:]

    h_new = int(round(h * scale_y))
    w_new = int(round(w * scale_x))
    flow_new = torch.zeros(size=[1, h_new, w_new], dtype=flow.dtype)
    valid_new = torch.zeros(size=[h_new, w_new], dtype=valid_flow_mask.dtype)

    jj, ii = torch.meshgrid(torch.arange(w), torch.arange(h), indexing="xy")

    ii_valid, jj_valid = ii[valid_flow_mask], jj[valid_flow_mask]

    ii_valid_new = torch.round(ii_valid.to(float) * scale_y).to(torch.long)
    jj_valid_new = torch.round(jj_valid.to(float) * scale_x).to(torch.long)

    within_bounds_mask = (0 <= ii_valid_new) & (ii_valid_new < h_new) & (0 <= jj_valid_new) & (jj_valid_new < w_new)

    ii_valid = ii_valid[within_bounds_mask]
    jj_valid = jj_valid[within_bounds_mask]
    ii_valid_new = ii_valid_new[within_bounds_mask]
    jj_valid_new = jj_valid_new[within_bounds_mask]

    valid_flow_new = flow[:, ii_valid, jj_valid]
    valid_flow_new *= scale_x

    flow_new[:, ii_valid_new, jj_valid_new] = valid_flow_new
    valid_new[ii_valid_new, jj_valid_new] = valid_flow_mask[ii_valid, jj_valid]

    return flow_new, valid_new.bool()


class Compose(torch.nn.Module):
    def __init__(self, transforms: List[Callable]):
        super().__init__()
        self.transforms = transforms

    @torch.inference_mode()
    def forward(self, images, disparities, masks):
        for t in self.transforms:
            images, disparities, masks = t(images, disparities, masks)
        return images, disparities, masks
