from typing import List, Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.prototype.models.depth.stereo.raft_stereo import grid_sample, make_coords_grid


def make_gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
    """Function to create a 2D Gaussian kernel."""

    x = torch.arange(kernel_size, dtype=torch.float32)
    y = torch.arange(kernel_size, dtype=torch.float32)
    x = x - (kernel_size - 1) / 2
    y = y - (kernel_size - 1) / 2
    x, y = torch.meshgrid(x, y)
    grid = (x**2 + y**2) / (2 * sigma**2)
    kernel = torch.exp(-grid)
    kernel = kernel / kernel.sum()
    return kernel


def _sequence_loss_fn(
    flow_preds: List[Tensor],
    flow_gt: Tensor,
    valid_flow_mask: Optional[Tensor],
    gamma: Tensor,
    max_flow: int = 256,
    exclude_large: bool = False,
    weights: Optional[Tensor] = None,
):
    """Loss function defined over sequence of flow predictions"""
    torch._assert(
        gamma < 1,
        "sequence_loss: `gamma` must be lower than 1, but got {}".format(gamma),
    )

    if exclude_large:
        # exclude invalid pixels and extremely large diplacements
        flow_norm = torch.sum(flow_gt**2, dim=1).sqrt()
        if valid_flow_mask is not None:
            valid_flow_mask = valid_flow_mask & (flow_norm < max_flow)
        else:
            valid_flow_mask = flow_norm < max_flow

    if valid_flow_mask is not None:
        valid_flow_mask = valid_flow_mask.unsqueeze(1)
    flow_preds = torch.stack(flow_preds)  # shape = (num_flow_updates, batch_size, 2, H, W)

    abs_diff = (flow_preds - flow_gt).abs()
    if valid_flow_mask is not None:
        abs_diff = abs_diff * valid_flow_mask.unsqueeze(0)

    abs_diff = abs_diff.mean(axis=(1, 2, 3, 4))
    num_predictions = flow_preds.shape[0]

    # allocating on CPU and moving to device during run-time can force
    # an unwanted GPU synchronization that produces a large overhead
    if weights is None or len(weights) != num_predictions:
        weights = gamma ** torch.arange(num_predictions - 1, -1, -1, device=flow_preds.device, dtype=flow_preds.dtype)

    flow_loss = (abs_diff * weights).sum()
    return flow_loss, weights


class SequenceLoss(nn.Module):
    def __init__(self, gamma: float = 0.8, max_flow: int = 256, exclude_large_flows: bool = False) -> None:
        """
        Args:
            gamma: value for the exponential weighting of the loss across frames
            max_flow: maximum flow value to exclude
            exclude_large_flows: whether to exclude large flows
        """

        super().__init__()
        self.max_flow = max_flow
        self.excluding_large = exclude_large_flows
        self.register_buffer("gamma", torch.tensor([gamma]))
        # cache the scale factor for the loss
        self._weights = None

    def forward(self, flow_preds: List[Tensor], flow_gt: Tensor, valid_flow_mask: Optional[Tensor]) -> Tensor:
        """
        Args:
            flow_preds: list of flow predictions of shape (batch_size, C, H, W)
            flow_gt: ground truth flow of shape (batch_size, C, H, W)
            valid_flow_mask: mask of valid flow pixels of shape (batch_size, H, W)
        """
        loss, weights = _sequence_loss_fn(
            flow_preds, flow_gt, valid_flow_mask, self.gamma, self.max_flow, self.excluding_large, self._weights
        )
        self._weights = weights
        return loss

    def set_gamma(self, gamma: float) -> None:
        self.gamma.fill_(gamma)
        # reset the cached scale factor
        self._weights = None


def _ssim_loss_fn(
    source: Tensor,
    reference: Tensor,
    kernel: Tensor,
    eps: float = 1e-8,
    c1: float = 0.01**2,
    c2: float = 0.03**2,
    use_padding: bool = False,
) -> Tensor:
    # ref: Algorithm section: https://en.wikipedia.org/wiki/Structural_similarity
    # ref: Alternative implementation: https://kornia.readthedocs.io/en/latest/_modules/kornia/metrics/ssim.html#ssim

    torch._assert(
        source.ndim == reference.ndim == 4,
        "SSIM: `source` and `reference` must be 4-dimensional tensors",
    )

    torch._assert(
        source.shape == reference.shape,
        "SSIM: `source` and `reference` must have the same shape, but got {} and {}".format(
            source.shape, reference.shape
        ),
    )

    B, C, H, W = source.shape
    kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)
    if use_padding:
        pad_size = kernel.shape[2] // 2
        source = F.pad(source, (pad_size, pad_size, pad_size, pad_size), "reflect")
        reference = F.pad(reference, (pad_size, pad_size, pad_size, pad_size), "reflect")

    mu1 = F.conv2d(source, kernel, groups=C)
    mu2 = F.conv2d(reference, kernel, groups=C)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1 * mu2
    mu_img1_sq = F.conv2d(source.pow(2), kernel, groups=C)
    mu_img2_sq = F.conv2d(reference.pow(2), kernel, groups=C)
    mu_img1_mu2 = F.conv2d(source * reference, kernel, groups=C)

    sigma1_sq = mu_img1_sq - mu1_sq
    sigma2_sq = mu_img2_sq - mu2_sq
    sigma12 = mu_img1_mu2 - mu1_mu2

    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim = numerator / (denominator + eps)

    # doing 1 - ssim because we want to maximize the ssim
    return 1 - ssim.mean(dim=(1, 2, 3))


class SSIM(nn.Module):
    def __init__(
        self,
        kernel_size: int = 11,
        max_val: float = 1.0,
        sigma: float = 1.5,
        eps: float = 1e-12,
        use_padding: bool = True,
    ) -> None:
        """SSIM loss function.

        Args:
            kernel_size: size of the Gaussian kernel
            max_val: constant scaling factor
            sigma: sigma of the Gaussian kernel
            eps: constant for division by zero
            use_padding: whether to pad the input tensor such that we have a score for each pixel
        """
        super().__init__()

        self.kernel_size = kernel_size
        self.max_val = max_val
        self.sigma = sigma

        gaussian_kernel = make_gaussian_kernel(kernel_size, sigma)
        self.register_buffer("gaussian_kernel", gaussian_kernel)

        self.c1 = (0.01 * self.max_val) ** 2
        self.c2 = (0.03 * self.max_val) ** 2

        self.use_padding = use_padding
        self.eps = eps

    def forward(self, source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """
        Args:
            source: source image of shape (batch_size, C, H, W)
            reference: reference image of shape (batch_size, C, H, W)

        Returns:
            SSIM loss of shape (batch_size,)
        """
        return _ssim_loss_fn(
            source,
            reference,
            kernel=self.gaussian_kernel,
            c1=self.c1,
            c2=self.c2,
            use_padding=self.use_padding,
            eps=self.eps,
        )


def _smoothness_loss_fn(img_gx: Tensor, img_gy: Tensor, val_gx: Tensor, val_gy: Tensor):
    # ref: https://github.com/nianticlabs/monodepth2/blob/b676244e5a1ca55564eb5d16ab521a48f823af31/layers.py#L202

    torch._assert(
        img_gx.ndim >= 3,
        "smoothness_loss: `img_gx` must be at least 3-dimensional tensor of shape (..., C, H, W)",
    )

    torch._assert(
        img_gx.ndim == val_gx.ndim,
        "smoothness_loss: `img_gx` and `depth_gx` must have the same dimensionality, but got {} and {}".format(
            img_gx.ndim, val_gx.ndim
        ),
    )

    for idx in range(img_gx.ndim):
        torch._assert(
            (img_gx.shape[idx] == val_gx.shape[idx] or (img_gx.shape[idx] == 1 or val_gx.shape[idx] == 1)),
            "smoothness_loss: `img_gx` and `depth_gx` must have either the same shape or broadcastable shape, but got {} and {}".format(
                img_gx.shape, val_gx.shape
            ),
        )

    # -3 is channel dimension
    weights_x = torch.exp(-torch.mean(torch.abs(val_gx), axis=-3, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(val_gy), axis=-3, keepdim=True))

    smoothness_x = img_gx * weights_x
    smoothness_y = img_gy * weights_y

    smoothness = (torch.abs(smoothness_x) + torch.abs(smoothness_y)).mean(axis=(-3, -2, -1))
    return smoothness


class SmoothnessLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def _x_gradient(self, img: Tensor) -> Tensor:
        if img.ndim > 4:
            original_shape = img.shape
            is_reshaped = True
            img = img.reshape(-1, *original_shape[-3:])
        else:
            is_reshaped = False

        padded = F.pad(img, (0, 1, 0, 0), mode="replicate")
        grad = padded[..., :, :-1] - padded[..., :, 1:]
        if is_reshaped:
            grad = grad.reshape(original_shape)
        return grad

    def _y_gradient(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim > 4:
            original_shape = x.shape
            is_reshaped = True
            x = x.reshape(-1, *original_shape[-3:])
        else:
            is_reshaped = False

        padded = F.pad(x, (0, 0, 0, 1), mode="replicate")
        grad = padded[..., :-1, :] - padded[..., 1:, :]
        if is_reshaped:
            grad = grad.reshape(original_shape)
        return grad

    def forward(self, images: Tensor, vals: Tensor) -> Tensor:
        """
        Args:
            images: tensor of shape (D1, D2, ..., DN, C, H, W)
            vals: tensor of shape (D1, D2, ..., DN, 1, H, W)

        Returns:
            smoothness loss of shape (D1, D2, ..., DN)
        """
        img_gx = self._x_gradient(images)
        img_gy = self._y_gradient(images)

        val_gx = self._x_gradient(vals)
        val_gy = self._y_gradient(vals)

        return _smoothness_loss_fn(img_gx, img_gy, val_gx, val_gy)


def _flow_sequence_consistency_loss_fn(
    flow_preds: List[Tensor],
    gamma: float = 0.8,
    resize_factor: float = 0.25,
    rescale_factor: float = 0.25,
    rescale_mode: str = "bilinear",
    weights: Optional[Tensor] = None,
):
    """Loss function defined over sequence of flow predictions"""

    # Simplified version of ref: https://arxiv.org/pdf/2006.11242.pdf
    # In the original paper, an additional refinement network is used to refine a flow prediction.
    # Each step performed by the recurrent module in Raft or CREStereo is a refinement step using a delta_flow update.
    # which should be consistent with the previous step. In this implementation, we simplify the overall loss
    # term and ignore left-right consistency loss or photometric loss which can be treated separately.

    torch._assert(
        rescale_factor <= 1.0,
        "sequence_consistency_loss: `rescale_factor` must be less than or equal to 1, but got {}".format(
            rescale_factor
        ),
    )

    flow_preds = torch.stack(flow_preds)  # shape = (num_flow_updates, batch_size, 2, H, W)
    N, B, C, H, W = flow_preds.shape

    # rescale flow predictions to account for bilinear upsampling artifacts
    if rescale_factor:
        flow_preds = (
            F.interpolate(
                flow_preds.view(N * B, C, H, W), scale_factor=resize_factor, mode=rescale_mode, align_corners=True
            )
        ) * rescale_factor
        flow_preds = torch.stack(torch.chunk(flow_preds, N, dim=0), dim=0)

    # force the next prediction to be similar to the previous prediction
    abs_diff = (flow_preds[1:] - flow_preds[:-1]).square()
    abs_diff = abs_diff.mean(axis=(1, 2, 3, 4))

    num_predictions = flow_preds.shape[0] - 1  # because we are comparing differences
    if weights is None or len(weights) != num_predictions:
        weights = gamma ** torch.arange(num_predictions - 1, -1, -1, device=flow_preds.device, dtype=flow_preds.dtype)

    flow_loss = (abs_diff * weights).sum()
    return flow_loss, weights


class FlowSequenceConsistencyLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 0.8,
        resize_factor: float = 0.25,
        rescale_factor: float = 0.25,
        rescale_mode: str = "bilinear",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.resize_factor = resize_factor
        self.rescale_factor = rescale_factor
        self.rescale_mode = rescale_mode
        self._weights = None

    def forward(self, flow_preds: List[Tensor]) -> Tensor:
        """
        Args:
            flow_preds: list of tensors of shape (batch_size, C, H, W)

        Returns:
            sequence consistency loss of shape (batch_size,)
        """
        loss, weights = _flow_sequence_consistency_loss_fn(
            flow_preds,
            gamma=self.gamma,
            resize_factor=self.resize_factor,
            rescale_factor=self.rescale_factor,
            rescale_mode=self.rescale_mode,
            weights=self._weights,
        )
        self._weights = weights
        return loss

    def set_gamma(self, gamma: float) -> None:
        self.gamma.fill_(gamma)
        # reset the cached scale factor
        self._weights = None


def _psnr_loss_fn(source: torch.Tensor, target: torch.Tensor, max_val: float) -> torch.Tensor:
    torch._assert(
        source.shape == target.shape,
        "psnr_loss: source and target must have the same shape, but got {} and {}".format(source.shape, target.shape),
    )

    # ref https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    return 10 * torch.log10(max_val**2 / ((source - target).pow(2).mean(axis=(-3, -2, -1))))


class PSNRLoss(nn.Module):
    def __init__(self, max_val: float = 256) -> None:
        """
        Args:
            max_val: maximum value of the input tensor. This refers to the maximum domain value of the input tensor.

        """
        super().__init__()
        self.max_val = max_val

    def forward(self, source: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            source: tensor of shape (D1, D2, ..., DN, C, H, W)
            target: tensor of shape (D1, D2, ..., DN, C, H, W)

        Returns:
            psnr loss of shape (D1, D2, ..., DN)
        """

        # multiply by -1 as we want to maximize the psnr
        return -1 * _psnr_loss_fn(source, target, self.max_val)


class FlowPhotoMetricLoss(nn.Module):
    def __init__(
        self,
        ssim_weight: float = 0.85,
        ssim_window_size: int = 11,
        ssim_max_val: float = 1.0,
        ssim_sigma: float = 1.5,
        ssim_eps: float = 1e-12,
        ssim_use_padding: bool = True,
        max_displacement_ratio: float = 0.15,
    ) -> None:
        super().__init__()

        self._ssim_loss = SSIM(
            kernel_size=ssim_window_size,
            max_val=ssim_max_val,
            sigma=ssim_sigma,
            eps=ssim_eps,
            use_padding=ssim_use_padding,
        )

        self._L1_weight = 1 - ssim_weight
        self._SSIM_weight = ssim_weight
        self._max_displacement_ratio = max_displacement_ratio

    def forward(
        self,
        source: Tensor,
        reference: Tensor,
        flow_pred: Tensor,
        valid_mask: Optional[Tensor] = None,
    ):
        """
        Args:
            source: tensor of shape (B, C, H, W)
            reference: tensor of shape (B, C, H, W)
            flow_pred: tensor of shape (B, 2, H, W)
            valid_mask: tensor of shape (B, H, W) or None

        Returns:
            photometric loss of shape

        """
        torch._assert(
            source.ndim == 4,
            "FlowPhotoMetricLoss: source must have 4 dimensions, but got {}".format(source.ndim),
        )
        torch._assert(
            reference.ndim == source.ndim,
            "FlowPhotoMetricLoss: source and other must have the same number of dimensions, but got {} and {}".format(
                source.ndim, reference.ndim
            ),
        )
        torch._assert(
            flow_pred.shape[1] == 2,
            "FlowPhotoMetricLoss: flow_pred must have 2 channels, but got {}".format(flow_pred.shape[1]),
        )
        torch._assert(
            flow_pred.ndim == 4,
            "FlowPhotoMetricLoss: flow_pred must have 4 dimensions, but got {}".format(flow_pred.ndim),
        )

        B, C, H, W = source.shape
        flow_channels = flow_pred.shape[1]

        max_displacements = []
        for dim in range(flow_channels):
            shape_index = -1 - dim
            max_displacements.append(int(self._max_displacement_ratio * source.shape[shape_index]))

        # mask out all pixels that have larger flow than the max flow allowed
        max_flow_mask = torch.logical_and(
            *[flow_pred[:, dim, :, :] < max_displacements[dim] for dim in range(flow_channels)]
        )

        if valid_mask is not None:
            valid_mask = torch.logical_and(valid_mask, max_flow_mask).unsqueeze(1)
        else:
            valid_mask = max_flow_mask.unsqueeze(1)

        grid = make_coords_grid(B, H, W, device=str(source.device))
        resampled_grids = grid - flow_pred
        resampled_grids = resampled_grids.permute(0, 2, 3, 1)
        resampled_source = grid_sample(reference, resampled_grids, mode="bilinear")

        # compute SSIM loss
        ssim_loss = self._ssim_loss(resampled_source * valid_mask, source * valid_mask)
        l1_loss = (resampled_source * valid_mask - source * valid_mask).abs().mean(axis=(-3, -2, -1))
        loss = self._L1_weight * l1_loss + self._SSIM_weight * ssim_loss

        return loss.mean()
