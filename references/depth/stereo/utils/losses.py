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
    valid_flow_mask: Tensor,
    gamma: Tensor,
    max_flow: int = 256,
    exclude_large: bool = False,
    weights: Optional[Tensor] = None,
):
    """Loss function defined over sequence of flow predictions"""
    torch._assert(
        gamma > 1,
        "sequence_loss: `gamma` must be greater than 1, but got {}".format(gamma),
    )

    if exclude_large:
        # exclude invalid pixels and extremely large diplacements
        flow_norm = torch.sum(flow_gt**2, dim=1).sqrt()
        valid_flow_mask = valid_flow_mask & (flow_norm < max_flow)

    valid_flow_mask = valid_flow_mask[:, None, :, :]
    flow_preds = torch.stack(flow_preds)  # shape = (num_flow_updates, batch_size, 2, H, W)

    abs_diff = (flow_preds - flow_gt).abs()
    abs_diff = (abs_diff * valid_flow_mask).mean(axis=(1, 2, 3, 4))

    num_predictions = flow_preds.shape[0]

    # alocating on CPU and moving to device during run-time can force
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
        self.gamma = self.register_buffer("gamma", torch.tensor([gamma]))
        # cache the scale factor for the loss
        self.weights = None

    def forward(self, flow_preds: List[Tensor], flow_gt: Tensor, valid_flow_mask: Tensor) -> Tensor:
        """
        Args:
            flow_preds: list of flow predictions of shape (batch_size, C, H, W)
            flow_gt: ground truth flow of shape (batch_size, C, H, W)
            valid_flow_mask: mask of valid flow pixels of shape (batch_size, H, W)
        """
        loss, weights = _sequence_loss_fn(
            flow_preds, flow_gt, valid_flow_mask, self.gamma, self.max_flow, self.excluding_large, self.weights
        )
        self.weights = weights
        return loss

    def set_gamma(self, gamma: float) -> None:
        self.gamma.fill_(gamma)
        # reset the cached scale factor
        self.weights = None


def _ssim_loss_fn(
    source: Tensor,
    reference: Tensor,
    kernel: Tensor,
    eps: float = 1e-8,
    C1: float = 0.01**2,
    C2: float = 0.03**2,
    use_padding: bool = False,
) -> Tensor:

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
    kernel = kernel[None, None, ...].repeat(C, 1, 1, 1)
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

    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim = numerator / (denominator + eps)
    return ssim.mean(dim=(1, 2, 3))


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

        self.C1 = (0.01 * self.max_val) ** 2
        self.C2 = (0.03 * self.max_val) ** 2

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
            C1=self.C1,
            C2=self.C2,
            use_padding=self.use_padding,
            eps=self.eps,
        )


def _smothness_loss_fn(img_gx: Tensor, img_gy: Tensor, depth_gx: Tensor, depth_gy: Tensor):
    torch._assert(
        img_gx.ndim >= 3,
        "smothness_loss: `img_gx` must be at least 3-dimensional tensor of shape (..., C, H, W)",
    )

    torch._assert(
        img_gx.ndim == depth_gx.ndim,
        "smothness_loss: `img_gx` and `depth_gx` must have the same dimensionality, but got {} and {}".format(
            img_gx.ndim, depth_gx.ndim
        ),
    )

    for idx in range(img_gx.ndim):
        torch._assert(
            (img_gx.shape[idx] == depth_gx.shape[idx] or (img_gx.shape[idx] == 1 or depth_gx.shape[idx] == 1)),
            "smothness_loss: `img_gx` and `depth_gx` must have either the same shape or broadcastable shape, but got {} and {}".format(
                img_gx.shape, depth_gx.shape
            ),
        )

    # -3 is channel dimension
    weights_x = torch.exp(-torch.mean(torch.abs(depth_gx), axis=-3, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(depth_gy), axis=-3, keepdim=True))

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

    def _y_grad(x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, images: Tensor, depths: Tensor, weights: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            images: tensor of shape (D1, D2, ..., DN, C, H, W)
            depths: tensor of shape (D1, D2, ..., DN, 1, H, W)

        Returns:
            smoothness loss of shape (D1, D2, ..., DN)
        """
        img_gx = self._x_gradient(images)
        img_gy = self._y_grad(images)

        depth_gx = self._x_gradient(depths)
        depth_gy = self._y_grad(depths)

        return _smothness_loss_fn(img_gx, img_gy, depth_gx, depth_gy, weights)


def _flow_sequence_consistency_loss_fn(
    flow_preds: List[Tensor],
    gamma: float = 0.8,
    resize_factor: float = 0.25,
    rescale_factor: float = 0.25,
    rescale_mode: str = "bilinear",
    weights: Optional[Tensor] = None,
):
    """Loss function defined over sequence of flow predictions"""

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

    flow_loss = abs_diff * weights
    return flow_loss


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

    def forward(self, flow_preds: List[Tensor], weights: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            flow_preds: list of tensors of shape (batch_size, C, H, W)

        Returns:
            sequence consistency loss of shape (batch_size,)
        """
        return _flow_sequence_consistency_loss_fn(
            flow_preds,
            gamma=self.gamma,
            resize_factor=self.resize_factor,
            rescale_factor=self.rescale_factor,
            rescale_mode=self.rescale_mode,
            weights=weights,
        )


def phototmetric_loss(
    source, other, flow_preds, valid_flow_mask, gamma=0.8, weights=None, max_flow=256, ssim_weight=0.1, ssim_loss=None
):
    B, C, H, W = source.shape
    flow_preds = torch.stack(flow_preds)
    flow_norm = torch.sum(flow_preds**2, dim=2).sqrt()
    valid_flow_mask = valid_flow_mask & (flow_norm < max_flow)
    valid_flow_mask = valid_flow_mask[:, None, :, :]

    N = flow_preds.shape[0]
    source = source.unsqueeze(0).repeat(N, 1, 1, 1, 1)
    other = other.unsqueeze(0).repeat(N, 1, 1, 1, 1)

    grid = make_coords_grid(B, H, W, device=source.device).unsqueeze(0)
    resampled_grids = flow_preds + grid
    resampled_grids = resampled_grids.permute(0, 1, 3, 4, 2)
    resampled_grids = resampled_grids.reshape(-1, H, W, 2)

    resampled_source = grid_sample(
        source.reshape(N * B, C, H, W),
        resampled_grids,
    )

    if ssim_loss is not None:
        ssim_score = ssim_loss(resampled_source.reshape(-1, C, H, W), other.reshape(-1, C, H, W))
        ssim_score = ssim_score.reshape(N, B, 1).mean(dim=(1, 2))
    else:
        ssim_score = torch.tensor([0.0], device=source.device)

    resampled_source = resampled_source.reshape(N, B, C, H, W)
    # ignore loss for out of bounds pixels
    resampled_mask = (resampled_source.sum(dim=2, keepdim=True) > 0).detach()
    valid_flow_mask = valid_flow_mask.permute(0, 2, 1, 3, 4)
    valid_flow_mask = torch.logical_or(valid_flow_mask, resampled_mask)
    abs_diff = (resampled_source - source).abs()
    abs_diff = (abs_diff * valid_flow_mask).mean(axis=(1, 2, 3, 4))

    num_predictions = flow_preds.shape[0]
    if weights is None or len(weights) != num_predictions:
        weights = torch.tensor([gamma], device=flow_preds.device, dtype=flow_preds.dtype) ** torch.arange(
            num_predictions - 1, -1, -1, device=flow_preds.device, dtype=flow_preds.dtype
        )

    # loss is combination of L1 and SSIM
    # use a minus for SSIM, since we want to maximize it
    loss = (abs_diff * weights).sum() * (1 - ssim_weight) + (1 - ssim_score).sum() * ssim_weight
    return loss, weights
