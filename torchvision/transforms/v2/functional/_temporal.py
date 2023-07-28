import PIL.Image
import torch

from torchvision import datapoints

from torchvision.utils import _log_api_usage_once

from ._utils import _get_kernel, _register_explicit_noop, _register_kernel_internal, is_simple_tensor


@_register_explicit_noop(
    PIL.Image.Image, datapoints.Image, datapoints.BoundingBoxes, datapoints.Mask, warn_passthrough=True
)
def uniform_temporal_subsample(inpt: datapoints._VideoTypeJIT, num_samples: int) -> datapoints._VideoTypeJIT:
    if not torch.jit.is_scripting():
        _log_api_usage_once(uniform_temporal_subsample)

    if torch.jit.is_scripting() or is_simple_tensor(inpt):
        return uniform_temporal_subsample_video(inpt, num_samples)
    elif isinstance(inpt, datapoints.Datapoint):
        kernel = _get_kernel(uniform_temporal_subsample, type(inpt))
        return kernel(inpt, num_samples)
    else:
        raise TypeError(
            f"Input can either be a plain tensor or any TorchVision datapoint, but got {type(inpt)} instead."
        )


@_register_kernel_internal(uniform_temporal_subsample, datapoints.Video)
def uniform_temporal_subsample_video(video: torch.Tensor, num_samples: int) -> torch.Tensor:
    # Reference: https://github.com/facebookresearch/pytorchvideo/blob/a0a131e/pytorchvideo/transforms/functional.py#L19
    t_max = video.shape[-4] - 1
    indices = torch.linspace(0, t_max, num_samples, device=video.device).long()
    return torch.index_select(video, -4, indices)
