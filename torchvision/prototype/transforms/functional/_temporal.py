import torch

from torchvision.prototype import features


def uniform_temporal_subsample_video(video: torch.Tensor, num_samples: int, temporal_dim: int = -4) -> torch.Tensor:
    # Reference: https://github.com/facebookresearch/pytorchvideo/blob/a0a131e/pytorchvideo/transforms/functional.py#L19
    t_max = video.shape[temporal_dim] - 1
    indices = torch.linspace(0, t_max, num_samples, device=video.device).long()
    return torch.index_select(video, temporal_dim, indices)


def uniform_temporal_subsample(
    inpt: features.VideoTypeJIT, num_samples: int, temporal_dim: int = -4
) -> features.VideoTypeJIT:
    if isinstance(inpt, torch.Tensor) and (torch.jit.is_scripting() or not isinstance(inpt, features.Video)):
        return uniform_temporal_subsample_video(inpt, num_samples, temporal_dim=temporal_dim)
    else:  # isinstance(inpt, features.Video)
        if temporal_dim != -4 and inpt.ndim - 4 != temporal_dim:
            raise ValueError("Video inputs must have temporal_dim equivalent to -4")
        output = uniform_temporal_subsample_video(
            inpt.as_subclass(torch.Tensor), num_samples, temporal_dim=temporal_dim
        )
        return features.Video.wrap_like(inpt, output)
