from typing import Any, Dict

from torchvision import datapoints
from torchvision.transforms.v2 import functional as F, Transform

from torchvision.transforms.v2.utils import is_simple_tensor


class UniformTemporalSubsample(Transform):
    """[BETA] Uniformly subsample ``num_samples`` indices from the temporal dimension of the video.

    .. v2betastatus:: UniformTemporalSubsample transform

    Videos are expected to be of shape ``[..., T, C, H, W]`` where ``T`` denotes the temporal dimension.

    When ``num_samples`` is larger than the size of temporal dimension of the video, it
    will sample frames based on nearest neighbor interpolation.

    Args:
        num_samples (int): The number of equispaced samples to be selected
    """

    _transformed_types = (is_simple_tensor, datapoints.Video)

    def __init__(self, num_samples: int):
        super().__init__()
        self.num_samples = num_samples

    def _transform(self, inpt: datapoints._VideoType, params: Dict[str, Any]) -> datapoints._VideoType:
        return F.uniform_temporal_subsample(inpt, self.num_samples)
