from typing import Any, Dict

from torchvision import datapoints
from torchvision.transforms.v2 import functional as F, Transform

from torchvision.transforms.v2.utils import is_simple_tensor


class UniformTemporalSubsample(Transform):
    _transformed_types = (is_simple_tensor, datapoints.Video)

    def __init__(self, num_samples: int):
        super().__init__()
        self.num_samples = num_samples

    def _transform(self, inpt: datapoints._VideoType, params: Dict[str, Any]) -> datapoints._VideoType:
        return F.uniform_temporal_subsample(inpt, self.num_samples)
