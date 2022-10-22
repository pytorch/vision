from typing import Any, Dict

from torchvision.prototype import features
from torchvision.prototype.transforms import functional as F, Transform


class UniformTemporalSubsample(Transform):
    _transformed_types = (features.is_simple_tensor, features.Video)

    def __init__(self, num_samples: int, temporal_dim: int = -4):
        super().__init__()
        self.num_samples = num_samples
        self.temporal_dim = temporal_dim

    def _transform(self, inpt: features.VideoType, params: Dict[str, Any]) -> features.VideoType:
        return F.uniform_temporal_subsample(inpt, self.num_samples, temporal_dim=self.temporal_dim)
