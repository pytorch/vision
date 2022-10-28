# TODO: Consturct a Mixup transform
import random

from torchvision.prototype import features
from torchvision.prototype.transforms import AutoAugmentPolicy, functional as F, InterpolationMode, Transform


class _Mixup:
    def __init__(
        self,
        *,
        alpha: float = 0.9,
        beta: float = 0.1,
    ) -> None:
        super().__init__()

    # Fetch a random image from the dataset
    random_index = random.randint(0, )