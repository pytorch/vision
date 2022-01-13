from torchvision.transforms import functional as _F

from .utils import _from_legacy_kernel

normalize_image = _from_legacy_kernel(_F.normalize)

erase_image = _from_legacy_kernel(_F.erase)
