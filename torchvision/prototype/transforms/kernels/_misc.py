from torchvision.transforms import functional as _F, functional_tensor as _FT


normalize_image = _F.normalize
gaussian_blur_image = _FT.gaussian_blur
