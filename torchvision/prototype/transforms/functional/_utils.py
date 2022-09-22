import torch


# The types defined in this file should be mirroring the ones in `.._utils.py`
# Unfortunately due to torch.jit.script limitations we must use fake types
# Keeping track of the actual types is useful in-case this limitation is lifted

# Real type: Union[torch.Tensor, PIL.Image.Image, features._Feature]
DType = torch.Tensor

# Real type: Union[torch.Tensor, PIL.Image.Image, features.Image]
ImageType = torch.Tensor

# Real type: Union[torch.Tensor, PIL.Image.Image]
LegacyImageType = torch.Tensor

# Real type: Union[torch.Tensor, features.Image]
TensorImageType = torch.Tensor
