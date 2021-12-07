import torch
import torchvision.transforms.functional as F


torch.jit.script(F.rotate)
