from torchvision.transforms import InterpolationMode, AutoAugmentPolicy  # usort: skip

from . import functional  # usort: skip

from ._transform import Transform  # usort: skip

from ._augment import RandomErasing, RandomMixup, RandomCutmix
from ._auto_augment import RandAugment, TrivialAugmentWide, AutoAugment, AugMix
from ._container import Compose, RandomApply, RandomChoice, RandomOrder
from ._geometry import HorizontalFlip, Resize, CenterCrop, RandomResizedCrop
from ._meta import ConvertBoundingBoxFormat, ConvertImageDtype, ConvertImageColorSpace
from ._misc import Identity, Normalize, ToDtype, Lambda
from ._presets import CocoEval, ImageNetEval, VocEval, Kinect400Eval, RaftEval
from ._type_conversion import DecodeImage, LabelToOneHot

# What are the migration plans for Classes without new API equivalents? There are two categories:
# 1. Deprecated methods which have equivalents on the new API (_legacy.py?):
# - Grayscale, RandomGrayscale: use ConvertImageColorSpace
# 2. Those without equivalents on the new API:
# - Pad, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomPerspective, FiveCrop, TenCrop, ColorJitter,
# RandomRotation, RandomAffine, GaussianBlur, RandomInvert, RandomPosterize, RandomSolarize, RandomAdjustSharpness,
# RandomAutocontrast, RandomEqualize, LinearTransformation (must be added)
# - PILToTensor, ToPILImage (_legacy.py?)
# - ToTensor (deprecate vfdev-5?)
# We need a plan for both categories implemented on the new API.
