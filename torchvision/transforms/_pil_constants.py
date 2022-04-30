
import PIL
from PIL import Image

if tuple(int(part) for part in PIL.__version__.split(".")) >= (9, 1, 0):
    FLIP_LEFT_RIGHT = Image.Transpose.FLIP_LEFT_RIGHT
    FLIP_TOP_BOTTOM = Image.Transpose.FLIP_TOP_BOTTOM
    BILINEAR = Image.Resampling.BILINEAR
    NEAREST = Image.Resampling.NEAREST
    AFFINE = Image.Transform.AFFINE
    BICUBIC = Image.Resampling.BICUBIC
    PERSPECTIVE = Image.Transform.PERSPECTIVE
    LINEAR = Image.Resampling.BILINEAR
else:
    FLIP_LEFT_RIGHT = Image.FLIP_LEFT_RIGHT
    FLIP_TOP_BOTTOM = Image.FLIP_TOP_BOTTOM
    BILINEAR = Image.BILINEAR
    NEAREST = Image.NEAREST
    AFFINE = Image.AFFINE
    BICUBIC = Image.BICUBIC
    PERSPECTIVE = Image.PERSPECTIVE
    LINEAR = Image.LINEAR
