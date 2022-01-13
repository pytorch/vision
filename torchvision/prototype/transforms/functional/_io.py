import unittest.mock
from typing import Dict, Any, Tuple

import numpy as np
import PIL.Image
import torch
from torchvision.io.video import read_video
from torchvision.prototype.features import Image, EncodedVideo, EncodedImage, Feature
from torchvision.prototype.utils._internal import ReadOnlyTensorBuffer

from .utils import Dispatcher


def decode_image_with_pil(encoded_image: torch.Tensor) -> torch.Tensor:
    image = torch.as_tensor(np.array(PIL.Image.open(ReadOnlyTensorBuffer(encoded_image)), copy=True))
    if image.ndim == 2:
        image = image.unsqueeze(2)
    return image.permute(2, 0, 1)


@Dispatcher
def decode_image(input: EncodedImage) -> Image:
    """ADDME"""
    pass


@decode_image.implements(EncodedImage)
def _decode_image(input: EncodedImage) -> Image:
    return Image(decode_image_with_pil(input))


def decode_video_with_av(encoded_video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    with unittest.mock.patch("torchvision.io.video.os.path.exists", return_value=True):
        return read_video(ReadOnlyTensorBuffer(encoded_video))


@Dispatcher
def decode_video(input: EncodedVideo) -> Tuple[Feature, Feature, Dict[str, Any]]:
    """ADDME"""
    pass


@decode_video.implements(EncodedVideo)
def _decode_video(input: EncodedVideo) -> Tuple[Feature, Feature, Dict[str, Any]]:
    video, audio, video_meta = decode_video_with_av(input)
    return Feature(video), Feature(audio), video_meta
