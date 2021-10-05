import io
import unittest.mock
from typing import Dict, Any

import PIL.Image
from torchvision.io.video import read_video
from torchvision.transforms.functional import pil_to_tensor

__all__ = ["pil", "av"]


def pil(buffer: io.IOBase, *, mode: str = "RGB") -> Dict[str, Any]:
    return dict(image=pil_to_tensor(PIL.Image.open(buffer).convert(mode.upper())))


def av(buffer: io.IOBase, **read_video_kwargs: Any) -> Dict[str, Any]:
    with unittest.mock.patch("torchvision.io.video.os.path.exists", return_value=True):
        return dict(
            zip(
                ("video", "audio", "video_meta"),
                read_video(buffer, **read_video_kwargs),  # type: ignore[arg-type]
            )
        )
