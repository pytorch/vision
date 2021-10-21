import io
import unittest.mock
from typing import Dict, Any

import av
import PIL.Image
import torch
from torchvision.transforms.functional import pil_to_tensor

__all__ = ["raw", "pil"]


def raw(buffer: io.IOBase) -> torch.Tensor:
    raise RuntimeError("This is just a sentinel and should never be called.")


def pil(buffer: io.IOBase, mode: str = "RGB") -> torch.Tensor:
    return pil_to_tensor(PIL.Image.open(buffer).convert(mode.upper()))


def av_kf(buffer: io.IOBase, **read_video_kwargs: Any) -> Dict[str, Any]:
    with unittest.mock.patch("torchvision.io.video.os.path.exists", return_value=True):
        keyframes, pts = [], []
        with av.open(buffer) as container:
            stream = container.streams.video[0]
            stream.codec_context.skip_frame = 'NONKEY'
            for frame in container.decode(stream):
                keyframes.append(frame.to_image())
                # TODO: convert to seconds
                pts.append(frame.pts)

        return {"keyframes": keyframes, "pts": pts}
