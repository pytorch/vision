import os

import pytest
import torch
from torchvision.io import _HAS_VIDEO_DECODER, VideoReader

try:
    import av
except ImportError:
    av = None

VIDEO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "videos")

test_videos = [
    "RATRACE_wave_f_nm_np1_fr_goo_37.avi",
    "TrumanShow_wave_f_nm_np1_fr_med_26.avi",
    "v_SoccerJuggling_g23_c01.avi",
    "v_SoccerJuggling_g24_c01.avi",
    "R6llTwEh07w.mp4",
    "SOX5yA1l24A.mp4",
    "WUzgd7C1pWA.mp4",
]


def _yuv420_to_444(mat):
    # logic taken from
    # https://en.wikipedia.org/wiki/YUV#Y%E2%80%B2UV420p_(and_Y%E2%80%B2V12_or_YV12)_to_RGB888_conversion
    width = mat.shape[-1]
    height = mat.shape[0] * 2 // 3
    luma = mat[:height]
    uv = mat[height:].reshape(2, height // 2, width // 2)
    uv2 = torch.nn.functional.interpolate(uv[None], scale_factor=2, mode="nearest")[0]
    yuv2 = torch.cat([luma[None], uv2]).permute(1, 2, 0)
    return yuv2


def _yuv420_to_rgb(mat):
    # ITU-R BT.709
    # taken from https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.709_conversion
    m = torch.tensor(
        [[1.0000, 0.0000, 1.5748], [1.0000, -0.1873, -0.4681], [1.0000, 1.8556, 0.0000]], device=mat.device
    )
    m = m * torch.tensor([255 / 219, 255 / 224, 255 / 224], device=mat.device)
    m = m.T
    offset = torch.tensor([16.0, 128.0, 128.0], device=mat.device)
    yuv2 = _yuv420_to_444(mat)
    res = (yuv2 - offset) @ m
    res.clamp_(min=0, max=255)
    return res.round().to(torch.uint8)


@pytest.mark.skipif(_HAS_VIDEO_DECODER is False, reason="Didn't compile with support for gpu decoder")
class TestVideoGPUDecoder:
    @pytest.mark.skipif(av is None, reason="PyAV unavailable")
    def test_frame_reading(self):
        for test_video in test_videos:
            full_path = os.path.join(VIDEO_DIR, test_video)
            decoder = VideoReader(full_path, device="cuda:0")
            with av.open(full_path) as container:
                for av_frame in container.decode(container.streams.video[0]):
                    av_frames_yuv = torch.tensor(av_frame.to_ndarray())
                    vision_frames = next(decoder)["data"]
                    av_frames = _yuv420_to_rgb(av_frames_yuv)
                    mean_delta = torch.mean(torch.abs(av_frames.float() - vision_frames.cpu().float()))
                    assert mean_delta < 0.7


if __name__ == "__main__":
    pytest.main([__file__])
