import math
import os

import pytest
import torch
import torchvision
from torchvision.io import _HAS_GPU_VIDEO_DECODER, VideoReader

try:
    import av
except ImportError:
    av = None

VIDEO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "videos")


@pytest.mark.skipif(_HAS_GPU_VIDEO_DECODER is False, reason="Didn't compile with support for gpu decoder")
class TestVideoGPUDecoder:
    @pytest.mark.skipif(av is None, reason="PyAV unavailable")
    @pytest.mark.parametrize(
        "video_file",
        [
            "RATRACE_wave_f_nm_np1_fr_goo_37.avi",
            "TrumanShow_wave_f_nm_np1_fr_med_26.avi",
            "v_SoccerJuggling_g23_c01.avi",
            "v_SoccerJuggling_g24_c01.avi",
            "R6llTwEh07w.mp4",
            "SOX5yA1l24A.mp4",
            "WUzgd7C1pWA.mp4",
        ],
    )
    def test_frame_reading(self, video_file):
        torchvision.set_video_backend("cuda")
        full_path = os.path.join(VIDEO_DIR, video_file)
        decoder = VideoReader(full_path)
        with av.open(full_path) as container:
            for av_frame in container.decode(container.streams.video[0]):
                av_frames = torch.tensor(av_frame.to_rgb(src_colorspace="ITU709").to_ndarray())
                vision_frames = next(decoder)["data"]
                mean_delta = torch.mean(torch.abs(av_frames.float() - vision_frames.cpu().float()))
                assert mean_delta < 0.75

    @pytest.mark.skipif(av is None, reason="PyAV unavailable")
    @pytest.mark.parametrize("keyframes", [True, False])
    @pytest.mark.parametrize(
        "full_path, duration",
        [
            (os.path.join(VIDEO_DIR, x), y)
            for x, y in [
                ("v_SoccerJuggling_g23_c01.avi", 8.0),
                ("v_SoccerJuggling_g24_c01.avi", 8.0),
                ("R6llTwEh07w.mp4", 10.0),
                ("SOX5yA1l24A.mp4", 11.0),
                ("WUzgd7C1pWA.mp4", 11.0),
            ]
        ],
    )
    def test_seek_reading(self, keyframes, full_path, duration):
        torchvision.set_video_backend("cuda")
        decoder = VideoReader(full_path)
        time = duration / 2
        decoder.seek(time, keyframes_only=keyframes)
        with av.open(full_path) as container:
            container.seek(int(time * 1000000), any_frame=not keyframes, backward=False)
            for av_frame in container.decode(container.streams.video[0]):
                av_frames = torch.tensor(av_frame.to_rgb(src_colorspace="ITU709").to_ndarray())
                vision_frames = next(decoder)["data"]
                mean_delta = torch.mean(torch.abs(av_frames.float() - vision_frames.cpu().float()))
                assert mean_delta < 0.75

    @pytest.mark.skipif(av is None, reason="PyAV unavailable")
    @pytest.mark.parametrize(
        "video_file",
        [
            "RATRACE_wave_f_nm_np1_fr_goo_37.avi",
            "TrumanShow_wave_f_nm_np1_fr_med_26.avi",
            "v_SoccerJuggling_g23_c01.avi",
            "v_SoccerJuggling_g24_c01.avi",
            "R6llTwEh07w.mp4",
            "SOX5yA1l24A.mp4",
            "WUzgd7C1pWA.mp4",
        ],
    )
    def test_metadata(self, video_file):
        torchvision.set_video_backend("cuda")
        full_path = os.path.join(VIDEO_DIR, video_file)
        decoder = VideoReader(full_path)
        video_metadata = decoder.get_metadata()["video"]
        with av.open(full_path) as container:
            video = container.streams.video[0]
            av_duration = float(video.duration * video.time_base)
            assert math.isclose(video_metadata["duration"], av_duration, rel_tol=1e-2)
            assert math.isclose(video_metadata["fps"], video.base_rate, rel_tol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
