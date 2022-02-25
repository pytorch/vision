import math
import os

import pytest
import torch
from torchvision.io import _HAS_VIDEO_DECODER, _HAS_VIDEO_OPT, VideoReader
from torchvision.prototype.features import EncodedData
from torchvision.prototype.utils._internal import ReadOnlyTensorBuffer
from torchvision.prototype.datasets.utils._video import KeyframeDecoder, RandomFrameDecoder
try:
    import av
except ImportError:
    av = None

VIDEO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "videos")


@pytest.mark.skipif(av is None, reason="PyAV unavailable")
class TestVideoDatasetUtils:
    # TODO: atm we separate backends in order to allow for testing on different systems;
    # once we have things packaged we should add this as test parametrisation
    # (this also applies for GPU decoding as well)

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
    def test_random_decoder_av(self, video_file):
        """Read a sequence of random frames from a video
        Checks that files are valid video frames and no error is thrown during decoding.
        """
        video_file = os.path.join(VIDEO_DIR, video_file)
        video = ReadOnlyTensorBuffer(EncodedData.from_path(video_file))
        print(next(video))
        pass

    def test_random_decoder_cpu(self, video_file):
        """Read a sequence of random frames from a video using CPU backend
        Checks that files are valid video frames and no error is thrown during decoding,
        and compares them to `pyav` output.
        """
        pass

    def test_random_decoder_GPU(self, video_file):
        """Read a sequence of random frames from a video using GPU backend
        Checks that files are valid video frames and no error is thrown during decoding,
        and compares them to `pyav` output.
        """
        pass

    def test_keyframe_decoder_av(self, video_file):
        """Read all keyframes from a video;
        Compare the output to naive keyframe reading with `pyav`
        """
        pass

    def test_keyframe_decoder_cpu(self, video_file):
        """Read all keyframes from a video using CPU backend;
        ATM should raise a warning and default to `pyav`
        TODO: should we fail or default to a working backend
        """
        pass

    def test_keyframe_decoder_GPU(self, video_file):
        """Read all keyframes from a video using CPU backend;
        ATM should raise a warning and default to `pyav`
        TODO: should we fail or default to a working backend
        """
        pass

    def test_clip_decoder(self, video_file):
        """ATM very crude test:
        check only if fails, or if the clip sampling is correct,
        don't bother with the content just yet.
        """
        pass


if __name__ == "__main__":
    pytest.main([__file__])
