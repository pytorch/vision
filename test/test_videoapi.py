import collections
import os
import sys
import pytest
from pytest import approx
import urllib

import torch
import torchvision
from torchvision.io import _HAS_VIDEO_OPT, VideoReader
from torchvision.datasets.utils import download_url

from common_utils import PY39_SKIP

try:
    import av

    # Do a version test too
    torchvision.io.video._check_av_available()
except ImportError:
    av = None


VIDEO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "videos")

CheckerConfig = ["duration", "video_fps", "audio_sample_rate"]
GroundTruth = collections.namedtuple("GroundTruth", " ".join(CheckerConfig))


def fate(name, path="."):
    """Download and return a path to a sample from the FFmpeg test suite.
    See the `FFmpeg Automated Test Environment <https://www.ffmpeg.org/fate.html>`_
    """

    file_name = name.split("/")[1]
    download_url("http://fate.ffmpeg.org/fate-suite/" + name, path, file_name)
    return os.path.join(path, file_name)


test_videos = {
    "RATRACE_wave_f_nm_np1_fr_goo_37.avi": GroundTruth(
        duration=2.0, video_fps=30.0, audio_sample_rate=None
    ),
    "SchoolRulesHowTheyHelpUs_wave_f_nm_np1_ba_med_0.avi": GroundTruth(
        duration=2.0, video_fps=30.0, audio_sample_rate=None
    ),
    "TrumanShow_wave_f_nm_np1_fr_med_26.avi": GroundTruth(
        duration=2.0, video_fps=30.0, audio_sample_rate=None
    ),
    "v_SoccerJuggling_g23_c01.avi": GroundTruth(
        duration=8.0, video_fps=29.97, audio_sample_rate=None
    ),
    "v_SoccerJuggling_g24_c01.avi": GroundTruth(
        duration=8.0, video_fps=29.97, audio_sample_rate=None
    ),
    "R6llTwEh07w.mp4": GroundTruth(
        duration=10.0, video_fps=30.0, audio_sample_rate=44100
    ),
    "SOX5yA1l24A.mp4": GroundTruth(
        duration=11.0, video_fps=29.97, audio_sample_rate=48000
    ),
    "WUzgd7C1pWA.mp4": GroundTruth(
        duration=11.0, video_fps=29.97, audio_sample_rate=48000
    ),
}


# Rollback once #4402 is fixed
@pytest.mark.skipif(sys.platform == "darwin", reason=(
    'These tests segfault on MacOS; temporarily disabling.'
))
@pytest.mark.skipif(_HAS_VIDEO_OPT is False, reason="Didn't compile with ffmpeg")
@PY39_SKIP
class TestVideoApi:
    @pytest.mark.skipif(av is None, reason="PyAV unavailable")
    def test_frame_reading(self):
        for test_video, config in test_videos.items():
            full_path = os.path.join(VIDEO_DIR, test_video)

            av_reader = av.open(full_path)

            if av_reader.streams.video:
                video_reader = VideoReader(full_path, "video")
                for av_frame in av_reader.decode(av_reader.streams.video[0]):
                    vr_frame = next(video_reader)

                    assert float(av_frame.pts * av_frame.time_base) == approx(vr_frame["pts"], abs=0.1)

                    av_array = torch.tensor(av_frame.to_rgb().to_ndarray()).permute(
                        2, 0, 1
                    )
                    vr_array = vr_frame["data"]
                    mean_delta = torch.mean(
                        torch.abs(av_array.float() - vr_array.float())
                    )
                    # on average the difference is very small and caused
                    # by decoding (around 1%)
                    # TODO: asses empirically how to set this? atm it's 1%
                    # averaged over all frames
                    assert mean_delta.item() < 2.5

            av_reader = av.open(full_path)
            if av_reader.streams.audio:
                video_reader = VideoReader(full_path, "audio")
                for av_frame in av_reader.decode(av_reader.streams.audio[0]):
                    vr_frame = next(video_reader)
                    assert float(av_frame.pts * av_frame.time_base) == approx(vr_frame["pts"], abs=0.1)

                    av_array = torch.tensor(av_frame.to_ndarray()).permute(1, 0)
                    vr_array = vr_frame["data"]

                    max_delta = torch.max(
                        torch.abs(av_array.float() - vr_array.float())
                    )
                    # we assure that there is never more than 1% difference in signal
                    assert max_delta.item() < 0.001

    def test_metadata(self):
        """
        Test that the metadata returned via pyav corresponds to the one returned
        by the new video decoder API
        """
        for test_video, config in test_videos.items():
            full_path = os.path.join(VIDEO_DIR, test_video)
            reader = VideoReader(full_path, "video")
            reader_md = reader.get_metadata()
            assert config.video_fps == approx(reader_md["video"]["fps"][0], abs=0.0001)
            assert config.duration == approx(reader_md["video"]["duration"][0], abs=0.5)

    def test_seek_start(self):
        for test_video, config in test_videos.items():
            full_path = os.path.join(VIDEO_DIR, test_video)

            video_reader = VideoReader(full_path, "video")
            num_frames = 0
            for frame in video_reader:
                num_frames += 1

            # now seek the container to 0 and do it again
            # It's often that starting seek can be inprecise
            # this way and it doesn't start at 0
            video_reader.seek(0)
            start_num_frames = 0
            for frame in video_reader:
                start_num_frames += 1

            assert start_num_frames == num_frames

            # now seek the container to < 0 to check for unexpected behaviour
            video_reader.seek(-1)
            start_num_frames = 0
            for frame in video_reader:
                start_num_frames += 1

            assert start_num_frames == num_frames

    def test_accurateseek_middle(self):
        for test_video, config in test_videos.items():
            full_path = os.path.join(VIDEO_DIR, test_video)

            stream = "video"
            video_reader = VideoReader(full_path, stream)
            md = video_reader.get_metadata()
            duration = md[stream]["duration"][0]
            if duration is not None:

                num_frames = 0
                for frame in video_reader:
                    num_frames += 1

                video_reader.seek(duration / 2)
                middle_num_frames = 0
                for frame in video_reader:
                    middle_num_frames += 1

                assert middle_num_frames < num_frames
                assert middle_num_frames == approx(num_frames // 2, abs=1)

                video_reader.seek(duration / 2)
                frame = next(video_reader)
                lb = duration / 2 - 1 / md[stream]["fps"][0]
                ub = duration / 2 + 1 / md[stream]["fps"][0]
                assert (lb <= frame["pts"]) and (ub >= frame["pts"])

    def test_fate_suite(self):
        # TODO: remove the try-except statement once the connectivity issues are resolved
        try:
            video_path = fate("sub/MovText_capability_tester.mp4", VIDEO_DIR)
        except (urllib.error.URLError, ConnectionError) as error:
            pytest.skip(f"Skipping due to connectivity issues: {error}")
        vr = VideoReader(video_path)
        metadata = vr.get_metadata()

        assert metadata["subtitles"]["duration"] is not None
        os.remove(video_path)


if __name__ == '__main__':
    pytest.main([__file__])
