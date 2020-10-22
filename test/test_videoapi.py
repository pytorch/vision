import collections
import os
import unittest


import torchvision
from torchvision.io import _HAS_VIDEO_OPT, VideoReader

try:
    import av

    # Do a version test too
    print("Success")
    torchvision.io.video._check_av_available()
except ImportError:
    av = None


VIDEO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "videos")

CheckerConfig = ["duration", "video_fps", "audio_sample_rate"]
GroundTruth = collections.namedtuple("GroundTruth", " ".join(CheckerConfig))

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


@unittest.skipIf(_HAS_VIDEO_OPT is False, "Didn't compile with ffmpeg")
class TestVideoApi(unittest.TestCase):
    def test_predefined_metadata(self):
        """
        Test that the source metadata corresponds to the one returned
        by the new video decoder API.
        """
        for test_video, config in test_videos.items():
            full_path = os.path.join(VIDEO_DIR, test_video)
            reader = VideoReader(full_path, "video")
            reader_md = reader.get_metadata()
            self.assertAlmostEqual(
                config.video_fps, reader_md["video"]["fps"][0], delta=0.0001
            )
            self.assertAlmostEqual(
                config.duration, reader_md["video"]["duration"][0], delta=0.5
            )

    def test_frame_reading(self):
        for test_video, config in test_videos.items():
            full_path = os.path.join(VIDEO_DIR, test_video)

            av_reader = av.open(full_path)
            video_reader = VideoReader(full_path, "video")

            if av_reader.streams.video:
                for av_frame in av_reader.decode(av_reader.streams.video[0]):
                    vr_frame = next(video_reader)
                    print(av_frame.pts, vr_frame["pts"])
                # ,
                # {"video": 0}


if __name__ == "__main__":
    unittest.main()
