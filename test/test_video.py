import os
import collections
import contextlib
import tempfile
import unittest


import numpy as np

import torch
import torchvision
from torchvision.io import _HAS_VIDEO_OPT



VIDEO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "videos")

CheckerConfig = [
    "duration",
    "video_fps",
    "audio_sample_rate",
    # We find for some videos (e.g. HMDB51 videos), the decoded audio frames and pts are
    # slightly different between TorchVision decoder and PyAv decoder. So omit it during check
    "check_aframes",
    "check_aframe_pts",
]
GroundTruth = collections.namedtuple("GroundTruth", " ".join(CheckerConfig))

all_check_config = GroundTruth(
    duration=0,
    video_fps=0,
    audio_sample_rate=0,
    check_aframes=True,
    check_aframe_pts=True,
)

test_videos = {
    "RATRACE_wave_f_nm_np1_fr_goo_37.avi": GroundTruth(
        duration=2.0,
        video_fps=30.0,
        audio_sample_rate=None,
        check_aframes=True,
        check_aframe_pts=True,
    ),
    "SchoolRulesHowTheyHelpUs_wave_f_nm_np1_ba_med_0.avi": GroundTruth(
        duration=2.0,
        video_fps=30.0,
        audio_sample_rate=None,
        check_aframes=True,
        check_aframe_pts=True,
    ),
    "TrumanShow_wave_f_nm_np1_fr_med_26.avi": GroundTruth(
        duration=2.0,
        video_fps=30.0,
        audio_sample_rate=None,
        check_aframes=True,
        check_aframe_pts=True,
    ),
    "v_SoccerJuggling_g23_c01.avi": GroundTruth(
        duration=8.0,
        video_fps=29.97,
        audio_sample_rate=None,
        check_aframes=True,
        check_aframe_pts=True,
    ),
    "v_SoccerJuggling_g24_c01.avi": GroundTruth(
        duration=8.0,
        video_fps=29.97,
        audio_sample_rate=None,
        check_aframes=True,
        check_aframe_pts=True,
    ),
    ### Last three test segfault on video reader (see issues)
    # "R6llTwEh07w.mp4": GroundTruth(
    #     duration=10.0,
    #     video_fps=30.0,
    #     audio_sample_rate=44100,
    #     # PyAv miss one audio frame at the beginning (pts=0)
    #     check_aframes=False,
    #     check_aframe_pts=False,
    # ),
    # "SOX5yA1l24A.mp4": GroundTruth(
    #     duration=11.0,
    #     video_fps=29.97,
    #     audio_sample_rate=48000,
    #     # PyAv miss one audio frame at the beginning (pts=0)
    #     check_aframes=False,
    #     check_aframe_pts=False,
    # ),
    # "WUzgd7C1pWA.mp4": GroundTruth(
    #     duration=11.0,
    #     video_fps=29.97,
    #     audio_sample_rate=48000,
    #     # PyAv miss one audio frame at the beginning (pts=0)
    #     check_aframes=False,
    #     check_aframe_pts=False,
    # ),
}


@unittest.skipIf(_HAS_VIDEO_OPT is False, "Didn't compile with ffmpeg")
class TestVideo(unittest.TestCase):
    def test_read_video_tensor(self):
        """
        Check if reading the video using the `next` based API yields the
        same sized and equal tensors as video_reader.
        """
        print("test read")
        torchvision.set_video_backend("video_reader")
        for test_video, config in test_videos.items():
            full_path = os.path.join(VIDEO_DIR, test_video)
            # pass 1: decode all frames using existing TV decoder
            tv_result, _, _ = torchvision.io.read_video(full_path, pts_unit="sec")
            tv_result = tv_result.permute(0, 3, 1, 2)
            # pass 2: decode all frames using new api
            reader = torch.classes.torchvision.Video(full_path, "video", True)
            frames = []
            t, _ = reader.next("")
            while t.numel() > 0:
                frames.append(t)
                t, _ = reader.next("")
            new_api = torch.stack(frames, 0)
            self.assertEqual(tv_result.size(), new_api.size())
            self.assertEqual(torch.equal(tv_result, new_api), True)
    
    @unittest.skipIf(not _HAS_VIDEO_OPT, "video_reader backend is not chosen")
    def test_pts(self):
        """Check if the frames have the same timestamps
        """
        print("test timestamp")
        torchvision.set_video_backend("video_reader")
        for test_video, config in test_videos.items():
            full_path = os.path.join(VIDEO_DIR, test_video)

            tv_timestamps, _ =  torchvision.io.read_video_timestamps(full_path, pts_unit='sec')
            # pass 2: decode all frames using new api
            reader = torch.classes.torchvision.Video(full_path, "video", True)
            pts = []
            t, p = reader.next("")
            while t.numel() > 0:
                pts.append(p)
                t, p = reader.next("")
            
            tv_timestamps = [float(p) for p in tv_timestamps]
            napi_pts = [float(p) for p in pts]
            for i in range(len(napi_pts)):
                self.assertAlmostEqual(napi_pts[i], tv_timestamps[i], delta=0.001)

    @unittest.skipIf(not _HAS_VIDEO_OPT, "video_reader backend is not chosen")
    def test_metadata(self):
        print("test fps")
        torchvision.set_video_backend("video_reader")
        for test_video, config in test_videos.items():
            full_path = os.path.join(VIDEO_DIR, test_video)
            reader = torch.classes.torchvision.Video(full_path, "video", True)
            reader_md = reader.get_metadata()
            self.assertAlmostEqual(config.video_fps, reader_md["video"]["fps"][0], delta=0.0001)
            self.assertAlmostEqual(config.duration, reader_md["video"]["duration"][0], delta=0.5)
      
if __name__ == '__main__':
    unittest.main()