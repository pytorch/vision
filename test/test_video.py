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
    # Last three test segfault on video reader (see issues)
    "R6llTwEh07w.mp4": GroundTruth(
        duration=10.0,
        video_fps=30.0,
        audio_sample_rate=44100,
        # PyAv miss one audio frame at the beginning (pts=0)
        check_aframes=False,
        check_aframe_pts=False,
    ),
    "SOX5yA1l24A.mp4": GroundTruth(
        duration=11.0,
        video_fps=29.97,
        audio_sample_rate=48000,
        # PyAv miss one audio frame at the beginning (pts=0)
        check_aframes=False,
        check_aframe_pts=False,
    ),
    "WUzgd7C1pWA.mp4": GroundTruth(
        duration=11.0,
        video_fps=29.97,
        audio_sample_rate=48000,
        # PyAv miss one audio frame at the beginning (pts=0)
        check_aframes=False,
        check_aframe_pts=False,
    ),
}


def _template_read_video(video_object, s=0, e=None):

    if e is None:
        e = float("inf")
    if e < s:
        raise ValueError(
            "end time should be larger than start time, got "
            "start time={} and end time={}".format(s, e)
        )

    video_object.set_current_stream("video")
    video_object.seek(s)
    video_frames = torch.empty(0)
    frames = []
    t, pts = video_object.next()
    while t.numel() > 0 and (pts >= s and pts <= e):
        frames.append(t)
        t, pts = video_object.next()
    if len(frames) > 0:
        video_frames = torch.stack(frames, 0)

    video_object.set_current_stream("audio")
    video_object.seek(s)
    audio_frames = torch.empty(0)
    frames = []
    t, pts = video_object.next()
    while t.numel() > 0 and (pts > s and pts <= e):
        frames.append(t)
        t, pts = video_object.next()
    if len(frames) > 0:
        audio_frames = torch.stack(frames, 0)

    return video_frames, audio_frames, video_object.get_metadata()


@unittest.skipIf(_HAS_VIDEO_OPT is False, "Didn't compile with ffmpeg")
class TestVideo(unittest.TestCase):
    def test_read_video_tensor(self):
        """
        Check if reading the video using the `next` based API yields the
        same sized and equal tensors as video_reader.
        """
        torchvision.set_video_backend("video_reader")
        for test_video, config in test_videos.items():
            full_path = os.path.join(VIDEO_DIR, test_video)
            # pass 1: decode all frames using existing TV decoder
            tv_result, _, _ = torchvision.io.read_video(full_path, pts_unit="sec")
            tv_result = tv_result.permute(0, 3, 1, 2)
            # pass 2: decode all frames using new api
            reader = torch.classes.torchvision.Video(full_path, "video")
            frames = []
            t, _ = reader.next()
            while t.numel() > 0:
                frames.append(t)
                t, _ = reader.next()
            new_api = torch.stack(frames, 0)
            self.assertEqual(tv_result.size(), new_api.size())
            self.assertEqual(torch.equal(tv_result, new_api), True)

    def test_pts(self):
        """Check if the frames have the same timestamps
        """
        torchvision.set_video_backend("video_reader")
        for test_video, config in test_videos.items():
            full_path = os.path.join(VIDEO_DIR, test_video)

            tv_timestamps, _ = torchvision.io.read_video_timestamps(full_path, pts_unit='sec')
            # pass 2: decode all frames using new api
            reader = torch.classes.torchvision.Video(full_path, "video")
            pts = []
            t, p = reader.next()
            while t.numel() > 0:
                pts.append(p)
                t, p = reader.next()

            tv_timestamps = [float(p) for p in tv_timestamps]
            napi_pts = [float(p) for p in pts]
            for i in range(len(napi_pts)):
                self.assertAlmostEqual(napi_pts[i], tv_timestamps[i], delta=0.001)

    def test_metadata(self):
        torchvision.set_video_backend("video_reader")
        for test_video, config in test_videos.items():
            full_path = os.path.join(VIDEO_DIR, test_video)
            reader = torch.classes.torchvision.Video(full_path, "video")
            reader_md = reader.get_metadata()
            self.assertAlmostEqual(config.video_fps, reader_md["video"]["fps"][0], delta=0.0001)
            self.assertAlmostEqual(config.duration, reader_md["video"]["duration"][0], delta=0.5)

    def test_video_reading_fn(self):
        torchvision.set_video_backend("video_reader")
        for test_video, config in test_videos.items():
            full_path = os.path.join(VIDEO_DIR, test_video)

            reader = torch.classes.torchvision.Video(full_path, "video")
            video, audio, metadata = _template_read_video(reader)
            tv_video, tv_audio, info = torchvision.io.read_video(full_path, pts_unit="sec")

            self.assertEqual(torch.equal(tv_video.permute(0, 3, 1, 2), video), True)
            self.assertEqual(torch.equal(tv_audio, audio), True)

    def test_partial_video_reading_fn(self):
        import random
        print("Test video reader")
        torchvision.set_video_backend("video_reader")
        for test_video, config in test_videos.items():
            full_path = os.path.join(VIDEO_DIR, test_video)

            # select two random points between 0 and duration
            r = []
            r.append(random.uniform(0, config.duration))
            r.append(random.uniform(0, config.duration))
            s = min(r)
            e = max(r)

            reader = torch.classes.torchvision.Video(full_path, "video")
            video, audio, metadata = _template_read_video(reader, s, e)
            tv_video, tv_audio, info = torchvision.io.read_video(full_path, start_pts=s, end_pts=e, pts_unit="sec")
            self.assertAlmostEqual(tv_video.size(0), video.size(0), delta=2.0)


if __name__ == '__main__':
    unittest.main()
