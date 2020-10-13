import os
import collections
import contextlib
import tempfile
import unittest
import random

import itertools


import numpy as np

import torch
import torchvision
from torchvision.io import _HAS_VIDEO_OPT, VideoReader

try:
    import av

    # Do a version test too
    torchvision.io.video._check_av_available()
except ImportError:
    av = None


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

DecoderResult = collections.namedtuple(
    "DecoderResult", "vframes vframe_pts vtimebase aframes aframe_pts atimebase"
)


def _read_from_stream(
    container, start_pts, end_pts, stream, stream_name, buffer_size=4
):
    """
    Args:
        container: pyav container
        start_pts/end_pts: the starting/ending Presentation TimeStamp where
            frames are read
        stream: pyav stream
        stream_name: a dictionary of streams. For example, {"video": 0} means
            video stream at stream index 0
        buffer_size: pts of frames decoded by PyAv is not guaranteed to be in
            ascending order. We need to decode more frames even when we meet end
            pts
    """
    # seeking in the stream is imprecise. Thus, seek to an ealier PTS by a margin
    margin = 1
    seek_offset = max(start_pts - margin, 0)

    container.seek(seek_offset, any_frame=False, backward=True, stream=stream)
    frames = {}
    buffer_count = 0
    for frame in container.decode(**stream_name):
        if frame.pts < start_pts:
            continue
        if frame.pts <= end_pts:
            frames[frame.pts] = frame
        else:
            buffer_count += 1
            if buffer_count >= buffer_size:
                break
    result = [frames[pts] for pts in sorted(frames)]

    return result


def _fraction_to_tensor(fraction):
    ret = torch.zeros([2], dtype=torch.int32)
    ret[0] = fraction.numerator
    ret[1] = fraction.denominator
    return ret


def _decode_frames_by_av_module(
    full_path,
    video_start_pts=0,
    video_end_pts=None,
    audio_start_pts=0,
    audio_end_pts=None,
):
    """
    Use PyAv to decode video frames. This provides a reference for our decoder
    to compare the decoding results.
    Input arguments:
        full_path: video file path
        video_start_pts/video_end_pts: the starting/ending Presentation TimeStamp where
            frames are read
    """
    if video_end_pts is None:
        video_end_pts = float("inf")
    if audio_end_pts is None:
        audio_end_pts = float("inf")
    container = av.open(full_path)

    video_frames = []
    vtimebase = torch.zeros([0], dtype=torch.int32)
    if container.streams.video:
        video_frames = _read_from_stream(
            container,
            video_start_pts,
            video_end_pts,
            container.streams.video[0],
            {"video": 0},
        )
        # container.streams.video[0].average_rate is not a reliable estimator of
        # frame rate. It can be wrong for certain codec, such as VP80
        # So we do not return video fps here
        vtimebase = _fraction_to_tensor(container.streams.video[0].time_base)

    audio_frames = []
    atimebase = torch.zeros([0], dtype=torch.int32)
    if container.streams.audio:
        audio_frames = _read_from_stream(
            container,
            audio_start_pts,
            audio_end_pts,
            container.streams.audio[0],
            {"audio": 0},
        )
        atimebase = _fraction_to_tensor(container.streams.audio[0].time_base)

    container.close()
    vframes = [frame.to_rgb().to_ndarray() for frame in video_frames]
    vframes = torch.as_tensor(np.stack(vframes))

    vframe_pts = torch.tensor([frame.pts for frame in video_frames], dtype=torch.int64)

    aframes = [frame.to_ndarray() for frame in audio_frames]
    if aframes:
        aframes = np.transpose(np.concatenate(aframes, axis=1))
        aframes = torch.as_tensor(aframes)
    else:
        aframes = torch.empty((1, 0), dtype=torch.float32)

    aframe_pts = torch.tensor(
        [audio_frame.pts for audio_frame in audio_frames], dtype=torch.int64
    )

    return DecoderResult(
        vframes=vframes.permute(0, 3, 1, 2),
        vframe_pts=vframe_pts,
        vtimebase=vtimebase,
        aframes=aframes,
        aframe_pts=aframe_pts,
        atimebase=atimebase,
    )


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
    video_pts = []
    for frame in itertools.takewhile(lambda x: x['pts'] <= e, video_object):
        if frame['pts'] < s:
            continue
        frames.append(frame['data'])
        video_pts.append(frame['pts'])
    if len(frames) > 0:
        video_frames = torch.stack(frames, 0)

    video_object.set_current_stream("audio")
    video_object.seek(s)
    audio_frames = torch.empty(0)
    frames = []
    audio_pts = []
    for frame in itertools.takewhile(lambda x: x['pts'] <= e, video_object):
        if frame['pts'] < s:
            continue
        frames.append(frame['data'])
        audio_pts.append(frame['pts'])
    if len(frames) > 0:
        audio_frames = torch.stack(frames, 0)

    return DecoderResult(
        vframes=video_frames,
        vframe_pts=video_pts,
        vtimebase=None,
        aframes=audio_frames,
        aframe_pts=audio_pts,
        atimebase=None,
    )
    return video_frames, audio_frames, video_object.get_metadata()


@unittest.skipIf(_HAS_VIDEO_OPT is False, "Didn't compile with ffmpeg")
class TestVideo(unittest.TestCase):
    @unittest.skipIf(av is None, "PyAV unavailable")
    def test_read_video_tensor(self):
        """
        Check if reading the video using the `next` based API yields the
        same sized tensors as the pyav alternative.
        """
        torchvision.set_video_backend("pyav")
        for test_video, config in test_videos.items():
            full_path = os.path.join(VIDEO_DIR, test_video)
            # pass 1: decode all frames using existing TV decoder
            tv_result, _, _ = torchvision.io.read_video(full_path, pts_unit="sec")
            tv_result = tv_result.permute(0, 3, 1, 2)
            # pass 2: decode all frames using new api
            reader = VideoReader(full_path, "video")
            frames = []
            for frame in reader:
                frames.append(frame['data'])
            new_api = torch.stack(frames, 0)
            self.assertEqual(tv_result.size(), new_api.size())

    # def test_partial_video_reading_fn(self):
    #     torchvision.set_video_backend("video_reader")
    #     for test_video, config in test_videos.items():
    #         full_path = os.path.join(VIDEO_DIR, test_video)

    #         # select two random points between 0 and duration
    #         r = []
    #         r.append(random.uniform(0, config.duration))
    #         r.append(random.uniform(0, config.duration))
    #         s = min(r)
    #         e = max(r)

    #         reader = VideoReader(full_path, "video")
    #         results = _template_read_video(reader, s, e)
    #         tv_video, tv_audio, info = torchvision.io.read_video(
    #             full_path, start_pts=s, end_pts=e, pts_unit="sec"
    #         )
    #         self.assertAlmostEqual(tv_video.size(0), results.vframes.size(0), delta=2.0)

    # def test_pts(self):
    #     """
    #     Check if every frame read from
    #     """
    #     torchvision.set_video_backend("video_reader")
    #     for test_video, config in test_videos.items():
    #         full_path = os.path.join(VIDEO_DIR, test_video)

    #         tv_timestamps, _ = torchvision.io.read_video_timestamps(
    #             full_path, pts_unit="sec"
    #         )
    #         # pass 2: decode all frames using new api
    #         reader = VideoReader(full_path, "video")
    #         pts = []
    #         t, p = next(reader)
    #         while t.numel() > 0:  # THIS NEEDS TO BE FIXED
    #             pts.append(p)
    #             t, p = next(reader)

    #         tv_timestamps = [float(p) for p in tv_timestamps]
    #         napi_pts = [float(p) for p in pts]
    #         for i in range(len(napi_pts)):
    #             self.assertAlmostEqual(napi_pts[i], tv_timestamps[i], delta=0.001)
    #     # check if pts of video frames are sorted in ascending order
    #     for i in range(len(napi_pts) - 1):
    #         self.assertEqual(napi_pts[i] < napi_pts[i + 1], True)

    @unittest.skipIf(av is None, "PyAV unavailable")
    def test_metadata(self):
        """
        Test that the metadata returned via pyav corresponds to the one returned
        by the new video decoder API
        """
        torchvision.set_video_backend("pyav")
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

    @unittest.skipIf(av is None, "PyAV unavailable")
    def test_video_reading_fn(self):
        """
        Test that the outputs of the pyav and ffmpeg outputs are mostly the same
        """
        for test_video, config in test_videos.items():
            full_path = os.path.join(VIDEO_DIR, test_video)

            ref_result = _decode_frames_by_av_module(full_path)

            reader = VideoReader(full_path, "video")
            newapi_result = _template_read_video(reader)

            # First we check if the frames are approximately the same
            # (note that every codec context has signature artefacts which
            # make a direct comparison not feasible)
            if newapi_result.vframes.numel() > 0 and ref_result.vframes.numel() > 0:
                mean_delta = torch.mean(
                    torch.abs(
                        newapi_result.vframes.float() - ref_result.vframes.float()
                    )
                )
            self.assertAlmostEqual(mean_delta, 0, delta=8.0)

            # Just a sanity check: are the two of the correct size?
            self.assertEqual(newapi_result.vframes.size(), ref_result.vframes.size())

            # Lastly, we compare the resulting audio streams
            if (
                config.check_aframes
                and newapi_result.aframes.numel() > 0
                and ref_result.aframes.numel() > 0
            ):
                """Audio stream is available and audio frame is required to return
                from decoder"""
                is_same = torch.all(
                    torch.eq(newapi_result.aframes, ref_result.aframes)
                ).item()
                self.assertEqual(is_same, True)


if __name__ == "__main__":
    unittest.main()
