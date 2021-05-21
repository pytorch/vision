import collections
import math
import os
import time
import unittest
from fractions import Fraction

import numpy as np
import torch
import torchvision.io as io
from numpy.random import randint
from torchvision.io import _HAS_VIDEO_OPT
from common_utils import PY39_SKIP


try:
    import av

    # Do a version test too
    io.video._check_av_available()
except ImportError:
    av = None


from urllib.error import URLError


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

"""av_seek_frame is imprecise so seek to a timestamp earlier by a margin
The unit of margin is second"""
seek_frame_margin = 0.25


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


def _get_timebase_by_av_module(full_path):
    container = av.open(full_path)
    video_time_base = container.streams.video[0].time_base
    if container.streams.audio:
        audio_time_base = container.streams.audio[0].time_base
    else:
        audio_time_base = None
    return video_time_base, audio_time_base


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
        vframes=vframes,
        vframe_pts=vframe_pts,
        vtimebase=vtimebase,
        aframes=aframes,
        aframe_pts=aframe_pts,
        atimebase=atimebase,
    )


def _pts_convert(pts, timebase_from, timebase_to, round_func=math.floor):
    """convert pts between different time bases
    Args:
        pts: presentation timestamp, float
        timebase_from: original timebase. Fraction
        timebase_to: new timebase. Fraction
        round_func: rounding function.
    """
    new_pts = Fraction(pts, 1) * timebase_from / timebase_to
    return int(round_func(new_pts))


def _get_video_tensor(video_dir, video_file):
    """open a video file, and represent the video data by a PT tensor"""
    full_path = os.path.join(video_dir, video_file)

    assert os.path.exists(full_path), "File not found: %s" % full_path

    with open(full_path, "rb") as fp:
        video_tensor = torch.from_numpy(np.frombuffer(fp.read(), dtype=np.uint8))

    return full_path, video_tensor


@unittest.skipIf(av is None, "PyAV unavailable")
@unittest.skipIf(_HAS_VIDEO_OPT is False, "Didn't compile with ffmpeg")
class TestVideoReader(unittest.TestCase):
    def check_separate_decoding_result(self, tv_result, config):
        """check the decoding results from TorchVision decoder
        """
        vframes, vframe_pts, vtimebase, vfps, vduration, \
            aframes, aframe_pts, atimebase, asample_rate, aduration = (
                tv_result
            )

        video_duration = vduration.item() * Fraction(
            vtimebase[0].item(), vtimebase[1].item()
        )
        self.assertAlmostEqual(video_duration, config.duration, delta=0.5)

        self.assertAlmostEqual(vfps.item(), config.video_fps, delta=0.5)
        if asample_rate.numel() > 0:
            self.assertEqual(asample_rate.item(), config.audio_sample_rate)
            audio_duration = aduration.item() * Fraction(
                atimebase[0].item(), atimebase[1].item()
            )
            self.assertAlmostEqual(audio_duration, config.duration, delta=0.5)

        # check if pts of video frames are sorted in ascending order
        for i in range(len(vframe_pts) - 1):
            self.assertEqual(vframe_pts[i] < vframe_pts[i + 1], True)

        if len(aframe_pts) > 1:
            # check if pts of audio frames are sorted in ascending order
            for i in range(len(aframe_pts) - 1):
                self.assertEqual(aframe_pts[i] < aframe_pts[i + 1], True)

    def check_probe_result(self, result, config):
        vtimebase, vfps, vduration, atimebase, asample_rate, aduration = result
        video_duration = vduration.item() * Fraction(
            vtimebase[0].item(), vtimebase[1].item()
        )
        self.assertAlmostEqual(video_duration, config.duration, delta=0.5)
        self.assertAlmostEqual(vfps.item(), config.video_fps, delta=0.5)
        if asample_rate.numel() > 0:
            self.assertEqual(asample_rate.item(), config.audio_sample_rate)
            audio_duration = aduration.item() * Fraction(
                atimebase[0].item(), atimebase[1].item()
            )
            self.assertAlmostEqual(audio_duration, config.duration, delta=0.5)

    def check_meta_result(self, result, config):
        self.assertAlmostEqual(result.video_duration, config.duration, delta=0.5)
        self.assertAlmostEqual(result.video_fps, config.video_fps, delta=0.5)
        if result.has_audio > 0:
            self.assertEqual(result.audio_sample_rate, config.audio_sample_rate)
            self.assertAlmostEqual(result.audio_duration, config.duration, delta=0.5)

    def compare_decoding_result(self, tv_result, ref_result, config=all_check_config):
        """
        Compare decoding results from two sources.
        Args:
            tv_result: decoding results from TorchVision decoder
            ref_result: reference decoding results which can be from either PyAv
                        decoder or TorchVision decoder with getPtsOnly = 1
            config: config of decoding results checker
        """
        vframes, vframe_pts, vtimebase, _vfps, _vduration, \
            aframes, aframe_pts, atimebase, _asample_rate, _aduration = (
                tv_result
            )
        if isinstance(ref_result, list):
            # the ref_result is from new video_reader decoder
            ref_result = DecoderResult(
                vframes=ref_result[0],
                vframe_pts=ref_result[1],
                vtimebase=ref_result[2],
                aframes=ref_result[5],
                aframe_pts=ref_result[6],
                atimebase=ref_result[7],
            )

        if vframes.numel() > 0 and ref_result.vframes.numel() > 0:
            mean_delta = torch.mean(
                torch.abs(vframes.float() - ref_result.vframes.float())
            )
            self.assertAlmostEqual(mean_delta, 0, delta=8.0)

        mean_delta = torch.mean(
            torch.abs(vframe_pts.float() - ref_result.vframe_pts.float())
        )
        self.assertAlmostEqual(mean_delta, 0, delta=1.0)

        is_same = torch.all(torch.eq(vtimebase, ref_result.vtimebase)).item()
        self.assertEqual(is_same, True)

        if (
            config.check_aframes
            and aframes.numel() > 0
            and ref_result.aframes.numel() > 0
        ):
            """Audio stream is available and audio frame is required to return
            from decoder"""
            is_same = torch.all(torch.eq(aframes, ref_result.aframes)).item()
            self.assertEqual(is_same, True)

        if (
            config.check_aframe_pts
            and aframe_pts.numel() > 0
            and ref_result.aframe_pts.numel() > 0
        ):
            """Audio stream is available"""
            is_same = torch.all(torch.eq(aframe_pts, ref_result.aframe_pts)).item()
            self.assertEqual(is_same, True)

            is_same = torch.all(torch.eq(atimebase, ref_result.atimebase)).item()
            self.assertEqual(is_same, True)

    @unittest.skip(
        "This stress test will iteratively decode the same set of videos."
        "It helps to detect memory leak but it takes lots of time to run."
        "By default, it is disabled"
    )
    def test_stress_test_read_video_from_file(self):
        num_iter = 10000
        # video related
        width, height, min_dimension, max_dimension = 0, 0, 0, 0
        video_start_pts, video_end_pts = 0, -1
        video_timebase_num, video_timebase_den = 0, 1
        # audio related
        samples, channels = 0, 0
        audio_start_pts, audio_end_pts = 0, -1
        audio_timebase_num, audio_timebase_den = 0, 1

        for _i in range(num_iter):
            for test_video, _config in test_videos.items():
                full_path = os.path.join(VIDEO_DIR, test_video)

                # pass 1: decode all frames using new decoder
                torch.ops.video_reader.read_video_from_file(
                    full_path,
                    seek_frame_margin,
                    0,  # getPtsOnly
                    1,  # readVideoStream
                    width,
                    height,
                    min_dimension,
                    max_dimension,
                    video_start_pts,
                    video_end_pts,
                    video_timebase_num,
                    video_timebase_den,
                    1,  # readAudioStream
                    samples,
                    channels,
                    audio_start_pts,
                    audio_end_pts,
                    audio_timebase_num,
                    audio_timebase_den,
                )

    @PY39_SKIP
    def test_read_video_from_file(self):
        """
        Test the case when decoder starts with a video file to decode frames.
        """
        # video related
        width, height, min_dimension, max_dimension = 0, 0, 0, 0
        video_start_pts, video_end_pts = 0, -1
        video_timebase_num, video_timebase_den = 0, 1
        # audio related
        samples, channels = 0, 0
        audio_start_pts, audio_end_pts = 0, -1
        audio_timebase_num, audio_timebase_den = 0, 1

        for test_video, config in test_videos.items():
            full_path = os.path.join(VIDEO_DIR, test_video)

            # pass 1: decode all frames using new decoder
            tv_result = torch.ops.video_reader.read_video_from_file(
                full_path,
                seek_frame_margin,
                0,  # getPtsOnly
                1,  # readVideoStream
                width,
                height,
                min_dimension,
                max_dimension,
                video_start_pts,
                video_end_pts,
                video_timebase_num,
                video_timebase_den,
                1,  # readAudioStream
                samples,
                channels,
                audio_start_pts,
                audio_end_pts,
                audio_timebase_num,
                audio_timebase_den,
            )
            # pass 2: decode all frames using av
            pyav_result = _decode_frames_by_av_module(full_path)
            # check results from TorchVision decoder
            self.check_separate_decoding_result(tv_result, config)
            # compare decoding results
            self.compare_decoding_result(tv_result, pyav_result, config)

    @PY39_SKIP
    def test_read_video_from_file_read_single_stream_only(self):
        """
        Test the case when decoder starts with a video file to decode frames, and
        only reads video stream and ignores audio stream
        """
        # video related
        width, height, min_dimension, max_dimension = 0, 0, 0, 0
        video_start_pts, video_end_pts = 0, -1
        video_timebase_num, video_timebase_den = 0, 1
        # audio related
        samples, channels = 0, 0
        audio_start_pts, audio_end_pts = 0, -1
        audio_timebase_num, audio_timebase_den = 0, 1

        for test_video, config in test_videos.items():
            full_path = os.path.join(VIDEO_DIR, test_video)
            for readVideoStream, readAudioStream in [(1, 0), (0, 1)]:
                # decode all frames using new decoder
                tv_result = torch.ops.video_reader.read_video_from_file(
                    full_path,
                    seek_frame_margin,
                    0,  # getPtsOnly
                    readVideoStream,
                    width,
                    height,
                    min_dimension,
                    max_dimension,
                    video_start_pts,
                    video_end_pts,
                    video_timebase_num,
                    video_timebase_den,
                    readAudioStream,
                    samples,
                    channels,
                    audio_start_pts,
                    audio_end_pts,
                    audio_timebase_num,
                    audio_timebase_den,
                )

                vframes, vframe_pts, vtimebase, vfps, vduration, \
                    aframes, aframe_pts, atimebase, asample_rate, aduration = (
                        tv_result
                    )

                self.assertEqual(vframes.numel() > 0, readVideoStream)
                self.assertEqual(vframe_pts.numel() > 0, readVideoStream)
                self.assertEqual(vtimebase.numel() > 0, readVideoStream)
                self.assertEqual(vfps.numel() > 0, readVideoStream)

                expect_audio_data = (
                    readAudioStream == 1 and config.audio_sample_rate is not None
                )
                self.assertEqual(aframes.numel() > 0, expect_audio_data)
                self.assertEqual(aframe_pts.numel() > 0, expect_audio_data)
                self.assertEqual(atimebase.numel() > 0, expect_audio_data)
                self.assertEqual(asample_rate.numel() > 0, expect_audio_data)

    def test_read_video_from_file_rescale_min_dimension(self):
        """
        Test the case when decoder starts with a video file to decode frames, and
        video min dimension between height and width is set.
        """
        # video related
        width, height, min_dimension, max_dimension = 0, 0, 128, 0
        video_start_pts, video_end_pts = 0, -1
        video_timebase_num, video_timebase_den = 0, 1
        # audio related
        samples, channels = 0, 0
        audio_start_pts, audio_end_pts = 0, -1
        audio_timebase_num, audio_timebase_den = 0, 1

        for test_video, _config in test_videos.items():
            full_path = os.path.join(VIDEO_DIR, test_video)

            tv_result = torch.ops.video_reader.read_video_from_file(
                full_path,
                seek_frame_margin,
                0,  # getPtsOnly
                1,  # readVideoStream
                width,
                height,
                min_dimension,
                max_dimension,
                video_start_pts,
                video_end_pts,
                video_timebase_num,
                video_timebase_den,
                1,  # readAudioStream
                samples,
                channels,
                audio_start_pts,
                audio_end_pts,
                audio_timebase_num,
                audio_timebase_den,
            )
            self.assertEqual(
                min_dimension, min(tv_result[0].size(1), tv_result[0].size(2))
            )

    def test_read_video_from_file_rescale_max_dimension(self):
        """
        Test the case when decoder starts with a video file to decode frames, and
        video min dimension between height and width is set.
        """
        # video related
        width, height, min_dimension, max_dimension = 0, 0, 0, 85
        video_start_pts, video_end_pts = 0, -1
        video_timebase_num, video_timebase_den = 0, 1
        # audio related
        samples, channels = 0, 0
        audio_start_pts, audio_end_pts = 0, -1
        audio_timebase_num, audio_timebase_den = 0, 1

        for test_video, _config in test_videos.items():
            full_path = os.path.join(VIDEO_DIR, test_video)

            tv_result = torch.ops.video_reader.read_video_from_file(
                full_path,
                seek_frame_margin,
                0,  # getPtsOnly
                1,  # readVideoStream
                width,
                height,
                min_dimension,
                max_dimension,
                video_start_pts,
                video_end_pts,
                video_timebase_num,
                video_timebase_den,
                1,  # readAudioStream
                samples,
                channels,
                audio_start_pts,
                audio_end_pts,
                audio_timebase_num,
                audio_timebase_den,
            )
            self.assertEqual(
                max_dimension, max(tv_result[0].size(1), tv_result[0].size(2))
            )

    def test_read_video_from_file_rescale_both_min_max_dimension(self):
        """
        Test the case when decoder starts with a video file to decode frames, and
        video min dimension between height and width is set.
        """
        # video related
        width, height, min_dimension, max_dimension = 0, 0, 64, 85
        video_start_pts, video_end_pts = 0, -1
        video_timebase_num, video_timebase_den = 0, 1
        # audio related
        samples, channels = 0, 0
        audio_start_pts, audio_end_pts = 0, -1
        audio_timebase_num, audio_timebase_den = 0, 1

        for test_video, _config in test_videos.items():
            full_path = os.path.join(VIDEO_DIR, test_video)

            tv_result = torch.ops.video_reader.read_video_from_file(
                full_path,
                seek_frame_margin,
                0,  # getPtsOnly
                1,  # readVideoStream
                width,
                height,
                min_dimension,
                max_dimension,
                video_start_pts,
                video_end_pts,
                video_timebase_num,
                video_timebase_den,
                1,  # readAudioStream
                samples,
                channels,
                audio_start_pts,
                audio_end_pts,
                audio_timebase_num,
                audio_timebase_den,
            )
            self.assertEqual(
                min_dimension, min(tv_result[0].size(1), tv_result[0].size(2))
            )
            self.assertEqual(
                max_dimension, max(tv_result[0].size(1), tv_result[0].size(2))
            )

    def test_read_video_from_file_rescale_width(self):
        """
        Test the case when decoder starts with a video file to decode frames, and
        video width is set.
        """
        # video related
        width, height, min_dimension, max_dimension = 256, 0, 0, 0
        video_start_pts, video_end_pts = 0, -1
        video_timebase_num, video_timebase_den = 0, 1
        # audio related
        samples, channels = 0, 0
        audio_start_pts, audio_end_pts = 0, -1
        audio_timebase_num, audio_timebase_den = 0, 1

        for test_video, _config in test_videos.items():
            full_path = os.path.join(VIDEO_DIR, test_video)

            tv_result = torch.ops.video_reader.read_video_from_file(
                full_path,
                seek_frame_margin,
                0,  # getPtsOnly
                1,  # readVideoStream
                width,
                height,
                min_dimension,
                max_dimension,
                video_start_pts,
                video_end_pts,
                video_timebase_num,
                video_timebase_den,
                1,  # readAudioStream
                samples,
                channels,
                audio_start_pts,
                audio_end_pts,
                audio_timebase_num,
                audio_timebase_den,
            )
            self.assertEqual(tv_result[0].size(2), width)

    def test_read_video_from_file_rescale_height(self):
        """
        Test the case when decoder starts with a video file to decode frames, and
        video height is set.
        """
        # video related
        width, height, min_dimension, max_dimension = 0, 224, 0, 0
        video_start_pts, video_end_pts = 0, -1
        video_timebase_num, video_timebase_den = 0, 1
        # audio related
        samples, channels = 0, 0
        audio_start_pts, audio_end_pts = 0, -1
        audio_timebase_num, audio_timebase_den = 0, 1

        for test_video, _config in test_videos.items():
            full_path = os.path.join(VIDEO_DIR, test_video)

            tv_result = torch.ops.video_reader.read_video_from_file(
                full_path,
                seek_frame_margin,
                0,  # getPtsOnly
                1,  # readVideoStream
                width,
                height,
                min_dimension,
                max_dimension,
                video_start_pts,
                video_end_pts,
                video_timebase_num,
                video_timebase_den,
                1,  # readAudioStream
                samples,
                channels,
                audio_start_pts,
                audio_end_pts,
                audio_timebase_num,
                audio_timebase_den,
            )
            self.assertEqual(tv_result[0].size(1), height)

    def test_read_video_from_file_rescale_width_and_height(self):
        """
        Test the case when decoder starts with a video file to decode frames, and
        both video height and width are set.
        """
        # video related
        width, height, min_dimension, max_dimension = 320, 240, 0, 0
        video_start_pts, video_end_pts = 0, -1
        video_timebase_num, video_timebase_den = 0, 1
        # audio related
        samples, channels = 0, 0
        audio_start_pts, audio_end_pts = 0, -1
        audio_timebase_num, audio_timebase_den = 0, 1

        for test_video, _config in test_videos.items():
            full_path = os.path.join(VIDEO_DIR, test_video)

            tv_result = torch.ops.video_reader.read_video_from_file(
                full_path,
                seek_frame_margin,
                0,  # getPtsOnly
                1,  # readVideoStream
                width,
                height,
                min_dimension,
                max_dimension,
                video_start_pts,
                video_end_pts,
                video_timebase_num,
                video_timebase_den,
                1,  # readAudioStream
                samples,
                channels,
                audio_start_pts,
                audio_end_pts,
                audio_timebase_num,
                audio_timebase_den,
            )
            self.assertEqual(tv_result[0].size(1), height)
            self.assertEqual(tv_result[0].size(2), width)

    @PY39_SKIP
    def test_read_video_from_file_audio_resampling(self):
        """
        Test the case when decoder starts with a video file to decode frames, and
        audio waveform are resampled
        """

        for samples in [9600, 96000]:  # downsampling  # upsampling
            # video related
            width, height, min_dimension, max_dimension = 0, 0, 0, 0
            video_start_pts, video_end_pts = 0, -1
            video_timebase_num, video_timebase_den = 0, 1
            # audio related
            channels = 0
            audio_start_pts, audio_end_pts = 0, -1
            audio_timebase_num, audio_timebase_den = 0, 1

            for test_video, _config in test_videos.items():
                full_path = os.path.join(VIDEO_DIR, test_video)

                tv_result = torch.ops.video_reader.read_video_from_file(
                    full_path,
                    seek_frame_margin,
                    0,  # getPtsOnly
                    1,  # readVideoStream
                    width,
                    height,
                    min_dimension,
                    max_dimension,
                    video_start_pts,
                    video_end_pts,
                    video_timebase_num,
                    video_timebase_den,
                    1,  # readAudioStream
                    samples,
                    channels,
                    audio_start_pts,
                    audio_end_pts,
                    audio_timebase_num,
                    audio_timebase_den,
                )
                vframes, vframe_pts, vtimebase, vfps, vduration, \
                    aframes, aframe_pts, atimebase, asample_rate, aduration = (
                        tv_result
                    )
                if aframes.numel() > 0:
                    self.assertEqual(samples, asample_rate.item())
                    self.assertEqual(1, aframes.size(1))
                    # when audio stream is found
                    duration = (
                        float(aframe_pts[-1])
                        * float(atimebase[0])
                        / float(atimebase[1])
                    )
                    self.assertAlmostEqual(
                        aframes.size(0),
                        int(duration * asample_rate.item()),
                        delta=0.1 * asample_rate.item(),
                    )

    @PY39_SKIP
    def test_compare_read_video_from_memory_and_file(self):
        """
        Test the case when video is already in memory, and decoder reads data in memory
        """
        # video related
        width, height, min_dimension, max_dimension = 0, 0, 0, 0
        video_start_pts, video_end_pts = 0, -1
        video_timebase_num, video_timebase_den = 0, 1
        # audio related
        samples, channels = 0, 0
        audio_start_pts, audio_end_pts = 0, -1
        audio_timebase_num, audio_timebase_den = 0, 1

        for test_video, config in test_videos.items():
            full_path, video_tensor = _get_video_tensor(VIDEO_DIR, test_video)

            # pass 1: decode all frames using cpp decoder
            tv_result_memory = torch.ops.video_reader.read_video_from_memory(
                video_tensor,
                seek_frame_margin,
                0,  # getPtsOnly
                1,  # readVideoStream
                width,
                height,
                min_dimension,
                max_dimension,
                video_start_pts,
                video_end_pts,
                video_timebase_num,
                video_timebase_den,
                1,  # readAudioStream
                samples,
                channels,
                audio_start_pts,
                audio_end_pts,
                audio_timebase_num,
                audio_timebase_den,
            )
            self.check_separate_decoding_result(tv_result_memory, config)
            # pass 2: decode all frames from file
            tv_result_file = torch.ops.video_reader.read_video_from_file(
                full_path,
                seek_frame_margin,
                0,  # getPtsOnly
                1,  # readVideoStream
                width,
                height,
                min_dimension,
                max_dimension,
                video_start_pts,
                video_end_pts,
                video_timebase_num,
                video_timebase_den,
                1,  # readAudioStream
                samples,
                channels,
                audio_start_pts,
                audio_end_pts,
                audio_timebase_num,
                audio_timebase_den,
            )

            self.check_separate_decoding_result(tv_result_file, config)
            # finally, compare results decoded from memory and file
            self.compare_decoding_result(tv_result_memory, tv_result_file)

    @PY39_SKIP
    def test_read_video_from_memory(self):
        """
        Test the case when video is already in memory, and decoder reads data in memory
        """
        # video related
        width, height, min_dimension, max_dimension = 0, 0, 0, 0
        video_start_pts, video_end_pts = 0, -1
        video_timebase_num, video_timebase_den = 0, 1
        # audio related
        samples, channels = 0, 0
        audio_start_pts, audio_end_pts = 0, -1
        audio_timebase_num, audio_timebase_den = 0, 1

        for test_video, config in test_videos.items():
            full_path, video_tensor = _get_video_tensor(VIDEO_DIR, test_video)

            # pass 1: decode all frames using cpp decoder
            tv_result = torch.ops.video_reader.read_video_from_memory(
                video_tensor,
                seek_frame_margin,
                0,  # getPtsOnly
                1,  # readVideoStream
                width,
                height,
                min_dimension,
                max_dimension,
                video_start_pts,
                video_end_pts,
                video_timebase_num,
                video_timebase_den,
                1,  # readAudioStream
                samples,
                channels,
                audio_start_pts,
                audio_end_pts,
                audio_timebase_num,
                audio_timebase_den,
            )
            # pass 2: decode all frames using av
            pyav_result = _decode_frames_by_av_module(full_path)

            self.check_separate_decoding_result(tv_result, config)
            self.compare_decoding_result(tv_result, pyav_result, config)

    @PY39_SKIP
    def test_read_video_from_memory_get_pts_only(self):
        """
        Test the case when video is already in memory, and decoder reads data in memory.
        Compare frame pts between decoding for pts only and full decoding
        for both pts and frame data
        """
        # video related
        width, height, min_dimension, max_dimension = 0, 0, 0, 0
        video_start_pts, video_end_pts = 0, -1
        video_timebase_num, video_timebase_den = 0, 1
        # audio related
        samples, channels = 0, 0
        audio_start_pts, audio_end_pts = 0, -1
        audio_timebase_num, audio_timebase_den = 0, 1

        for test_video, config in test_videos.items():
            full_path, video_tensor = _get_video_tensor(VIDEO_DIR, test_video)

            # pass 1: decode all frames using cpp decoder
            tv_result = torch.ops.video_reader.read_video_from_memory(
                video_tensor,
                seek_frame_margin,
                0,  # getPtsOnly
                1,  # readVideoStream
                width,
                height,
                min_dimension,
                max_dimension,
                video_start_pts,
                video_end_pts,
                video_timebase_num,
                video_timebase_den,
                1,  # readAudioStream
                samples,
                channels,
                audio_start_pts,
                audio_end_pts,
                audio_timebase_num,
                audio_timebase_den,
            )
            self.assertAlmostEqual(config.video_fps, tv_result[3].item(), delta=0.01)

            # pass 2: decode all frames to get PTS only using cpp decoder
            tv_result_pts_only = torch.ops.video_reader.read_video_from_memory(
                video_tensor,
                seek_frame_margin,
                1,  # getPtsOnly
                1,  # readVideoStream
                width,
                height,
                min_dimension,
                max_dimension,
                video_start_pts,
                video_end_pts,
                video_timebase_num,
                video_timebase_den,
                1,  # readAudioStream
                samples,
                channels,
                audio_start_pts,
                audio_end_pts,
                audio_timebase_num,
                audio_timebase_den,
            )

            self.assertEqual(tv_result_pts_only[0].numel(), 0)
            self.assertEqual(tv_result_pts_only[5].numel(), 0)
            self.compare_decoding_result(tv_result, tv_result_pts_only)

    @PY39_SKIP
    def test_read_video_in_range_from_memory(self):
        """
        Test the case when video is already in memory, and decoder reads data in memory.
        In addition, decoder takes meaningful start- and end PTS as input, and decode
        frames within that interval
        """
        for test_video, config in test_videos.items():
            full_path, video_tensor = _get_video_tensor(VIDEO_DIR, test_video)
            # video related
            width, height, min_dimension, max_dimension = 0, 0, 0, 0
            video_start_pts, video_end_pts = 0, -1
            video_timebase_num, video_timebase_den = 0, 1
            # audio related
            samples, channels = 0, 0
            audio_start_pts, audio_end_pts = 0, -1
            audio_timebase_num, audio_timebase_den = 0, 1
            # pass 1: decode all frames using new decoder
            tv_result = torch.ops.video_reader.read_video_from_memory(
                video_tensor,
                seek_frame_margin,
                0,  # getPtsOnly
                1,  # readVideoStream
                width,
                height,
                min_dimension,
                max_dimension,
                video_start_pts,
                video_end_pts,
                video_timebase_num,
                video_timebase_den,
                1,  # readAudioStream
                samples,
                channels,
                audio_start_pts,
                audio_end_pts,
                audio_timebase_num,
                audio_timebase_den,
            )
            vframes, vframe_pts, vtimebase, vfps, vduration, \
                aframes, aframe_pts, atimebase, asample_rate, aduration = (
                    tv_result
                )
            self.assertAlmostEqual(config.video_fps, vfps.item(), delta=0.01)

            for num_frames in [4, 8, 16, 32, 64, 128]:
                start_pts_ind_max = vframe_pts.size(0) - num_frames
                if start_pts_ind_max <= 0:
                    continue
                # randomly pick start pts
                start_pts_ind = randint(0, start_pts_ind_max)
                end_pts_ind = start_pts_ind + num_frames - 1
                video_start_pts = vframe_pts[start_pts_ind]
                video_end_pts = vframe_pts[end_pts_ind]

                video_timebase_num, video_timebase_den = vtimebase[0], vtimebase[1]
                if len(atimebase) > 0:
                    # when audio stream is available
                    audio_timebase_num, audio_timebase_den = atimebase[0], atimebase[1]
                    audio_start_pts = _pts_convert(
                        video_start_pts.item(),
                        Fraction(video_timebase_num.item(), video_timebase_den.item()),
                        Fraction(audio_timebase_num.item(), audio_timebase_den.item()),
                        math.floor,
                    )
                    audio_end_pts = _pts_convert(
                        video_end_pts.item(),
                        Fraction(video_timebase_num.item(), video_timebase_den.item()),
                        Fraction(audio_timebase_num.item(), audio_timebase_den.item()),
                        math.ceil,
                    )

                # pass 2: decode frames in the randomly generated range
                tv_result = torch.ops.video_reader.read_video_from_memory(
                    video_tensor,
                    seek_frame_margin,
                    0,  # getPtsOnly
                    1,  # readVideoStream
                    width,
                    height,
                    min_dimension,
                    max_dimension,
                    video_start_pts,
                    video_end_pts,
                    video_timebase_num,
                    video_timebase_den,
                    1,  # readAudioStream
                    samples,
                    channels,
                    audio_start_pts,
                    audio_end_pts,
                    audio_timebase_num,
                    audio_timebase_den,
                )

                # pass 3: decode frames in range using PyAv
                video_timebase_av, audio_timebase_av = _get_timebase_by_av_module(
                    full_path
                )

                video_start_pts_av = _pts_convert(
                    video_start_pts.item(),
                    Fraction(video_timebase_num.item(), video_timebase_den.item()),
                    Fraction(
                        video_timebase_av.numerator, video_timebase_av.denominator
                    ),
                    math.floor,
                )
                video_end_pts_av = _pts_convert(
                    video_end_pts.item(),
                    Fraction(video_timebase_num.item(), video_timebase_den.item()),
                    Fraction(
                        video_timebase_av.numerator, video_timebase_av.denominator
                    ),
                    math.ceil,
                )
                if audio_timebase_av:
                    audio_start_pts = _pts_convert(
                        video_start_pts.item(),
                        Fraction(video_timebase_num.item(), video_timebase_den.item()),
                        Fraction(
                            audio_timebase_av.numerator, audio_timebase_av.denominator
                        ),
                        math.floor,
                    )
                    audio_end_pts = _pts_convert(
                        video_end_pts.item(),
                        Fraction(video_timebase_num.item(), video_timebase_den.item()),
                        Fraction(
                            audio_timebase_av.numerator, audio_timebase_av.denominator
                        ),
                        math.ceil,
                    )

                pyav_result = _decode_frames_by_av_module(
                    full_path,
                    video_start_pts_av,
                    video_end_pts_av,
                    audio_start_pts,
                    audio_end_pts,
                )

                self.assertEqual(tv_result[0].size(0), num_frames)
                if pyav_result.vframes.size(0) == num_frames:
                    # if PyAv decodes a different number of video frames, skip
                    # comparing the decoding results between Torchvision video reader
                    # and PyAv
                    self.compare_decoding_result(tv_result, pyav_result, config)

    def test_probe_video_from_file(self):
        """
        Test the case when decoder probes a video file
        """
        for test_video, config in test_videos.items():
            full_path = os.path.join(VIDEO_DIR, test_video)
            probe_result = torch.ops.video_reader.probe_video_from_file(full_path)
            self.check_probe_result(probe_result, config)

    def test_probe_video_from_memory(self):
        """
        Test the case when decoder probes a video in memory
        """
        for test_video, config in test_videos.items():
            full_path, video_tensor = _get_video_tensor(VIDEO_DIR, test_video)
            probe_result = torch.ops.video_reader.probe_video_from_memory(video_tensor)
            self.check_probe_result(probe_result, config)

    def test_probe_video_from_memory_script(self):
        scripted_fun = torch.jit.script(io._probe_video_from_memory)
        self.assertIsNotNone(scripted_fun)

        for test_video, config in test_videos.items():
            full_path, video_tensor = _get_video_tensor(VIDEO_DIR, test_video)
            probe_result = scripted_fun(video_tensor)
            self.check_meta_result(probe_result, config)

    @PY39_SKIP
    def test_read_video_from_memory_scripted(self):
        """
        Test the case when video is already in memory, and decoder reads data in memory
        """
        # video related
        width, height, min_dimension, max_dimension = 0, 0, 0, 0
        video_start_pts, video_end_pts = 0, -1
        video_timebase_num, video_timebase_den = 0, 1
        # audio related
        samples, channels = 0, 0
        audio_start_pts, audio_end_pts = 0, -1
        audio_timebase_num, audio_timebase_den = 0, 1

        scripted_fun = torch.jit.script(io._read_video_from_memory)
        self.assertIsNotNone(scripted_fun)

        for test_video, _config in test_videos.items():
            full_path, video_tensor = _get_video_tensor(VIDEO_DIR, test_video)

            # decode all frames using cpp decoder
            scripted_fun(
                video_tensor,
                seek_frame_margin,
                1,  # readVideoStream
                width,
                height,
                min_dimension,
                max_dimension,
                [video_start_pts, video_end_pts],
                video_timebase_num,
                video_timebase_den,
                1,  # readAudioStream
                samples,
                channels,
                [audio_start_pts, audio_end_pts],
                audio_timebase_num,
                audio_timebase_den,
            )
            # FUTURE: check value of video / audio frames


if __name__ == "__main__":
    unittest.main()
