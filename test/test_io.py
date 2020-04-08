import os
import contextlib
import tempfile
import torch
import torchvision.datasets.utils as utils
import torchvision.io as io
from torchvision import get_video_backend
import unittest
import warnings
from urllib.error import URLError

from common_utils import get_tmp_dir


try:
    import av
    # Do a version test too
    io.video._check_av_available()
except ImportError:
    av = None


def _create_video_frames(num_frames, height, width):
    y, x = torch.meshgrid(torch.linspace(-2, 2, height), torch.linspace(-2, 2, width))
    data = []
    for i in range(num_frames):
        xc = float(i) / num_frames
        yc = 1 - float(i) / (2 * num_frames)
        d = torch.exp(-((x - xc) ** 2 + (y - yc) ** 2) / 2) * 255
        data.append(d.unsqueeze(2).repeat(1, 1, 3).byte())

    return torch.stack(data, 0)


@contextlib.contextmanager
def temp_video(num_frames, height, width, fps, lossless=False, video_codec=None, options=None):
    if lossless:
        if video_codec is not None:
            raise ValueError("video_codec can't be specified together with lossless")
        if options is not None:
            raise ValueError("options can't be specified together with lossless")
        video_codec = 'libx264rgb'
        options = {'crf': '0'}

    if video_codec is None:
        if get_video_backend() == "pyav":
            video_codec = 'libx264'
        else:
            # when video_codec is not set, we assume it is libx264rgb which accepts
            # RGB pixel formats as input instead of YUV
            video_codec = 'libx264rgb'
    if options is None:
        options = {}

    data = _create_video_frames(num_frames, height, width)
    with tempfile.NamedTemporaryFile(suffix='.mp4') as f:
        f.close()
        io.write_video(f.name, data, fps=fps, video_codec=video_codec, options=options)
        yield f.name, data
    os.unlink(f.name)

@unittest.skipIf(get_video_backend() != "pyav" and not io._HAS_VIDEO_OPT,
                 "video_reader backend not available")
@unittest.skipIf(av is None, "PyAV unavailable")
class Tester(unittest.TestCase):
    # compression adds artifacts, thus we add a tolerance of
    # 6 in 0-255 range
    TOLERANCE = 6

    def test_write_read_video(self):
        with temp_video(10, 300, 300, 5, lossless=True) as (f_name, data):
            lv, _, info = io.read_video(f_name)
            self.assertTrue(data.equal(lv))
            self.assertEqual(info["video_fps"], 5)

    @unittest.skipIf(not io._HAS_VIDEO_OPT, "video_reader backend is not chosen")
    def test_probe_video_from_file(self):
        with temp_video(10, 300, 300, 5) as (f_name, data):
            video_info = io._probe_video_from_file(f_name)
            self.assertAlmostEqual(video_info.video_duration, 2, delta=0.1)
            self.assertAlmostEqual(video_info.video_fps, 5, delta=0.1)

    @unittest.skipIf(not io._HAS_VIDEO_OPT, "video_reader backend is not chosen")
    def test_probe_video_from_memory(self):
        with temp_video(10, 300, 300, 5) as (f_name, data):
            with open(f_name, "rb") as fp:
                filebuffer = fp.read()
            video_info = io._probe_video_from_memory(filebuffer)
            self.assertAlmostEqual(video_info.video_duration, 2, delta=0.1)
            self.assertAlmostEqual(video_info.video_fps, 5, delta=0.1)

    def test_read_timestamps(self):
        with temp_video(10, 300, 300, 5) as (f_name, data):
            pts, _ = io.read_video_timestamps(f_name)
            # note: not all formats/codecs provide accurate information for computing the
            # timestamps. For the format that we use here, this information is available,
            # so we use it as a baseline
            container = av.open(f_name)
            stream = container.streams[0]
            pts_step = int(round(float(1 / (stream.average_rate * stream.time_base))))
            num_frames = int(round(float(stream.average_rate * stream.time_base * stream.duration)))
            expected_pts = [i * pts_step for i in range(num_frames)]

            self.assertEqual(pts, expected_pts)
            container.close()

    def test_read_partial_video(self):
        with temp_video(10, 300, 300, 5, lossless=True) as (f_name, data):
            pts, _ = io.read_video_timestamps(f_name)
            for start in range(5):
                for l in range(1, 4):
                    lv, _, _ = io.read_video(f_name, pts[start], pts[start + l - 1])
                    s_data = data[start:(start + l)]
                    self.assertEqual(len(lv), l)
                    self.assertTrue(s_data.equal(lv))

            if get_video_backend() == "pyav":
                # for "video_reader" backend, we don't decode the closest early frame
                # when the given start pts is not matching any frame pts
                lv, _, _ = io.read_video(f_name, pts[4] + 1, pts[7])
                self.assertEqual(len(lv), 4)
                self.assertTrue(data[4:8].equal(lv))

    def test_read_partial_video_bframes(self):
        # do not use lossless encoding, to test the presence of B-frames
        options = {'bframes': '16', 'keyint': '10', 'min-keyint': '4'}
        with temp_video(100, 300, 300, 5, options=options) as (f_name, data):
            pts, _ = io.read_video_timestamps(f_name)
            for start in range(0, 80, 20):
                for l in range(1, 4):
                    lv, _, _ = io.read_video(f_name, pts[start], pts[start + l - 1])
                    s_data = data[start:(start + l)]
                    self.assertEqual(len(lv), l)
                    self.assertTrue((s_data.float() - lv.float()).abs().max() < self.TOLERANCE)

            lv, _, _ = io.read_video(f_name, pts[4] + 1, pts[7])
            # TODO fix this
            if get_video_backend() == 'pyav':
                self.assertEqual(len(lv), 4)
                self.assertTrue((data[4:8].float() - lv.float()).abs().max() < self.TOLERANCE)
            else:
                self.assertEqual(len(lv), 3)
                self.assertTrue((data[5:8].float() - lv.float()).abs().max() < self.TOLERANCE)

    def test_read_packed_b_frames_divx_file(self):
        with get_tmp_dir() as temp_dir:
            name = "hmdb51_Turnk_r_Pippi_Michel_cartwheel_f_cm_np2_le_med_6.avi"
            f_name = os.path.join(temp_dir, name)
            url = "https://download.pytorch.org/vision_tests/io/" + name
            try:
                utils.download_url(url, temp_dir)
                pts, fps = io.read_video_timestamps(f_name)

                self.assertEqual(pts, sorted(pts))
                self.assertEqual(fps, 30)
            except URLError:
                msg = "could not download test file '{}'".format(url)
                warnings.warn(msg, RuntimeWarning)
                raise unittest.SkipTest(msg)

    def test_read_timestamps_from_packet(self):
        with temp_video(10, 300, 300, 5, video_codec='mpeg4') as (f_name, data):
            pts, _ = io.read_video_timestamps(f_name)
            # note: not all formats/codecs provide accurate information for computing the
            # timestamps. For the format that we use here, this information is available,
            # so we use it as a baseline
            container = av.open(f_name)
            stream = container.streams[0]
            # make sure we went through the optimized codepath
            self.assertIn(b'Lavc', stream.codec_context.extradata)
            pts_step = int(round(float(1 / (stream.average_rate * stream.time_base))))
            num_frames = int(round(float(stream.average_rate * stream.time_base * stream.duration)))
            expected_pts = [i * pts_step for i in range(num_frames)]

            self.assertEqual(pts, expected_pts)
            container.close()

    def test_read_video_pts_unit_sec(self):
        with temp_video(10, 300, 300, 5, lossless=True) as (f_name, data):
            lv, _, info = io.read_video(f_name, pts_unit='sec')

            self.assertTrue(data.equal(lv))
            self.assertEqual(info["video_fps"], 5)
            self.assertEqual(info, {"video_fps": 5})

    def test_read_timestamps_pts_unit_sec(self):
        with temp_video(10, 300, 300, 5) as (f_name, data):
            pts, _ = io.read_video_timestamps(f_name, pts_unit='sec')

            container = av.open(f_name)
            stream = container.streams[0]
            pts_step = int(round(float(1 / (stream.average_rate * stream.time_base))))
            num_frames = int(round(float(stream.average_rate * stream.time_base * stream.duration)))
            expected_pts = [i * pts_step * stream.time_base for i in range(num_frames)]

            self.assertEqual(pts, expected_pts)
            container.close()

    def test_read_partial_video_pts_unit_sec(self):
        with temp_video(10, 300, 300, 5, lossless=True) as (f_name, data):
            pts, _ = io.read_video_timestamps(f_name, pts_unit='sec')

            for start in range(5):
                for l in range(1, 4):
                    lv, _, _ = io.read_video(f_name, pts[start], pts[start + l - 1], pts_unit='sec')
                    s_data = data[start:(start + l)]
                    self.assertEqual(len(lv), l)
                    self.assertTrue(s_data.equal(lv))

            container = av.open(f_name)
            stream = container.streams[0]
            lv, _, _ = io.read_video(f_name,
                                     int(pts[4] * (1.0 / stream.time_base) + 1) * stream.time_base, pts[7],
                                     pts_unit='sec')
            if get_video_backend() == "pyav":
                # for "video_reader" backend, we don't decode the closest early frame
                # when the given start pts is not matching any frame pts
                self.assertEqual(len(lv), 4)
                self.assertTrue(data[4:8].equal(lv))
            container.close()

    def test_read_video_corrupted_file(self):
        with tempfile.NamedTemporaryFile(suffix='.mp4') as f:
            f.write(b'This is not an mpg4 file')
            video, audio, info = io.read_video(f.name)
            self.assertIsInstance(video, torch.Tensor)
            self.assertIsInstance(audio, torch.Tensor)
            self.assertEqual(video.numel(), 0)
            self.assertEqual(audio.numel(), 0)
            self.assertEqual(info, {})

    def test_read_video_timestamps_corrupted_file(self):
        with tempfile.NamedTemporaryFile(suffix='.mp4') as f:
            f.write(b'This is not an mpg4 file')
            video_pts, video_fps = io.read_video_timestamps(f.name)
            self.assertEqual(video_pts, [])
            self.assertIs(video_fps, None)

    def test_read_video_partially_corrupted_file(self):
        with temp_video(5, 4, 4, 5, lossless=True) as (f_name, data):
            with open(f_name, 'r+b') as f:
                size = os.path.getsize(f_name)
                bytes_to_overwrite = size // 10
                # seek to the middle of the file
                f.seek(5 * bytes_to_overwrite)
                # corrupt 10% of the file from the middle
                f.write(b'\xff' * bytes_to_overwrite)
            # this exercises the container.decode assertion check
            video, audio, info = io.read_video(f.name, pts_unit='sec')
            # check that size is not equal to 5, but 3
            # TODO fix this
            if get_video_backend() == 'pyav':
                self.assertEqual(len(video), 3)
            else:
                self.assertEqual(len(video), 4)
            # but the valid decoded content is still correct
            self.assertTrue(video[:3].equal(data[:3]))
            # and the last few frames are wrong
            self.assertFalse(video.equal(data))

    # TODO add tests for audio


if __name__ == '__main__':
    unittest.main()
