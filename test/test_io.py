import os
import contextlib
import tempfile
import torch
import torchvision.datasets.utils as utils
import torchvision.io as io
from torchvision import get_video_backend
import unittest
import sys
import warnings

from common_utils import get_tmp_dir

if sys.version_info < (3,):
    from urllib2 import URLError
else:
    from urllib.error import URLError

try:
    import av
    # Do a version test too
    io.video._check_av_available()
except ImportError:
    av = None

_video_backend = get_video_backend()


def _read_video(filename, start_pts=0, end_pts=None):
    if _video_backend == "pyav":
        return io.read_video(filename, start_pts, end_pts)
    else:
        if end_pts is None:
            end_pts = -1
        return io._read_video_from_file(
            filename,
            video_pts_range=(start_pts, end_pts),
        )


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
        assert video_codec is None, "video_codec can't be specified together with lossless"
        assert options is None, "options can't be specified together with lossless"
        video_codec = 'libx264rgb'
        options = {'crf': '0'}

    if video_codec is None:
        if _video_backend == "pyav":
            video_codec = 'libx264'
        else:
            # when video_codec is not set, we assume it is libx264rgb which accepts
            # RGB pixel formats as input instead of YUV
            video_codec = 'libx264rgb'
    if options is None:
        options = {}

    data = _create_video_frames(num_frames, height, width)
    with tempfile.NamedTemporaryFile(suffix='.mp4') as f:
        io.write_video(f.name, data, fps=fps, video_codec=video_codec, options=options)
        yield f.name, data


@unittest.skipIf(av is None, "PyAV unavailable")
@unittest.skipIf('win' in sys.platform, 'temporarily disabled on Windows')
class Tester(unittest.TestCase):
    # compression adds artifacts, thus we add a tolerance of
    # 6 in 0-255 range
    TOLERANCE = 6

    def test_write_read_video(self):
        with temp_video(10, 300, 300, 5, lossless=True) as (f_name, data):
            lv, _, info = _read_video(f_name)
            self.assertTrue(data.equal(lv))
            self.assertEqual(info["video_fps"], 5)

    def test_read_timestamps(self):
        with temp_video(10, 300, 300, 5) as (f_name, data):
            if _video_backend == "pyav":
                pts, _ = io.read_video_timestamps(f_name)
            else:
                pts, _, _ = io._read_video_timestamps_from_file(f_name)
            # note: not all formats/codecs provide accurate information for computing the
            # timestamps. For the format that we use here, this information is available,
            # so we use it as a baseline
            container = av.open(f_name)
            stream = container.streams[0]
            pts_step = int(round(float(1 / (stream.average_rate * stream.time_base))))
            num_frames = int(round(float(stream.average_rate * stream.time_base * stream.duration)))
            expected_pts = [i * pts_step for i in range(num_frames)]

            self.assertEqual(pts, expected_pts)

    def test_read_partial_video(self):
        with temp_video(10, 300, 300, 5, lossless=True) as (f_name, data):
            if _video_backend == "pyav":
                pts, _ = io.read_video_timestamps(f_name)
            else:
                pts, _, _ = io._read_video_timestamps_from_file(f_name)
            for start in range(5):
                for l in range(1, 4):
                    lv, _, _ = _read_video(f_name, pts[start], pts[start + l - 1])
                    s_data = data[start:(start + l)]
                    self.assertEqual(len(lv), l)
                    self.assertTrue(s_data.equal(lv))

            if _video_backend == "pyav":
                # for "video_reader" backend, we don't decode the closest early frame
                # when the given start pts is not matching any frame pts
                lv, _, _ = _read_video(f_name, pts[4] + 1, pts[7])
                self.assertEqual(len(lv), 4)
                self.assertTrue(data[4:8].equal(lv))

    def test_read_partial_video_bframes(self):
        # do not use lossless encoding, to test the presence of B-frames
        options = {'bframes': '16', 'keyint': '10', 'min-keyint': '4'}
        with temp_video(100, 300, 300, 5, options=options) as (f_name, data):
            if _video_backend == "pyav":
                pts, _ = io.read_video_timestamps(f_name)
            else:
                pts, _, _ = io._read_video_timestamps_from_file(f_name)
            for start in range(0, 80, 20):
                for l in range(1, 4):
                    lv, _, _ = _read_video(f_name, pts[start], pts[start + l - 1])
                    s_data = data[start:(start + l)]
                    self.assertEqual(len(lv), l)
                    self.assertTrue((s_data.float() - lv.float()).abs().max() < self.TOLERANCE)

            lv, _, _ = io.read_video(f_name, pts[4] + 1, pts[7])
            self.assertEqual(len(lv), 4)
            self.assertTrue((data[4:8].float() - lv.float()).abs().max() < self.TOLERANCE)

    def test_read_packed_b_frames_divx_file(self):
        with get_tmp_dir() as temp_dir:
            name = "hmdb51_Turnk_r_Pippi_Michel_cartwheel_f_cm_np2_le_med_6.avi"
            f_name = os.path.join(temp_dir, name)
            url = "https://download.pytorch.org/vision_tests/io/" + name
            try:
                utils.download_url(url, temp_dir)
                if _video_backend == "pyav":
                    pts, fps = io.read_video_timestamps(f_name)
                else:
                    pts, _, info = io._read_video_timestamps_from_file(f_name)
                    fps = info["video_fps"]

                self.assertEqual(pts, sorted(pts))
                self.assertEqual(fps, 30)
            except URLError:
                msg = "could not download test file '{}'".format(url)
                warnings.warn(msg, RuntimeWarning)
                raise unittest.SkipTest(msg)

    def test_read_timestamps_from_packet(self):
        with temp_video(10, 300, 300, 5, video_codec='mpeg4') as (f_name, data):
            if _video_backend == "pyav":
                pts, _ = io.read_video_timestamps(f_name)
            else:
                pts, _, _ = io._read_video_timestamps_from_file(f_name)
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

    # TODO add tests for audio


if __name__ == '__main__':
    unittest.main()
