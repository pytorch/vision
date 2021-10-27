import contextlib
import os
import sys
import tempfile

import pytest
import torch
import torchvision.io as io
from common_utils import assert_equal
from torchvision import get_video_backend


try:
    import av

    # Do a version test too
    io.video._check_av_available()
except ImportError:
    av = None


VIDEO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "videos")


def _create_video_frames(num_frames, height, width):
    y, x = torch.meshgrid(torch.linspace(-2, 2, height), torch.linspace(-2, 2, width), indexing="ij")
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
        video_codec = "libx264rgb"
        options = {"crf": "0"}

    if video_codec is None:
        if get_video_backend() == "pyav":
            video_codec = "libx264"
        else:
            # when video_codec is not set, we assume it is libx264rgb which accepts
            # RGB pixel formats as input instead of YUV
            video_codec = "libx264rgb"
    if options is None:
        options = {}

    data = _create_video_frames(num_frames, height, width)
    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        f.close()
        io.write_video(f.name, data, fps=fps, video_codec=video_codec, options=options)
        yield f.name, data
    os.unlink(f.name)


@pytest.mark.skipif(
    get_video_backend() != "pyav" and not io._HAS_VIDEO_OPT, reason="video_reader backend not available"
)
@pytest.mark.skipif(av is None, reason="PyAV unavailable")
class TestVideo:
    # compression adds artifacts, thus we add a tolerance of
    # 6 in 0-255 range
    TOLERANCE = 6

    def test_write_read_video(self):
        with temp_video(10, 300, 300, 5, lossless=True) as (f_name, data):
            lv, _, info = io.read_video(f_name)
            assert_equal(data, lv)
            assert info["video_fps"] == 5

    @pytest.mark.skipif(not io._HAS_VIDEO_OPT, reason="video_reader backend is not chosen")
    def test_probe_video_from_file(self):
        with temp_video(10, 300, 300, 5) as (f_name, data):
            video_info = io._probe_video_from_file(f_name)
            assert pytest.approx(2, rel=0.0, abs=0.1) == video_info.video_duration
            assert pytest.approx(5, rel=0.0, abs=0.1) == video_info.video_fps

    @pytest.mark.skipif(not io._HAS_VIDEO_OPT, reason="video_reader backend is not chosen")
    def test_probe_video_from_memory(self):
        with temp_video(10, 300, 300, 5) as (f_name, data):
            with open(f_name, "rb") as fp:
                filebuffer = fp.read()
            video_info = io._probe_video_from_memory(filebuffer)
            assert pytest.approx(2, rel=0.0, abs=0.1) == video_info.video_duration
            assert pytest.approx(5, rel=0.0, abs=0.1) == video_info.video_fps

    def test_read_timestamps(self):
        with temp_video(10, 300, 300, 5) as (f_name, data):
            pts, _ = io.read_video_timestamps(f_name)
            # note: not all formats/codecs provide accurate information for computing the
            # timestamps. For the format that we use here, this information is available,
            # so we use it as a baseline
            with av.open(f_name) as container:
                stream = container.streams[0]
                pts_step = int(round(float(1 / (stream.average_rate * stream.time_base))))
                num_frames = int(round(float(stream.average_rate * stream.time_base * stream.duration)))
                expected_pts = [i * pts_step for i in range(num_frames)]

            assert pts == expected_pts

    @pytest.mark.parametrize("start", range(5))
    @pytest.mark.parametrize("offset", range(1, 4))
    def test_read_partial_video(self, start, offset):
        with temp_video(10, 300, 300, 5, lossless=True) as (f_name, data):
            pts, _ = io.read_video_timestamps(f_name)

            lv, _, _ = io.read_video(f_name, pts[start], pts[start + offset - 1])
            s_data = data[start : (start + offset)]
            assert len(lv) == offset
            assert_equal(s_data, lv)

            if get_video_backend() == "pyav":
                # for "video_reader" backend, we don't decode the closest early frame
                # when the given start pts is not matching any frame pts
                lv, _, _ = io.read_video(f_name, pts[4] + 1, pts[7])
                assert len(lv) == 4
                assert_equal(data[4:8], lv)

    @pytest.mark.parametrize("start", range(0, 80, 20))
    @pytest.mark.parametrize("offset", range(1, 4))
    def test_read_partial_video_bframes(self, start, offset):
        # do not use lossless encoding, to test the presence of B-frames
        options = {"bframes": "16", "keyint": "10", "min-keyint": "4"}
        with temp_video(100, 300, 300, 5, options=options) as (f_name, data):
            pts, _ = io.read_video_timestamps(f_name)

            lv, _, _ = io.read_video(f_name, pts[start], pts[start + offset - 1])
            s_data = data[start : (start + offset)]
            assert len(lv) == offset
            assert_equal(s_data, lv, rtol=0.0, atol=self.TOLERANCE)

            lv, _, _ = io.read_video(f_name, pts[4] + 1, pts[7])
            # TODO fix this
            if get_video_backend() == "pyav":
                assert len(lv) == 4
                assert_equal(data[4:8], lv, rtol=0.0, atol=self.TOLERANCE)
            else:
                assert len(lv) == 3
                assert_equal(data[5:8], lv, rtol=0.0, atol=self.TOLERANCE)

    def test_read_packed_b_frames_divx_file(self):
        name = "hmdb51_Turnk_r_Pippi_Michel_cartwheel_f_cm_np2_le_med_6.avi"
        f_name = os.path.join(VIDEO_DIR, name)
        pts, fps = io.read_video_timestamps(f_name)

        assert pts == sorted(pts)
        assert fps == 30

    def test_read_timestamps_from_packet(self):
        with temp_video(10, 300, 300, 5, video_codec="mpeg4") as (f_name, data):
            pts, _ = io.read_video_timestamps(f_name)
            # note: not all formats/codecs provide accurate information for computing the
            # timestamps. For the format that we use here, this information is available,
            # so we use it as a baseline
            with av.open(f_name) as container:
                stream = container.streams[0]
                # make sure we went through the optimized codepath
                assert b"Lavc" in stream.codec_context.extradata
                pts_step = int(round(float(1 / (stream.average_rate * stream.time_base))))
                num_frames = int(round(float(stream.average_rate * stream.time_base * stream.duration)))
                expected_pts = [i * pts_step for i in range(num_frames)]

            assert pts == expected_pts

    def test_read_video_pts_unit_sec(self):
        with temp_video(10, 300, 300, 5, lossless=True) as (f_name, data):
            lv, _, info = io.read_video(f_name, pts_unit="sec")

            assert_equal(data, lv)
            assert info["video_fps"] == 5
            assert info == {"video_fps": 5}

    def test_read_timestamps_pts_unit_sec(self):
        with temp_video(10, 300, 300, 5) as (f_name, data):
            pts, _ = io.read_video_timestamps(f_name, pts_unit="sec")

            with av.open(f_name) as container:
                stream = container.streams[0]
                pts_step = int(round(float(1 / (stream.average_rate * stream.time_base))))
                num_frames = int(round(float(stream.average_rate * stream.time_base * stream.duration)))
                expected_pts = [i * pts_step * stream.time_base for i in range(num_frames)]

            assert pts == expected_pts

    @pytest.mark.parametrize("start", range(5))
    @pytest.mark.parametrize("offset", range(1, 4))
    def test_read_partial_video_pts_unit_sec(self, start, offset):
        with temp_video(10, 300, 300, 5, lossless=True) as (f_name, data):
            pts, _ = io.read_video_timestamps(f_name, pts_unit="sec")

            lv, _, _ = io.read_video(f_name, pts[start], pts[start + offset - 1], pts_unit="sec")
            s_data = data[start : (start + offset)]
            assert len(lv) == offset
            assert_equal(s_data, lv)

            with av.open(f_name) as container:
                stream = container.streams[0]
                lv, _, _ = io.read_video(
                    f_name, int(pts[4] * (1.0 / stream.time_base) + 1) * stream.time_base, pts[7], pts_unit="sec"
                )
            if get_video_backend() == "pyav":
                # for "video_reader" backend, we don't decode the closest early frame
                # when the given start pts is not matching any frame pts
                assert len(lv) == 4
                assert_equal(data[4:8], lv)

    def test_read_video_corrupted_file(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            f.write(b"This is not an mpg4 file")
            video, audio, info = io.read_video(f.name)
            assert isinstance(video, torch.Tensor)
            assert isinstance(audio, torch.Tensor)
            assert video.numel() == 0
            assert audio.numel() == 0
            assert info == {}

    def test_read_video_timestamps_corrupted_file(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            f.write(b"This is not an mpg4 file")
            video_pts, video_fps = io.read_video_timestamps(f.name)
            assert video_pts == []
            assert video_fps is None

    @pytest.mark.skip(reason="Temporarily disabled due to new pyav")
    def test_read_video_partially_corrupted_file(self):
        with temp_video(5, 4, 4, 5, lossless=True) as (f_name, data):
            with open(f_name, "r+b") as f:
                size = os.path.getsize(f_name)
                bytes_to_overwrite = size // 10
                # seek to the middle of the file
                f.seek(5 * bytes_to_overwrite)
                # corrupt 10% of the file from the middle
                f.write(b"\xff" * bytes_to_overwrite)
            # this exercises the container.decode assertion check
            video, audio, info = io.read_video(f.name, pts_unit="sec")
            # check that size is not equal to 5, but 3
            # TODO fix this
            if get_video_backend() == "pyav":
                assert len(video) == 3
            else:
                assert len(video) == 4
            # but the valid decoded content is still correct
            assert_equal(video[:3], data[:3])
            # and the last few frames are wrong
            with pytest.raises(AssertionError):
                assert_equal(video, data)

    @pytest.mark.skipif(sys.platform == "win32", reason="temporarily disabled on Windows")
    def test_write_video_with_audio(self, tmpdir):
        f_name = os.path.join(VIDEO_DIR, "R6llTwEh07w.mp4")
        video_tensor, audio_tensor, info = io.read_video(f_name, pts_unit="sec")

        out_f_name = os.path.join(tmpdir, "testing.mp4")
        io.video.write_video(
            out_f_name,
            video_tensor,
            round(info["video_fps"]),
            video_codec="libx264rgb",
            options={"crf": "0"},
            audio_array=audio_tensor,
            audio_fps=info["audio_fps"],
            audio_codec="aac",
        )

        out_video_tensor, out_audio_tensor, out_info = io.read_video(out_f_name, pts_unit="sec")

        assert info["video_fps"] == out_info["video_fps"]
        assert_equal(video_tensor, out_video_tensor)

        audio_stream = av.open(f_name).streams.audio[0]
        out_audio_stream = av.open(out_f_name).streams.audio[0]

        assert info["audio_fps"] == out_info["audio_fps"]
        assert audio_stream.rate == out_audio_stream.rate
        assert pytest.approx(out_audio_stream.frames, rel=0.0, abs=1) == audio_stream.frames
        assert audio_stream.frame_size == out_audio_stream.frame_size

    # TODO add tests for audio


if __name__ == "__main__":
    pytest.main(__file__)
