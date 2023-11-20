import collections
import os
import urllib

import pytest
import torch
import torchvision
from pytest import approx
from torchvision.datasets.utils import download_url
from torchvision.io import _HAS_VIDEO_OPT, VideoReader


# WARNING: these tests have been skipped forever on the CI because the video ops
# are never properly available. This is bad, but things have been in a terrible
# state for a long time already as we write this comment, and we'll hopefully be
# able to get rid of this all soon.


try:
    import av

    # Do a version test too
    torchvision.io.video._check_av_available()
except ImportError:
    av = None


VIDEO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "videos")

CheckerConfig = ["duration", "video_fps", "audio_sample_rate"]
GroundTruth = collections.namedtuple("GroundTruth", " ".join(CheckerConfig))


def backends():
    backends_ = ["video_reader"]
    if av is not None:
        backends_.append("pyav")
    return backends_


def fate(name, path="."):
    """Download and return a path to a sample from the FFmpeg test suite.
    See the `FFmpeg Automated Test Environment <https://www.ffmpeg.org/fate.html>`_
    """

    file_name = name.split("/")[1]
    download_url("http://fate.ffmpeg.org/fate-suite/" + name, path, file_name)
    return os.path.join(path, file_name)


test_videos = {
    "RATRACE_wave_f_nm_np1_fr_goo_37.avi": GroundTruth(duration=2.0, video_fps=30.0, audio_sample_rate=None),
    "SchoolRulesHowTheyHelpUs_wave_f_nm_np1_ba_med_0.avi": GroundTruth(
        duration=2.0, video_fps=30.0, audio_sample_rate=None
    ),
    "TrumanShow_wave_f_nm_np1_fr_med_26.avi": GroundTruth(duration=2.0, video_fps=30.0, audio_sample_rate=None),
    "v_SoccerJuggling_g23_c01.avi": GroundTruth(duration=8.0, video_fps=29.97, audio_sample_rate=None),
    "v_SoccerJuggling_g24_c01.avi": GroundTruth(duration=8.0, video_fps=29.97, audio_sample_rate=None),
    "R6llTwEh07w.mp4": GroundTruth(duration=10.0, video_fps=30.0, audio_sample_rate=44100),
    "SOX5yA1l24A.mp4": GroundTruth(duration=11.0, video_fps=29.97, audio_sample_rate=48000),
    "WUzgd7C1pWA.mp4": GroundTruth(duration=11.0, video_fps=29.97, audio_sample_rate=48000),
}


@pytest.mark.skipif(_HAS_VIDEO_OPT is False, reason="Didn't compile with ffmpeg")
class TestVideoApi:
    @pytest.mark.skipif(av is None, reason="PyAV unavailable")
    @pytest.mark.parametrize("test_video", test_videos.keys())
    @pytest.mark.parametrize("backend", backends())
    def test_frame_reading(self, test_video, backend):
        torchvision.set_video_backend(backend)
        full_path = os.path.join(VIDEO_DIR, test_video)
        with av.open(full_path) as av_reader:
            if av_reader.streams.video:
                av_frames, vr_frames = [], []
                av_pts, vr_pts = [], []
                # get av frames
                for av_frame in av_reader.decode(av_reader.streams.video[0]):
                    av_frames.append(torch.tensor(av_frame.to_rgb().to_ndarray()).permute(2, 0, 1))
                    av_pts.append(av_frame.pts * av_frame.time_base)

                # get vr frames
                video_reader = VideoReader(full_path, "video")
                for vr_frame in video_reader:
                    vr_frames.append(vr_frame["data"])
                    vr_pts.append(vr_frame["pts"])

                # same number of frames
                assert len(vr_frames) == len(av_frames)
                assert len(vr_pts) == len(av_pts)

                # compare the frames and ptss
                for i in range(len(vr_frames)):
                    assert float(av_pts[i]) == approx(vr_pts[i], abs=0.1)

                    mean_delta = torch.mean(torch.abs(av_frames[i].float() - vr_frames[i].float()))
                    # on average the difference is very small and caused
                    # by decoding (around 1%)
                    # TODO: asses empirically how to set this? atm it's 1%
                    # averaged over all frames
                    assert mean_delta.item() < 2.55

                del vr_frames, av_frames, vr_pts, av_pts

        # test audio reading compared to PYAV
        with av.open(full_path) as av_reader:
            if av_reader.streams.audio:
                av_frames, vr_frames = [], []
                av_pts, vr_pts = [], []
                # get av frames
                for av_frame in av_reader.decode(av_reader.streams.audio[0]):
                    av_frames.append(torch.tensor(av_frame.to_ndarray()).permute(1, 0))
                    av_pts.append(av_frame.pts * av_frame.time_base)
                av_reader.close()

                # get vr frames
                video_reader = VideoReader(full_path, "audio")
                for vr_frame in video_reader:
                    vr_frames.append(vr_frame["data"])
                    vr_pts.append(vr_frame["pts"])

                # same number of frames
                assert len(vr_frames) == len(av_frames)
                assert len(vr_pts) == len(av_pts)

                # compare the frames and ptss
                for i in range(len(vr_frames)):
                    assert float(av_pts[i]) == approx(vr_pts[i], abs=0.1)
                    max_delta = torch.max(torch.abs(av_frames[i].float() - vr_frames[i].float()))
                    # we assure that there is never more than 1% difference in signal
                    assert max_delta.item() < 0.001

    @pytest.mark.parametrize("stream", ["video", "audio"])
    @pytest.mark.parametrize("test_video", test_videos.keys())
    @pytest.mark.parametrize("backend", backends())
    def test_frame_reading_mem_vs_file(self, test_video, stream, backend):
        torchvision.set_video_backend(backend)
        full_path = os.path.join(VIDEO_DIR, test_video)

        reader = VideoReader(full_path)
        reader_md = reader.get_metadata()

        if stream in reader_md:
            # Test video reading from file vs from memory
            vr_frames, vr_frames_mem = [], []
            vr_pts, vr_pts_mem = [], []
            # get vr frames
            video_reader = VideoReader(full_path, stream)
            for vr_frame in video_reader:
                vr_frames.append(vr_frame["data"])
                vr_pts.append(vr_frame["pts"])

            # get vr frames = read from memory
            f = open(full_path, "rb")
            fbytes = f.read()
            f.close()
            video_reader_from_mem = VideoReader(fbytes, stream)

            for vr_frame_from_mem in video_reader_from_mem:
                vr_frames_mem.append(vr_frame_from_mem["data"])
                vr_pts_mem.append(vr_frame_from_mem["pts"])

            # same number of frames
            assert len(vr_frames) == len(vr_frames_mem)
            assert len(vr_pts) == len(vr_pts_mem)

            # compare the frames and ptss
            for i in range(len(vr_frames)):
                assert vr_pts[i] == vr_pts_mem[i]
                mean_delta = torch.mean(torch.abs(vr_frames[i].float() - vr_frames_mem[i].float()))
                # on average the difference is very small and caused
                # by decoding (around 1%)
                # TODO: asses empirically how to set this? atm it's 1%
                # averaged over all frames
                assert mean_delta.item() < 2.55

            del vr_frames, vr_pts, vr_frames_mem, vr_pts_mem
        else:
            del reader, reader_md

    @pytest.mark.parametrize("test_video,config", test_videos.items())
    @pytest.mark.parametrize("backend", backends())
    def test_metadata(self, test_video, config, backend):
        """
        Test that the metadata returned via pyav corresponds to the one returned
        by the new video decoder API
        """
        torchvision.set_video_backend(backend)
        full_path = os.path.join(VIDEO_DIR, test_video)
        reader = VideoReader(full_path, "video")
        reader_md = reader.get_metadata()
        assert config.video_fps == approx(reader_md["video"]["fps"][0], abs=0.0001)
        assert config.duration == approx(reader_md["video"]["duration"][0], abs=0.5)

    @pytest.mark.parametrize("test_video", test_videos.keys())
    @pytest.mark.parametrize("backend", backends())
    def test_seek_start(self, test_video, backend):
        torchvision.set_video_backend(backend)
        full_path = os.path.join(VIDEO_DIR, test_video)
        video_reader = VideoReader(full_path, "video")
        num_frames = 0
        for _ in video_reader:
            num_frames += 1

        # now seek the container to 0 and do it again
        # It's often that starting seek can be inprecise
        # this way and it doesn't start at 0
        video_reader.seek(0)
        start_num_frames = 0
        for _ in video_reader:
            start_num_frames += 1

        assert start_num_frames == num_frames

        # now seek the container to < 0 to check for unexpected behaviour
        video_reader.seek(-1)
        start_num_frames = 0
        for _ in video_reader:
            start_num_frames += 1

        assert start_num_frames == num_frames

    @pytest.mark.parametrize("test_video", test_videos.keys())
    @pytest.mark.parametrize("backend", ["video_reader"])
    def test_accurateseek_middle(self, test_video, backend):
        torchvision.set_video_backend(backend)
        full_path = os.path.join(VIDEO_DIR, test_video)
        stream = "video"
        video_reader = VideoReader(full_path, stream)
        md = video_reader.get_metadata()
        duration = md[stream]["duration"][0]
        if duration is not None:
            num_frames = 0
            for _ in video_reader:
                num_frames += 1

            video_reader.seek(duration / 2)
            middle_num_frames = 0
            for _ in video_reader:
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

    @pytest.mark.skipif(av is None, reason="PyAV unavailable")
    @pytest.mark.parametrize("test_video,config", test_videos.items())
    @pytest.mark.parametrize("backend", backends())
    def test_keyframe_reading(self, test_video, config, backend):
        torchvision.set_video_backend(backend)
        full_path = os.path.join(VIDEO_DIR, test_video)

        av_reader = av.open(full_path)
        # reduce streams to only keyframes
        av_stream = av_reader.streams.video[0]
        av_stream.codec_context.skip_frame = "NONKEY"

        av_keyframes = []
        vr_keyframes = []
        if av_reader.streams.video:

            # get all keyframes using pyav. Then, seek randomly into video reader
            # and assert that all the returned values are in AV_KEYFRAMES

            for av_frame in av_reader.decode(av_stream):
                av_keyframes.append(float(av_frame.pts * av_frame.time_base))

        if len(av_keyframes) > 1:
            video_reader = VideoReader(full_path, "video")
            for i in range(1, len(av_keyframes)):
                seek_val = (av_keyframes[i] + av_keyframes[i - 1]) / 2
                data = next(video_reader.seek(seek_val, True))
                vr_keyframes.append(data["pts"])

            data = next(video_reader.seek(config.duration, True))
            vr_keyframes.append(data["pts"])

            assert len(av_keyframes) == len(vr_keyframes)
            # NOTE: this video gets different keyframe with different
            # loaders (0.333 pyav, 0.666 for us)
            if test_video != "TrumanShow_wave_f_nm_np1_fr_med_26.avi":
                for i in range(len(av_keyframes)):
                    assert av_keyframes[i] == approx(vr_keyframes[i], rel=0.001)

    def test_src(self):
        with pytest.raises(ValueError, match="src cannot be empty"):
            VideoReader(src="")
        with pytest.raises(ValueError, match="src must be either string"):
            VideoReader(src=2)
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            VideoReader(path="path")


if __name__ == "__main__":
    pytest.main([__file__])
