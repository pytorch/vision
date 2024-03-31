import pytest
import torch
from common_utils import assert_equal, get_list_of_videos
from torchvision import io
from torchvision.datasets.video_utils import unfold, VideoClips


class TestVideo:
    def test_unfold(self):
        a = torch.arange(7)

        r = unfold(a, 3, 3, 1)
        expected = torch.tensor(
            [
                [0, 1, 2],
                [3, 4, 5],
            ]
        )
        assert_equal(r, expected)

        r = unfold(a, 3, 2, 1)
        expected = torch.tensor([[0, 1, 2], [2, 3, 4], [4, 5, 6]])
        assert_equal(r, expected)

        r = unfold(a, 3, 2, 2)
        expected = torch.tensor(
            [
                [0, 2, 4],
                [2, 4, 6],
            ]
        )
        assert_equal(r, expected)

    @pytest.mark.skipif(not io.video._av_available(), reason="this test requires av")
    def test_video_clips(self, tmpdir):
        video_list = get_list_of_videos(tmpdir, num_videos=3)
        video_clips = VideoClips(video_list, 5, 5, num_workers=2)
        assert video_clips.num_clips() == 1 + 2 + 3
        for i, (v_idx, c_idx) in enumerate([(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)]):
            video_idx, clip_idx = video_clips.get_clip_location(i)
            assert video_idx == v_idx
            assert clip_idx == c_idx

        video_clips = VideoClips(video_list, 6, 6)
        assert video_clips.num_clips() == 0 + 1 + 2
        for i, (v_idx, c_idx) in enumerate([(1, 0), (2, 0), (2, 1)]):
            video_idx, clip_idx = video_clips.get_clip_location(i)
            assert video_idx == v_idx
            assert clip_idx == c_idx

        video_clips = VideoClips(video_list, 6, 1)
        assert video_clips.num_clips() == 0 + (10 - 6 + 1) + (15 - 6 + 1)
        for i, v_idx, c_idx in [(0, 1, 0), (4, 1, 4), (5, 2, 0), (6, 2, 1)]:
            video_idx, clip_idx = video_clips.get_clip_location(i)
            assert video_idx == v_idx
            assert clip_idx == c_idx

    @pytest.mark.skipif(not io.video._av_available(), reason="this test requires av")
    def test_video_clips_custom_fps(self, tmpdir):
        video_list = get_list_of_videos(tmpdir, num_videos=3, sizes=[12, 12, 12], fps=[3, 4, 6])
        num_frames = 4
        for fps in [1, 3, 4, 10]:
            video_clips = VideoClips(video_list, num_frames, num_frames, fps)
            for i in range(video_clips.num_clips()):
                video, audio, info, video_idx = video_clips.get_clip(i)
                assert video.shape[0] == num_frames
                assert info["video_fps"] == fps
                # TODO add tests checking that the content is right

    def test_compute_clips_for_video(self):
        video_pts = torch.arange(30)
        # case 1: single clip
        num_frames = 13
        orig_fps = 30
        duration = float(len(video_pts)) / orig_fps
        new_fps = 13
        clips, idxs = VideoClips.compute_clips_for_video(video_pts, num_frames, num_frames, orig_fps, new_fps)
        resampled_idxs = VideoClips._resample_video_idx(int(duration * new_fps), orig_fps, new_fps)
        assert len(clips) == 1
        assert_equal(clips, idxs)
        assert_equal(idxs[0], resampled_idxs)

        # case 2: all frames appear only once
        num_frames = 4
        orig_fps = 30
        duration = float(len(video_pts)) / orig_fps
        new_fps = 12
        clips, idxs = VideoClips.compute_clips_for_video(video_pts, num_frames, num_frames, orig_fps, new_fps)
        resampled_idxs = VideoClips._resample_video_idx(int(duration * new_fps), orig_fps, new_fps)
        assert len(clips) == 3
        assert_equal(clips, idxs)
        assert_equal(idxs.flatten(), resampled_idxs)

        # case 3: frames aren't enough for a clip
        num_frames = 32
        orig_fps = 30
        new_fps = 13
        with pytest.warns(UserWarning):
            clips, idxs = VideoClips.compute_clips_for_video(video_pts, num_frames, num_frames, orig_fps, new_fps)
        assert len(clips) == 0
        assert len(idxs) == 0


if __name__ == "__main__":
    pytest.main([__file__])
