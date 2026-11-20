import sys

import pytest
import torch
from common_utils import assert_equal, get_list_of_videos
from torchvision.datasets.samplers import DistributedSampler, RandomClipSampler, UniformClipSampler
from torchvision.datasets.video_utils import VideoClips

try:
    import torchcodec  # noqa: F401

    _torchcodec_available = True
except ImportError:
    _torchcodec_available = False


@pytest.mark.skipif(
    not (_torchcodec_available and sys.platform == "linux"),
    reason="this test requires torchcodec (linux only)",
)
class TestDatasetsSamplers:
    def test_random_clip_sampler(self, tmpdir):
        video_list = get_list_of_videos(tmpdir, num_videos=3, sizes=[25, 25, 25])
        video_clips = VideoClips(video_list, 5, 5)
        sampler = RandomClipSampler(video_clips, 3)
        assert len(sampler) == 3 * 3
        indices = torch.tensor(list(iter(sampler)))
        videos = torch.div(indices, 5, rounding_mode="floor")
        v_idxs, count = torch.unique(videos, return_counts=True)
        assert_equal(v_idxs, torch.tensor([0, 1, 2]))
        assert_equal(count, torch.tensor([3, 3, 3]))

    def test_random_clip_sampler_unequal(self, tmpdir):
        video_list = get_list_of_videos(tmpdir, num_videos=3, sizes=[10, 25, 25])
        video_clips = VideoClips(video_list, 5, 5)
        sampler = RandomClipSampler(video_clips, 3)
        assert len(sampler) == 2 + 3 + 3
        indices = list(iter(sampler))
        assert 0 in indices
        assert 1 in indices
        # remove elements of the first video, to simplify testing
        indices.remove(0)
        indices.remove(1)
        indices = torch.tensor(indices) - 2
        videos = torch.div(indices, 5, rounding_mode="floor")
        v_idxs, count = torch.unique(videos, return_counts=True)
        assert_equal(v_idxs, torch.tensor([0, 1]))
        assert_equal(count, torch.tensor([3, 3]))

    def test_uniform_clip_sampler(self, tmpdir):
        video_list = get_list_of_videos(tmpdir, num_videos=3, sizes=[25, 25, 25])
        video_clips = VideoClips(video_list, 5, 5)
        sampler = UniformClipSampler(video_clips, 3)
        assert len(sampler) == 3 * 3
        indices = torch.tensor(list(iter(sampler)))
        videos = torch.div(indices, 5, rounding_mode="floor")
        v_idxs, count = torch.unique(videos, return_counts=True)
        assert_equal(v_idxs, torch.tensor([0, 1, 2]))
        assert_equal(count, torch.tensor([3, 3, 3]))
        assert_equal(indices, torch.tensor([0, 2, 4, 5, 7, 9, 10, 12, 14]))

    def test_uniform_clip_sampler_insufficient_clips(self, tmpdir):
        video_list = get_list_of_videos(tmpdir, num_videos=3, sizes=[10, 25, 25])
        video_clips = VideoClips(video_list, 5, 5)
        sampler = UniformClipSampler(video_clips, 3)
        assert len(sampler) == 3 * 3
        indices = torch.tensor(list(iter(sampler)))
        assert_equal(indices, torch.tensor([0, 0, 1, 2, 4, 6, 7, 9, 11]))

    def test_distributed_sampler_and_uniform_clip_sampler(self, tmpdir):
        video_list = get_list_of_videos(tmpdir, num_videos=3, sizes=[25, 25, 25])
        video_clips = VideoClips(video_list, 5, 5)
        clip_sampler = UniformClipSampler(video_clips, 3)

        distributed_sampler_rank0 = DistributedSampler(
            clip_sampler,
            num_replicas=2,
            rank=0,
            group_size=3,
        )
        indices = torch.tensor(list(iter(distributed_sampler_rank0)))
        assert len(distributed_sampler_rank0) == 6
        assert_equal(indices, torch.tensor([0, 2, 4, 10, 12, 14]))

        distributed_sampler_rank1 = DistributedSampler(
            clip_sampler,
            num_replicas=2,
            rank=1,
            group_size=3,
        )
        indices = torch.tensor(list(iter(distributed_sampler_rank1)))
        assert len(distributed_sampler_rank1) == 6
        assert_equal(indices, torch.tensor([5, 7, 9, 0, 2, 4]))


@pytest.mark.filterwarnings("ignore:There aren't enough frames in the current video:UserWarning")
@pytest.mark.parametrize("sampler_cls", [UniformClipSampler, RandomClipSampler])
class TestClipSamplersMetadata:
    def make_video_clips(self, frame_counts):
        video_paths = [f"video_{i}.mp4" for i in range(len(frame_counts))]
        return VideoClips(
            video_paths,
            clip_length_in_frames=5,
            frames_between_clips=5,
            _precomputed_metadata={
                "video_paths": video_paths,
                "video_pts": [torch.arange(count) for count in frame_counts],
                "video_fps": [30.0] * len(frame_counts),
            },
        )

    @pytest.mark.parametrize("frame_counts", [[], [0], [4], [0, 3, 4]])
    def test_no_clips(self, sampler_cls, frame_counts):
        sampler = sampler_cls(self.make_video_clips(frame_counts), 3)
        assert len(sampler) == 0
        assert list(sampler) == []

        for rank in range(2):
            distributed_sampler = DistributedSampler(sampler, num_replicas=2, rank=rank, group_size=3)
            assert len(distributed_sampler) == 0
            assert list(distributed_sampler) == []

    @pytest.mark.parametrize(
        "frame_counts, expected_uniform",
        [([5], [0, 0, 0]), ([10, 25], [0, 0, 1, 2, 4, 6]), ([0, 10, 0, 25, 0], [0, 0, 1, 2, 4, 6])],
    )
    def test_nonempty_clips(self, sampler_cls, frame_counts, expected_uniform):
        video_clips = self.make_video_clips(frame_counts)
        sampler = sampler_cls(video_clips, 3)
        indices = list(sampler)
        assert len(indices) == len(sampler)
        if sampler_cls is UniformClipSampler:
            assert indices == expected_uniform
        else:
            assert len(set(indices)) == len(indices)
            offset = 0
            for clips in video_clips.clips:
                assert sum(offset <= index < offset + len(clips) for index in indices) == min(len(clips), 3)
                offset += len(clips)

    def test_zero_requested_clips(self, sampler_cls):
        sampler = sampler_cls(self.make_video_clips([10, 25]), 0)
        assert len(sampler) == 0
        assert list(sampler) == []


if __name__ == "__main__":
    pytest.main([__file__])
