import contextlib
import sys
import os
import torch
import unittest

from torchvision import io
from torchvision.datasets.samplers import (
    DistributedSampler,
    RandomClipSampler,
    UniformClipSampler,
)
from torchvision.datasets.video_utils import VideoClips, unfold
from torchvision import get_video_backend

from common_utils import get_tmp_dir


@contextlib.contextmanager
def get_list_of_videos(num_videos=5, sizes=None, fps=None):
    with get_tmp_dir() as tmp_dir:
        names = []
        for i in range(num_videos):
            if sizes is None:
                size = 5 * (i + 1)
            else:
                size = sizes[i]
            if fps is None:
                f = 5
            else:
                f = fps[i]
            data = torch.randint(0, 256, (size, 300, 400, 3), dtype=torch.uint8)
            name = os.path.join(tmp_dir, "{}.mp4".format(i))
            names.append(name)
            io.write_video(name, data, fps=f)

        yield names


@unittest.skipIf(not io.video._av_available(), "this test requires av")
class Tester(unittest.TestCase):
    def test_random_clip_sampler(self):
        with get_list_of_videos(num_videos=3, sizes=[25, 25, 25]) as video_list:
            video_clips = VideoClips(video_list, 5, 5)
            sampler = RandomClipSampler(video_clips, 3)
            self.assertEqual(len(sampler), 3 * 3)
            indices = torch.tensor(list(iter(sampler)))
            videos = indices // 5
            v_idxs, count = torch.unique(videos, return_counts=True)
            self.assertTrue(v_idxs.equal(torch.tensor([0, 1, 2])))
            self.assertTrue(count.equal(torch.tensor([3, 3, 3])))

    def test_random_clip_sampler_unequal(self):
        with get_list_of_videos(num_videos=3, sizes=[10, 25, 25]) as video_list:
            video_clips = VideoClips(video_list, 5, 5)
            sampler = RandomClipSampler(video_clips, 3)
            self.assertEqual(len(sampler), 2 + 3 + 3)
            indices = list(iter(sampler))
            self.assertIn(0, indices)
            self.assertIn(1, indices)
            # remove elements of the first video, to simplify testing
            indices.remove(0)
            indices.remove(1)
            indices = torch.tensor(indices) - 2
            videos = indices // 5
            v_idxs, count = torch.unique(videos, return_counts=True)
            self.assertTrue(v_idxs.equal(torch.tensor([0, 1])))
            self.assertTrue(count.equal(torch.tensor([3, 3])))

    def test_uniform_clip_sampler(self):
        with get_list_of_videos(num_videos=3, sizes=[25, 25, 25]) as video_list:
            video_clips = VideoClips(video_list, 5, 5)
            sampler = UniformClipSampler(video_clips, 3)
            self.assertEqual(len(sampler), 3 * 3)
            indices = torch.tensor(list(iter(sampler)))
            videos = indices // 5
            v_idxs, count = torch.unique(videos, return_counts=True)
            self.assertTrue(v_idxs.equal(torch.tensor([0, 1, 2])))
            self.assertTrue(count.equal(torch.tensor([3, 3, 3])))
            self.assertTrue(indices.equal(torch.tensor([0, 2, 4, 5, 7, 9, 10, 12, 14])))

    def test_uniform_clip_sampler_insufficient_clips(self):
        with get_list_of_videos(num_videos=3, sizes=[10, 25, 25]) as video_list:
            video_clips = VideoClips(video_list, 5, 5)
            sampler = UniformClipSampler(video_clips, 3)
            self.assertEqual(len(sampler), 3 * 3)
            indices = torch.tensor(list(iter(sampler)))
            self.assertTrue(indices.equal(torch.tensor([0, 0, 1, 2, 4, 6, 7, 9, 11])))

    def test_distributed_sampler_and_uniform_clip_sampler(self):
        with get_list_of_videos(num_videos=3, sizes=[25, 25, 25]) as video_list:
            video_clips = VideoClips(video_list, 5, 5)
            clip_sampler = UniformClipSampler(video_clips, 3)

            distributed_sampler_rank0 = DistributedSampler(
                clip_sampler,
                num_replicas=2,
                rank=0,
                group_size=3,
            )
            indices = torch.tensor(list(iter(distributed_sampler_rank0)))
            self.assertEqual(len(distributed_sampler_rank0), 6)
            self.assertTrue(indices.equal(torch.tensor([0, 2, 4, 10, 12, 14])))

            distributed_sampler_rank1 = DistributedSampler(
                clip_sampler,
                num_replicas=2,
                rank=1,
                group_size=3,
            )
            indices = torch.tensor(list(iter(distributed_sampler_rank1)))
            self.assertEqual(len(distributed_sampler_rank1), 6)
            self.assertTrue(indices.equal(torch.tensor([5, 7, 9, 0, 2, 4])))


if __name__ == '__main__':
    unittest.main()
