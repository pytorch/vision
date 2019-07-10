import contextlib
import os
import torch
import unittest

from torchvision import io
from torchvision.datasets.video_utils import VideoClips, unfold, RandomClipSampler

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
            data = torch.randint(0, 255, (size, 300, 400, 3), dtype=torch.uint8)
            name = os.path.join(tmp_dir, "{}.mp4".format(i))
            names.append(name)
            io.write_video(name, data, fps=f)

        yield names


class Tester(unittest.TestCase):

    def test_unfold(self):
        a = torch.arange(7)

        r = unfold(a, 3, 3, 1)
        expected = torch.tensor([
            [0, 1, 2],
            [3, 4, 5],
        ])
        self.assertTrue(r.equal(expected))

        r = unfold(a, 3, 2, 1)
        expected = torch.tensor([
            [0, 1, 2],
            [2, 3, 4],
            [4, 5, 6]
        ])
        self.assertTrue(r.equal(expected))

        r = unfold(a, 3, 2, 2)
        expected = torch.tensor([
            [0, 2, 4],
            [2, 4, 6],
        ])
        self.assertTrue(r.equal(expected))

    def test_video_clips(self):
        with get_list_of_videos(num_videos=3) as video_list:
            video_clips = VideoClips(video_list, 5, 5)
            self.assertEqual(video_clips.num_clips(), 1 + 2 + 3)
            for i, (v_idx, c_idx) in enumerate([(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)]):
                video_idx, clip_idx = video_clips.get_clip_location(i)
                self.assertEqual(video_idx, v_idx)
                self.assertEqual(clip_idx, c_idx)

            video_clips = VideoClips(video_list, 6, 6)
            self.assertEqual(video_clips.num_clips(), 0 + 1 + 2)
            for i, (v_idx, c_idx) in enumerate([(1, 0), (2, 0), (2, 1)]):
                video_idx, clip_idx = video_clips.get_clip_location(i)
                self.assertEqual(video_idx, v_idx)
                self.assertEqual(clip_idx, c_idx)

            video_clips = VideoClips(video_list, 6, 1)
            self.assertEqual(video_clips.num_clips(), 0 + (10 - 6 + 1) + (15 - 6 + 1))
            for i, v_idx, c_idx in [(0, 1, 0), (4, 1, 4), (5, 2, 0), (6, 2, 1)]:
                video_idx, clip_idx = video_clips.get_clip_location(i)
                self.assertEqual(video_idx, v_idx)
                self.assertEqual(clip_idx, c_idx)

    def test_video_sampler(self):
        with get_list_of_videos(num_videos=3, sizes=[25, 25, 25]) as video_list:
            video_clips = VideoClips(video_list, 5, 5)
            sampler = RandomClipSampler(video_clips, 3)
            self.assertEqual(len(sampler), 3 * 3)
            indices = torch.tensor(list(iter(sampler)))
            videos = indices // 5
            v_idxs, count = torch.unique(videos, return_counts=True)
            self.assertTrue(v_idxs.equal(torch.tensor([0, 1, 2])))
            self.assertTrue(count.equal(torch.tensor([3, 3, 3])))

    def test_video_sampler_unequal(self):
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

    def test_video_clips_custom_fps(self):
        with get_list_of_videos(num_videos=3, sizes=[12, 12, 12], fps=[3, 4, 6]) as video_list:
            num_frames = 4
            for fps in [1, 3, 4, 10]:
                video_clips = VideoClips(video_list, num_frames, num_frames, fps)
                for i in range(video_clips.num_clips()):
                    video, audio, info, video_idx = video_clips.get_clip(i)
                    self.assertEqual(video.shape[0], num_frames)
                    self.assertEqual(info["video_fps"], fps)
                    # TODO add tests checking that the content is right


if __name__ == '__main__':
    unittest.main()
