import contextlib
import os
import torch
import unittest

from torchvision import io
from torchvision.datasets.video_utils import VideoClips, unfold

from common_utils import get_tmp_dir


@contextlib.contextmanager
def get_list_of_videos(num_videos=5):
    with get_tmp_dir() as tmp_dir:
        names = []
        for i in range(num_videos):
            data = torch.randint(0, 255, (5 * (i + 1), 300, 400, 3), dtype=torch.uint8)
            name = os.path.join(tmp_dir, "{}.mp4".format(i))
            names.append(name)
            io.write_video(name, data, fps=5)

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


if __name__ == '__main__':
    unittest.main()
