import os
import tempfile
import torch
import torchvision.io as io
import unittest


class Tester(unittest.TestCase):
    # compression adds artifacts, thus we add a tolerance of
    # 5 in 0-255 range
    TOLERANCE = 5

    def _create_video_frames(self, num_frames, height, width):
        y, x = torch.meshgrid(torch.linspace(-2, 2, height), torch.linspace(-2, 2, width))
        data = []
        for i in range(num_frames):
            xc = float(i) / num_frames
            yc = 1 - float(i) / (2 * num_frames)
            d = torch.exp(-((x - xc) ** 2 + (y - yc) ** 2) / 2) * 255
            data.append(d.unsqueeze(2).repeat(1, 1, 3).byte())

        return torch.stack(data, 0)

    def test_write_read_video(self):
        with tempfile.NamedTemporaryFile(suffix='.mp4') as f:
            data = self._create_video_frames(10, 300, 300)
            io.write_video(f.name, data, fps=5)

            lv, _ = io.read_video(f.name)

            self.assertTrue((data.float() - lv.float()).abs().max() < self.TOLERANCE)

    def test_read_timestamps(self):
        with tempfile.NamedTemporaryFile(suffix='.mp4') as f:
            data = self._create_video_frames(10, 300, 300)
            io.write_video(f.name, data, fps=5)

            pts = io.read_video_timestamps(f.name)
            self.assertEqual(pts, [0, 2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432])

    def test_read_partial_video(self):
        with tempfile.NamedTemporaryFile(suffix='.mp4') as f:
            data = self._create_video_frames(10, 300, 300)
            io.write_video(f.name, data, fps=5)

            pts = io.read_video_timestamps(f.name)

            for start in range(5):
                for l in range(1, 4):
                    lv, _ = io.read_video(f.name, pts[start], pts[start + l - 1])
                    s_data = data[start:(start + l)]
                    self.assertEqual(len(lv), l)
                    self.assertTrue((s_data.float() - lv.float()).abs().max() < self.TOLERANCE)

            lv, _ = io.read_video(f.name, pts[4] + 1, pts[7])
            self.assertEqual(len(lv), 4)
            self.assertTrue((data[4:8].float() - lv.float()).abs().max() < self.TOLERANCE)

    # TODO add tests for audio


if __name__ == '__main__':
    unittest.main()
