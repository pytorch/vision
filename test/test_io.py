import os
import tempfile
import torch
import torchvision.io as io
import unittest


class Tester(unittest.TestCase):

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

            # compression adds artifacts, thus we add a tolerance of
            # 5 in 0-255 range
            self.assertTrue((data.float() - lv.float()).abs().max() < 5)

    def test_read_timestamps(self):
        with tempfile.NamedTemporaryFile(suffix='.mp4') as f:
            data = self._create_video_frames(10, 300, 300)
            io.write_video(f.name, data, fps=5)

            lv = io.read_video_timestamps(f.name)
            print(lv)
            import av
            container = av.open(f.name)
            from IPython import embed; embed()



if __name__ == '__main__':
    unittest.main()
