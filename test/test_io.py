import os
import tempfile
import torch
import torchvision.io as io
import unittest


try:
    import av
except ImportError:
    av = None


class Tester(unittest.TestCase):
    # compression adds artifacts, thus we add a tolerance of
    # 6 in 0-255 range
    TOLERANCE = 6

    def _create_video_frames(self, num_frames, height, width):
        y, x = torch.meshgrid(torch.linspace(-2, 2, height), torch.linspace(-2, 2, width))
        data = []
        for i in range(num_frames):
            xc = float(i) / num_frames
            yc = 1 - float(i) / (2 * num_frames)
            d = torch.exp(-((x - xc) ** 2 + (y - yc) ** 2) / 2) * 255
            data.append(d.unsqueeze(2).repeat(1, 1, 3).byte())

        return torch.stack(data, 0)

    @unittest.skipIf(av is None, "PyAV unavailable")
    def test_write_read_video(self):
        with tempfile.NamedTemporaryFile(suffix='.mp4') as f:
            data = self._create_video_frames(10, 300, 300)
            io.write_video(f.name, data, fps=5)

            lv, _, info = io.read_video(f.name)

            self.assertTrue((data.float() - lv.float()).abs().max() < self.TOLERANCE)
            self.assertEqual(info["video_fps"], 5)

    @unittest.skipIf(av is None, "PyAV unavailable")
    def test_read_timestamps(self):
        with tempfile.NamedTemporaryFile(suffix='.mp4') as f:
            data = self._create_video_frames(10, 300, 300)
            io.write_video(f.name, data, fps=5)

            pts, _ = io.read_video_timestamps(f.name)

            # note: not all formats/codecs provide accurate information for computing the
            # timestamps. For the format that we use here, this information is available,
            # so we use it as a baseline
            container = av.open(f.name)
            stream = container.streams[0]
            pts_step = int(round(float(1 / (stream.average_rate * stream.time_base))))
            num_frames = int(round(float(stream.average_rate * stream.time_base * stream.duration)))
            expected_pts = [i * pts_step for i in range(num_frames)]

            self.assertEqual(pts, expected_pts)

    @unittest.skipIf(av is None, "PyAV unavailable")
    def test_read_partial_video(self):
        with tempfile.NamedTemporaryFile(suffix='.mp4') as f:
            data = self._create_video_frames(10, 300, 300)
            io.write_video(f.name, data, fps=5)

            pts, _ = io.read_video_timestamps(f.name)

            for start in range(5):
                for l in range(1, 4):
                    lv, _, _ = io.read_video(f.name, pts[start], pts[start + l - 1])
                    s_data = data[start:(start + l)]
                    self.assertEqual(len(lv), l)
                    self.assertTrue((s_data.float() - lv.float()).abs().max() < self.TOLERANCE)

            lv, _, _ = io.read_video(f.name, pts[4] + 1, pts[7])
            self.assertEqual(len(lv), 4)
            self.assertTrue((data[4:8].float() - lv.float()).abs().max() < self.TOLERANCE)

    # TODO add tests for audio


if __name__ == '__main__':
    unittest.main()
