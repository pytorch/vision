import os
import tempfile
import torch
import torchvision.io as io
import unittest


class Tester(unittest.TestCase):
    
    def test_write_read_video(self):
        with tempfile.NamedTemporaryFile(suffix='.mp4') as f:
            data = torch.randint(0, 255, (10, 300, 300, 3), dtype=torch.uint8)
            io.write_video(f.name, data, fps=5)

            lv, _ = io.read_video(f.name)
            print((data.float() - lv.float()).abs().max())



if __name__ == '__main__':
    unittest.main()
