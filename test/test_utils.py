import numpy as np
import os
import sys
import tempfile
import torch
import torchvision.utils as utils
import unittest
from io import BytesIO
import torchvision.transforms.functional as F
from PIL import Image


class Tester(unittest.TestCase):

    def test_make_grid_not_inplace(self):
        t = torch.rand(5, 3, 10, 10)
        t_clone = t.clone()

        utils.make_grid(t, normalize=False)
        self.assertTrue(torch.equal(t, t_clone), 'make_grid modified tensor in-place')

        utils.make_grid(t, normalize=True, scale_each=False)
        self.assertTrue(torch.equal(t, t_clone), 'make_grid modified tensor in-place')

        utils.make_grid(t, normalize=True, scale_each=True)
        self.assertTrue(torch.equal(t, t_clone), 'make_grid modified tensor in-place')

    def test_normalize_in_make_grid(self):
        t = torch.rand(5, 3, 10, 10) * 255
        norm_max = torch.tensor(1.0)
        norm_min = torch.tensor(0.0)

        grid = utils.make_grid(t, normalize=True)
        grid_max = torch.max(grid)
        grid_min = torch.min(grid)

        # Rounding the result to one decimal for comparison
        n_digits = 1
        rounded_grid_max = torch.round(grid_max * 10 ** n_digits) / (10 ** n_digits)
        rounded_grid_min = torch.round(grid_min * 10 ** n_digits) / (10 ** n_digits)

        self.assertTrue(torch.equal(norm_max, rounded_grid_max), 'Normalized max is not equal to 1')
        self.assertTrue(torch.equal(norm_min, rounded_grid_min), 'Normalized min is not equal to 0')

    @unittest.skipIf('win' in sys.platform, 'temporarily disabled on Windows')
    def test_save_image(self):
        with tempfile.NamedTemporaryFile(suffix='.png') as f:
            t = torch.rand(2, 3, 64, 64)
            utils.save_image(t, f.name)
            self.assertTrue(os.path.exists(f.name), 'The image is not present after save')

    @unittest.skipIf('win' in sys.platform, 'temporarily disabled on Windows')
    def test_save_image_single_pixel(self):
        with tempfile.NamedTemporaryFile(suffix='.png') as f:
            t = torch.rand(1, 3, 1, 1)
            utils.save_image(t, f.name)
            self.assertTrue(os.path.exists(f.name), 'The pixel image is not present after save')

    @unittest.skipIf('win' in sys.platform, 'temporarily disabled on Windows')
    def test_save_image_file_object(self):
        with tempfile.NamedTemporaryFile(suffix='.png') as f:
            t = torch.rand(2, 3, 64, 64)
            utils.save_image(t, f.name)
            img_orig = Image.open(f.name)
            fp = BytesIO()
            utils.save_image(t, fp, format='png')
            img_bytes = Image.open(fp)
            self.assertTrue(torch.equal(F.to_tensor(img_orig), F.to_tensor(img_bytes)),
                            'Image not stored in file object')

    @unittest.skipIf('win' in sys.platform, 'temporarily disabled on Windows')
    def test_save_image_single_pixel_file_object(self):
        with tempfile.NamedTemporaryFile(suffix='.png') as f:
            t = torch.rand(1, 3, 1, 1)
            utils.save_image(t, f.name)
            img_orig = Image.open(f.name)
            fp = BytesIO()
            utils.save_image(t, fp, format='png')
            img_bytes = Image.open(fp)
            self.assertTrue(torch.equal(F.to_tensor(img_orig), F.to_tensor(img_bytes)),
                            'Pixel Image not stored in file object')

    def test_draw_boxes(self):
        img = torch.full((3, 100, 100), 255, dtype=torch.uint8)
        boxes = torch.tensor([[0, 0, 20, 20], [0, 0, 0, 0],
                             [10, 15, 30, 35], [23, 35, 93, 95]], dtype=torch.float)
        labels = ["a", "b", "c", "d"]
        colors = ["green", "#FF00FF", (0, 255, 0), "red"]
        result = utils.draw_bounding_boxes(img, boxes, labels=labels, colors=colors)

        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "fakedata", "draw_boxes_util.png")
        if not os.path.exists(path):
            res = Image.fromarray(result.permute(1, 2, 0).contiguous().numpy())
            res.save(path)

        expected = torch.as_tensor(np.array(Image.open(path))).permute(2, 0, 1)
        self.assertTrue(torch.equal(result, expected))


if __name__ == '__main__':
    unittest.main()
