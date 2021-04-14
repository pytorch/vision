import numpy as np
import os
import sys
import tempfile
import torch
import torchvision.utils as utils
import unittest
from io import BytesIO
import torchvision.transforms.functional as F
from PIL import Image, __version__ as PILLOW_VERSION


PILLOW_VERSION = tuple(int(x) for x in PILLOW_VERSION.split('.'))

boxes = torch.tensor([[0, 0, 20, 20], [0, 0, 0, 0],
                     [10, 15, 30, 35], [23, 35, 93, 95]], dtype=torch.float)

masks = torch.tensor([
    [
        [-2.2799, -2.2799, -2.2799, -2.2799, -2.2799],
        [5.0914, 5.0914, 5.0914, 5.0914, 5.0914],
        [-2.2799, -2.2799, -2.2799, -2.2799, -2.2799],
        [-2.2799, -2.2799, -2.2799, -2.2799, -2.2799],
        [-2.2799, -2.2799, -2.2799, -2.2799, -2.2799]
    ],
    [
        [5.0914, 5.0914, 5.0914, 5.0914, 5.0914],
        [-2.2799, -2.2799, -2.2799, -2.2799, -2.2799],
        [5.0914, 5.0914, 5.0914, 5.0914, 5.0914],
        [5.0914, 5.0914, 5.0914, 5.0914, 5.0914],
        [-1.4541, -1.4541, -1.4541, -1.4541, -1.4541]
    ],
    [
        [-1.4541, -1.4541, -1.4541, -1.4541, -1.4541],
        [-1.4541, -1.4541, -1.4541, -1.4541, -1.4541],
        [-1.4541, -1.4541, -1.4541, -1.4541, -1.4541],
        [-1.4541, -1.4541, -1.4541, -1.4541, -1.4541],
        [5.0914, 5.0914, 5.0914, 5.0914, 5.0914],
    ]
], dtype=torch.float)


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

    @unittest.skipIf(sys.platform in ('win32', 'cygwin'), 'temporarily disabled on Windows')
    def test_save_image(self):
        with tempfile.NamedTemporaryFile(suffix='.png') as f:
            t = torch.rand(2, 3, 64, 64)
            utils.save_image(t, f.name)
            self.assertTrue(os.path.exists(f.name), 'The image is not present after save')

    @unittest.skipIf(sys.platform in ('win32', 'cygwin'), 'temporarily disabled on Windows')
    def test_save_image_single_pixel(self):
        with tempfile.NamedTemporaryFile(suffix='.png') as f:
            t = torch.rand(1, 3, 1, 1)
            utils.save_image(t, f.name)
            self.assertTrue(os.path.exists(f.name), 'The pixel image is not present after save')

    @unittest.skipIf(sys.platform in ('win32', 'cygwin'), 'temporarily disabled on Windows')
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

    @unittest.skipIf(sys.platform in ('win32', 'cygwin'), 'temporarily disabled on Windows')
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
        img_cp = img.clone()
        boxes_cp = boxes.clone()
        labels = ["a", "b", "c", "d"]
        colors = ["green", "#FF00FF", (0, 255, 0), "red"]
        result = utils.draw_bounding_boxes(img, boxes, labels=labels, colors=colors, fill=True)

        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "fakedata", "draw_boxes_util.png")
        if not os.path.exists(path):
            res = Image.fromarray(result.permute(1, 2, 0).contiguous().numpy())
            res.save(path)

        if PILLOW_VERSION >= (8, 2):
            # The reference image is only valid for new PIL versions
            expected = torch.as_tensor(np.array(Image.open(path))).permute(2, 0, 1)
            self.assertTrue(torch.equal(result, expected))

        # Check if modification is not in place
        self.assertTrue(torch.all(torch.eq(boxes, boxes_cp)).item())
        self.assertTrue(torch.all(torch.eq(img, img_cp)).item())

    def test_draw_boxes_vanilla(self):
        img = torch.full((3, 100, 100), 0, dtype=torch.uint8)
        img_cp = img.clone()
        boxes_cp = boxes.clone()
        result = utils.draw_bounding_boxes(img, boxes, fill=False, width=7)

        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "fakedata", "draw_boxes_vanilla.png")
        if not os.path.exists(path):
            res = Image.fromarray(result.permute(1, 2, 0).contiguous().numpy())
            res.save(path)

        expected = torch.as_tensor(np.array(Image.open(path))).permute(2, 0, 1)
        self.assertTrue(torch.equal(result, expected))
        # Check if modification is not in place
        self.assertTrue(torch.all(torch.eq(boxes, boxes_cp)).item())
        self.assertTrue(torch.all(torch.eq(img, img_cp)).item())

    def test_draw_invalid_boxes(self):
        img_tp = ((1, 1, 1), (1, 2, 3))
        img_wrong1 = torch.full((3, 5, 5), 255, dtype=torch.float)
        img_wrong2 = torch.full((1, 3, 5, 5), 255, dtype=torch.uint8)
        boxes = torch.tensor([[0, 0, 20, 20], [0, 0, 0, 0],
                             [10, 15, 30, 35], [23, 35, 93, 95]], dtype=torch.float)
        self.assertRaises(TypeError, utils.draw_bounding_boxes, img_tp, boxes)
        self.assertRaises(ValueError, utils.draw_bounding_boxes, img_wrong1, boxes)
        self.assertRaises(ValueError, utils.draw_bounding_boxes, img_wrong2, boxes)

    def test_draw_segmentation_masks_colors(self):
        img = torch.full((3, 5, 5), 255, dtype=torch.uint8)
        img_cp = img.clone()
        masks_cp = masks.clone()
        colors = ["#FF00FF", (0, 255, 0), "red"]
        result = utils.draw_segmentation_masks(img, masks, colors=colors)

        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets",
                            "fakedata", "draw_segm_masks_colors_util.png")

        if not os.path.exists(path):
            res = Image.fromarray(result.permute(1, 2, 0).contiguous().numpy())
            res.save(path)

        expected = torch.as_tensor(np.array(Image.open(path))).permute(2, 0, 1)
        self.assertTrue(torch.equal(result, expected))
        # Check if modification is not in place
        self.assertTrue(torch.all(torch.eq(img, img_cp)).item())
        self.assertTrue(torch.all(torch.eq(masks, masks_cp)).item())

    def test_draw_segmentation_masks_no_colors(self):
        img = torch.full((3, 20, 20), 255, dtype=torch.uint8)
        img_cp = img.clone()
        masks_cp = masks.clone()
        result = utils.draw_segmentation_masks(img, masks, colors=None)

        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets",
                            "fakedata", "draw_segm_masks_no_colors_util.png")

        if not os.path.exists(path):
            res = Image.fromarray(result.permute(1, 2, 0).contiguous().numpy())
            res.save(path)

        expected = torch.as_tensor(np.array(Image.open(path))).permute(2, 0, 1)
        self.assertTrue(torch.equal(result, expected))
        # Check if modification is not in place
        self.assertTrue(torch.all(torch.eq(img, img_cp)).item())
        self.assertTrue(torch.all(torch.eq(masks, masks_cp)).item())

    def test_draw_invalid_masks(self):
        img_tp = ((1, 1, 1), (1, 2, 3))
        img_wrong1 = torch.full((3, 5, 5), 255, dtype=torch.float)
        img_wrong2 = torch.full((1, 3, 5, 5), 255, dtype=torch.uint8)
        img_wrong3 = torch.full((4, 5, 5), 255, dtype=torch.uint8)

        self.assertRaises(TypeError, utils.draw_segmentation_masks, img_tp, masks)
        self.assertRaises(ValueError, utils.draw_segmentation_masks, img_wrong1, masks)
        self.assertRaises(ValueError, utils.draw_segmentation_masks, img_wrong2, masks)
        self.assertRaises(ValueError, utils.draw_segmentation_masks, img_wrong3, masks)


if __name__ == '__main__':
    unittest.main()
