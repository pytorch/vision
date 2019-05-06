import os
import torch
import torchvision.utils as utils
import unittest


class Tester(unittest.TestCase):

    def test_make_grid_not_inplace(self):
        t = torch.rand(5, 3, 10, 10)
        t_clone = t.clone()

        utils.make_grid(t, normalize=False)
        assert torch.equal(t, t_clone), 'make_grid modified tensor in-place'

        utils.make_grid(t, normalize=True, scale_each=False)
        assert torch.equal(t, t_clone), 'make_grid modified tensor in-place'

        utils.make_grid(t, normalize=True, scale_each=True)
        assert torch.equal(t, t_clone), 'make_grid modified tensor in-place'

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

        assert torch.equal(norm_max, rounded_grid_max), 'Normalized max is not equal to 1'
        assert torch.equal(norm_min, rounded_grid_min), 'Normalized min is not equal to 0'

    def test_save_image(self):
        t = torch.rand(2, 3, 64, 64)
        file_name = 'test_image.png'
        utils.save_image(t, file_name)
        assert os.path.exists(file_name), 'The image is not present after save'
        os.remove(file_name)


if __name__ == '__main__':
    unittest.main()
