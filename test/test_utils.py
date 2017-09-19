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

    def test_make_grid_raises_with_variable(self):
        t = torch.autograd.Variable(torch.rand(3, 10, 10))
        with self.assertRaises(TypeError):
            utils.make_grid(t)

        with self.assertRaises(TypeError):
            utils.make_grid([t, t, t, t])


if __name__ == '__main__':
    unittest.main()
