import torchvision.transforms.functional_tensor as F_t
import unittest
import torch


class Tester(unittest.TestCase):

    def test_vflip(self):
        img_tensor = torch.randn(3,16,16)
        vflipped_img = F_t.vflip(img_tensor)
        vflipped_img_again = F_t.vflip(vflipped_img)
        self.assertEqual(vflipped_img.shape, img_tensor.shape)
        self.assertTrue(torch.equal(img_tensor, vflipped_img_again))

    def test_hflip(self):
        img_tensor = torch.randn(3,16,16)
        hflipped_img = F_t.hflip(img_tensor)
        hflipped_img_again = F_t.hflip(hflipped_img)
        self.assertEqual(hflipped_img.shape, img_tensor.shape)
        self.assertTrue(torch.equal(img_tensor, hflipped_img_again))

if __name__ == '__main__':
    unittest.main()