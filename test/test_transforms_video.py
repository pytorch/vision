import torch
import torchvision.transforms._transforms_video as transforms
from torchvision.transforms import Compose
import unittest
import random
import numpy as np

try:
    from scipy import stats
except ImportError:
    stats = None


class TestVideoTransforms(unittest.TestCase):

    def test_random_crop_video(self):
        numFrames = random.randint(4, 128)
        height = random.randint(10, 32) * 2
        width = random.randint(10, 32) * 2
        oheight = random.randint(5, (height - 2) / 2) * 2
        owidth = random.randint(5, (width - 2) / 2) * 2
        clip = torch.randint(0, 256, (numFrames, height, width, 3), dtype=torch.uint8)
        result = Compose([
            transforms.ToTensorVideo(),
            transforms.RandomCropVideo((oheight, owidth)),
        ])(clip)
        self.assertEqual(result.size(2), oheight)
        self.assertEqual(result.size(3), owidth)

        transforms.RandomCropVideo((oheight, owidth)).__repr__()

    def test_random_resized_crop_video(self):
        numFrames = random.randint(4, 128)
        height = random.randint(10, 32) * 2
        width = random.randint(10, 32) * 2
        oheight = random.randint(5, (height - 2) / 2) * 2
        owidth = random.randint(5, (width - 2) / 2) * 2
        clip = torch.randint(0, 256, (numFrames, height, width, 3), dtype=torch.uint8)
        result = Compose([
            transforms.ToTensorVideo(),
            transforms.RandomResizedCropVideo((oheight, owidth)),
        ])(clip)
        self.assertEqual(result.size(2), oheight)
        self.assertEqual(result.size(3), owidth)

        transforms.RandomResizedCropVideo((oheight, owidth)).__repr__()

    def test_center_crop_video(self):
        numFrames = random.randint(4, 128)
        height = random.randint(10, 32) * 2
        width = random.randint(10, 32) * 2
        oheight = random.randint(5, (height - 2) / 2) * 2
        owidth = random.randint(5, (width - 2) / 2) * 2

        clip = torch.ones((numFrames, height, width, 3), dtype=torch.uint8) * 255
        oh1 = (height - oheight) // 2
        ow1 = (width - owidth) // 2
        clipNarrow = clip[:, oh1:oh1 + oheight, ow1:ow1 + owidth, :]
        clipNarrow.fill_(0)
        result = Compose([
            transforms.ToTensorVideo(),
            transforms.CenterCropVideo((oheight, owidth)),
        ])(clip)

        msg = "height: " + str(height) + " width: " \
            + str(width) + " oheight: " + str(oheight) + " owidth: " + str(owidth)
        self.assertEqual(result.sum().item(), 0, msg)

        oheight += 1
        owidth += 1
        result = Compose([
            transforms.ToTensorVideo(),
            transforms.CenterCropVideo((oheight, owidth)),
        ])(clip)
        sum1 = result.sum()

        msg = "height: " + str(height) + " width: " \
            + str(width) + " oheight: " + str(oheight) + " owidth: " + str(owidth)
        self.assertEqual(sum1.item() > 1, True, msg)

        oheight += 1
        owidth += 1
        result = Compose([
            transforms.ToTensorVideo(),
            transforms.CenterCropVideo((oheight, owidth)),
        ])(clip)
        sum2 = result.sum()

        msg = "height: " + str(height) + " width: " \
            + str(width) + " oheight: " + str(oheight) + " owidth: " + str(owidth)
        self.assertTrue(sum2.item() > 1, msg)
        self.assertTrue(sum2.item() > sum1.item(), msg)

    @unittest.skipIf(stats is None, 'scipy.stats is not available')
    def test_normalize_video(self):
        def samples_from_standard_normal(tensor):
            p_value = stats.kstest(list(tensor.view(-1)), 'norm', args=(0, 1)).pvalue
            return p_value > 0.0001

        random_state = random.getstate()
        random.seed(42)
        for channels in [1, 3]:
            numFrames = random.randint(4, 128)
            height = random.randint(32, 256)
            width = random.randint(32, 256)
            mean = random.random()
            std = random.random()
            clip = torch.normal(mean, std, size=(channels, numFrames, height, width))
            mean = [clip[c].mean().item() for c in range(channels)]
            std = [clip[c].std().item() for c in range(channels)]
            normalized = transforms.NormalizeVideo(mean, std)(clip)
            self.assertTrue(samples_from_standard_normal(normalized))
        random.setstate(random_state)

        # Checking the optional in-place behaviour
        tensor = torch.rand((3, 128, 16, 16))
        tensor_inplace = transforms.NormalizeVideo((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)(tensor)
        self.assertTrue(torch.equal(tensor, tensor_inplace))

        transforms.NormalizeVideo((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True).__repr__()

    def test_to_tensor_video(self):
        numFrames, height, width = 64, 4, 4
        trans = transforms.ToTensorVideo()

        with self.assertRaises(TypeError):
            trans(np.random.rand(numFrames, height, width, 1).tolist())
            trans(torch.rand((numFrames, height, width, 1), dtype=torch.float))

        with self.assertRaises(ValueError):
            trans(torch.ones((3, numFrames, height, width, 3), dtype=torch.uint8))
            trans(torch.ones((height, width, 3), dtype=torch.uint8))
            trans(torch.ones((width, 3), dtype=torch.uint8))
            trans(torch.ones((3), dtype=torch.uint8))

        trans.__repr__()

    @unittest.skipIf(stats is None, 'scipy.stats not available')
    def test_random_horizontal_flip_video(self):
        random_state = random.getstate()
        random.seed(42)
        clip = torch.rand((3, 4, 112, 112), dtype=torch.float)
        hclip = clip.flip((-1))

        num_samples = 250
        num_horizontal = 0
        for _ in range(num_samples):
            out = transforms.RandomHorizontalFlipVideo()(clip)
            if torch.all(torch.eq(out, hclip)):
                num_horizontal += 1

        p_value = stats.binom_test(num_horizontal, num_samples, p=0.5)
        random.setstate(random_state)
        self.assertGreater(p_value, 0.0001)

        num_samples = 250
        num_horizontal = 0
        for _ in range(num_samples):
            out = transforms.RandomHorizontalFlipVideo(p=0.7)(clip)
            if torch.all(torch.eq(out, hclip)):
                num_horizontal += 1

        p_value = stats.binom_test(num_horizontal, num_samples, p=0.7)
        random.setstate(random_state)
        self.assertGreater(p_value, 0.0001)

        transforms.RandomHorizontalFlipVideo().__repr__()


if __name__ == '__main__':
    unittest.main()
