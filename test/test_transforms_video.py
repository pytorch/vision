import random
import warnings

import numpy as np
import pytest
import torch
from common_utils import assert_equal
from torchvision.transforms import Compose

try:
    from scipy import stats
except ImportError:
    stats = None


with warnings.catch_warnings(record=True):
    warnings.simplefilter("always")
    import torchvision.transforms._transforms_video as transforms


class TestVideoTransforms:
    def test_random_crop_video(self):
        numFrames = random.randint(4, 128)
        height = random.randint(10, 32) * 2
        width = random.randint(10, 32) * 2
        oheight = random.randint(5, (height - 2) / 2) * 2
        owidth = random.randint(5, (width - 2) / 2) * 2
        clip = torch.randint(0, 256, (numFrames, height, width, 3), dtype=torch.uint8)
        result = Compose(
            [
                transforms.ToTensorVideo(),
                transforms.RandomCropVideo((oheight, owidth)),
            ]
        )(clip)
        assert result.size(2) == oheight
        assert result.size(3) == owidth

        transforms.RandomCropVideo((oheight, owidth)).__repr__()

    def test_random_resized_crop_video(self):
        numFrames = random.randint(4, 128)
        height = random.randint(10, 32) * 2
        width = random.randint(10, 32) * 2
        oheight = random.randint(5, (height - 2) / 2) * 2
        owidth = random.randint(5, (width - 2) / 2) * 2
        clip = torch.randint(0, 256, (numFrames, height, width, 3), dtype=torch.uint8)
        result = Compose(
            [
                transforms.ToTensorVideo(),
                transforms.RandomResizedCropVideo((oheight, owidth)),
            ]
        )(clip)
        assert result.size(2) == oheight
        assert result.size(3) == owidth

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
        clipNarrow = clip[:, oh1 : oh1 + oheight, ow1 : ow1 + owidth, :]
        clipNarrow.fill_(0)
        result = Compose(
            [
                transforms.ToTensorVideo(),
                transforms.CenterCropVideo((oheight, owidth)),
            ]
        )(clip)

        msg = (
            "height: " + str(height) + " width: " + str(width) + " oheight: " + str(oheight) + " owidth: " + str(owidth)
        )
        assert result.sum().item() == 0, msg

        oheight += 1
        owidth += 1
        result = Compose(
            [
                transforms.ToTensorVideo(),
                transforms.CenterCropVideo((oheight, owidth)),
            ]
        )(clip)
        sum1 = result.sum()

        msg = (
            "height: " + str(height) + " width: " + str(width) + " oheight: " + str(oheight) + " owidth: " + str(owidth)
        )
        assert sum1.item() > 1, msg

        oheight += 1
        owidth += 1
        result = Compose(
            [
                transforms.ToTensorVideo(),
                transforms.CenterCropVideo((oheight, owidth)),
            ]
        )(clip)
        sum2 = result.sum()

        msg = (
            "height: " + str(height) + " width: " + str(width) + " oheight: " + str(oheight) + " owidth: " + str(owidth)
        )
        assert sum2.item() > 1, msg
        assert sum2.item() > sum1.item(), msg

    @pytest.mark.skipif(stats is None, reason="scipy.stats is not available")
    @pytest.mark.parametrize("channels", [1, 3])
    def test_normalize_video(self, channels):
        def samples_from_standard_normal(tensor):
            p_value = stats.kstest(list(tensor.view(-1)), "norm", args=(0, 1)).pvalue
            return p_value > 0.0001

        random_state = random.getstate()
        random.seed(42)

        numFrames = random.randint(4, 128)
        height = random.randint(32, 256)
        width = random.randint(32, 256)
        mean = random.random()
        std = random.random()
        clip = torch.normal(mean, std, size=(channels, numFrames, height, width))
        mean = [clip[c].mean().item() for c in range(channels)]
        std = [clip[c].std().item() for c in range(channels)]
        normalized = transforms.NormalizeVideo(mean, std)(clip)
        assert samples_from_standard_normal(normalized)
        random.setstate(random_state)

        # Checking the optional in-place behaviour
        tensor = torch.rand((3, 128, 16, 16))
        tensor_inplace = transforms.NormalizeVideo((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)(tensor)
        assert_equal(tensor, tensor_inplace)

        transforms.NormalizeVideo((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True).__repr__()

    def test_to_tensor_video(self):
        numFrames, height, width = 64, 4, 4
        trans = transforms.ToTensorVideo()

        with pytest.raises(TypeError):
            np_rng = np.random.RandomState(0)
            trans(np_rng.rand(numFrames, height, width, 1).tolist())
        with pytest.raises(TypeError):
            trans(torch.rand((numFrames, height, width, 1), dtype=torch.float))

        with pytest.raises(ValueError):
            trans(torch.ones((3, numFrames, height, width, 3), dtype=torch.uint8))
        with pytest.raises(ValueError):
            trans(torch.ones((height, width, 3), dtype=torch.uint8))
        with pytest.raises(ValueError):
            trans(torch.ones((width, 3), dtype=torch.uint8))
        with pytest.raises(ValueError):
            trans(torch.ones((3), dtype=torch.uint8))

        trans.__repr__()

    @pytest.mark.parametrize("p", (0, 1))
    def test_random_horizontal_flip_video(self, p):
        clip = torch.rand((3, 4, 112, 112), dtype=torch.float)
        hclip = clip.flip(-1)

        out = transforms.RandomHorizontalFlipVideo(p=p)(clip)
        if p == 0:
            torch.testing.assert_close(out, clip)
        elif p == 1:
            torch.testing.assert_close(out, hclip)

        transforms.RandomHorizontalFlipVideo().__repr__()


if __name__ == "__main__":
    pytest.main([__file__])
