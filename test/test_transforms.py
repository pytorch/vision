import torch
import torchvision.transforms as transforms
import unittest
import random
import numpy as np


class Tester(unittest.TestCase):

    def test_crop(self):
        height = random.randint(10, 32) * 2
        width = random.randint(10, 32) * 2
        oheight = random.randint(5, (height - 2) / 2) * 2
        owidth = random.randint(5, (width - 2) / 2) * 2

        img = torch.ones(3, height, width)
        oh1 = (height - oheight) // 2
        ow1 = (width - owidth) // 2
        imgnarrow = img[:, oh1:oh1 + oheight, ow1:ow1 + owidth]
        imgnarrow.fill_(0)
        result = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop((oheight, owidth)),
            transforms.ToTensor(),
        ])(img)
        assert result.sum() == 0, "height: " + str(height) + " width: " \
                                  + str(width) + " oheight: " + str(oheight) + " owidth: " + str(owidth)
        oheight += 1
        owidth += 1
        result = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop((oheight, owidth)),
            transforms.ToTensor(),
        ])(img)
        sum1 = result.sum()
        assert sum1 > 1, "height: " + str(height) + " width: " \
                         + str(width) + " oheight: " + str(oheight) + " owidth: " + str(owidth)
        oheight += 1
        owidth += 1
        result = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop((oheight, owidth)),
            transforms.ToTensor(),
        ])(img)
        sum2 = result.sum()
        assert sum2 > 0, "height: " + str(height) + " width: " \
                         + str(width) + " oheight: " + str(oheight) + " owidth: " + str(owidth)
        assert sum2 > sum1, "height: " + str(height) + " width: " \
                            + str(width) + " oheight: " + str(oheight) + " owidth: " + str(owidth)

    def test_scale(self):
        height = random.randint(24, 32) * 2
        width = random.randint(24, 32) * 2
        osize = random.randint(5, 12) * 2

        img = torch.ones(3, height, width)
        result = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Scale(osize),
            transforms.ToTensor(),
        ])(img)
        # print img.size()
        # print 'output size:', osize
        # print result.size()
        assert osize in result.size()
        if height < width:
            assert result.size(1) <= result.size(2)
        elif width < height:
            assert result.size(1) >= result.size(2)

    def test_random_crop(self):
        height = random.randint(10, 32) * 2
        width = random.randint(10, 32) * 2
        oheight = random.randint(5, (height - 2) / 2) * 2
        owidth = random.randint(5, (width - 2) / 2) * 2
        img = torch.ones(3, height, width)
        result = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop((oheight, owidth)),
            transforms.ToTensor(),
        ])(img)
        assert result.size(1) == oheight
        assert result.size(2) == owidth

        padding = random.randint(1, 20)
        result = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop((oheight, owidth), padding=padding),
            transforms.ToTensor(),
        ])(img)
        assert result.size(1) == oheight
        assert result.size(2) == owidth

    def test_pad(self):
        height = random.randint(10, 32) * 2
        width = random.randint(10, 32) * 2
        img = torch.ones(3, height, width)
        padding = random.randint(1, 20)
        result = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(padding),
            transforms.ToTensor(),
        ])(img)
        assert result.size(1) == height + 2 * padding
        assert result.size(2) == width + 2 * padding

    def test_lambda(self):
        trans = transforms.Lambda(lambda x: x.add(10))
        x = torch.randn(10)
        y = trans(x)
        assert (y.equal(torch.add(x, 10)))

        trans = transforms.Lambda(lambda x: x.add_(10))
        x = torch.randn(10)
        y = trans(x)
        assert (y.equal(x))

    def test_to_tensor(self):
        channels = 3
        height, width = 4, 4
        trans = transforms.ToTensor()
        input_data = torch.ByteTensor(channels, height, width).random_(0, 255).float().div_(255)
        img = transforms.ToPILImage()(input_data)
        output = trans(img)
        assert np.allclose(input_data.numpy(), output.numpy())

        ndarray = np.random.randint(low=0, high=255, size=(height, width, channels))
        output = trans(ndarray)
        expected_output = ndarray.transpose((2, 0, 1)) / 255.0
        assert np.allclose(output.numpy(), expected_output)

    def test_tensor_to_pil_image(self):
        trans = transforms.ToPILImage()
        to_tensor = transforms.ToTensor()

        img_data = torch.Tensor(3, 4, 4).uniform_()
        img = trans(img_data)
        assert img.getbands() == ('R', 'G', 'B')
        r, g, b = img.split()

        expected_output = img_data.mul(255).int().float().div(255)
        assert np.allclose(expected_output[0].numpy(), to_tensor(r).numpy())
        assert np.allclose(expected_output[1].numpy(), to_tensor(g).numpy())
        assert np.allclose(expected_output[2].numpy(), to_tensor(b).numpy())

        # single channel image
        img_data = torch.Tensor(1, 4, 4).uniform_()
        img = trans(img_data)
        assert img.getbands() == ('L',)
        l, = img.split()
        expected_output = img_data.mul(255).int().float().div(255)
        assert np.allclose(expected_output[0].numpy(), to_tensor(l).numpy())

    def test_tensor_gray_to_pil_image(self):
        trans = transforms.ToPILImage()
        to_tensor = transforms.ToTensor()

        img_data_byte = torch.ByteTensor(1, 4, 4).random_(0, 255)
        img_data_short = torch.ShortTensor(1, 4, 4).random_()
        img_data_int = torch.IntTensor(1, 4, 4).random_()
        img_data_float = torch.FloatTensor(1, 4, 4).uniform_()

        img_byte = trans(img_data_byte)
        img_short = trans(img_data_short)
        img_int = trans(img_data_int)
        img_float = trans(img_data_float)
        assert img_byte.mode == 'L'
        assert img_short.mode == 'I;16'
        assert img_int.mode == 'I'
        #assert img_float.mode == 'F'

        assert np.allclose(img_data_short.numpy(), to_tensor(img_short).numpy())
        assert np.allclose(img_data_int.numpy(), to_tensor(img_int).numpy())
        # would cause breaking changes as ToTensor converts to range [0, 1]
        #assert np.allclose(img_data_byte.numpy(), to_tensor(img_byte).numpy())
        #assert np.allclose(img_data_float.numpy(), to_tensor(img_float).numpy())

    def test_ndarray_to_pil_image(self):
        trans = transforms.ToPILImage()
        img_data = torch.ByteTensor(4, 4, 3).random_(0, 255).numpy()
        img = trans(img_data)
        assert img.getbands() == ('R', 'G', 'B')
        r, g, b = img.split()

        assert np.allclose(r, img_data[:, :, 0])
        assert np.allclose(g, img_data[:, :, 1])
        assert np.allclose(b, img_data[:, :, 2])

        # single channel image
        img_data = torch.ByteTensor(4, 4, 1).random_(0, 255).numpy()
        img = trans(img_data)
        assert img.getbands() == ('L',)
        l, = img.split()
        assert np.allclose(l, img_data[:, :, 0])

    def test_ndarray_bad_types_to_pil_image(self):
        trans = transforms.ToPILImage()
        with self.assertRaises(AssertionError):
            trans(np.ones([4, 4, 1], np.int64))
            trans(np.ones([4, 4, 1], np.uint16))
            trans(np.ones([4, 4, 1], np.uint32))
            trans(np.ones([4, 4, 1], np.float64))

    def test_ndarray_gray_float32_to_pil_image(self):
        trans = transforms.ToPILImage()
        img_data = torch.FloatTensor(4, 4, 1).random_().numpy()
        img = trans(img_data)
        assert img.mode == 'F'
        assert np.allclose(img, img_data[:, :, 0])

    def test_ndarray_gray_int16_to_pil_image(self):
        trans = transforms.ToPILImage()
        img_data = torch.ShortTensor(4, 4, 1).random_().numpy()
        img = trans(img_data)
        assert img.mode == 'I;16'
        assert np.allclose(img, img_data[:, :, 0])

    def test_ndarray_gray_int32_to_pil_image(self):
        trans = transforms.ToPILImage()
        img_data = torch.IntTensor(4, 4, 1).random_().numpy()
        img = trans(img_data)
        assert img.mode == 'I'
        assert np.allclose(img, img_data[:, :, 0])


if __name__ == '__main__':
    unittest.main()
