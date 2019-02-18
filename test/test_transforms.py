from __future__ import division
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch._utils_internal import get_file_path_2
import unittest
import math
import random
import numpy as np
from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None

try:
    from scipy import stats
except ImportError:
    stats = None

GRACE_HOPPER = get_file_path_2('assets/grace_hopper_517x606.jpg')


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

    def test_five_crop(self):
        to_pil_image = transforms.ToPILImage()
        h = random.randint(5, 25)
        w = random.randint(5, 25)
        for single_dim in [True, False]:
            crop_h = random.randint(1, h)
            crop_w = random.randint(1, w)
            if single_dim:
                crop_h = min(crop_h, crop_w)
                crop_w = crop_h
                transform = transforms.FiveCrop(crop_h)
            else:
                transform = transforms.FiveCrop((crop_h, crop_w))

            img = torch.FloatTensor(3, h, w).uniform_()
            results = transform(to_pil_image(img))

            assert len(results) == 5
            for crop in results:
                assert crop.size == (crop_w, crop_h)

            to_pil_image = transforms.ToPILImage()
            tl = to_pil_image(img[:, 0:crop_h, 0:crop_w])
            tr = to_pil_image(img[:, 0:crop_h, w - crop_w:])
            bl = to_pil_image(img[:, h - crop_h:, 0:crop_w])
            br = to_pil_image(img[:, h - crop_h:, w - crop_w:])
            center = transforms.CenterCrop((crop_h, crop_w))(to_pil_image(img))
            expected_output = (tl, tr, bl, br, center)
            assert results == expected_output

    def test_ten_crop(self):
        to_pil_image = transforms.ToPILImage()
        h = random.randint(5, 25)
        w = random.randint(5, 25)
        for should_vflip in [True, False]:
            for single_dim in [True, False]:
                crop_h = random.randint(1, h)
                crop_w = random.randint(1, w)
                if single_dim:
                    crop_h = min(crop_h, crop_w)
                    crop_w = crop_h
                    transform = transforms.TenCrop(crop_h,
                                                   vertical_flip=should_vflip)
                    five_crop = transforms.FiveCrop(crop_h)
                else:
                    transform = transforms.TenCrop((crop_h, crop_w),
                                                   vertical_flip=should_vflip)
                    five_crop = transforms.FiveCrop((crop_h, crop_w))

                img = to_pil_image(torch.FloatTensor(3, h, w).uniform_())
                results = transform(img)
                expected_output = five_crop(img)

                # Checking if FiveCrop and TenCrop can be printed as string
                transform.__repr__()
                five_crop.__repr__()

                if should_vflip:
                    vflipped_img = img.transpose(Image.FLIP_TOP_BOTTOM)
                    expected_output += five_crop(vflipped_img)
                else:
                    hflipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    expected_output += five_crop(hflipped_img)

                assert len(results) == 10
                assert expected_output == results

    def test_randomresized_params(self):
        height = random.randint(24, 32) * 2
        width = random.randint(24, 32) * 2
        img = torch.ones(3, height, width)
        to_pil_image = transforms.ToPILImage()
        img = to_pil_image(img)
        size = 100
        epsilon = 0.05
        for i in range(10):
            scale_min = round(random.random(), 2)
            scale_range = (scale_min, scale_min + round(random.random(), 2))
            aspect_min = round(random.random(), 2)
            aspect_ratio_range = (aspect_min, aspect_min + round(random.random(), 2))
            randresizecrop = transforms.RandomResizedCrop(size, scale_range, aspect_ratio_range)
            _, _, h, w = randresizecrop.get_params(img, scale_range, aspect_ratio_range)
            aspect_ratio_obtained = w / h
            assert (min(aspect_ratio_range) - epsilon <= aspect_ratio_obtained <= max(aspect_ratio_range) + epsilon or
                    aspect_ratio_obtained == 1.0)

    def test_resize(self):
        height = random.randint(24, 32) * 2
        width = random.randint(24, 32) * 2
        osize = random.randint(5, 12) * 2

        img = torch.ones(3, height, width)
        result = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(osize),
            transforms.ToTensor(),
        ])(img)
        assert osize in result.size()
        if height < width:
            assert result.size(1) <= result.size(2)
        elif width < height:
            assert result.size(1) >= result.size(2)

        result = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([osize, osize]),
            transforms.ToTensor(),
        ])(img)
        assert osize in result.size()
        assert result.size(1) == osize
        assert result.size(2) == osize

        oheight = random.randint(5, 12) * 2
        owidth = random.randint(5, 12) * 2
        result = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((oheight, owidth)),
            transforms.ToTensor(),
        ])(img)
        assert result.size(1) == oheight
        assert result.size(2) == owidth

        result = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([oheight, owidth]),
            transforms.ToTensor(),
        ])(img)
        assert result.size(1) == oheight
        assert result.size(2) == owidth

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

        result = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop((height, width)),
            transforms.ToTensor()
        ])(img)
        assert result.size(1) == height
        assert result.size(2) == width
        assert np.allclose(img.numpy(), result.numpy())

        result = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop((height + 1, width + 1), pad_if_needed=True),
            transforms.ToTensor(),
        ])(img)
        assert result.size(1) == height + 1
        assert result.size(2) == width + 1

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

    def test_pad_with_tuple_of_pad_values(self):
        height = random.randint(10, 32) * 2
        width = random.randint(10, 32) * 2
        img = transforms.ToPILImage()(torch.ones(3, height, width))

        padding = tuple([random.randint(1, 20) for _ in range(2)])
        output = transforms.Pad(padding)(img)
        assert output.size == (width + padding[0] * 2, height + padding[1] * 2)

        padding = tuple([random.randint(1, 20) for _ in range(4)])
        output = transforms.Pad(padding)(img)
        assert output.size[0] == width + padding[0] + padding[2]
        assert output.size[1] == height + padding[1] + padding[3]

        # Checking if Padding can be printed as string
        transforms.Pad(padding).__repr__()

    def test_pad_with_non_constant_padding_modes(self):
        """Unit tests for edge, reflect, symmetric padding"""
        img = torch.zeros(3, 27, 27).byte()
        img[:, :, 0] = 1  # Constant value added to leftmost edge
        img = transforms.ToPILImage()(img)
        img = F.pad(img, 1, (200, 200, 200))

        # pad 3 to all sidess
        edge_padded_img = F.pad(img, 3, padding_mode='edge')
        # First 6 elements of leftmost edge in the middle of the image, values are in order:
        # edge_pad, edge_pad, edge_pad, constant_pad, constant value added to leftmost edge, 0
        edge_middle_slice = np.asarray(edge_padded_img).transpose(2, 0, 1)[0][17][:6]
        assert np.all(edge_middle_slice == np.asarray([200, 200, 200, 200, 1, 0]))
        assert transforms.ToTensor()(edge_padded_img).size() == (3, 35, 35)

        # Pad 3 to left/right, 2 to top/bottom
        reflect_padded_img = F.pad(img, (3, 2), padding_mode='reflect')
        # First 6 elements of leftmost edge in the middle of the image, values are in order:
        # reflect_pad, reflect_pad, reflect_pad, constant_pad, constant value added to leftmost edge, 0
        reflect_middle_slice = np.asarray(reflect_padded_img).transpose(2, 0, 1)[0][17][:6]
        assert np.all(reflect_middle_slice == np.asarray([0, 0, 1, 200, 1, 0]))
        assert transforms.ToTensor()(reflect_padded_img).size() == (3, 33, 35)

        # Pad 3 to left, 2 to top, 2 to right, 1 to bottom
        symmetric_padded_img = F.pad(img, (3, 2, 2, 1), padding_mode='symmetric')
        # First 6 elements of leftmost edge in the middle of the image, values are in order:
        # sym_pad, sym_pad, sym_pad, constant_pad, constant value added to leftmost edge, 0
        symmetric_middle_slice = np.asarray(symmetric_padded_img).transpose(2, 0, 1)[0][17][:6]
        assert np.all(symmetric_middle_slice == np.asarray([0, 1, 200, 200, 1, 0]))
        assert transforms.ToTensor()(symmetric_padded_img).size() == (3, 32, 34)

    def test_pad_raises_with_invalid_pad_sequence_len(self):
        with self.assertRaises(ValueError):
            transforms.Pad(())

        with self.assertRaises(ValueError):
            transforms.Pad((1, 2, 3))

        with self.assertRaises(ValueError):
            transforms.Pad((1, 2, 3, 4, 5))

    def test_lambda(self):
        trans = transforms.Lambda(lambda x: x.add(10))
        x = torch.randn(10)
        y = trans(x)
        assert (y.equal(torch.add(x, 10)))

        trans = transforms.Lambda(lambda x: x.add_(10))
        x = torch.randn(10)
        y = trans(x)
        assert (y.equal(x))

        # Checking if Lambda can be printed as string
        trans.__repr__()

    def test_random_apply(self):
        random_state = random.getstate()
        random.seed(42)
        random_apply_transform = transforms.RandomApply(
            [
                transforms.RandomRotation((-45, 45)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ], p=0.75
        )
        img = transforms.ToPILImage()(torch.rand(3, 10, 10))
        num_samples = 250
        num_applies = 0
        for _ in range(num_samples):
            out = random_apply_transform(img)
            if out != img:
                num_applies += 1

        p_value = stats.binom_test(num_applies, num_samples, p=0.75)
        random.setstate(random_state)
        assert p_value > 0.0001

        # Checking if RandomApply can be printed as string
        random_apply_transform.__repr__()

    def test_random_choice(self):
        random_state = random.getstate()
        random.seed(42)
        random_choice_transform = transforms.RandomChoice(
            [
                transforms.Resize(15),
                transforms.Resize(20),
                transforms.CenterCrop(10)
            ]
        )
        img = transforms.ToPILImage()(torch.rand(3, 25, 25))
        num_samples = 250
        num_resize_15 = 0
        num_resize_20 = 0
        num_crop_10 = 0
        for _ in range(num_samples):
            out = random_choice_transform(img)
            if out.size == (15, 15):
                num_resize_15 += 1
            elif out.size == (20, 20):
                num_resize_20 += 1
            elif out.size == (10, 10):
                num_crop_10 += 1

        p_value = stats.binom_test(num_resize_15, num_samples, p=0.33333)
        assert p_value > 0.0001
        p_value = stats.binom_test(num_resize_20, num_samples, p=0.33333)
        assert p_value > 0.0001
        p_value = stats.binom_test(num_crop_10, num_samples, p=0.33333)
        assert p_value > 0.0001

        random.setstate(random_state)
        # Checking if RandomChoice can be printed as string
        random_choice_transform.__repr__()

    def test_random_order(self):
        random_state = random.getstate()
        random.seed(42)
        random_order_transform = transforms.RandomOrder(
            [
                transforms.Resize(20),
                transforms.CenterCrop(10)
            ]
        )
        img = transforms.ToPILImage()(torch.rand(3, 25, 25))
        num_samples = 250
        num_normal_order = 0
        resize_crop_out = transforms.CenterCrop(10)(transforms.Resize(20)(img))
        for _ in range(num_samples):
            out = random_order_transform(img)
            if out == resize_crop_out:
                num_normal_order += 1

        p_value = stats.binom_test(num_normal_order, num_samples, p=0.5)
        random.setstate(random_state)
        assert p_value > 0.0001

        # Checking if RandomOrder can be printed as string
        random_order_transform.__repr__()

    def test_to_tensor(self):
        test_channels = [1, 3, 4]
        height, width = 4, 4
        trans = transforms.ToTensor()

        for channels in test_channels:
            input_data = torch.ByteTensor(channels, height, width).random_(0, 255).float().div_(255)
            img = transforms.ToPILImage()(input_data)
            output = trans(img)
            assert np.allclose(input_data.numpy(), output.numpy())

            ndarray = np.random.randint(low=0, high=255, size=(height, width, channels)).astype(np.uint8)
            output = trans(ndarray)
            expected_output = ndarray.transpose((2, 0, 1)) / 255.0
            assert np.allclose(output.numpy(), expected_output)

            ndarray = np.random.rand(height, width, channels).astype(np.float32)
            output = trans(ndarray)
            expected_output = ndarray.transpose((2, 0, 1))
            assert np.allclose(output.numpy(), expected_output)

        # separate test for mode '1' PIL images
        input_data = torch.ByteTensor(1, height, width).bernoulli_()
        img = transforms.ToPILImage()(input_data.mul(255)).convert('1')
        output = trans(img)
        assert np.allclose(input_data.numpy(), output.numpy())

    @unittest.skipIf(accimage is None, 'accimage not available')
    def test_accimage_to_tensor(self):
        trans = transforms.ToTensor()

        expected_output = trans(Image.open(GRACE_HOPPER).convert('RGB'))
        output = trans(accimage.Image(GRACE_HOPPER))

        self.assertEqual(expected_output.size(), output.size())
        assert np.allclose(output.numpy(), expected_output.numpy())

    @unittest.skipIf(accimage is None, 'accimage not available')
    def test_accimage_resize(self):
        trans = transforms.Compose([
            transforms.Resize(256, interpolation=Image.LINEAR),
            transforms.ToTensor(),
        ])

        # Checking if Compose, Resize and ToTensor can be printed as string
        trans.__repr__()

        expected_output = trans(Image.open(GRACE_HOPPER).convert('RGB'))
        output = trans(accimage.Image(GRACE_HOPPER))

        self.assertEqual(expected_output.size(), output.size())
        self.assertLess(np.abs((expected_output - output).mean()), 1e-3)
        self.assertLess((expected_output - output).var(), 1e-5)
        # note the high absolute tolerance
        assert np.allclose(output.numpy(), expected_output.numpy(), atol=5e-2)

    @unittest.skipIf(accimage is None, 'accimage not available')
    def test_accimage_crop(self):
        trans = transforms.Compose([
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ])

        # Checking if Compose, CenterCrop and ToTensor can be printed as string
        trans.__repr__()

        expected_output = trans(Image.open(GRACE_HOPPER).convert('RGB'))
        output = trans(accimage.Image(GRACE_HOPPER))

        self.assertEqual(expected_output.size(), output.size())
        assert np.allclose(output.numpy(), expected_output.numpy())

    def test_1_channel_tensor_to_pil_image(self):
        to_tensor = transforms.ToTensor()

        img_data_float = torch.Tensor(1, 4, 4).uniform_()
        img_data_byte = torch.ByteTensor(1, 4, 4).random_(0, 255)
        img_data_short = torch.ShortTensor(1, 4, 4).random_()
        img_data_int = torch.IntTensor(1, 4, 4).random_()

        inputs = [img_data_float, img_data_byte, img_data_short, img_data_int]
        expected_outputs = [img_data_float.mul(255).int().float().div(255).numpy(),
                            img_data_byte.float().div(255.0).numpy(),
                            img_data_short.numpy(),
                            img_data_int.numpy()]
        expected_modes = ['L', 'L', 'I;16', 'I']

        for img_data, expected_output, mode in zip(inputs, expected_outputs, expected_modes):
            for transform in [transforms.ToPILImage(), transforms.ToPILImage(mode=mode)]:
                img = transform(img_data)
                assert img.mode == mode
                assert np.allclose(expected_output, to_tensor(img).numpy())

    def test_1_channel_ndarray_to_pil_image(self):
        img_data_float = torch.Tensor(4, 4, 1).uniform_().numpy()
        img_data_byte = torch.ByteTensor(4, 4, 1).random_(0, 255).numpy()
        img_data_short = torch.ShortTensor(4, 4, 1).random_().numpy()
        img_data_int = torch.IntTensor(4, 4, 1).random_().numpy()

        inputs = [img_data_float, img_data_byte, img_data_short, img_data_int]
        expected_modes = ['F', 'L', 'I;16', 'I']
        for img_data, mode in zip(inputs, expected_modes):
            for transform in [transforms.ToPILImage(), transforms.ToPILImage(mode=mode)]:
                img = transform(img_data)
                assert img.mode == mode
                assert np.allclose(img_data[:, :, 0], img)

    def test_2_channel_ndarray_to_pil_image(self):
        def verify_img_data(img_data, mode):
            if mode is None:
                img = transforms.ToPILImage()(img_data)
                assert img.mode == 'LA'  # default should assume LA
            else:
                img = transforms.ToPILImage(mode=mode)(img_data)
                assert img.mode == mode
            split = img.split()
            for i in range(2):
                assert np.allclose(img_data[:, :, i], split[i])

        img_data = torch.ByteTensor(4, 4, 2).random_(0, 255).numpy()
        for mode in [None, 'LA']:
            verify_img_data(img_data, mode)

        transforms.ToPILImage().__repr__()

        with self.assertRaises(ValueError):
            # should raise if we try a mode for 4 or 1 or 3 channel images
            transforms.ToPILImage(mode='RGBA')(img_data)
            transforms.ToPILImage(mode='P')(img_data)
            transforms.ToPILImage(mode='RGB')(img_data)

    def test_2_channel_tensor_to_pil_image(self):
        def verify_img_data(img_data, expected_output, mode):
            if mode is None:
                img = transforms.ToPILImage()(img_data)
                assert img.mode == 'LA'  # default should assume LA
            else:
                img = transforms.ToPILImage(mode=mode)(img_data)
                assert img.mode == mode
            split = img.split()
            for i in range(2):
                assert np.allclose(expected_output[i].numpy(), F.to_tensor(split[i]).numpy())

        img_data = torch.Tensor(2, 4, 4).uniform_()
        expected_output = img_data.mul(255).int().float().div(255)
        for mode in [None, 'LA']:
            verify_img_data(img_data, expected_output, mode=mode)

        with self.assertRaises(ValueError):
            # should raise if we try a mode for 4 or 1 or 3 channel images
            transforms.ToPILImage(mode='RGBA')(img_data)
            transforms.ToPILImage(mode='P')(img_data)
            transforms.ToPILImage(mode='RGB')(img_data)

    def test_3_channel_tensor_to_pil_image(self):
        def verify_img_data(img_data, expected_output, mode):
            if mode is None:
                img = transforms.ToPILImage()(img_data)
                assert img.mode == 'RGB'  # default should assume RGB
            else:
                img = transforms.ToPILImage(mode=mode)(img_data)
                assert img.mode == mode
            split = img.split()
            for i in range(3):
                assert np.allclose(expected_output[i].numpy(), F.to_tensor(split[i]).numpy())

        img_data = torch.Tensor(3, 4, 4).uniform_()
        expected_output = img_data.mul(255).int().float().div(255)
        for mode in [None, 'RGB', 'HSV', 'YCbCr']:
            verify_img_data(img_data, expected_output, mode=mode)

        with self.assertRaises(ValueError):
            # should raise if we try a mode for 4 or 1 or 2 channel images
            transforms.ToPILImage(mode='RGBA')(img_data)
            transforms.ToPILImage(mode='P')(img_data)
            transforms.ToPILImage(mode='LA')(img_data)

        with self.assertRaises(ValueError):
            transforms.ToPILImage()(torch.Tensor(1, 3, 4, 4).uniform_())

    def test_3_channel_ndarray_to_pil_image(self):
        def verify_img_data(img_data, mode):
            if mode is None:
                img = transforms.ToPILImage()(img_data)
                assert img.mode == 'RGB'  # default should assume RGB
            else:
                img = transforms.ToPILImage(mode=mode)(img_data)
                assert img.mode == mode
            split = img.split()
            for i in range(3):
                assert np.allclose(img_data[:, :, i], split[i])

        img_data = torch.ByteTensor(4, 4, 3).random_(0, 255).numpy()
        for mode in [None, 'RGB', 'HSV', 'YCbCr']:
            verify_img_data(img_data, mode)

        # Checking if ToPILImage can be printed as string
        transforms.ToPILImage().__repr__()

        with self.assertRaises(ValueError):
            # should raise if we try a mode for 4 or 1 or 2 channel images
            transforms.ToPILImage(mode='RGBA')(img_data)
            transforms.ToPILImage(mode='P')(img_data)
            transforms.ToPILImage(mode='LA')(img_data)

    def test_4_channel_tensor_to_pil_image(self):
        def verify_img_data(img_data, expected_output, mode):
            if mode is None:
                img = transforms.ToPILImage()(img_data)
                assert img.mode == 'RGBA'  # default should assume RGBA
            else:
                img = transforms.ToPILImage(mode=mode)(img_data)
                assert img.mode == mode

            split = img.split()
            for i in range(4):
                assert np.allclose(expected_output[i].numpy(), F.to_tensor(split[i]).numpy())

        img_data = torch.Tensor(4, 4, 4).uniform_()
        expected_output = img_data.mul(255).int().float().div(255)
        for mode in [None, 'RGBA', 'CMYK', 'RGBX']:
            verify_img_data(img_data, expected_output, mode)

        with self.assertRaises(ValueError):
            # should raise if we try a mode for 3 or 1 or 2 channel images
            transforms.ToPILImage(mode='RGB')(img_data)
            transforms.ToPILImage(mode='P')(img_data)
            transforms.ToPILImage(mode='LA')(img_data)

    def test_4_channel_ndarray_to_pil_image(self):
        def verify_img_data(img_data, mode):
            if mode is None:
                img = transforms.ToPILImage()(img_data)
                assert img.mode == 'RGBA'  # default should assume RGBA
            else:
                img = transforms.ToPILImage(mode=mode)(img_data)
                assert img.mode == mode
            split = img.split()
            for i in range(4):
                assert np.allclose(img_data[:, :, i], split[i])

        img_data = torch.ByteTensor(4, 4, 4).random_(0, 255).numpy()
        for mode in [None, 'RGBA', 'CMYK', 'RGBX']:
            verify_img_data(img_data, mode)

        with self.assertRaises(ValueError):
            # should raise if we try a mode for 3 or 1 or 2 channel images
            transforms.ToPILImage(mode='RGB')(img_data)
            transforms.ToPILImage(mode='P')(img_data)
            transforms.ToPILImage(mode='LA')(img_data)

    def test_2d_tensor_to_pil_image(self):
        to_tensor = transforms.ToTensor()

        img_data_float = torch.Tensor(4, 4).uniform_()
        img_data_byte = torch.ByteTensor(4, 4).random_(0, 255)
        img_data_short = torch.ShortTensor(4, 4).random_()
        img_data_int = torch.IntTensor(4, 4).random_()

        inputs = [img_data_float, img_data_byte, img_data_short, img_data_int]
        expected_outputs = [img_data_float.mul(255).int().float().div(255).numpy(),
                            img_data_byte.float().div(255.0).numpy(),
                            img_data_short.numpy(),
                            img_data_int.numpy()]
        expected_modes = ['L', 'L', 'I;16', 'I']

        for img_data, expected_output, mode in zip(inputs, expected_outputs, expected_modes):
            for transform in [transforms.ToPILImage(), transforms.ToPILImage(mode=mode)]:
                img = transform(img_data)
                assert img.mode == mode
                assert np.allclose(expected_output, to_tensor(img).numpy())

    def test_2d_ndarray_to_pil_image(self):
        img_data_float = torch.Tensor(4, 4).uniform_().numpy()
        img_data_byte = torch.ByteTensor(4, 4).random_(0, 255).numpy()
        img_data_short = torch.ShortTensor(4, 4).random_().numpy()
        img_data_int = torch.IntTensor(4, 4).random_().numpy()

        inputs = [img_data_float, img_data_byte, img_data_short, img_data_int]
        expected_modes = ['F', 'L', 'I;16', 'I']
        for img_data, mode in zip(inputs, expected_modes):
            for transform in [transforms.ToPILImage(), transforms.ToPILImage(mode=mode)]:
                img = transform(img_data)
                assert img.mode == mode
                assert np.allclose(img_data, img)

    def test_tensor_bad_types_to_pil_image(self):
        with self.assertRaises(ValueError):
            transforms.ToPILImage()(torch.ones(1, 3, 4, 4))

    def test_ndarray_bad_types_to_pil_image(self):
        trans = transforms.ToPILImage()
        with self.assertRaises(TypeError):
            trans(np.ones([4, 4, 1], np.int64))
            trans(np.ones([4, 4, 1], np.uint16))
            trans(np.ones([4, 4, 1], np.uint32))
            trans(np.ones([4, 4, 1], np.float64))

        with self.assertRaises(ValueError):
            transforms.ToPILImage()(np.ones([1, 4, 4, 3]))

    @unittest.skipIf(stats is None, 'scipy.stats not available')
    def test_random_vertical_flip(self):
        random_state = random.getstate()
        random.seed(42)
        img = transforms.ToPILImage()(torch.rand(3, 10, 10))
        vimg = img.transpose(Image.FLIP_TOP_BOTTOM)

        num_samples = 250
        num_vertical = 0
        for _ in range(num_samples):
            out = transforms.RandomVerticalFlip()(img)
            if out == vimg:
                num_vertical += 1

        p_value = stats.binom_test(num_vertical, num_samples, p=0.5)
        random.setstate(random_state)
        assert p_value > 0.0001

        num_samples = 250
        num_vertical = 0
        for _ in range(num_samples):
            out = transforms.RandomVerticalFlip(p=0.7)(img)
            if out == vimg:
                num_vertical += 1

        p_value = stats.binom_test(num_vertical, num_samples, p=0.7)
        random.setstate(random_state)
        assert p_value > 0.0001

        # Checking if RandomVerticalFlip can be printed as string
        transforms.RandomVerticalFlip().__repr__()

    @unittest.skipIf(stats is None, 'scipy.stats not available')
    def test_random_horizontal_flip(self):
        random_state = random.getstate()
        random.seed(42)
        img = transforms.ToPILImage()(torch.rand(3, 10, 10))
        himg = img.transpose(Image.FLIP_LEFT_RIGHT)

        num_samples = 250
        num_horizontal = 0
        for _ in range(num_samples):
            out = transforms.RandomHorizontalFlip()(img)
            if out == himg:
                num_horizontal += 1

        p_value = stats.binom_test(num_horizontal, num_samples, p=0.5)
        random.setstate(random_state)
        assert p_value > 0.0001

        num_samples = 250
        num_horizontal = 0
        for _ in range(num_samples):
            out = transforms.RandomHorizontalFlip(p=0.7)(img)
            if out == himg:
                num_horizontal += 1

        p_value = stats.binom_test(num_horizontal, num_samples, p=0.7)
        random.setstate(random_state)
        assert p_value > 0.0001

        # Checking if RandomHorizontalFlip can be printed as string
        transforms.RandomHorizontalFlip().__repr__()

    @unittest.skipIf(stats is None, 'scipy.stats is not available')
    def test_normalize(self):
        def samples_from_standard_normal(tensor):
            p_value = stats.kstest(list(tensor.view(-1)), 'norm', args=(0, 1)).pvalue
            return p_value > 0.0001

        random_state = random.getstate()
        random.seed(42)
        for channels in [1, 3]:
            img = torch.rand(channels, 10, 10)
            mean = [img[c].mean() for c in range(channels)]
            std = [img[c].std() for c in range(channels)]
            normalized = transforms.Normalize(mean, std)(img)
            assert samples_from_standard_normal(normalized)
        random.setstate(random_state)

        # Checking if Normalize can be printed as string
        transforms.Normalize(mean, std).__repr__()

    def test_adjust_brightness(self):
        x_shape = [2, 2, 3]
        x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
        x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)
        x_pil = Image.fromarray(x_np, mode='RGB')

        # test 0
        y_pil = F.adjust_brightness(x_pil, 1)
        y_np = np.array(y_pil)
        assert np.allclose(y_np, x_np)

        # test 1
        y_pil = F.adjust_brightness(x_pil, 0.5)
        y_np = np.array(y_pil)
        y_ans = [0, 2, 6, 27, 67, 113, 18, 4, 117, 45, 127, 0]
        y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
        assert np.allclose(y_np, y_ans)

        # test 2
        y_pil = F.adjust_brightness(x_pil, 2)
        y_np = np.array(y_pil)
        y_ans = [0, 10, 26, 108, 255, 255, 74, 16, 255, 180, 255, 2]
        y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
        assert np.allclose(y_np, y_ans)

    def test_adjust_contrast(self):
        x_shape = [2, 2, 3]
        x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
        x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)
        x_pil = Image.fromarray(x_np, mode='RGB')

        # test 0
        y_pil = F.adjust_contrast(x_pil, 1)
        y_np = np.array(y_pil)
        assert np.allclose(y_np, x_np)

        # test 1
        y_pil = F.adjust_contrast(x_pil, 0.5)
        y_np = np.array(y_pil)
        y_ans = [43, 45, 49, 70, 110, 156, 61, 47, 160, 88, 170, 43]
        y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
        assert np.allclose(y_np, y_ans)

        # test 2
        y_pil = F.adjust_contrast(x_pil, 2)
        y_np = np.array(y_pil)
        y_ans = [0, 0, 0, 22, 184, 255, 0, 0, 255, 94, 255, 0]
        y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
        assert np.allclose(y_np, y_ans)

    def test_adjust_saturation(self):
        x_shape = [2, 2, 3]
        x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
        x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)
        x_pil = Image.fromarray(x_np, mode='RGB')

        # test 0
        y_pil = F.adjust_saturation(x_pil, 1)
        y_np = np.array(y_pil)
        assert np.allclose(y_np, x_np)

        # test 1
        y_pil = F.adjust_saturation(x_pil, 0.5)
        y_np = np.array(y_pil)
        y_ans = [2, 4, 8, 87, 128, 173, 39, 25, 138, 133, 215, 88]
        y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
        assert np.allclose(y_np, y_ans)

        # test 2
        y_pil = F.adjust_saturation(x_pil, 2)
        y_np = np.array(y_pil)
        y_ans = [0, 6, 22, 0, 149, 255, 32, 0, 255, 4, 255, 0]
        y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
        assert np.allclose(y_np, y_ans)

    def test_adjust_hue(self):
        x_shape = [2, 2, 3]
        x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
        x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)
        x_pil = Image.fromarray(x_np, mode='RGB')

        with self.assertRaises(ValueError):
            F.adjust_hue(x_pil, -0.7)
            F.adjust_hue(x_pil, 1)

        # test 0: almost same as x_data but not exact.
        # probably because hsv <-> rgb floating point ops
        y_pil = F.adjust_hue(x_pil, 0)
        y_np = np.array(y_pil)
        y_ans = [0, 5, 13, 54, 139, 226, 35, 8, 234, 91, 255, 1]
        y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
        assert np.allclose(y_np, y_ans)

        # test 1
        y_pil = F.adjust_hue(x_pil, 0.25)
        y_np = np.array(y_pil)
        y_ans = [13, 0, 12, 224, 54, 226, 234, 8, 99, 1, 222, 255]
        y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
        assert np.allclose(y_np, y_ans)

        # test 2
        y_pil = F.adjust_hue(x_pil, -0.25)
        y_np = np.array(y_pil)
        y_ans = [0, 13, 2, 54, 226, 58, 8, 234, 152, 255, 43, 1]
        y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
        assert np.allclose(y_np, y_ans)

    def test_adjust_gamma(self):
        x_shape = [2, 2, 3]
        x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
        x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)
        x_pil = Image.fromarray(x_np, mode='RGB')

        # test 0
        y_pil = F.adjust_gamma(x_pil, 1)
        y_np = np.array(y_pil)
        assert np.allclose(y_np, x_np)

        # test 1
        y_pil = F.adjust_gamma(x_pil, 0.5)
        y_np = np.array(y_pil)
        y_ans = [0, 35, 57, 117, 185, 240, 97, 45, 244, 151, 255, 15]
        y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
        assert np.allclose(y_np, y_ans)

        # test 2
        y_pil = F.adjust_gamma(x_pil, 2)
        y_np = np.array(y_pil)
        y_ans = [0, 0, 0, 11, 71, 200, 5, 0, 214, 31, 255, 0]
        y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
        assert np.allclose(y_np, y_ans)

    def test_adjusts_L_mode(self):
        x_shape = [2, 2, 3]
        x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
        x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)
        x_rgb = Image.fromarray(x_np, mode='RGB')

        x_l = x_rgb.convert('L')
        assert F.adjust_brightness(x_l, 2).mode == 'L'
        assert F.adjust_saturation(x_l, 2).mode == 'L'
        assert F.adjust_contrast(x_l, 2).mode == 'L'
        assert F.adjust_hue(x_l, 0.4).mode == 'L'
        assert F.adjust_gamma(x_l, 0.5).mode == 'L'

    def test_color_jitter(self):
        color_jitter = transforms.ColorJitter(2, 2, 2, 0.1)

        x_shape = [2, 2, 3]
        x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
        x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)
        x_pil = Image.fromarray(x_np, mode='RGB')
        x_pil_2 = x_pil.convert('L')

        for i in range(10):
            y_pil = color_jitter(x_pil)
            assert y_pil.mode == x_pil.mode

            y_pil_2 = color_jitter(x_pil_2)
            assert y_pil_2.mode == x_pil_2.mode

        # Checking if ColorJitter can be printed as string
        color_jitter.__repr__()

    def test_linear_transformation(self):
        x = torch.randn(250, 10, 10, 3)
        flat_x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        # compute principal components
        sigma = torch.mm(flat_x.t(), flat_x) / flat_x.size(0)
        u, s, _ = np.linalg.svd(sigma.numpy())
        zca_epsilon = 1e-10  # avoid division by 0
        d = torch.Tensor(np.diag(1. / np.sqrt(s + zca_epsilon)))
        u = torch.Tensor(u)
        principal_components = torch.mm(torch.mm(u, d), u.t())
        # initialize whitening matrix
        whitening = transforms.LinearTransformation(principal_components)
        # pass first vector
        xwhite = whitening(x[0].view(10, 10, 3))
        # estimate covariance
        xwhite = xwhite.view(1, 300).numpy()
        cov = np.dot(xwhite, xwhite.T) / x.size(0)
        assert np.allclose(cov, np.identity(1), rtol=1e-3)

        # Checking if LinearTransformation can be printed as string
        whitening.__repr__()

    def test_rotate(self):
        x = np.zeros((100, 100, 3), dtype=np.uint8)
        x[40, 40] = [255, 255, 255]

        with self.assertRaises(TypeError):
            F.rotate(x, 10)

        img = F.to_pil_image(x)

        result = F.rotate(img, 45)
        assert result.size == (100, 100)
        r, c, ch = np.where(result)
        assert all(x in r for x in [49, 50])
        assert all(x in c for x in [36])
        assert all(x in ch for x in [0, 1, 2])

        result = F.rotate(img, 45, expand=True)
        assert result.size == (142, 142)
        r, c, ch = np.where(result)
        assert all(x in r for x in [70, 71])
        assert all(x in c for x in [57])
        assert all(x in ch for x in [0, 1, 2])

        result = F.rotate(img, 45, center=(40, 40))
        assert result.size == (100, 100)
        r, c, ch = np.where(result)
        assert all(x in r for x in [40])
        assert all(x in c for x in [40])
        assert all(x in ch for x in [0, 1, 2])

        result_a = F.rotate(img, 90)
        result_b = F.rotate(img, -270)

        assert np.all(np.array(result_a) == np.array(result_b))

    def test_affine(self):
        input_img = np.zeros((40, 40, 3), dtype=np.uint8)
        pts = []
        cnt = [20, 20]
        for pt in [(16, 16), (20, 16), (20, 20)]:
            for i in range(-5, 5):
                for j in range(-5, 5):
                    input_img[pt[0] + i, pt[1] + j, :] = [255, 155, 55]
                    pts.append((pt[0] + i, pt[1] + j))
        pts = list(set(pts))

        with self.assertRaises(TypeError):
            F.affine(input_img, 10)

        pil_img = F.to_pil_image(input_img)

        def _to_3x3_inv(inv_result_matrix):
            result_matrix = np.zeros((3, 3))
            result_matrix[:2, :] = np.array(inv_result_matrix).reshape((2, 3))
            result_matrix[2, 2] = 1
            return np.linalg.inv(result_matrix)

        def _test_transformation(a, t, s, sh):
            a_rad = math.radians(a)
            s_rad = math.radians(sh)
            # 1) Check transformation matrix:
            c_matrix = np.array([[1.0, 0.0, cnt[0]], [0.0, 1.0, cnt[1]], [0.0, 0.0, 1.0]])
            c_inv_matrix = np.linalg.inv(c_matrix)
            t_matrix = np.array([[1.0, 0.0, t[0]],
                                 [0.0, 1.0, t[1]],
                                 [0.0, 0.0, 1.0]])
            r_matrix = np.array([[s * math.cos(a_rad), -s * math.sin(a_rad + s_rad), 0.0],
                                 [s * math.sin(a_rad), s * math.cos(a_rad + s_rad), 0.0],
                                 [0.0, 0.0, 1.0]])
            true_matrix = np.dot(t_matrix, np.dot(c_matrix, np.dot(r_matrix, c_inv_matrix)))
            result_matrix = _to_3x3_inv(F._get_inverse_affine_matrix(center=cnt, angle=a,
                                                                     translate=t, scale=s, shear=sh))
            assert np.sum(np.abs(true_matrix - result_matrix)) < 1e-10
            # 2) Perform inverse mapping:
            true_result = np.zeros((40, 40, 3), dtype=np.uint8)
            inv_true_matrix = np.linalg.inv(true_matrix)
            for y in range(true_result.shape[0]):
                for x in range(true_result.shape[1]):
                    res = np.dot(inv_true_matrix, [x, y, 1])
                    _x = int(res[0] + 0.5)
                    _y = int(res[1] + 0.5)
                    if 0 <= _x < input_img.shape[1] and 0 <= _y < input_img.shape[0]:
                        true_result[y, x, :] = input_img[_y, _x, :]

            result = F.affine(pil_img, angle=a, translate=t, scale=s, shear=sh)
            assert result.size == pil_img.size
            # Compute number of different pixels:
            np_result = np.array(result)
            n_diff_pixels = np.sum(np_result != true_result) / 3
            # Accept 3 wrong pixels
            assert n_diff_pixels < 3, \
                "a={}, t={}, s={}, sh={}\n".format(a, t, s, sh) +\
                "n diff pixels={}\n".format(np.sum(np.array(result)[:, :, 0] != true_result[:, :, 0]))

        # Test rotation
        a = 45
        _test_transformation(a=a, t=(0, 0), s=1.0, sh=0.0)

        # Test translation
        t = [10, 15]
        _test_transformation(a=0.0, t=t, s=1.0, sh=0.0)

        # Test scale
        s = 1.2
        _test_transformation(a=0.0, t=(0.0, 0.0), s=s, sh=0.0)

        # Test shear
        sh = 45.0
        _test_transformation(a=0.0, t=(0.0, 0.0), s=1.0, sh=sh)

        # Test rotation, scale, translation, shear
        for a in range(-90, 90, 25):
            for t1 in range(-10, 10, 5):
                for s in [0.75, 0.98, 1.0, 1.1, 1.2]:
                    for sh in range(-15, 15, 5):
                        _test_transformation(a=a, t=(t1, t1), s=s, sh=sh)

    def test_random_rotation(self):

        with self.assertRaises(ValueError):
            transforms.RandomRotation(-0.7)
            transforms.RandomRotation([-0.7])
            transforms.RandomRotation([-0.7, 0, 0.7])

        t = transforms.RandomRotation(10)
        angle = t.get_params(t.degrees)
        assert angle > -10 and angle < 10

        t = transforms.RandomRotation((-10, 10))
        angle = t.get_params(t.degrees)
        assert angle > -10 and angle < 10

        # Checking if RandomRotation can be printed as string
        t.__repr__()

    def test_random_affine(self):

        with self.assertRaises(ValueError):
            transforms.RandomAffine(-0.7)
            transforms.RandomAffine([-0.7])
            transforms.RandomAffine([-0.7, 0, 0.7])

            transforms.RandomAffine([-90, 90], translate=2.0)
            transforms.RandomAffine([-90, 90], translate=[-1.0, 1.0])
            transforms.RandomAffine([-90, 90], translate=[-1.0, 0.0, 1.0])

            transforms.RandomAffine([-90, 90], translate=[0.2, 0.2], scale=[0.0])
            transforms.RandomAffine([-90, 90], translate=[0.2, 0.2], scale=[-1.0, 1.0])
            transforms.RandomAffine([-90, 90], translate=[0.2, 0.2], scale=[0.5, -0.5])
            transforms.RandomAffine([-90, 90], translate=[0.2, 0.2], scale=[0.5, 3.0, -0.5])

            transforms.RandomAffine([-90, 90], translate=[0.2, 0.2], scale=[0.5, 0.5], shear=-7)
            transforms.RandomAffine([-90, 90], translate=[0.2, 0.2], scale=[0.5, 0.5], shear=[-10])
            transforms.RandomAffine([-90, 90], translate=[0.2, 0.2], scale=[0.5, 0.5], shear=[-10, 0, 10])

        x = np.zeros((100, 100, 3), dtype=np.uint8)
        img = F.to_pil_image(x)

        t = transforms.RandomAffine(10, translate=[0.5, 0.3], scale=[0.7, 1.3], shear=[-10, 10])
        for _ in range(100):
            angle, translations, scale, shear = t.get_params(t.degrees, t.translate, t.scale, t.shear,
                                                             img_size=img.size)
            assert -10 < angle < 10
            assert -img.size[0] * 0.5 <= translations[0] <= img.size[0] * 0.5, \
                "{} vs {}".format(translations[0], img.size[0] * 0.5)
            assert -img.size[1] * 0.5 <= translations[1] <= img.size[1] * 0.5, \
                "{} vs {}".format(translations[1], img.size[1] * 0.5)
            assert 0.7 < scale < 1.3
            assert -10 < shear < 10

        # Checking if RandomAffine can be printed as string
        t.__repr__()

        t = transforms.RandomAffine(10, resample=Image.BILINEAR)
        assert "Image.BILINEAR" in t.__repr__()

    def test_to_grayscale(self):
        """Unit tests for grayscale transform"""

        x_shape = [2, 2, 3]
        x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
        x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)
        x_pil = Image.fromarray(x_np, mode='RGB')
        x_pil_2 = x_pil.convert('L')
        gray_np = np.array(x_pil_2)

        # Test Set: Grayscale an image with desired number of output channels
        # Case 1: RGB -> 1 channel grayscale
        trans1 = transforms.Grayscale(num_output_channels=1)
        gray_pil_1 = trans1(x_pil)
        gray_np_1 = np.array(gray_pil_1)
        assert gray_pil_1.mode == 'L', 'mode should be L'
        assert gray_np_1.shape == tuple(x_shape[0:2]), 'should be 1 channel'
        np.testing.assert_equal(gray_np, gray_np_1)

        # Case 2: RGB -> 3 channel grayscale
        trans2 = transforms.Grayscale(num_output_channels=3)
        gray_pil_2 = trans2(x_pil)
        gray_np_2 = np.array(gray_pil_2)
        assert gray_pil_2.mode == 'RGB', 'mode should be RGB'
        assert gray_np_2.shape == tuple(x_shape), 'should be 3 channel'
        np.testing.assert_equal(gray_np_2[:, :, 0], gray_np_2[:, :, 1])
        np.testing.assert_equal(gray_np_2[:, :, 1], gray_np_2[:, :, 2])
        np.testing.assert_equal(gray_np, gray_np_2[:, :, 0])

        # Case 3: 1 channel grayscale -> 1 channel grayscale
        trans3 = transforms.Grayscale(num_output_channels=1)
        gray_pil_3 = trans3(x_pil_2)
        gray_np_3 = np.array(gray_pil_3)
        assert gray_pil_3.mode == 'L', 'mode should be L'
        assert gray_np_3.shape == tuple(x_shape[0:2]), 'should be 1 channel'
        np.testing.assert_equal(gray_np, gray_np_3)

        # Case 4: 1 channel grayscale -> 3 channel grayscale
        trans4 = transforms.Grayscale(num_output_channels=3)
        gray_pil_4 = trans4(x_pil_2)
        gray_np_4 = np.array(gray_pil_4)
        assert gray_pil_4.mode == 'RGB', 'mode should be RGB'
        assert gray_np_4.shape == tuple(x_shape), 'should be 3 channel'
        np.testing.assert_equal(gray_np_4[:, :, 0], gray_np_4[:, :, 1])
        np.testing.assert_equal(gray_np_4[:, :, 1], gray_np_4[:, :, 2])
        np.testing.assert_equal(gray_np, gray_np_4[:, :, 0])

        # Checking if Grayscale can be printed as string
        trans4.__repr__()

    @unittest.skipIf(stats is None, 'scipy.stats not available')
    def test_random_grayscale(self):
        """Unit tests for random grayscale transform"""

        # Test Set 1: RGB -> 3 channel grayscale
        random_state = random.getstate()
        random.seed(42)
        x_shape = [2, 2, 3]
        x_np = np.random.randint(0, 256, x_shape, np.uint8)
        x_pil = Image.fromarray(x_np, mode='RGB')
        x_pil_2 = x_pil.convert('L')
        gray_np = np.array(x_pil_2)

        num_samples = 250
        num_gray = 0
        for _ in range(num_samples):
            gray_pil_2 = transforms.RandomGrayscale(p=0.5)(x_pil)
            gray_np_2 = np.array(gray_pil_2)
            if np.array_equal(gray_np_2[:, :, 0], gray_np_2[:, :, 1]) and \
                    np.array_equal(gray_np_2[:, :, 1], gray_np_2[:, :, 2]) and \
                    np.array_equal(gray_np, gray_np_2[:, :, 0]):
                num_gray = num_gray + 1

        p_value = stats.binom_test(num_gray, num_samples, p=0.5)
        random.setstate(random_state)
        assert p_value > 0.0001

        # Test Set 2: grayscale -> 1 channel grayscale
        random_state = random.getstate()
        random.seed(42)
        x_shape = [2, 2, 3]
        x_np = np.random.randint(0, 256, x_shape, np.uint8)
        x_pil = Image.fromarray(x_np, mode='RGB')
        x_pil_2 = x_pil.convert('L')
        gray_np = np.array(x_pil_2)

        num_samples = 250
        num_gray = 0
        for _ in range(num_samples):
            gray_pil_3 = transforms.RandomGrayscale(p=0.5)(x_pil_2)
            gray_np_3 = np.array(gray_pil_3)
            if np.array_equal(gray_np, gray_np_3):
                num_gray = num_gray + 1

        p_value = stats.binom_test(num_gray, num_samples, p=1.0)  # Note: grayscale is always unchanged
        random.setstate(random_state)
        assert p_value > 0.0001

        # Test set 3: Explicit tests
        x_shape = [2, 2, 3]
        x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
        x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)
        x_pil = Image.fromarray(x_np, mode='RGB')
        x_pil_2 = x_pil.convert('L')
        gray_np = np.array(x_pil_2)

        # Case 3a: RGB -> 3 channel grayscale (grayscaled)
        trans2 = transforms.RandomGrayscale(p=1.0)
        gray_pil_2 = trans2(x_pil)
        gray_np_2 = np.array(gray_pil_2)
        assert gray_pil_2.mode == 'RGB', 'mode should be RGB'
        assert gray_np_2.shape == tuple(x_shape), 'should be 3 channel'
        np.testing.assert_equal(gray_np_2[:, :, 0], gray_np_2[:, :, 1])
        np.testing.assert_equal(gray_np_2[:, :, 1], gray_np_2[:, :, 2])
        np.testing.assert_equal(gray_np, gray_np_2[:, :, 0])

        # Case 3b: RGB -> 3 channel grayscale (unchanged)
        trans2 = transforms.RandomGrayscale(p=0.0)
        gray_pil_2 = trans2(x_pil)
        gray_np_2 = np.array(gray_pil_2)
        assert gray_pil_2.mode == 'RGB', 'mode should be RGB'
        assert gray_np_2.shape == tuple(x_shape), 'should be 3 channel'
        np.testing.assert_equal(x_np, gray_np_2)

        # Case 3c: 1 channel grayscale -> 1 channel grayscale (grayscaled)
        trans3 = transforms.RandomGrayscale(p=1.0)
        gray_pil_3 = trans3(x_pil_2)
        gray_np_3 = np.array(gray_pil_3)
        assert gray_pil_3.mode == 'L', 'mode should be L'
        assert gray_np_3.shape == tuple(x_shape[0:2]), 'should be 1 channel'
        np.testing.assert_equal(gray_np, gray_np_3)

        # Case 3d: 1 channel grayscale -> 1 channel grayscale (unchanged)
        trans3 = transforms.RandomGrayscale(p=0.0)
        gray_pil_3 = trans3(x_pil_2)
        gray_np_3 = np.array(gray_pil_3)
        assert gray_pil_3.mode == 'L', 'mode should be L'
        assert gray_np_3.shape == tuple(x_shape[0:2]), 'should be 1 channel'
        np.testing.assert_equal(gray_np, gray_np_3)

        # Checking if RandomGrayscale can be printed as string
        trans3.__repr__()


if __name__ == '__main__':
    unittest.main()
