import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import unittest
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

GRACE_HOPPER = 'assets/grace_hopper_517x606.jpg'


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
                if should_vflip:
                    vflipped_img = img.transpose(Image.FLIP_TOP_BOTTOM)
                    expected_output += five_crop(vflipped_img)
                else:
                    hflipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    expected_output += five_crop(hflipped_img)

                assert len(results) == 10
                assert expected_output == results

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

    def test_to_tensor(self):
        test_channels = [1, 3, 4]
        height, width = 4, 4
        trans = transforms.ToTensor()

        for channels in test_channels:
            input_data = torch.ByteTensor(channels, height, width).random_(0, 255).float().div_(255)
            img = transforms.ToPILImage()(input_data)
            output = trans(img)
            assert np.allclose(input_data.numpy(), output.numpy())

            ndarray = np.random.randint(low=0, high=255, size=(height, width, channels))
            output = trans(ndarray)
            expected_output = ndarray.transpose((2, 0, 1)) / 255.0
            assert np.allclose(output.numpy(), expected_output)

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
            # should raise if we try a mode for 4 or 1 channel images
            transforms.ToPILImage(mode='RGBA')(img_data)
            transforms.ToPILImage(mode='P')(img_data)

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

        with self.assertRaises(ValueError):
            # should raise if we try a mode for 4 or 1 channel images
            transforms.ToPILImage(mode='RGBA')(img_data)
            transforms.ToPILImage(mode='P')(img_data)

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
        for mode in [None, 'RGBA', 'CMYK']:
            verify_img_data(img_data, expected_output, mode)

        with self.assertRaises(ValueError):
            # should raise if we try a mode for 3 or 1 channel images
            transforms.ToPILImage(mode='RGB')(img_data)
            transforms.ToPILImage(mode='P')(img_data)

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
        for mode in [None, 'RGBA', 'CMYK']:
            verify_img_data(img_data, mode)

        with self.assertRaises(ValueError):
            # should raise if we try a mode for 3 or 1 channel images
            transforms.ToPILImage(mode='RGB')(img_data)
            transforms.ToPILImage(mode='P')(img_data)

    def test_ndarray_bad_types_to_pil_image(self):
        trans = transforms.ToPILImage()
        with self.assertRaises(TypeError):
            trans(np.ones([4, 4, 1], np.int64))
            trans(np.ones([4, 4, 1], np.uint16))
            trans(np.ones([4, 4, 1], np.uint32))
            trans(np.ones([4, 4, 1], np.float64))

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

    @unittest.skipIf(stats is None, 'scipt.stats is not available')
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

    def test_translate(self):
        x = np.zeros((100, 100, 1), dtype=np.uint8)
        x[40, 40] = 255

        with self.assertRaises(TypeError):
            F.translate(x, horizontal=10)

        img = F.to_pil_image(x)

        result = F.translate(img, horizontal=10)
        assert result.size == (100, 100)
        r, c = np.where(result)
        np.testing.assert_equal(r, 40)
        np.testing.assert_equal(c, 30)

        result = F.translate(img, horizontal=-10)
        assert result.size == (100, 100)
        r, c = np.where(result)
        np.testing.assert_equal(r, 40)
        np.testing.assert_equal(c, 50)

        result = F.translate(img, vertical=10)
        assert result.size == (100, 100)
        r, c = np.where(result)
        np.testing.assert_equal(r, 30)
        np.testing.assert_equal(c, 40)

        result = F.translate(img, vertical=-10)
        assert result.size == (100, 100)
        r, c = np.where(result)
        np.testing.assert_equal(r, 50)
        np.testing.assert_equal(c, 40)

        result = F.translate(img, horizontal=10, vertical=-10)
        assert result.size == (100, 100)
        r, c = np.where(result)
        np.testing.assert_equal(r, 50)
        np.testing.assert_equal(c, 30)

    def test_random_translation(self):

        with self.assertRaises(ValueError):
            transforms.RandomTranslation(horizontal=-10)
            transforms.RandomTranslation(horizontal=[-10])
            transforms.RandomTranslation(horizontal=[-10, 0, 10])
            transforms.RandomTranslation(vertical=-10)
            transforms.RandomTranslation(vertical=[-10])
            transforms.RandomTranslation(vertical=[-10, 0, 10])

        t = transforms.RandomTranslation(horizontal=10, vertical=10)
        h, v = t.get_params(t.horizontal, t.vertical)
        assert h > -10 and h < 10
        assert v > -10 and v < 10

        t = transforms.RandomTranslation(horizontal=(-10, 10), vertical=(-10, 10))
        h, v = t.get_params(t.horizontal, t.vertical)
        assert h > -10 and h < 10
        assert v > -10 and v < 10

if __name__ == '__main__':
    unittest.main()
