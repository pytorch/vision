import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import unittest
import random

class Tester(unittest.TestCase):
    def test_crop(self):
        height = random.randint(10, 32) * 2
        width = random.randint(10, 32) * 2
        oheight = random.randint(5, (height - 2) / 2) * 2
        owidth =  random.randint(5, (width - 2) / 2) * 2
        
        img = torch.ones(3, height, width)
        oh1 = (height - oheight) / 2
        ow1 = (width - owidth) / 2
        imgnarrow = img[:, oh1 :oh1 + oheight, ow1 :ow1 + owidth]
        imgnarrow.fill_(0)
        result = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop((oheight, owidth)),
            transforms.ToTensor(),
        ])(img)
        assert result.sum() == 0, "height: " + str(height) + " width: " \
            + str( width) + " oheight: " + str(oheight) + " owidth: " + str(owidth)
        oheight += 1
        owidth += 1
        result = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop((oheight, owidth)),
            transforms.ToTensor(),
        ])(img)
        sum1 = result.sum()
        assert sum1 > 1, "height: " + str(height) + " width: " \
            + str( width) + " oheight: " + str(oheight) + " owidth: " + str(owidth)
        oheight += 1
        owidth += 1        
        result = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop((oheight, owidth)),
            transforms.ToTensor(),
        ])(img)
        sum2 = result.sum()
        assert sum2 > 0, "height: " + str(height) + " width: " \
            + str( width) + " oheight: " + str(oheight) + " owidth: " + str(owidth)
        assert sum2 > sum1, "height: " + str(height) + " width: " \
            + str( width) + " oheight: " + str(oheight) + " owidth: " + str(owidth)

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
        owidth =  random.randint(5, (width - 2) / 2) * 2
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
        assert result.size(1) == height + 2*padding
        assert result.size(2) == width + 2*padding

    def test_lambda(self):
        trans = transforms.Lambda(lambda x: x.add(10))
        x = torch.randn(10)
        y = trans(x)
        assert(y.equal(torch.add(x, 10)))

        trans = transforms.Lambda(lambda x: x.add_(10))
        x = torch.randn(10)
        y = trans(x)
        assert(y.equal(x))
        

if __name__ == '__main__':
    unittest.main()
