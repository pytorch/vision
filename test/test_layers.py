import torch
from torch.autograd import gradcheck

from torchvision import layers


import unittest


class Tester(unittest.TestCase):

    def test_roi_align(self):
        outputs = []
        dtype = torch.float32
        x = torch.rand(1, 1, 10, 10, dtype=dtype)
        rois = torch.tensor([
            [0, 0, 0, 10, 10],
            [0, 0, 5, 5, 10],
            [0, 5, 5, 10, 10]], dtype=dtype)


        for device in ['cpu', 'cuda']:
            device = torch.device(device)
            x_n = x.to(device)
            rois_n = rois.to(device)
            output = layers.roi_align(x_n, rois_n, (5, 5), 0.5, 1).to('cpu')
            outputs.append(output)

        assert (outputs[0] - outputs[1]).abs().max() < 1e-6


    def test_roi_align_gradient(self):
        dtype = torch.float64
        device = torch.device('cuda')
        m = layers.ROIAlign((5, 5), 0.5, 1).to(dtype=dtype, device=device)
        x = torch.rand(1, 1, 10, 10, dtype=dtype, device=device)
        rois = torch.tensor([
            [0, 0, 0, 10, 10],
            [0, 0, 5, 5, 10],
            [0, 5, 5, 10, 10]], dtype=dtype, device=device)

        def func(input):
            return m(input, rois)

        assert gradcheck(func, (x,)), 'gradcheck failed for roi_align'


    def test_nms(self):
        boxes = torch.tensor([
            [0, 0, 100, 100],
            [2, 2, 98, 98],
            [50, 50, 200, 200],
            [50, 50, 200, 200]], dtype=torch.float32)
        scores = torch.tensor([1, 2, 0.5, 1], dtype=torch.float32)
        keep = layers.nms(boxes, scores, 0.5)
        assert keep.tolist() == [1, 3]

if __name__ == '__main__':
    unittest.main()
