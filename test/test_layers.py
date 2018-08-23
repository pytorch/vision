import torch
from torch.autograd import gradcheck

from torchvision import layers


import unittest


class ROIPoolTester(unittest.TestCase):

    def test_roi_pool_basic_cpu(self):
        dtype = torch.float32
        device = torch.device('cpu')
        x = torch.rand(1, 1, 10, 10, dtype=dtype, device=device)
        rois = torch.tensor([[0, 0, 0, 4, 4]],  # format is (xyxy)
                            dtype=dtype, device=device)

        pool_h, pool_w = (5, 5)
        roi_pool = layers.ROIPool((pool_h, pool_w), 1)
        y = roi_pool(x, rois)

        gt_y = torch.zeros(rois.size(0), x.size(1), pool_h, pool_w)

        for n in range(0, gt_y.size(0)):
            start_h, end_h = int(rois[n, 2].item()), int(rois[n, 4].item()) + 1
            start_w, end_w = int(rois[n, 1].item()), int(rois[n, 3].item()) + 1
            roi_x = x[:, :, start_h:end_h, start_w:end_w]
            bin_h, bin_w = roi_x.size(2) // pool_h, roi_x.size(3) // pool_w
            for j in range(0, pool_h):
                for i in range(0, pool_w):
                    gt_y[n, :, j, i] = torch.max(roi_x[:, :, j * bin_h:(j + 1) * bin_h, i * bin_w:(i + 1) * bin_w])

        assert torch.equal(gt_y, y), 'ROIPool layer incorrect'

    def test_roi_pool_cpu(self):
        dtype = torch.float32
        device = torch.device('cpu')
        x = torch.rand(2, 1, 10, 10, dtype=dtype, device=device)
        rois = torch.tensor([[0, 0, 0, 9, 9],  # format is (xyxy)
                             [0, 0, 5, 4, 9],
                             [0, 5, 5, 9, 9],
                             [1, 0, 0, 9, 9]],
                            dtype=dtype, device=device)

        pool_h, pool_w = (5, 5)
        roi_pool = layers.ROIPool((pool_h, pool_w), 1)
        y = roi_pool(x, rois)

        gt_y = torch.zeros(rois.size(0), x.size(1), pool_h, pool_w, device=device)
        for n in range(0, gt_y.size(0)):
            for r, roi in enumerate(rois):
                if roi[0] == n:
                    start_h, end_h = int(roi[2].item()), int(roi[4].item()) + 1
                    start_w, end_w = int(roi[1].item()), int(roi[3].item()) + 1
                    roi_x = x[roi[0].long():roi[0].long() + 1, :, start_h:end_h, start_w:end_w]
                    bin_h, bin_w = roi_x.size(2) // pool_h, roi_x.size(3) // pool_w
                    for j in range(0, pool_h):
                        for i in range(0, pool_w):
                            gt_y[r, :, j, i] = torch.max(gt_y[r, :, j, i],
                                                         torch.max(roi_x[:, :,
                                                                         j * bin_h:(j + 1) * bin_h,
                                                                         i * bin_w:(i + 1) * bin_w])
                                                         )

        assert torch.equal(gt_y, y), 'ROIPool layer incorrect'

    def test_roi_pool_gradient_cpu(self):
        dtype = torch.float32
        device = torch.device('cpu')
        layer = layers.ROIPool((5, 5), 1).to(dtype=dtype, device=device)
        x = torch.ones(1, 1, 10, 10, dtype=dtype, device=device, requires_grad=True)
        cx = torch.ones(1, 1, 10, 10, dtype=dtype, requires_grad=True).cuda()
        rois = torch.tensor([
            [0, 0, 0, 9, 9],
            [0, 0, 5, 4, 9],
            [0, 0, 0, 4, 4]],
            dtype=dtype, device=device)

        y = layer(x, rois)
        s = y.sum()
        s.backward()

        gt_grad = torch.tensor([[[[2., 1., 2., 1., 2., 0., 1., 0., 1., 0.],
                                  [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
                                  [2., 1., 2., 1., 2., 0., 1., 0., 1., 0.],
                                  [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
                                  [2., 1., 2., 1., 2., 0., 1., 0., 1., 0.],
                                  [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
                                  [2., 1., 2., 1., 2., 0., 1., 0., 1., 0.],
                                  [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
                                  [2., 1., 2., 1., 2., 0., 1., 0., 1., 0.],
                                  [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.]]]], device=device)

        assert torch.equal(x.grad, gt_grad), 'gradient incorrect for roi_pool'

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_roi_pool_basic_gpu(self):
        dtype = torch.float32
        device = torch.device('cuda')
        x = torch.rand(1, 1, 10, 10, dtype=dtype, device=device)
        rois = torch.tensor([[0, 0, 0, 4, 4]],  # format is (xyxy)
                            dtype=dtype, device=device)

        pool_h, pool_w = (5, 5)
        roi_pool = layers.ROIPool((pool_h, pool_w), 1)
        y = roi_pool(x, rois)

        gt_y = torch.zeros(rois.size(0), x.size(1), pool_h, pool_w)

        for n in range(0, gt_y.size(0)):
            start_h, end_h = int(rois[n, 2].item()), int(rois[n, 4].item()) + 1
            start_w, end_w = int(rois[n, 1].item()), int(rois[n, 3].item()) + 1
            roi_x = x[:, :, start_h:end_h, start_w:end_w]
            bin_h, bin_w = roi_x.size(2) // pool_h, roi_x.size(3) // pool_w
            for j in range(0, pool_h):
                for i in range(0, pool_w):
                    gt_y[n, :, j, i] = torch.max(roi_x[:, :, j * bin_h:(j + 1) * bin_h, i * bin_w:(i + 1) * bin_w])

        assert torch.equal(gt_y.cuda(), y), 'ROIPool layer incorrect'

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_roi_pool_gpu(self):
        dtype = torch.float32
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        x = torch.rand(2, 1, 10, 10, dtype=dtype, device=device)
        rois = torch.tensor([[0, 0, 0, 9, 9],  # format is (xyxy)
                             [0, 0, 5, 4, 9],
                             [0, 5, 5, 9, 9],
                             [1, 0, 0, 9, 9]],
                            dtype=dtype, device=device)

        pool_h, pool_w = (5, 5)
        roi_pool = layers.ROIPool((pool_h, pool_w), 1)
        y = roi_pool(x, rois)

        gt_y = torch.zeros(rois.size(0), x.size(1), pool_h, pool_w, device=device)
        for n in range(0, gt_y.size(0)):
            for r, roi in enumerate(rois):
                if roi[0] == n:
                    start_h, end_h = int(roi[2].item()), int(roi[4].item()) + 1
                    start_w, end_w = int(roi[1].item()), int(roi[3].item()) + 1
                    roi_x = x[roi[0].long():roi[0].long() + 1, :, start_h:end_h, start_w:end_w]
                    bin_h, bin_w = roi_x.size(2) // pool_h, roi_x.size(3) // pool_w
                    for j in range(0, pool_h):
                        for i in range(0, pool_w):
                            gt_y[r, :, j, i] = torch.max(gt_y[r, :, j, i],
                                                         torch.max(roi_x[:, :,
                                                                         j * bin_h:(j + 1) * bin_h,
                                                                         i * bin_w:(i + 1) * bin_w])
                                                         )

        assert torch.equal(gt_y.cuda(), y), 'ROIPool layer incorrect'

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_roi_pool_gradient_gpu(self):
        dtype = torch.float32
        device = torch.device('cuda')
        layer = layers.ROIPool((5, 5), 1).to(dtype=dtype, device=device)
        x = torch.ones(1, 1, 10, 10, dtype=dtype, device=device, requires_grad=True)
        rois = torch.tensor([
            [0, 0, 0, 9, 9],
            [0, 0, 5, 4, 9],
            [0, 0, 0, 4, 4]],
            dtype=dtype, device=device)

        def func(input):
            return layer(input, rois)

        x.requires_grad = True
        y = layer(x, rois)
        # print(argmax, argmax.shape)
        s = y.sum()
        s.backward()
        gt_grad = torch.tensor([[[[2., 1., 2., 1., 2., 0., 1., 0., 1., 0.],
                                  [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
                                  [2., 1., 2., 1., 2., 0., 1., 0., 1., 0.],
                                  [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
                                  [2., 1., 2., 1., 2., 0., 1., 0., 1., 0.],
                                  [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
                                  [2., 1., 2., 1., 2., 0., 1., 0., 1., 0.],
                                  [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
                                  [2., 1., 2., 1., 2., 0., 1., 0., 1., 0.],
                                  [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.]]]], device=device)

        assert torch.equal(x.grad, gt_grad), 'gradient incorrect for roi_pool'


if __name__ == '__main__':
    unittest.main()
