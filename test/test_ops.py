import numpy as np
import torch
from torch.autograd import gradcheck

from torchvision import ops

from itertools import product
import unittest


class RoIPoolTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dtype = torch.float64

    def slow_roi_pooling(self, x, rois, pool_h, pool_w, spatial_scale=1,
                         device=torch.device('cpu'), dtype=torch.float64):
        c = x.size(1)
        y = torch.zeros(rois.size(0), c, pool_h, pool_w, dtype=dtype, device=device)

        rois = torch.round(rois * spatial_scale)

        for n in range(0, y.size(0)):
            for r, roi in enumerate(rois):
                if roi[0] == n:
                    start_h, end_h = int(roi[2].item()), int(roi[4].item()) + 1
                    start_w, end_w = int(roi[1].item()), int(roi[3].item()) + 1
                    roi_x = x[roi[0].long(), :, start_h:end_h, start_w:end_w]
                    bin_h, bin_w = roi_x.size(-2) / float(pool_h), roi_x.size(-1) / float(pool_w)

                    for j in range(0, pool_h):
                        cj = slice(int(np.floor(j * bin_h)), int(np.ceil((j + 1) * bin_h)))
                        for i in range(0, pool_w):
                            ci = slice(int(np.floor(i * bin_w)), int(np.ceil((i + 1) * bin_w)))
                            t = roi_x[:, cj, ci].reshape(c, -1)
                            if t.numel() > 0:
                                y[r, :, j, i] = torch.max(t, 1)[0]
        return y

    def test_roi_pool_basic_cpu(self):
        device = torch.device('cpu')
        x = torch.rand(1, 1, 10, 10, dtype=self.dtype, device=device)
        rois = torch.tensor([[0, 0, 0, 4, 4]],  # format is (xyxy)
                            dtype=self.dtype, device=device)

        pool_h, pool_w = (5, 5)
        roi_pool = ops.RoIPool((pool_h, pool_w), 1)
        y = roi_pool(x, rois)

        gt_y = self.slow_roi_pooling(x, rois, pool_h, pool_w, device=device, dtype=self.dtype)

        assert torch.allclose(gt_y, y), 'RoIPool layer incorrect on CPU'

        # non-contiguous
        y = roi_pool(x.permute(0, 1, 3, 2), rois)
        gt_y = self.slow_roi_pooling(x.permute(0, 1, 3, 2), rois, pool_h, pool_w, device=device, dtype=self.dtype)
        assert torch.allclose(gt_y, y), 'RoIPool layer incorrect on CPU'

    def test_roi_pool_cpu(self):
        device = torch.device('cpu')
        x = torch.rand(2, 1, 10, 10, dtype=self.dtype, device=device)
        rois = torch.tensor([[0, 0, 0, 9, 9],  # format is (xyxy)
                             [0, 0, 5, 4, 9],
                             [0, 5, 5, 9, 9],
                             [1, 0, 0, 9, 9]],
                            dtype=self.dtype, device=device)

        pool_h, pool_w = (5, 5)
        roi_pool = ops.RoIPool((pool_h, pool_w), 1)
        y = roi_pool(x, rois)

        gt_y = self.slow_roi_pooling(x, rois, pool_h, pool_w, device=device, dtype=self.dtype)

        assert torch.allclose(gt_y, y), 'RoIPool layer incorrect on CPU for batch > 1'

        # non-contiguous
        y = roi_pool(x.permute(0, 1, 3, 2), rois)
        gt_y = self.slow_roi_pooling(x.permute(0, 1, 3, 2), rois, pool_h, pool_w, device=device, dtype=self.dtype)
        assert torch.allclose(gt_y, y), 'RoIPool layer incorrect on CPU for batch > 1'

    def test_roi_pool_cpu_empty_rois(self):
        device = torch.device('cpu')
        x = torch.tensor(
            [[[[0.1767, 1.2851, 4.2325, 4.8645, 7.1496]],
              [[2.5916, 4.3361, 3.8143, 6.1329, 2.0230]],
              [[1.4492, 3.3384, 4.0816, 6.3116, 5.1068]]]],
            dtype=self.dtype, device=device)
        rois = torch.tensor(
            [[0., 1., 0., 4., 0.],
             [0., 2., 0., 3., 0.],
             [0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.],
             [0., 2., 0., 2., 0.]],
            dtype=self.dtype, device=device)

        pool_h, pool_w = (1, 2)
        roi_pool = ops.RoIPool((pool_h, pool_w), 1)
        y = roi_pool(x, rois)

        gt_y = self.slow_roi_pooling(x, rois, pool_h, pool_w, device=device, dtype=self.dtype)

        assert torch.allclose(gt_y, y), 'RoIPool layer incorrect on CPU empty rois'

        # non-contiguous
        y = roi_pool(x.permute(0, 1, 3, 2), rois)
        gt_y = self.slow_roi_pooling(x.permute(0, 1, 3, 2), rois, pool_h, pool_w, device=device, dtype=self.dtype)
        assert torch.allclose(gt_y, y), 'RoIPool layer incorrect on CPU for empty rois non-contiguous'

    def test_roi_pool_gradient_cpu(self):
        device = torch.device('cpu')
        x = torch.ones(1, 1, 10, 10, dtype=self.dtype, device=device, requires_grad=True)
        rois = torch.tensor([
            [0, 0, 0, 9, 9],
            [0, 0, 5, 4, 9],
            [0, 0, 0, 4, 4]],
            dtype=self.dtype, device=device)

        layer = ops.RoIPool((5, 5), 1).to(dtype=self.dtype, device=device)

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
                                  [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.]]]],
                               device=device, dtype=self.dtype)

        assert torch.allclose(x.grad, gt_grad), 'gradient incorrect for roi_pool'

    def test_roi_pool_align_non_cont_grad_cpu(self):
        devices = ['cpu']
        if torch.cuda.is_available():
            devices.append('cuda')

        for d in devices:
            device = torch.device(d)
            rois = torch.tensor([
                [0, 0, 0, 9, 9],
                [0, 0, 5, 5, 9],
                [0, 5, 5, 9, 9]], dtype=self.dtype, device=device)

            grad_cont = torch.rand(3, 1, 5, 5, dtype=self.dtype, device=device)
            grad = grad_cont.permute(2, 1, 3, 0).contiguous().permute(3, 1, 0, 2)

            for op in ['RoIPool', 'RoIAlign']:
                x = torch.rand(1, 1, 10, 10, dtype=self.dtype, device=device, requires_grad=True)
                kwargs = {}
                if op == 'RoIAlign':
                    kwargs['sampling_ratio'] = 1
                m = getattr(ops, op)((5, 5), 1, **kwargs)

                y = m(x, rois)
                y.backward(grad_cont)

                g1 = x.grad.detach().clone()
                del x.grad

                y = m(x, rois)
                y.backward(grad)

                g2 = x.grad.detach().clone()
                del x.grad
                assert torch.allclose(g1, g2), 'gradient incorrect for {}'.format(op)

    def test_roi_pool_gradcheck_cpu(self):
        device = torch.device('cpu')
        x = torch.rand(1, 1, 10, 10, dtype=self.dtype, device=device, requires_grad=True)
        rois = torch.tensor([
            [0, 0, 0, 9, 9],
            [0, 0, 5, 5, 9],
            [0, 5, 5, 9, 9]], dtype=self.dtype, device=device)

        m = ops.RoIPool((5, 5), 1).to(dtype=self.dtype, device=device)

        def func(input):
            return m(input, rois)

        assert gradcheck(func, (x,)), 'gradcheck failed for roi_pool CPU'
        assert gradcheck(func, (x.permute(0, 1, 3, 2),)), 'gradcheck failed for roi_pool CPU'

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_roi_pool_basic_cuda(self):
        device = torch.device('cuda')
        x = torch.rand(1, 1, 10, 10, dtype=self.dtype, device=device)
        rois = torch.tensor([[0, 0, 0, 4, 4]],  # format is (xyxy)
                            dtype=self.dtype, device=device)

        pool_h, pool_w = (5, 5)
        roi_pool = ops.RoIPool((pool_h, pool_w), 1)
        y = roi_pool(x, rois)

        gt_y = self.slow_roi_pooling(x, rois, pool_h, pool_w, device=device, dtype=self.dtype)

        assert torch.allclose(gt_y.cuda(), y), 'RoIPool layer incorrect'

        y = roi_pool(x.permute(0, 1, 3, 2), rois)
        gt_y = self.slow_roi_pooling(x.permute(0, 1, 3, 2), rois, pool_h, pool_w, device=device, dtype=self.dtype)
        assert torch.allclose(gt_y.cuda(), y), 'RoIPool layer incorrect'

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_roi_pool_cuda(self):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        x = torch.rand(2, 1, 10, 10, dtype=self.dtype, device=device)
        rois = torch.tensor([[0, 0, 0, 9, 9],  # format is (xyxy)
                             [0, 0, 5, 4, 9],
                             [0, 5, 5, 9, 9],
                             [1, 0, 0, 9, 9]],
                            dtype=self.dtype, device=device)

        pool_h, pool_w = (5, 5)
        roi_pool = ops.RoIPool((pool_h, pool_w), 1)
        y = roi_pool(x, rois)

        gt_y = self.slow_roi_pooling(x, rois, pool_h, pool_w, device=device, dtype=self.dtype)

        assert torch.allclose(gt_y.cuda(), y), 'RoIPool layer incorrect'

        y = roi_pool(x.permute(0, 1, 3, 2), rois)
        gt_y = self.slow_roi_pooling(x.permute(0, 1, 3, 2), rois, pool_h, pool_w, device=device, dtype=self.dtype)
        assert torch.allclose(gt_y.cuda(), y), 'RoIPool layer incorrect'

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_roi_pool_gradient_cuda(self):
        device = torch.device('cuda')
        layer = ops.RoIPool((5, 5), 1).to(dtype=self.dtype, device=device)
        x = torch.ones(1, 1, 10, 10, dtype=self.dtype, device=device, requires_grad=True)
        rois = torch.tensor([
            [0, 0, 0, 9, 9],
            [0, 0, 5, 4, 9],
            [0, 0, 0, 4, 4]],
            dtype=self.dtype, device=device)

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
                                  [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.]]]],
                               device=device, dtype=self.dtype)

        assert torch.allclose(x.grad, gt_grad), 'gradient incorrect for roi_pool'

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_roi_pool_gradcheck_cuda(self):
        device = torch.device('cuda')
        x = torch.rand(1, 1, 10, 10, dtype=self.dtype, device=device, requires_grad=True)
        rois = torch.tensor([
            [0, 0, 0, 9, 9],
            [0, 0, 5, 5, 9],
            [0, 5, 5, 9, 9]], dtype=self.dtype, device=device)

        m = ops.RoIPool((5, 5), 1).to(dtype=self.dtype, device=device)

        def func(input):
            return m(input, rois)

        assert gradcheck(func, (x,)), 'gradcheck failed for roi_pool CUDA'
        assert gradcheck(func, (x.permute(0, 1, 3, 2),)), 'gradcheck failed for roi_pool CUDA'


class RoIAlignTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(123)
        cls.dtype = torch.float32
        cls.x = torch.rand(1, 1, 10, 10, dtype=cls.dtype)
        cls.single_roi = torch.tensor([[0, 0, 0, 4, 4]],  # format is (xyxy)
                                      dtype=cls.dtype)
        cls.rois = torch.tensor([[0, 0, 0, 9, 9],  # format is (xyxy)
                                 [0, 0, 5, 4, 9],
                                 [0, 5, 5, 9, 9]],
                                dtype=cls.dtype)

        cls.gt_y_single = torch.tensor(
            [[[[0.41617328, 0.5040753, 0.25266218, 0.4296828, 0.29928464],
               [0.5210769, 0.57222337, 0.2524979, 0.32063985, 0.32635176],
               [0.73108256, 0.6114335, 0.62033176, 0.8188273, 0.5562218],
               [0.83115816, 0.70803946, 0.7084047, 0.74928707, 0.7769296],
               [0.54266506, 0.45964524, 0.5780159, 0.80522037, 0.7321807]]]], dtype=cls.dtype)

        cls.gt_y_multiple = torch.tensor(
            [[[[0.49311584, 0.35972416, 0.40843594, 0.3638034, 0.49751836],
               [0.70881474, 0.75481665, 0.5826779, 0.34767765, 0.46865487],
               [0.4740328, 0.69306874, 0.3617804, 0.47145438, 0.66130304],
               [0.6861706, 0.17634538, 0.47194335, 0.42473823, 0.37930614],
               [0.62666404, 0.49973848, 0.37911576, 0.5842756, 0.7176864]]],
             [[[0.67499936, 0.6607055, 0.42656037, 0.46134934, 0.42144877],
               [0.7471722, 0.7235433, 0.14512213, 0.13031253, 0.289369],
               [0.8443615, 0.6659734, 0.23614208, 0.14719573, 0.4268827],
               [0.69429564, 0.5621515, 0.5019923, 0.40678093, 0.34556213],
               [0.51315194, 0.7177093, 0.6494485, 0.6775592, 0.43865064]]],
             [[[0.24465509, 0.36108392, 0.64635646, 0.4051828, 0.33956185],
               [0.49006107, 0.42982674, 0.34184104, 0.15493104, 0.49633422],
               [0.54400194, 0.5265246, 0.22381854, 0.3929715, 0.6757667],
               [0.32961223, 0.38482672, 0.68877804, 0.71822757, 0.711909],
               [0.561259, 0.71047884, 0.84651315, 0.8541089, 0.644432]]]], dtype=cls.dtype)

        cls.x_grad = torch.tensor(
            [[[[0.075625, 0.15125, 0.15124999, 0.15125002, 0.15812504,
                0.15812503, 0.15124999, 0.15124999, 0.15125006, 0.0756249],
               [0.15125, 0.30250007, 0.3025, 0.30250007, 0.31625012,
                0.31625003, 0.3025, 0.3025, 0.30250013, 0.1512498],
               [0.15124999, 0.3025, 0.30249995, 0.3025, 0.31625006,
                0.31625, 0.30249995, 0.30249995, 0.30250007, 0.15124978],
               [0.15125002, 0.30250007, 0.3025, 0.30250007, 0.31625012,
                0.3162501, 0.3025, 0.3025, 0.30250013, 0.15124981],
               [0.15812504, 0.31625012, 0.31625006, 0.31625012, 0.33062524,
                0.3306251, 0.31625006, 0.31625006, 0.3162502, 0.15812483],
               [0.5181251, 1.0962502, 1.0362502, 1.0962503, 0.69062525, 0.6906252,
                1.0962502, 1.0362502, 1.0962503, 0.5181248],
               [0.93125, 1.9925, 1.8624997, 1.9925, 1.0962502, 1.0962502,
                1.9925, 1.8624998, 1.9925, 0.9312496],
               [0.8712501, 1.8625, 1.7425002, 1.8625001, 1.0362502, 1.0362502,
                1.8625, 1.7425001, 1.8625002, 0.8712497],
               [0.93125004, 1.9925, 1.8625002, 1.9925, 1.0962503, 1.0962503,
                1.9925001, 1.8625001, 1.9925001, 0.93124974],
               [0.43562484, 0.9312497, 0.8712497, 0.9312497, 0.5181249, 0.5181248,
                0.9312496, 0.8712497, 0.93124974, 0.43562466]]]], dtype=cls.dtype)

    def test_roi_align_basic_cpu(self):
        device = torch.device('cpu')
        x = self.x.to(device)
        single_roi = self.single_roi.to(device)
        gt_y_single = self.gt_y_single.to(device)

        pool_h, pool_w = (5, 5)
        roi_align = ops.RoIAlign((pool_h, pool_w), spatial_scale=1, sampling_ratio=2).to(device=device)
        y = roi_align(x, single_roi)

        assert torch.allclose(gt_y_single, y), 'RoIAlign layer incorrect for single ROI on CPU'

        y = roi_align(x.transpose(2, 3).contiguous().transpose(2, 3), single_roi)
        assert torch.allclose(gt_y_single, y), 'RoIAlign layer incorrect for single ROI on CPU'

    def test_roi_align_cpu(self):
        device = torch.device('cpu')
        x = self.x.to(device)
        rois = self.rois.to(device)
        gt_y_multiple = self.gt_y_multiple.to(device)

        pool_h, pool_w = (5, 5)
        roi_align = ops.RoIAlign((pool_h, pool_w), spatial_scale=1, sampling_ratio=2).to(device=device)
        y = roi_align(x, rois)

        assert torch.allclose(gt_y_multiple, y), 'RoIAlign layer incorrect for multiple ROIs on CPU'

        y = roi_align(x.transpose(2, 3).contiguous().transpose(2, 3), rois)
        assert torch.allclose(gt_y_multiple, y), 'RoIAlign layer incorrect for multiple ROIs on CPU'

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_roi_align_basic_cuda(self):
        device = torch.device('cuda')
        x = self.x.to(device)
        single_roi = self.single_roi.to(device)
        gt_y_single = self.gt_y_single.to(device)

        pool_h, pool_w = (5, 5)
        roi_align = ops.RoIAlign((pool_h, pool_w), spatial_scale=1, sampling_ratio=2).to(device=device)
        y = roi_align(x, single_roi)

        assert torch.allclose(gt_y_single, y), 'RoIAlign layer incorrect for single ROI on CUDA'

        y = roi_align(x.transpose(2, 3).contiguous().transpose(2, 3), single_roi)
        assert torch.allclose(gt_y_single, y), 'RoIAlign layer incorrect for single ROI on CUDA'

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_roi_align_cuda(self):
        device = torch.device('cuda')
        x = self.x.to(device)
        rois = self.rois.to(device)
        gt_y_multiple = self.gt_y_multiple.to(device)

        pool_h, pool_w = (5, 5)
        roi_align = ops.RoIAlign((pool_h, pool_w), spatial_scale=1, sampling_ratio=2).to(device=device)
        y = roi_align(x, rois)

        assert torch.allclose(gt_y_multiple, y), 'RoIAlign layer incorrect for multiple ROIs on CUDA'

        y = roi_align(x.transpose(2, 3).contiguous().transpose(2, 3), rois)
        assert torch.allclose(gt_y_multiple, y), 'RoIAlign layer incorrect for multiple ROIs on CUDA'

    def test_roi_align_gradient_cpu(self):
        """
        Compute gradients for RoIAlign with multiple bounding boxes on CPU
        """
        device = torch.device('cpu')
        pool_h, pool_w = (5, 5)
        roi_align = ops.RoIAlign((pool_h, pool_w), spatial_scale=1, sampling_ratio=2).to(device=device)

        x = self.x.to(device).clone()
        rois = self.rois.to(device)
        gt_grad = self.x_grad.to(device)

        x.requires_grad = True
        y = roi_align(x, rois)
        s = y.sum()
        s.backward()

        assert torch.allclose(x.grad, gt_grad), 'gradient incorrect for RoIAlign CPU'

    def test_roi_align_gradcheck_cpu(self):
        dtype = torch.float64
        device = torch.device('cpu')
        m = ops.RoIAlign((5, 5), 0.5, 1).to(dtype=dtype, device=device)
        x = torch.rand(1, 1, 10, 10, dtype=dtype, device=device, requires_grad=True)
        rois = self.rois.to(device=device, dtype=dtype)

        def func(input):
            return m(input, rois)

        assert gradcheck(func, (x,)), 'gradcheck failed for RoIAlign CPU'
        assert gradcheck(func, (x.transpose(2, 3),)), 'gradcheck failed for RoIAlign CPU'

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_roi_align_gradient_cuda(self):
        """
        Compute gradients for RoIAlign with multiple bounding boxes on the GPU
        """
        device = torch.device('cuda')
        pool_h, pool_w = (5, 5)
        roi_align = ops.RoIAlign((pool_h, pool_w), spatial_scale=1, sampling_ratio=2).to(device=device)

        x = self.x.to(device).clone()
        rois = self.rois.to(device)
        gt_grad = self.x_grad.to(device)

        x.requires_grad = True
        y = roi_align(x, rois)
        s = y.sum()
        s.backward()

        assert torch.allclose(x.grad, gt_grad), 'gradient incorrect for RoIAlign CUDA'

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_roi_align_gradcheck_cuda(self):
        dtype = torch.float64
        device = torch.device('cuda')
        m = ops.RoIAlign((5, 5), 0.5, 1).to(dtype=dtype, device=device)
        x = torch.rand(1, 1, 10, 10, dtype=dtype, device=device, requires_grad=True)
        rois = self.rois.to(device=device, dtype=dtype)

        def func(input):
            return m(input, rois)

        assert gradcheck(func, (x,)), 'gradcheck failed for RoIAlign CUDA'
        assert gradcheck(func, (x.transpose(2, 3),)), 'gradcheck failed for RoIAlign CUDA'


class NMSTester(unittest.TestCase):
    def reference_nms(self, boxes, scores, iou_threshold):
        """
        Args:
            box_scores (N, 5): boxes in corner-form and probabilities.
            iou_threshold: intersection over union threshold.
        Returns:
             picked: a list of indexes of the kept boxes
        """
        picked = []
        _, indexes = scores.sort(descending=True)
        while len(indexes) > 0:
            current = indexes[0]
            picked.append(current.item())
            if len(indexes) == 1:
                break
            current_box = boxes[current, :]
            indexes = indexes[1:]
            rest_boxes = boxes[indexes, :]
            iou = ops.box_iou(rest_boxes, current_box.unsqueeze(0)).squeeze(1)
            indexes = indexes[iou <= iou_threshold]

        return torch.as_tensor(picked)

    def _create_tensors(self, N):
        boxes = torch.rand(N, 4) * 100
        boxes[:, 2:] += torch.rand(N, 2) * 100
        scores = torch.rand(N)
        return boxes, scores

    def test_nms(self):
        boxes, scores = self._create_tensors(1000)
        err_msg = 'NMS incompatible between CPU and reference implementation for IoU={}'
        for iou in [0.2, 0.5, 0.8]:
            keep_ref = self.reference_nms(boxes, scores, iou)
            keep = ops.nms(boxes, scores, iou)
            assert torch.allclose(keep, keep_ref), err_msg.format(iou)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_nms_cuda(self):
        boxes, scores = self._create_tensors(1000)
        err_msg = 'NMS incompatible between CPU and CUDA for IoU={}'

        for iou in [0.2, 0.5, 0.8]:
            r_cpu = ops.nms(boxes, scores, iou)
            r_cuda = ops.nms(boxes.cuda(), scores.cuda(), iou)

            assert torch.allclose(r_cpu, r_cuda.cpu()), err_msg.format(iou)


if __name__ == '__main__':
    unittest.main()
