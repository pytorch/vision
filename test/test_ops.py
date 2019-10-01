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
                         device=None, dtype=torch.float64):
        if device is None:
            device = torch.device("cpu")
        c = x.size(1)
        y = torch.zeros(rois.size(0), c, pool_h, pool_w, dtype=dtype, device=device)

        for n in range(0, x.size(0)):
            for r, roi in enumerate(rois):
                if roi[0] == n:
                    roi[1:] = torch.round(roi[1:] * spatial_scale)
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

        # spatial-scale != 1
        y = ops.RoIPool((pool_h, pool_w), 2)(x.permute(0, 1, 3, 2), rois)
        gt_y = self.slow_roi_pooling(x.permute(0, 1, 3, 2), rois, pool_h, pool_w,
                                     spatial_scale=2, device=device, dtype=self.dtype)
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

        @torch.jit.script
        def script_func(input, rois):
            return torch.ops.torchvision.roi_pool(input, rois, 1.0, 5, 5)[0]

        assert gradcheck(lambda x: script_func(x, rois), (x,)), 'gradcheck failed for scripted roi_pool'

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

        @torch.jit.script
        def script_func(input, rois):
            return torch.ops.torchvision.roi_pool(input, rois, 1.0, 5, 5)[0]

        assert gradcheck(lambda x: script_func(x, rois), (x,)), 'gradcheck failed for scripted roi_pool on CUDA'


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

        @torch.jit.script
        def script_func(input, rois):
            return torch.ops.torchvision.roi_align(input, rois, 0.5, 5, 5, 1)[0]

        assert gradcheck(lambda x: script_func(x, rois), (x,)), 'gradcheck failed for scripted roi_align'

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

        @torch.jit.script
        def script_func(input, rois):
            return torch.ops.torchvision.roi_align(input, rois, 0.5, 5, 5, 1)[0]

        assert gradcheck(lambda x: script_func(x, rois), (x,)), 'gradcheck failed for scripted roi_align on CUDA'


def bilinear_interpolate(data, height, width, y, x):
    if y < -1.0 or y > height or x < -1.0 or x > width:
        return 0.

    if y <= 0:
        y = 0.
    if x <= 0:
        x = 0.

    y_low, x_low = int(y), int(x)
    y_high, x_high = 0, 0

    if y_low >= height - 1:
        y_high = y_low = height - 1
        y = float(y_low)
    else:
        y_high = y_low + 1

    if x_low >= width - 1:
        x_high = x_low = width - 1
        x = float(x_low)
    else:
        x_high = x_low + 1

    ly = y - y_low
    lx = x - x_low
    hy, hx = 1. - ly, 1. - lx

    v1 = data[y_low * width + x_low]
    v2 = data[y_low * width + x_high]
    v3 = data[y_high * width + x_low]
    v4 = data[y_high * width + x_high]
    w1, w2, w3, w4 = hy * hx, hy * lx, ly * hx, ly * lx

    return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4


class PSRoIAlignTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dtype = torch.float64

    def slow_ps_roi_align(self, in_data, rois, pool_h, pool_w, device, spatial_scale=1,
                          sampling_ratio=-1, dtype=torch.float64):
        if device is None:
            device = torch.device("cpu")
        num_input_channels = in_data.size(1)
        assert num_input_channels % (pool_h * pool_w) == 0, "input channels must be divisible by ph * pw"
        num_output_channels = int(num_input_channels / (pool_h * pool_w))
        out_data = torch.zeros(rois.size(0), num_output_channels, pool_h, pool_w, dtype=dtype, device=device)

        for n in range(0, in_data.size(0)):
            for r, roi in enumerate(rois):
                if roi[0] != n:
                    continue
                roi[1:] = (roi[1:] * spatial_scale) - 0.5
                c_in = 0
                roi_height = float(roi[4].item() - roi[2].item())
                roi_width = float(roi[3].item() - roi[1].item())
                bin_h, bin_w = roi_height / float(pool_h), roi_width / float(pool_w)
                for c_out in range(0, num_output_channels):
                    for j in range(0, pool_h):
                        start_h = float(j) * bin_h + roi[2].item()

                        for i in range(0, pool_w):
                            start_w = float(i) * bin_w + roi[1].item()

                            roi_bin_grid_h = sampling_ratio if sampling_ratio > 0 else int(np.ceil(roi_height / pool_h))
                            roi_bin_grid_w = sampling_ratio if sampling_ratio > 0 else int(np.ceil(roi_width / pool_w))

                            val = 0.
                            for iy in range(0, roi_bin_grid_h):
                                y = start_h + (iy + 0.5) * bin_h / float(roi_bin_grid_h)
                                for ix in range(0, roi_bin_grid_w):
                                    x = start_w + (ix + 0.5) * bin_w / float(roi_bin_grid_w)
                                    val += bilinear_interpolate(
                                        in_data[n, c_in, :, :].flatten(),
                                        in_data.size(-2),
                                        in_data.size(-1),
                                        y, x
                                    )
                            count = roi_bin_grid_h * roi_bin_grid_w
                            out_data[r, c_out, j, i] = val / count
                            c_in += 1
        return out_data

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_ps_roi_align_basic_cuda(self):
        print("Testing test_ps_roi_align_basic_cuda")
        device = torch.device('cuda')
        pool_size = 3
        x = torch.rand(1, 2 * (pool_size ** 2), 7, 7, dtype=self.dtype, device=device)
        rois = torch.tensor([[0, 0, 0, 5, 5]],  # format is (xyxy)
                            dtype=self.dtype, device=device)

        pool_h, pool_w = (pool_size, pool_size)
        print("before creating PSROIAlign Layer")
        ps_roi_align = ops.PSRoIAlign((pool_h, pool_w), spatial_scale=1, sampling_ratio=2)
        print("after creating PSROIAlign Layer")
        print("x_is_cuda: {}".format(x.is_cuda))
        print("rois_is_cuda: {}".format(rois.is_cuda))
        y = ps_roi_align(x, rois)
        print("after feedforward of data in PSROIAlign")

        gt_y = self.slow_ps_roi_align(x, rois, pool_h, pool_w, device,
                                      spatial_scale=1, sampling_ratio=2,
                                      dtype=self.dtype)
        assert torch.allclose(gt_y.cuda(), y), 'PSRoIAlign layer incorrect'

        y = ps_roi_align(x.permute(0, 1, 3, 2), rois)
        gt_y = self.slow_ps_roi_align(x.permute(0, 1, 3, 2), rois, pool_h, pool_w, device,
                                      spatial_scale=1, sampling_ratio=-1,
                                      dtype=self.dtype)
        assert torch.allclose(gt_y.cuda(), y), 'PSRoIAlign layer incorrect'

    # def test_ps_roi_align_basic_cpu(self):
    #     device = torch.device('cpu')
    #     pool_size = 3
    #     x = torch.rand(1, 2 * (pool_size ** 2), 7, 7, dtype=self.dtype, device=device)
    #     rois = torch.tensor([[0, 0, 0, 5, 5]],  # format is (xyxy)
    #                         dtype=self.dtype, device=device)
    #
    #     pool_h, pool_w = (pool_size, pool_size)
    #     ps_roi_align = ops.PSRoIAlign((pool_h, pool_w), spatial_scale=1, sampling_ratio=2)
    #     y = ps_roi_align(x, rois)
    #
    #     gt_y = self.slow_ps_roi_align(x, rois, pool_h, pool_w, device,
    #                                   spatial_scale=1, sampling_ratio=2,
    #                                   dtype=self.dtype)
    #     assert torch.allclose(gt_y, y), 'PSRoIAlign layer incorrect on CPU'
    #
    #     y = ps_roi_align(x.permute(0, 1, 3, 2), rois)
    #     gt_y = self.slow_ps_roi_align(x.permute(0, 1, 3, 2), rois, pool_h, pool_w, device,
    #                                   spatial_scale=1, sampling_ratio=-1,
    #                                   dtype=self.dtype)
    #     assert torch.allclose(gt_y, y), 'PSRoIAlign layer incorrect on CPU'

    # @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    # def test_ps_roi_align_cuda(self):
    #     print("Testing test_ps_roi_align_cuda")
    #     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #     pool_size = 5
    #     x = torch.rand(2, 2 * (pool_size ** 2), 10, 10, dtype=self.dtype, device=device)
    #     rois = torch.tensor([[0, 0, 0, 9, 9],  # format is (xyxy)
    #                          [0, 0, 5, 4, 9],
    #                          [0, 5, 5, 9, 9],
    #                          [1, 0, 0, 9, 9]],
    #                         dtype=self.dtype, device=device)
    #
    #     pool_h, pool_w = (pool_size, pool_size)
    #     ps_roi_align = ops.PSRoIAlign((pool_h, pool_w), spatial_scale=1, sampling_ratio=2)
    #     y = ps_roi_align(x, rois)
    #
    #     gt_y = self.slow_ps_roi_align(x, rois, pool_h, pool_w, device,
    #                                   spatial_scale=1, sampling_ratio=2,
    #                                   dtype=self.dtype)
    #     assert torch.allclose(gt_y.cuda(), y), 'PSRoIAlign layer incorrect'
    #
    #     y = ps_roi_align(x.permute(0, 1, 3, 2), rois)
    #     gt_y = self.slow_ps_roi_align(x.permute(0, 1, 3, 2), rois, pool_h, pool_w,
    #                                   device, spatial_scale=1, sampling_ratio=2,
    #                                   dtype=self.dtype)
    #     assert torch.allclose(gt_y.cuda(), y), 'PSRoIAlign layer incorrect'

    # def test_ps_roi_align_cpu(self):
    #     device = torch.device('cpu')
    #     pool_size = 5
    #     x = torch.rand(2, 2 * (pool_size ** 2), 10, 10, dtype=self.dtype, device=device)
    #     rois = torch.tensor([[0, 0, 0, 9, 9],  # format is (xyxy)
    #                          [0, 0, 5, 4, 9],
    #                          [0, 5, 5, 9, 9],
    #                          [1, 0, 0, 9, 9]],
    #                         dtype=self.dtype, device=device)
    #
    #     pool_h, pool_w = (pool_size, pool_size)
    #     ps_roi_align = ops.PSRoIAlign((pool_h, pool_w), spatial_scale=1, sampling_ratio=2)
    #     y = ps_roi_align(x, rois)
    #
    #     gt_y = self.slow_ps_roi_align(x, rois, pool_h, pool_w, device,
    #                                   spatial_scale=1, sampling_ratio=2,
    #                                   dtype=self.dtype)
    #     assert torch.allclose(gt_y, y), 'PSRoIAlign layer incorrect on CPU'
    #
    #     y = ps_roi_align(x.permute(0, 1, 3, 2), rois)
    #     gt_y = self.slow_ps_roi_align(x.permute(0, 1, 3, 2), rois, pool_h, pool_w,
    #                                   device, spatial_scale=1, sampling_ratio=2,
    #                                   dtype=self.dtype)
    #     assert torch.allclose(gt_y, y), 'PSRoIAlign layer incorrect on CPU'

    # @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    # def test_ps_roi_align_gradient_cuda(self):
    #     print("Testing test_ps_roi_align_gradient_cuda")
    #     device = torch.device('cuda')
    #     pool_size = 3
    #     layer = ops.PSRoIAlign((pool_size, pool_size), spatial_scale=1,
    #                            sampling_ratio=-1).to(dtype=self.dtype, device=device)
    #     x = torch.ones(1, pool_size ** 2, 5, 5, dtype=self.dtype, device=device, requires_grad=True)
    #     rois = torch.tensor([
    #         [0, 0, 0, 4, 4],
    #         [0, 0, 3, 5, 5],
    #         [0, 1, 0, 2, 4]],
    #         dtype=self.dtype, device=device)
    #
    #     y = layer(x, rois)
    #     s = y.sum()
    #     s.backward()
    #     gt_grad = torch.tensor([[[[8.125e-01, 6.875e-01, 0.0, 0.0, 0.0, ],
    #                               [2.7083333333e-01, 2.2916666667e-01, 0.0, 0.0, 0.0, ],
    #                               [1.0416666667e-01, 6.25e-02, 0.0, 0.0, 0.0, ],
    #                               [5.2083333333e-01, 3.125e-01, 0.0, 0.0, 0.0, ],
    #                               [0.0, 0.0, 0.0, 0.0, 0.0, ]],
    #                              [[8.3266726847e-17, 1.125e00, 3.750e-01, 0.0, 0.0, ],
    #                               [2.7755575616e-17, 3.750e-01, 1.250e-01, 0.0, 0.0, ],
    #                               [0.0, 3.4722222222e-02, 9.7222222222e-02, 3.4722222222e-02, 0.0, ],
    #                               [0.0, 1.7361111111e-01, 4.8611111111e-01, 1.7361111111e-01, 0.0, ],
    #                               [0.0, 0.0, 0.0, 0.0, 0.0, ]],
    #                              [[0.0, 5.000e-01, 4.375e-01, 5.000e-01, 6.25e-02, ],
    #                               [0.0, 1.6666666667e-01, 1.4583333333e-01, 1.6666666667e-01, 2.0833333333e-02, ],
    #                               [0.0, 0.0, 0.0, 6.25e-02, 1.0416666667e-01, ],
    #                               [0.0, 0.0, 0.0, 3.125e-01, 5.2083333333e-01, ],
    #                               [0.0, 0.0, 0.0, 0.0, 0.0, ]],
    #                              [[0.0, 0.0, 0.0, 0.0, 0.0, ],
    #                               [5.4166666667e-01, 4.5833333333e-01, 0.0, 0.0, 0.0, ],
    #                               [5.4166666667e-01, 4.5833333333e-01, 0.0, 0.0, 0.0, ],
    #                               [3.125e-01, 1.875e-01, 0.0, 0.0, 0.0, ],
    #                               [3.125e-01, 1.875e-01, 0.0, 0.0, 0.0, ]],
    #                              [[0.0, 0.0, 0.0, 0.0, 0.0, ],
    #                               [5.5511151231e-17, 7.500e-01, 2.500e-01, 0.0, 0.0, ],
    #                               [5.5511151231e-17, 7.500e-01, 2.500e-01, 0.0, 0.0, ],
    #                               [0.0, 1.0416666667e-01, 2.9166666667e-01, 1.0416666667e-01, 0.0, ],
    #                               [0.0, 1.0416666667e-01, 2.9166666667e-01, 1.0416666667e-01, 0.0, ]],
    #                              [[0.0, 0.0, 0.0, 0.0, 0.0, ],
    #                               [0.0, 3.3333333333e-01, 2.9166666667e-01, 3.3333333333e-01, 4.1666666667e-02, ],
    #                               [0.0, 3.3333333333e-01, 2.9166666667e-01, 3.3333333333e-01, 4.1666666667e-02, ],
    #                               [0.0, 0.0, 0.0, 1.875e-01, 3.125e-01, ],
    #                               [0.0, 0.0, 0.0, 1.875e-01, 3.125e-01, ]],
    #                              [[0.0, 0.0, 0.0, 0.0, 0.0, ],
    #                               [0.0, 0.0, 0.0, 0.0, 0.0, ],
    #                               [2.7083333333e-01, 2.2916666667e-01, 0.0, 0.0, 0.0, ],
    #                               [7.2222222222e-01, 6.1111111111e-01, 0.0, 0.0, 0.0, ],
    #                               [7.1527777778e-01, 4.5138888889e-01, 0.0, 0.0, 0.0, ]],
    #                              [[0.0, 0.0, 0.0, 0.0, 0.0, ],
    #                               [0.0, 0.0, 0.0, 0.0, 0.0, ],
    #                               [2.7755575616e-17, 3.750e-01, 1.250e-01, 0.0, 0.0, ],
    #                               [7.4014868308e-17, 1.000e00, 3.3333333333e-01, 0.0, 0.0, ],
    #                               [9.2518585385e-18, 3.3333333333e-01, 6.25e-01, 2.0833333333e-01, 0.0, ]],
    #                              [[0.0, 0.0, 0.0, 0.0, 0.0, ],
    #                               [0.0, 0.0, 0.0, 0.0, 0.0, ],
    #                               [0.0, 1.6666666667e-01, 1.4583333333e-01, 1.6666666667e-01, 2.0833333333e-02, ],
    #                               [0.0, 4.4444444444e-01, 3.8888888889e-01, 4.4444444444e-01, 5.5555555556e-02, ],
    #                               [0.0, 5.5555555556e-02, 4.8611111111e-02, 4.3055555556e-01, 6.3194444444e-01, ]]]],
    #                            device=device, dtype=self.dtype)
    #     assert torch.allclose(x.grad, gt_grad), 'gradient incorrect for PSRoIAlign'

    # def test_ps_roi_align_gradient_cpu(self):
    #     device = torch.device('cpu')
    #     pool_size = 3
    #     layer = ops.PSRoIAlign((pool_size, pool_size), spatial_scale=1,
    #                            sampling_ratio=-1).to(dtype=self.dtype, device=device)
    #     x = torch.ones(1, pool_size ** 2, 5, 5, dtype=self.dtype, device=device, requires_grad=True)
    #     rois = torch.tensor([
    #         [0, 0, 0, 4, 4],
    #         [0, 0, 3, 5, 5],
    #         [0, 1, 0, 2, 4]],
    #         dtype=self.dtype, device=device)
    #
    #     y = layer(x, rois)
    #     s = y.sum()
    #     s.backward()
    #     gt_grad = torch.tensor([[[[8.125e-01, 6.875e-01, 0.0, 0.0, 0.0, ],
    #                               [2.7083333333e-01, 2.2916666667e-01, 0.0, 0.0, 0.0, ],
    #                               [1.0416666667e-01, 6.25e-02, 0.0, 0.0, 0.0, ],
    #                               [5.2083333333e-01, 3.125e-01, 0.0, 0.0, 0.0, ],
    #                               [0.0, 0.0, 0.0, 0.0, 0.0, ]],
    #                              [[8.3266726847e-17, 1.125e00, 3.750e-01, 0.0, 0.0, ],
    #                               [2.7755575616e-17, 3.750e-01, 1.250e-01, 0.0, 0.0, ],
    #                               [0.0, 3.4722222222e-02, 9.7222222222e-02, 3.4722222222e-02, 0.0, ],
    #                               [0.0, 1.7361111111e-01, 4.8611111111e-01, 1.7361111111e-01, 0.0, ],
    #                               [0.0, 0.0, 0.0, 0.0, 0.0, ]],
    #                              [[0.0, 5.000e-01, 4.375e-01, 5.000e-01, 6.25e-02, ],
    #                               [0.0, 1.6666666667e-01, 1.4583333333e-01, 1.6666666667e-01, 2.0833333333e-02, ],
    #                               [0.0, 0.0, 0.0, 6.25e-02, 1.0416666667e-01, ],
    #                               [0.0, 0.0, 0.0, 3.125e-01, 5.2083333333e-01, ],
    #                               [0.0, 0.0, 0.0, 0.0, 0.0, ]],
    #                              [[0.0, 0.0, 0.0, 0.0, 0.0, ],
    #                               [5.4166666667e-01, 4.5833333333e-01, 0.0, 0.0, 0.0, ],
    #                               [5.4166666667e-01, 4.5833333333e-01, 0.0, 0.0, 0.0, ],
    #                               [3.125e-01, 1.875e-01, 0.0, 0.0, 0.0, ],
    #                               [3.125e-01, 1.875e-01, 0.0, 0.0, 0.0, ]],
    #                              [[0.0, 0.0, 0.0, 0.0, 0.0, ],
    #                               [5.5511151231e-17, 7.500e-01, 2.500e-01, 0.0, 0.0, ],
    #                               [5.5511151231e-17, 7.500e-01, 2.500e-01, 0.0, 0.0, ],
    #                               [0.0, 1.0416666667e-01, 2.9166666667e-01, 1.0416666667e-01, 0.0, ],
    #                               [0.0, 1.0416666667e-01, 2.9166666667e-01, 1.0416666667e-01, 0.0, ]],
    #                              [[0.0, 0.0, 0.0, 0.0, 0.0, ],
    #                               [0.0, 3.3333333333e-01, 2.9166666667e-01, 3.3333333333e-01, 4.1666666667e-02, ],
    #                               [0.0, 3.3333333333e-01, 2.9166666667e-01, 3.3333333333e-01, 4.1666666667e-02, ],
    #                               [0.0, 0.0, 0.0, 1.875e-01, 3.125e-01, ],
    #                               [0.0, 0.0, 0.0, 1.875e-01, 3.125e-01, ]],
    #                              [[0.0, 0.0, 0.0, 0.0, 0.0, ],
    #                               [0.0, 0.0, 0.0, 0.0, 0.0, ],
    #                               [2.7083333333e-01, 2.2916666667e-01, 0.0, 0.0, 0.0, ],
    #                               [7.2222222222e-01, 6.1111111111e-01, 0.0, 0.0, 0.0, ],
    #                               [7.1527777778e-01, 4.5138888889e-01, 0.0, 0.0, 0.0, ]],
    #                              [[0.0, 0.0, 0.0, 0.0, 0.0, ],
    #                               [0.0, 0.0, 0.0, 0.0, 0.0, ],
    #                               [2.7755575616e-17, 3.750e-01, 1.250e-01, 0.0, 0.0, ],
    #                               [7.4014868308e-17, 1.000e00, 3.3333333333e-01, 0.0, 0.0, ],
    #                               [9.2518585385e-18, 3.3333333333e-01, 6.25e-01, 2.0833333333e-01, 0.0, ]],
    #                              [[0.0, 0.0, 0.0, 0.0, 0.0, ],
    #                               [0.0, 0.0, 0.0, 0.0, 0.0, ],
    #                               [0.0, 1.6666666667e-01, 1.4583333333e-01, 1.6666666667e-01, 2.0833333333e-02, ],
    #                               [0.0, 4.4444444444e-01, 3.8888888889e-01, 4.4444444444e-01, 5.5555555556e-02, ],
    #                               [0.0, 5.5555555556e-02, 4.8611111111e-02, 4.3055555556e-01, 6.3194444444e-01, ]]]],
    #                            device=device, dtype=self.dtype)
    #     assert torch.allclose(x.grad, gt_grad), 'gradient incorrect for PSRoIAlign on CPU'

    # @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    # def test_ps_roi_align_gradcheck_cuda(self):
    #     print("Testing test_ps_roi_align_gradcheck_cuda")
    #     device = torch.device('cuda')
    #     pool_size = 5
    #     x = torch.rand(1, pool_size ** 2, 10, 10, dtype=self.dtype, device=device, requires_grad=True)
    #     rois = torch.tensor([
    #         [0, 0, 0, 9, 9],
    #         [0, 0, 5, 5, 9],
    #         [0, 5, 5, 9, 9]], dtype=self.dtype, device=device)
    #
    #     m = ops.PSRoIAlign((pool_size, pool_size), spatial_scale=1,
    #                        sampling_ratio=2).to(dtype=self.dtype, device=device)
    #
    #     def func(input):
    #         return m(input, rois)
    #
    #     assert gradcheck(func, (x,)), 'gradcheck failed for PSRoIAlign CUDA'
    #     assert gradcheck(func, (x.permute(0, 1, 3, 2),)), 'gradcheck failed for PSRoIAlign CUDA'

    # def test_ps_roi_align_gradcheck_cpu(self):
    #     device = torch.device('cpu')
    #     pool_size = 5
    #     x = torch.rand(1, pool_size ** 2, 10, 10, dtype=self.dtype, device=device, requires_grad=True)
    #     rois = torch.tensor([
    #         [0, 0, 0, 9, 9],
    #         [0, 0, 5, 5, 9],
    #         [0, 5, 5, 9, 9]], dtype=self.dtype, device=device)
    #
    #     m = ops.PSRoIAlign((pool_size, pool_size), spatial_scale=1,
    #                        sampling_ratio=2).to(dtype=self.dtype, device=device)
    #
    #     def func(input):
    #         return m(input, rois)
    #
    #     assert gradcheck(func, (x,)), 'gradcheck failed for PSRoIAlign on CPU'
    #     assert gradcheck(func, (x.permute(0, 1, 3, 2),)), 'gradcheck failed for PSRoIAlign on CPU'


class PSRoIPoolTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dtype = torch.float64

    def slow_ps_roi_pooling(self, x, rois, pool_h, pool_w, device, spatial_scale=1,
                            dtype=torch.float64):
        if device is None:
            device = torch.device("cpu")
        num_input_channels = x.size(1)
        assert num_input_channels % (pool_h * pool_w) == 0, "input channels must be divisible by ph * pw"
        num_output_channels = int(num_input_channels / (pool_h * pool_w))
        y = torch.zeros(rois.size(0), num_output_channels, pool_h, pool_w, dtype=dtype, device=device)

        rois = torch.round(rois * spatial_scale).int()
        for n in range(0, x.size(0)):
            for r, roi in enumerate(rois):
                if roi[0] != n:
                    continue
                c_in = 0
                for c_out in range(0, num_output_channels):
                    roi_height = max(roi[4].item() - roi[2].item(), 1)
                    roi_width = max(roi[3].item() - roi[1].item(), 1)
                    bin_h, bin_w = roi_height / float(pool_h), roi_width / float(pool_w)

                    for j in range(0, pool_h):
                        start_h = int(np.floor(j * bin_h)) + roi[2].item()
                        end_h = int(np.ceil((j + 1) * bin_w)) + roi[2].item()

                        # range-check
                        start_h = min(max(start_h, 0), x.size(2))
                        end_h = min(max(end_h, 0), x.size(2))

                        for i in range(0, pool_w):
                            start_w = int(np.floor(i * bin_w)) + roi[1].item()
                            end_w = int(np.ceil((i + 1) * bin_w)) + roi[1].item()

                            # range-check
                            start_w = min(max(start_w, 0), x.size(3))
                            end_w = min(max(end_w, 0), x.size(3))

                            is_empty = (end_h <= start_h) or (end_w <= start_w)
                            area = (end_h - start_h) * (end_w - start_w)

                            if not is_empty:
                                t = torch.sum(x[n, c_in, slice(start_h, end_h), slice(start_w, end_w)])
                                y[r, c_out, j, i] = t / area
                            c_in += 1
        return y

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_ps_roi_pool_basic_cuda(self):
        device = torch.device('cuda')
        pool_size = 3
        x = torch.rand(1, pool_size ** 2, 10, 10, dtype=self.dtype, device=device)
        rois = torch.tensor([[0, 0, 0, 4, 4]],  # format is (xyxy)
                            dtype=self.dtype, device=device)

        pool_h, pool_w = (pool_size, pool_size)
        ps_roi_pool = ops.PSRoIPool((pool_h, pool_w), 1)
        y = ps_roi_pool(x, rois)

        gt_y = self.slow_ps_roi_pooling(x, rois, pool_h, pool_w, device, dtype=self.dtype)
        assert torch.allclose(gt_y.cuda(), y), 'PSRoIPool layer incorrect'

        y = ps_roi_pool(x.permute(0, 1, 3, 2), rois)
        gt_y = self.slow_ps_roi_pooling(x.permute(0, 1, 3, 2), rois, pool_h, pool_w, device, dtype=self.dtype)
        assert torch.allclose(gt_y.cuda(), y), 'PSRoIPool layer incorrect'

    def test_ps_roi_pool_basic_cpu(self):
        device = torch.device('cpu')
        pool_size = 3
        x = torch.rand(1, pool_size ** 2, 10, 10, dtype=self.dtype, device=device)
        rois = torch.tensor([[0, 0, 0, 4, 4]],  # format is (xyxy)
                            dtype=self.dtype, device=device)

        pool_h, pool_w = (pool_size, pool_size)
        ps_roi_pool = ops.PSRoIPool((pool_h, pool_w), 1)
        y = ps_roi_pool(x, rois)

        gt_y = self.slow_ps_roi_pooling(x, rois, pool_h, pool_w, device, dtype=self.dtype)
        assert torch.allclose(gt_y, y), 'PSRoIPool layer incorrect on CPU'

        y = ps_roi_pool(x.permute(0, 1, 3, 2), rois)
        gt_y = self.slow_ps_roi_pooling(x.permute(0, 1, 3, 2), rois, pool_h, pool_w, device, dtype=self.dtype)
        assert torch.allclose(gt_y, y), 'PSRoIPool layer incorrect on CPU'

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_ps_roi_pool_cuda(self):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        pool_size = 5
        x = torch.rand(2, 2 * (pool_size ** 2), 10, 10, dtype=self.dtype, device=device)
        rois = torch.tensor([[0, 0, 0, 9, 9],  # format is (xyxy)
                             [0, 0, 5, 4, 9],
                             [0, 5, 5, 9, 9],
                             [1, 0, 0, 9, 9]],
                            dtype=self.dtype, device=device)

        pool_h, pool_w = (pool_size, pool_size)
        ps_roi_pool = ops.PSRoIPool((pool_h, pool_w), 1)
        y = ps_roi_pool(x, rois)

        gt_y = self.slow_ps_roi_pooling(x, rois, pool_h, pool_w, device, dtype=self.dtype)

        assert torch.allclose(gt_y.cuda(), y), 'PSRoIPool layer incorrect'

        y = ps_roi_pool(x.permute(0, 1, 3, 2), rois)
        gt_y = self.slow_ps_roi_pooling(x.permute(0, 1, 3, 2), rois, pool_h, pool_w, device, dtype=self.dtype)
        assert torch.allclose(gt_y.cuda(), y), 'PSRoIPool layer incorrect'

    def test_ps_roi_pool_cpu(self):
        device = torch.device('cpu')
        pool_size = 5
        x = torch.rand(2, 2 * (pool_size ** 2), 10, 10, dtype=self.dtype, device=device)
        rois = torch.tensor([[0, 0, 0, 9, 9],  # format is (xyxy)
                             [0, 0, 5, 4, 9],
                             [0, 5, 5, 9, 9],
                             [1, 0, 0, 9, 9]],
                            dtype=self.dtype, device=device)

        pool_h, pool_w = (pool_size, pool_size)
        ps_roi_pool = ops.PSRoIPool((pool_h, pool_w), 1)
        y = ps_roi_pool(x, rois)

        gt_y = self.slow_ps_roi_pooling(x, rois, pool_h, pool_w, device, dtype=self.dtype)
        assert torch.allclose(gt_y, y), 'PSRoIPool layer incorrect on CPU'

        y = ps_roi_pool(x.permute(0, 1, 3, 2), rois)
        gt_y = self.slow_ps_roi_pooling(x.permute(0, 1, 3, 2), rois, pool_h, pool_w, device, dtype=self.dtype)
        assert torch.allclose(gt_y, y), 'PSRoIPool layer incorrect on CPU'

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_ps_roi_pool_gradient_cuda(self):
        device = torch.device('cuda')
        pool_size = 3
        layer = ops.PSRoIPool((pool_size, pool_size), 1).to(dtype=self.dtype, device=device)
        x = torch.ones(1, pool_size ** 2, 5, 5, dtype=self.dtype, device=device, requires_grad=True)
        rois = torch.tensor([
            [0, 0, 0, 4, 4],
            [0, 0, 3, 5, 5],
            [0, 1, 0, 2, 4]],
            dtype=self.dtype, device=device)

        y = layer(x, rois)
        s = y.sum()
        s.backward()
        gt_grad = torch.tensor([[[[0.2500, 0.7500, 0.0000, 0.0000, 0.0000],
                                  [0.2500, 0.7500, 0.0000, 0.0000, 0.0000],
                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                  [0.5000, 0.5000, 0.0000, 0.0000, 0.0000],
                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

                                 [[0.0000, 0.7500, 0.2500, 0.0000, 0.0000],
                                  [0.0000, 0.7500, 0.2500, 0.0000, 0.0000],
                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                  [0.0000, 1. / 3, 1. / 3, 1. / 3, 0.0000],
                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

                                 [[0.0000, 0.5000, 0.2500, 0.2500, 0.0000],
                                  [0.0000, 0.5000, 0.2500, 0.2500, 0.0000],
                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                  [0.0000, 0.0000, 0.0000, 0.5000, 0.5000],
                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

                                 [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                  [0.2500, 0.7500, 0.0000, 0.0000, 0.0000],
                                  [0.2500, 0.7500, 0.0000, 0.0000, 0.0000],
                                  [0.2500, 0.2500, 0.0000, 0.0000, 0.0000],
                                  [0.2500, 0.2500, 0.0000, 0.0000, 0.0000]],

                                 [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                  [0.0000, 0.7500, 0.2500, 0.0000, 0.0000],
                                  [0.0000, 0.7500, 0.2500, 0.0000, 0.0000],
                                  [0.0000, 1. / 6, 1. / 6, 1. / 6, 0.0000],
                                  [0.0000, 1. / 6, 1. / 6, 1. / 6, 0.0000]],

                                 [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                  [0.0000, 0.5000, 0.2500, 0.2500, 0.0000],
                                  [0.0000, 0.5000, 0.2500, 0.2500, 0.0000],
                                  [0.0000, 0.0000, 0.0000, 0.2500, 0.2500],
                                  [0.0000, 0.0000, 0.0000, 0.2500, 0.2500]],

                                 [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                  [0.2500, 0.7500, 0.0000, 0.0000, 0.0000],
                                  [0.2500, 0.7500, 0.0000, 0.0000, 0.0000],
                                  [0.5000, 0.5000, 0.0000, 0.0000, 0.0000]],

                                 [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                  [0.0000, 0.7500, 0.2500, 0.0000, 0.0000],
                                  [0.0000, 0.7500, 0.2500, 0.0000, 0.0000],
                                  [0.0000, 1. / 3, 1. / 3, 1. / 3, 0.0000]],

                                 [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                  [0.0000, 0.5000, 0.2500, 0.2500, 0.0000],
                                  [0.0000, 0.5000, 0.2500, 0.2500, 0.0000],
                                  [0.0000, 0.0000, 0.0000, 0.5000, 0.5000]]]],
                               device=device, dtype=self.dtype)
        assert torch.allclose(x.grad, gt_grad), 'gradient incorrect for PSRoIPool'

    def test_ps_roi_pool_gradient_cpu(self):
        device = torch.device('cpu')
        pool_size = 3
        layer = ops.PSRoIPool((pool_size, pool_size), 1).to(dtype=self.dtype, device=device)
        x = torch.ones(1, pool_size ** 2, 5, 5, dtype=self.dtype, device=device, requires_grad=True)
        rois = torch.tensor([
            [0, 0, 0, 4, 4],
            [0, 0, 3, 5, 5],
            [0, 1, 0, 2, 4]],
            dtype=self.dtype, device=device)

        y = layer(x, rois)
        s = y.sum()
        s.backward()
        gt_grad = torch.tensor([[[[0.2500, 0.7500, 0.0000, 0.0000, 0.0000],
                                  [0.2500, 0.7500, 0.0000, 0.0000, 0.0000],
                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                  [0.5000, 0.5000, 0.0000, 0.0000, 0.0000],
                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

                                 [[0.0000, 0.7500, 0.2500, 0.0000, 0.0000],
                                  [0.0000, 0.7500, 0.2500, 0.0000, 0.0000],
                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                  [0.0000, 1. / 3, 1. / 3, 1. / 3, 0.0000],
                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

                                 [[0.0000, 0.5000, 0.2500, 0.2500, 0.0000],
                                  [0.0000, 0.5000, 0.2500, 0.2500, 0.0000],
                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                  [0.0000, 0.0000, 0.0000, 0.5000, 0.5000],
                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

                                 [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                  [0.2500, 0.7500, 0.0000, 0.0000, 0.0000],
                                  [0.2500, 0.7500, 0.0000, 0.0000, 0.0000],
                                  [0.2500, 0.2500, 0.0000, 0.0000, 0.0000],
                                  [0.2500, 0.2500, 0.0000, 0.0000, 0.0000]],

                                 [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                  [0.0000, 0.7500, 0.2500, 0.0000, 0.0000],
                                  [0.0000, 0.7500, 0.2500, 0.0000, 0.0000],
                                  [0.0000, 1. / 6, 1. / 6, 1. / 6, 0.0000],
                                  [0.0000, 1. / 6, 1. / 6, 1. / 6, 0.0000]],

                                 [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                  [0.0000, 0.5000, 0.2500, 0.2500, 0.0000],
                                  [0.0000, 0.5000, 0.2500, 0.2500, 0.0000],
                                  [0.0000, 0.0000, 0.0000, 0.2500, 0.2500],
                                  [0.0000, 0.0000, 0.0000, 0.2500, 0.2500]],

                                 [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                  [0.2500, 0.7500, 0.0000, 0.0000, 0.0000],
                                  [0.2500, 0.7500, 0.0000, 0.0000, 0.0000],
                                  [0.5000, 0.5000, 0.0000, 0.0000, 0.0000]],

                                 [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                  [0.0000, 0.7500, 0.2500, 0.0000, 0.0000],
                                  [0.0000, 0.7500, 0.2500, 0.0000, 0.0000],
                                  [0.0000, 1. / 3, 1. / 3, 1. / 3, 0.0000]],

                                 [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                  [0.0000, 0.5000, 0.2500, 0.2500, 0.0000],
                                  [0.0000, 0.5000, 0.2500, 0.2500, 0.0000],
                                  [0.0000, 0.0000, 0.0000, 0.5000, 0.5000]]]],
                               device=device, dtype=self.dtype)
        assert torch.allclose(x.grad, gt_grad), 'gradient incorrect for PSRoIPool on CPU'

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_ps_roi_pool_gradcheck_cuda(self):
        device = torch.device('cuda')
        pool_size = 5
        x = torch.rand(1, pool_size ** 2, 10, 10, dtype=self.dtype, device=device, requires_grad=True)
        rois = torch.tensor([
            [0, 0, 0, 9, 9],
            [0, 0, 5, 5, 9],
            [0, 5, 5, 9, 9]], dtype=self.dtype, device=device)

        m = ops.PSRoIPool((pool_size, pool_size), 1).to(dtype=self.dtype, device=device)

        def func(input):
            return m(input, rois)

        assert gradcheck(func, (x,)), 'gradcheck failed for PSRoIPool CUDA'
        assert gradcheck(func, (x.permute(0, 1, 3, 2),)), 'gradcheck failed for PSRoIPool CUDA'

    def test_ps_roi_pool_gradcheck_cpu(self):
        device = torch.device('cpu')
        pool_size = 5
        x = torch.rand(1, pool_size ** 2, 10, 10, dtype=self.dtype, device=device, requires_grad=True)
        rois = torch.tensor([
            [0, 0, 0, 9, 9],
            [0, 0, 5, 5, 9],
            [0, 5, 5, 9, 9]], dtype=self.dtype, device=device)

        m = ops.PSRoIPool((pool_size, pool_size), 1).to(dtype=self.dtype, device=device)

        def func(input):
            return m(input, rois)

        assert gradcheck(func, (x,)), 'gradcheck failed for PSRoIPool on CPU'
        assert gradcheck(func, (x.permute(0, 1, 3, 2),)), 'gradcheck failed for PSRoIPool on CPU'


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
