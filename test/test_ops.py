import math
import unittest

import numpy as np

import torch
from torch import Tensor
from torch.autograd import gradcheck
from torch.jit.annotations import Tuple
from torch.nn.modules.utils import _pair
from torchvision import ops


class OpTester(object):
    @classmethod
    def setUpClass(cls):
        cls.dtype = torch.float64

    def test_forward_cpu_contiguous(self):
        self._test_forward(device=torch.device('cpu'), contiguous=True)

    def test_forward_cpu_non_contiguous(self):
        self._test_forward(device=torch.device('cpu'), contiguous=False)

    def test_backward_cpu_contiguous(self):
        self._test_backward(device=torch.device('cpu'), contiguous=True)

    def test_backward_cpu_non_contiguous(self):
        self._test_backward(device=torch.device('cpu'), contiguous=False)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_forward_cuda_contiguous(self):
        self._test_forward(device=torch.device('cuda'), contiguous=True)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_forward_cuda_non_contiguous(self):
        self._test_forward(device=torch.device('cuda'), contiguous=False)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_backward_cuda_contiguous(self):
        self._test_backward(device=torch.device('cuda'), contiguous=True)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_backward_cuda_non_contiguous(self):
        self._test_backward(device=torch.device('cuda'), contiguous=False)

    def _test_forward(self, device, contiguous):
        pass

    def _test_backward(self, device, contiguous):
        pass


class RoIOpTester(OpTester):
    def _test_forward(self, device, contiguous, x_dtype=None, rois_dtype=None):
        x_dtype = self.dtype if x_dtype is None else x_dtype
        rois_dtype = self.dtype if rois_dtype is None else rois_dtype
        pool_size = 5
        # n_channels % (pool_size ** 2) == 0 required for PS opeartions.
        n_channels = 2 * (pool_size ** 2)
        x = torch.rand(2, n_channels, 10, 10, dtype=x_dtype, device=device)
        if not contiguous:
            x = x.permute(0, 1, 3, 2)
        rois = torch.tensor([[0, 0, 0, 9, 9],  # format is (xyxy)
                             [0, 0, 5, 4, 9],
                             [0, 5, 5, 9, 9],
                             [1, 0, 0, 9, 9]],
                            dtype=rois_dtype, device=device)

        pool_h, pool_w = pool_size, pool_size
        y = self.fn(x, rois, pool_h, pool_w, spatial_scale=1, sampling_ratio=-1)
        # the following should be true whether we're running an autocast test or not.
        self.assertTrue(y.dtype == x.dtype)
        gt_y = self.expected_fn(x, rois, pool_h, pool_w, spatial_scale=1,
                                sampling_ratio=-1, device=device, dtype=self.dtype)

        tol = 1e-3 if (x_dtype is torch.half or rois_dtype is torch.half) else 1e-5
        self.assertTrue(torch.allclose(gt_y.to(y.dtype), y, rtol=tol, atol=tol))

    def _test_backward(self, device, contiguous):
        pool_size = 2
        x = torch.rand(1, 2 * (pool_size ** 2), 5, 5, dtype=self.dtype, device=device, requires_grad=True)
        if not contiguous:
            x = x.permute(0, 1, 3, 2)
        rois = torch.tensor([[0, 0, 0, 4, 4],  # format is (xyxy)
                             [0, 0, 2, 3, 4],
                             [0, 2, 2, 4, 4]],
                            dtype=self.dtype, device=device)

        def func(z):
            return self.fn(z, rois, pool_size, pool_size, spatial_scale=1, sampling_ratio=1)

        script_func = self.get_script_fn(rois, pool_size)

        self.assertTrue(gradcheck(func, (x,)))
        self.assertTrue(gradcheck(script_func, (x,)))

    def test_boxes_shape(self):
        self._test_boxes_shape()

    def _helper_boxes_shape(self, func):
        # test boxes as Tensor[N, 5]
        with self.assertRaises(AssertionError):
            a = torch.linspace(1, 8 * 8, 8 * 8).reshape(1, 1, 8, 8)
            boxes = torch.tensor([[0, 0, 3, 3]], dtype=a.dtype)
            func(a, boxes, output_size=(2, 2))

        # test boxes as List[Tensor[N, 4]]
        with self.assertRaises(AssertionError):
            a = torch.linspace(1, 8 * 8, 8 * 8).reshape(1, 1, 8, 8)
            boxes = torch.tensor([[0, 0, 3]], dtype=a.dtype)
            ops.roi_pool(a, [boxes], output_size=(2, 2))

    def fn(*args, **kwargs):
        pass

    def get_script_fn(*args, **kwargs):
        pass

    def expected_fn(*args, **kwargs):
        pass


class RoIPoolTester(RoIOpTester, unittest.TestCase):
    def fn(self, x, rois, pool_h, pool_w, spatial_scale=1, sampling_ratio=-1, **kwargs):
        return ops.RoIPool((pool_h, pool_w), spatial_scale)(x, rois)

    def get_script_fn(self, rois, pool_size):
        @torch.jit.script
        def script_fn(input, rois, pool_size):
            # type: (Tensor, Tensor, int) -> Tensor
            return ops.roi_pool(input, rois, pool_size, 1.0)[0]
        return lambda x: script_fn(x, rois, pool_size)

    def expected_fn(self, x, rois, pool_h, pool_w, spatial_scale=1, sampling_ratio=-1,
                    device=None, dtype=torch.float64):
        if device is None:
            device = torch.device("cpu")

        n_channels = x.size(1)
        y = torch.zeros(rois.size(0), n_channels, pool_h, pool_w, dtype=dtype, device=device)

        def get_slice(k, block):
            return slice(int(np.floor(k * block)), int(np.ceil((k + 1) * block)))

        for roi_idx, roi in enumerate(rois):
            batch_idx = int(roi[0])
            j_begin, i_begin, j_end, i_end = (int(round(x.item() * spatial_scale)) for x in roi[1:])
            roi_x = x[batch_idx, :, i_begin:i_end + 1, j_begin:j_end + 1]

            roi_h, roi_w = roi_x.shape[-2:]
            bin_h = roi_h / pool_h
            bin_w = roi_w / pool_w

            for i in range(0, pool_h):
                for j in range(0, pool_w):
                    bin_x = roi_x[:, get_slice(i, bin_h), get_slice(j, bin_w)]
                    if bin_x.numel() > 0:
                        y[roi_idx, :, i, j] = bin_x.reshape(n_channels, -1).max(dim=1)[0]
        return y

    def _test_boxes_shape(self):
        self._helper_boxes_shape(ops.roi_pool)


class PSRoIPoolTester(RoIOpTester, unittest.TestCase):
    def fn(self, x, rois, pool_h, pool_w, spatial_scale=1, sampling_ratio=-1, **kwargs):
        return ops.PSRoIPool((pool_h, pool_w), 1)(x, rois)

    def get_script_fn(self, rois, pool_size):
        @torch.jit.script
        def script_fn(input, rois, pool_size):
            # type: (Tensor, Tensor, int) -> Tensor
            return ops.ps_roi_pool(input, rois, pool_size, 1.0)[0]
        return lambda x: script_fn(x, rois, pool_size)

    def expected_fn(self, x, rois, pool_h, pool_w, spatial_scale=1, sampling_ratio=-1,
                    device=None, dtype=torch.float64):
        if device is None:
            device = torch.device("cpu")
        n_input_channels = x.size(1)
        self.assertEqual(n_input_channels % (pool_h * pool_w), 0, "input channels must be divisible by ph * pw")
        n_output_channels = int(n_input_channels / (pool_h * pool_w))
        y = torch.zeros(rois.size(0), n_output_channels, pool_h, pool_w, dtype=dtype, device=device)

        def get_slice(k, block):
            return slice(int(np.floor(k * block)), int(np.ceil((k + 1) * block)))

        for roi_idx, roi in enumerate(rois):
            batch_idx = int(roi[0])
            j_begin, i_begin, j_end, i_end = (int(round(x.item() * spatial_scale)) for x in roi[1:])
            roi_x = x[batch_idx, :, i_begin:i_end + 1, j_begin:j_end + 1]

            roi_height = max(i_end - i_begin, 1)
            roi_width = max(j_end - j_begin, 1)
            bin_h, bin_w = roi_height / float(pool_h), roi_width / float(pool_w)

            for i in range(0, pool_h):
                for j in range(0, pool_w):
                    bin_x = roi_x[:, get_slice(i, bin_h), get_slice(j, bin_w)]
                    if bin_x.numel() > 0:
                        area = bin_x.size(-2) * bin_x.size(-1)
                        for c_out in range(0, n_output_channels):
                            c_in = c_out * (pool_h * pool_w) + pool_w * i + j
                            t = torch.sum(bin_x[c_in, :, :])
                            y[roi_idx, c_out, i, j] = t / area
        return y

    def _test_boxes_shape(self):
        self._helper_boxes_shape(ops.ps_roi_pool)


def bilinear_interpolate(data, y, x, snap_border=False):
    height, width = data.shape

    if snap_border:
        if -1 < y <= 0:
            y = 0
        elif height - 1 <= y < height:
            y = height - 1

        if -1 < x <= 0:
            x = 0
        elif width - 1 <= x < width:
            x = width - 1

    y_low = int(math.floor(y))
    x_low = int(math.floor(x))
    y_high = y_low + 1
    x_high = x_low + 1

    wy_h = y - y_low
    wx_h = x - x_low
    wy_l = 1 - wy_h
    wx_l = 1 - wx_h

    val = 0
    for wx, xp in zip((wx_l, wx_h), (x_low, x_high)):
        for wy, yp in zip((wy_l, wy_h), (y_low, y_high)):
            if 0 <= yp < height and 0 <= xp < width:
                val += wx * wy * data[yp, xp]
    return val


class RoIAlignTester(RoIOpTester, unittest.TestCase):
    def fn(self, x, rois, pool_h, pool_w, spatial_scale=1, sampling_ratio=-1, aligned=False, **kwargs):
        return ops.RoIAlign((pool_h, pool_w), spatial_scale=spatial_scale,
                            sampling_ratio=sampling_ratio, aligned=aligned)(x, rois)

    def get_script_fn(self, rois, pool_size):
        @torch.jit.script
        def script_fn(input, rois, pool_size):
            # type: (Tensor, Tensor, int) -> Tensor
            return ops.roi_align(input, rois, pool_size, 1.0)[0]
        return lambda x: script_fn(x, rois, pool_size)

    def expected_fn(self, in_data, rois, pool_h, pool_w, spatial_scale=1, sampling_ratio=-1, aligned=False,
                    device=None, dtype=torch.float64):
        if device is None:
            device = torch.device("cpu")
        n_channels = in_data.size(1)
        out_data = torch.zeros(rois.size(0), n_channels, pool_h, pool_w, dtype=dtype, device=device)

        offset = 0.5 if aligned else 0.

        for r, roi in enumerate(rois):
            batch_idx = int(roi[0])
            j_begin, i_begin, j_end, i_end = (x.item() * spatial_scale - offset for x in roi[1:])

            roi_h = i_end - i_begin
            roi_w = j_end - j_begin
            bin_h = roi_h / pool_h
            bin_w = roi_w / pool_w

            for i in range(0, pool_h):
                start_h = i_begin + i * bin_h
                grid_h = sampling_ratio if sampling_ratio > 0 else int(np.ceil(bin_h))
                for j in range(0, pool_w):
                    start_w = j_begin + j * bin_w
                    grid_w = sampling_ratio if sampling_ratio > 0 else int(np.ceil(bin_w))

                    for channel in range(0, n_channels):

                        val = 0
                        for iy in range(0, grid_h):
                            y = start_h + (iy + 0.5) * bin_h / grid_h
                            for ix in range(0, grid_w):
                                x = start_w + (ix + 0.5) * bin_w / grid_w
                                val += bilinear_interpolate(in_data[batch_idx, channel, :, :], y, x, snap_border=True)
                        val /= grid_h * grid_w

                        out_data[r, channel, i, j] = val
        return out_data

    def _test_boxes_shape(self):
        self._helper_boxes_shape(ops.roi_align)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_roi_align_autocast(self):
        for x_dtype in (torch.float, torch.half):
            for rois_dtype in (torch.float, torch.half):
                with torch.cuda.amp.autocast():
                    self._test_forward(torch.device("cuda"), contiguous=False, x_dtype=x_dtype, rois_dtype=rois_dtype)


class PSRoIAlignTester(RoIOpTester, unittest.TestCase):
    def fn(self, x, rois, pool_h, pool_w, spatial_scale=1, sampling_ratio=-1, **kwargs):
        return ops.PSRoIAlign((pool_h, pool_w), spatial_scale=spatial_scale,
                              sampling_ratio=sampling_ratio)(x, rois)

    def get_script_fn(self, rois, pool_size):
        @torch.jit.script
        def script_fn(input, rois, pool_size):
            # type: (Tensor, Tensor, int) -> Tensor
            return ops.ps_roi_align(input, rois, pool_size, 1.0)[0]
        return lambda x: script_fn(x, rois, pool_size)

    def expected_fn(self, in_data, rois, pool_h, pool_w, device, spatial_scale=1,
                    sampling_ratio=-1, dtype=torch.float64):
        if device is None:
            device = torch.device("cpu")
        n_input_channels = in_data.size(1)
        self.assertEqual(n_input_channels % (pool_h * pool_w), 0, "input channels must be divisible by ph * pw")
        n_output_channels = int(n_input_channels / (pool_h * pool_w))
        out_data = torch.zeros(rois.size(0), n_output_channels, pool_h, pool_w, dtype=dtype, device=device)

        for r, roi in enumerate(rois):
            batch_idx = int(roi[0])
            j_begin, i_begin, j_end, i_end = (x.item() * spatial_scale - 0.5 for x in roi[1:])

            roi_h = i_end - i_begin
            roi_w = j_end - j_begin
            bin_h = roi_h / pool_h
            bin_w = roi_w / pool_w

            for i in range(0, pool_h):
                start_h = i_begin + i * bin_h
                grid_h = sampling_ratio if sampling_ratio > 0 else int(np.ceil(bin_h))
                for j in range(0, pool_w):
                    start_w = j_begin + j * bin_w
                    grid_w = sampling_ratio if sampling_ratio > 0 else int(np.ceil(bin_w))
                    for c_out in range(0, n_output_channels):
                        c_in = c_out * (pool_h * pool_w) + pool_w * i + j

                        val = 0
                        for iy in range(0, grid_h):
                            y = start_h + (iy + 0.5) * bin_h / grid_h
                            for ix in range(0, grid_w):
                                x = start_w + (ix + 0.5) * bin_w / grid_w
                                val += bilinear_interpolate(in_data[batch_idx, c_in, :, :], y, x, snap_border=True)
                        val /= grid_h * grid_w

                        out_data[r, c_out, i, j] = val
        return out_data

    def _test_boxes_shape(self):
        self._helper_boxes_shape(ops.ps_roi_align)


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

    def _create_tensors_with_iou(self, N, iou_thresh):
        # force last box to have a pre-defined iou with the first box
        # let b0 be [x0, y0, x1, y1], and b1 be [x0, y0, x1 + d, y1],
        # then, in order to satisfy ops.iou(b0, b1) == iou_thresh,
        # we need to have d = (x1 - x0) * (1 - iou_thresh) / iou_thresh
        # Adjust the threshold upward a bit with the intent of creating
        # at least one box that exceeds (barely) the threshold and so
        # should be suppressed.
        boxes = torch.rand(N, 4) * 100
        boxes[:, 2:] += boxes[:, :2]
        boxes[-1, :] = boxes[0, :]
        x0, y0, x1, y1 = boxes[-1].tolist()
        iou_thresh += 1e-5
        boxes[-1, 2] += (x1 - x0) * (1 - iou_thresh) / iou_thresh
        scores = torch.rand(N)
        return boxes, scores

    def test_nms(self):
        err_msg = 'NMS incompatible between CPU and reference implementation for IoU={}'
        for iou in [0.2, 0.5, 0.8]:
            boxes, scores = self._create_tensors_with_iou(1000, iou)
            keep_ref = self.reference_nms(boxes, scores, iou)
            keep = ops.nms(boxes, scores, iou)
            self.assertTrue(torch.allclose(keep, keep_ref), err_msg.format(iou))
        self.assertRaises(RuntimeError, ops.nms, torch.rand(4), torch.rand(3), 0.5)
        self.assertRaises(RuntimeError, ops.nms, torch.rand(3, 5), torch.rand(3), 0.5)
        self.assertRaises(RuntimeError, ops.nms, torch.rand(3, 4), torch.rand(3, 2), 0.5)
        self.assertRaises(RuntimeError, ops.nms, torch.rand(3, 4), torch.rand(4), 0.5)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_nms_cuda(self):
        err_msg = 'NMS incompatible between CPU and CUDA for IoU={}'

        for iou in [0.2, 0.5, 0.8]:
            boxes, scores = self._create_tensors_with_iou(1000, iou)
            r_cpu = ops.nms(boxes, scores, iou)
            r_cuda = ops.nms(boxes.cuda(), scores.cuda(), iou)

            is_eq = torch.allclose(r_cpu, r_cuda.cpu())
            if not is_eq:
                # if the indices are not the same, ensure that it's because the scores
                # are duplicate
                is_eq = torch.allclose(scores[r_cpu], scores[r_cuda.cpu()])
            self.assertTrue(is_eq, err_msg.format(iou))


class NewEmptyTensorTester(unittest.TestCase):
    def test_new_empty_tensor(self):
        input = torch.tensor([2., 2.], requires_grad=True)
        new_shape = [3, 3]
        out = torch.ops.torchvision._new_empty_tensor_op(input, new_shape)
        assert out.size() == torch.Size([3, 3])
        assert out.dtype == input.dtype


class DeformConvTester(OpTester, unittest.TestCase):
    def expected_fn(self, x, weight, offset, bias, stride=1, padding=0, dilation=1):
        stride_h, stride_w = _pair(stride)
        pad_h, pad_w = _pair(padding)
        dil_h, dil_w = _pair(dilation)
        weight_h, weight_w = weight.shape[-2:]

        n_batches, n_in_channels, in_h, in_w = x.shape
        n_out_channels = weight.shape[0]

        out_h = (in_h + 2 * pad_h - (dil_h * (weight_h - 1) + 1)) // stride_h + 1
        out_w = (in_w + 2 * pad_w - (dil_w * (weight_w - 1) + 1)) // stride_w + 1

        n_offset_grps = offset.shape[1] // (2 * weight_h * weight_w)
        in_c_per_offset_grp = n_in_channels // n_offset_grps

        n_weight_grps = n_in_channels // weight.shape[1]
        in_c_per_weight_grp = weight.shape[1]
        out_c_per_weight_grp = n_out_channels // n_weight_grps

        out = torch.zeros(n_batches, n_out_channels, out_h, out_w, device=x.device, dtype=x.dtype)
        for b in range(n_batches):
            for c_out in range(n_out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        for di in range(weight_h):
                            for dj in range(weight_w):
                                for c in range(in_c_per_weight_grp):
                                    weight_grp = c_out // out_c_per_weight_grp
                                    c_in = weight_grp * in_c_per_weight_grp + c

                                    offset_grp = c_in // in_c_per_offset_grp
                                    offset_idx = 2 * (offset_grp * (weight_h * weight_w) + di * weight_w + dj)

                                    pi = stride_h * i - pad_h + dil_h * di + offset[b, offset_idx, i, j]
                                    pj = stride_w * j - pad_w + dil_w * dj + offset[b, offset_idx + 1, i, j]

                                    out[b, c_out, i, j] += (weight[c_out, c, di, dj] *
                                                            bilinear_interpolate(x[b, c_in, :, :], pi, pj))
        out += bias.view(1, n_out_channels, 1, 1)
        return out

    def get_fn_args(self, device, contiguous):
        batch_sz = 33
        n_in_channels = 6
        n_out_channels = 2
        n_weight_grps = 2
        n_offset_grps = 3

        stride = (2, 1)
        pad = (1, 0)
        dilation = (2, 1)

        stride_h, stride_w = stride
        pad_h, pad_w = pad
        dil_h, dil_w = dilation
        weight_h, weight_w = (3, 2)
        in_h, in_w = (5, 4)

        out_h = (in_h + 2 * pad_h - (dil_h * (weight_h - 1) + 1)) // stride_h + 1
        out_w = (in_w + 2 * pad_w - (dil_w * (weight_w - 1) + 1)) // stride_w + 1

        x = torch.rand(batch_sz, n_in_channels, in_h, in_w, device=device, dtype=self.dtype, requires_grad=True)

        offset = torch.randn(batch_sz, n_offset_grps * 2 * weight_h * weight_w, out_h, out_w,
                             device=device, dtype=self.dtype, requires_grad=True)

        weight = torch.randn(n_out_channels, n_in_channels // n_weight_grps, weight_h, weight_w,
                             device=device, dtype=self.dtype, requires_grad=True)

        bias = torch.randn(n_out_channels, device=device, dtype=self.dtype, requires_grad=True)

        if not contiguous:
            x = x.permute(0, 1, 3, 2).contiguous().permute(0, 1, 3, 2)
            offset = offset.permute(1, 3, 0, 2).contiguous().permute(2, 0, 3, 1)
            weight = weight.permute(3, 2, 0, 1).contiguous().permute(2, 3, 1, 0)

        return x, weight, offset, bias, stride, pad, dilation

    def _test_forward(self, device, contiguous):
        x, _, offset, _, stride, padding, dilation = self.get_fn_args(device, contiguous)
        in_channels = 6
        out_channels = 2
        kernel_size = (3, 2)
        groups = 2

        layer = ops.DeformConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups).to(device=x.device, dtype=x.dtype)
        res = layer(x, offset)

        weight = layer.weight.data
        bias = layer.bias.data
        expected = self.expected_fn(x, weight, offset, bias, stride=stride, padding=padding, dilation=dilation)

        self.assertTrue(torch.allclose(res, expected), '\nres:\n{}\nexpected:\n{}'.format(res, expected))

        # test for wrong sizes
        with self.assertRaises(RuntimeError):
            wrong_offset = torch.rand_like(offset[:, :2])
            res = layer(x, wrong_offset)

    def _test_backward(self, device, contiguous):
        x, weight, offset, bias, stride, padding, dilation = self.get_fn_args(device, contiguous)

        def func(x_, offset_, weight_, bias_):
            return ops.deform_conv2d(x_, offset_, weight_, bias_, stride=stride, padding=padding, dilation=dilation)

        gradcheck(func, (x, offset, weight, bias), nondet_tol=1e-5)

        @torch.jit.script
        def script_func(x_, offset_, weight_, bias_, stride_, pad_, dilation_):
            # type: (Tensor, Tensor, Tensor, Tensor, Tuple[int, int], Tuple[int, int], Tuple[int, int]) -> Tensor
            return ops.deform_conv2d(x_, offset_, weight_, bias_, stride=stride_, padding=pad_, dilation=dilation_)

        gradcheck(lambda z, off, wei, bi: script_func(z, off, wei, bi, stride, padding, dilation),
                  (x, offset, weight, bias), nondet_tol=1e-5)

        # Test from https://github.com/pytorch/vision/issues/2598
        # Run on CUDA only
        if "cuda" in device.type:
            # compare grads computed on CUDA with grads computed on CPU
            true_cpu_grads = None

            init_weight = torch.randn(9, 9, 3, 3, requires_grad=True)
            img = torch.randn(8, 9, 1000, 110)
            offset = torch.rand(8, 2 * 3 * 3, 1000, 110)

            if not contiguous:
                img = img.permute(0, 1, 3, 2).contiguous().permute(0, 1, 3, 2)
                offset = offset.permute(1, 3, 0, 2).contiguous().permute(2, 0, 3, 1)
                weight = init_weight.permute(3, 2, 0, 1).contiguous().permute(2, 3, 1, 0)
            else:
                weight = init_weight

            for d in ["cpu", "cuda"]:

                out = ops.deform_conv2d(img.to(d), offset.to(d), weight.to(d), padding=1)
                out.mean().backward()
                if true_cpu_grads is None:
                    true_cpu_grads = init_weight.grad
                    self.assertTrue(true_cpu_grads is not None)
                else:
                    self.assertTrue(init_weight.grad is not None)
                    res_grads = init_weight.grad.to("cpu")
                    self.assertTrue(true_cpu_grads.allclose(res_grads))


class FrozenBNTester(unittest.TestCase):
    def test_frozenbatchnorm2d_repr(self):
        num_features = 32
        t = ops.misc.FrozenBatchNorm2d(num_features)

        # Check integrity of object __repr__ attribute
        expected_string = f"FrozenBatchNorm2d({num_features})"
        self.assertEqual(t.__repr__(), expected_string)

    def test_frozenbatchnorm2d_eps(self):
        sample_size = (4, 32, 28, 28)
        x = torch.rand(sample_size)
        state_dict = dict(weight=torch.rand(sample_size[1]),
                          bias=torch.rand(sample_size[1]),
                          running_mean=torch.rand(sample_size[1]),
                          running_var=torch.rand(sample_size[1]),
                          num_batches_tracked=torch.tensor(100))

        # Check that default eps is zero for backward-compatibility
        fbn = ops.misc.FrozenBatchNorm2d(sample_size[1])
        fbn.load_state_dict(state_dict, strict=False)
        bn = torch.nn.BatchNorm2d(sample_size[1], eps=0).eval()
        bn.load_state_dict(state_dict)
        # Difference is expected to fall in an acceptable range
        self.assertTrue(torch.allclose(fbn(x), bn(x), atol=1e-6))

        # Check computation for eps > 0
        fbn = ops.misc.FrozenBatchNorm2d(sample_size[1], eps=1e-5)
        fbn.load_state_dict(state_dict, strict=False)
        bn = torch.nn.BatchNorm2d(sample_size[1], eps=1e-5).eval()
        bn.load_state_dict(state_dict)
        self.assertTrue(torch.allclose(fbn(x), bn(x), atol=1e-6))

    def test_frozenbatchnorm2d_n_arg(self):
        """Ensure a warning is thrown when passing `n` kwarg
        (remove this when support of `n` is dropped)"""
        self.assertWarns(DeprecationWarning, ops.misc.FrozenBatchNorm2d, 32, eps=1e-5, n=32)


class BoxConversionTester(unittest.TestCase):
    @staticmethod
    def _get_box_sequences():
        # Define here the argument type of `boxes` supported by region pooling operations
        box_tensor = torch.tensor([[0, 0, 0, 100, 100], [1, 0, 0, 100, 100]], dtype=torch.float)
        box_list = [torch.tensor([[0, 0, 100, 100]], dtype=torch.float),
                    torch.tensor([[0, 0, 100, 100]], dtype=torch.float)]
        box_tuple = tuple(box_list)
        return box_tensor, box_list, box_tuple

    def test_check_roi_boxes_shape(self):
        # Ensure common sequences of tensors are supported
        for box_sequence in self._get_box_sequences():
            self.assertIsNone(ops._utils.check_roi_boxes_shape(box_sequence))

    def test_convert_boxes_to_roi_format(self):
        # Ensure common sequences of tensors yield the same result
        ref_tensor = None
        for box_sequence in self._get_box_sequences():
            if ref_tensor is None:
                ref_tensor = box_sequence
            else:
                self.assertTrue(torch.equal(ref_tensor, ops._utils.convert_boxes_to_roi_format(box_sequence)))


if __name__ == '__main__':
    unittest.main()
