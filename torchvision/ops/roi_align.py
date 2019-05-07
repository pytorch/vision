import torch
from torch import nn

from torch.autograd import Function
from torch.autograd.function import once_differentiable

from torch.nn.modules.utils import _pair

from torchvision import _C
from ._utils import convert_boxes_to_roi_format


class _RoIAlignFunction(Function):
    @staticmethod
    def forward(ctx, input, roi, output_size, spatial_scale, sampling_ratio):
        ctx.save_for_backward(roi)
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        ctx.input_shape = input.size()
        output = _C.roi_align_forward(
            input, roi, spatial_scale,
            output_size[0], output_size[1], sampling_ratio)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        rois, = ctx.saved_tensors
        output_size = ctx.output_size
        spatial_scale = ctx.spatial_scale
        sampling_ratio = ctx.sampling_ratio
        bs, ch, h, w = ctx.input_shape
        grad_input = _C.roi_align_backward(
            grad_output, rois, spatial_scale,
            output_size[0], output_size[1], bs, ch, h, w, sampling_ratio)
        return grad_input, None, None, None, None


def roi_align(input, boxes, output_size, spatial_scale=1.0, sampling_ratio=-1):
    """
    Performs Region of Interest (RoI) Align operator described in Mask R-CNN

    Arguments:
        input (Tensor[N, C, H, W]): input tensor
        boxes (Tensor[K, 5] or List[Tensor[L, 4]]): the box coordinates in x1,y1,x2,y2
            format where the regions will be taken from. If a single Tensor is passed,
            then the first column should contain the batch index. If a list of Tensors
            is passed, then each Tensor will correspond to the boxes for an element i
            in a batch
        output_size (int or Tuple[int, int]): the size of the output after the cropping
            is performed, as (height, width)
        spatial_scale (float): a scaling factor that maps the input coordinates to
            the box coordinates. Default: 1.0
        sampling_ratio (int): number of sampling points in the interpolation grid
            used to compute the output value of each pooled output bin. If > 0,
            then exactly sampling_ratio x sampling_ratio grid points are used. If
            <= 0, then an adaptive number of grid points are used (computed as
            ceil(roi_width / pooled_w), and likewise for height). Default: -1

    Returns:
        output (Tensor[K, C, output_size[0], output_size[1]])
    """
    rois = boxes
    if not isinstance(rois, torch.Tensor):
        rois = convert_boxes_to_roi_format(rois)
    return _RoIAlignFunction.apply(input, rois, output_size, spatial_scale, sampling_ratio)


class RoIAlign(nn.Module):
    """
    See roi_align
    """
    def __init__(self, output_size, spatial_scale, sampling_ratio):
        super(RoIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

    def forward(self, input, rois):
        return roi_align(input, rois, self.output_size, self.spatial_scale, self.sampling_ratio)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'output_size=' + str(self.output_size)
        tmpstr += ', spatial_scale=' + str(self.spatial_scale)
        tmpstr += ', sampling_ratio=' + str(self.sampling_ratio)
        tmpstr += ')'
        return tmpstr
