# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from torchvision.ops import roi_align
from torchvision.ops.boxes import box_area

from typing import Optional, List



# TODO: https://github.com/pytorch/pytorch/issues/26744
@torch.jit.script
class LevelMapper(object):
    """Determine which FPN level each RoI in a set of RoIs should map to based
    on the heuristic in the FPN paper.

    Arguments:
        k_min (int)
        k_max (int)
        canonical_scale (int)
        canonical_level (int)
        eps (float)
    """

    def __init__(self, k_min, k_max, canonical_scale=224, canonical_level=4, eps=1e-6):
        # type: (int, int, int, int, float)
        self.k_min = k_min
        self.k_max = k_max
        self.s0 = canonical_scale
        self.lvl0 = canonical_level
        self.eps = eps

    def run(self, boxlists):
        # type: (List[Tensor])
        """
        Arguments:
            boxlists (List[BoxList])
        """
        # Compute level ids
        s = torch.sqrt(torch.cat([box_area(boxlist) for boxlist in boxlists]))

        # Eqn.(1) in FPN paper
        target_lvls = torch.floor(self.lvl0 + torch.log2(s / self.s0 + self.eps))
        target_lvls = torch.clamp(target_lvls, min=self.k_min, max=self.k_max)
        return target_lvls.to(torch.int64) - self.k_min


class MultiScaleRoIAlign(nn.Module):
    """
    Multi-scale RoIAlign pooling, which is useful for detection with or without FPN.

    It infers the scale of the pooling via the heuristics present in the FPN paper.

    Arguments:
        featmap_names (List[str]): the names of the feature maps that will be used
            for the pooling.
        output_size (List[Tuple[int, int]] or List[int]): output size for the pooled region
        sampling_ratio (int): sampling ratio for ROIAlign

    Examples::

        >>> m = torchvision.ops.MultiScaleRoIAlign(['feat1', 'feat3'], 3, 2)
        >>> i = OrderedDict()
        >>> i['feat1'] = torch.rand(1, 5, 64, 64)
        >>> i['feat2'] = torch.rand(1, 5, 32, 32)  # this feature won't be used in the pooling
        >>> i['feat3'] = torch.rand(1, 5, 16, 16)
        >>> # create some random bounding boxes
        >>> boxes = torch.rand(6, 4) * 256; boxes[:, 2:] += boxes[:, :2]
        >>> # original image size, before computing the feature maps
        >>> image_sizes = [(512, 512)]
        >>> output = m(i, [boxes], image_sizes)
        >>> print(output.shape)
        >>> torch.Size([6, 5, 3, 3])

    """
    __annotations__ = {
        'scales': Optional[List[float]],
        'map_levels': Optional[LevelMapper]
    }

    def __init__(self, featmap_names, output_size, sampling_ratio):
        super(MultiScaleRoIAlign, self).__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.featmap_names = list(map(lambda x: str(x), featmap_names))
        self.sampling_ratio = sampling_ratio
        self.output_size = tuple(output_size)
        self.scales = None
        self.map_levels = None

    def convert_to_roi_format(self, boxes):
        # type: (List[Tensor])
        concat_boxes = torch.cat(boxes, dim=0)
        device, dtype = concat_boxes.device, concat_boxes.dtype

        id_tensors = []
        for i, b in enumerate(boxes):
            id_tensors.append(torch.full((len(b), 1), i, dtype=dtype, device=device))

        ids = torch.cat(id_tensors, dim=0)
        rois = torch.cat([ids, concat_boxes], dim=1)
        return rois

    def infer_scale(self, feature, original_size):
        # type: (Tensor, List[int])
        # assumption: the scale is of the form 2 ** (-k), with k integer
        size = feature.shape[-2:]
        possible_scales = torch.jit.annotate(List[float], [])
        for s1, s2 in zip(size, original_size):
            approx_scale = float(s1) / float(s2)
            exp = float(torch.tensor(approx_scale).log2().round().item())
            scale = 2. ** exp
            possible_scales.append(scale)
        # assert possible_scales[0] == possible_scales[1], str(possible_scales)
        return possible_scales[0]

    def setup_scales(self, features, image_shapes):
        # type: (List[Tensor], List[Tuple[int, int]])
        max_x = 0
        max_y = 0
        for shape in image_shapes:
            max_x = max(shape[0], max_x)
            max_y = max(shape[1], max_y)
        original_input_shape = (max_x, max_y)
        # original_input_shape = tuple(max(s) for s in zip(*image_shapes))
        print(original_input_shape)
        scales = [self.infer_scale(feat, original_input_shape) for feat in features]
        # get the levels in the feature map by leveraging the fact that the network always
        # downsamples by a factor of 2 at each level.
        lvl_min = -int(torch.log2(torch.tensor(scales[0], dtype=torch.float32)).item())
        lvl_max = -int(torch.log2(torch.tensor(scales[-1], dtype=torch.float32)).item())
        self.scales = scales

        # TODO: https://github.com/pytorch/pytorch/issues/26744
        self.map_levels = LevelMapper(lvl_min, lvl_max, 224, 4, 1e-6)

    def forward(self, x, boxes, image_shapes):
        # type: (Dict[str, Tensor], List[Tensor], List[Tuple[int, int]])
        """
        Arguments:
            x (OrderedDict[Tensor]): feature maps for each level. They are assumed to have
                all the same number of channels, but they can have different sizes.
            boxes (List[Tensor[N, 4]]): boxes to be used to perform the pooling operation, in
                (x1, y1, x2, y2) format and in the image reference size, not the feature map
                reference.
            image_shapes (List[Tuple[height, width]]): the sizes of each image before they
                have been fed to a CNN to obtain feature maps. This allows us to infer the
                scale factor for each one of the levels to be pooled.
        Returns:
            result (Tensor)
        """
        new_x = []
        for k, v in x.items():
            if k in self.featmap_names:
                new_x.append(v)
        num_levels = len(new_x)
        rois = self.convert_to_roi_format(boxes)
        if self.scales is None:
            self.setup_scales(new_x, image_shapes)

        scales = self.scales
        assert scales is not None

        if num_levels == 1:
            return roi_align(
                new_x[0], rois,
                output_size=self.output_size,
                spatial_scale=scales[0],
                sampling_ratio=self.sampling_ratio
            )


        mapper = self.map_levels
        assert mapper is not None

        levels = mapper.run(boxes)

        num_rois = len(rois)
        num_channels = new_x[0].shape[1]

        dtype, device = new_x[0].dtype, new_x[0].device
        result = torch.zeros(
            (num_rois, num_channels,) + self.output_size,
            dtype=dtype,
            device=device,
        )

        for level, (per_level_feature, scale) in enumerate(zip(new_x, scales)):
            idx_in_level = torch.nonzero(levels == level).squeeze(1)
            rois_per_level = rois[idx_in_level]

            result[idx_in_level] = roi_align(
                per_level_feature, rois_per_level,
                output_size=self.output_size,
                spatial_scale=scale, sampling_ratio=self.sampling_ratio
            )

        return result
