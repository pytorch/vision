import torch

def peaksmask2D(heatmap: torch.tensor, threshold: float):
    """
    Performs peak detection across a batch of 2D heatmaps with a 3x3 window. Peaks cannot be found on
    pixels bordering the image. 

    Arguments:
        heatmap (Tensor[batch_size, in_channels, in_height, in_width]): input tensor
        threshold (float): the minimum magnitude of the peak

    Returns:
        output (Tensor[batch_size, in_channels, in_height, in_width]): boolean mask of where the peaks are

    Examples::
        >>> # in ObjectsAsPoitnts you use peaks in heatmap to lookup
        >>> # the value in boxes and offsets;
        >>> heatmaps, boxes, offsets = model(images)
        >>> peaks = peaks2D_mask(heatmaps)
        >>> # now use our peak mask to extract values
        >>> bbox_shapes = boxes[peaks]
        >>> bbox_offsets = offsets[peaks]
        >>> confidences = heatmaps[peaks]
    """

    heatmap_top_left = heatmap[:, :, :-2, :-2]
    heatmap_top = heatmap[:, :, :-2, 1:-1]
    heatmap_top_right = heatmap[:, :, :-2, 2:]

    heatmap_left = heatmap[:, :, 1:-1, :-2]
    heatmap_crop = heatmap[:, :, 1:-1, 1:-1]
    heatmap_right = heatmap[:, :, 1:-1, 2:]

    heatmap_bottom_left = heatmap[:, :, 2:, :-2]
    heatmap_bottom = heatmap[:, :, 2:, 1:-1]
    heatmap_bottom_right = heatmap[:, :, 2:, 2:]

    is_peak = heatmap_crop > threshold
    is_peak &= heatmap_top_left < heatmap_crop
    is_peak &= heatmap_top < heatmap_crop
    is_peak &= heatmap_top_right < heatmap_crop
    is_peak &= heatmap_left < heatmap_crop
    is_peak &= heatmap_bottom_left < heatmap_crop
    is_peak &= heatmap_right <= heatmap_crop
    is_peak &= heatmap_bottom <= heatmap_crop
    is_peak &= heatmap_bottom_right <= heatmap_crop

    full_is_peak = torch.zeros_like(heatmap, dtype=torch.bool)
    full_is_peak[:, :, 1:-1, 1:-1] = is_peak

    return full_is_peak