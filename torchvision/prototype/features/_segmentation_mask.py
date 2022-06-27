from __future__ import annotations

from ._feature import _Feature


class SegmentationMask(_Feature):
    def horizontal_flip(self) -> SegmentationMask:
        output = self._F.horizontal_flip_segmentation_mask(self)
        return SegmentationMask.new_like(self, output)

    def vertical_flip(self) -> SegmentationMask:
        output = self._F.vertical_flip_segmentation_mask(self)
        return SegmentationMask.new_like(self, output)

    def resize(self, size, *, interpolation, max_size, antialias) -> SegmentationMask:
        interpolation, antialias  # unused
        output = self._F.resize_segmentation_mask(self, size, max_size=max_size)
        return SegmentationMask.new_like(self, output)

    def center_crop(self, output_size) -> SegmentationMask:
        output = self._F.center_crop_segmentation_mask(self, output_size=output_size)
        return SegmentationMask.new_like(self, output)

    def resized_crop(self, top, left, height, width, *, size, interpolation, antialias) -> SegmentationMask:
        # TODO: untested right now
        interpolation, antialias  # unused
        output = self._F.resized_crop_segmentation_mask(self, top, left, height, width, size=list(size))
        return SegmentationMask.new_like(self, output)

    def pad(self, padding, *, fill, padding_mode) -> SegmentationMask:
        fill  # unused
        output = self._F.pad_segmentation_mask(self, padding, padding_mode=padding_mode)
        return SegmentationMask.new_like(self, output)
