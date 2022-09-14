from torchvision.prototype import features, transforms


class RandomCrop(transforms.RandomCrop):
    def _transform(self, inpt, params):
        if not isinstance(inpt, features.Mask):
            return super()._transform(inpt, params)

        # `Mask`'s should be padded with 255 to indicate an area that should not be used in the loss calculation. See
        # https://stackoverflow.com/questions/49629933/ground-truth-pixel-labels-in-pascal-voc-for-semantic-segmentation
        # for details.
        # FIXME: Using different values for `fill` based on the input type is not supported by `transforms.RandomCrop`.
        #  Thus, we emulate it here. See https://github.com/pytorch/vision/issues/6568.
        fill = self.fill
        try:
            self.fill = 255
            return super()._transform(inpt, params)
        finally:
            self.fill = fill
