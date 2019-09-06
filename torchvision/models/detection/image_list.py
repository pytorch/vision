# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from __future__ import division

import torch
import math

torch = torch.nested.monkey_patch(torch)


class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
    """

    def __init__(self, tensors):
        """
        Arguments:
            tensors (tensor)
            image_sizes (list[tuple[int, int]])
        """
        if isinstance(tensors, list):
            print("ImageList based on a list")
            tensors = self._batch_images(tensors).unbind()
            self._tensors = torch.nested_tensor(tensors)
            # self._image_sizes = image_sizes
        else:
            print("ImageList based on a Tensor?")
            raise RuntimeError("")

    def _batch_images(self, images, size_divisible=32):    
        # concatenate    
        max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))    

        stride = size_divisible    
        max_size = list(max_size)    
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)    
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)    
        max_size = tuple(max_size)    

        batch_shape = (len(images),) + max_size    
        batched_imgs = images[0].new(*batch_shape).zero_()    
        for img, pad_img in zip(images, batched_imgs):    
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
        return batched_imgs

    @property
    def tensors(self):
        print("Getting tensors")
        return self._tensors

    @property
    def image_sizes(self):
        print("Getting image_sizes")
        return [img.shape[-2:] for img in self._tensors.unbind()]

    @property
    def shape(self):
        print("Getting tensors shape")
        return self._batch_images(self._tensors.unbind()).shape

    @tensors.setter
    def tensors(self, tensors):
        print("Setting tensors")
        raise RuntimeError("")

    @image_sizes.setter
    def image_sizes(self, image_sizes):
        print("Setting image_sizes")
        raise RuntimeError("")

    def to(self, *args, **kwargs):
        cast_tensor = self.tensors.to(*args, **kwargs)
        return ImageList(cast_tensor, self.image_sizes)
