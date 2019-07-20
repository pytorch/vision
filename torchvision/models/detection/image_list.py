# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from __future__ import division

import torch


class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
    """

    def __init__(self, tensors, image_sizes):
        raise NotImplementedError("")
