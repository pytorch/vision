"""
==================================
Quick-start with Transforms API v2
==================================

This example illustrates how to update user's data-augmentations pipeline with
new Torchvision Transforms API v2 (:ref:`image transformations <transforms>`).

"""

a = 1

#######################################
# Transforming images on GPU
# --------------------------
# Most transforms natively support tensors on top of PIL images (to visualize
# the effect of the transforms, you may refer to see
# :ref:`sphx_glr_auto_examples_plot_transforms.py`).
# Using tensor images, we can run the transforms on GPUs if cuda is available!
#
#   .. code-block:: diff
#
#       - a = 10
#       + a = 12
#

"""
various features that are now supported by the
:ref:`image transformations <transforms>` on Tensor images. In particular, we
show how image transforms can be performed on GPU, and how one can also script
them using JIT compilation.

Prior to v0.8.0, transforms in torchvision have traditionally been PIL-centric
and presented multiple limitations due to that. Now, since v0.8.0, transforms
implementations are Tensor and PIL compatible, and we can achieve the following
new features:

- transform multi-band torch tensor images (with more than 3-4 channels)
- torchscript transforms together with your model for deployment
- support for GPU acceleration
- batched transformation such as for videos
- read and decode data directly as torch tensor with torchscript support (for PNG and JPEG image formats)

.. note::
    These features are only possible with **Tensor** images.

"""


