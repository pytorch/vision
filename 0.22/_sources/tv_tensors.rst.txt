.. _tv_tensors:

TVTensors
==========

.. currentmodule:: torchvision.tv_tensors

TVTensors are :class:`torch.Tensor` subclasses which the v2 :ref:`transforms
<transforms>` use under the hood to dispatch their inputs to the appropriate
lower-level kernels. Most users do not need to manipulate TVTensors directly.

Refer to
:ref:`sphx_glr_auto_examples_transforms_plot_transforms_getting_started.py` for
an introduction to TVTensors, or
:ref:`sphx_glr_auto_examples_transforms_plot_tv_tensors.py` for more advanced
info.

.. autosummary::
    :toctree: generated/
    :template: class.rst

    Image
    Video
    BoundingBoxFormat
    BoundingBoxes
    Mask
    TVTensor
    set_return_type
    wrap
