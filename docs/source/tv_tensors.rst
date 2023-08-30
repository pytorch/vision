.. _tv_tensors:

TVTensors
==========

.. currentmodule:: torchvision.tv_tensors

TVTensors are :class:`torch.Tensor` subclasses which the v2 :ref:`transforms
<transforms>` use under the hood to dispatch their inputs to the appropriate
lower-level kernels. Most users do not need to manipulate TVTensors directly and
can simply rely on dataset wrapping - see e.g.
:ref:`sphx_glr_auto_examples_transforms_plot_transforms_e2e.py`.

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
