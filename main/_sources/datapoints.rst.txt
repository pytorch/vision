Datapoints
==========

.. currentmodule:: torchvision.datapoints

Datapoints are tensor subclasses which the :mod:`~torchvision.transforms.v2` v2 transforms use under the hood to
dispatch their inputs to the appropriate lower-level kernels. Most users do not
need to manipulate datapoints directly and can simply rely on dataset wrapping -
see e.g. :ref:`sphx_glr_auto_examples_v2_transforms_plot_transforms_v2_e2e.py`.

.. autosummary::
    :toctree: generated/
    :template: class.rst

    Image
    Video
    BoundingBoxFormat
    BoundingBoxes
    Mask
    Datapoint
    set_return_type
    wrap
