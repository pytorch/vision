.. _transforms:

Transforming and augmenting images
==================================

.. currentmodule:: torchvision.transforms


.. note::
    In 0.15, we released a new set of transforms available in the
    ``torchvision.transforms.v2`` namespace, which add support for transforming
    not just images but also bounding boxes, masks, or videos. These transforms
    are fully backward compatible with the current ones, and you'll see them
    documented below with a `v2.` prefix. To get started with those new
    transforms, you can check out
    :ref:`sphx_glr_auto_examples_plot_transforms_v2_e2e.py`.
    Note that these transforms are still BETA, and while we don't expect major
    breaking changes in the future, some APIs may still change according to user
    feedback. Please submit any feedback you may have `here
    <https://github.com/pytorch/vision/issues/6753>`_, and you can also check
    out `this issue <https://github.com/pytorch/vision/issues/7319>`_ to learn
    more about the APIs that we suspect might involve future changes.

Transforms are common image transformations available in the
``torchvision.transforms`` module. They can be chained together using
:class:`Compose`.
Most transform classes have a function equivalent: :ref:`functional
transforms <functional_transforms>` give fine-grained control over the
transformations.
This is useful if you have to build a more complex transformation pipeline
(e.g. in the case of segmentation tasks).

Most transformations accept both `PIL <https://pillow.readthedocs.io>`_ images
and tensor images, although some transformations are PIL-only and some are
tensor-only. The :ref:`conversion_transforms` may be used to convert to and from
PIL images, or for converting dtypes and ranges.

The transformations that accept tensor images also accept batches of tensor
images. A Tensor Image is a tensor with ``(C, H, W)`` shape, where ``C`` is a
number of channels, ``H`` and ``W`` are image height and width. A batch of
Tensor Images is a tensor of ``(B, C, H, W)`` shape, where ``B`` is a number
of images in the batch.

The expected range of the values of a tensor image is implicitly defined by
the tensor dtype. Tensor images with a float dtype are expected to have
values in ``[0, 1)``. Tensor images with an integer dtype are expected to
have values in ``[0, MAX_DTYPE]`` where ``MAX_DTYPE`` is the largest value
that can be represented in that dtype.

Randomized transformations will apply the same transformation to all the
images of a given batch, but they will produce different transformations
across calls. For reproducible transformations across calls, you may use
:ref:`functional transforms <functional_transforms>`.

The following examples illustrate the use of the available transforms:

    * :ref:`sphx_glr_auto_examples_plot_transforms.py`

        .. figure:: ../source/auto_examples/images/sphx_glr_plot_transforms_001.png
            :align: center
            :scale: 65%

    * :ref:`sphx_glr_auto_examples_plot_scripted_tensor_transforms.py`

        .. figure:: ../source/auto_examples/images/sphx_glr_plot_scripted_tensor_transforms_001.png
            :align: center
            :scale: 30%

.. warning::

    Since v0.8.0 all random transformations are using torch default random generator to sample random parameters.
    It is a backward compatibility breaking change and user should set the random state as following:

    .. code:: python

        # Previous versions
        # import random
        # random.seed(12)

        # Now
        import torch
        torch.manual_seed(17)

    Please, keep in mind that the same seed for torch random generator and Python random generator will not
    produce the same results.


Transforms scriptability
------------------------

.. TODO: Add note about v2 scriptability (in next PR)

In order to script the transformations, please use ``torch.nn.Sequential`` instead of :class:`Compose`.

.. code:: python

    transforms = torch.nn.Sequential(
        transforms.CenterCrop(10),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    )
    scripted_transforms = torch.jit.script(transforms)

Make sure to use only scriptable transformations, i.e. that work with ``torch.Tensor`` and does not require
`lambda` functions or ``PIL.Image``.

For any custom transformations to be used with ``torch.jit.script``, they should be derived from ``torch.nn.Module``.


Geometry
--------

.. autosummary::
    :toctree: generated/
    :template: class.rst

    Resize
    v2.Resize
    v2.ScaleJitter
    v2.RandomShortestSize
    v2.RandomResize
    RandomCrop
    v2.RandomCrop
    RandomResizedCrop
    v2.RandomResizedCrop
    v2.RandomIoUCrop
    CenterCrop
    v2.CenterCrop
    FiveCrop
    v2.FiveCrop
    TenCrop
    v2.TenCrop
    Pad
    v2.Pad
    v2.RandomZoomOut
    RandomRotation
    v2.RandomRotation
    RandomAffine
    v2.RandomAffine
    RandomPerspective
    v2.RandomPerspective
    ElasticTransform
    v2.ElasticTransform
    RandomHorizontalFlip
    v2.RandomHorizontalFlip
    RandomVerticalFlip
    v2.RandomVerticalFlip


Color
-----

.. autosummary::
    :toctree: generated/
    :template: class.rst

    ColorJitter
    v2.ColorJitter
    v2.RandomChannelPermutation
    v2.RandomPhotometricDistort
    Grayscale
    v2.Grayscale
    RandomGrayscale
    v2.RandomGrayscale
    GaussianBlur
    v2.GaussianBlur
    RandomInvert
    v2.RandomInvert
    RandomPosterize
    v2.RandomPosterize
    RandomSolarize
    v2.RandomSolarize
    RandomAdjustSharpness
    v2.RandomAdjustSharpness
    RandomAutocontrast
    v2.RandomAutocontrast
    RandomEqualize
    v2.RandomEqualize

Composition
-----------

.. autosummary::
    :toctree: generated/
    :template: class.rst

    Compose
    v2.Compose
    RandomApply
    v2.RandomApply
    RandomChoice
    v2.RandomChoice
    RandomOrder
    v2.RandomOrder

Miscellaneous
-------------

.. autosummary::
    :toctree: generated/
    :template: class.rst

    LinearTransformation
    v2.LinearTransformation
    Normalize
    v2.Normalize
    RandomErasing
    v2.RandomErasing
    Lambda
    v2.Lambda
    v2.SanitizeBoundingBoxes
    v2.ClampBoundingBoxes
    v2.UniformTemporalSubsample

.. _conversion_transforms:

Conversion
----------

.. note::
    Beware, some of these conversion transforms below will scale the values
    while performing the conversion, while some may not do any scaling. By
    scaling, we mean e.g. that a ``uint8`` -> ``float32`` would map the [0,
    255] range into [0, 1] (and vice-versa).
    
.. autosummary::
    :toctree: generated/
    :template: class.rst

    ToPILImage
    v2.ToPILImage
    v2.ToImagePIL
    ToTensor
    v2.ToTensor
    PILToTensor
    v2.PILToTensor
    v2.ToImageTensor
    ConvertImageDtype
    v2.ConvertImageDtype
    v2.ToDtype
    v2.ConvertBoundingBoxFormat

Auto-Augmentation
-----------------

`AutoAugment <https://arxiv.org/pdf/1805.09501.pdf>`_ is a common Data Augmentation technique that can improve the accuracy of Image Classification models.
Though the data augmentation policies are directly linked to their trained dataset, empirical studies show that
ImageNet policies provide significant improvements when applied to other datasets.
In TorchVision we implemented 3 policies learned on the following datasets: ImageNet, CIFAR10 and SVHN.
The new transform can be used standalone or mixed-and-matched with existing transforms:

.. autosummary::
    :toctree: generated/
    :template: class.rst

    AutoAugmentPolicy
    AutoAugment
    v2.AutoAugment
    RandAugment
    v2.RandAugment
    TrivialAugmentWide
    v2.TrivialAugmentWide
    AugMix
    v2.AugMix

CutMix - MixUp
--------------

CutMix and MixUp are special transforms that
are meant to be used on batches rather than on individual images, because they
are combining pairs of images together. These can be used after the dataloader
(once the samples are batched), or part of a collation function. See
:ref:`sphx_glr_auto_examples_plot_cutmix_mixup.py` for detailed usage examples.

.. autosummary::
    :toctree: generated/
    :template: class.rst

    v2.CutMix
    v2.MixUp

.. _functional_transforms:

Functional Transforms
---------------------

.. currentmodule:: torchvision.transforms.functional


.. note::
    You'll find below the documentation for the existing
    ``torchvision.transforms.functional`` namespace. The
    ``torchvision.transforms.v2.functional`` namespace exists as well and can be
    used! The same functionals are present, so you simply need to change your
    import to rely on the ``v2`` namespace.

Functional transforms give you fine-grained control of the transformation pipeline.
As opposed to the transformations above, functional transforms don't contain a random number
generator for their parameters.
That means you have to specify/generate all parameters, but the functional transform will give you
reproducible results across calls.

Example:
you can apply a functional transform with the same parameters to multiple images like this:

.. code:: python

    import torchvision.transforms.functional as TF
    import random

    def my_segmentation_transforms(image, segmentation):
        if random.random() > 0.5:
            angle = random.randint(-30, 30)
            image = TF.rotate(image, angle)
            segmentation = TF.rotate(segmentation, angle)
        # more transforms ...
        return image, segmentation


Example:
you can use a functional transform to build transform classes with custom behavior:

.. code:: python

    import torchvision.transforms.functional as TF
    import random

    class MyRotationTransform:
        """Rotate by one of the given angles."""

        def __init__(self, angles):
            self.angles = angles

        def __call__(self, x):
            angle = random.choice(self.angles)
            return TF.rotate(x, angle)

    rotation_transform = MyRotationTransform(angles=[-30, -15, 0, 15, 30])


.. autosummary::
    :toctree: generated/
    :template: function.rst

    adjust_brightness
    adjust_contrast
    adjust_gamma
    adjust_hue
    adjust_saturation
    adjust_sharpness
    affine
    autocontrast
    center_crop
    convert_image_dtype
    crop
    equalize
    erase
    five_crop
    gaussian_blur
    get_dimensions
    get_image_num_channels
    get_image_size
    hflip
    invert
    normalize
    pad
    perspective
    pil_to_tensor
    posterize
    resize
    resized_crop
    rgb_to_grayscale
    rotate
    solarize
    ten_crop
    to_grayscale
    to_pil_image
    to_tensor
    vflip

Developer tools
---------------

.. currentmodule:: torchvision.transforms.v2.functional

.. autosummary::
    :toctree: generated/
    :template: function.rst

    register_kernel
