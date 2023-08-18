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
    :ref:`sphx_glr_auto_examples_v2_transforms_plot_transforms_v2_e2e.py`.
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

    * :ref:`sphx_glr_auto_examples_others_plot_transforms.py`

        .. figure:: ../source/auto_examples/others/images/sphx_glr_plot_transforms_001.png
            :align: center
            :scale: 65%

    * :ref:`sphx_glr_auto_examples_others_plot_scripted_tensor_transforms.py`

        .. figure:: ../source/auto_examples/others/images/sphx_glr_plot_scripted_tensor_transforms_001.png
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



V2 - Recommended
----------------

Geometry
^^^^^^^^


Resizing
""""""""

.. autosummary::
    :toctree: generated/
    :template: class.rst

    v2.Resize
    v2.ScaleJitter
    v2.RandomShortestSize
    v2.RandomResize

Functionals

.. autosummary::
    :toctree: generated/
    :template: function.rst

    v2.functional.resize

Cropping
""""""""

.. autosummary::
    :toctree: generated/
    :template: class.rst

    v2.RandomCrop
    v2.RandomResizedCrop
    v2.RandomIoUCrop
    v2.CenterCrop
    v2.FiveCrop
    v2.TenCrop

Functionals

.. autosummary::
    :toctree: generated/
    :template: function.rst

    v2.functional.crop
    v2.functional.resized_crop
    v2.functional.ten_crop
    v2.functional.center_crop
    v2.functional.five_crop

Others
""""""

.. autosummary::
    :toctree: generated/
    :template: class.rst

    v2.RandomHorizontalFlip
    v2.RandomVerticalFlip
    v2.Pad
    v2.RandomZoomOut
    v2.RandomRotation
    v2.RandomAffine
    v2.RandomPerspective
    v2.ElasticTransform

Functionals

.. autosummary::
    :toctree: generated/
    :template: function.rst

    v2.functional.horizontal_flip
    v2.functional.vertical_flip
    v2.functional.pad
    v2.functional.rotate
    v2.functional.affine
    v2.functional.perspective
    v2.functional.elastic

Color
^^^^^

.. autosummary::
    :toctree: generated/
    :template: class.rst

    v2.ColorJitter
    v2.RandomChannelPermutation
    v2.RandomPhotometricDistort
    v2.Grayscale
    v2.RandomGrayscale
    v2.GaussianBlur
    v2.RandomInvert
    v2.RandomPosterize
    v2.RandomSolarize
    v2.RandomAdjustSharpness
    v2.RandomAutocontrast
    v2.RandomEqualize

Functionals

.. autosummary::
    :toctree: generated/
    :template: function.rst

    v2.functional.permute_channels
    v2.functional.rgb_to_grayscale
    v2.functional.to_grayscale
    v2.functional.gaussian_blur
    v2.functional.invert
    v2.functional.posterize
    v2.functional.solarize
    v2.functional.adjust_sharpness
    v2.functional.autocontrast
    v2.functional.adjust_contrast
    v2.functional.equalize
    v2.functional.adjust_brightness
    v2.functional.adjust_saturation
    v2.functional.adjust_hue
    v2.functional.adjust_gamma


Composition
^^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :template: class.rst

    v2.Compose
    v2.RandomApply
    v2.RandomChoice
    v2.RandomOrder

Miscellaneous
^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :template: class.rst

    v2.LinearTransformation
    v2.Normalize
    v2.RandomErasing
    v2.Lambda
    v2.SanitizeBoundingBoxes
    v2.ClampBoundingBoxes
    v2.UniformTemporalSubsample

Functionals

.. autosummary::
    :toctree: generated/
    :template: function.rst

    v2.functional.normalize
    v2.functional.erase
    v2.functional.clamp_bounding_boxes
    v2.functional.uniform_temporal_subsample

.. _conversion_transforms:

Conversion
^^^^^^^^^^

.. note::
    Beware, some of these conversion transforms below will scale the values
    while performing the conversion, while some may not do any scaling. By
    scaling, we mean e.g. that a ``uint8`` -> ``float32`` would map the [0,
    255] range into [0, 1] (and vice-versa).

.. autosummary::
    :toctree: generated/
    :template: class.rst

    v2.ToImage
    v2.ToPureTensor
    v2.PILToTensor
    v2.ToPILImage
    v2.ToDtype
    v2.ConvertBoundingBoxFormat

functionals

.. autosummary::
    :toctree: generated/
    :template: functional.rst

    v2.functional.to_image
    v2.functional.pil_to_tensor
    v2.functional.to_pil_image
    v2.functional.to_dtype
    v2.functional.convert_bounding_box_format


Deprecated

.. autosummary::
    :toctree: generated/
    :template: class.rst

    v2.ToTensor
    v2.functional.to_tensor
    v2.ConvertImageDtype
    v2.functional.convert_image_dtype

Auto-Augmentation
^^^^^^^^^^^^^^^^^

`AutoAugment <https://arxiv.org/pdf/1805.09501.pdf>`_ is a common Data Augmentation technique that can improve the accuracy of Image Classification models.
Though the data augmentation policies are directly linked to their trained dataset, empirical studies show that
ImageNet policies provide significant improvements when applied to other datasets.
In TorchVision we implemented 3 policies learned on the following datasets: ImageNet, CIFAR10 and SVHN.
The new transform can be used standalone or mixed-and-matched with existing transforms:

.. autosummary::
    :toctree: generated/
    :template: class.rst

    v2.AutoAugment
    v2.RandAugment
    v2.TrivialAugmentWide
    v2.AugMix


CutMix - MixUp
^^^^^^^^^^^^^^

CutMix and MixUp are special transforms that
are meant to be used on batches rather than on individual images, because they
are combining pairs of images together. These can be used after the dataloader
(once the samples are batched), or part of a collation function. See
:ref:`sphx_glr_auto_examples_v2_transforms_plot_cutmix_mixup.py` for detailed usage examples.

.. autosummary::
    :toctree: generated/
    :template: class.rst

    v2.CutMix
    v2.MixUp

Developer tools
^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :template: function.rst

    v2.functional.register_kernel


V1
--

Geometry
^^^^^^^^

.. autosummary::
    :toctree: generated/
    :template: class.rst

    Resize
    RandomCrop
    RandomResizedCrop
    CenterCrop
    FiveCrop
    TenCrop
    Pad
    RandomRotation
    RandomAffine
    RandomPerspective
    ElasticTransform
    RandomHorizontalFlip
    RandomVerticalFlip


Color
^^^^^

.. autosummary::
    :toctree: generated/
    :template: class.rst

    ColorJitter
    Grayscale
    RandomGrayscale
    GaussianBlur
    RandomInvert
    RandomPosterize
    RandomSolarize
    RandomAdjustSharpness
    RandomAutocontrast
    RandomEqualize

Composition
^^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :template: class.rst

    Compose
    RandomApply
    RandomChoice
    RandomOrder

Miscellaneous
^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :template: class.rst

    LinearTransformation
    Normalize
    RandomErasing
    Lambda

Conversion
^^^^^^^^^^

.. note::
    Beware, some of these conversion transforms below will scale the values
    while performing the conversion, while some may not do any scaling. By
    scaling, we mean e.g. that a ``uint8`` -> ``float32`` would map the [0,
    255] range into [0, 1] (and vice-versa).
    
.. autosummary::
    :toctree: generated/
    :template: class.rst

    ToPILImage
    ToTensor
    PILToTensor
    ConvertImageDtype

Auto-Augmentation
^^^^^^^^^^^^^^^^^

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
    RandAugment
    TrivialAugmentWide
    AugMix


.. _functional_transforms:

Functional Transforms
^^^^^^^^^^^^^^^^^^^^^

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


Torchscript support
-------------------

Most transforms (v1 and v2) support torchscript. For composing transforms, use
:class:`torch.nn.Sequential` instead of ``Compose``:

.. code:: python

    transforms = torch.nn.Sequential(
        CenterCrop(10),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    )
    scripted_transforms = torch.jit.script(transforms)

.. warning::

    v2 transforms support torchscript, but if you call ``torch.jit.script()`` on
    a v2 **class** transform, you'll actually end up with its (scripted) v1
    equivalent.  This may lead to slightly different results between the
    scripted and eager executions due to implementation differences between v1
    and v2.

    If you really need torchscript support for the v2 tranforms, we recommend
    scripting the **functionals** from the
    ``torchvision.transforms.v2.functional`` namespace to avoid surprises.

For any custom transformations to be used with ``torch.jit.script``, they should be derived from ``torch.nn.Module``.
