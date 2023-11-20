.. _transforms:

Transforming and augmenting images
==================================

.. currentmodule:: torchvision.transforms

Torchvision supports common computer vision transformations in the
``torchvision.transforms`` and ``torchvision.transforms.v2`` modules. Transforms
can be used to transform or augment data for training or inference of different
tasks (image classification, detection, segmentation, video classification).

.. code:: python

    # Image Classification
    import torch
    from torchvision.transforms import v2

    H, W = 32, 32
    img = torch.randint(0, 256, size=(3, H, W), dtype=torch.uint8)

    transforms = v2.Compose([
        v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = transforms(img)

.. code:: python

    # Detection (re-using imports and transforms from above)
    from torchvision import tv_tensors

    img = torch.randint(0, 256, size=(3, H, W), dtype=torch.uint8)
    boxes = torch.randint(0, H // 2, size=(3, 4))
    boxes[:, 2:] += boxes[:, :2]
    boxes = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(H, W))

    # The same transforms can be used!
    img, boxes = transforms(img, boxes)
    # And you can pass arbitrary input structures
    output_dict = transforms({"image": img, "boxes": boxes})

Transforms are typically passed as the ``transform`` or ``transforms`` argument
to the :ref:`Datasets <datasets>`.

Start here
----------

Whether you're new to Torchvision transforms, or you're already experienced with
them, we encourage you to start with
:ref:`sphx_glr_auto_examples_transforms_plot_transforms_getting_started.py` in
order to learn more about what can be done with the new v2 transforms.

Then, browse the sections in below this page for general information and
performance tips. The available transforms and functionals are listed in the
:ref:`API reference <v2_api_ref>`.

More information and tutorials can also be found in our :ref:`example gallery
<gallery>`, e.g. :ref:`sphx_glr_auto_examples_transforms_plot_transforms_e2e.py`
or :ref:`sphx_glr_auto_examples_transforms_plot_custom_transforms.py`.

.. _conventions:

Supported input types and conventions
-------------------------------------

Most transformations accept both `PIL <https://pillow.readthedocs.io>`_ images
and tensor inputs. Both CPU and CUDA tensors are supported.
The result of both backends (PIL or Tensors) should be very
close. In general, we recommend relying on the tensor backend :ref:`for
performance <transforms_perf>`.  The :ref:`conversion transforms
<conversion_transforms>` may be used to convert to and from PIL images, or for
converting dtypes and ranges.

Tensor image are expected to be of shape ``(C, H, W)``, where ``C`` is the
number of channels, and ``H`` and ``W`` refer to height and width. Most
transforms support batched tensor input. A batch of Tensor images is a tensor of
shape ``(N, C, H, W)``, where ``N`` is a number of images in the batch. The
:ref:`v2 <v1_or_v2>` transforms generally accept an arbitrary number of leading
dimensions ``(..., C, H, W)`` and can handle batched images or batched videos.

.. _range_and_dtype:

Dtype and expected value range
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The expected range of the values of a tensor image is implicitly defined by
the tensor dtype. Tensor images with a float dtype are expected to have
values in ``[0, 1]``. Tensor images with an integer dtype are expected to
have values in ``[0, MAX_DTYPE]`` where ``MAX_DTYPE`` is the largest value
that can be represented in that dtype. Typically, images of dtype
``torch.uint8`` are expected to have values in ``[0, 255]``.

Use :class:`~torchvision.transforms.v2.ToDtype` to convert both the dtype and
range of the inputs.

.. _v1_or_v2:

V1 or V2? Which one should I use?
---------------------------------

**TL;DR** We recommending using the ``torchvision.transforms.v2`` transforms
instead of those in ``torchvision.transforms``. They're faster and they can do
more things. Just change the import and you should be good to go. Moving
forward, new features and improvements will only be considered for the v2
transforms.

In Torchvision 0.15 (March 2023), we released a new set of transforms available
in the ``torchvision.transforms.v2`` namespace. These transforms have a lot of
advantages compared to the v1 ones (in ``torchvision.transforms``):

- They can transform images **but also** bounding boxes, masks, or videos. This
  provides support for tasks beyond image classification: detection, segmentation,
  video classification, etc. See
  :ref:`sphx_glr_auto_examples_transforms_plot_transforms_getting_started.py`
  and :ref:`sphx_glr_auto_examples_transforms_plot_transforms_e2e.py`.
- They support more transforms like :class:`~torchvision.transforms.v2.CutMix`
  and :class:`~torchvision.transforms.v2.MixUp`. See
  :ref:`sphx_glr_auto_examples_transforms_plot_cutmix_mixup.py`.
- They're :ref:`faster <transforms_perf>`.
- They support arbitrary input structures (dicts, lists, tuples, etc.).
- Future improvements and features will be added to the v2 transforms only.

These transforms are **fully backward compatible** with the v1 ones, so if
you're already using tranforms from ``torchvision.transforms``, all you need to
do to is to update the import to ``torchvision.transforms.v2``. In terms of
output, there might be negligible differences due to implementation differences.

.. _transforms_perf:

Performance considerations
--------------------------

We recommend the following guidelines to get the best performance out of the
transforms:

- Rely on the v2 transforms from ``torchvision.transforms.v2``
- Use tensors instead of PIL images
- Use ``torch.uint8`` dtype, especially for resizing
- Resize with bilinear or bicubic mode

This is what a typical transform pipeline could look like:

.. code:: python

    from torchvision.transforms import v2
    transforms = v2.Compose([
        v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
        v2.ToDtype(torch.uint8, scale=True),  # optional, most input are already uint8 at this point
        # ...
        v2.RandomResizedCrop(size=(224, 224), antialias=True),  # Or Resize(antialias=True)
        # ...
        v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

The above should give you the best performance in a typical training environment
that relies on the :class:`torch.utils.data.DataLoader` with ``num_workers >
0``.

Transforms tend to be sensitive to the input strides / memory format. Some
transforms will be faster with channels-first images while others prefer
channels-last. Like ``torch`` operators, most transforms will preserve the
memory format of the input, but this may not always be respected due to
implementation details. You may want to experiment a bit if you're chasing the
very best performance.  Using :func:`torch.compile` on individual transforms may
also help factoring out the memory format variable (e.g. on
:class:`~torchvision.transforms.v2.Normalize`). Note that we're talking about
**memory format**, not :ref:`tensor shape <conventions>`.

Note that resize transforms like :class:`~torchvision.transforms.v2.Resize`
and :class:`~torchvision.transforms.v2.RandomResizedCrop` typically prefer
channels-last input and tend **not** to benefit from :func:`torch.compile` at
this time.

.. _functional_transforms:

Transform classes, functionals, and kernels
-------------------------------------------

Transforms are available as classes like
:class:`~torchvision.transforms.v2.Resize`, but also as functionals like
:func:`~torchvision.transforms.v2.functional.resize` in the
``torchvision.transforms.v2.functional`` namespace.
This is very much like the :mod:`torch.nn` package which defines both classes
and functional equivalents in :mod:`torch.nn.functional`.

The functionals support PIL images, pure tensors, or :ref:`TVTensors
<tv_tensors>`, e.g. both ``resize(image_tensor)`` and ``resize(boxes)`` are
valid.

.. note::

    Random transforms like :class:`~torchvision.transforms.v2.RandomCrop` will
    randomly sample some parameter each time they're called. Their functional
    counterpart (:func:`~torchvision.transforms.v2.functional.crop`) does not do
    any kind of random sampling and thus have a slighlty different
    parametrization. The ``get_params()`` class method of the transforms class
    can be used to perform parameter sampling when using the functional APIs.


The ``torchvision.transforms.v2.functional`` namespace also contains what we
call the "kernels". These are the low-level functions that implement the
core functionalities for specific types, e.g. ``resize_bounding_boxes`` or
```resized_crop_mask``. They are public, although not documented. Check the
`code
<https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/functional/__init__.py>`_
to see which ones are available (note that those starting with a leading
underscore are **not** public!). Kernels are only really useful if you want
:ref:`torchscript support <transforms_torchscript>` for types like bounding
boxes or masks.

.. _transforms_torchscript:

Torchscript support
-------------------

Most transform classes and functionals support torchscript. For composing
transforms, use :class:`torch.nn.Sequential` instead of
:class:`~torchvision.transforms.v2.Compose`:

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

    If you really need torchscript support for the v2 transforms, we recommend
    scripting the **functionals** from the
    ``torchvision.transforms.v2.functional`` namespace to avoid surprises.


Also note that the functionals only support torchscript for pure tensors, which
are always treated as images. If you need torchscript support for other types
like bounding boxes or masks, you can rely on the :ref:`low-level kernels
<functional_transforms>`.

For any custom transformations to be used with ``torch.jit.script``, they should
be derived from ``torch.nn.Module``.

See also: :ref:`sphx_glr_auto_examples_others_plot_scripted_tensor_transforms.py`.

.. _v2_api_ref:

V2 API reference - Recommended
------------------------------

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
    255] range into [0, 1] (and vice-versa). See :ref:`range_and_dtype`.

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
:ref:`sphx_glr_auto_examples_transforms_plot_cutmix_mixup.py` for detailed usage examples.

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


V1 API Reference
----------------

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
    255] range into [0, 1] (and vice-versa). See :ref:`range_and_dtype`.
    
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



Functional Transforms
^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: torchvision.transforms.functional

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
