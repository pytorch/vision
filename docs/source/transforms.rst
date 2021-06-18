.. _transforms:

torchvision.transforms
======================

.. currentmodule:: torchvision.transforms

Transforms are common image transformations. They can be chained together using :class:`Compose`.
Most transform classes have a function equivalent: :ref:`functional
transforms <functional_transforms>` give fine-grained control over the
transformations.
This is useful if you have to build a more complex transformation pipeline
(e.g. in the case of segmentation tasks).

Most transformations accept both `PIL <https://pillow.readthedocs.io>`_
images and tensor images, although some transformations are :ref:`PIL-only
<transforms_pil_only>` and some are :ref:`tensor-only
<transforms_tensor_only>`. The :ref:`conversion_transforms` may be used to
convert to and from PIL images.

The transformations that accept tensor images also accept batches of tensor
images. A Tensor Image is a tensor with ``(C, H, W)`` shape, where ``C`` is a
number of channels, ``H`` and ``W`` are image height and width. A batch of
Tensor Images is a tensor of ``(B, C, H, W)`` shape, where ``B`` is a number
of images in the batch.

The expected range of the values of a tensor image is implicitely defined by
the tensor dtype. Tensor images with a float dtype are expected to have
values in ``[0, 1)``. Tensor images with an integer dtype are expected to
have values in ``[0, MAX_DTYPE]`` where ``MAX_DTYPE`` is the largest value
that can be represented in that dtype.

Randomized transformations will apply the same transformation to all the
images of a given batch, but they will produce different transformations
across calls. For reproducible transformations across calls, you may use
:ref:`functional transforms <functional_transforms>`.

The following examples illustate the use of the available transforms:

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


Scriptable transforms
---------------------

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


Compositions of transforms
--------------------------

.. autoclass:: Compose


Transforms on PIL Image and torch.\*Tensor
------------------------------------------

.. autoclass:: CenterCrop
    :members:

.. autoclass:: ColorJitter
    :members:

.. autoclass:: FiveCrop
    :members:

.. autoclass:: Grayscale
    :members:

.. autoclass:: Pad
    :members:

.. autoclass:: RandomAffine
    :members:

.. autoclass:: RandomApply

.. autoclass:: RandomCrop
    :members:

.. autoclass:: RandomGrayscale
    :members:

.. autoclass:: RandomHorizontalFlip
    :members:

.. autoclass:: RandomPerspective
    :members:

.. autoclass:: RandomResizedCrop
    :members:

.. autoclass:: RandomRotation
    :members:

.. autoclass:: RandomSizedCrop
    :members:

.. autoclass:: RandomVerticalFlip
    :members:

.. autoclass:: Resize
    :members:

.. autoclass:: Scale
    :members:

.. autoclass:: TenCrop
    :members:

.. autoclass:: GaussianBlur
    :members:

.. autoclass:: RandomInvert
    :members:

.. autoclass:: RandomPosterize
    :members:

.. autoclass:: RandomSolarize
    :members:

.. autoclass:: RandomAdjustSharpness
    :members:

.. autoclass:: RandomAutocontrast
    :members:

.. autoclass:: RandomEqualize
    :members:

.. _transforms_pil_only:

Transforms on PIL Image only
----------------------------

.. autoclass:: RandomChoice

.. autoclass:: RandomOrder

.. _transforms_tensor_only:

Transforms on torch.\*Tensor only
---------------------------------

.. autoclass:: LinearTransformation
    :members:

.. autoclass:: Normalize
    :members:

.. autoclass:: RandomErasing
    :members:

.. autoclass:: ConvertImageDtype

.. _conversion_transforms:

Conversion Transforms
---------------------

.. autoclass:: ToPILImage
    :members:

.. autoclass:: ToTensor
    :members:


Generic Transforms
------------------

.. autoclass:: Lambda
    :members:


AutoAugment Transforms
----------------------

`AutoAugment <https://arxiv.org/pdf/1805.09501.pdf>`_ is a common Data Augmentation technique that can improve the accuracy of Image Classification models.
Though the data augmentation policies are directly linked to their trained dataset, empirical studies show that
ImageNet policies provide significant improvements when applied to other datasets.
In TorchVision we implemented 3 policies learned on the following datasets: ImageNet, CIFAR10 and SVHN.
The new transform can be used standalone or mixed-and-matched with existing transforms:

.. autoclass:: AutoAugmentPolicy
    :members:

.. autoclass:: AutoAugment
    :members:


.. _functional_transforms:

Functional Transforms
---------------------

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


.. automodule:: torchvision.transforms.functional
    :members:
