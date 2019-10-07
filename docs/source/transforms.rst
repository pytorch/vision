torchvision.transforms
======================

.. currentmodule:: torchvision.transforms

Transforms are common image transformations. They can be chained together using :class:`Compose`.
Additionally, there is the :mod:`torchvision.transforms.functional` module.
Functional transforms give fine-grained control over the transformations.
This is useful if you have to build a more complex transformation pipeline
(e.g. in the case of segmentation tasks).

.. Note::
    Most transform classese have an equivalent in :mod:`torchvision.transforms.functional`.


.. autoclass:: Compose

Transforms on PIL Image
-----------------------

.. autoclass:: CenterCrop

**Example**

.. plot::
   :include-source:

    from torchvision.transforms import CenterCrop
    from torchvision.utils import _plot_images, _sample_image

    _plot_images(
        _sample_image(),
        CenterCrop(256)(_sample_image()),
        CenterCrop((200, 300))(_sample_image()),
    )


.. autoclass:: ColorJitter

**Example**

.. plot::
   :include-source:

    from torchvision.transforms import ColorJitter
    from torchvision.utils import _plot_images, _sample_image

    transform = ColorJitter(brightness=0.5)
    _plot_images(*[transform(_sample_image()) for _ in range(5)])

    transform = ColorJitter(contrast=0.5)
    _plot_images(*[transform(_sample_image()) for _ in range(5)])

    transform = ColorJitter(saturation=0.5)
    _plot_images(*[transform(_sample_image()) for _ in range(5)])

    transform = ColorJitter(hue=0.1)
    _plot_images(*[transform(_sample_image()) for _ in range(5)])


.. autoclass:: FiveCrop

**Example**

.. plot::
   :include-source:

    from torchvision.transforms import FiveCrop
    from torchvision.utils import _plot_images, _sample_image

    transform = FiveCrop(256)
    _plot_images(*transform(_sample_image()))


.. autoclass:: Grayscale

**Example**

.. plot::
   :include-source:

    from torchvision.transforms import Grayscale
    from torchvision.utils import _plot_images, _sample_image

    transform = Grayscale(3)
    _plot_images(transform(_sample_image()))


.. autoclass:: Pad

**Example**

.. plot::
   :include-source:

    from torchvision.transforms import Pad
    from torchvision.utils import _plot_images, _sample_image

    transform = Pad(100)
    _plot_images(
        Pad(100, padding_mode="constant", fill=0)(_sample_image()),
        Pad(100, padding_mode="constant", fill=(128, 128, 128))(_sample_image()),
        Pad(100, padding_mode="constant", fill=(255, 255, 255))(_sample_image()),
        Pad(100, padding_mode="edge")(_sample_image()),
        Pad(100, padding_mode="reflect")(_sample_image()),
        Pad(100, padding_mode="symmetric")(_sample_image()),
    )


.. autoclass:: RandomAffine

**Example**

.. plot::
   :include-source:

    from torchvision.transforms import RandomAffine
    from torchvision.utils import _plot_images, _sample_image

    transform = RandomAffine(degrees=30)
    _plot_images(*[transform(_sample_image()) for _ in range(5)])

    transform = RandomAffine(degrees=0, translate=(.4, .2))
    _plot_images(*[transform(_sample_image()) for _ in range(5)])

    transform = RandomAffine(degrees=0, scale=(.5, 1.5))
    _plot_images(*[transform(_sample_image()) for _ in range(5)])

    transform = RandomAffine(degrees=0, shear=30)
    _plot_images(*[transform(_sample_image()) for _ in range(5)])



.. autoclass:: RandomApply

.. autoclass:: RandomChoice

.. autoclass:: RandomCrop

**Example**

.. plot::
   :include-source:

    from torchvision.transforms import RandomCrop
    from torchvision.utils import _plot_images, _sample_image

    transform = RandomCrop(150)
    _plot_images(*[transform(_sample_image()) for _ in range(5)])


.. autoclass:: RandomGrayscale

**Example**

.. plot::
   :include-source:

    from torchvision.transforms import RandomGrayscale
    from torchvision.utils import _plot_images, _sample_image

    _plot_images(
        RandomGrayscale(p=1.0)(_sample_image()),
        RandomGrayscale(p=0.0)(_sample_image()),
    )


.. autoclass:: RandomHorizontalFlip

**Example**

.. plot::
   :include-source:

    from torchvision.transforms import RandomHorizontalFlip
    from torchvision.utils import _plot_images, _sample_image

    _plot_images(
        RandomHorizontalFlip(p=1.0)(_sample_image()),
        RandomHorizontalFlip(p=0.0)(_sample_image()),
    )


.. autoclass:: RandomOrder

.. autoclass:: RandomPerspective

**Example**

.. plot::
   :include-source:

    from torchvision.transforms import RandomPerspective
    from torchvision.utils import _plot_images, _sample_image

    transform = RandomPerspective(p=1.0)
    _plot_images(*[transform(_sample_image()) for _ in range(5)])


.. autoclass:: RandomResizedCrop

**Example**

.. plot::
   :include-source:

    from torchvision.transforms import RandomResizedCrop
    from torchvision.utils import _plot_images, _sample_image

    transform = RandomResizedCrop(250)
    _plot_images(*[transform(_sample_image()) for _ in range(5)])


.. autoclass:: RandomRotation

**Example**

.. plot::
   :include-source:

    from torchvision.transforms import RandomRotation
    from torchvision.utils import _plot_images, _sample_image

    transform = RandomRotation(30)
    _plot_images(*[transform(_sample_image()) for _ in range(5)])


.. autoclass:: RandomSizedCrop


.. autoclass:: RandomVerticalFlip

**Example**

.. plot::
   :include-source:

    from torchvision.transforms import RandomVerticalFlip
    from torchvision.utils import _plot_images, _sample_image

    transform = RandomVerticalFlip()
    _plot_images(
        RandomVerticalFlip(p=0.0)(_sample_image()),
        RandomVerticalFlip(p=1.0)(_sample_image()),
    )


.. autoclass:: Resize

**Example**

.. plot::
   :include-source:

    from torchvision.transforms import Resize
    from torchvision.utils import _plot_images, _sample_image

    _plot_images(
        Resize(200)(_sample_image()),
        Resize((200, 200))(_sample_image()),
        Resize(32)(_sample_image()),
    )


.. autoclass:: Scale

.. autoclass:: TenCrop

**Example**

.. plot::
   :include-source:

    from torchvision.transforms import TenCrop
    from torchvision.utils import _plot_images, _sample_image

    _plot_images(*TenCrop(200)(_sample_image()))


Transforms on torch.\*Tensor
----------------------------

.. autoclass:: LinearTransformation

.. autoclass:: Normalize
	:members: __call__
	:special-members:

.. autoclass:: RandomErasing

Conversion Transforms
---------------------

.. autoclass:: ToPILImage
	:members: __call__
	:special-members:

.. autoclass:: ToTensor
	:members: __call__
	:special-members:

Generic Transforms
------------------

.. autoclass:: Lambda


Functional Transforms
---------------------

Functional transforms give you fine-grained control of the transformation pipeline.
As opposed to the transformations above, functional transforms don't contain a random number
generator for their parameters.
That means you have to specify/generate all parameters, but you can reuse the functional transform.

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
