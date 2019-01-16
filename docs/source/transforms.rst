torchvision.transforms
======================

.. currentmodule:: torchvision.transforms

Transforms are common image transformations. They can be chained together using :class:`Compose`.
Additionally, there is the :mod:`torchvision.transforms.functional` module.
Functional transforms give fine-grained control over the transformations.
This is useful if you have to build a more complex transformation pipeline
(e.g. in the case of segmentation tasks).

.. autoclass:: Compose

Transforms on PIL Image
-----------------------

.. autoclass:: CenterCrop

.. autoclass:: ColorJitter

.. autoclass:: FiveCrop

.. autoclass:: Grayscale

.. autoclass:: Pad

.. autoclass:: RandomAffine

.. autoclass:: RandomApply

.. autoclass:: RandomChoice

.. autoclass:: RandomCrop

.. autoclass:: RandomGrayscale

.. autoclass:: RandomHorizontalFlip

.. autoclass:: RandomOrder

.. autoclass:: RandomResizedCrop

.. autoclass:: RandomRotation

.. autoclass:: RandomSizedCrop

.. autoclass:: RandomVerticalFlip

.. autoclass:: Resize

.. autoclass:: Scale

.. autoclass:: TenCrop

Transforms on torch.\*Tensor
----------------------------

.. autoclass:: LinearTransformation

.. autoclass:: Normalize
	:members: __call__
	:special-members:

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
For example, you can apply a functional transform to multiple images like this:

.. code:: python

    import torchvision.transforms.functional as TF
    import random

    def my_segmentation_transforms(image, segmentation):
        if random.random() > 5:
            angle = random.randint(-30, 30)
            image = TF.rotate(image, angle)
            segmentation = TF.rotate(segmentation, angle)
        # more transforms ...
        return image, segmentation

.. automodule:: torchvision.transforms.functional
    :members:
