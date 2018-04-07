torchvision.transforms
======================

.. currentmodule:: torchvision.transforms

Transforms are common image transforms. They can be chained together using :class:`Compose`

.. autoclass:: Compose

Transforms on PIL Image
-----------------------

.. autoclass:: CenterCrop

.. autoclass:: ColorJitter

.. autoclass:: FiveCrop

.. autoclass:: Grayscale

.. autoclass:: LinearTransformation

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

