torchvision.transforms
======================

.. currentmodule:: torchvision.transforms

Transforms are common image transformations. They can be chained together using :class:`Compose`.
Additionally, there is the :mod:`torchvision.transforms.functional` module.
Functional transforms give fine-grained control over the transformations.
This is useful if you have to build a more complex transformation pipeline
(e.g. in the case of segmentation tasks).

All transformations accept PIL Image, Tensor Image or batch of Tensor Images as input. Tensor Image is a tensor with
``(C, H, W)`` shape, where ``C`` is a number of channels, ``H`` and ``W`` are image height and width. Batch of
Tensor Images is a tensor of ``(B, C, H, W)`` shape, where ``B`` is a number of images in the batch. Deterministic or
random transformations applied on the batch of Tensor Images identically transform all the images of the batch.


Scriptable transforms
^^^^^^^^^^^^^^^^^^^^^

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

.. autoclass:: RandomPerspective

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
