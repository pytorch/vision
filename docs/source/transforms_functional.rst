torchvision.transforms.functional
=================================

Functional transforms give you fine-grained control of the transformation
pipeline.
As opposed to the stateful transformations from :mod:`torchvision.transforms`
functional transforms don't contain any state or random number generator for
their parameters.
That means that you have to specify/generate all parameters,
but you can reuse a functional transform.

Example:
you can apply a functional transform with the same parameters to multiple
images like this:

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
