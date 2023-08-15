"""
=====================================
How to write your own Datapoint class
=====================================

This guide is intended for downstream library maintainers. We explain how to
write your own datapoint class, and how to make it compatible with the built-in
Torchvision v2 transforms. Before continuing, make sure you have read
:ref:`sphx_glr_auto_examples_plot_datapoints.py`.
"""

# %%
import torch
import torchvision

# We are using BETA APIs, so we deactivate the associated warning, thereby acknowledging that
# some APIs may slightly change in the future
torchvision.disable_beta_transforms_warning()

from torchvision import datapoints
from torchvision.transforms import v2

# %%
# We will create a very simple class that just inherits from the base
# :class:`~torchvision.datapoints.Datapoint` class. It will be enough to cover
# what you need to know to implement your more elaborate uses-cases. If you need
# to create a class that carries meta-data, take a look at how the
# :class:`~torchvision.datapoints.BoundingBoxes` class is `implemented
# <https://github.com/pytorch/vision/blob/main/torchvision/datapoints/_bounding_box.py>`_.


class MyDatapoint(datapoints.Datapoint):
    pass


my_dp = MyDatapoint([1, 2, 3])
my_dp

# %%
# Now that we have defined our custom Datapoint class, we want it to be
# compatible with the built-in torchvision transforms, and the functional API.
# For that, we need to implement a kernel which performs the core of the
# transformation, and then "hook" it to the functional that we want to support
# via :func:`~torchvision.transforms.v2.functional.register_kernel`.
#
# We illustrate this process below: we create a kernel for the "horizontal flip"
# operation of our MyDatapoint class, and register it to the functional API.

from torchvision.transforms.v2 import functional as F


@F.register_kernel(functional="hflip", datapoint_cls=MyDatapoint)
def hflip_my_datapoint(my_dp, *args, **kwargs):
    print("Flipping!")
    out = my_dp.flip(-1)
    return MyDatapoint.wrap_like(my_dp, out)


# %%
# To understand why ``wrap_like`` is used, see
# :ref:`datapoint_unwrapping_behaviour`. Ignore the ``*args, **kwargs`` for now,
# we will explain it below in :ref:`param_forwarding`.
#
# .. note::
#
#     In our call to ``register_kernel`` above we used a string
#     ``functional="hflip"`` to refer to the functional we want to hook into. We
#     could also have used the  functional *itself*, i.e.
#     ``@register_kernel(functional=F.hflip, ...)``.
#
#     The functionals that you can be hooked into are the ones in
#     ``torchvision.transforms.v2.functional`` and they are documented in
#     :ref:`functional_transforms`.
#
# Now that we have registered our kernel, we can call the functional API on a
# ``MyDatapoint`` instance:

my_dp = MyDatapoint(torch.rand(3, 256, 256))
_ = F.hflip(my_dp)

# %%
# And we can also use the
# :class:`~torchvision.transforms.v2.RandomHorizontalFlip` transform, since it relies on :func:`~torchvision.transforms.v2.functional.hflip` internally:
t = v2.RandomHorizontalFlip(p=1)
_ = t(my_dp)

# %%
# .. note::
#
#     We cannot register a kernel for a transform class, we can only register a
#     kernel for a **functional**. The reason we can't register a transform
#     class is because one transform may internally rely on more than one
#     functional, so in general we can't register a single kernel for a given
#     class.
#
# .. _param_forwarding:
#
# Parameter forwarding, and ensuring future compatibility of your kernels
# -----------------------------------------------------------------------
#
# The functional API that you're hooking into is public and therefore
# **backward** compatible: we guarantee that the parameters of these functionals
# won't be removed or renamed without a proper deprecation cycle. However, we
# don't guarantee **forward** compatibility, and we may add new parameters in
# the future.
#
# Imagine that in a future version, Torchvision adds a new ``inplace`` parameter
# to its :func:`~torchvision.transforms.v2.functional.hflip` functional. If you
# already defined and registered your own kernel as

def hflip_my_datapoint(my_dp):  # noqa
    print("Flipping!")
    out = my_dp.flip(-1)
    return MyDatapoint.wrap_like(my_dp, out)


# %%
# then calling ``F.hflip(my_dp)`` will **fail**, because ``hflip`` will try to
# pass the new ``inplace`` parameter to your kernel, but your kernel doesn't
# accept it.
#
# For this reason, we recommend to always define your kernels with
# ``*args, **kwargs`` in their signature, as done above. This way, your kernel
# will be able to accept any new parameter that we may add in the future.
# (Technically, adding `**kwargs` only should be enough).
