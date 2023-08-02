"""
=====================================
How to write your own Datapoint class
=====================================

This guide is intended for downstream library maintainers. We explain how to
write your own datapoint class, and how to make it compatible with the built-in
Torchvision V2 transforms. Before continuing, make sure you have read
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
#
# We will just create a very simple class that just inherits from the base
# :class:`~torchvision.datapoints.Datapoint` class. It will be enough to cover
# what you need to know to implement your own custom uses-cases. If you need to
# create a class that carries meta-data, take a look at how the
# :class:`~torchvision.datapoints.BoundingBoxes` class is implemented.

class MyDatapoint(datapoints.Datapoint):
    pass

my_dp = MyDatapoint([1, 2, 3])
my_dp

#%%
from torchvision.transforms.v2.functional import register_kernel, resize

# TODO: THIS didn't raise an error:
# @register_kernel(MyDatapoint, resize)

# TODO Let devs pass strings

@register_kernel(resize, MyDatapoint)
def resize_my_datapoint(my_dp, size, *args, **kwargs):
    print(f"Resizing {my_dp} to {size}")
    return torch.rand(3, *size)



my_dp = MyDatapoint(torch.rand(3, 256, 256))
out = v2.Resize((224, 224))(my_dp)
print(type(out), out.shape)
# %%
