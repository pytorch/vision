
"""
===========================
How to use CutMix and MixUp
===========================

.. note::
    Try on `collab <https://colab.research.google.com/github/pytorch/vision/blob/gh-pages/main/_generated_ipynb_notebooks/plot_cutmix_mixup.ipynb>`_
    or :ref:`go to the end <sphx_glr_download_auto_examples_transforms_plot_cutmix_mixup.py>` to download the full example code.

:class:`~torchvision.transforms.v2.CutMix` and
:class:`~torchvision.transforms.v2.MixUp` are popular augmentation strategies
that can improve classification accuracy.

These transforms are slightly different from the rest of the Torchvision
transforms, because they expect
**batches** of samples as input, not individual images. In this example we'll
explain how to use them: after the ``DataLoader``, or as part of a collation
function.
"""

# %%
import torch
from torchvision.datasets import FakeData
from torchvision.transforms import v2


NUM_CLASSES = 100

# %%
# Pre-processing pipeline
# -----------------------
#
# We'll use a simple but typical image classification pipeline:

preproc = v2.Compose([
    v2.PILToTensor(),
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),  # to float32 in [0, 1]
    v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # typically from ImageNet
])

dataset = FakeData(size=1000, num_classes=NUM_CLASSES, transform=preproc)

img, label = dataset[0]
print(f"{type(img) = }, {img.dtype = }, {img.shape = }, {label = }")

# %%
#
# One important thing to note is that neither CutMix nor MixUp are part of this
# pre-processing pipeline. We'll add them a bit later once we define the
# DataLoader. Just as a refresher, this is what the DataLoader and training loop
# would look like if we weren't using CutMix or MixUp:

from torch.utils.data import DataLoader

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

for images, labels in dataloader:
    print(f"{images.shape = }, {labels.shape = }")
    print(labels.dtype)
    # <rest of the training loop here>
    break
# %%

# %%
# Where to use MixUp and CutMix
# -----------------------------
#
# After the DataLoader
# ^^^^^^^^^^^^^^^^^^^^
#
# Now let's add CutMix and MixUp. The simplest way to do this right after the
# DataLoader: the Dataloader has already batched the images and labels for us,
# and this is exactly what these transforms expect as input:

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

cutmix = v2.CutMix(num_classes=NUM_CLASSES)
mixup = v2.MixUp(num_classes=NUM_CLASSES)
cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

for images, labels in dataloader:
    print(f"Before CutMix/MixUp: {images.shape = }, {labels.shape = }")
    images, labels = cutmix_or_mixup(images, labels)
    print(f"After CutMix/MixUp: {images.shape = }, {labels.shape = }")

    # <rest of the training loop here>
    break
# %%
#
# Note how the labels were also transformed: we went from a batched label of
# shape (batch_size,) to a tensor of shape (batch_size, num_classes). The
# transformed labels can still be passed as-is to a loss function like
# :func:`torch.nn.functional.cross_entropy`.
#
# As part of the collation function
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Passing the transforms after the DataLoader is the simplest way to use CutMix
# and MixUp, but one disadvantage is that it does not take advantage of the
# DataLoader multi-processing. For that, we can pass those transforms as part of
# the collation function (refer to the `PyTorch docs
# <https://pytorch.org/docs/stable/data.html#dataloader-collate-fn>`_ to learn
# more about collation).

from torch.utils.data import default_collate


def collate_fn(batch):
    return cutmix_or_mixup(*default_collate(batch))


dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2, collate_fn=collate_fn)

for images, labels in dataloader:
    print(f"{images.shape = }, {labels.shape = }")
    # No need to call cutmix_or_mixup, it's already been called as part of the DataLoader!
    # <rest of the training loop here>
    break

# %%
# Non-standard input format
# -------------------------
#
# So far we've used a typical sample structure where we pass ``(images,
# labels)`` as inputs. MixUp and CutMix will magically work by default with most
# common sample structures: tuples where the second parameter is a tensor label,
# or dict with a "label[s]" key. Look at the documentation of the
# ``labels_getter`` parameter for more details.
#
# If your samples have a different structure, you can still use CutMix and MixUp
# by passing a callable to the ``labels_getter`` parameter. For example:

batch = {
    "imgs": torch.rand(4, 3, 224, 224),
    "target": {
        "classes": torch.randint(0, NUM_CLASSES, size=(4,)),
        "some_other_key": "this is going to be passed-through"
    }
}


def labels_getter(batch):
    return batch["target"]["classes"]


out = v2.CutMix(num_classes=NUM_CLASSES, labels_getter=labels_getter)(batch)
print(f"{out['imgs'].shape = }, {out['target']['classes'].shape = }")
