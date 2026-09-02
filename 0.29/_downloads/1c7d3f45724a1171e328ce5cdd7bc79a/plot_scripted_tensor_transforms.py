"""
===================
Torchscript support
===================

.. note::
    Try on `Colab <https://colab.research.google.com/github/pytorch/vision/blob/gh-pages/main/_generated_ipynb_notebooks/plot_scripted_tensor_transforms.ipynb>`_
    or :ref:`go to the end <sphx_glr_download_auto_examples_others_plot_scripted_tensor_transforms.py>` to download the full example code.

This example illustrates `torchscript
<https://pytorch.org/docs/stable/jit.html>`_ support of the torchvision
:ref:`transforms <transforms>` on Tensor images.
"""

# %%
from pathlib import Path

import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import torchvision.transforms as v1
from torchvision.io import decode_image

plt.rcParams["savefig.bbox"] = 'tight'
torch.manual_seed(1)

# If you're trying to run that on Colab, you can download the assets and the
# helpers from https://github.com/pytorch/vision/tree/main/gallery/
import sys
sys.path += ["../transforms"]
from helpers import plot
ASSETS_PATH = Path('../assets')


# %%
# Most transforms support torchscript. For composing transforms, we use
# :class:`torch.nn.Sequential` instead of
# :class:`~torchvision.transforms.v2.Compose`:

dog1 = decode_image(str(ASSETS_PATH / 'dog1.jpg'))
dog2 = decode_image(str(ASSETS_PATH / 'dog2.jpg'))

transforms = torch.nn.Sequential(
    v1.RandomCrop(224),
    v1.RandomHorizontalFlip(p=0.3),
)

scripted_transforms = torch.jit.script(transforms)

plot([dog1, scripted_transforms(dog1), dog2, scripted_transforms(dog2)])


# %%
# .. warning::
#
#     Above we have used transforms from the ``torchvision.transforms``
#     namespace, i.e. the "v1" transforms. The v2 transforms from the
#     ``torchvision.transforms.v2`` namespace are the :ref:`recommended
#     <v1_or_v2>` way to use transforms in your code.
#
#     The v2 transforms also support torchscript, but if you call
#     ``torch.jit.script()`` on a v2 **class** transform, you'll actually end up
#     with its (scripted) v1 equivalent.  This may lead to slightly different
#     results between the scripted and eager executions due to implementation
#     differences between v1 and v2.
#
#     If you really need torchscript support for the v2 transforms, **we
#     recommend scripting the functionals** from the
#     ``torchvision.transforms.v2.functional`` namespace to avoid surprises.
#
# Below we now show how to combine image transformations and a model forward
# pass, while using ``torch.jit.script`` to obtain a single scripted module.
#
# Let's define a ``Predictor`` module that transforms the input tensor and then
# applies an ImageNet model on it.

from torchvision.models import resnet18, ResNet18_Weights


class Predictor(nn.Module):

    def __init__(self):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        self.resnet18 = resnet18(weights=weights, progress=False).eval()
        self.transforms = weights.transforms(antialias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.transforms(x)
            y_pred = self.resnet18(x)
            return y_pred.argmax(dim=1)


# %%
# Now, let's define scripted and non-scripted instances of ``Predictor`` and
# apply it on multiple tensor images of the same size

device = "cuda" if torch.cuda.is_available() else "cpu"

predictor = Predictor().to(device)
scripted_predictor = torch.jit.script(predictor).to(device)

batch = torch.stack([dog1, dog2]).to(device)

res = predictor(batch)
res_scripted = scripted_predictor(batch)

# %%
# We can verify that the prediction of the scripted and non-scripted models are
# the same:

import json

with open(Path('../assets') / 'imagenet_class_index.json') as labels_file:
    labels = json.load(labels_file)

for i, (pred, pred_scripted) in enumerate(zip(res, res_scripted)):
    assert pred == pred_scripted
    print(f"Prediction for Dog {i + 1}: {labels[str(pred.item())]}")

# %%
# Since the model is scripted, it can be easily dumped on disk and re-used

import tempfile

with tempfile.NamedTemporaryFile() as f:
    scripted_predictor.save(f.name)

    dumped_scripted_predictor = torch.jit.load(f.name)
    res_scripted_dumped = dumped_scripted_predictor(batch)
assert (res_scripted_dumped == res_scripted).all()

# %%
