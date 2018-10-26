'''
This is an example hubconf.py for pytorch/vision repo

## Users can get this published model by calling:
hub_model = hub.load(
    'pytorch/vision:master', # repo_owner/repo_name:branch
    'resnet18', # entrypoint
    1234, # args for callable [not applicable to resnet]
    pretrained=True) # kwargs for callable

## Protocol on repo owner side
1. The "published" models should be at least in a branch/tag. It can't be a random commit.
2. Repo owner should have the following field defined in hubconf.py
  2.1 Function/entrypoint with function signature "def resnet18(pretrained=False, *args, **kwargs):"
  2.2 Pretrained allows users to load pretrained weights from repo owner.
  2.3 Args and kwargs are passed to the callable _resnet18,
  2.4 Docstring of function works as a help message, explaining what does the model do and what's
      the allowed arguments.
  2.5 Dependencies is a list optionally provided by repo owner, to specify what packages are required
      to run the model.

## Hub_dir

hub_dir specifies where the intermediate files/folders will be saved. By default this is ~/.torch/hub.
Users can change it by either setting the environment variable TORCH_HUB_DIR or calling hub.set_dir(PATH_TO_HUB_DIR).
By default, we don't cleanup files after loading so that users can use cache next time.

## Cache logic

We used the cache by default if it exists in hub_dir.
Users can force a fresh reload by calling hub.load(..., force_reload=True).
'''

import torch.utils.model_zoo as model_zoo

# Optional list of dependencies required by the package
dependencies = ['torch', 'math']


def resnet18(pretrained=False, *args, **kwargs):
    """
    Resnet18 model
    pretrained (bool): a recommended kwargs for all entrypoints
    args & kwargs are arguments for the function
    """
    from torchvision.models.resnet import resnet18 as _resnet18
    model = _resnet18(*args, **kwargs)
    checkpoint = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
    if pretrained:
        model.load_state_dict(model_zoo.load_url(checkpoint, progress=False))
    return model


def resnet50(pretrained=False, *args, **kwargs):
    """
    Resnet50 model
    pretrained (bool): a recommended kwargs for all entrypoints
    args & kwargs are arguments for the function
    """
    from torchvision.models.resnet import resnet50 as _resnet50
    model = _resnet50(*args, **kwargs)
    checkpoint = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
    if pretrained:
        model.load_state_dict(model_zoo.load_url(checkpoint, progress=False))
    return model
