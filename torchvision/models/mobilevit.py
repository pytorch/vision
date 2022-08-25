# TODO: Implement v1 and v2 versions of the mobile ViT model.

from torch import nn
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._utils import _ovewrite_named_param
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torch import Tensor
from typing import Callable, Optional, Any, List
from ..transforms._presets import ImageClassification
from functools import partial

__all__ = ["MobileViT", "MobileViT_Weights", "MobileViT_V2_Weights"]

# TODO: Is this correct? Maybe not? Need to check the training script...
_COMMON_META = {
    "categories": _IMAGENET_CATEGORIES,
}

# TODO: Update this...
# Paper links: v1 https://arxiv.org/abs/2110.02178
# v2 (what the difference with the V1 paper?)
# TODO: Need a mobile ViT block...
# TODO: Adding weights... Start with V1.
# Things to be done: write the V1, mobileViTblock, weights, documentation...

class MobileViT_Weights(WeightsEnum):
    # TODO: Update these...
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/mobilevit.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#mobilenetv2",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 71.878,
                    "acc@5": 90.286,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    # TODO: Will be updated later...
    IMAGENET1K_V2 = Weights(
        url="https://download.pytorch.org/models/mobilevit.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe-with-reg-tuning",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 72.154,
                    "acc@5": 90.822,
                }
            },
            "_docs": """
                These weights improve upon the results of the original paper by using a modified version of TorchVision's
                `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V2


class MobileViT_V2_Weights(WeightsEnum):
    pass



class MobileViTBlock(nn.Module):
    def forward(self, x: Tensor):
        return x

class MobileViTV2Block(MobileViTBlock):
    def forward(self, x: Tensor):
        return x

class MobileViT(nn.Module):
    """
    Implements MobileViT from the `"MobileViT: Light-Weight, General-Purpose, and Mobile-Friendly Vision Transfomer" <https://arxiv.org/abs/2110.02178>`_ paper.
    Args:
        TODO: Arguments to be updated...
    """

    def __init__(
        self,
        num_classes: int,
        # TODO: Should this be optional?
        block: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()
        _log_api_usage_once(self)
        # TODO: Add blocks... In progress...
        self.num_classes = num_classes

        if block is None:
            block = MobileViTBlock

    # TODO: This is the core thing to implement...
    def forward(self, x):
        return x


def _mobile_vit(
    # TODO: Update the parameters...
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> MobileViT:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = MobileViT(
        # TODO: Update these...Will pass different configurations depending on the size of the mdoel...
        **kwargs,
    )

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


@register_model()
def mobile_vit_s():
    pass


@register_model()
def mobile_vit_s():
    pass

@register_model()
def mobile_vit_s():
    pass