import os
from collections import OrderedDict

from torch import nn

from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, TwoSidesFeaturePyramidNetwork, \
    LastLevelMaxPool
from .. import resnet
from ..model_utils import IntermediateLayerGetter, IntermediateLayerGetterLateFusionSummation, \
    IntermediateLayerGetterLateFusionConcat, IntermediateLayerGetterLateAttentionFusion

input_types = {"RGB", "RGBD", "Depth", "Combined"}
fusion_types = {"Sum", "Concat", "Attention"}


class BackboneWithFPNGivenBody(nn.Sequential):
    """
    Adds a FPN on top of a model.

    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediatLayerGetter apply here.

    Arguments:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN.

    Attributes:
        out_channels (int): the number of channels in the FPN
    """

    def __init__(self, body, in_channels_list, out_channels, two_sides_fpn=False):

        if two_sides_fpn:
            fpn = TwoSidesFeaturePyramidNetwork(
                in_channels_list=in_channels_list,
                out_channels=out_channels,
                extra_blocks=LastLevelMaxPool(),
            )
        else:
            fpn = FeaturePyramidNetwork(
                in_channels_list=in_channels_list,
                out_channels=out_channels,
                extra_blocks=LastLevelMaxPool(),
            )
        super(BackboneWithFPNGivenBody, self).__init__(OrderedDict(
            [("body", body), ("fpn", fpn)]))
        self.out_channels = out_channels


class BackboneWithFPN(BackboneWithFPNGivenBody):
    """
    Adds a FPN on top of a model.

    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediatLayerGetter apply here.

    Arguments:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN.

    Attributes:
        out_channels (int): the number of channels in the FPN
    """

    def __init__(self, backbone, return_layers, in_channels_list, out_channels, two_sides_fpn=False):
        super().__init__(IntermediateLayerGetter(backbone, return_layers=return_layers), in_channels_list, out_channels,
                         two_sides_fpn)


class BackboneWithLateFusionFPN(BackboneWithFPNGivenBody):
    """
    Adds a FPN on top of a model.

    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediatLayerGetter apply here.

    Arguments:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN.

    Attributes:
        out_channels (int): the number of channels in the FPN
    """

    def __init__(self, backbone1, backbone2, return_layers, in_channels_list, out_channels, two_sides_fpn=False,
                 fusion_type="concat"):
        global fusion_types
        assert fusion_type in fusion_types, f"Invalid fusion type {fusion_type}. Valid fusion types: {fusion_types}"

        body = None
        if fusion_type == "Concat":
            body = IntermediateLayerGetterLateFusionConcat(backbone1, backbone2, return_layers=return_layers)

        elif fusion_type == "Sum":
            body = IntermediateLayerGetterLateFusionSummation(backbone1, backbone2, return_layers=return_layers)

        elif fusion_type == "Attention":
            body = IntermediateLayerGetterLateAttentionFusion(backbone1, backbone2, return_layers, in_channels_list)

        super().__init__(body, in_channels_list, out_channels, two_sides_fpn)


class ResnetFPNNamespace:

    @staticmethod
    def assert_single_backbone_params(backbone_name, pretrained, backbone_names=resnet.__all__):
        assert isinstance(backbone_name, str) and backbone_name in backbone_names, \
            f"Invalid argument backbone_name={backbone_name}. Valid backbone names are {backbone_names}"
        assert isinstance(pretrained, str) and os.path.exists(str(pretrained)) or isinstance(pretrained, bool), \
            f"Invalid argument pretrained={pretrained}. 'pretrained' should be a valid path or a bool flag"

    @staticmethod
    def assert_general_backbone_params(num_of_layers_in_pyramid, two_sides_fpn):
        assert isinstance(num_of_layers_in_pyramid, int), \
            f"Invalid argument num_of_layers_in_pyramid={num_of_layers_in_pyramid}. 'num_of_layers_in_pyramid' should be an integer"
        assert isinstance(two_sides_fpn, bool), \
            f"Invalid argument two_sides_fpn={two_sides_fpn}. 'two_sides_fpn' should be a boolean"

    @staticmethod
    def resnet_fpn_backbone(backbone1=None, backbone2=None, fusion_type=None, num_of_layers_in_pyramid=4,
                            two_sides_fpn=False):
        assert backbone1 or backbone2, "Invalid arguments. 'resnet_fpn_backbone' should get at least one backbone " \
                                       "argument backbone1={backbone1}, backbone2={backbone2}"

        return_layers = {'layer1': 0, 'layer2': 1, 'layer3': 2, 'layer4': 3}
        return_layers = {k: v for k, v in zip(list(return_layers.keys())[:num_of_layers_in_pyramid],
                                              list(return_layers.values())[:num_of_layers_in_pyramid])}

        fpn_in_channels = backbone1.inplanes // (4 if fusion_type == "Concat" else 8)
        in_channels_list = [
            fpn_in_channels,
            fpn_in_channels * 2,
            fpn_in_channels * 4,
            fpn_in_channels * 8,
        ]
        out_channels = 256

        if backbone1 and backbone2 and fusion_type:
            return BackboneWithLateFusionFPN(
                backbone1,
                backbone2,
                return_layers,
                in_channels_list[:num_of_layers_in_pyramid],
                out_channels,
                two_sides_fpn,
                fusion_type
            )

        return BackboneWithFPN(
            backbone1 if backbone1 else backbone2,
            return_layers,
            in_channels_list[:num_of_layers_in_pyramid],
            out_channels,
            two_sides_fpn
        )

    class SingleBackboneNamespace:

        @staticmethod
        def assert_backbone_params(backbone_name, pretrained, num_of_layers_in_pyramid, two_sides_fpn,
                                   valid_backbone_names):
            ResnetFPNNamespace.assert_single_backbone_params(backbone_name, pretrained, valid_backbone_names)
            ResnetFPNNamespace.assert_general_backbone_params(num_of_layers_in_pyramid, two_sides_fpn)

        @staticmethod
        def resnet_fpn_single_input_backbone(backbone_name="resnet50", pretrained=False, num_of_layers_in_pyramid=4,
                                             two_sides_fpn=False, valid_backbone_names=resnet.__all__):
            ResnetFPNNamespace.SingleBackboneNamespace.assert_backbone_params(backbone_name, pretrained,
                                                                              num_of_layers_in_pyramid, two_sides_fpn,
                                                                              valid_backbone_names)

            backbone = resnet.__dict__[backbone_name](
                pretrained=pretrained,
                norm_layer=misc_nn_ops.FrozenBatchNorm2d
            )

            return ResnetFPNNamespace.resnet_fpn_backbone(
                backbone1=backbone,
                num_of_layers_in_pyramid=num_of_layers_in_pyramid,
                two_sides_fpn=two_sides_fpn
            )

        @staticmethod
        def resnet_rgb_fpn_backbone(backbone_name="resnet50", pretrained=False, num_of_layers_in_pyramid=4,
                                    two_sides_fpn=False):
            print("Creating RGB resnet FPN backbone")
            return ResnetFPNNamespace.SingleBackboneNamespace.resnet_fpn_single_input_backbone(
                backbone_name=backbone_name,
                pretrained=pretrained,
                num_of_layers_in_pyramid=num_of_layers_in_pyramid,
                two_sides_fpn=two_sides_fpn,
                valid_backbone_names=resnet.model_names["RGB"]
            )

        @staticmethod
        def resnet_rgbd_fpn_backbone(backbone_name="resnet50_rgbd", pretrained=False, num_of_layers_in_pyramid=4,
                                     two_sides_fpn=False):
            print("Creating RGBD resnet FPN backbone")
            assert backbone_name in resnet.model_names["RGBD"], f"Invalid backbone name for function" \
                                                                f" {ResnetFPNNamespace.SingleBackboneNamespace.resnet_rgbd_fpn_backbone.__name__}." \
                                                                f" Valid names: {resnet.model_names['RGBD']}"
            return ResnetFPNNamespace.SingleBackboneNamespace.resnet_fpn_single_input_backbone(
                backbone_name=backbone_name,
                pretrained=pretrained,
                num_of_layers_in_pyramid=num_of_layers_in_pyramid,
                two_sides_fpn=two_sides_fpn,
                valid_backbone_names=resnet.model_names["RGBD"]
            )

        @staticmethod
        def resnet_depth_fpn_backbone(backbone_name="resnet50_depth", pretrained=False, num_of_layers_in_pyramid=4,
                                      two_sides_fpn=False):
            print("Creating Depth resnet FPN backbone")
            assert backbone_name in resnet.model_names["Depth"], f"Invalid backbone name for function" \
                                                                 f" {ResnetFPNNamespace.SingleBackboneNamespace.resnet_depth_fpn_backbone.__name__}." \
                                                                 f" Valid names: {resnet.model_names['Depth']}"
            return ResnetFPNNamespace.SingleBackboneNamespace.resnet_fpn_single_input_backbone(
                backbone_name=backbone_name,
                pretrained=pretrained,
                num_of_layers_in_pyramid=num_of_layers_in_pyramid,
                two_sides_fpn=two_sides_fpn,
                valid_backbone_names=resnet.model_names["Depth"]
            )

    class DoubleBackboneNamespace:

        @staticmethod
        def assert_backbone_params(rgb_backbone_params, depth_backbone_params, num_of_layers_in_pyramid, two_sides_fpn,
                                   fusion_type):
            global fusion_types
            ResnetFPNNamespace.assert_single_backbone_params(rgb_backbone_params["name"],
                                                             rgb_backbone_params["pretrained"],
                                                             resnet.model_names["RGB"])
            ResnetFPNNamespace.assert_single_backbone_params(depth_backbone_params["name"],
                                                             depth_backbone_params["pretrained"],
                                                             resnet.model_names["Depth"])
            ResnetFPNNamespace.assert_general_backbone_params(num_of_layers_in_pyramid, two_sides_fpn)
            assert fusion_type in fusion_types, f"Invalid fusion_type={fusion_type}, Valid fusion types: {fusion_types}"

        @staticmethod
        def resnet_late_fusion_fpn_backbone(rgb_backbone_params, depth_backbone_params, num_of_layers_in_pyramid=4,
                                            two_sides_fpn=False, fusion_type="Sum"):
            ResnetFPNNamespace.DoubleBackboneNamespace.assert_backbone_params(rgb_backbone_params,
                                                                              depth_backbone_params,
                                                                              num_of_layers_in_pyramid, two_sides_fpn,
                                                                              fusion_type)
            print(f"Creating late fusion resnet FPN backbone with '{fusion_type}' fusion type")

            rgb_backbone = resnet.__dict__[rgb_backbone_params["name"]](
                pretrained=rgb_backbone_params["pretrained"],
                norm_layer=misc_nn_ops.FrozenBatchNorm2d
            )

            depth_backbone = resnet.__dict__[depth_backbone_params["name"]](
                pretrained=depth_backbone_params["pretrained"],
                norm_layer=misc_nn_ops.FrozenBatchNorm2d
            )

            return ResnetFPNNamespace.resnet_fpn_backbone(
                backbone1=rgb_backbone,
                backbone2=depth_backbone,
                fusion_type=fusion_type,
                num_of_layers_in_pyramid=num_of_layers_in_pyramid,
                two_sides_fpn=two_sides_fpn
            )

        @staticmethod
        def resnet_sum_fusion_fpn_backbone(rgb_backbone_params, depth_backbone_params, num_of_layers_in_pyramid=4,
                                           two_sides_fpn=False):
            return ResnetFPNNamespace.DoubleBackboneNamespace.resnet_late_fusion_fpn_backbone(
                rgb_backbone_params=rgb_backbone_params,
                depth_backbone_params=depth_backbone_params,
                num_of_layers_in_pyramid=num_of_layers_in_pyramid,
                two_sides_fpn=two_sides_fpn,
                fusion_type="Sum"
            )

        @staticmethod
        def resnet_concat_fusion_fpn_backbone(rgb_backbone_params, depth_backbone_params, num_of_layers_in_pyramid=4,
                                              two_sides_fpn=False):
            return ResnetFPNNamespace.DoubleBackboneNamespace.resnet_late_fusion_fpn_backbone(
                rgb_backbone_params=rgb_backbone_params,
                depth_backbone_params=depth_backbone_params,
                num_of_layers_in_pyramid=num_of_layers_in_pyramid,
                two_sides_fpn=two_sides_fpn,
                fusion_type="Concat"
            )

        @staticmethod
        def resnet_attention_fusion_fpn_backbone(rgb_backbone_params, depth_backbone_params, num_of_layers_in_pyramid=4,
                                                 two_sides_fpn=False):
            return ResnetFPNNamespace.DoubleBackboneNamespace.resnet_late_fusion_fpn_backbone(
                rgb_backbone_params=rgb_backbone_params,
                depth_backbone_params=depth_backbone_params,
                num_of_layers_in_pyramid=num_of_layers_in_pyramid,
                two_sides_fpn=two_sides_fpn,
                fusion_type="Attention"
            )


fpn_registry = {
    "RGB": ResnetFPNNamespace.SingleBackboneNamespace.resnet_rgb_fpn_backbone,
    "Depth": ResnetFPNNamespace.SingleBackboneNamespace.resnet_depth_fpn_backbone,
    "RGBD": ResnetFPNNamespace.SingleBackboneNamespace.resnet_rgbd_fpn_backbone,
    "Combined": {
        "Sum": ResnetFPNNamespace.DoubleBackboneNamespace.resnet_sum_fusion_fpn_backbone,
        "Concat": ResnetFPNNamespace.DoubleBackboneNamespace.resnet_concat_fusion_fpn_backbone,
        "Attention": ResnetFPNNamespace.DoubleBackboneNamespace.resnet_attention_fusion_fpn_backbone,
    }
}


def fpn_factory(backbone_params, input_type, fusion_type):
    global fpn_registry
    assert input_type in input_types, f"Invalid input type {input_type}. Valid input types: {input_types}"
    if input_type == "RGB":
        return fpn_registry["RGB"](**backbone_params)
    elif input_type == "RGBD":
        return fpn_registry["RGBD"](**backbone_params)
    elif input_type == "Depth":
        return fpn_registry["Depth"](**backbone_params)
    elif input_type == "Combined":
        return fpn_registry["Combined"][fusion_type](**backbone_params)
