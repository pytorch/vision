import torch
from torch import nn


def resize_network_output(net, output_size):
    if isinstance(net, torch.nn.DataParallel):
        return resize_network_output(net.module, output_size)

    output_layer = net._modules.keys()[-1]
    old_output_layer = net._modules[output_layer]

    if isinstance(old_output_layer, nn.Sequential):
        return resize_network_output(old_output_layer, output_size)
    elif isinstance(old_output_layer, nn.modules.pooling.AvgPool2d):
        # Go back in the layer sequence and find the last conv layer and resize that
        # Only happens for squeezenet1_0
        for name, layer in list(net._modules.iteritems())[::-1][1:]:
            if isinstance(layer, nn.modules.conv.Conv2d):
                net._modules[name] = nn.modules.conv.Conv2d(layer.in_channels, output_size, layer.kernel_size,
                                                            layer.stride, layer.padding, layer.dilation, layer.groups)
                return
        assert False

    assert isinstance(old_output_layer, nn.Linear), 'Class of old_output_layer {}'.format(old_output_layer.__class__.__name__)
    input_size = old_output_layer.weight.size()[1]

    net._modules[output_layer] = nn.Linear(input_size, output_size)
