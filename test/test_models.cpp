#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include "../torchvision/csrc/models/models.h"

#include <pybind11/pybind11.h>

using namespace vision::models;

template <typename Model>
torch::Tensor test_model(torch::Tensor x, const std::string& input_path)
{
    Model network;
    torch::load(network, input_path);
    return network->forward(x);
}

torch::Tensor test_alexnet(torch::Tensor x)
{
    return test_model<AlexNet>(x, "alexnet.pt");
}

torch::Tensor test_vgg11(torch::Tensor x)
{
    return test_model<VGG11>(x, "vgg11.pt");
}
torch::Tensor test_vgg13(torch::Tensor x)
{
    return test_model<VGG13>(x, "vgg13.pt");
}
torch::Tensor test_vgg16(torch::Tensor x)
{
    return test_model<VGG16>(x, "vgg16.pt");
}
torch::Tensor test_vgg19(torch::Tensor x)
{
    return test_model<VGG19>(x, "vgg19.pt");
}

torch::Tensor test_vgg11bn(torch::Tensor x)
{
    return test_model<VGG11BN>(x, "vgg11bn.pt");
}
torch::Tensor test_vgg13bn(torch::Tensor x)
{
    return test_model<VGG13BN>(x, "vgg13bn.pt");
}
torch::Tensor test_vgg16bn(torch::Tensor x)
{
    return test_model<VGG16BN>(x, "vgg16bn.pt");
}
torch::Tensor test_vgg19bn(torch::Tensor x)
{
    return test_model<VGG19BN>(x, "vgg19bn.pt");
}

torch::Tensor test_resnet18(torch::Tensor x)
{
    return test_model<ResNet18>(x, "resnet18.pt");
}
torch::Tensor test_resnet34(torch::Tensor x)
{
    return test_model<ResNet34>(x, "resnet34.pt");
}
torch::Tensor test_resnet50(torch::Tensor x)
{
    return test_model<ResNet50>(x, "resnet50.pt");
}
torch::Tensor test_resnet101(torch::Tensor x)
{
    return test_model<ResNet101>(x, "resnet101.pt");
}
torch::Tensor test_resnet152(torch::Tensor x)
{
    return test_model<ResNet152>(x, "resnet152.pt");
}

torch::Tensor test_squeezenet1_0(torch::Tensor x)
{
    return test_model<SqueezeNet1_0>(x, "squeezenet1_0.pt");
}
torch::Tensor test_squeezenet1_1(torch::Tensor x)
{
    return test_model<SqueezeNet1_1>(x, "squeezenet1_1.pt");
}

torch::Tensor test_densenet121(torch::Tensor x)
{
    return test_model<DenseNet121>(x, "densenet121.pt");
}
torch::Tensor test_densenet169(torch::Tensor x)
{
    return test_model<DenseNet169>(x, "densenet169.pt");
}
torch::Tensor test_densenet201(torch::Tensor x)
{
    return test_model<DenseNet201>(x, "densenet201.pt");
}
torch::Tensor test_densenet161(torch::Tensor x)
{
    return test_model<DenseNet161>(x, "densenet161.pt");
}

torch::Tensor test_inceptionv3(torch::Tensor x)
{
    InceptionV3 network;
    torch::load(network, "inceptionv3.pt");
    return network->forward(x)[0];
}
