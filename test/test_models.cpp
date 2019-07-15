#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>

#include "../torchvision/csrc/models/models.h"

using namespace vision::models;

template <typename Model>
torch::Tensor forward_model(const std::string& input_path, torch::Tensor x) {
  Model network;
  torch::load(network, input_path);
  network->eval();
  return network->forward(x);
}

torch::Tensor forward_alexnet(const std::string& input_path, torch::Tensor x) {
  return forward_model<AlexNet>(input_path, x);
}

torch::Tensor forward_vgg11(const std::string& input_path, torch::Tensor x) {
  return forward_model<VGG11>(input_path, x);
}
torch::Tensor forward_vgg13(const std::string& input_path, torch::Tensor x) {
  return forward_model<VGG13>(input_path, x);
}
torch::Tensor forward_vgg16(const std::string& input_path, torch::Tensor x) {
  return forward_model<VGG16>(input_path, x);
}
torch::Tensor forward_vgg19(const std::string& input_path, torch::Tensor x) {
  return forward_model<VGG19>(input_path, x);
}

torch::Tensor forward_vgg11bn(const std::string& input_path, torch::Tensor x) {
  return forward_model<VGG11BN>(input_path, x);
}
torch::Tensor forward_vgg13bn(const std::string& input_path, torch::Tensor x) {
  return forward_model<VGG13BN>(input_path, x);
}
torch::Tensor forward_vgg16bn(const std::string& input_path, torch::Tensor x) {
  return forward_model<VGG16BN>(input_path, x);
}
torch::Tensor forward_vgg19bn(const std::string& input_path, torch::Tensor x) {
  return forward_model<VGG19BN>(input_path, x);
}

torch::Tensor forward_resnet18(const std::string& input_path, torch::Tensor x) {
  return forward_model<ResNet18>(input_path, x);
}
torch::Tensor forward_resnet34(const std::string& input_path, torch::Tensor x) {
  return forward_model<ResNet34>(input_path, x);
}
torch::Tensor forward_resnet50(const std::string& input_path, torch::Tensor x) {
  return forward_model<ResNet50>(input_path, x);
}
torch::Tensor forward_resnet101(
    const std::string& input_path,
    torch::Tensor x) {
  return forward_model<ResNet101>(input_path, x);
}
torch::Tensor forward_resnet152(
    const std::string& input_path,
    torch::Tensor x) {
  return forward_model<ResNet152>(input_path, x);
}
torch::Tensor forward_resnext50_32x4d(
    const std::string& input_path,
    torch::Tensor x) {
  return forward_model<ResNext50_32x4d>(input_path, x);
}
torch::Tensor forward_resnext101_32x8d(
    const std::string& input_path,
    torch::Tensor x) {
  return forward_model<ResNext101_32x8d>(input_path, x);
}
torch::Tensor forward_wide_resnet50_2(
    const std::string& input_path,
    torch::Tensor x) {
  return forward_model<WideResNet50_2>(input_path, x);
}
torch::Tensor forward_wide_resnet101_2(
    const std::string& input_path,
    torch::Tensor x) {
  return forward_model<WideResNet101_2>(input_path, x);
}

torch::Tensor forward_squeezenet1_0(
    const std::string& input_path,
    torch::Tensor x) {
  return forward_model<SqueezeNet1_0>(input_path, x);
}
torch::Tensor forward_squeezenet1_1(
    const std::string& input_path,
    torch::Tensor x) {
  return forward_model<SqueezeNet1_1>(input_path, x);
}

torch::Tensor forward_densenet121(
    const std::string& input_path,
    torch::Tensor x) {
  return forward_model<DenseNet121>(input_path, x);
}
torch::Tensor forward_densenet169(
    const std::string& input_path,
    torch::Tensor x) {
  return forward_model<DenseNet169>(input_path, x);
}
torch::Tensor forward_densenet201(
    const std::string& input_path,
    torch::Tensor x) {
  return forward_model<DenseNet201>(input_path, x);
}
torch::Tensor forward_densenet161(
    const std::string& input_path,
    torch::Tensor x) {
  return forward_model<DenseNet161>(input_path, x);
}

torch::Tensor forward_mobilenetv2(
    const std::string& input_path,
    torch::Tensor x) {
  return forward_model<MobileNetV2>(input_path, x);
}

torch::Tensor forward_googlenet(
    const std::string& input_path,
    torch::Tensor x) {
  GoogLeNet network;
  torch::load(network, input_path);
  network->eval();
  return network->forward(x).output;
}
torch::Tensor forward_inceptionv3(
    const std::string& input_path,
    torch::Tensor x) {
  InceptionV3 network;
  torch::load(network, input_path);
  network->eval();
  return network->forward(x).output;
}

torch::Tensor forward_mnasnet0_5(const std::string& input_path, torch::Tensor x) {
  return forward_model<MNASNet0_5>(input_path, x);
}
torch::Tensor forward_mnasnet0_75(const std::string& input_path, torch::Tensor x) {
  return forward_model<MNASNet0_75>(input_path, x);
}
torch::Tensor forward_mnasnet1_0(const std::string& input_path, torch::Tensor x) {
  return forward_model<MNASNet1_0>(input_path, x);
}
torch::Tensor forward_mnasnet1_3(const std::string& input_path, torch::Tensor x) {
  return forward_model<MNASNet1_3>(input_path, x);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_alexnet", &forward_alexnet, "forward_alexnet");

  m.def("forward_vgg11", &forward_vgg11, "forward_vgg11");
  m.def("forward_vgg13", &forward_vgg13, "forward_vgg13");
  m.def("forward_vgg16", &forward_vgg16, "forward_vgg16");
  m.def("forward_vgg19", &forward_vgg19, "forward_vgg19");

  m.def("forward_vgg11bn", &forward_vgg11bn, "forward_vgg11bn");
  m.def("forward_vgg13bn", &forward_vgg13bn, "forward_vgg13bn");
  m.def("forward_vgg16bn", &forward_vgg16bn, "forward_vgg16bn");
  m.def("forward_vgg19bn", &forward_vgg19bn, "forward_vgg19bn");

  m.def("forward_resnet18", &forward_resnet18, "forward_resnet18");
  m.def("forward_resnet34", &forward_resnet34, "forward_resnet34");
  m.def("forward_resnet50", &forward_resnet50, "forward_resnet50");
  m.def("forward_resnet101", &forward_resnet101, "forward_resnet101");
  m.def("forward_resnet152", &forward_resnet152, "forward_resnet152");
  m.def(
      "forward_resnext50_32x4d",
      &forward_resnext50_32x4d,
      "forward_resnext50_32x4d");
  m.def(
      "forward_resnext101_32x8d",
      &forward_resnext101_32x8d,
      "forward_resnext101_32x8d");
  m.def(
      "forward_wide_resnet50_2",
      &forward_wide_resnet50_2,
      "forward_wide_resnet50_2");
  m.def(
      "forward_wide_resnet101_2",
      &forward_wide_resnet101_2,
      "forward_wide_resnet101_2");

  m.def(
      "forward_squeezenet1_0", &forward_squeezenet1_0, "forward_squeezenet1_0");
  m.def(
      "forward_squeezenet1_1", &forward_squeezenet1_1, "forward_squeezenet1_1");

  m.def("forward_densenet121", &forward_densenet121, "forward_densenet121");
  m.def("forward_densenet169", &forward_densenet169, "forward_densenet169");
  m.def("forward_densenet201", &forward_densenet201, "forward_densenet201");
  m.def("forward_densenet161", &forward_densenet161, "forward_densenet161");

  m.def("forward_mobilenetv2", &forward_mobilenetv2, "forward_mobilenetv2");

  m.def("forward_googlenet", &forward_googlenet, "forward_googlenet");
  m.def("forward_inceptionv3", &forward_inceptionv3, "forward_inceptionv3");

  m.def("forward_mnasnet0_5", &forward_mnasnet0_5, "forward_mnasnet0_5");
  m.def("forward_mnasnet0_75", &forward_mnasnet0_75, "forward_mnasnet0_75");
  m.def("forward_mnasnet1_0", &forward_mnasnet1_0, "forward_mnasnet1_0");
  m.def("forward_mnasnet1_3", &forward_mnasnet1_3, "forward_mnasnet1_3");
}
