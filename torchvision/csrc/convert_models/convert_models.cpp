#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>

#include "../models/models.h"

using namespace vision::models;

template <typename Model>
void convert_and_save_model(
    const std::string& input_path,
    const std::string& output_path) {
  Model network;
  torch::load(network, input_path);
  torch::save(network, output_path);

  auto index = input_path.find("_python");
  auto name = input_path.substr(0, index);
  std::cout << "finished loading and saving " << name << std::endl;
}

int main(int argc, const char* argv[]) {
  convert_and_save_model<AlexNet>("alexnet_python.pt", "alexnet_cpp.pt");

  convert_and_save_model<VGG11>("vgg11_python.pt", "vgg11_cpp.pt");
  convert_and_save_model<VGG13>("vgg13_python.pt", "vgg13_cpp.pt");
  convert_and_save_model<VGG16>("vgg16_python.pt", "vgg16_cpp.pt");
  convert_and_save_model<VGG19>("vgg19_python.pt", "vgg19_cpp.pt");

  convert_and_save_model<VGG11BN>("vgg11bn_python.pt", "vgg11bn_cpp.pt");
  convert_and_save_model<VGG13BN>("vgg13bn_python.pt", "vgg13bn_cpp.pt");
  convert_and_save_model<VGG16BN>("vgg16bn_python.pt", "vgg16bn_cpp.pt");
  convert_and_save_model<VGG19BN>("vgg19bn_python.pt", "vgg19bn_cpp.pt");

  convert_and_save_model<ResNet18>("resnet18_python.pt", "resnet18_cpp.pt");
  convert_and_save_model<ResNet34>("resnet34_python.pt", "resnet34_cpp.pt");
  convert_and_save_model<ResNet50>("resnet50_python.pt", "resnet50_cpp.pt");
  convert_and_save_model<ResNet101>("resnet101_python.pt", "resnet101_cpp.pt");
  convert_and_save_model<ResNet152>("resnet152_python.pt", "resnet152_cpp.pt");
  convert_and_save_model<ResNext50_32x4d>(
      "resnext50_32x4d_python.pt", "resnext50_32x4d_cpp.pt");
  convert_and_save_model<ResNext101_32x8d>(
      "resnext101_32x8d_python.pt", "resnext101_32x8d_cpp.pt");

  convert_and_save_model<SqueezeNet1_0>(
      "squeezenet1_0_python.pt", "squeezenet1_0_cpp.pt");
  convert_and_save_model<SqueezeNet1_1>(
      "squeezenet1_1_python.pt", "squeezenet1_1_cpp.pt");

  convert_and_save_model<DenseNet121>(
      "densenet121_python.pt", "densenet121_cpp.pt");
  convert_and_save_model<DenseNet169>(
      "densenet169_python.pt", "densenet169_cpp.pt");
  convert_and_save_model<DenseNet201>(
      "densenet201_python.pt", "densenet201_cpp.pt");
  convert_and_save_model<DenseNet161>(
      "densenet161_python.pt", "densenet161_cpp.pt");

  convert_and_save_model<MobileNetV2>(
      "mobilenetv2_python.pt", "mobilenetv2_cpp.pt");

  convert_and_save_model<ShuffleNetV2_x0_5>(
      "shufflenetv2_x0_5_python.pt", "shufflenetv2_x0_5_cpp.pt");
  convert_and_save_model<ShuffleNetV2_x1_0>(
      "shufflenetv2_x1_0_python.pt", "shufflenetv2_x1_0_cpp.pt");
  convert_and_save_model<ShuffleNetV2_x1_5>(
      "shufflenetv2_x1_5_python.pt", "shufflenetv2_x1_5_cpp.pt");
  convert_and_save_model<ShuffleNetV2_x2_0>(
      "shufflenetv2_x2_0_python.pt", "shufflenetv2_x2_0_cpp.pt");

  convert_and_save_model<GoogLeNet>("googlenet_python.pt", "googlenet_cpp.pt");
  convert_and_save_model<InceptionV3>(
      "inceptionv3_python.pt", "inceptionv3_cpp.pt");

  return 0;
}
