#include <iostream>
#include <torch/script.h>
#include <torch/torch.h>
#include <torchvision/vision.h>

int main() {
  torch::DeviceType device_type;
  device_type = torch::kCPU;

  torch::jit::script::Module model;
  try {
    std::cout << "Loading model\n";
    // Deserialize the ScriptModule from a file using torch::jit::load().
    model = torch::jit::load("resnet18.pt");
    std::cout << "Model loaded\n";
  } catch (const torch::Error& e) {
    std::cout << "error loading the model\n";
    return -1;
  } catch (const std::exception& e) {
    std::cout << "Other error: " << e.what() << "\n";
    return -1;
  }

  // TorchScript models require a List[IValue] as input
  std::vector<torch::jit::IValue> inputs;

  // Create a random input tensor and run it through the model.
  inputs.push_back(torch::rand({1, 3, 10, 10}));
  auto out = model.forward(inputs);
  std::cout << out << "\n";

  if (torch::cuda::is_available()) {
    // Move model and inputs to GPU
    model.to(torch::kCUDA);

    // Add GPU inputs
    inputs.clear();
    torch::TensorOptions options = torch::TensorOptions{torch::kCUDA};
    inputs.push_back(torch::rand({1, 3, 10, 10}, options));

    auto gpu_out = model.forward(inputs);
    std::cout << gpu_out << "\n";
  }
}
