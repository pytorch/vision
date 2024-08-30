#include <torch/script.h>
#include <torch/torch.h>
#include <cstring>
#include <iostream>

#ifdef _WIN32
#include <torchvision/vision.h>
#endif // _WIN32

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cout << "Usage: run_model <path_to_scripted_model>\n";
    return -1;
  }
  torch::DeviceType device_type;
  device_type = torch::kCPU;

  torch::jit::script::Module model;
  try {
    std::cout << "Loading model\n";
    // Deserialize the ScriptModule from a file using torch::jit::load().
    model = torch::jit::load(argv[1]);
    std::cout << "Model loaded\n";
  } catch (const torch::Error& e) {
    std::cout << "error loading the model.\n";
    return -1;
  } catch (const std::exception& e) {
    std::cout << "Other error: " << e.what() << "\n";
    return -1;
  }

  // TorchScript models require a List[IValue] as input
  std::vector<torch::jit::IValue> inputs;

  if (std::strstr(argv[1], "fasterrcnn") != NULL) {
    // Faster RCNN accepts a List[Tensor] as main input
    std::vector<torch::Tensor> images;
    images.push_back(torch::rand({3, 256, 275}));
    images.push_back(torch::rand({3, 256, 275}));
    inputs.push_back(images);
  } else {
    inputs.push_back(torch::rand({1, 3, 10, 10}));
  }
  auto out = model.forward(inputs);
  std::cout << out << "\n";

  if (torch::cuda::is_available()) {
    // Move model and inputs to GPU
    model.to(torch::kCUDA);

    // Add GPU inputs
    inputs.clear();
    torch::TensorOptions options = torch::TensorOptions{torch::kCUDA};
    if (std::strstr(argv[1], "fasterrcnn") != NULL) {
      // Faster RCNN accepts a List[Tensor] as main input
      std::vector<torch::Tensor> images;
      images.push_back(torch::rand({3, 256, 275}, options));
      images.push_back(torch::rand({3, 256, 275}, options));
      inputs.push_back(images);
    } else {
      inputs.push_back(torch::rand({1, 3, 10, 10}, options));
    }

    auto gpu_out = model.forward(inputs);
    std::cout << gpu_out << "\n";
  }
}
