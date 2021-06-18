#include <torch/script.h>
#include <torch/torch.h>
#include <torchvision/vision.h>
#include <torchvision/ops/nms.h>


int main() {
  torch::DeviceType device_type;
  device_type = torch::kCPU;

  torch::jit::script::Module module;
  try {
    std::cout << "Loading model\n";
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load("fasterrcnn_resnet50_fpn.pt");
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

  // Faster RCNN accepts a List[Tensor] as main input
  std::vector<torch::Tensor> images;
  images.push_back(torch::rand({3, 256, 275}));
  images.push_back(torch::rand({3, 256, 275}));

  inputs.push_back(images);
  auto output = module.forward(inputs);

  std::cout << "ok\n";
  std::cout << "output" << output << "\n";

  if (torch::cuda::is_available()) {
    // Move traced model to GPU
    module.to(torch::kCUDA);

    // Add GPU inputs
    images.clear();
    inputs.clear();

    torch::TensorOptions options = torch::TensorOptions{torch::kCUDA};
    images.push_back(torch::rand({3, 256, 275}, options));
    images.push_back(torch::rand({3, 256, 275}, options));

    inputs.push_back(images);
    auto output = module.forward(inputs);

    std::cout << "ok\n";
    std::cout << "output" << output << "\n";
  }
  return 0;
}
