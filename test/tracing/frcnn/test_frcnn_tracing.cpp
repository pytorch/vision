#include <torch/torch.h>
#include <torch/script.h>
#include <torchvision/nms.h>
#include <torchvision/ROIAlign.h>
#include <ATen/ATen.h>

#ifdef _WIN32
// This is necessary until operators are automatically registered on include
static auto registry = torch::RegisterOperators()
                           .op("torchvision::nms", &nms)
                           .op("torchvision::roi_align", &roi_align);
#endif

int main() {
    torch::DeviceType device_type;
    device_type = torch::kCPU;

    torch::jit::script::Module module;
    try {
        std::cout << "Loading model\n";
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load("fasterrcnn_resnet50_fpn.pt");
        std::cout << "Model loaded\n";
    }
    catch (const c10::Error& e) {
        std::cout << "error loading the model\n";
        return -1;
    } catch (const std::exception& e) {
      std::cout << "Other error: " << e.what() << "\n";
      return -1;
    }

    std::vector<torch::jit::IValue> inputs;
    std::vector<at::Tensor> images;
    images.push_back(torch::rand({3, 256, 275}));
    images.push_back(torch::rand({3, 256, 275}));

    inputs.push_back(images);
    auto output = model.forward(inputs);

    std::cout << "ok\n";
    std::cout << "output" << output << "\n";
    return 0;
}
