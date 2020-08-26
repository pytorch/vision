#include <torch/torch.h>
#include <torch/script.h>
#include <torchvision/nms.h>
#include <ATen/ATen.h>

int main() {
    torch::DeviceType device_type;
    device_type = torch::kCPU;

    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load("fasterrcnn_resnet50_fpn.pt");
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::rand({3, 256, 275}));
    inputs.push_back(torch::rand({3, 256, 275}));

    std::cout << "ok\n";
    return 0;
}
