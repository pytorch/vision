#include <torchvision/vision.h>
#include <torch/csrc/jit/runtime/operator.h>

int main() {
  auto ops = torch::jit::getAllOperators();
  int vision_ops_count = 0;
  for (const auto &op : ops) {
    const auto &schema = op->schema();
    const auto &ns = schema.getNamespace();
    if (ns.has_value() && ns.value() == "torchvision")
      ++vision_ops_count;
  }
  return vision_ops_count;
}
