#include "torchvision/csrc/datasets/datasets.h"
#include "torchvision/csrc/datasets/datasetsimpl.h"

using namespace std;
using namespace vision::datasets;
using namespace vision::datasets::datasetsimpl;

int main() {
  torch::nn::Sequential(
      torch::nn::Sequential(),
      torch::nn::Sequential(),
      torch::nn::Sequential());
  auto T = torch::tensor(10);
  cout << T << endl;
}
