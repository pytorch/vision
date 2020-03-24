#include <iostream>

#include <torchvision/models/resnet.h>

int main()
{
  auto model = vision::models::ResNet18();
  model->eval();

  // Create a random input tensor and run it through the model.
  auto in = torch::rand({1, 3, 10, 10});
  auto out = model->forward(in);

  std::cout << out;
}
