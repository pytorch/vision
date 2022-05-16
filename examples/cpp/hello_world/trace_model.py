import os.path as osp

import torch
import torchvision

HERE = osp.dirname(osp.abspath(__file__))
ASSETS = osp.dirname(osp.dirname(HERE))

model = torchvision.models.resnet18()
model.eval()

traced_model = torch.jit.script(model)
traced_model.save("resnet18.pt")
