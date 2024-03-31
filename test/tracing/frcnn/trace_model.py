import os.path as osp

import torch
import torchvision

HERE = osp.dirname(osp.abspath(__file__))
ASSETS = osp.dirname(osp.dirname(HERE))

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
model.eval()

traced_model = torch.jit.script(model)
traced_model.save("fasterrcnn_resnet50_fpn.pt")
