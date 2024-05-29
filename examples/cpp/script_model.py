import torch
from torchvision import models

for model, name in (
    (models.resnet18(weights=None), "resnet18"),
    (models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None), "fasterrcnn_resnet50_fpn"),
):
    model.eval()
    traced_model = torch.jit.script(model)
    traced_model.save(f"{name}.pt")
