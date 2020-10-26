import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

print(torch.__version__)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

model.eval()
script_model = torch.jit.script(model)
torch.jit.save(script_model, "app/src/main/assets/frcnn_resnet50_fpn.pt")

model_300_500 = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, min_size=300, max_size=500)
model_300_500.eval()
script_model_300_500 = torch.jit.script(model_300_500)
torch.jit.save(script_model_300_500, "app/src/main/assets/frcnn_resnet50_fpn_300_500.pt")
