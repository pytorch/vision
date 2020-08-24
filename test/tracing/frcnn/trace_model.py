
import torch
import torchvision

model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
input_imgs = torch.rand(2, 3, 256, 275).split(1)
input_imgs = [x.squeeze() for x in input_imgs]

traced_model = torch.jit.trace(model, input_imgs)
traced_model.save("fasterrcnn_resnet50_fpn.pt")
