
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.io.image import read_image
from torchvision.models import resnet18


class Predictor(nn.Module):

    def __init__(self):
        super().__init__()
        self.resnet18 = resnet18(pretrained=True).eval()
        self.transforms = nn.Sequential(
            T.Resize(256),
            T.CenterCrop(224),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        )

    def forward(self, x: torch.Tensor) -> int:
        with torch.no_grad():
            x = x.float() / 255.0
            x = self.transforms(x)
            y_pred = self.resnet18(x.unsqueeze(0))
            return y_pred[0].argmax().item()


if __name__ == "__main__":

    path = "ILSVRC2012_val_00017789.JPEG"
    img_tensor = read_image(path)

    predictor = Predictor()
    scripted_predictor = torch.jit.script(predictor)
    res1 = scripted_predictor(img_tensor)
    res2 = predictor(img_tensor)
    print("Scripted predictor: ", res1)
    print("Original predictor: ", res2)

    scripted_predictor.save("scripted_predictor.pt")

    scripted_predictor = torch.jit.load("scripted_predictor.pt")
    res1 = scripted_predictor(img_tensor)
    print("Reloaded Scripted predictor: ", res1)
