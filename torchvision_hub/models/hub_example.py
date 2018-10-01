from torchvision_hub.models.resnet import resnet18

def wrapper1(pretrained=False, **kwargs):
    model = resnet18(pretrained, **kwargs)
    return model

