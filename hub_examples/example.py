from torchvision.models.alexnet import alexnet

def wrapper1(pretrained=False, **kwargs):
    model = alexnet(pretrained, **kwargs)
    return model


