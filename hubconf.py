import torch.utils.model_zoo as model_zoo

# Optional list of dependencies required by the package
dependencies = ['torch', 'math']

def wrapper1(pretrained=False, *args, **kwargs):
    """
    pretrained (bool): a recommended kwargs for all entrypoints
    args & kwargs are arguments for the function
    """
    from torchvision.models.resnet import resnet18
    model = resnet18(*args, **kwargs)
    checkpoint = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
    if pretrained:
        model.load_state_dict(model_zoo.load_url(checkpoint, progress=False))
    return model

def wrapper2(pretrained=False, *args, **kwargs):
    """
    pretrained (bool): a recommended kwargs for all entrypoints
    args & kwargs are arguments for the function
    """
    from torchvision.models.resnet import resnet50
    model = resnet50(*args, **kwargs)
    checkpoint = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
    if pretrained:
        model.load_state_dict(model_zoo.load_url(checkpoint, progress=False))
    return model

