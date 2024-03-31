import torch


def freeze_batch_norm(model):
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()


def unfreeze_batch_norm(model):
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.train()
