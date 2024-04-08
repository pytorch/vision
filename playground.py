import torch
import torchvision

def fn1(x):
    return torch.ops.torchvision.my_custom_op1(x)

def fn2(x):
    return torch.ops.torchvision.my_custom_op2(x)

opt_fn1 = torch.compile(fn1, backend="eager")
opt_fn2 = torch.compile(fn2, backend="eager")
print(opt_fn2(torch.randn(3, 3)))
print(opt_fn1(torch.randn(3, 3)))
