from torchvision.prototype import datasets

dataset = datasets.load("ucf101")
for i in dataset:
    print(i)
    break 