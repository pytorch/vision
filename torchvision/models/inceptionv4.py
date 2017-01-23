import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = ['InceptionV4', 'inceptionv4']

model_urls = {
    'inceptionv4': 'https://s3.amazonaws.com/pytorch/models/inceptionv4-58153ba9.pth'
}

class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False) # verify bias false
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001, momentum=0, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Mixed_3a(nn.Module):

    def __init__(self):
        super(Mixed_3a, self).__init__()
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.conv = BasicConv2d(64, 96, kernel_size=3, stride=2)

    def forward(self, x):
        x0 = self.maxpool(x)
        x1 = self.conv(x)
        out = torch.cat((x0, x1), 1)
        return out

class Mixed_4a(nn.Module):

    def __init__(self):
        super(Mixed_4a, self).__init__()

        self.block0 = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1)
        )

        self.block1 = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 64, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(64, 64, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(64, 96, kernel_size=(3,3), stride=1)
        )

    def forward(self, x):
        x0 = self.block0(x)
        x1 = self.block1(x)
        out = torch.cat((x0, x1), 1)
        return out

class Mixed_5a(nn.Module):

    def __init__(self):
        super(Mixed_5a, self).__init__()
        self.conv = BasicConv2d(192, 192, kernel_size=3, stride=2)
        self.maxpool = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.maxpool(x)
        out = torch.cat((x0, x1), 1)
        return out

class Inception_A(nn.Module):

    def __init__(self):
        super(Inception_A, self).__init__()
        self.block0 = BasicConv2d(384, 96, kernel_size=1, stride=1)

        self.block1 = nn.Sequential(
            BasicConv2d(384, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1)
        )

        self.block2 = nn.Sequential(
            BasicConv2d(384, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1),
            BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1)
        )

        self.block3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(384, 96, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.block0(x)
        x1 = self.block1(x)
        x2 = self.block2(x)
        x3 = self.block3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out

class Reduction_A(nn.Module):

    def __init__(self):
        super(Reduction_A, self).__init__()
        self.block0 = BasicConv2d(384, 384, kernel_size=3, stride=2)

        self.block1 = nn.Sequential(
            BasicConv2d(384, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224, kernel_size=3, stride=1, padding=1),
            BasicConv2d(224, 256, kernel_size=3, stride=2)
        )
        
        self.block2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.block0(x)
        x1 = self.block1(x)
        x2 = self.block2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out

class Inception_B(nn.Module):

    def __init__(self):
        super(Inception_B, self).__init__()
        self.block0 = BasicConv2d(1024, 384, kernel_size=1, stride=1)
        
        self.block1 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(224, 256, kernel_size=(7,1), stride=1, padding=(3,0))
        )

        self.block2 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(192, 224, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(224, 224, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(224, 256, kernel_size=(1,7), stride=1, padding=(0,3))
        )

        self.block3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(1024, 128, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.block0(x)
        x1 = self.block1(x)
        x2 = self.block2(x)
        x3 = self.block3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out

class Reduction_B(nn.Module):

    def __init__(self):
        super(Reduction_B, self).__init__()

        self.block0 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=3, stride=2)
        )

        self.block1 = nn.Sequential(
            BasicConv2d(1024, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(256, 320, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(320, 320, kernel_size=3, stride=2)
        )

        self.block2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.block0(x)
        x1 = self.block1(x)
        x2 = self.block2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out

class Inception_C(nn.Module):

    def __init__(self):
        super(Inception_C, self).__init__()
        self.block0 = BasicConv2d(1536, 256, kernel_size=1, stride=1)
        
        self.block1_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.block1_1a = BasicConv2d(384, 256, kernel_size=(1,3), stride=1, padding=(0,1))
        self.block1_1b = BasicConv2d(384, 256, kernel_size=(3,1), stride=1, padding=(1,0))
        
        self.block2_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.block2_1 = BasicConv2d(384, 448, kernel_size=(3,1), stride=1, padding=(1,0))
        self.block2_2 = BasicConv2d(448, 512, kernel_size=(1,3), stride=1, padding=(0,1))
        self.block2_3a = BasicConv2d(512, 256, kernel_size=(1,3), stride=1, padding=(0,1))
        self.block2_3b = BasicConv2d(512, 256, kernel_size=(3,1), stride=1, padding=(1,0))
        
        self.block3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(1536, 256, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.block0(x)
        
        x1_0 = self.block1_0(x)
        x1_1a = self.block1_1a(x1_0)
        x1_1b = self.block1_1b(x1_0)
        x1 = torch.cat((x1_1a, x1_1b), 1)

        x2_0 = self.block2_0(x)
        x2_1 = self.block2_1(x2_0)
        x2_2 = self.block2_2(x2_1)
        x2_3a = self.block2_3a(x2_2)
        x2_3b = self.block2_3b(x2_2)
        x2 = torch.cat((x2_3a, x2_3b), 1)

        x3 = self.block3(x)

        out = torch.cat((x0, x1, x2, x3), 1)
        return out

class InceptionV4(nn.Module):

    def __init__(self, num_classes=1001):
        super(InceptionV4, self).__init__()
        self.features = nn.Sequential(
            BasicConv2d(3, 32, kernel_size=3, stride=2),
            BasicConv2d(32, 32, kernel_size=3, stride=1),
            BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1),
            Mixed_3a(),
            Mixed_4a(),
            Mixed_5a(),
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Reduction_A(), # Mixed_6a
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Reduction_B(), # Mixed_7a
            Inception_C(),
            Inception_C(),
            Inception_C(),
            nn.AvgPool2d(8, count_include_pad=False)
        )
        self.classif = nn.Linear(1536, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classif(x) 
        return x


def inceptionv4(pretrained=False):
    r"""InceptionV4 model architecture from the
    `"Inception-v4, Inception-ResNet..." <https://arxiv.org/abs/1602.07261>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = InceptionV4()
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['inceptionv4']))
    return model
