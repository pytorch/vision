import os

import pytest
import torch
from common_utils import cpu_and_cuda, needs_cuda
from einops import rearrange
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from torchvision.transforms.autoaugment import PatchMix
from torchvision.transforms.functional import to_pil_image


@needs_cuda
@pytest.mark.parametrize("batch_size ", (4, 7))
@pytest.mark.parametrize("prob", (1.0, 0.5, 0.0))
@pytest.mark.parametrize("mix_num", (1, 2, 3, 4))
@pytest.mark.parametrize("device", cpu_and_cuda())
def test_patchmix(batch_size, prob, mix_num, device):
    data_set = datasets.ImageFolder(
        root=os.path.join("/tmp/ILSVRC2012", "train"),
        transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]),
    )
    data_loader = DataLoader(dataset=data_set, batch_size=batch_size, num_workers=4, shuffle=True)

    patchmix = PatchMix(10, prob, mix_num, 16)

    for images, _ in data_loader:
        images = images.to(device)
        target = torch.arange(batch_size).to(device)
        org_img = rearrange(images, "b c w h -> c (b w) h")
        mix_img, mo_target, mm_target = patchmix(images, target)
        mix_img = rearrange(mix_img, "b c w h -> c (b w) h")
        result = torch.cat([org_img, mix_img], dim=-1)
        to_pil_image(result).save("result.png")
        print(target)
        print(mo_target)
        print(mm_target)
        break
