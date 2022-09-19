import pytest
import torch.jit

from common_utils import cpu_and_gpu, script
from prototype_transforms_dispatcher_infos import DISPATCHER_INFOS

from torchvision.prototype import features


class TestCommon:
    @pytest.mark.xfail(reason="dispatchers are currently not scriptable")
    @pytest.mark.parametrize(
        ("info", "args_kwargs"),
        [
            pytest.param(info, args_kwargs, id=f"{info.dispatcher.__name__}-{idx}")
            for info in DISPATCHER_INFOS
            for idx, args_kwargs in enumerate(info.sample_inputs(features.Image))
            if features.Image in info.kernels
        ],
    )
    @pytest.mark.parametrize("device", cpu_and_gpu())
    def test_scripted_smoke(self, info, args_kwargs, device):
        fn = script(info.dispatcher)

        (image_feature, *other_args), kwargs = args_kwargs.load(device)
        image_simple_tensor = torch.Tensor(image_feature)

        fn(image_simple_tensor, *other_args, **kwargs)
