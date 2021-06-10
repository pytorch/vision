import torchvision
import torch
from torchvision import models
import pytest
import traceback


def get_available_quantizable_models():
    # TODO add a registration mechanism to torchvision.models
    return [k for k, v in models.quantization.__dict__.items() if callable(v) and k[0].lower() == k[0] and k[0] != "_"]


@pytest.mark.skipif(not ('fbgemm' in torch.backends.quantized.supported_engines and
                         'qnnpack' in torch.backends.quantized.supported_engines),
                    reason="This Pytorch Build has not been built with fbgemm and qnnpack")
# inception_v3 was causing timeouts on circleci
# See https://github.com/pytorch/vision/issues/1857
@pytest.mark.parametrize('name', get_available_quantizable_models())
def test_classification_model(name):

    # First check if quantize=True provides models that can run with input data
    input_shape = (1, 3, 224, 224)
    model = torchvision.models.quantization.__dict__[name](pretrained=False, quantize=True)
    x = torch.rand(input_shape)
    model(x)

    for eval_mode in [True, False]:
        model = torchvision.models.quantization.__dict__[name](pretrained=False, quantize=False)
        if eval_mode:
            model.eval()
            model.qconfig = torch.quantization.default_qconfig
        else:
            model.train()
            model.qconfig = torch.quantization.default_qat_qconfig

        model.fuse_model()
        if eval_mode:
            torch.quantization.prepare(model, inplace=True)
        else:
            torch.quantization.prepare_qat(model, inplace=True)
            model.eval()

        torch.quantization.convert(model, inplace=True)

    try:
        torch.jit.script(model)
    except Exception as e:
        tb = traceback.format_exc()
        raise AssertionError(f"model cannot be scripted. Traceback = {str(tb)}") from e


if __name__ == '__main__':
    pytest.main([__file__])
