import torchvision
from common_utils import TestCase, map_nested_tensor_object
from collections import OrderedDict
from itertools import product
import torch
import numpy as np
from torchvision import models
import unittest
import traceback
import random


def set_rng_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_available_quantizable_models():
    # TODO add a registration mechanism to torchvision.models
    return [k for k, v in models.quantization.__dict__.items() if callable(v) and k[0].lower() == k[0] and k[0] != "_"]


# list of models that are not scriptable
scriptable_quantizable_models_blacklist = []


@unittest.skipUnless('fbgemm' in torch.backends.quantized.supported_engines and
                     'qnnpack' in torch.backends.quantized.supported_engines,
                     "This Pytorch Build has not been built with fbgemm and qnnpack")
class ModelTester(TestCase):
    def check_quantized_model(self, model, input_shape):
        x = torch.rand(input_shape)
        model(x)
        return

    def check_script(self, model, name):
        if name in scriptable_quantizable_models_blacklist:
            return
        scriptable = True
        msg = ""
        try:
            torch.jit.script(model)
        except Exception as e:
            tb = traceback.format_exc()
            scriptable = False
            msg = str(e) + str(tb)
        self.assertTrue(scriptable, msg)

    def _test_classification_model(self, name, input_shape):
        # First check if quantize=True provides models that can run with input data

        model = torchvision.models.quantization.__dict__[name](pretrained=False, quantize=True)
        self.check_quantized_model(model, input_shape)

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

        self.check_script(model, name)


for model_name in get_available_quantizable_models():
    # for-loop bodies don't define scopes, so we have to save the variables
    # we want to close over in some way
    def do_test(self, model_name=model_name):
        input_shape = (1, 3, 224, 224)
        if model_name in ['inception_v3']:
            input_shape = (1, 3, 299, 299)
        self._test_classification_model(model_name, input_shape)

    # inception_v3 was causing timeouts on circleci
    # See https://github.com/pytorch/vision/issues/1857
    if model_name not in ['inception_v3']:
        setattr(ModelTester, "test_" + model_name, do_test)


if __name__ == '__main__':
    unittest.main()
