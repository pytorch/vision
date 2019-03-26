import torch
from torchvision import models
import unittest


def get_available_models():
    # TODO add a registration mechanism to torchvision.models
    return [k for k, v in models.__dict__.items() if callable(v) and k[0].lower() == k[0]]


class Tester(unittest.TestCase):
    def _test_model(self, name, input_shape):
        # passing num_class equal to a number other than 1000 helps in making the test more enforcing in nature
        model = models.__dict__[name](num_classes=50)
        model.eval()
        x = torch.rand(input_shape)
        out = model(x)
        self.assertEqual(out.shape[-1], 50)


for model_name in get_available_models():
    # for-loop bodies don't define scopes, so we have to save the variables
    # we want to close over in some way
    def do_test(self, model_name=model_name):
        input_shape = (1, 3, 224, 224)
        if model_name in ['inception_v3']:
            input_shape = (1, 3, 299, 299)
        self._test_model(model_name, input_shape)

    setattr(Tester, "test_" + model_name, do_test)


if __name__ == '__main__':
    unittest.main()
