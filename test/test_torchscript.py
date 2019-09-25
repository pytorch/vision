import unittest
import torch
import torchvision
import contextlib


@contextlib.contextmanager
def freeze_rng_state():
    rng_state = torch.get_rng_state()
    if torch.cuda.is_available():
        cuda_rng_state = torch.cuda.get_rng_state()
    yield
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(cuda_rng_state)
    torch.set_rng_state(rng_state)


class TestTorchScript(unittest.TestCase):
    def checkModule(self, nn_module, args):
        """
        Check that a nn.Module's results in Script mode match eager and that it
        can be exported
        """
        sm = torch.jit.script(nn_module)

        # TODO: Add these back in
        # with freeze_rng_state():
        #     eager_out = nn_module(*args)

        # with freeze_rng_state():
        #     script_out = sm(*args)

        # self.assertEqual(eager_out, script_out)
        # self.assertExportImportModule(sm, args)

        return sm

    def test_maskrcnn(self):
        # print("Making")
        m = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)
        inputs = (torch.randn(2, 2),)
        sm = self.checkModule(m, inputs)
        # print(sm.graph)


if __name__ == '__main__':
    unittest.main()
