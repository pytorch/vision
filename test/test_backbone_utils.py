import unittest

import torch
import torchvision
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from torchvision.models._utils import IntermediateLayerGetter, IntermediateLayerGetter2


class ResnetFPNBackboneTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dtype = torch.float32

    def test_resnet18_fpn_backbone(self):
        device = torch.device('cpu')
        x = torch.rand(1, 3, 300, 300, dtype=self.dtype, device=device)
        resnet18_fpn = resnet_fpn_backbone(backbone_name='resnet18', pretrained=False)
        y = resnet18_fpn(x)
        self.assertEqual(list(y.keys()), ['0', '1', '2', '3', 'pool'])

    def test_resnet50_fpn_backbone(self):
        device = torch.device('cpu')
        x = torch.rand(1, 3, 300, 300, dtype=self.dtype, device=device)
        resnet50_fpn = resnet_fpn_backbone(backbone_name='resnet50', pretrained=False)
        y = resnet50_fpn(x)
        self.assertEqual(list(y.keys()), ['0', '1', '2', '3', 'pool'])


class IntermediateLayerGetterTester(unittest.TestCase):
    def test_old_new_match(self):
        model = torchvision.models.resnet18(pretrained=False)

        return_layers = {'layer2': '5', 'layer4': 'pool'}

        old_model = IntermediateLayerGetter2(model, return_layers).eval()
        new_model = IntermediateLayerGetter(model, return_layers).eval()

        # check that we have same parameters
        for (n1, p1), (n2, p2) in zip(old_model.named_parameters(), new_model.named_parameters()):
            self.assertEqual(n1, n2)
            self.assertTrue(p1.equal(p2))

        # and state_dict matches
        for (n1, p1), (n2, p2) in zip(old_model.state_dict().items(), new_model.state_dict().items()):
            self.assertEqual(n1, n2)
            self.assertTrue(p1.equal(p2))

        # check that we actually compute the same thing
        x = torch.rand(2, 3, 224, 224)
        old_out = old_model(x)
        new_out = new_model(x)
        self.assertEqual(old_out.keys(), new_out.keys())
        for k in old_out.keys():
            o1 = old_out[k]
            o2 = new_out[k]
            self.assertTrue(o1.equal(o2))

        # check torchscriptability
        script_new_model = torch.jit.script(new_model)
        new_out_script = script_new_model(x)
        self.assertEqual(old_out.keys(), new_out_script.keys())
        for k in old_out.keys():
            o1 = old_out[k]
            o2 = new_out_script[k]
            self.assertTrue(o1.equal(o2))

        # check assert that non-existing keys raise error
        with self.assertRaises(ValueError):
            _ = IntermediateLayerGetter(model, {'layer5': '0'})


if __name__ == "__main__":
    unittest.main()
