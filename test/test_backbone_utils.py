import unittest
import torch
import torchvision
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.feature_extraction import build_feature_graph_net
from torchvision.models.feature_extraction import IntermediateLayerGetter

import pytest


@pytest.mark.parametrize('backbone_name', ('resnet18', 'resnet50'))
def test_resnet_fpn_backbone(backbone_name):
    x = torch.rand(1, 3, 300, 300, dtype=torch.float32, device='cpu')
    y = resnet_fpn_backbone(backbone_name=backbone_name, pretrained=False)(x)
    assert list(y.keys()) == ['0', '1', '2', '3', 'pool']


# Needed by TestFeatureExtraction.test_feature_graph_net_leaf_module_and_function
def leaf_function(x):
    return int(x)


class TestFeatureExtraction(unittest.TestCase):
    model = torchvision.models.resnet18(pretrained=False, num_classes=1).eval()
    return_layers = {
        'layer1': 'layer1',
        'layer2': 'layer2',
        'layer3': 'layer3',
        'layer4': 'layer4'
    }
    inp = torch.rand(1, 3, 224, 224, dtype=torch.float32, device='cpu')
    expected_out_shapes = [
        torch.Size([1, 64, 56, 56]),
        torch.Size([1, 128, 28, 28]),
        torch.Size([1, 256, 14, 14]),
        torch.Size([1, 512, 7, 7])
    ]

    def test_build_feature_graph_net(self):
        # Check that it works with both a list and dict for return nodes
        build_feature_graph_net(self.model, self.return_layers)
        build_feature_graph_net(self.model, list(self.return_layers.keys()))
        # Check must specify return nodes
        with pytest.raises(AssertionError):
            build_feature_graph_net(self.model)
        # Check return_nodes and train_return_nodes / eval_return nodes
        # mutual exclusivity
        with pytest.raises(AssertionError):
            build_feature_graph_net(
                self.model, return_nodes=self.return_layers,
                train_return_nodes=self.return_layers)
        # Check train_return_nodes / eval_return nodes must both be specified
        with pytest.raises(AssertionError):
            build_feature_graph_net(
                self.model, train_return_nodes=self.return_layers)

    def test_feature_graph_net_forward_backward(self):
        model = build_feature_graph_net(self.model, self.return_layers)
        out = model(self.inp)
        # Check output shape
        for o, e in zip(out.values(), self.expected_out_shapes):
            assert o.shape == e
        # Backward
        sum([o.mean() for o in out.values()]).backward()

    def test_intermediate_layer_getter_forward_backward(self):
        model = IntermediateLayerGetter(self.model, self.return_layers).eval()
        out = model(self.inp)
        # Check output shape
        for o, e in zip(out.values(), self.expected_out_shapes):
            assert o.shape == e
        # Backward
        sum([o.mean() for o in out.values()]).backward()

    def test_feature_extraction_methods_equivalence(self):
        ilg_model = IntermediateLayerGetter(
            self.model, self.return_layers).eval()
        fgn_model = build_feature_graph_net(self.model, self.return_layers)

        # Check that we have same parameters
        for (n1, p1), (n2, p2) in zip(ilg_model.named_parameters(),
                                      fgn_model.named_parameters()):
            self.assertEqual(n1, n2)
            self.assertTrue(p1.equal(p2))

        # And state_dict matches
        for (n1, p1), (n2, p2) in zip(ilg_model.state_dict().items(),
                                      fgn_model.state_dict().items()):
            self.assertEqual(n1, n2)
            self.assertTrue(p1.equal(p2))

        with torch.no_grad():
            ilg_out = ilg_model(self.inp)
            fgn_out = fgn_model(self.inp)

        self.assertEqual(ilg_out.keys(), fgn_out.keys())
        for k in ilg_out.keys():
            o1 = ilg_out[k]
            o2 = fgn_out[k]
            self.assertTrue(o1.equal(o2))

    def test_intermediate_layer_getter_scriptable_forward_backward(self):
        ilg_model = IntermediateLayerGetter(
            self.model, self.return_layers).eval()
        ilg_model = torch.jit.script(ilg_model)
        ilg_out = ilg_model(self.inp)
        sum([o.mean() for o in ilg_out.values()]).backward()

    def test_feature_graph_net_scriptable_forward_backward(self):
        fgn_model = build_feature_graph_net(self.model, self.return_layers)
        fgn_model = torch.jit.script(fgn_model)
        fgn_out = fgn_model(self.inp)
        sum([o.mean() for o in fgn_out.values()]).backward()

    def test_feature_graph_net_train_eval(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dropout = torch.nn.Dropout(p=1.)

            def forward(self, x):
                x = x.mean()
                x = self.dropout(x)  # dropout
                if self.training:
                    x += 100  # add
                else:
                    x *= 0  # mul
                x -= 0  # sub
                return x

        model = TestModel()

        train_return_nodes = ['dropout', 'add', 'sub']
        eval_return_nodes = ['dropout', 'mul', 'sub']

        def checks(model, mode):
            with torch.no_grad():
                out = model(torch.ones(10, 10))
            if mode == 'train':
                # Check that dropout is respected
                assert out['dropout'].item() == 0
                # Check that control flow dependent on training_mode is respected
                assert out['sub'].item() == 100
                assert 'add' in out
                assert 'mul' not in out
            elif mode == 'eval':
                # Check that dropout is respected
                assert out['dropout'].item() == 1
                # Check that control flow dependent on training_mode is respected
                assert out['sub'].item() == 0
                assert 'mul' in out
                assert 'add' not in out

        # Starting from train mode
        model.train()
        fgn_model = build_feature_graph_net(
            model, train_return_nodes=train_return_nodes,
            eval_return_nodes=eval_return_nodes)
        # Check that the models stay in their original training state
        assert model.training
        assert fgn_model.training
        # Check outputs
        checks(fgn_model, 'train')
        # Check outputs after switching to eval mode
        fgn_model.eval()
        checks(fgn_model, 'eval')

        # Starting from eval mode
        model.eval()
        fgn_model = build_feature_graph_net(
            model, train_return_nodes=train_return_nodes,
            eval_return_nodes=eval_return_nodes)
        # Check that the models stay in their original training state
        assert not model.training
        assert not fgn_model.training
        # Check outputs
        checks(fgn_model, 'eval')
        # Check outputs after switching to train mode
        fgn_model.train()
        checks(fgn_model, 'train')

    def test_feature_graph_net_leaf_module_and_function(self):
        class LeafModule(torch.nn.Module):
            def forward(self, x):
                # This would raise a TypeError if it were not in a leaf module
                int(x.shape[0])
                return torch.nn.functional.relu(x + 4)

        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 1, 3)
                self.leaf_module = LeafModule()

            def forward(self, x):
                leaf_function(x.shape[0])
                x = self.conv(x)
                return self.leaf_module(x)

        model = build_feature_graph_net(
            TestModule(), return_nodes=['leaf_module'],
            tracer_kwargs={'leaf_modules': [LeafModule],
                           'autowrap_functions': [leaf_function]})

        # Check that LeafModule is not in the list of nodes
        assert 'relu' not in [str(n) for n in model.graph.nodes]
        assert 'leaf_module' in [str(n) for n in model.graph.nodes]

        # Check forward
        out = model(self.inp)
        # And backward
        out['leaf_module'].mean().backward()
