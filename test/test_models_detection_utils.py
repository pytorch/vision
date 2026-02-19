import copy

import pytest
import torch
from common_utils import assert_equal
from torchvision.models.detection import _utils, backbone_utils
from torchvision.models.detection.transform import GeneralizedRCNNTransform


class TestModelsDetectionUtils:
    def test_balanced_positive_negative_sampler(self):
        sampler = _utils.BalancedPositiveNegativeSampler(4, 0.25)
        # keep all 6 negatives first, then add 3 positives, last two are ignore
        matched_idxs = [torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, -1, -1])]
        pos, neg = sampler(matched_idxs)
        # we know the number of elements that should be sampled for the positive (1)
        # and the negative (3), and their location. Let's make sure that they are
        # there
        assert pos[0].sum() == 1
        assert pos[0][6:9].sum() == 1
        assert neg[0].sum() == 3
        assert neg[0][0:6].sum() == 3

    def test_box_linear_coder(self):
        box_coder = _utils.BoxLinearCoder(normalize_by_size=True)
        # Generate a random 10x4 boxes tensor, with coordinates < 50.
        boxes = torch.rand(10, 4) * 50
        boxes.clamp_(min=1.0)  # tiny boxes cause numerical instability in box regression
        boxes[:, 2:] += boxes[:, :2]

        proposals = torch.tensor([0, 0, 101, 101] * 10).reshape(10, 4).float()

        rel_codes = box_coder.encode(boxes, proposals)
        pred_boxes = box_coder.decode(rel_codes, boxes)
        torch.allclose(proposals, pred_boxes)

    @pytest.mark.parametrize("train_layers, exp_froz_params", [(0, 53), (1, 43), (2, 24), (3, 11), (4, 1), (5, 0)])
    def test_resnet_fpn_backbone_frozen_layers(self, train_layers, exp_froz_params):
        # we know how many initial layers and parameters of the network should
        # be frozen for each trainable_backbone_layers parameter value
        # i.e. all 53 params are frozen if trainable_backbone_layers=0
        # ad first 24 params are frozen if trainable_backbone_layers=2
        model = backbone_utils.resnet_fpn_backbone("resnet50", weights=None, trainable_layers=train_layers)
        # boolean list that is true if the param at that index is frozen
        is_frozen = [not parameter.requires_grad for _, parameter in model.named_parameters()]
        # check that expected initial number of layers are frozen
        assert all(is_frozen[:exp_froz_params])

    def test_validate_resnet_inputs_detection(self):
        # default number of backbone layers to train
        ret = backbone_utils._validate_trainable_layers(
            is_trained=True, trainable_backbone_layers=None, max_value=5, default_value=3
        )
        assert ret == 3
        # can't go beyond 5
        with pytest.raises(ValueError, match=r"Trainable backbone layers should be in the range"):
            ret = backbone_utils._validate_trainable_layers(
                is_trained=True, trainable_backbone_layers=6, max_value=5, default_value=3
            )
        # if not trained, should use all trainable layers and warn
        with pytest.warns(UserWarning):
            ret = backbone_utils._validate_trainable_layers(
                is_trained=False, trainable_backbone_layers=0, max_value=5, default_value=3
            )
        assert ret == 5

    def test_transform_copy_targets(self):
        transform = GeneralizedRCNNTransform(300, 500, torch.zeros(3), torch.ones(3))
        image = [torch.rand(3, 200, 300), torch.rand(3, 200, 200)]
        targets = [{"boxes": torch.rand(3, 4)}, {"boxes": torch.rand(2, 4)}]
        targets_copy = copy.deepcopy(targets)
        out = transform(image, targets)  # noqa: F841
        assert_equal(targets[0]["boxes"], targets_copy[0]["boxes"])
        assert_equal(targets[1]["boxes"], targets_copy[1]["boxes"])

    def test_not_float_normalize(self):
        transform = GeneralizedRCNNTransform(300, 500, torch.zeros(3), torch.ones(3))
        image = [torch.randint(0, 255, (3, 200, 300), dtype=torch.uint8)]
        targets = [{"boxes": torch.rand(3, 4)}]
        with pytest.raises(TypeError):
            out = transform(image, targets)  # noqa: F841


class TestModelsDetectionUtilsExport:
    """Export tests for detection utility components."""

    @pytest.mark.parametrize("strict", [False, True])
    def test_box_coder_decode_export(self, strict):
        """Exported BoxCoder.decode should match eager, using the same pattern
        as test_box_linear_coder."""
        from torch.export import export

        class BoxCoderDecodeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.box_coder = _utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

            def forward(self, rel_codes, boxes):
                return self.box_coder.decode(rel_codes, [boxes])

        torch.manual_seed(0)
        boxes = torch.rand(10, 4) * 50
        boxes[:, 2:] += boxes[:, :2]
        rel_codes = torch.randn(10, 4)

        model = BoxCoderDecodeModule()
        with torch.no_grad():
            ep = export(model, (rel_codes, boxes), strict=strict)

        eager_out = model(rel_codes, boxes)
        export_out = ep.module()(rel_codes, boxes)
        torch.testing.assert_close(eager_out, export_out, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("strict", [False, True])
    def test_box_coder_decode_multi_class_export(self, strict):
        """BoxCoder.decode with multi-class box regression (num_classes * 4 columns)."""
        from torch.export import export

        num_classes = 5

        class BoxCoderDecodeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.box_coder = _utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

            def forward(self, rel_codes, boxes):
                return self.box_coder.decode(rel_codes, [boxes])

        torch.manual_seed(0)
        boxes = torch.rand(10, 4) * 50
        boxes[:, 2:] += boxes[:, :2]
        rel_codes = torch.randn(10, num_classes * 4)

        model = BoxCoderDecodeModule()
        with torch.no_grad():
            ep = export(model, (rel_codes, boxes), strict=strict)

        eager_out = model(rel_codes, boxes)
        export_out = ep.module()(rel_codes, boxes)
        torch.testing.assert_close(eager_out, export_out, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("strict", [False, True])
    def test_transform_batch_images_export(self, strict):
        """Exported batch_images should match eager, using the
        same inputs as test_transform_copy_targets."""
        from torch.export import Dim, export

        transform = GeneralizedRCNNTransform(300, 500, torch.zeros(3), torch.ones(3))

        class BatchImagesModule(torch.nn.Module):
            def __init__(self, t):
                super().__init__()
                self.size_divisible = t.size_divisible

            def forward(self, image):
                return transform.batch_images([image], self.size_divisible)

        model = BatchImagesModule(transform)
        model.eval()

        # batch_images pads to stride-32 multiples, creating // guards
        # that require constrained dims (32*k aligned)
        _h = Dim("_h", min=4, max=25)
        _w = Dim("_w", min=4, max=25)
        h = 32 * _h
        w = 32 * _w

        x = torch.rand(3, 192, 256)  # 32-aligned example
        with torch.no_grad():
            ep = export(
                model, (x,),
                dynamic_shapes={"image": {1: h, 2: w}},
                strict=strict,
            )

        # Same input
        eager_out = model(x)
        export_out = ep.module()(x)
        torch.testing.assert_close(eager_out, export_out, atol=1e-6, rtol=1e-6)

        # Different 32-aligned size
        x2 = torch.rand(3, 160, 320)
        eager_out2 = model(x2)
        export_out2 = ep.module()(x2)
        torch.testing.assert_close(eager_out2, export_out2, atol=1e-6, rtol=1e-6)



if __name__ == "__main__":
    pytest.main([__file__])
