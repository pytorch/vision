import pytest
import torch
import torch.nn.functional as F
from common_utils import cpu_and_gpu
from torchvision import ops


def get_boxes(dtype, device):
    box1 = torch.tensor([-1, -1, 1, 1], dtype=dtype, device=device)
    box2 = torch.tensor([0, 0, 1, 1], dtype=dtype, device=device)
    box3 = torch.tensor([0, 1, 1, 2], dtype=dtype, device=device)
    box4 = torch.tensor([1, 1, 2, 2], dtype=dtype, device=device)

    box1s = torch.stack([box2, box2], dim=0)
    box2s = torch.stack([box3, box4], dim=0)

    return box1, box2, box3, box4, box1s, box2s


def assert_iou_loss(iou_fn, box1, box2, expected_loss, dtype, device, reduction="none"):
    computed_loss = iou_fn(box1, box2, reduction=reduction)
    expected_loss = torch.tensor(expected_loss, device=device)
    torch.testing.assert_close(computed_loss, expected_loss)


def assert_empty_loss(iou_fn, dtype, device):
    box1 = torch.randn([0, 4], dtype=dtype, device=device).requires_grad_()
    box2 = torch.randn([0, 4], dtype=dtype, device=device).requires_grad_()
    loss = iou_fn(box1, box2, reduction="mean")
    loss.backward()
    torch.testing.assert_close(loss, torch.tensor(0.0, device=device))
    assert box1.grad is not None, "box1.grad should not be None after backward is called"
    assert box2.grad is not None, "box2.grad should not be None after backward is called"
    loss = iou_fn(box1, box2, reduction="none")
    assert loss.numel() == 0, f"{str(iou_fn)} for two empty box should be empty"


class TestGeneralizedBoxIouLoss:
    # We refer to original test: https://github.com/facebookresearch/fvcore/blob/main/tests/test_giou_loss.py
    @pytest.mark.parametrize("device", cpu_and_gpu())
    @pytest.mark.parametrize("dtype", [torch.float32, torch.half])
    def test_giou_loss(self, dtype, device):

        box1, box2, box3, box4, box1s, box2s = get_boxes(dtype, device)

        # Identical boxes should have loss of 0
        assert_iou_loss(ops.generalized_box_iou_loss, box1, box1, 0.0, dtype=dtype, device=device)

        # quarter size box inside other box = IoU of 0.25
        assert_iou_loss(ops.generalized_box_iou_loss, box1, box2, 0.75, dtype=dtype, device=device)

        # Two side by side boxes, area=union
        # IoU=0 and GIoU=0 (loss 1.0)
        assert_iou_loss(ops.generalized_box_iou_loss, box2, box3, 1.0, dtype=dtype, device=device)

        # Two diagonally adjacent boxes, area=2*union
        # IoU=0 and GIoU=-0.5 (loss 1.5)
        assert_iou_loss(ops.generalized_box_iou_loss, box2, box4, 1.5, dtype=dtype, device=device)

        # Test batched loss and reductions
        assert_iou_loss(ops.generalized_box_iou_loss, box1s, box2s, 2.5, dtype=dtype, device=device, reduction="sum")
        assert_iou_loss(ops.generalized_box_iou_loss, box1s, box2s, 1.25, dtype=dtype, device=device, reduction="mean")

    @pytest.mark.parametrize("device", cpu_and_gpu())
    @pytest.mark.parametrize("dtype", [torch.float32, torch.half])
    def test_empty_inputs(self, dtype, device):
        assert_empty_loss(ops.generalized_box_iou_loss, dtype, device)


class TestCIOULoss:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.half])
    @pytest.mark.parametrize("device", cpu_and_gpu())
    def test_ciou_loss(self, dtype, device):
        box1, box2, box3, box4, box1s, box2s = get_boxes(dtype, device)

        assert_iou_loss(ops.complete_box_iou_loss, box1, box1, 0.0, dtype=dtype, device=device)
        assert_iou_loss(ops.complete_box_iou_loss, box1, box2, 0.8125, dtype=dtype, device=device)
        assert_iou_loss(ops.complete_box_iou_loss, box1, box3, 1.1923, dtype=dtype, device=device)
        assert_iou_loss(ops.complete_box_iou_loss, box1, box4, 1.2500, dtype=dtype, device=device)
        assert_iou_loss(ops.complete_box_iou_loss, box1s, box2s, 1.2250, dtype=dtype, device=device, reduction="mean")
        assert_iou_loss(ops.complete_box_iou_loss, box1s, box2s, 2.4500, dtype=dtype, device=device, reduction="sum")

    @pytest.mark.parametrize("device", cpu_and_gpu())
    @pytest.mark.parametrize("dtype", [torch.float32, torch.half])
    def test_empty_inputs(self, dtype, device):
        assert_empty_loss(ops.complete_box_iou_loss, dtype, device)


class TestDIouLoss:
    @pytest.mark.parametrize("device", cpu_and_gpu())
    @pytest.mark.parametrize("dtype", [torch.float32, torch.half])
    def test_distance_iou_loss(self, dtype, device):
        box1, box2, box3, box4, box1s, box2s = get_boxes(dtype, device)

        assert_iou_loss(ops.distance_box_iou_loss, box1, box1, 0.0, dtype=dtype, device=device)
        assert_iou_loss(ops.distance_box_iou_loss, box1, box2, 0.8125, dtype=dtype, device=device)
        assert_iou_loss(ops.distance_box_iou_loss, box1, box3, 1.1923, dtype=dtype, device=device)
        assert_iou_loss(ops.distance_box_iou_loss, box1, box4, 1.2500, dtype=dtype, device=device)
        assert_iou_loss(ops.distance_box_iou_loss, box1s, box2s, 1.2250, dtype=dtype, device=device, reduction="mean")
        assert_iou_loss(ops.distance_box_iou_loss, box1s, box2s, 2.4500, dtype=dtype, device=device, reduction="sum")

    @pytest.mark.parametrize("device", cpu_and_gpu())
    @pytest.mark.parametrize("dtype", [torch.float32, torch.half])
    def test_empty_distance_iou_inputs(self, dtype, device):
        assert_empty_loss(ops.distance_box_iou_loss, dtype, device)


class TestFocalLoss:
    def _generate_diverse_input_target_pair(self, shape=(5, 2), **kwargs):
        def logit(p):
            return torch.log(p / (1 - p))

        def generate_tensor_with_range_type(shape, range_type, **kwargs):
            if range_type != "random_binary":
                low, high = {
                    "small": (0.0, 0.2),
                    "big": (0.8, 1.0),
                    "zeros": (0.0, 0.0),
                    "ones": (1.0, 1.0),
                    "random": (0.0, 1.0),
                }[range_type]
                return torch.testing.make_tensor(shape, low=low, high=high, **kwargs)
            else:
                return torch.randint(0, 2, shape, **kwargs)

        # This function will return inputs and targets with shape: (shape[0]*9, shape[1])
        inputs = []
        targets = []
        for input_range_type, target_range_type in [
            ("small", "zeros"),
            ("small", "ones"),
            ("small", "random_binary"),
            ("big", "zeros"),
            ("big", "ones"),
            ("big", "random_binary"),
            ("random", "zeros"),
            ("random", "ones"),
            ("random", "random_binary"),
        ]:
            inputs.append(logit(generate_tensor_with_range_type(shape, input_range_type, **kwargs)))
            targets.append(generate_tensor_with_range_type(shape, target_range_type, **kwargs))

        return torch.cat(inputs), torch.cat(targets)

    @pytest.mark.parametrize("alpha", [-1.0, 0.0, 0.58, 1.0])
    @pytest.mark.parametrize("gamma", [0, 2])
    @pytest.mark.parametrize("device", cpu_and_gpu())
    @pytest.mark.parametrize("dtype", [torch.float32, torch.half])
    @pytest.mark.parametrize("seed", [0, 1])
    def test_correct_ratio(self, alpha, gamma, device, dtype, seed):
        if device == "cpu" and dtype is torch.half:
            pytest.skip("Currently torch.half is not fully supported on cpu")
        # For testing the ratio with manual calculation, we require the reduction to be "none"
        reduction = "none"
        torch.random.manual_seed(seed)
        inputs, targets = self._generate_diverse_input_target_pair(dtype=dtype, device=device)
        focal_loss = ops.sigmoid_focal_loss(inputs, targets, gamma=gamma, alpha=alpha, reduction=reduction)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction=reduction)

        assert torch.all(
            focal_loss <= ce_loss
        ), "focal loss must be less or equal to cross entropy loss with same input"

        loss_ratio = (focal_loss / ce_loss).squeeze()
        prob = torch.sigmoid(inputs)
        p_t = prob * targets + (1 - prob) * (1 - targets)
        correct_ratio = (1.0 - p_t) ** gamma
        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            correct_ratio = correct_ratio * alpha_t

        tol = 1e-3 if dtype is torch.half else 1e-5
        torch.testing.assert_close(correct_ratio, loss_ratio, atol=tol, rtol=tol)

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    @pytest.mark.parametrize("device", cpu_and_gpu())
    @pytest.mark.parametrize("dtype", [torch.float32, torch.half])
    @pytest.mark.parametrize("seed", [2, 3])
    def test_equal_ce_loss(self, reduction, device, dtype, seed):
        if device == "cpu" and dtype is torch.half:
            pytest.skip("Currently torch.half is not fully supported on cpu")
        # focal loss should be equal ce_loss if alpha=-1 and gamma=0
        alpha = -1
        gamma = 0
        torch.random.manual_seed(seed)
        inputs, targets = self._generate_diverse_input_target_pair(dtype=dtype, device=device)
        inputs_fl = inputs.clone().requires_grad_()
        targets_fl = targets.clone()
        inputs_ce = inputs.clone().requires_grad_()
        targets_ce = targets.clone()
        focal_loss = ops.sigmoid_focal_loss(inputs_fl, targets_fl, gamma=gamma, alpha=alpha, reduction=reduction)
        ce_loss = F.binary_cross_entropy_with_logits(inputs_ce, targets_ce, reduction=reduction)

        tol = 1e-3 if dtype is torch.half else 1e-5
        torch.testing.assert_close(focal_loss, ce_loss, atol=tol, rtol=tol)

        focal_loss.backward()
        ce_loss.backward()
        torch.testing.assert_close(inputs_fl.grad, inputs_ce.grad)

    @pytest.mark.parametrize("alpha", [-1.0, 0.0, 0.58, 1.0])
    @pytest.mark.parametrize("gamma", [0, 2])
    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    @pytest.mark.parametrize("device", cpu_and_gpu())
    @pytest.mark.parametrize("dtype", [torch.float32, torch.half])
    @pytest.mark.parametrize("seed", [4, 5])
    def test_jit(self, alpha, gamma, reduction, device, dtype, seed):
        if device == "cpu" and dtype is torch.half:
            pytest.skip("Currently torch.half is not fully supported on cpu")
        script_fn = torch.jit.script(ops.sigmoid_focal_loss)
        torch.random.manual_seed(seed)
        inputs, targets = self._generate_diverse_input_target_pair(dtype=dtype, device=device)
        focal_loss = ops.sigmoid_focal_loss(inputs, targets, gamma=gamma, alpha=alpha, reduction=reduction)
        if device == "cpu":
            scripted_focal_loss = script_fn(inputs, targets, gamma=gamma, alpha=alpha, reduction=reduction)
        else:
            with torch.jit.fuser("fuser2"):
                # Use fuser2 to prevent a bug on fuser: https://github.com/pytorch/pytorch/issues/75476
                # We may remove this condition once the bug is resolved
                scripted_focal_loss = script_fn(inputs, targets, gamma=gamma, alpha=alpha, reduction=reduction)

        tol = 1e-3 if dtype is torch.half else 1e-5
        torch.testing.assert_close(focal_loss, scripted_focal_loss, rtol=tol, atol=tol)


if __name__ == "__main__":
    pytest.main([__file__])
