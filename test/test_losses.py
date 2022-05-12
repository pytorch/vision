import pytest
import torch
from common_utils import cpu_and_gpu
from torchvision import ops


class TestGeneralizedBoxIouLoss:
    # We refer to original test: https://github.com/facebookresearch/fvcore/blob/main/tests/test_giou_loss.py
    @pytest.mark.parametrize("device", cpu_and_gpu())
    @pytest.mark.parametrize("dtype", [torch.float32, torch.half])
    def test_giou_loss(self, dtype, device) -> None:
        box1 = torch.tensor([-1, -1, 1, 1], dtype=dtype, device=device)
        box2 = torch.tensor([0, 0, 1, 1], dtype=dtype, device=device)
        box3 = torch.tensor([0, 1, 1, 2], dtype=dtype, device=device)
        box4 = torch.tensor([1, 1, 2, 2], dtype=dtype, device=device)

        box1s = torch.stack([box2, box2], dim=0)
        box2s = torch.stack([box3, box4], dim=0)

        def assert_giou_loss(box1, box2, expected_loss, reduction="none"):
            tol = 1e-3 if dtype is torch.half else 1e-5
            computed_loss = ops.generalized_box_iou_loss(box1, box2, reduction=reduction)
            expected_loss = torch.tensor(expected_loss, device=device)
            torch.testing.assert_close(computed_loss, expected_loss, rtol=tol, atol=tol)

        # Identical boxes should have loss of 0
        assert_giou_loss(box1, box1, 0.0)

        # quarter size box inside other box = IoU of 0.25
        assert_giou_loss(box1, box2, 0.75)

        # Two side by side boxes, area=union
        # IoU=0 and GIoU=0 (loss 1.0)
        assert_giou_loss(box2, box3, 1.0)

        # Two diagonally adjacent boxes, area=2*union
        # IoU=0 and GIoU=-0.5 (loss 1.5)
        assert_giou_loss(box2, box4, 1.5)

        # Test batched loss and reductions
        assert_giou_loss(box1s, box2s, 2.5, reduction="sum")
        assert_giou_loss(box1s, box2s, 1.25, reduction="mean")

    @pytest.mark.parametrize("device", cpu_and_gpu())
    @pytest.mark.parametrize("dtype", [torch.float32, torch.half])
    def test_empty_inputs(self, dtype, device) -> None:
        box1 = torch.randn([0, 4], dtype=dtype).requires_grad_()
        box2 = torch.randn([0, 4], dtype=dtype).requires_grad_()

        loss = ops.generalized_box_iou_loss(box1, box2, reduction="mean")
        loss.backward()

        tol = 1e-3 if dtype is torch.half else 1e-5
        torch.testing.assert_close(loss, torch.tensor(0.0), rtol=tol, atol=tol)
        assert box1.grad is not None, "box1.grad should not be None after backward is called"
        assert box2.grad is not None, "box2.grad should not be None after backward is called"

        loss = ops.generalized_box_iou_loss(box1, box2, reduction="none")
        assert loss.numel() == 0, "giou_loss for two empty box should be empty"


class TestCIOULoss:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.half])
    @pytest.mark.parametrize("device", cpu_and_gpu())
    def test_ciou_loss(self, dtype, device):
        box1 = torch.tensor([-1, -1, 1, 1], dtype=dtype, device=device)
        box2 = torch.tensor([0, 0, 1, 1], dtype=dtype, device=device)
        box3 = torch.tensor([0, 1, 1, 2], dtype=dtype, device=device)
        box4 = torch.tensor([1, 1, 2, 2], dtype=dtype, device=device)

        box1s = torch.stack([box2, box2], dim=0)
        box2s = torch.stack([box3, box4], dim=0)

        def assert_ciou_loss(box1, box2, expected_output, reduction="none"):

            output = ops.complete_box_iou_loss(box1, box2, reduction=reduction)
            # TODO: When passing the dtype, the torch.half test doesn't pass...
            expected_output = torch.tensor(expected_output, device=device)
            tol = 1e-5 if dtype != torch.half else 1e-3
            torch.testing.assert_close(output, expected_output, rtol=tol, atol=tol)

        assert_ciou_loss(box1, box1, 0.0)

        assert_ciou_loss(box1, box2, 0.8125)

        assert_ciou_loss(box1, box3, 1.1923)

        assert_ciou_loss(box1, box4, 1.2500)

        assert_ciou_loss(box1s, box2s, 1.2250, reduction="mean")
        assert_ciou_loss(box1s, box2s, 2.4500, reduction="sum")

    @pytest.mark.parametrize("device", cpu_and_gpu())
    @pytest.mark.parametrize("dtype", [torch.float32, torch.half])
    def test_empty_inputs(self, dtype, device) -> None:
        box1 = torch.randn([0, 4], dtype=dtype).requires_grad_()
        box2 = torch.randn([0, 4], dtype=dtype).requires_grad_()

        loss = ops.complete_box_iou_loss(box1, box2, reduction="mean")
        loss.backward()

        tol = 1e-3 if dtype is torch.half else 1e-5
        torch.testing.assert_close(loss, torch.tensor(0.0), rtol=tol, atol=tol)
        assert box1.grad is not None, "box1.grad should not be None after backward is called"
        assert box2.grad is not None, "box2.grad should not be None after backward is called"

        loss = ops.complete_box_iou_loss(box1, box2, reduction="none")
        assert loss.numel() == 0, "ciou_loss for two empty box should be empty"


@pytest.mark.parametrize("device", cpu_and_gpu())
@pytest.mark.parametrize("dtype", [torch.float32, torch.half])
def test_distance_iou_loss(dtype, device):
    box1 = torch.tensor([-1, -1, 1, 1], dtype=dtype, device=device)
    box2 = torch.tensor([0, 0, 1, 1], dtype=dtype, device=device)
    box3 = torch.tensor([0, 1, 1, 2], dtype=dtype, device=device)
    box4 = torch.tensor([1, 1, 2, 2], dtype=dtype, device=device)

    box1s = torch.stack(
        [box2, box2],
        dim=0,
    )
    box2s = torch.stack(
        [box3, box4],
        dim=0,
    )

    def assert_distance_iou_loss(box1, box2, expected_output, reduction="none"):
        output = ops.distance_box_iou_loss(box1, box2, reduction=reduction)
        # TODO: When passing the dtype, the torch.half fails as usual.
        expected_output = torch.tensor(expected_output, device=device)
        tol = 1e-5 if dtype != torch.half else 1e-3
        torch.testing.assert_close(output, expected_output, rtol=tol, atol=tol)

    assert_distance_iou_loss(box1, box1, 0.0)

    assert_distance_iou_loss(box1, box2, 0.8125)

    assert_distance_iou_loss(box1, box3, 1.1923)

    assert_distance_iou_loss(box1, box4, 1.2500)

    assert_distance_iou_loss(box1s, box2s, 1.2250, reduction="mean")
    assert_distance_iou_loss(box1s, box2s, 2.4500, reduction="sum")


@pytest.mark.parametrize("device", cpu_and_gpu())
@pytest.mark.parametrize("dtype", [torch.float32, torch.half])
def test_empty_distance_iou_inputs(dtype, device) -> None:
    box1 = torch.randn([0, 4], dtype=dtype, device=device).requires_grad_()
    box2 = torch.randn([0, 4], dtype=dtype, device=device).requires_grad_()

    loss = ops.distance_box_iou_loss(box1, box2, reduction="mean")
    loss.backward()

    tol = 1e-3 if dtype is torch.half else 1e-5
    torch.testing.assert_close(loss, torch.tensor(0.0, device=device), rtol=tol, atol=tol)
    assert box1.grad is not None, "box1.grad should not be None after backward is called"
    assert box2.grad is not None, "box2.grad should not be None after backward is called"

    loss = ops.distance_box_iou_loss(box1, box2, reduction="none")
    assert loss.numel() == 0, "diou_loss for two empty box should be empty"


if __name__ == "__main__":
    pytest.main([__file__])
