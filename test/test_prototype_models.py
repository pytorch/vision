import pytest
import test_models as TM
import torch
from common_utils import cpu_and_cuda, set_rng_seed
from torchvision.prototype import models


@pytest.mark.parametrize("model_fn", (models.depth.stereo.raft_stereo_base,))
@pytest.mark.parametrize("model_mode", ("standard", "scripted"))
@pytest.mark.parametrize("dev", cpu_and_cuda())
def test_raft_stereo(model_fn, model_mode, dev):
    # A simple test to make sure the model can do forward pass and jit scriptable
    set_rng_seed(0)

    # Use corr_pyramid and corr_block with smaller num_levels and radius to prevent nan output
    # get the idea from test_models.test_raft
    corr_pyramid = models.depth.stereo.raft_stereo.CorrPyramid1d(num_levels=2)
    corr_block = models.depth.stereo.raft_stereo.CorrBlock1d(num_levels=2, radius=2)
    model = model_fn(corr_pyramid=corr_pyramid, corr_block=corr_block).eval().to(dev)

    if model_mode == "scripted":
        model = torch.jit.script(model)

    img1 = torch.rand(1, 3, 64, 64).to(dev)
    img2 = torch.rand(1, 3, 64, 64).to(dev)
    num_iters = 3

    preds = model(img1, img2, num_iters=num_iters)
    depth_pred = preds[-1]

    assert len(preds) == num_iters, "Number of predictions should be the same as model.num_iters"

    assert depth_pred.shape == torch.Size(
        [1, 1, 64, 64]
    ), f"The output shape of depth_pred should be [1, 1, 64, 64] but instead it is {preds[0].shape}"

    # Test against expected file output
    TM._assert_expected(depth_pred, name=model_fn.__name__, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("model_fn", (models.depth.stereo.crestereo_base,))
@pytest.mark.parametrize("model_mode", ("standard", "scripted"))
@pytest.mark.parametrize("dev", cpu_and_cuda())
def test_crestereo(model_fn, model_mode, dev):
    set_rng_seed(0)

    model = model_fn().eval().to(dev)

    if model_mode == "scripted":
        model = torch.jit.script(model)

    img1 = torch.rand(1, 3, 64, 64).to(dev)
    img2 = torch.rand(1, 3, 64, 64).to(dev)
    iterations = 3

    preds = model(img1, img2, flow_init=None, num_iters=iterations)
    disparity_pred = preds[-1]

    # all the pyramid levels except the highest res make only half the number of iterations
    expected_iterations = (iterations // 2) * (len(model.resolutions) - 1)
    expected_iterations += iterations
    assert (
        len(preds) == expected_iterations
    ), "Number of predictions should be the number of iterations multiplied by the number of pyramid levels"

    assert disparity_pred.shape == torch.Size(
        [1, 2, 64, 64]
    ), f"Predicted disparity should have the same spatial shape as the input. Inputs shape {img1.shape[2:]}, Prediction shape {disparity_pred.shape[2:]}"

    assert all(
        d.shape == torch.Size([1, 2, 64, 64]) for d in preds
    ), "All predicted disparities are expected to have the same shape"

    # test a backward pass with a dummy loss as well
    preds = torch.stack(preds, dim=0)
    targets = torch.ones_like(preds, requires_grad=False)
    loss = torch.nn.functional.mse_loss(preds, targets)

    try:
        loss.backward()
    except Exception as e:
        assert False, f"Backward pass failed with an unexpected exception: {e.__class__.__name__} {e}"

    TM._assert_expected(disparity_pred, name=model_fn.__name__, atol=1e-2, rtol=1e-2)
