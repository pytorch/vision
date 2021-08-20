import torch
from torch import nn, Tensor


def stochastic_depth(input: Tensor, p: float, mode: str, training: bool = True) -> Tensor:
    """
    Implements the Stochastic Depth from `"Deep Networks with Stochastic Depth"
    <https://arxiv.org/abs/1603.09382>`_ used for randomly dropping residual
    branches of residual architectures.

    Args:
        input (Tensor[N, ...]): The input tensor or arbitrary dimensions with the first one
                    being its batch i.e. a batch with ``N`` rows.
        p (float): probability of the input to be zeroed.
        mode (str): ``"batch"`` or ``"row"``.
                    ``"batch"`` randomly zeroes the entire input, ``"row"`` zeroes
                    randomly selected rows from the batch.
        training: apply stochastic depth if is ``True``. Default: ``True``

    Returns:
        Tensor[N, ...]: The randomly zeroed tensor.
    """
    if p < 0.0 or p > 1.0:
        raise ValueError("drop probability has to be between 0 and 1, but got {}".format(p))
    if not training or p == 0.0:
        return input

    survival_rate = 1.0 - p
    if mode == "batch":
        keep = torch.rand(size=(1, ), dtype=input.dtype, device=input.device) < survival_rate
    elif mode == "row":
        keep = torch.rand(size=(input.size(0),), dtype=input.dtype, device=input.device) < survival_rate
        keep = keep[(None, ) * (input.ndim - 1)].T
    else:
        raise ValueError("mode has to be either 'batch' or 'row', but got {}".format(mode))
    return input / survival_rate * keep


class StochasticDepth(nn.Module):
    """
    See :func:`stochastic_depth`.
    """
    def __init__(self, p: float, mode: str) -> None:
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, input: Tensor) -> Tensor:
        return stochastic_depth(input, self.p, self.mode, self.training)

    def __repr__(self) -> str:
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'p=' + str(self.p)
        tmpstr += ', mode=' + str(self.mode)
        tmpstr += ')'
        return tmpstr
