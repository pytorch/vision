import functools

import torch
import torch.overrides


def dispatches(dispatch_fn):
    @functools.wraps(dispatch_fn)
    def wrapper(input, *args, **kwargs):
        if torch.overrides.has_torch_function_unary(input):
            return torch.overrides.handle_torch_function(dispatch_fn, (input,), input, *args, **kwargs)

        raise RuntimeError()

    return wrapper


def implements(dispatch_fn, feature_type):
    def wrapper(kernel_fn):
        feature_type._KERNELS[dispatch_fn.__wrapped__] = kernel_fn
        return kernel_fn

    return wrapper
