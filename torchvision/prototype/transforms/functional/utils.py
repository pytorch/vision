import types


def _from_legacy_kernel(legacy_kernel, new_name=None):
    kernel = types.FunctionType(
        code=legacy_kernel.__code__,
        globals=legacy_kernel.__globals__,
        name=new_name or f"{legacy_kernel.__name__}_image",
        argdefs=legacy_kernel.__defaults__,
        closure=legacy_kernel.__closure__,
    )
    kernel.__annotations__ = legacy_kernel.__annotations__
    return kernel
