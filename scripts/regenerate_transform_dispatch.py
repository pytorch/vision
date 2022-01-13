import contextlib
import enum
import functools
import importlib
import inspect
import pathlib
import typing
import warnings
from copy import copy
from textwrap import dedent
from textwrap import indent as _indent
from typing import Any

import torchvision.prototype.transforms.functional as F
from torchvision import transforms
from torchvision.prototype import features
from torchvision.prototype.utils._internal import camel_to_snake_case

try:
    import yaml
except ModuleNotFoundError:
    raise ModuleNotFoundError() from None

HERE = pathlib.Path(__file__).parent

for module, name in [
    (features, "BoundingBoxFormat"),
    (transforms, "InterpolationMode"),
]:
    getattr(module, name).__module__ = module.__package__.rsplit(".", 1)[1]

ENUMS_MAP = {
    enum.__name__: enum
    for enum in [
        features.BoundingBoxFormat,
        transforms.InterpolationMode,
    ]
}


class ManualAnnotation:
    def __init__(self, repr):
        self.repr = repr

    def __repr__(self):
        return self.repr

    def __eq__(self, other):
        if not isinstance(other, ManualAnnotation):
            return NotImplemented

        return self.repr == other.repr


# TODO: typing module
FEATURE_SPECIFIC_DEFAULT = ManualAnnotation("FEATURE_SPECIFIC_DEFAULT")
GENERIC_FEATURE_TYPE = ManualAnnotation("T")


FUNCTIONAL_ROOT = HERE.parent / "torchvision" / "prototype" / "transforms" / "functional"


def main(config_path=FUNCTIONAL_ROOT / "dispatch.yaml", dispatch_path=FUNCTIONAL_ROOT / "_dispatch.py"):
    with open(config_path) as file:
        dispatch_config = yaml.load(file, yaml.Loader)

    functions = []

    for dispatcher_name, feature_type_configs in dispatch_config.items():
        try:
            feature_type_configs = validate_feature_type_configs(feature_type_configs)
            kernel_params, implementer_params = make_kernel_and_implementer_params(feature_type_configs)
            dispatcher_params = make_dispatcher_params(implementer_params)
        except Exception as error:
            raise type(error)(f"{error}\nThe error happened while working on dispatcher '{dispatcher_name}'") from error

        functions.append(DispatcherFunction(name=dispatcher_name, params=dispatcher_params))
        functions.extend(
            [
                ImplementerFunction(
                    dispatcher_name=dispatcher_name,
                    feature_type=feature_type,
                    params=implementer_params[feature_type],
                    output_type=config["output_type"],
                    kernel=config["kernel"],
                    kernel_params=kernel_params[feature_type],
                    conversion_map=config["meta_conversion"],
                    kernel_param_name_map=config["kwargs_overwrite"],
                    meta_overwrite=config["meta_overwrite"],
                )
                for feature_type, config in feature_type_configs.items()
            ]
        )

    # make_file_content(functions)
    with open(dispatch_path, "w") as file:
        file.write(ufmt_format(make_file_content(functions)))


def validate_feature_type_configs(feature_type_configs):
    def get_feature_type(name):
        try:
            return getattr(features, name)
        except AttributeError as error:
            # unknown feature type
            raise TypeError() from error

    try:
        feature_type_configs = {
            get_feature_type(feature_type_name): config for feature_type_name, config in feature_type_configs.items()
        }
    except AttributeError as error:
        # unknown feature type
        raise TypeError() from error

    for feature_type, config in feature_type_configs.items():
        unknown_keys = config.keys() - {
            "kernel",
            "meta_conversion",
            "kwargs_overwrite",
            "meta_overwrite",
            "output_type",
            "output_type",
        }
        if unknown_keys:
            raise KeyError(unknown_keys)

        try:
            config["kernel"] = getattr(F, config["kernel"])
        except KeyError:
            # no kernel provided
            raise
        except AttributeError:
            # kernel not accessible
            raise

        # check kernel signature with current transforms logic
        # better: check unary
        # signature = inspect.signature(config["kernel"])

        for key in ["meta_conversion", "kwargs_overwrite", "meta_overwrite"]:
            config.setdefault(key, dict())

        for meta_attr, value in config["meta_conversion"].copy().items():
            if meta_attr not in feature_type._META_ATTRS:
                raise KeyError(meta_attr)

            if not isinstance(value, str):
                continue

            parts = value.split(".")
            if len(parts) != 2:
                continue

            enum, member = parts

            with contextlib.suppress(KeyError):
                config["meta_conversion"][meta_attr] = ENUMS_MAP[enum][member]

        # TODO: remove
        try:
            config["output_type"] = getattr(features, config.get("output_type", feature_type.__name__))
        except AttributeError as error:
            # unknown feature type
            raise TypeError() from error

    # TODO: bunchify the individual configs
    return feature_type_configs


def make_kernel_and_implementer_params(feature_type_configs):
    kernel_params = {}
    implementer_params = {}
    for feature_type, config in feature_type_configs.items():
        kernel_params[feature_type] = [
            Parameter.from_regular(param) for param in list(inspect.signature(config["kernel"]).parameters.values())[1:]
        ]
        implementer_params[feature_type] = [
            Parameter(
                name=config["kwargs_overwrite"].get(kernel_param.name, kernel_param.name),
                kind=inspect.Parameter.KEYWORD_ONLY,
                default=kernel_param.default,
                annotation=kernel_param.annotation,
            )
            for kernel_param in kernel_params[feature_type]
            if not config["kwargs_overwrite"].get(kernel_param.name, "").startswith(".")
        ]
    return kernel_params, implementer_params


def make_dispatcher_params(implementer_params):
    # not using a set here to keep the order
    dispatcher_param_names = []
    for params in implementer_params.values():
        dispatcher_param_names.extend([param.name for param in params])
    dispatcher_param_names = unique(dispatcher_param_names)

    dispatcher_params = []
    need_kwargs_ignore = set()
    for name in dispatcher_param_names:
        dispatcher_param_candidates = set()
        for feature_type, params in implementer_params.items():
            params = {param.name: param for param in params}
            if name not in params:
                need_kwargs_ignore.add(feature_type)
                continue

            dispatcher_param_candidates.add(params[name])

        if len(dispatcher_param_candidates) == 1:
            dispatcher_params.append(copy(dispatcher_param_candidates.pop()))
            continue

        annotations = {param.annotation for param in dispatcher_param_candidates}
        if len(annotations) > 1:
            raise TypeError(
                f"Found multiple annotations for parameter `{name}`: "
                f"{', '.join([str(annotation) for annotation in annotations])}"
            )

        dispatcher_params.append(
            Parameter(
                name=name,
                kind=Parameter.KEYWORD_ONLY,
                default=FEATURE_SPECIFIC_DEFAULT,
                annotation=annotations.pop(),
            )
        )

    for feature_type in need_kwargs_ignore:
        implementer_params[feature_type].append(Parameter(name="_", kind=Parameter.VAR_KEYWORD, annotation=Any))

    return dispatcher_params


def make_file_content(functions):
    header = dedent(
        f"""
        # THIS FILE IS auto-generated!!

        from typing import Any, TypeVar, List, Optional

        import torch
        from torchvision import transforms
        from torchvision.prototype import features

        from .. import functional as F

        # TODO: add explanation
        # just a sentinel to have a default argument for parameters that have different default for features
        # the actual value is not used
        {FEATURE_SPECIFIC_DEFAULT} = object()

        {GENERIC_FEATURE_TYPE} = TypeVar("{GENERIC_FEATURE_TYPE}", bound=features.Feature)
        """
    ).strip()

    __all__ = "\n".join(
        (
            "__all__ = [",
            *[
                indent(f"{format_value(function.name)},")
                for function in functions
                if isinstance(function, DispatcherFunction)
            ],
            "]",
        )
    )
    return (
        "\n\n\n".join(
            (
                header,
                __all__,
                *[str(function) for function in functions],
            )
        )
        + "\n"
    )


class Parameter(inspect.Parameter):
    @classmethod
    def from_regular(cls, param):
        return cls(param.name, param.kind, default=param.default, annotation=param.annotation)

    def __str__(self):
        @contextlib.contextmanager
        def tmp_override(**tmp_values):
            values = {name: getattr(self, name) for name in tmp_values}
            for name, tmp_value in tmp_values.items():
                setattr(self, f"_{name}", tmp_value)
            try:
                yield
            finally:
                for name, value in values.items():
                    setattr(self, f"_{name}", value)

        tmp_values = dict()

        if isinstance(self.default, enum.Enum):
            tmp_values["default"] = ManualAnnotation(format_value(self.default))

        if (
            hasattr(self.annotation, "__origin__")
            and self.annotation.__origin__ is typing.Union
            and type(None) in self.annotation.__args__
        ):
            tmp_values["annotation"] = ManualAnnotation(
                f"Optional[{', '.join(inspect.formatannotation(arg) for arg in self.annotation.__args__ if arg is not type(None))}]"
            )

        with tmp_override(**tmp_values):
            return super().__str__()


class Signature(inspect.Signature):
    def __str__(self):
        parts = super().__str__().split(repr(FEATURE_SPECIFIC_DEFAULT))
        return f"{FEATURE_SPECIFIC_DEFAULT},  # type: ignore[assignment]\n".join(
            [
                parts[0],
                *[part.lstrip(",") for part in parts[1:]],
            ]
        )


class Function:
    def __init__(self, *, decorator=None, name, signature, docstring=None, body=("pass",)):
        self.decorator = decorator
        self.name = name
        self.signature = signature
        self.docstring = docstring
        self.body = body

    def __str__(self):
        lines = []
        if self.decorator:
            lines.append(f"@{self.decorator}")
        lines.append(f"def {self.name}{self.signature}:")
        if self.docstring:
            lines.append(indent('"""' + self.docstring + '"""'))
        lines.extend([indent(line) for line in self.body])
        return "\n".join(lines)


class DispatcherFunction(Function):
    def __init__(self, *, name, params, input_name="input"):
        for param in params:
            param._kind = Parameter.KEYWORD_ONLY
        signature = Signature(
            parameters=[
                Parameter(
                    name=input_name,
                    kind=Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=GENERIC_FEATURE_TYPE,
                ),
                *params,
            ],
            return_annotation=GENERIC_FEATURE_TYPE,
        )
        super().__init__(
            decorator="F.utils.Dispatcher",
            name=name,
            signature=signature,
            docstring="ADDME",
        )


class ImplementerFunction(Function):
    def __init__(
        self,
        *,
        dispatcher_name,
        feature_type,
        params,
        output_type,
        kernel,
        kernel_params,
        conversion_map,
        kernel_param_name_map,
        meta_overwrite,
    ):
        feature_type_usage = ManualAnnotation(f"features.{feature_type.__name__}")
        output_type_usage = ManualAnnotation(f"features.{output_type.__name__}")

        body = ["converted_input = input.data", ""]

        output_conversions = []
        for attr, intermediate_value in conversion_map.items():

            def make_conversion_call(input, old, new):
                return (
                    f"F.convert_{attr}_{camel_to_snake_case(feature_type.__name__)}("
                    f"{input}, old_{attr}={old}, new_{attr}={new}"
                    f")"
                )

            input_name = f"input.{attr}"
            intermediate_name = f"intermediate_{attr}"
            body.extend(
                [
                    f"{intermediate_name} = {format_value(intermediate_value)}",
                    f"converted_input = {make_conversion_call('converted_input', input_name, intermediate_name)}",
                    "",
                ]
            )
            output_conversions = [f"output = {make_conversion_call('output', intermediate_name, input_name)}"]

        kernel_call = self._make_kernel_call(
            kernel=kernel,
            kernel_params=kernel_params,
            kernel_param_name_map=kernel_param_name_map,
        )
        body.extend(
            [
                f"output = {kernel_call}",
                *reversed(output_conversions),
                "",
            ]
        )

        feature_type_wrapper = self._make_feature_type_wrapper(
            feature_type_usage=feature_type_usage,
            output_type_usage=output_type_usage,
            meta_overwrite=meta_overwrite,
        )
        body.append(f"return {feature_type_wrapper}")

        super().__init__(
            decorator=f"{dispatcher_name}.implements({feature_type_usage})",
            name=f"_{dispatcher_name}_{camel_to_snake_case(feature_type.__name__)}",
            signature=Signature(
                parameters=[
                    Parameter(
                        name="input",
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=feature_type_usage,
                    ),
                    *params,
                ],
                return_annotation=output_type_usage,
            ),
            body=body,
        )

    def _make_kernel_call(
        self,
        *,
        kernel,
        kernel_params,
        kernel_param_name_map,
    ):
        call_args = ["converted_input"]
        for param in kernel_params:
            dispatcher_param_name = kernel_param_name_map.get(param.name, param.name)
            if dispatcher_param_name.startswith("."):
                dispatcher_param_name = f"input{dispatcher_param_name}"
            call_args.append(f"{param.name}={dispatcher_param_name}")
        return f"F.{kernel.__name__}({', '.join(call_args)})"

    def _make_feature_type_wrapper(self, *, feature_type_usage, output_type_usage, meta_overwrite):
        retain_type = output_type_usage == feature_type_usage

        call_args = []
        if retain_type:
            method = ".new_like("
            call_args.append("input")
        else:
            method = "("

        call_args.append("output")
        call_args.extend(
            f"{meta_name}={dispatcher_param_name}" for meta_name, dispatcher_param_name in meta_overwrite.items()
        )

        return f"{output_type_usage}{method}{', '.join(call_args)})"


def ufmt_format(content):
    try:
        import ufmt
    except ModuleNotFoundError:
        return content

    with open(HERE.parent / ".pre-commit-config.yaml") as file:
        repo = next(
            repo for repo in yaml.load(file, yaml.Loader)["repos"] for hook in repo["hooks"] if hook["id"] == "ufmt"
        )

    expected_versions = {ufmt: repo["rev"].replace("v", "")}
    for dependency in repo["hooks"][0]["additional_dependencies"]:
        name, version = [item.strip() for item in dependency.split("==")]
        expected_versions[importlib.import_module(name)] = version

    for module, expected_version in expected_versions.items():
        if module.__version__ != expected_version:
            warnings.warn("foo")

    from ufmt.core import make_black_config
    from usort.config import Config as UsortConfig

    black_config = make_black_config(HERE)
    usort_config = UsortConfig.find(HERE)

    return ufmt.ufmt_string(path=HERE, content=content, usort_config=usort_config, black_config=black_config)


def indent(text, level=1):
    return _indent(text, prefix=" " * (level * 4))


def format_value(value):
    if isinstance(value, str):
        return f'"{value}"'
    elif isinstance(value, enum.Enum):
        return f"{value.__module__}.{type(value).__name__}.{value.name}"
    else:
        return repr(value)


def unique(seq):
    return functools.reduce(
        lambda unique_list, item: (unique_list.append(item) or unique_list) if item not in unique_list else unique_list,
        seq,
        [],
    )


if __name__ == "__main__":
    main()
