import contextlib
import enum
import importlib
import inspect
import pathlib
import re
import sys
import typing
import warnings
from copy import copy
from typing import Any

import torchvision.prototype.transforms.functional as F
from torchvision import transforms
from torchvision.prototype import features

try:
    import yaml
except ModuleNotFoundError:
    raise ModuleNotFoundError()


ENUMS = [
    (features, ["BoundingBoxFormat", "ColorSpace"]),
    (transforms, ["InterpolationMode"]),
]

ENUMS_MAP = {name: getattr(module, name) for module, names in ENUMS for name in names}

META_CONVERTER_MAP = {
    (features.Image, "color_space"): F.convert_color_space,
    (features.BoundingBox, "format"): F.convert_bounding_box_format,
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
FEATURE_SPECIFIC_PARAM = ManualAnnotation("Dispatcher.FEATURE_SPECIFIC_PARAM")
FEATURE_SPECIFIC_DEFAULT = ManualAnnotation("FEATURE_SPECIFIC_DEFAULT")
GENERIC_FEATURE_TYPE = ManualAnnotation("T")


def main(dispatch_config):
    functions = []
    for dispatcher_name, feature_type_configs in dispatch_config.items():
        try:
            feature_type_configs = validate_feature_type_configs(feature_type_configs)
            kernel_params, implementer_params = make_kernel_and_implementer_params(feature_type_configs)
            dispatcher_params = make_dispatcher_params(implementer_params)
        except Exception as error:
            raise RuntimeError(
                f"while working on dispatcher '{dispatcher_name}' the following error was raised:\n\n"
                f"{type(error).__name__}: {error}"
            ) from None

        functions.append(DispatcherFunction(name=dispatcher_name, params=dispatcher_params))
        functions.extend(
            [
                ImplementerFunction(
                    dispatcher_name=dispatcher_name,
                    feature_type=feature_type,
                    params=implementer_params[feature_type],
                    pil_kernel=config.get("pil_kernel"),
                    kernel=config["kernel"],
                    kernel_params=kernel_params[feature_type],
                    conversion_map=config["meta_conversion"],
                    kernel_param_name_map=config["kwargs_overwrite"],
                    meta_overwrite=config["meta_overwrite"],
                )
                for feature_type, config in feature_type_configs.items()
            ]
        )

    return ufmt_format(make_file_content(functions))


def validate_feature_type_configs(feature_type_configs):
    try:
        feature_type_configs = {
            getattr(features, feature_type_name): config for feature_type_name, config in feature_type_configs.items()
        }
    except AttributeError:
        # unknown feature type
        raise TypeError() from None

    for feature_type, config in tuple(feature_type_configs.items()):
        if not isinstance(config, dict):
            feature_type_configs[feature_type] = config = dict(kernel=config)

        unknown_keys = config.keys() - {
            "kernel",
            "pil_kernel",
            "meta_conversion",
            "kwargs_overwrite",
            "meta_overwrite",
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

        if "pil_kernel" in config and feature_type is not features.Image:
            raise TypeError

        for key in ["meta_conversion", "kwargs_overwrite", "meta_overwrite"]:
            if key not in config:
                config[key] = dict()
                continue

            for meta_attr, value in tuple(config[key].items()):
                # if meta_attr not in feature_type._META_ATTRS:
                #     raise KeyError(meta_attr)

                config[key][meta_attr] = maybe_convert_to_enum(value)

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
        dispatcher_param_candidates = {}
        for feature_type, params in implementer_params.items():
            params = {param.name: param for param in params}
            if name not in params:
                need_kwargs_ignore.add(feature_type)
                continue
            else:
                dispatcher_param_candidates[feature_type] = params[name]

        if len(dispatcher_param_candidates) == 1:
            param = next(iter(dispatcher_param_candidates.values()))
            if len(implementer_params) == 1:
                dispatcher_params.append(copy(param))
            else:
                param._default = FEATURE_SPECIFIC_PARAM
                dispatcher_params.append(
                    Parameter(
                        name=name,
                        kind=Parameter.KEYWORD_ONLY,
                        default=FEATURE_SPECIFIC_PARAM,
                        annotation=param.annotation,
                    )
                )
            continue

        annotations = {param.annotation for param in dispatcher_param_candidates.values()}
        if len(annotations) > 1:
            raise TypeError(
                f"Found multiple annotations for parameter `{name}`: "
                f"{', '.join([str(annotation) for annotation in annotations])}"
            )

        defaults = {param.default for param in dispatcher_param_candidates.values()}
        default = FEATURE_SPECIFIC_DEFAULT if len(defaults) > 1 else defaults.pop()

        dispatcher_params.append(
            Parameter(
                name=name,
                kind=Parameter.KEYWORD_ONLY,
                default=default,
                annotation=annotations.pop(),
            )
        )

    without_default = []
    with_default = []
    for param in dispatcher_params:
        (without_default if param.default in (Parameter.empty, FEATURE_SPECIFIC_PARAM) else with_default).append(param)
    dispatcher_params = [*without_default, *with_default]

    for feature_type in need_kwargs_ignore:
        implementer_params[feature_type].append(Parameter(name="_", kind=Parameter.VAR_KEYWORD, annotation=Any))

    return dispatcher_params


def make_file_content(functions):
    enums = "\n".join(f"from {module.__package__} import {', '.join(names)}" for module, names in ENUMS)

    header = f"""
# THIS FILE IS AUTOGENERATED
#
# FROM torchvision/prototype/transforms/functional/dispatch.yaml
# WITH scripts/regenerate_transforms_dispatch.py
#
# DO NOT CHANGE MANUALLY!

from typing import Any, TypeVar, List, Optional, Tuple

import torch
import torchvision.transforms.functional as _F
import torchvision.prototype.transforms.functional as F
from torchvision.prototype import features
{enums}

Dispatcher = F.utils.Dispatcher

# This is just a sentinel to have a default argument for a dispatcher if the feature specific implementations use
# different defaults. The actual value is never used.
{FEATURE_SPECIFIC_DEFAULT} = object()

{GENERIC_FEATURE_TYPE} = TypeVar("{GENERIC_FEATURE_TYPE}", bound=features.Feature)
"""
    header = "\n".join(line.strip() for line in header.splitlines())

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

        # OPtional only has one
        # check docs ther ewas something about checking in the patch notes maybe?
        if (
            hasattr(self.annotation, "__origin__")
            and self.annotation.__origin__ is typing.Union
            and type(None) in self.annotation.__args__
        ):
            annotations = [
                inspect.formatannotation(arg) for arg in self.annotation.__args__ if arg is not type(None)  # noqa: E721
            ]
            tmp_values["annotation"] = ManualAnnotation(f"Optional[{', '.join(annotations)}]")
        elif isinstance(self.annotation, enum.EnumMeta):
            tmp_values["annotation"] = ManualAnnotation(self.annotation.__name__)

        with tmp_override(**tmp_values):
            return super().__str__()


class Signature(inspect.Signature):
    def __str__(self):
        text = super().__str__()
        for separator in [FEATURE_SPECIFIC_PARAM, FEATURE_SPECIFIC_DEFAULT]:
            parts = text.split(repr(separator))
            text = f"{separator},  # type: ignore[assignment]\n".join(
                [
                    parts[0],
                    *[part.lstrip(",") for part in parts[1:]],
                ]
            )
        return text


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
            decorator="Dispatcher",
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
        pil_kernel,
        kernel,
        kernel_params,
        conversion_map,
        kernel_param_name_map,
        meta_overwrite,
        input_name="input",
    ):
        feature_type_usage = ManualAnnotation(f"features.{feature_type.__name__}")

        body = []

        feature_specific_params = []
        for param in params:
            if param.default is FEATURE_SPECIFIC_PARAM:
                feature_specific_params.append(param.name)
                param._default = Parameter.empty

        output_conversions = []
        for idx, (attr, intermediate_value) in enumerate(conversion_map.items()):

            converter = META_CONVERTER_MAP[(feature_type, attr)]

            def make_conversion_call(input, old, new):
                return f"F.{converter.__name__}({input}, old_{attr}={old}, new_{attr}={new})"

            input_attr = f"input.{attr}"
            intermediate_name = f"intermediate_{attr}"
            body.extend(
                [
                    f"{intermediate_name} = {format_value(intermediate_value)}",
                    f"converted_input = {make_conversion_call(input_name, input_attr, intermediate_name)}",
                    "",
                ]
            )
            if idx == 0:
                input_name = "converted_input"

            output_conversions = [f"output = {make_conversion_call('output', intermediate_name, input_attr)}"]

        kernel_call = self._make_kernel_call(
            input_name=input_name,
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
            meta_overwrite=meta_overwrite,
        )
        body.append(f"return {feature_type_wrapper}")

        super().__init__(
            decorator=self._make_decorator(
                dispatcher_name=dispatcher_name,
                feature_type_usage=feature_type_usage,
                feature_specific_params=feature_specific_params,
                pil_kernel=pil_kernel,
            ),
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
                return_annotation=feature_type_usage,
            ),
            body=body,
        )

    def _make_decorator(self, *, dispatcher_name, feature_type_usage, feature_specific_params, pil_kernel):
        decorator = f"{dispatcher_name}.implements({feature_type_usage}"
        if feature_specific_params:
            decorator += f", feature_specific_params={tuple(feature_specific_params)}"
        if pil_kernel:
            decorator += f", pil_kernel=_F.{pil_kernel}"
        return f"{decorator})"

    def _make_kernel_call(
        self,
        *,
        kernel,
        input_name,
        kernel_params,
        kernel_param_name_map,
    ):
        call_args = [input_name]
        for param in kernel_params:
            dispatcher_param_name = kernel_param_name_map.get(param.name, param.name)
            if dispatcher_param_name.startswith("."):
                dispatcher_param_name = f"input{dispatcher_param_name}"
            call_args.append(f"{param.name}={dispatcher_param_name}")
        return f"F.{kernel.__name__}({', '.join(call_args)})"

    def _make_feature_type_wrapper(self, *, feature_type_usage, meta_overwrite):
        call_args = ["input", "output"]
        call_args.extend(
            f"{meta_name}={format_value(dispatcher_param_name)}"
            for meta_name, dispatcher_param_name in meta_overwrite.items()
        )
        return f"{feature_type_usage}.new_like({', '.join(call_args)})"


def ufmt_format(content):
    try:
        import ufmt
    except ModuleNotFoundError:
        return content

    HERE = pathlib.Path(__file__).parent

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


def maybe_convert_to_enum(value):
    if not isinstance(value, str):
        return value

    parts = value.split(".")
    if len(parts) != 2:
        return value

    enum, member = parts

    try:
        return ENUMS_MAP[enum][member]
    except KeyError:
        return value


def indent(text, level=1):
    return "\n".join(" " * (level * 4) + line for line in text.splitlines())


def camel_to_snake_case(camel_case: str) -> str:
    return "_".join([part.lower() for part in re.findall("[A-Z][^A-Z]*", camel_case)])


def format_value(value):
    if isinstance(value, str):
        return f'"{value}"'
    elif isinstance(value, enum.Enum):
        return f"{type(value).__name__}.{value.name}"
    else:
        return repr(value)


def unique(seq):
    unique_seq = []
    for item in seq:
        if item not in unique_seq:
            unique_seq.append(item)
    return unique_seq


if __name__ == "__main__":
    try:
        with open(pathlib.Path(F.__path__[0]) / "dispatch.yaml") as file:
            dispatch_config = yaml.load(file, yaml.Loader)
        content = main(dispatch_config)
        with open(pathlib.Path(F.__path__[0]) / "_dispatch.py", "w") as file:
            file.write(content)
    except Exception as error:
        msg = str(error)
        print(msg or f"Unspecified {type(error)} was raised during execution.", file=sys.stderr)
        sys.exit(1)
