import enum
from typing import Any, Callable, Dict, List, Tuple, Type, Union

import PIL.Image

import torch
from torch import nn
from torch.utils._pytree import tree_flatten, tree_unflatten, TreeSpec
from torchvision.prototype import datapoints
from torchvision.prototype.transforms import functional as F
from torchvision.prototype.transforms.utils import check_type, is_simple_tensor
from torchvision.utils import _log_api_usage_once


class Transform(nn.Module):

    # Class attribute defining transformed types. Other types are passed-through without any transformation
    # We support both Types and callables that are able to do further checks on the type of the input.
    _transformed_types: Tuple[Union[Type, Callable[[Any], bool]], ...] = (torch.Tensor, PIL.Image.Image)

    def __init__(self) -> None:
        super().__init__()
        _log_api_usage_once(self)

    def _check_inputs(self, flat_inputs: List[Any]) -> None:
        pass

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        return dict()

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        raise NotImplementedError

    def forward(self, *inputs: Any) -> Any:
        flat_inputs, spec = tree_flatten(inputs if len(inputs) > 1 else inputs[0])

        self._check_inputs(flat_inputs)

        params = self._get_params(flat_inputs)

        flat_outputs = [
            self._transform(inpt, params) if check_type(inpt, self._transformed_types) else inpt for inpt in flat_inputs
        ]

        return tree_unflatten(flat_outputs, spec)

    def extra_repr(self) -> str:
        extra = []
        for name, value in self.__dict__.items():
            if name.startswith("_") or name == "training":
                continue

            if not isinstance(value, (bool, int, float, str, tuple, list, enum.Enum)):
                continue

            extra.append(f"{name}={value}")

        return ", ".join(extra)


class _RandomApplyTransform(Transform):
    def __init__(self, p: float = 0.5) -> None:
        if not (0.0 <= p <= 1.0):
            raise ValueError("`p` should be a floating point value in the interval [0.0, 1.0].")

        super().__init__()
        self.p = p

    def forward(self, *inputs: Any) -> Any:
        # We need to almost duplicate `Transform.forward()` here since we always want to check the inputs, but return
        # early afterwards in case the random check triggers. The same result could be achieved by calling
        # `super().forward()` after the random check, but that would call `self._check_inputs` twice.

        inputs = inputs if len(inputs) > 1 else inputs[0]
        flat_inputs, spec = tree_flatten(inputs)

        self._check_inputs(flat_inputs)

        if torch.rand(1) >= self.p:
            return inputs

        params = self._get_params(flat_inputs)

        flat_outputs = [
            self._transform(inpt, params) if check_type(inpt, self._transformed_types) else inpt for inpt in flat_inputs
        ]

        return tree_unflatten(flat_outputs, spec)


class _DetectionBatchTransform(Transform):
    @staticmethod
    def _flatten_and_extract_data(
        inputs: Any, **types_or_checks: Tuple[Union[Type, Callable[[Any], bool]], ...]
    ) -> Tuple[Tuple[List[Any], TreeSpec, List[Dict[str, int]]], List[Dict[str, Any]]]:
        batch = inputs if len(inputs) > 1 else inputs[0]
        flat_batch = []
        sample_specs = []

        offset = 0
        batch_idcs = []
        batch_data = []
        for sample_idx, sample in enumerate(batch):
            flat_sample, sample_spec = tree_flatten(sample)
            flat_batch.extend(flat_sample)
            sample_specs.append(sample_spec)

            sample_types_or_checks = types_or_checks.copy()
            sample_idcs = {}
            sample_data = {}
            for flat_idx, item in enumerate(flat_sample, offset):
                if not sample_types_or_checks:
                    break

                for key, types_or_checks_ in sample_types_or_checks.items():
                    if check_type(item, types_or_checks_):
                        break
                else:
                    continue

                del sample_types_or_checks[key]
                sample_idcs[key] = flat_idx
                sample_data[key] = item

            if sample_types_or_checks:
                # TODO: improve message
                raise TypeError(
                    f"Sample at index {sample_idx} in the batch is missing {sample_types_or_checks.keys()}`"
                )

            batch_idcs.append(sample_idcs)
            batch_data.append(sample_data)
            offset += len(flat_sample)

        batch_spec = TreeSpec(list, context=None, children_specs=sample_specs)

        return (flat_batch, batch_spec, batch_idcs), batch_data

    @staticmethod
    def _to_image_tensor(batch: List[Dict[str, Any]], *, key: str = "image") -> List[Dict[str, Any]]:
        for sample in batch:
            image = sample.pop(key)
            if isinstance(image, PIL.Image.Image):
                image = F.pil_to_tensor(image)
            elif isinstance(image, datapoints.Image):
                image = image.as_subclass(torch.Tensor)
            sample[key] = image
        return batch

    @staticmethod
    def _unflatten_and_insert_data(
        flat_batch_with_spec: Tuple[List[Any], TreeSpec, List[Dict[str, int]]],
        batch: List[Dict[str, Any]],
    ) -> Any:
        flat_batch, batch_spec, batch_idcs = flat_batch_with_spec

        for sample_idx, sample_idcs in enumerate(batch_idcs):
            for key, flat_idx in sample_idcs.items():
                inpt = flat_batch[flat_idx]
                item = batch[sample_idx][key]

                if not is_simple_tensor(inpt) and is_simple_tensor(item):
                    if isinstance(inpt, datapoints._datapoint.Datapoint):
                        item = type(inpt).wrap_like(inpt, item)
                    elif isinstance(inpt, PIL.Image.Image):
                        item = F.to_image_pil(item)

                flat_batch[flat_idx] = item

        return tree_unflatten(flat_batch, batch_spec)
