import collections.abc
import inspect
import re
from typing import Any, Callable, Dict, Optional, Type, Union, cast, Set, Collection

import torch
from torch import nn
from torchvision.prototype import features
from torchvision.prototype.utils._internal import add_suggestion


class Transform(nn.Module):
    """Base class for transforms.

    A transform operates on a full sample at once, which might be a nested container of elements to transform. The
    non-container elements of the sample will be dispatched to feature transforms based on their type in case it is
    supported by the transform. Each transform needs to define at least one feature transform, which is canonical done
    as static method:

    .. code-block::

        class ImageIdentity(Transform):
            @staticmethod
            def image(input):
                return input

    To achieve correct results for a complete sample, each transform should implement feature transforms for every
    :class:`Feature` it can handle:

    .. code-block::

        class Identity(Transform):
            @staticmethod
            def image(input):
                return input

            @staticmethod
            def bounding_box(input):
                return input

            ...

    If the name of a static method in camel-case matches the name of a :class:`Feature`, the feature transform is
    auto-registered. Supported pairs are:

    +----------------+----------------+
    | method name    | `Feature`      |
    +================+================+
    | `image`        | `Image`        |
    +----------------+----------------+
    | `bounding_box` | `BoundingBox`  |
    +----------------+----------------+
    | `label`        | `Label`        |
    +----------------+----------------+

    If you don't want to stick to this scheme, you can disable the auto-registration and perform it manually:

    .. code-block::

        def my_image_transform(input):
            ...

        class MyTransform(Transform, auto_register=False):
            def __init__(self):
                super().__init__()
                self.register_feature_transform(Image, my_image_transform)
                self.register_feature_transform(BoundingBox, self.my_bounding_box_transform)

            @staticmethod
            def my_bounding_box_transform(input):
                ...

    In any case, the registration will assert that the feature transform can be invoked with
    ``feature_transform(input, **params)``.

    .. warning::

        Feature transforms are **registered on the class and not on the instance**. This means you cannot have two
        instances of the same :class:`Transform` with different feature transforms.

    If the feature transforms needs additional parameters, you need to
    overwrite the :meth:`~Transform.get_params` method. It needs to return the parameter dictionary that will be
    unpacked and its contents passed to each feature transform:

    .. code-block::

        class Rotate(Transform):
            def __init__(self, degrees):
                super().__init__()
                self.degrees = degrees

            def get_params(self, sample):
                return dict(degrees=self.degrees)

            def image(input, *, degrees):
                ...

    The :meth:`~Transform.get_params` method will be invoked once per sample. Thus, in case of randomly sampled
    parameters they will be the same for all features of the whole sample.

    .. code-block::

        class RandomRotate(Transform)
            def __init__(self, range):
                super().__init__()
                self._dist = torch.distributions.Uniform(range)

            def get_params(self, sample):
                return dict(degrees=self._dist.sample().item())

            @staticmethod
            def image(input, *, degrees):
                ...

    In case the sampling depends on one or more features at runtime, the complete ``sample`` gets passed to the
    :meth:`Transform.get_params` method. Derivative transforms that only changes the parameter sampling, but the
    feature transformations are identical, can simply wrap the transform they dispatch to:

    .. code-block::

        class RandomRotate(Transform, wraps=Rotate):
            def get_params(self, sample):
                return dict(degrees=float(torch.rand(())) * 30.0)

    To transform a sample, you simply call an instance of the transform with it:

    .. code-block::

        transform = MyTransform()
        sample = dict(input=Image(torch.tensor(...)), target=BoundingBox(torch.tensor(...)), ...)
        transformed_sample = transform(sample)

    .. note::

        To use a :class:`Transform` with a dataset, simply use it as map:

        .. code-block::

            torchvision.datasets.load(...).map(MyTransform())
    """

    _BUILTIN_FEATURE_TYPES = (
        features.BoundingBox,
        features.Image,
        features.Label,
    )
    _FEATURE_NAME_MAP = {
        "_".join([part.lower() for part in re.findall("[A-Z][^A-Z]*", feature_type.__name__)]): feature_type
        for feature_type in _BUILTIN_FEATURE_TYPES
    }
    _feature_transforms: Dict[Type[features.Feature], Callable]

    NO_OP_FEATURE_TYPES: Collection[Type[features.Feature]] = ()

    def __init_subclass__(
        cls, *, wraps: Optional[Type["Transform"]] = None, auto_register: bool = True, verbose: bool = False
    ):
        cls._feature_transforms = {} if wraps is None else wraps._feature_transforms.copy()
        if wraps:
            cls.NO_OP_FEATURE_TYPES = wraps.NO_OP_FEATURE_TYPES
        if auto_register:
            cls._auto_register(verbose=verbose)

    @staticmethod
    def _has_allowed_signature(feature_transform: Callable) -> bool:
        """Checks if ``feature_transform`` can be invoked with ``feature_transform(input, **params)``"""

        parameters = tuple(inspect.signature(feature_transform).parameters.values())
        if not parameters:
            return False
        elif len(parameters) == 1:
            return parameters[0].kind != inspect.Parameter.KEYWORD_ONLY
        else:
            return parameters[1].kind != inspect.Parameter.POSITIONAL_ONLY

    @classmethod
    def register_feature_transform(cls, feature_type: Type[features.Feature], transform: Callable) -> None:
        """Registers a transform for given feature on the class.

        If a transform object is called or :meth:`Transform.apply` is invoked, inputs are dispatched to the registered
        transforms based on their type.

        Args:
            feature_type: Feature type the transformation is registered for.
            transform: Feature transformation.

        Raises:
            TypeError: If ``transform`` cannot be invoked with ``transform(input, **params)``.
        """
        if not cls._has_allowed_signature(transform):
            raise TypeError("Feature transform cannot be invoked with transform(input, **params)")
        cls._feature_transforms[feature_type] = transform

    @classmethod
    def _auto_register(cls, *, verbose: bool = False) -> None:
        """Auto-registers methods on the class as feature transforms if they meet the following criteria:

        1. They are static.
        2. They can be invoked with `cls.feature_transform(input, **params)`.
        3. They are public.
        4. Their name in camel case matches the name of a builtin feature, e.g. 'bounding_box' and 'BoundingBox'.

        The name from 4. determines for which feature the method is registered.

        .. note::

            The ``auto_register`` and ``verbose`` flags need to be passed as keyword arguments to the class:

            .. code-block::

                class MyTransform(Transform, auto_register=True, verbose=True):
                    ...

        Args:
            verbose: If ``True``, prints to STDOUT which methods were registered or why a method was not registered
        """
        for name, value in inspect.getmembers(cls):
            # check if attribute is a static method and was defined in the subclass
            # TODO: this needs to be revisited to allow subclassing of custom transforms
            if not (name in cls.__dict__ and inspect.isfunction(value)):
                continue

            not_registered_prefix = f"{cls.__name__}.{name}() was not registered as feature transform, because"

            if not cls._has_allowed_signature(value):
                if verbose:
                    print(f"{not_registered_prefix} it cannot be invoked with {name}(input, **params).")
                continue

            if name.startswith("_"):
                if verbose:
                    print(f"{not_registered_prefix} it is private.")
                continue

            try:
                feature_type = cls._FEATURE_NAME_MAP[name]
            except KeyError:
                if verbose:
                    print(
                        add_suggestion(
                            f"{not_registered_prefix} its name doesn't match any known feature type.",
                            word=name,
                            possibilities=cls._FEATURE_NAME_MAP.keys(),
                            close_match_hint=lambda close_match: (
                                f"Did you mean to name it '{close_match}' "
                                f"to be registered for type '{cls._FEATURE_NAME_MAP[close_match]}'?"
                            ),
                        )
                    )
                continue

            cls.register_feature_transform(feature_type, value)
            if verbose:
                print(
                    f"{cls.__name__}.{name}() was registered as feature transform for type '{feature_type.__name__}'."
                )

    @classmethod
    def from_callable(
        cls,
        feature_transform: Union[Callable, Dict[Type[features.Feature], Callable]],
        *,
        name: str = "FromCallable",
        get_params: Optional[Union[Dict[str, Any], Callable[[Any], Dict[str, Any]]]] = None,
    ) -> "Transform":
        """Creates a new transform from a callable.

        Args:
            feature_transform: Feature transform that will be registered to handle :class:`Image`'s. Can be passed as
                dictionary in which case each key-value-pair is needs to consists of a ``Feature`` type and the
                corresponding transform.
            name: Name of the transform.
            get_params: Parameter dictionary ``params`` that will be passed to ``feature_transform(input, **params)``.
                Can be passed as callable in which case it will be called with the transform instance (``self``) and
                the input of the transform.

        Raises:
            TypeError: If ``feature_transform`` cannot be invoked with ``feature_transform(input, **params)``.
        """
        if get_params is None:
            get_params = dict()
        attributes = dict(
            get_params=get_params if callable(get_params) else lambda self, sample: get_params,  # type: ignore[misc]
        )
        transform_cls = cast(Type[Transform], type(name, (cls,), attributes))

        if callable(feature_transform):
            feature_transform = {features.Image: feature_transform}
        for feature_type, transform in feature_transform.items():
            transform_cls.register_feature_transform(feature_type, transform)

        return transform_cls()

    @classmethod
    def supported_feature_types(cls) -> Set[Type[features.Feature]]:
        return set(cls._feature_transforms.keys())

    @classmethod
    def supports(cls, obj: Any) -> bool:
        """Checks if object or type is supported.

        Args:
            obj: Object or type.
        """
        # TODO: should this handle containers?
        feature_type = obj if isinstance(obj, type) else type(obj)
        return feature_type is torch.Tensor or feature_type in cls.supported_feature_types()

    @classmethod
    def transform(cls, input: Union[torch.Tensor, features.Feature], **params: Any) -> torch.Tensor:
        """Applies the registered feature transform to the input based on its type.

        This can be uses as feature type generic functional interface:

            .. code-block::

                transform = Rotate.transform
                transformed_image = transform(Image(torch.tensor(...)), degrees=30.0)
                transformed_bbox = transform(BoundingBox(torch.tensor(...)), degrees=-10.0)

        Args:
            input: ``input`` in ``feature_transform(input, **params)``
            **params: Parameter dictionary ``params`` in ``feature_transform(input, **params)``.

        Returns:
            Transformed input.
        """
        feature_type = type(input)
        if not cls.supports(feature_type):
            raise TypeError(f"{cls.__name__}() is not able to handle inputs of type {feature_type}.")

        if feature_type is torch.Tensor:
            # To keep BC, we treat all regular torch.Tensor's as images
            feature_type = features.Image
            input = feature_type(input)
        feature_type = cast(Type[features.Feature], feature_type)

        feature_transform = cls._feature_transforms[feature_type]
        output = feature_transform(input, **params)

        if type(output) is torch.Tensor:
            output = feature_type(output, like=input)
        return output

    def _transform_recursively(self, sample: Any, *, params: Dict[str, Any]) -> Any:
        """Recurses through a sample and invokes :meth:`Transform.transform` on non-container elements.

        If an element is not supported by the transform, it is returned untransformed.

        Args:
            sample: Sample.
            params: Parameter dictionary ``params`` that will be passed to ``feature_transform(input, **params)``.
        """
        # We explicitly exclude str's here since they are self-referential and would cause an infinite recursion loop:
        # "a" == "a"[0][0]...
        if isinstance(sample, collections.abc.Sequence) and not isinstance(sample, str):
            return [self._transform_recursively(item, params=params) for item in sample]
        elif isinstance(sample, collections.abc.Mapping):
            return {name: self._transform_recursively(item, params=params) for name, item in sample.items()}
        else:
            feature_type = type(sample)
            if not self.supports(feature_type):
                if (
                    not issubclass(feature_type, features.Feature)
                    # issubclass is not a strict check, but also allows the type checked against. Thus, we need to
                    # check it separately
                    or feature_type is features.Feature
                    or feature_type in self.NO_OP_FEATURE_TYPES
                ):
                    return sample

                raise TypeError(
                    f"{type(self).__name__}() is not able to handle inputs of type {feature_type}. "
                    f"If you want it to be a no-op, add the feature type to {type(self).__name__}.NO_OP_FEATURE_TYPES."
                )

            return self.transform(cast(Union[torch.Tensor, features.Feature], sample), **params)

    def get_params(self, sample: Any) -> Dict[str, Any]:
        """Returns the parameter dictionary used to transform the current sample.

        .. note::

            Since ``sample`` might be a nested container, it is recommended to use the
            :class:`torchvision.datasets.utils.Query` class if you need to extract information from it.

        Args:
            sample: Current sample.

        Returns:
            Parameter dictionary ``params`` in ``feature_transform(input, **params)``.
        """
        return dict()

    def forward(
        self,
        *inputs: Any,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        if not self._feature_transforms:
            raise RuntimeError(f"{type(self).__name__}() has no registered feature transform.")

        sample = inputs if len(inputs) > 1 else inputs[0]
        if params is None:
            params = self.get_params(sample)
        return self._transform_recursively(sample, params=params)
