import collections.abc
import difflib
import enum
import functools
import inspect
import os
import os.path
import textwrap
import warnings
from typing import Collection, Sequence, Callable, Any, Iterator, NoReturn, Mapping, TypeVar, Iterable, Tuple, cast

__all__ = [
    "StrEnum",
    "sequence_to_str",
    "add_suggestion",
    "FrozenMapping",
    "make_repr",
    "FrozenBunch",
    "kwonly_to_pos_or_kw",
]


class StrEnumMeta(enum.EnumMeta):
    def __getitem__(self, item):
        return super().__getitem__(item.upper() if isinstance(item, str) else item)


class StrEnum(enum.Enum, metaclass=StrEnumMeta):
    pass


def sequence_to_str(seq: Sequence, separate_last: str = "") -> str:
    if len(seq) == 1:
        return f"'{seq[0]}'"

    return f"""'{"', '".join([str(item) for item in seq[:-1]])}', {separate_last}'{seq[-1]}'"""


def add_suggestion(
    msg: str,
    *,
    word: str,
    possibilities: Collection[str],
    close_match_hint: Callable[[str], str] = lambda close_match: f"Did you mean '{close_match}'?",
    alternative_hint: Callable[
        [Sequence[str]], str
    ] = lambda possibilities: f"Can be {sequence_to_str(possibilities, separate_last='or ')}.",
) -> str:
    if not isinstance(possibilities, collections.abc.Sequence):
        possibilities = sorted(possibilities)
    suggestions = difflib.get_close_matches(word, possibilities, 1)
    hint = close_match_hint(suggestions[0]) if suggestions else alternative_hint(possibilities)
    if not hint:
        return msg

    return f"{msg.strip()} {hint}"


K = TypeVar("K")
D = TypeVar("D")


class FrozenMapping(Mapping[K, D]):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        data = dict(*args, **kwargs)
        self.__dict__["__data__"] = data
        self.__dict__["__final_hash__"] = hash(tuple(data.items()))

    def __getitem__(self, item: K) -> D:
        return cast(Mapping[K, D], self.__dict__["__data__"])[item]

    def __iter__(self) -> Iterator[K]:
        return iter(self.__dict__["__data__"].keys())

    def __len__(self) -> int:
        return len(self.__dict__["__data__"])

    def __immutable__(self) -> NoReturn:
        raise RuntimeError(f"'{type(self).__name__}' object is immutable")

    def __setitem__(self, key: K, value: Any) -> NoReturn:
        self.__immutable__()

    def __delitem__(self, key: K) -> NoReturn:
        self.__immutable__()

    def __hash__(self) -> int:
        return cast(int, self.__dict__["__final_hash__"])

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, FrozenMapping):
            return NotImplemented

        return hash(self) == hash(other)

    def __repr__(self) -> str:
        return repr(self.__dict__["__data__"])


def make_repr(name: str, items: Iterable[Tuple[str, Any]]) -> str:
    def to_str(sep: str) -> str:
        return sep.join([f"{key}={value}" for key, value in items])

    prefix = f"{name}("
    postfix = ")"
    body = to_str(", ")

    line_length = int(os.environ.get("COLUMNS", 80))
    body_too_long = (len(prefix) + len(body) + len(postfix)) > line_length
    multiline_body = len(str(body).splitlines()) > 1
    if not (body_too_long or multiline_body):
        return prefix + body + postfix

    body = textwrap.indent(to_str(",\n"), " " * 2)
    return f"{prefix}\n{body}\n{postfix}"


class FrozenBunch(FrozenMapping):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'") from error

    def __setattr__(self, key: Any, value: Any) -> NoReturn:
        self.__immutable__()

    def __delattr__(self, item: Any) -> NoReturn:
        self.__immutable__()

    def __repr__(self) -> str:
        return make_repr(type(self).__name__, self.items())


def kwonly_to_pos_or_kw(fn: Callable[..., D]) -> Callable[..., D]:
    """Decorates a function that uses keyword only parameters to also allow them being passed as positionals.

    For example, consider the use case of changing the signature of ``old_fn`` into the one from ``new_fn``:

    .. code::

        def old_fn(foo, bar, baz=None):
            ...

        def new_fn(foo, *, bar, baz=None):
            ...

    Calling ``old_fn("foo", "bar, "baz")`` was valid, but the same call is no longer valid with ``new_fn``. To keep BC
    and at the same time warn the user of the deprecation, this decorator can be used:

    .. code::

        @kwonly_to_pos_or_kw
        def new_fn(foo, *, bar, baz=None):
            ...

        new_fn("foo", "bar, "baz")
    """
    params = inspect.signature(fn).parameters

    try:
        keyword_only_start_idx = next(
            idx for idx, param in enumerate(params.values()) if param.kind == param.KEYWORD_ONLY
        )
    except StopIteration:
        raise TypeError(f"Found no keyword-only parameter on function '{fn.__name__}'") from None

    keyword_only_params = tuple(inspect.signature(fn).parameters)[keyword_only_start_idx:]

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> D:
        args, keyword_only_args = args[:keyword_only_start_idx], args[keyword_only_start_idx:]
        if keyword_only_args:
            keyword_only_kwargs = dict(zip(keyword_only_params, keyword_only_args))
            warnings.warn(
                f"Using {sequence_to_str(tuple(keyword_only_kwargs.keys()), separate_last='and ')} as positional "
                f"parameter(s) is deprecated. Please use keyword parameter(s) instead."
            )
            kwargs.update(keyword_only_kwargs)

        return fn(*args, **kwargs)

    return wrapper
