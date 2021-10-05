import collections.abc
import difflib
import io
import pathlib
from typing import Collection, Sequence, Callable, Union, Any, Tuple, TypeVar


__all__ = [
    "INFINITE_BUFFER_SIZE",
    "sequence_to_str",
    "add_suggestion",
    "create_categories_file",
    "read_mat",
    "getitem",
    "path_accessor",
    "path_comparator",
]

D = TypeVar("D")

# pseudo-infinite until a true infinite buffer is supported by all datapipes
INFINITE_BUFFER_SIZE = 1_000_000_000


def sequence_to_str(seq: Sequence, separate_last: str = "") -> str:
    if len(seq) == 1:
        return f"'{seq[0]}'"

    return f"""'{"', '".join([str(item) for item in seq[:-1]])}', """ f"""{separate_last}'{seq[-1]}'."""


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
    return f"{msg.strip()} {hint}"


def create_categories_file(root: Union[str, pathlib.Path], name: str, categories: Sequence[str]) -> None:
    with open(pathlib.Path(root) / f"{name}.categories", "w") as fh:
        fh.write("\n".join(categories) + "\n")


def read_mat(buffer: io.IOBase, **kwargs: Any) -> Any:
    try:
        import scipy.io as sio
    except ImportError as error:
        raise ModuleNotFoundError("Package `scipy` is required to be installed to read .mat files.") from error

    return sio.loadmat(buffer, **kwargs)


def getitem(*items: Any) -> Callable[[Any], Any]:
    def wrapper(obj: Any):
        for item in items:
            obj = obj[item]
        return obj

    return wrapper


def path_accessor(getter: Union[str, Callable[[pathlib.Path], D]]) -> Callable[[Tuple[str, Any]], D]:
    if isinstance(getter, str):
        name = getter

        def getter(path: pathlib.Path) -> D:
            return getattr(path, name)

    def wrapper(data: Tuple[str, Any]) -> D:
        return getter(pathlib.Path(data[0]))  # type: ignore[operator]

    return wrapper


def path_comparator(getter: Union[str, Callable[[pathlib.Path], D]], value: D) -> Callable[[Tuple[str, Any]], bool]:
    accessor = path_accessor(getter)

    def wrapper(data: Tuple[str, Any]) -> bool:
        return accessor(data) == value

    return wrapper
