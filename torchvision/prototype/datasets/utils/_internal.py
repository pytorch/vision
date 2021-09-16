import collections.abc
import difflib
import io
import pathlib
from typing import Any, Tuple
from typing import Collection, Sequence, Callable
from typing import Union

import numpy as np
import PIL.Image

__all__ = [
    "read_mat",
    "image_buffer_from_array",
    "sequence_to_str",
    "add_suggestion",
    "create_categories_file",
]


def read_mat(file: io.BufferedIOBase, **kwargs: Any) -> Any:
    try:
        import scipy.io as sio
    except ImportError as error:
        raise ModuleNotFoundError(
            "Package `scipy` is required to be installed to read .mat files."
        ) from error

    return sio.loadmat(file, **kwargs)


def image_buffer_from_array(array: np.array, *, format: str) -> Tuple[str, io.BytesIO]:
    image = PIL.Image.fromarray(array)
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return f"tmp.{format}", buffer


def sequence_to_str(seq: Sequence, separate_last: str = "") -> str:
    if len(seq) == 1:
        return f"'{seq[0]}'"

    return (
        f"""'{"', '".join([str(item) for item in seq[:-1]])}', """
        f"""{separate_last}'{seq[-1]}'"""
    )


def add_suggestion(
    msg: str,
    *,
    word: str,
    possibilities: Collection[str],
    close_match_hint: Callable[
        [str], str
    ] = lambda close_match: f"Did you mean '{close_match}'?",
    alternative_hint: Callable[
        [Sequence[str]], str
    ] = lambda possibilities: f"Can be {sequence_to_str(possibilities, separate_last='or ')}.",
) -> str:
    if not isinstance(possibilities, collections.abc.Sequence):
        possibilities = sorted(possibilities)
    suggestions = difflib.get_close_matches(word, possibilities, 1)
    hint = (
        close_match_hint(suggestions[0])
        if suggestions
        else alternative_hint(possibilities)
    )
    return f"{msg.strip()} {hint}"


def create_categories_file(
    root: Union[str, pathlib.Path], name: str, categories: Sequence[str]
) -> None:
    with open(root / f"{name}.categories", "w") as fh:
        fh.write("\n".join(categories) + "\n")
