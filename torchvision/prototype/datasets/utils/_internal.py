import collections.abc
import difflib
import io
import pathlib
from typing import Collection, Sequence, Callable, Union, Iterator, Tuple, TypeVar, Dict

import numpy as np
import PIL.Image
from torch.utils.data import IterDataPipe


__all__ = [
    "INFINITE_BUFFER_SIZE",
    "sequence_to_str",
    "add_suggestion",
    "create_categories_file",
    "image_buffer_from_array",
    "SequenceIterator",
    "MappingIterator",
    "Enumerator",
]


K = TypeVar("K")
D = TypeVar("D")


# pseudo-infinite until a true infinite buffer is supported by all datapipes
INFINITE_BUFFER_SIZE = 1_000_000_000


def sequence_to_str(seq: Sequence, separate_last: str = "") -> str:
    if len(seq) == 1:
        return f"'{seq[0]}'"

    return (
        f"""'{"', '".join([str(item) for item in seq[:-1]])}', """
        f"""{separate_last}'{seq[-1]}'."""
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
    with open(pathlib.Path(root) / f"{name}.categories", "w") as fh:
        fh.write("\n".join(categories) + "\n")


def image_buffer_from_array(array: np.array, *, format: str = "png") -> io.BytesIO:
    image = PIL.Image.fromarray(array)
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return buffer


class SequenceIterator(IterDataPipe[D]):
    def __init__(self, datapipe: IterDataPipe[Sequence[D]]):
        self.datapipe = datapipe

    def __iter__(self) -> Iterator[D]:
        for sequence in self.datapipe:
            yield from iter(sequence)


class MappingIterator(IterDataPipe[Union[Tuple[K, D], D]]):
    def __init__(self, datapipe: IterDataPipe[Dict[K, D]], *, drop_key: bool = False) -> None:
        self.datapipe = datapipe
        self.drop_key = drop_key

    def __iter__(self) -> Iterator[Union[Tuple[K, D], D]]:
        for mapping in self.datapipe:
            yield from iter(mapping.values() if self.drop_key else mapping.items())  # type: ignore[call-overload]


class Enumerator(IterDataPipe[Tuple[int, D]]):
    def __init__(self, datapipe: IterDataPipe[D], start: int = 0) -> None:
        self.datapipe = datapipe
        self.start = start

    def __iter__(self) -> Iterator[Tuple[int, D]]:
        yield from enumerate(self.datapipe, self.start)
