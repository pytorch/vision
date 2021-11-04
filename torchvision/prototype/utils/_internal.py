import collections.abc
import difflib
import enum
from typing import Sequence, Collection, Callable

__all__ = ["StrEnum", "sequence_to_str", "add_suggestion"]


class StrEnumMeta(enum.EnumMeta):
    def __getitem__(self, item):
        return super().__getitem__(item.upper() if isinstance(item, str) else item)


class StrEnum(enum.Enum, metaclass=StrEnumMeta):
    pass


def sequence_to_str(seq: Sequence, separate_last: str = "") -> str:
    if len(seq) == 1:
        return f"'{seq[0]}'"

    return f"""'{"', '".join([str(item) for item in seq[:-1]])}', {separate_last}'{seq[-1]}'."""


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
