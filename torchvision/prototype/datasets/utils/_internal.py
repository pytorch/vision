import collections.abc
import difflib
from typing import Collection, Sequence, Callable


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
