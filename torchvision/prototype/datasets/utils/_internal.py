import collections.abc
import difflib
import io
import os.path
import pathlib
from typing import Collection, Sequence, Callable, Union, Any
from typing import Tuple, Iterator

from torchdata.datapipes.iter import IterDataPipe


__all__ = [
    "INFINITE_BUFFER_SIZE",
    "sequence_to_str",
    "add_suggestion",
    "create_categories_file",
    "read_mat",
    "RarArchiveReader",
]

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


class RarArchiveReader(IterDataPipe[Tuple[str, io.BufferedIOBase]]):
    def __init__(self, datapipe: IterDataPipe[Tuple[str, io.BufferedIOBase]]):
        self._rarfile = self._verify_dependencies()
        super().__init__()
        self.datapipe = datapipe

    @staticmethod
    def _verify_dependencies():
        try:
            import rarfile
        except ImportError as error:
            raise ModuleNotFoundError(
                "Package `rarfile` is required to be installed to use this datapipe. "
                "Please use `pip install rarfile` or `conda -c conda-forge install rarfile` to install it."
            ) from error

        # check if at least one system library for reading rar archives is available to be used by rarfile
        rarfile.tool_setup()

        return rarfile

    def __iter__(self) -> Iterator[Tuple[str, io.BufferedIOBase]]:
        for path, stream in self.datapipe:
            rar = self._rarfile.RarFile(stream)
            for info in rar.infolist():
                if info.filename.endswith("/"):
                    continue

                inner_path = os.path.join(path, info.filename)
                file_obj = rar.open(info)

                yield inner_path, file_obj
