from __future__ import annotations

import abc
import pathlib
from typing import Optional, Tuple, Callable, BinaryIO, Any, Union, NoReturn, Set
from urllib.parse import urlparse

from torch.hub import tqdm
from torchdata.datapipes.iter import (
    IterableWrapper,
    FileLister,
    FileOpener,
    IterDataPipe,
    ZipArchiveLoader,
    TarArchiveLoader,
    RarArchiveLoader,
    OnlineReader,
    HashChecker,
)
from torchvision.datasets.utils import _detect_file_type, extract_archive, _decompress
from typing_extensions import Literal


class OnlineResource(abc.ABC):
    def __init__(
        self,
        url: str,
        *,
        file_name: str,
        sha256: Optional[str] = None,
        preprocess: Optional[Union[Literal["decompress", "extract"], Callable[[pathlib.Path], None]]] = None,
    ) -> None:
        self.url = url
        self.file_name = file_name
        self.sha256 = sha256

        if isinstance(preprocess, str):
            if preprocess == "decompress":
                preprocess = self._decompress
            elif preprocess == "extract":
                preprocess = self._extract
            else:
                raise ValueError(
                    f"Only `'decompress'` or `'extract'` are valid if `preprocess` is passed as string,"
                    f"but got {preprocess} instead."
                )
        self._preprocess = preprocess

    @staticmethod
    def _extract(file: pathlib.Path) -> None:
        extract_archive(str(file), to_path=str(file).replace("".join(file.suffixes), ""), remove_finished=False)

    @staticmethod
    def _decompress(file: pathlib.Path) -> None:
        _decompress(str(file), remove_finished=True)

    @classmethod
    def from_http(cls, url: str, *, file_name: Optional[str] = None, **kwargs: Any) -> OnlineResource:
        return cls(url, file_name=file_name or pathlib.Path(urlparse(url).path).name, **kwargs)

    @classmethod
    def from_gdrive(cls, id: str, **kwargs: Any) -> OnlineResource:
        return cls(f"https://drive.google.com/uc?export=download&id={id}", **kwargs)

    def download(self, root: Union[str, pathlib.Path], *, skip_integrity_check: bool = False) -> pathlib.Path:
        root = pathlib.Path(root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        file = root / self.file_name

        if not file.exists():
            dp = IterableWrapper([self.url])
            dp = OnlineReader(dp)
            stream = list(dp)[0][1]

            with open(file, "wb") as fh, tqdm() as progress_bar:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):  # type: ignore[no-any-return]
                    # filter out keep-alive new chunks
                    if not chunk:
                        continue

                    fh.write(chunk)
                    progress_bar.update(len(chunk))

        if self.sha256 and not skip_integrity_check:
            dp = IterableWrapper([str(file)])
            dp = FileOpener(dp, mode="rb")
            dp = HashChecker(dp, {str(file): self.sha256}, hash_type="sha256")
            list(dp)

        return file

    def _loader(self, path: pathlib.Path) -> IterDataPipe[Tuple[str, BinaryIO]]:
        if path.is_dir():
            return FileOpener(FileLister(str(path), recursive=True), mode="rb")

        dp = FileOpener(IterableWrapper((str(path),)), mode="rb")

        archive_loader = self._guess_archive_loader(path)
        if archive_loader:
            dp = archive_loader(dp)

        return dp

    _ARCHIVE_LOADERS = {
        ".tar": TarArchiveLoader,
        ".zip": ZipArchiveLoader,
        ".rar": RarArchiveLoader,
    }

    def _guess_archive_loader(
        self, path: pathlib.Path
    ) -> Optional[Callable[[IterDataPipe[Tuple[str, BinaryIO]]], IterDataPipe[Tuple[str, BinaryIO]]]]:
        try:
            _, archive_type, _ = _detect_file_type(path.name)
        except RuntimeError:
            return None
        return self._ARCHIVE_LOADERS.get(archive_type)  # type: ignore[arg-type]

    def load(
        self, root: Union[str, pathlib.Path], *, skip_integrity_check: bool = False
    ) -> IterDataPipe[Tuple[str, BinaryIO]]:
        root = pathlib.Path(root)
        path = root / self.file_name

        # Instead of the raw file, there might also be files with fewer suffixes after decompression or directories
        # with no suffixes at all. `pathlib.Path().stem` will only give us the name with the last suffix removed, which
        # is not sufficient for files with multiple suffixes, e.g. foo.tar.gz.
        stem = path.name.replace("".join(path.suffixes), "")

        def find_candidates() -> Set[pathlib.Path]:
            # Although it looks like we could glob for f"{stem}*" to find the file candidates as well as the folder
            # candidate simultaneously, that would also pick up other files that share the same prefix. For example, the
            # test split of the stanford-cars dataset uses the files
            # - cars_test.tgz
            # - cars_test_annos_withlabels.mat
            # Globbing for `"cars_test*"` picks up both.
            candidates = {file for file in path.parent.glob(f"{stem}.*")}
            folder_candidate = path.parent / stem
            if folder_candidate.exists():
                candidates.add(folder_candidate)

            return candidates

        candidates = find_candidates()

        if not candidates:
            self.download(root, skip_integrity_check=skip_integrity_check)
            if self._preprocess is not None:
                self._preprocess(path)
            candidates = find_candidates()

        # We use the path with the fewest suffixes. This gives us the
        # extracted > decompressed > raw
        # priority that we want for the best I/O performance.
        return self._loader(min(candidates, key=lambda candidate: len(candidate.suffixes)))


class ManualDownloadResource(OnlineResource):
    def __init__(self, url: str, *, instructions: str, **kwargs: Any) -> None:
        super().__init__(url, **kwargs)
        self._instructions = instructions

    def download(self, root: Union[str, pathlib.Path], **_: Any) -> NoReturn:
        root = pathlib.Path(root)
        raise RuntimeError(
            f"The file {self.file_name} cannot be downloaded automatically. "
            f"Please follow the instructions below and place it in {root}\n\n"
            f"{self._instructions}"
        )

    @classmethod
    def from_kaggle(cls, challenge_url: str, *, file_name: str, **kwargs: Any) -> ManualDownloadResource:
        instructions = "\n".join(
            (
                "1. Register and login at https://www.kaggle.com",
                f"2. Navigate to {challenge_url}",
                "3. Click 'Join Competition' and follow the instructions there",
                "4. Navigate to the 'Data' tab",
                f"5. Select {file_name} in the 'Data Explorer' and click the download button",
            )
        )
        return cls(challenge_url, instructions=instructions, file_name=file_name, **kwargs)
