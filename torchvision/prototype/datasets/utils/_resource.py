import abc
import hashlib
import itertools
import pathlib
import warnings
from typing import Optional, Sequence, Tuple, Callable, IO, Any, Union, NoReturn
from urllib.parse import urlparse

from torchdata.datapipes.iter import (
    IterableWrapper,
    FileLister,
    FileLoader,
    IterDataPipe,
    ZipArchiveReader,
    TarArchiveReader,
    RarArchiveLoader,
)
from torchvision.datasets.utils import (
    download_url,
    _detect_file_type,
    extract_archive,
    _decompress,
    download_file_from_google_drive,
)


class OnlineResource(abc.ABC):
    def __init__(
        self,
        *,
        file_name: str,
        sha256: Optional[str] = None,
        decompress: bool = False,
        extract: bool = False,
        preprocess: Optional[Callable[[pathlib.Path], pathlib.Path]] = None,
        loader: Optional[Callable[[pathlib.Path], IterDataPipe[Tuple[str, IO]]]] = None,
    ) -> None:
        self.file_name = file_name
        self.sha256 = sha256

        if preprocess and (decompress or extract):
            warnings.warn("The parameters 'decompress' and 'extract' are ignored when 'preprocess' is passed.")
        elif extract:
            preprocess = self._extract
        elif decompress:
            preprocess = self._decompress
        self._preprocess = preprocess

        if loader is None:
            loader = self._default_loader
        self._loader = loader

    @staticmethod
    def _extract(file: pathlib.Path) -> pathlib.Path:
        return pathlib.Path(
            extract_archive(str(file), to_path=str(file).replace("".join(file.suffixes), ""), remove_finished=False)
        )

    @staticmethod
    def _decompress(file: pathlib.Path) -> pathlib.Path:
        return pathlib.Path(_decompress(str(file), remove_finished=True))

    def _default_loader(self, path: pathlib.Path) -> IterDataPipe[Tuple[str, IO]]:
        if path.is_dir():
            return FileLoader(FileLister(str(path), recursive=True))

        dp = FileLoader(IterableWrapper((str(path),)))

        archive_loader = self._guess_archive_loader(path)
        if archive_loader:
            dp = archive_loader(dp)

        return dp

    _ARCHIVE_LOADERS = {
        ".tar": TarArchiveReader,
        ".zip": ZipArchiveReader,
        ".rar": RarArchiveLoader,
    }

    def _guess_archive_loader(
        self, path: pathlib.Path
    ) -> Optional[Callable[[IterDataPipe[Tuple[str, IO]]], IterDataPipe[Tuple[str, IO]]]]:
        try:
            _, archive_type, _ = _detect_file_type(path.name)
        except RuntimeError:
            return None
        return self._ARCHIVE_LOADERS.get(archive_type)  # type: ignore[arg-type]

    def load(
        self, root: Union[str, pathlib.Path], *, skip_integrity_check: bool = False
    ) -> IterDataPipe[Tuple[str, IO]]:
        root = pathlib.Path(root)
        path = root / self.file_name
        # Instead of the raw file, there might also be files with fewer suffixes after decompression or directories
        # with no suffixes at all. Thus, we look for all paths that share the same name without suffixes as the raw
        # file.
        path_candidates = {file for file in path.parent.glob(path.name.replace("".join(path.suffixes), "") + "*")}
        # If we don't find anything, we try to download the raw file.
        if not path_candidates:
            path_candidates = {self.download(root, skip_integrity_check=skip_integrity_check)}
        # If the only thing we find is the raw file, we use it and optionally perform some preprocessing steps.
        if path_candidates == {path}:
            if self._preprocess:
                path = self._preprocess(path)
        # Otherwise we use the path with the fewest suffixes. This gives us the extracted > decompressed > raw priority
        # that we want.
        else:
            path = min(path_candidates, key=lambda path: len(path.suffixes))
        return self._loader(path)

    @abc.abstractmethod
    def _download(self, root: pathlib.Path) -> None:
        pass

    def download(self, root: Union[str, pathlib.Path], *, skip_integrity_check: bool = False) -> pathlib.Path:
        root = pathlib.Path(root)
        self._download(root)
        path = root / self.file_name
        if self.sha256 and not skip_integrity_check:
            self._check_sha256(path)
        return path

    def _check_sha256(self, path: pathlib.Path, *, chunk_size: int = 1024 * 1024) -> None:
        hash = hashlib.sha256()
        with open(path, "rb") as file:
            for chunk in iter(lambda: file.read(chunk_size), b""):
                hash.update(chunk)
        sha256 = hash.hexdigest()
        if sha256 != self.sha256:
            raise RuntimeError(
                f"After the download, the SHA256 checksum of {path} didn't match the expected one: "
                f"{sha256} != {self.sha256}"
            )


class HttpResource(OnlineResource):
    def __init__(
        self, url: str, *, file_name: Optional[str] = None, mirrors: Optional[Sequence[str]] = None, **kwargs: Any
    ) -> None:
        super().__init__(file_name=file_name or pathlib.Path(urlparse(url).path).name, **kwargs)
        self.url = url
        self.mirrors = mirrors

    def _download(self, root: pathlib.Path) -> None:
        for url in itertools.chain((self.url,), self.mirrors or ()):
            try:
                download_url(url, str(root), filename=self.file_name, md5=None)
            # TODO: make this more precise
            except Exception:
                continue

            return
        else:
            # TODO: make this more informative
            raise RuntimeError("Download failed!")


class GDriveResource(OnlineResource):
    def __init__(self, id: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.id = id

    def _download(self, root: pathlib.Path) -> None:
        download_file_from_google_drive(self.id, root=str(root), filename=self.file_name, md5=None)


class ManualDownloadResource(OnlineResource):
    def __init__(self, instructions: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.instructions = instructions

    def _download(self, root: pathlib.Path) -> NoReturn:
        raise RuntimeError(
            f"The file {self.file_name} cannot be downloaded automatically. "
            f"Please follow the instructions below and place it in {root}\n\n"
            f"{self.instructions}"
        )
