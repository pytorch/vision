import abc
import hashlib
import itertools
import pathlib
from typing import Optional, Sequence, Tuple, Callable, IO, Any, Union, NoReturn
from urllib.parse import urlparse

from torchdata.datapipes.iter import (
    IterableWrapper,
    FileLister,
    FileOpener,
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
    _get_redirect_url,
    _get_google_drive_file_id,
)


class OnlineResource(abc.ABC):
    def __init__(
        self,
        *,
        file_name: str,
        sha256: Optional[str] = None,
        decompress: bool = False,
        extract: bool = False,
    ) -> None:
        self.file_name = file_name
        self.sha256 = sha256

        self._preprocess: Optional[Callable[[pathlib.Path], pathlib.Path]]
        if extract:
            self._preprocess = self._extract
        elif decompress:
            self._preprocess = self._decompress
        else:
            self._preprocess = None

    @staticmethod
    def _extract(file: pathlib.Path) -> pathlib.Path:
        return pathlib.Path(
            extract_archive(str(file), to_path=str(file).replace("".join(file.suffixes), ""), remove_finished=False)
        )

    @staticmethod
    def _decompress(file: pathlib.Path) -> pathlib.Path:
        return pathlib.Path(_decompress(str(file), remove_finished=True))

    def _loader(self, path: pathlib.Path) -> IterDataPipe[Tuple[str, IO]]:
        if path.is_dir():
            return FileOpener(FileLister(str(path), recursive=True), mode="rb")

        dp = FileOpener(IterableWrapper((str(path),)), mode="rb")

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
        # with no suffixes at all.
        stem = path.name.replace("".join(path.suffixes), "")

        # In a first step, we check for a folder with the same stem as the raw file. If it exists, we use it since
        # extracted files give the best I/O performance. Note that OnlineResource._extract() makes sure that an archive
        # is always extracted in a folder with the corresponding file name.
        folder_candidate = path.parent / stem
        if folder_candidate.exists() and folder_candidate.is_dir():
            return self._loader(folder_candidate)

        # If there is no folder, we look for all files that share the same stem as the raw file, but might have a
        # different suffix.
        file_candidates = {file for file in path.parent.glob(stem + ".*")}
        # If we don't find anything, we download the raw file.
        if not file_candidates:
            file_candidates = {self.download(root, skip_integrity_check=skip_integrity_check)}
        # If the only thing we find is the raw file, we use it and optionally perform some preprocessing steps.
        if file_candidates == {path}:
            if self._preprocess is not None:
                path = self._preprocess(path)
        # Otherwise, we use the path with the fewest suffixes. This gives us the decompressed > raw priority that we
        # want for the best I/O performance.
        else:
            path = min(file_candidates, key=lambda path: len(path.suffixes))
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
        self, url: str, *, file_name: Optional[str] = None, mirrors: Sequence[str] = (), **kwargs: Any
    ) -> None:
        super().__init__(file_name=file_name or pathlib.Path(urlparse(url).path).name, **kwargs)
        self.url = url
        self.mirrors = mirrors
        self._resolved = False

    def resolve(self) -> OnlineResource:
        if self._resolved:
            return self

        redirect_url = _get_redirect_url(self.url)
        if redirect_url == self.url:
            self._resolved = True
            return self

        meta = {
            attr.lstrip("_"): getattr(self, attr)
            for attr in (
                "file_name",
                "sha256",
                "_preprocess",
                "_loader",
            )
        }

        gdrive_id = _get_google_drive_file_id(redirect_url)
        if gdrive_id:
            return GDriveResource(gdrive_id, **meta)

        http_resource = HttpResource(redirect_url, **meta)
        http_resource._resolved = True
        return http_resource

    def _download(self, root: pathlib.Path) -> None:
        if not self._resolved:
            return self.resolve()._download(root)

        for url in itertools.chain((self.url,), self.mirrors):

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


class KaggleDownloadResource(ManualDownloadResource):
    def __init__(self, challenge_url: str, *, file_name: str, **kwargs: Any) -> None:
        instructions = "\n".join(
            (
                "1. Register and login at https://www.kaggle.com",
                f"2. Navigate to {challenge_url}",
                "3. Click 'Join Competition' and follow the instructions there",
                "4. Navigate to the 'Data' tab",
                f"5. Select {file_name} in the 'Data Explorer' and click the download button",
            )
        )
        super().__init__(instructions, file_name=file_name, **kwargs)
