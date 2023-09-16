import abc
import hashlib
import itertools
import pathlib
from typing import Any, Callable, IO, Literal, NoReturn, Optional, Sequence, Set, Tuple, Union
from urllib.parse import urlparse

from torchdata.datapipes.iter import (
    FileLister,
    FileOpener,
    IterableWrapper,
    IterDataPipe,
    RarArchiveLoader,
    TarArchiveLoader,
    ZipArchiveLoader,
)
from torchvision.datasets.utils import (
    _decompress,
    _detect_file_type,
    _get_google_drive_file_id,
    _get_redirect_url,
    download_file_from_google_drive,
    download_url,
    extract_archive,
)


class OnlineResource(abc.ABC):
    def __init__(
        self,
        *,
        file_name: str,
        sha256: Optional[str] = None,
        preprocess: Optional[Union[Literal["decompress", "extract"], Callable[[pathlib.Path], None]]] = None,
    ) -> None:
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

    def _loader(self, path: pathlib.Path) -> IterDataPipe[Tuple[str, IO]]:
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
            while chunk := file.read(chunk_size):
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
