import functools
import os
import os.path
import pathlib
from typing import Any, BinaryIO, Collection, Dict, List, Optional, Tuple, Union

from torchdata.datapipes.iter import FileLister, FileOpener, Filter, IterDataPipe, Mapper
from torchvision.prototype.datasets.utils import EncodedData, EncodedImage
from torchvision.prototype.datasets.utils._internal import hint_sharding, hint_shuffling
from torchvision.prototype.tv_tensors import Label


__all__ = ["from_data_folder", "from_image_folder"]


def _is_not_top_level_file(path: str, *, root: pathlib.Path) -> bool:
    rel_path = pathlib.Path(path).relative_to(root)
    return rel_path.is_dir() or rel_path.parent != pathlib.Path(".")


def _prepare_sample(
    data: Tuple[str, BinaryIO],
    *,
    root: pathlib.Path,
    categories: List[str],
) -> Dict[str, Any]:
    path, buffer = data
    category = pathlib.Path(path).relative_to(root).parts[0]
    return dict(
        path=path,
        data=EncodedData.from_file(buffer),
        label=Label.from_category(category, categories=categories),
    )


def from_data_folder(
    root: Union[str, pathlib.Path],
    *,
    valid_extensions: Optional[Collection[str]] = None,
    recursive: bool = True,
) -> Tuple[IterDataPipe, List[str]]:
    root = pathlib.Path(root).expanduser().resolve()
    categories = sorted(entry.name for entry in os.scandir(root) if entry.is_dir())
    masks: Union[List[str], str] = [f"*.{ext}" for ext in valid_extensions] if valid_extensions is not None else ""
    dp = FileLister(str(root), recursive=recursive, masks=masks)
    dp: IterDataPipe = Filter(dp, functools.partial(_is_not_top_level_file, root=root))
    dp = hint_sharding(dp)
    dp = hint_shuffling(dp)
    dp = FileOpener(dp, mode="rb")
    return Mapper(dp, functools.partial(_prepare_sample, root=root, categories=categories)), categories


def _data_to_image_key(sample: Dict[str, Any]) -> Dict[str, Any]:
    sample["image"] = EncodedImage(sample.pop("data").data)
    return sample


def from_image_folder(
    root: Union[str, pathlib.Path],
    *,
    valid_extensions: Collection[str] = ("jpg", "jpeg", "png", "ppm", "bmp", "pgm", "tif", "tiff", "webp"),
    **kwargs: Any,
) -> Tuple[IterDataPipe, List[str]]:
    valid_extensions = [valid_extension for ext in valid_extensions for valid_extension in (ext.lower(), ext.upper())]
    dp, categories = from_data_folder(root, valid_extensions=valid_extensions, **kwargs)
    return Mapper(dp, _data_to_image_key), categories
