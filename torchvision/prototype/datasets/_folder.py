import io
import os
import os.path
import pathlib
from typing import Callable, Optional, Collection
from typing import Union, Tuple, List, Dict, Any

import torch
from torch.utils.data import IterDataPipe
from torch.utils.data.datapipes.iter import FileLister, FileLoader, Mapper, Shuffler, Filter
from torchvision.prototype.datasets.decoder import pil
from torchvision.prototype.datasets.utils._internal import INFINITE_BUFFER_SIZE


__all__ = ["from_data_folder", "from_image_folder"]


def _is_not_top_level_file(path: str, *, root: pathlib.Path) -> bool:
    rel_path = pathlib.Path(path).relative_to(root)
    return rel_path.is_dir() or rel_path.parent != pathlib.Path(".")


def _collate_and_decode_data(
    data: Tuple[str, io.IOBase],
    *,
    root: pathlib.Path,
    categories: List[str],
    decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
) -> Dict[str, Any]:
    path, buffer = data
    data = decoder(buffer) if decoder else buffer
    category = pathlib.Path(path).relative_to(root).parts[0]
    label = torch.tensor(categories.index(category))
    return dict(
        path=path,
        data=data,
        label=label,
        category=category,
    )


def from_data_folder(
    root: Union[str, pathlib.Path],
    *,
    decoder: Optional[Callable[[io.IOBase], torch.Tensor]] = None,
    valid_extensions: Optional[Collection[str]] = None,
    recursive: bool = True,
) -> Tuple[IterDataPipe, List[str]]:
    root = pathlib.Path(root).expanduser().resolve()
    categories = sorted(entry.name for entry in os.scandir(root) if entry.is_dir())
    masks: Union[List[str], str] = [f"*.{ext}" for ext in valid_extensions] if valid_extensions is not None else ""
    dp = FileLister(str(root), recursive=recursive, masks=masks)
    dp: IterDataPipe = Filter(dp, _is_not_top_level_file, fn_kwargs=dict(root=root))
    dp = Shuffler(dp, buffer_size=INFINITE_BUFFER_SIZE)
    dp = FileLoader(dp)
    return (
        Mapper(dp, _collate_and_decode_data, fn_kwargs=dict(root=root, categories=categories, decoder=decoder)),
        categories,
    )


def _data_to_image_key(sample: Dict[str, Any]) -> Dict[str, Any]:
    sample["image"] = sample.pop("data")
    return sample


def from_image_folder(
    root: Union[str, pathlib.Path],
    *,
    decoder: Optional[Callable[[io.IOBase], torch.Tensor]] = pil,
    valid_extensions: Collection[str] = ("jpg", "jpeg", "png", "ppm", "bmp", "pgm", "tif", "tiff", "webp"),
    **kwargs: Any,
) -> Tuple[IterDataPipe, List[str]]:
    valid_extensions = [valid_extension for ext in valid_extensions for valid_extension in (ext.lower(), ext.upper())]
    dp, categories = from_data_folder(root, decoder=decoder, valid_extensions=valid_extensions, **kwargs)
    return Mapper(dp, _data_to_image_key), categories
