import io
import os
import os.path
import pathlib
from typing import Union, Tuple, List, Dict, Any

from torch.utils.data import IterDataPipe
from torch.utils.data.datapipes.iter import FileLister, Concater, FileLoader, Mapper

__all__ = ["from_image_folder"]


def _collate(
    data: Tuple[str, io.IOBase],
    *,
    label: int,
    category: str,
) -> Dict[str, Any]:
    path, image_file = data
    return dict(
        path=path,
        image=image_file,
        label=label,
        category=category,
    )


def from_image_folder(
    root: Union[str, pathlib.Path],
) -> Tuple[IterDataPipe, List[str]]:
    root = pathlib.Path(root).expanduser().resolve()
    categories = sorted({item.name for item in os.scandir(root) if item.is_dir})
    category_dps = []
    category_dp: IterDataPipe
    for label, category in enumerate(categories):
        category_dp = FileLister(str(root / category), recursive=True)
        category_dp = FileLoader(category_dp)
        category_dp = Mapper(
            category_dp,
            _collate,
            fn_kwargs=dict(label=label, category=category),
        )
        category_dps.append(category_dp)
    return Concater(*category_dps), categories
