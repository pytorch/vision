import pathlib
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

from torchdata.datapipes.iter import Demultiplexer, Filter, IterDataPipe, IterKeyZipper, JsonParser, Mapper, UnBatcher
from torchvision.prototype.datasets.utils import Dataset, EncodedImage, HttpResource, OnlineResource
from torchvision.prototype.datasets.utils._internal import (
    getitem,
    hint_sharding,
    hint_shuffling,
    INFINITE_BUFFER_SIZE,
    path_accessor,
    path_comparator,
)
from torchvision.prototype.tv_tensors import Label

from .._api import register_dataset, register_info

NAME = "clevr"


@register_info(NAME)
def _info() -> Dict[str, Any]:
    return dict()


@register_dataset(NAME)
class CLEVR(Dataset):
    """
    - **homepage**: https://cs.stanford.edu/people/jcjohns/clevr/
    """

    def __init__(
        self, root: Union[str, pathlib.Path], *, split: str = "train", skip_integrity_check: bool = False
    ) -> None:
        self._split = self._verify_str_arg(split, "split", ("train", "val", "test"))

        super().__init__(root, skip_integrity_check=skip_integrity_check)

    def _resources(self) -> List[OnlineResource]:
        archive = HttpResource(
            "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip",
            sha256="5cd61cf1096ed20944df93c9adb31e74d189b8459a94f54ba00090e5c59936d1",
        )
        return [archive]

    def _classify_archive(self, data: Tuple[str, Any]) -> Optional[int]:
        path = pathlib.Path(data[0])
        if path.parents[1].name == "images":
            return 0
        elif path.parent.name == "scenes":
            return 1
        else:
            return None

    def _filter_scene_anns(self, data: Tuple[str, Any]) -> bool:
        key, _ = data
        return key == "scenes"

    def _add_empty_anns(self, data: Tuple[str, BinaryIO]) -> Tuple[Tuple[str, BinaryIO], None]:
        return data, None

    def _prepare_sample(self, data: Tuple[Tuple[str, BinaryIO], Optional[Dict[str, Any]]]) -> Dict[str, Any]:
        image_data, scenes_data = data
        path, buffer = image_data

        return dict(
            path=path,
            image=EncodedImage.from_file(buffer),
            label=Label(len(scenes_data["objects"])) if scenes_data else None,
        )

    def _datapipe(self, resource_dps: List[IterDataPipe]) -> IterDataPipe[Dict[str, Any]]:
        archive_dp = resource_dps[0]
        images_dp, scenes_dp = Demultiplexer(
            archive_dp,
            2,
            self._classify_archive,
            drop_none=True,
            buffer_size=INFINITE_BUFFER_SIZE,
        )

        images_dp = Filter(images_dp, path_comparator("parent.name", self._split))
        images_dp = hint_shuffling(images_dp)
        images_dp = hint_sharding(images_dp)

        if self._split != "test":
            scenes_dp = Filter(scenes_dp, path_comparator("name", f"CLEVR_{self._split}_scenes.json"))
            scenes_dp = JsonParser(scenes_dp)
            scenes_dp = Mapper(scenes_dp, getitem(1, "scenes"))
            scenes_dp = UnBatcher(scenes_dp)

            dp = IterKeyZipper(
                images_dp,
                scenes_dp,
                key_fn=path_accessor("name"),
                ref_key_fn=getitem("image_filename"),
                buffer_size=INFINITE_BUFFER_SIZE,
            )
        else:
            for _, file in scenes_dp:
                file.close()
            dp = Mapper(images_dp, self._add_empty_anns)

        return Mapper(dp, self._prepare_sample)

    def __len__(self) -> int:
        return 70_000 if self._split == "train" else 15_000
