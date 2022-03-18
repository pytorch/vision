import pathlib
from typing import Any, Dict, List, Optional, Tuple, BinaryIO

from torchdata.datapipes.iter import IterDataPipe, Mapper, Filter, IterKeyZipper, Demultiplexer, JsonParser, UnBatcher
from torchvision.prototype.datasets.utils import (
    Dataset,
    DatasetConfig,
    DatasetInfo,
    HttpResource,
    OnlineResource,
)
from torchvision.prototype.datasets.utils._internal import (
    INFINITE_BUFFER_SIZE,
    hint_sharding,
    hint_shuffling,
    path_comparator,
    path_accessor,
    getitem,
)
from torchvision.prototype.features import Label, EncodedImage


class CLEVR(Dataset):
    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            "clevr",
            homepage="https://cs.stanford.edu/people/jcjohns/clevr/",
            valid_options=dict(split=("train", "val", "test")),
        )

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
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

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
    ) -> IterDataPipe[Dict[str, Any]]:
        archive_dp = resource_dps[0]
        images_dp, scenes_dp = Demultiplexer(
            archive_dp,
            2,
            self._classify_archive,
            drop_none=True,
            buffer_size=INFINITE_BUFFER_SIZE,
        )

        images_dp = Filter(images_dp, path_comparator("parent.name", config.split))
        images_dp = hint_sharding(images_dp)
        images_dp = hint_shuffling(images_dp)

        if config.split != "test":
            scenes_dp = Filter(scenes_dp, path_comparator("name", f"CLEVR_{config.split}_scenes.json"))
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
            dp = Mapper(images_dp, self._add_empty_anns)

        return Mapper(dp, self._prepare_sample)
