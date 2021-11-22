import io
import pathlib
import re
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, BinaryIO

import torch
from torchdata.datapipes.iter import (
    IterDataPipe,
    Demultiplexer,
    Mapper,
    Shuffler,
    Filter,
    IterKeyZipper,
    ZipArchiveReader,
)
from torchvision.prototype.datasets.utils import (
    Dataset,
    DatasetConfig,
    DatasetInfo,
    HttpResource,
    OnlineResource,
    DatasetType,
)
from torchvision.prototype.datasets.utils._internal import INFINITE_BUFFER_SIZE, read_flo, InScenePairer, path_accessor


class SINTEL(Dataset):

    _FILE_NAME_PATTERN = re.compile(r"(frame|image)_(?P<idx>\d+)[.](flo|png)")

    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            "sintel",
            type=DatasetType.IMAGE,
            homepage="http://sintel.is.tue.mpg.de/",
            valid_options=dict(
                split=("train", "test"),
                pass_name=("clean", "final", "both"),
            ),
        )

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        archive = HttpResource(
            "http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip",
            sha256="bdc80abbe6ae13f96f6aa02e04d98a251c017c025408066a00204cd2c7104c5f",
        )
        return [archive]

    def _filter_split(self, data: Tuple[str, Any], *, split: str) -> bool:
        path = pathlib.Path(data[0])
        # The dataset contains has the folder "training", while allowed options for `split` are
        # "train" and "test", we don't check for equality here ("train" != "training") and instead
        # check if split is in the folder name
        return split in path.parents[2].name

    def _filter_pass_name_and_flow(self, data: Tuple[str, Any], *, pass_name: str) -> bool:
        path = pathlib.Path(data[0])
        if pass_name == "both":
            matched = path.parents[1].name in ["clean", "final", "flow"]
        else:
            matched = path.parents[1].name in [pass_name, "flow"]
        return matched

    def _classify_archive(self, data: Tuple[str, Any], *, pass_name: str) -> Optional[int]:
        path = pathlib.Path(data[0])
        suffix = path.suffix
        if suffix == ".flo":
            return 0
        elif suffix == ".png":
            return 1
        else:
            return None

    def _flows_key(self, data: Tuple[str, Any]) -> Tuple[str, int]:
        path = pathlib.Path(data[0])
        category = path.parent.name
        idx = int(self._FILE_NAME_PATTERN.match(path.name).group("idx"))  # type: ignore[union-attr]
        return category, idx

    def _add_fake_flow_data(self, data: Tuple[str, Any]) -> Tuple[Tuple[None, None], Tuple[str, Any]]:
        return ((None, None), data)

    def _images_key(self, data: Tuple[Tuple[str, Any], Tuple[str, Any]]) -> Tuple[str, int]:
        return self._flows_key(data[0])

    def _collate_and_decode_sample(
        self,
        data: Tuple[Tuple[Optional[str], Optional[BinaryIO]], Tuple[Tuple[str, BinaryIO], Tuple[str, BinaryIO]]],
        *,
        decoder: Optional[Callable[[BinaryIO], torch.Tensor]],
    ) -> Dict[str, Any]:
        flow_data, images_data = data
        flow_path, flow_buffer = flow_data
        image1_data, image2_data = images_data
        image1_path, image1_buffer = image1_data
        image2_path, image2_buffer = image2_data

        return dict(
            image1=decoder(image1_buffer) if decoder else image1_buffer,
            image1_path=image1_path,
            image2=decoder(image2_buffer) if decoder else image2_buffer,
            image2_path=image2_path,
            flow=read_flo(flow_buffer) if flow_buffer else None,
            flow_path=flow_path,
            scene=pathlib.Path(image1_path).parent.name,
        )

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
    ) -> IterDataPipe[Dict[str, Any]]:
        dp = resource_dps[0]
        archive_dp = ZipArchiveReader(dp)

        curr_split = Filter(archive_dp, self._filter_split, fn_kwargs=dict(split=config.split))
        filtered_curr_split = Filter(
            curr_split, self._filter_pass_name_and_flow, fn_kwargs=dict(pass_name=config.pass_name)
        )
        if config.split == "train":
            flo_dp, pass_images_dp = Demultiplexer(
                filtered_curr_split,
                2,
                partial(self._classify_archive, pass_name=config.pass_name),
                drop_none=True,
                buffer_size=INFINITE_BUFFER_SIZE,
            )
            flo_dp = Shuffler(flo_dp, buffer_size=INFINITE_BUFFER_SIZE)
            pass_images_dp: IterDataPipe[Tuple[str, Any], Tuple[str, Any]] = InScenePairer(
                pass_images_dp, scene_fn=path_accessor("parent", "name")
            )
            zipped_dp = IterKeyZipper(
                flo_dp,
                pass_images_dp,
                key_fn=self._flows_key,
                ref_key_fn=self._images_key,
            )
        else:
            pass_images_dp = Shuffler(filtered_curr_split, buffer_size=INFINITE_BUFFER_SIZE)
            pass_images_dp = InScenePairer(pass_images_dp, scene_fn=path_accessor("parent", "name"))
            zipped_dp = Mapper(pass_images_dp, self._add_fake_flow_data)

        return Mapper(zipped_dp, self._collate_and_decode_sample, fn_kwargs=dict(decoder=decoder))
