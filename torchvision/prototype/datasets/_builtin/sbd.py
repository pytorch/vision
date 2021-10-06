import io
import pathlib
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import IterDataPipe
from torch.utils.data.datapipes.iter import Mapper, TarArchiveReader, Shuffler, Demultiplexer, Filter
from torchdata.datapipes.iter import KeyZipper, LineReader
from torchvision.prototype.datasets.utils import (
    Dataset,
    DatasetConfig,
    DatasetInfo,
    HttpResource,
    OnlineResource,
    DatasetType,
)
from torchvision.prototype.datasets.utils._internal import (
    create_categories_file,
    INFINITE_BUFFER_SIZE,
    read_mat,
    getitem,
    path_accessor,
    path_comparator,
)

HERE = pathlib.Path(__file__).parent


class SBD(Dataset):
    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            "sbd",
            type=DatasetType.IMAGE,
            categories=HERE / "caltech256.categories",
            homepage="http://home.bharathh.info/pubs/codes/SBD/download.html",
            valid_options=dict(
                split=("train", "val", "train_noval"),
                boundaries=(True, False),
                segmentation=(False, True),
            ),
        )

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        archive = HttpResource(
            "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz",
            sha256="6a5a2918d5c73ce032fdeba876574d150d9d04113ab87540a1304cbcc715be53",
        )
        extra_split = HttpResource(
            "http://home.bharathh.info/pubs/codes/SBD/train_noval.txt",
            sha256="0b2068f7a359d2907431803e1cd63bf6162da37d7d503b589d3b08c6fd0c2432",
        )
        return [archive, extra_split]

    def _classify_archive(self, data: Tuple[str, Any]) -> Optional[int]:
        path = pathlib.Path(data[0])
        parent, grandparent, *_ = path.parents

        if parent.name == "dataset":
            return 0
        elif grandparent.name == "dataset":
            if parent.name == "img":
                return 1
            elif parent.name == "cls":
                return 2
            else:
                return None
        else:
            return None

    def _decode_ann(
        self, data: Dict[str, Any], *, decode_boundaries: bool, decode_segmentation: bool
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        raw_anns = data["GTcls"][0]
        raw_boundaries = raw_anns["Boundaries"][0]
        raw_segmentation = raw_anns["Segmentation"][0]

        # the boundaries are stored in sparse CSC format, which is not supported by PyTorch
        boundaries = (
            torch.as_tensor(np.stack([raw_boundary[0].toarray() for raw_boundary in raw_boundaries]))
            if decode_boundaries
            else None
        )
        segmentation = torch.as_tensor(raw_segmentation) if decode_segmentation else None

        return boundaries, segmentation

    def _collate_and_decode_sample(
        self,
        data: Tuple[Tuple[Any, Tuple[str, io.IOBase]], Tuple[str, io.IOBase]],
        *,
        config: DatasetConfig,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
    ) -> Dict[str, Any]:
        split_and_image_data, ann_data = data
        _, image_data = split_and_image_data
        image_path, image_buffer = image_data
        ann_path, ann_buffer = ann_data

        image = decoder(image_buffer) if decoder else image_buffer

        if config.boundaries or config.segmentation:
            boundaries, segmentation = self._decode_ann(
                read_mat(ann_buffer), decode_boundaries=config.boundaries, decode_segmentation=config.segmentation
            )
        else:
            boundaries = segmentation = None

        return dict(
            image_path=image_path,
            image=image,
            ann_path=ann_path,
            boundaries=boundaries,
            segmentation=segmentation,
        )

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
    ) -> IterDataPipe[Dict[str, Any]]:
        archive_dp, extra_split_dp = resource_dps

        archive_dp = resource_dps[0]
        archive_dp = TarArchiveReader(archive_dp)
        split_dp, images_dp, anns_dp = Demultiplexer(
            archive_dp,
            3,
            self._classify_archive,  # type: ignore[arg-type]
            buffer_size=INFINITE_BUFFER_SIZE,
            drop_none=True,
        )

        if config.split == "train_noval":
            split_dp = extra_split_dp
        split_dp = LineReader(split_dp, decode=True)
        split_dp = Shuffler(split_dp)

        dp = split_dp
        for level, data_dp in enumerate((images_dp, anns_dp)):
            dp = KeyZipper(
                dp,
                data_dp,
                key_fn=getitem(*[0] * level, 1),
                ref_key_fn=path_accessor("stem"),
                buffer_size=INFINITE_BUFFER_SIZE,
            )
        return Mapper(dp, self._collate_and_decode_sample, fn_kwargs=dict(config=config, decoder=decoder))

    def generate_categories_file(self, root: Union[str, pathlib.Path]) -> None:
        dp = self.resources(self.default_config)[0].to_datapipe(pathlib.Path(root) / self.name)
        dp = TarArchiveReader(dp)
        dp: IterDataPipe = Filter(dp, path_comparator("name", "category_names.m"))
        dp = LineReader(dp)
        dp: IterDataPipe = Mapper(dp, bytes.decode, input_col=1)
        lines = tuple(zip(*iter(dp)))[1]

        pattern = re.compile(r"\s*'(?P<category>\w+)';\s*%(?P<label>\d+)")
        categories_and_labels = [
            pattern.match(line).groups()  # type: ignore[union-attr]
            # the first and last line contain no information
            for line in lines[1:-1]
        ]
        categories = tuple(
            zip(*sorted(categories_and_labels, key=lambda category_and_label: int(category_and_label[1])))
        )[0]

        create_categories_file(HERE, self.name, categories)


if __name__ == "__main__":
    from torchvision.prototype.datasets import home

    root = home()
    SBD().generate_categories_file(root)
