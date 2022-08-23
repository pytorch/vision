import pathlib
import warnings
from typing import List, Any, Dict, Optional, Tuple, BinaryIO

from torch.utils.model_zoo import tqdm
from torchdata.datapipes.iter import IterDataPipe, Demultiplexer, LineReader, Zipper, Mapper, IterKeyZipper
from torchvision.prototype.datasets.utils import Dataset, DatasetInfo, DatasetConfig, OnlineResource, HttpResource
from torchvision.prototype.datasets.utils._internal import (
    hint_sharding,
    hint_shuffling,
    INFINITE_BUFFER_SIZE,
    path_accessor,
)
from torchvision.prototype.features import EncodedImage


class SBU(Dataset):

    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            name="sbu",
            homepage="http://www.cs.virginia.edu/~vicente/sbucaptions/",
        )

    def _preprocess(self, path: pathlib.Path) -> pathlib.Path:
        folder = OnlineResource._extract(path)
        data_folder = folder / "dataset"
        image_folder = data_folder / "images"
        image_folder.mkdir()
        broken_urls = []
        with open(data_folder / "SBU_captioned_photo_dataset_urls.txt") as fh:
            urls = fh.read().splitlines()

            # TODO: Use workers to download images
            for url in tqdm(urls):
                try:
                    # TODO: suppress print statements within HttpResource.download()
                    HttpResource(url).download(image_folder)
                except Exception:
                    broken_urls.append(url)

        if broken_urls:
            broken_urls_file = folder.parent / "broken_urls.txt"
            warnings.warn(
                f"Failed to download {len(broken_urls)} ({len(broken_urls) / len(urls):.2%}) images. "
                f"They are logged in {broken_urls_file}."
            )
            with open(broken_urls_file, "w") as fh:
                fh.write("\n".join(broken_urls) + "\n")

        return folder

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        return [
            HttpResource(
                "http://www.cs.virginia.edu/~vicente/sbucaptions/SBUCaptionedPhotoDataset.tar.gz",
                sha256="2bf37d5e1c9e1c6eae7d5103030d58a7f2117fc5e8c6aa9620f0df165acebf09",
                preprocess=self._preprocess,
            )
        ]

    def _classify_files(self, data: Tuple[str, Any]) -> Optional[int]:
        path = pathlib.Path(data[0])
        if path.parent.name == "images":
            return 0
        elif path.name == "SBU_captioned_photo_dataset_urls.txt":
            return 1
        elif path.name == "SBU_captioned_photo_dataset_captions.txt":
            return 2
        else:
            return None

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
    ) -> IterDataPipe[Dict[str, Any]]:

        images_dp, urls_dp, captions_dp = Demultiplexer(
            resource_dps[0], 3, self._classify_files, drop_none=True, buffer_size=INFINITE_BUFFER_SIZE
        )

        images_dp = hint_shuffling(images_dp)
        images_dp = hint_sharding(images_dp)

        urls_dp = LineReader(urls_dp, decode=True, return_path=False)
        captions_dp = LineReader(captions_dp, decode=True, return_path=False)
        anns_dp = Zipper(urls_dp, captions_dp)

        dp = IterKeyZipper(images_dp, anns_dp, path_accessor("name"), buffer_size=INFINITE_BUFFER_SIZE)
        return Mapper(dp, self._prepare_sample)

    def _prepare_sample(self, data: Tuple[Tuple[str, BinaryIO], Tuple[str, str]]) -> Dict[str, Any]:
        (path, buffer), (_, caption) = data
        return dict(
            path=path,
            image=EncodedImage.from_file(buffer),
            caption=caption.strip(),
        )
