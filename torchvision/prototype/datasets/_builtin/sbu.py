import io
from typing import List, Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image as pil_image
from torchdata.datapipes.iter import IterDataPipe, Demultiplexer, LineReader, HttpReader, Zipper, Mapper
from torchvision.prototype import features
from torchvision.prototype.datasets.utils import Dataset, DatasetInfo, DatasetConfig, OnlineResource, HttpResource
from torchvision.prototype.datasets.utils._internal import hint_sharding, hint_shuffling


class SBU(Dataset):
    PHOTO_URLS = 0
    PHOTO_CAPTIONS = 1

    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            name="sbu",
            homepage="http://www.cs.virginia.edu/~vicente/sbucaptions/",
        )

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        return [
            HttpResource(
                "http://www.cs.virginia.edu/~vicente/sbucaptions/SBUCaptionedPhotoDataset.tar.gz",
                sha256="2bf37d5e1c9e1c6eae7d5103030d58a7f2117fc5e8c6aa9620f0df165acebf09",
            )
        ]

    def _classify_files(self, data: Tuple[str, Any]) -> Optional[int]:
        path, stream = data
        if path.endswith("SBU_captioned_photo_dataset_urls.txt"):
            return SBU.PHOTO_URLS
        elif path.endswith("SBU_captioned_photo_dataset_captions.txt"):
            return SBU.PHOTO_CAPTIONS
        else:
            return None

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
    ) -> IterDataPipe[Dict[str, Any]]:

        photo_urls_dp, photo_captions_dp = Demultiplexer(
            resource_dps[0], 2, self._classify_files, drop_none=True, buffer_size=-1
        )

        photo_urls_dp = LineReader(photo_urls_dp, decode=True, return_path=False)
        photo_urls_dp = HttpReader(photo_urls_dp)

        photo_captions_dp = LineReader(photo_captions_dp, decode=True, return_path=False)

        dp = Zipper(photo_urls_dp, photo_captions_dp)
        dp = hint_shuffling(dp)
        dp = hint_sharding(dp)

        return Mapper(dp, self._prepare_sample)

    def _prepare_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        image, caption = sample
        _, image = image

        # TODO: handle missing images
        # TODO: check for method to convert to tensor without PIL and Numpy
        image = pil_image.open(io.BytesIO(image.data))
        image = np.array(image)

        return {"image": features.Image(image), "caption": caption}
