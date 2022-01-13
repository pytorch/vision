from collections import UserDict
from typing import Any, TypeVar, Dict, Mapping

from torchvision.prototype import features
from torchvision.prototype.transforms import Transform

from . import functional as F

T = TypeVar("T", bound=features.Feature)


class DecodeImages(Transform):
    def supports(self, obj: Any) -> bool:
        return (obj if isinstance(obj, type) else type(obj)) is features.EncodedImage

    def _dispatch(self, feature: T, params: Dict[str, Any]) -> T:
        if not self.supports(feature):
            return feature

        return features.Image(F.decode_image_with_pil(feature))


class DecodeVideos(Transform):
    def supports(self, obj: Any) -> bool:
        return (obj if isinstance(obj, type) else type(obj)) is features.EncodedVideo

    def _dispatch(self, feature: T, params: Dict[str, Any]) -> Mapping[str, Any]:
        if not self.supports(feature):
            return feature

        video, audio, video_meta = F.decode_video_with_av(feature)
        decoded_video = UserDict(video=features.Feature(video), audio=features.Feature(audio), video_meta=video_meta)
        decoded_video.__inline__ = True

        return decoded_video
