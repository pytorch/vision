import collections.abc
from typing import Any, Callable, Iterator, Optional, Tuple, TypeVar, cast

from torchvision.prototype.features import BoundingBox, Image

T = TypeVar("T")


class SampleQuery:
    def __init__(self, sample: Any) -> None:
        self.sample = sample

    @staticmethod
    def _query_recursively(sample: Any, fn: Callable[[Any], Optional[T]]) -> Iterator[T]:
        if isinstance(sample, (collections.abc.Sequence, collections.abc.Mapping)):
            for item in sample.values() if isinstance(sample, collections.abc.Mapping) else sample:
                yield from SampleQuery._query_recursively(item, fn)
        else:
            result = fn(sample)
            if result is not None:
                yield result

    def query(self, fn: Callable[[Any], Optional[T]]) -> T:
        results = set(self._query_recursively(self.sample, fn))
        if not results:
            raise RuntimeError("Query turned up empty.")
        elif len(results) > 1:
            raise RuntimeError(f"Found more than one result: {results}")

        return results.pop()

    def image_size(self) -> Tuple[int, int]:
        def fn(sample: Any) -> Optional[Tuple[int, int]]:
            if isinstance(sample, Image):
                return cast(Tuple[int, int], sample.shape[-2:])
            elif isinstance(sample, BoundingBox):
                return sample.image_size
            else:
                return None

        return self.query(fn)
