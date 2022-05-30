from typing import Any, Dict, Optional, Tuple

from ._feature import DEFAULT, Feature


class Label(Feature):
    category: Optional[str]

    @classmethod
    def _parse_meta_data(
        cls,
        category: Optional[str] = DEFAULT,  # type: ignore[assignment]
    ) -> Dict[str, Tuple[Any, Any]]:
        return dict(category=(category, None))
