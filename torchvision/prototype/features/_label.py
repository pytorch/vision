from typing import Dict, Any, Optional
from typing import Tuple

from ._feature import Feature, DEFAULT


class Label(Feature):
    category: Optional[str]

    @classmethod
    def _parse_meta_data(
        cls,
        category: Optional[str] = DEFAULT,
    ) -> Dict[str, Tuple[Any, Any]]:
        return dict(category=(category, None))
