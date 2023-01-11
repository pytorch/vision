import enum
from typing import Sequence, Type, TypeVar

T = TypeVar("T", bound=enum.Enum)



# Could we just define 
# class ColorSpace(enum.Enum):
#    RGB = "RGB"  # instead of auto()
#    ...
# and avoid this helper class altogether?
# I sort of remember having some issue with this class before (perhaps with the
# WeightsEnum?)
class StrEnumMeta(enum.EnumMeta):
    auto = enum.auto

    def from_str(self: Type[T], member: str) -> T:  # type: ignore[misc]
        try:
            return self[member]
        except KeyError:
            # TODO: use `add_suggestion` from torchvision.prototype.utils._internal to improve the error message as
            #  soon as it is migrated.
            raise ValueError(f"Unknown value '{member}' for {self.__name__}.") from None


# Can this just be a simple class inherited from Enum (without the metaclass)?
class StrEnum(enum.Enum, metaclass=StrEnumMeta):
    pass


def sequence_to_str(seq: Sequence, separate_last: str = "") -> str:
    if not seq:
        return ""
    if len(seq) == 1:
        return f"'{seq[0]}'"

    head = "'" + "', '".join([str(item) for item in seq[:-1]]) + "'"
    tail = f"{'' if separate_last and len(seq) == 2 else ','} {separate_last}'{seq[-1]}'"

    return head + tail
