import enum


class StrEnumMeta(enum.EnumMeta):
    auto = enum.auto

    def from_str(self, member: str):
        try:
            return self[member]
        except KeyError:
            # TODO: use `add_suggestion` from torchvision.prototype.utils._internal to improve the error message as
            #  soon as it is migrated.
            raise ValueError(f"Unknown value '{member}' for {self.__name__}.") from None


class StrEnum(enum.Enum, metaclass=StrEnumMeta):
    pass
