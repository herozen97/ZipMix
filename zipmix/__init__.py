"""Public ZipMix package API."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["ZipMixModel"]

if TYPE_CHECKING:
    from zipmix.model import ZipMixModel


def __getattr__(name: str):
    if name == "ZipMixModel":
        from zipmix.model import ZipMixModel as _ZipMixModel

        return _ZipMixModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
