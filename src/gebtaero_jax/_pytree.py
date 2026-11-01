"""Tiny helper to declare frozen dataclasses that are JAX pytrees.

Fields declared with ``static_field()`` become pytree metadata (hashable,
not traced); every other field is a pytree child and can hold arrays that are
differentiated through.
"""

from __future__ import annotations

import dataclasses
from typing import Any, TypeVar

import jax

T = TypeVar("T")


def static_field(**kwargs: Any) -> Any:
    """Dataclass field that is treated as static (non-differentiable) metadata."""
    metadata = dict(kwargs.pop("metadata", {}))
    metadata["static"] = True
    return dataclasses.field(metadata=metadata, **kwargs)


def pytree_dataclass(cls: type[T]) -> type[T]:
    """Make ``cls`` a frozen dataclass registered as a JAX pytree."""
    cls = dataclasses.dataclass(frozen=True)(cls)
    fields = dataclasses.fields(cls)
    data = [f.name for f in fields if not f.metadata.get("static", False)]
    meta = [f.name for f in fields if f.metadata.get("static", False)]
    jax.tree_util.register_dataclass(cls, data_fields=data, meta_fields=meta)

    def replace(self: T, **changes: Any) -> T:
        return dataclasses.replace(self, **changes)

    cls.replace = replace  # type: ignore[attr-defined]
    return cls
