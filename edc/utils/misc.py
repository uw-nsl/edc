from __future__ import annotations

from typing import TYPE_CHECKING

from contextlib import contextmanager

if TYPE_CHECKING:
    from typing import TypeVar
    from collections.abc import Iterable, Iterator

    from contextvars import ContextVar

    T = TypeVar("T")

__all__ = [
    "set_ctx",
    "flatten"
]

@contextmanager
def set_ctx(ctx_var: ContextVar[T], value: T):
    ctx = ctx_var.set(value)
    yield ctx
    ctx_var.reset(ctx)

def flatten(iterable: Iterable[Iterable[T]]) -> Iterator[T]:
    return (elem for sub_iter in iterable for elem in sub_iter)
