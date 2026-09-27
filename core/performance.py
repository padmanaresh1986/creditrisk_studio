from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Iterator, Callable

from core.logging_utils import emit_log


@contextmanager
def timed(label: str, channel: str = "app", level: str = "DEBUG") -> Iterator[None]:
    start = perf_counter()
    try:
        yield
    finally:
        emit_log(f"{label} completed in {perf_counter() - start:.3f}s", level, channel)


def timed_call(fn: Callable, *args, label: str | None = None, channel: str = "app", **kwargs):
    name = label or getattr(fn, "__name__", "operation")
    start = perf_counter()
    try:
        return fn(*args, **kwargs)
    finally:
        emit_log(f"{name} completed in {perf_counter() - start:.3f}s", "DEBUG", channel)
