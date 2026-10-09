"""The relational lane on polars' in-memory engine, as `docs/howto/small-models.md` tells a user to run it.

Every verb is the `specsolve` arm's, run inside ``pl.Config(engine_affinity=...)``.
The timed rounds share one process with every other arm, so the affinity is
set for the call and restored after it, never left on the process.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from bench.arms import specsolve

SINKS = specsolve.SINKS
REQUIRES = specsolve.REQUIRES

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from bench.arms import Counts

prepare = specsolve.prepare


def _in_memory(verb: Callable[..., Any], *args: Any) -> Any:
    """*verb* called with polars' engine affinity on in-memory, and the affinity restored after it."""
    import polars as pl

    with pl.Config(engine_affinity='in-memory'):
        return verb(*args)


def build_and_emit(sink: str, prepared: tuple[Path, dict[str, str]]) -> Counts:
    """The `specsolve` arm's build and hand-over, on the in-memory engine."""
    return _in_memory(specsolve.build_and_emit, sink, prepared)


def build_only(prepared: tuple[Path, dict[str, str]]) -> Counts:
    """The `specsolve` arm's build alone, on the in-memory engine."""
    return _in_memory(specsolve.build_only, prepared)


def objective(prepared: tuple[Path, dict[str, str]]) -> float:
    """The `specsolve` arm's solve, on the in-memory engine."""
    return _in_memory(specsolve.objective, prepared)
