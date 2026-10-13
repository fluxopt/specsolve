"""Which parameters a build streams from their parquet file, and the copies it streams them from.

A parameter given as a parquet path of at least [`ROWS`][] rows is not held in
memory: the build reads it from the file each time it needs it. It reads a
copy taken at the build, so a file rewritten afterwards reaches neither the
model, an expression read off it, nor its archive. A smaller parameter is held
in memory, where reading it again costs nothing.
"""

from __future__ import annotations

import shutil
import tempfile
import weakref
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from specsolve.frames import as_frame
from specsolve.relational.collect import collected_all

if TYPE_CHECKING:
    from collections.abc import Mapping

    from specsolve.inputs import Source

__all__ = ['ROWS', 'Copies', 'large']

#: The rows from which a parameter streams.
ROWS = 250_000


def large(sources: Mapping[str, Source]) -> set[str]:
    """The names in *sources* given as a parquet path of at least [`ROWS`][] rows.

    The rows are counted off each file's footer, every footer in one
    concurrent pass, since opening a file costs a fixed time however small it is.
    """
    scans = {
        name: scan
        for name, source in sources.items()
        if isinstance(source, (str, Path)) and (scan := as_frame(source)) is not None
    }
    counts = collected_all(scan.select(pl.len()) for scan in scans.values())
    return {name for name, count in zip(scans, counts, strict=True) if count.item() >= ROWS}


class Copies:
    """The parquet files one build streams its large parameters from.

    The directory is made at the first copy and deleted once nothing refers to
    this object: the model that built from it, and every result whose
    expressions read from it. So a result outlives ``model.close()`` with
    its reads intact.

    Attributes:
        names: The parameters streamed from a copy.
    """

    def __init__(self) -> None:
        self._directory: Path | None = None
        self.names: set[str] = set()

    def take(self, name: str, source: str | Path) -> Path:
        """A copy of *source* that parameter *name* may be streamed from."""
        if self._directory is None:
            self._directory = Path(tempfile.mkdtemp(prefix='specsolve-'))
            weakref.finalize(self, shutil.rmtree, self._directory, ignore_errors=True)
        copy = self._directory / f'{name}.parquet'
        shutil.copyfile(source, copy)
        return copy

    def keep(self, name: str) -> None:
        """Mark *name* as streamed, once its copy has passed the checks."""
        self.names.add(name)
