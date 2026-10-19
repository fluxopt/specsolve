"""The one place that reads and writes parquet: through polars, or through pyarrow where polars has no parquet.

The browser's polars (Pyodide) is built without parquet. There, a file is
read and written through pyarrow wherever pyarrow can be imported, and
refused by name where it cannot. pyarrow is not a dependency: a page loads it
when it first needs a file, so it is looked for at every call rather than
once.

A frame crosses between the two libraries as Arrow IPC bytes, which both write
and read in the browser; ``DataFrame.to_arrow`` panics on 32-bit WebAssembly.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from functools import cache
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Literal

import polars as pl

from specsolve._relational.collect import collected
from specsolve.errors import SpecsolveError

if TYPE_CHECKING:
    import pyarrow as pa

__all__ = ['columns', 'read', 'require', 'scan', 'write']

#: What a frame is written with, through either library.
COMPRESSION = 'zstd'

#: One file, several read as one in order, or bytes in memory.
type Source = str | Path | Sequence[Path] | BinaryIO


@cache
def _polars_has_parquet() -> bool:
    """Whether this polars reads and writes parquet; asked once, since a polars does not change in a process."""
    try:
        pl.DataFrame({'probe': [0]}).write_parquet(io.BytesIO())
    except AttributeError:  # a polars built without parquet lacks the binding the call reaches for
        return False
    return True


def require(what: str) -> Literal['polars', 'pyarrow']:
    """The library that reads and writes parquet here, or a refusal of *what*, which needs it.

    Raises:
        SpecsolveError: Neither this polars nor an importable pyarrow reads
            parquet.
    """
    if _polars_has_parquet():
        return 'polars'
    if find_spec('pyarrow') is not None:
        return 'pyarrow'
    raise SpecsolveError(
        f"{what} reads or writes parquet, and this polars was built without it, as the browser's (Pyodide) "
        f'is. Load pyarrow first, which specsolve then reads and writes parquet with — in Pyodide, '
        f"`await pyodide.loadPackage('pyarrow')` — or pass each source as a table and read the "
        f'answer off the result or the sweep in memory.'
    )


def read(source: Source, *, columns: Sequence[str] | None = None) -> pl.DataFrame:
    """*source* read whole, *columns* alone where given."""
    given = _one_or_many(source)
    taken = None if columns is None else list(columns)
    if require('reading parquet') == 'polars':
        return pl.read_parquet(given, columns=taken)
    import pyarrow.parquet as pq

    paths = given if isinstance(given, list) else [given]
    return pl.concat([_from_arrow(pq.read_table(path, columns=taken)) for path in paths], how='vertical')


def columns(path: Path) -> list[str]:
    """The column names *path* holds, read off its footer alone."""
    if require('reading parquet') == 'polars':
        return list(pl.read_parquet_schema(path))
    import pyarrow.parquet as pq

    return pq.read_schema(path).names


def scan(source: str | Path | Sequence[Path]) -> pl.LazyFrame:
    """*source* as a lazy frame, read when it is collected; *source* has to outlive it.

    Through pyarrow the footer is read now and the rows whole at the collect,
    so nothing is pushed down into the file; a file gone by then raises
    ``FileNotFoundError``, as polars' own scan does.
    """
    if require('reading parquet') == 'polars':
        return pl.scan_parquet(_one_or_many(source))
    return pl.defer(lambda: read(source), schema=_schema(source))


def write(frame: pl.DataFrame | pl.LazyFrame, target: Path | BinaryIO) -> None:
    """*frame* at *target*; a lazy frame is sunk where polars writes parquet, so it streams to disk."""
    if require('writing parquet') == 'polars':
        if isinstance(frame, pl.LazyFrame):
            frame.sink_parquet(target, compression=COMPRESSION)
        else:
            frame.write_parquet(target, compression=COMPRESSION)
        return
    import pyarrow.parquet as pq

    whole = frame.pipe(collected) if isinstance(frame, pl.LazyFrame) else frame
    pq.write_table(_to_arrow(whole), target, compression=COMPRESSION)


def _one_or_many(source: Source) -> str | Path | BinaryIO | list[Path]:
    """*source* as polars takes it: one path or buffer as it is, several as a list."""
    return source if isinstance(source, (str, Path, io.IOBase)) else list(source)  # pyrefly: ignore[bad-return]  — an IOBase is the BinaryIO it was given as


def _schema(source: Source) -> pl.Schema:
    """The polars schema of *source*'s first file, off its footer alone."""
    import pyarrow.parquet as pq

    given = _one_or_many(source)
    return _from_arrow(pq.read_schema(given[0] if isinstance(given, list) else given).empty_table()).schema


def _to_arrow(frame: pl.DataFrame) -> pa.Table:
    """*frame* as an Arrow table whose string and binary views are plain large strings and binaries.

    The views are cast on the pyarrow side. polars' own oldest IPC, which
    writes no views, panics on 32-bit WebAssembly for a string column
    broadcast from one literal, and pyarrow writes no dictionary of string
    views to parquet.
    """
    import pyarrow as pa

    buffer = io.BytesIO()
    frame.write_ipc_stream(buffer)
    table = pa.ipc.open_stream(buffer.getvalue()).read_all()
    return table.cast(
        pa.schema([field.with_type(_unviewed(field.type)) for field in table.schema], table.schema.metadata)
    )


def _unviewed(kind: pa.DataType) -> pa.DataType:
    """*kind* with a string or binary view, also as a dictionary's values, made a large string or binary."""
    import pyarrow as pa

    if pa.types.is_string_view(kind):
        return pa.large_string()
    if pa.types.is_binary_view(kind):
        return pa.large_binary()
    if pa.types.is_dictionary(kind):
        return pa.dictionary(kind.index_type, _unviewed(kind.value_type), kind.ordered)
    return kind


def _from_arrow(table: pa.Table) -> pl.DataFrame:
    import pyarrow as pa

    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as stream:
        stream.write_table(table)
    return pl.read_ipc_stream(sink.getvalue().to_pybytes())
