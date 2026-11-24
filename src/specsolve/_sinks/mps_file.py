"""The ``mps_file`` sink: the model as MPS text.

MPS is column-major and the engine's matrix is row-major, so this writer sorts
the matrix by column. The names are the LP writer's — ``x0`` a column, ``c0`` a
row, ``s0`` a set — and every section is written in index order.
"""

from __future__ import annotations

from pathlib import Path
from typing import IO, TYPE_CHECKING

import numpy as np
import polars as pl

from specsolve._names import COEFF, COL, ROW
from specsolve._sinks.capabilities import Capabilities
from specsolve._sinks.tables import SENSE_CODES, ranges
from specsolve._sinks.text import append_lines, chunk_key, digits, number

if TYPE_CHECKING:
    from specsolve._sinks.tables import SinkTables


#: What this writer emits. It writes no quadratic extension section, so a
#: quadratic model is refused rather than written without its quadratic part.
MPS_FILE_CAPABILITIES = Capabilities(supports=frozenset({'integrality', 'sos'}))


#: The MPS spelling of each [`SENSE_CODES`][] comparison; a new sense raises here at import.
_MPS_SENSE = {sense: {'<=': 'L', '>=': 'G', '==': 'E'}[sense] for sense in SENSE_CODES}

#: What an integer column is wrapped in; the name field is a constant nothing reads.
_MARKER = "    MARKER 'MARKER' '{}'"

#: Nonzeros per column chunk; it bounds the writer's peak memory, not its speed.
EMIT_BUDGET = 500_000


def write_mps_file(tables: SinkTables, path: str | Path) -> None:
    """Write the model as MPS text."""
    path = Path(path)
    coefficients, starts = _column_major(tables)

    with open(path, 'wb') as f:
        f.write(b'NAME\n')
        if tables.objective_sense == 'maximize':
            f.write(b'OBJSENSE\n    MAX\n')

        f.write(b'ROWS\n N  obj\n')
        append_lines(_row_lines(tables), f)

        f.write(b'COLUMNS\n')
        width = tables.matrix.height / max(1, tables.column_count)
        for lo, hi in ranges(tables.column_count, EMIT_BUDGET, width):
            owned = coefficients.slice(int(starts[lo]), int(starts[hi] - starts[lo]))
            append_lines(_column_lines(tables, lo, hi, owned), f)

        f.write(b'RHS\n')
        if tables.objective_constant:
            f.write(f'    rhs obj {-tables.objective_constant!r}\n'.encode())
        append_lines(_rhs_lines(tables), f)

        f.write(b'BOUNDS\n')
        _write_bounds(tables, f)

        if tables.sos.height:
            f.write(b'SOS\n')
            append_lines(_set_lines(tables), f)

        f.write(b'ENDATA\n')


def _column_major(tables: SinkTables) -> tuple[pl.DataFrame, np.ndarray[tuple[int, ...], np.dtype[np.int64]]]:
    """The matrix in ``(col, row)`` order, and the offsets a column range slices it by."""
    coefficients = tables.matrix_chunk(0, tables.row_count).sort(COL, ROW)
    counts = np.bincount(coefficients[COL].to_numpy(), minlength=tables.column_count)
    return coefficients, np.concatenate(([0], np.cumsum(counts)))


def _row_lines(tables: SinkTables) -> pl.LazyFrame:
    """One ``ROWS`` line per constraint row, after the objective's ``N``."""
    return tables.rows.lazy().select(
        pl.concat_str(
            pl.lit(' '),
            pl.col('sense').replace_strict(_MPS_SENSE, return_dtype=pl.String),
            pl.lit('  c'),
            digits(pl.col(ROW)),
        )
    )


def _rhs_lines(tables: SinkTables) -> pl.LazyFrame:
    """Each row's right-hand side, in row order."""
    return tables.rows.lazy().select(
        pl.concat_str(pl.lit('    rhs c'), digits(pl.col(ROW)), pl.lit(' '), number(pl.col('rhs')))
    )


def _column_lines(tables: SinkTables, lo: int, hi: int, coefficients: pl.DataFrame) -> pl.LazyFrame:
    """Every ``COLUMNS`` line for columns ``[lo, hi)``, one sorted stream.

    A column's lines occupy ``slots`` consecutive keys — the integer marker, its
    objective coefficient, each coefficient at its row index, the closing
    marker — so one sort fixes both the column order and the order within a
    column. Every column gets an objective line, coefficient or not: a column
    MPS never names is a column the reader does not have.
    """
    slots = tables.row_count + 3

    def _key(within: pl.Expr) -> pl.Expr:
        return chunk_key(pl.col(COL), lo, slots, within)

    columns = (
        tables.cols.lazy().slice(lo, hi - lo).with_row_index(COL, offset=lo).with_columns(pl.col(COL).cast(pl.Int64))
    )
    integral = columns.filter(pl.col('vtype') != 'continuous')
    name = pl.concat_str(pl.lit('    x'), digits(pl.col(COL)))
    cost = columns.join(tables.obj.lazy().with_columns(pl.col(COL).cast(pl.Int64)), on=COL, how='left').select(
        _key(pl.lit(1, dtype=pl.Int64)),
        pl.concat_str(name, pl.lit(' obj '), number(pl.col(COEFF).fill_null(0.0))).alias('line'),
    )
    terms = (
        coefficients.lazy()
        .with_columns(pl.col(COL).cast(pl.Int64))
        .select(
            _key(pl.col(ROW).cast(pl.Int64) + 2),
            pl.concat_str(name, pl.lit(' c'), digits(pl.col(ROW)), pl.lit(' '), number(pl.col(COEFF))).alias('line'),
        )
    )
    markers = [
        integral.select(_key(pl.lit(at, dtype=pl.Int64)), pl.lit(_MARKER.format(word)).alias('line'))
        for at, word in ((0, 'INTORG'), (slots - 1, 'INTEND'))
    ]
    return pl.concat([*markers, cost, terms]).sort('key').select('line')


def _write_bounds(tables: SinkTables, f: IO[bytes]) -> None:
    """Every column's lower bound, then every column's upper.

    A reader given every lower bound does not apply the MPS rule that an ``UP``
    below zero implies an unbounded lower one.
    """
    for keyword, unbounded, column in (('LO', 'MI', 'lb'), ('UP', 'PL', 'ub')):
        name = pl.concat_str(pl.lit(' bnd x'), digits(pl.col(COL)))
        append_lines(
            tables.cols.lazy()
            .with_row_index(COL)
            .select(
                pl.when(pl.col(column).is_infinite())
                .then(pl.concat_str(pl.lit(f' {unbounded}'), name))
                .otherwise(pl.concat_str(pl.lit(f' {keyword}'), name, pl.lit(' '), number(pl.col(column))))
            ),
            f,
        )


def _set_lines(tables: SinkTables) -> pl.LazyFrame:
    """Each special-ordered set as its header line and one line per member.

    The stream arrives grouped by set and ascending in weight, so the key is the
    member's own index, doubled to leave the header a slot before its first
    member.
    """
    members = tables.sos.lazy().with_row_index('ord').with_columns(pl.col('ord').cast(pl.Int64))
    headers = (
        members.group_by('set', maintain_order=True)
        .agg(pl.col('type').first(), pl.col('ord').min())
        .select(
            (pl.col('ord') * 2).alias('key'),
            pl.concat_str(pl.lit(' S'), digits(pl.col('type')), pl.lit(' s'), digits(pl.col('set'))).alias('line'),
        )
    )
    lines = members.select(
        (pl.col('ord') * 2 + 1).alias('key'),
        pl.concat_str(pl.lit('    x'), digits(pl.col(COL)), pl.lit(' '), digits(pl.col('weight'))).alias('line'),
    )
    return pl.concat([headers, lines]).sort('key').select('line')
