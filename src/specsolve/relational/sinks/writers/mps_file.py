"""The ``mps_file`` sink: the model as MPS text.

MPS is column-major and the engine's matrix is row-major, so this writer sorts
the matrix by column. The names are the LP writer's — ``x0`` a column, ``c0`` a
row, ``s0`` a set, or the declared names both take — and every section is
written in label order.
"""

from __future__ import annotations

from pathlib import Path
from typing import IO, TYPE_CHECKING

import numpy as np
import polars as pl

from specsolve.relational.sinks.capabilities import Capabilities
from specsolve.relational.sinks.handoff import SENSE_CODES, ranges
from specsolve.relational.sinks.writers.base import NUMBERED, Names, chunk_key, digits, number, sink

if TYPE_CHECKING:
    from specsolve.relational.sinks.handoff import Handoff


#: What this writer emits. It writes no quadratic extension section, so a
#: quadratic model is refused rather than written without its quadratic part.
MPS_FILE_CAPABILITIES = Capabilities(supports={'integrality': 'native', 'sos': 'native'})


#: The MPS spelling of each [`SENSE_CODES`][] comparison; a new sense raises here at import.
_MPS_SENSE = {sense: {'<=': 'L', '>=': 'G', '==': 'E'}[sense] for sense in SENSE_CODES}

#: What an integer column is wrapped in; the name field is a constant nothing reads.
_MARKER = "    MARKER 'MARKER' '{}'"

#: Nonzeros per column chunk; it bounds the writer's peak memory, not its speed.
EMIT_BUDGET = 500_000


def write_mps_file(handoff: Handoff, path: str | Path, names: Names = NUMBERED) -> None:
    """Write the model as MPS text, its columns and rows called what *names* calls them."""
    path = Path(path)
    entries, starts = _column_major(handoff)

    with open(path, 'wb') as f:
        f.write(b'NAME\n')
        if handoff.objective_sense == 'maximize':
            f.write(b'OBJSENSE\n    MAX\n')

        f.write(b'ROWS\n N  obj\n')
        sink(_row_lines(handoff, names), f)

        f.write(b'COLUMNS\n')
        width = handoff.matrix.height / max(1, handoff.column_count)
        for lo, hi in ranges(handoff.column_count, EMIT_BUDGET, width):
            owned = entries.slice(int(starts[lo]), int(starts[hi] - starts[lo]))
            sink(_column_lines(handoff, lo, hi, owned, names), f)

        f.write(b'RHS\n')
        if handoff.objective_constant:
            f.write(f'    rhs obj {-handoff.objective_constant!r}\n'.encode())
        sink(_rhs_lines(handoff, names), f)

        f.write(b'BOUNDS\n')
        _write_bounds(handoff, f, names)

        if handoff.sos.height:
            f.write(b'SOS\n')
            sink(_set_lines(handoff, names), f)

        f.write(b'ENDATA\n')


def _column_major(handoff: Handoff) -> tuple[pl.DataFrame, np.ndarray[tuple[int, ...], np.dtype[np.int64]]]:
    """The matrix in ``(col, row)`` order, and where each column's entries begin.

    The offsets let a column range slice the matrix rather than filter it once per chunk.
    """
    entries = handoff.matrix_block(0, handoff.row_count).sort('col', 'row')
    counts = np.bincount(entries['col'].to_numpy(), minlength=handoff.column_count)
    return entries, np.concatenate(([0], np.cumsum(counts)))


def _row_lines(handoff: Handoff, names: Names) -> pl.LazyFrame:
    """One ``ROWS`` entry per constraint row, after the objective's ``N``."""
    return handoff.rows.lazy().select(
        pl.concat_str(
            pl.lit(' '),
            pl.col('sense').replace_strict(_MPS_SENSE, return_dtype=pl.String),
            pl.lit('  '),
            names.row(pl.col('row')),
        )
    )


def _rhs_lines(handoff: Handoff, names: Names) -> pl.LazyFrame:
    """Each row's right-hand side, in row order."""
    return handoff.rows.lazy().select(
        pl.concat_str(pl.lit('    rhs '), names.row(pl.col('row')), pl.lit(' '), number(pl.col('rhs')))
    )


def _column_lines(handoff: Handoff, lo: int, hi: int, entries: pl.DataFrame, names: Names) -> pl.LazyFrame:
    """Every ``COLUMNS`` line for columns ``[lo, hi)``, one sorted stream.

    A column's lines occupy ``slots`` consecutive keys — the integer marker, its
    objective coefficient, each matrix entry at its row index, the closing
    marker — so one sort settles both the column order and the order within a
    column. Every column gets an objective line, coefficient or not: a column
    MPS never names is a column the reader does not have.
    """
    slots = handoff.row_count + 3

    def _key(within: pl.Expr) -> pl.Expr:
        return chunk_key(pl.col('col'), lo, slots, within)

    columns = (
        handoff.cols.lazy()
        .slice(lo, hi - lo)
        .with_row_index('col', offset=lo)
        .with_columns(pl.col('col').cast(pl.Int64))
    )
    integral = columns.filter(pl.col('vtype') != 'continuous')
    name = pl.concat_str(pl.lit('    '), names.column(pl.col('col')))
    cost = columns.join(handoff.obj.lazy().with_columns(pl.col('col').cast(pl.Int64)), on='col', how='left').select(
        _key(pl.lit(1, dtype=pl.Int64)),
        pl.concat_str(name, pl.lit(' obj '), number(pl.col('coeff').fill_null(0.0))).alias('line'),
    )
    terms = (
        entries.lazy()
        .with_columns(pl.col('col').cast(pl.Int64))
        .select(
            _key(pl.col('row').cast(pl.Int64) + 2),
            pl.concat_str(name, pl.lit(' '), names.row(pl.col('row')), pl.lit(' '), number(pl.col('coeff'))).alias(
                'line'
            ),
        )
    )
    markers = [
        integral.select(_key(pl.lit(at, dtype=pl.Int64)), pl.lit(_MARKER.format(word)).alias('line'))
        for at, word in ((0, 'INTORG'), (slots - 1, 'INTEND'))
    ]
    return pl.concat([*markers, cost, terms]).sort('key').select('line')


def _write_bounds(handoff: Handoff, f: IO[bytes], names: Names) -> None:
    """Every column's lower bound, then every column's upper.

    A reader given every lower bound does not apply the MPS rule that an ``UP``
    below zero implies an unbounded lower one.
    """
    for keyword, unbounded, column in (('LO', 'MI', 'lb'), ('UP', 'PL', 'ub')):
        name = pl.concat_str(pl.lit(' bnd '), names.column(pl.col('col')))
        sink(
            handoff.cols.lazy()
            .with_row_index('col')
            .select(
                pl.when(pl.col(column).is_infinite())
                .then(pl.concat_str(pl.lit(f' {unbounded}'), name))
                .otherwise(pl.concat_str(pl.lit(f' {keyword}'), name, pl.lit(' '), number(pl.col(column))))
            ),
            f,
        )


def _set_lines(handoff: Handoff, names: Names) -> pl.LazyFrame:
    """Each special-ordered set as its header line and one line per member.

    The stream arrives grouped by set and ascending in weight, so the key is the
    member's own index, doubled to leave the header a slot before its first
    member.
    """
    members = handoff.sos.lazy().with_row_index('ord').with_columns(pl.col('ord').cast(pl.Int64))
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
        pl.concat_str(pl.lit('    '), names.column(pl.col('col')), pl.lit(' '), digits(pl.col('weight'))).alias('line'),
    )
    return pl.concat([headers, lines]).sort('key').select('line')
