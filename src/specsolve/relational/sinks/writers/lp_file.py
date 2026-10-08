"""The ``lp_file`` sink: the model as LP text.

Every section is a lazy frame sunk straight into the open file, in label
order, so a model writes the same bytes twice.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, get_args

import polars as pl
from mathspec import program

from specsolve.relational.collect import collected
from specsolve.relational.sinks.capabilities import Capabilities
from specsolve.relational.sinks.handoff import SENSE_CODES
from specsolve.relational.sinks.writers.text import append_lines, chunk_key, digits, number

if TYPE_CHECKING:
    from specsolve.relational.sinks.handoff import Handoff


#: Every construct the language can reach, quadratic rows included. A reader
#: may still refuse a section: HiGHS's refuses two.
LP_FILE_CAPABILITIES = Capabilities(
    supports=frozenset(
        {'integrality', 'sos', 'quadratic_objective', 'nonconvex_quadratic_objective', 'quadratic_constraint'}
    )
)


#: The LP spelling of each [`SENSE_CODES`][] comparison.
_LP_SENSE = {sense: '=' if sense == '==' else sense for sense in SENSE_CODES}

#: The section each non-continuous domain is listed under; a new domain raises here at import.
_LP_DOMAIN_SECTION = {
    domain: {'binary': 'binary', 'integer': 'general'}[domain]
    for domain in get_args(program.VariableDomain)
    if domain != 'continuous'
}

#: Nonzeros per constraint chunk; it bounds the writer's peak memory, not its speed.
EMIT_BUDGET = 2_000_000


def write_lp_file(handoff: Handoff, path: str | Path) -> None:
    """Write the model as LP text."""
    path = Path(path)
    objective = handoff.obj.lazy().sort('col').select(_term(pl.col('coeff'), pl.col('col')))
    bounds = (
        handoff.cols.lazy()
        .with_row_index('col')
        .select(
            pl.concat_str(
                _bound(pl.col('lb'), '-infinity').alias('lb'),
                pl.lit(' <= x').alias('open'),
                digits(pl.col('col')),
                pl.lit(' <= ').alias('close'),
                _bound(pl.col('ub'), '+infinity').alias('ub'),
            )
        )
    )

    with open(path, 'wb') as f:
        f.write((b'max' if handoff.objective_sense == 'maximize' else b'min') + b'\n\nobj:\n')
        if handoff.objective_constant:
            f.write(f'{handoff.objective_constant:+.17g}\n'.encode())
        append_lines(objective, f)
        if handoff.quad.height:
            f.write(b'+ [\n')
            append_lines(_quadratic_terms(handoff), f)
            f.write(b'] / 2\n')

        f.write(b'\ns.t.\n\n')
        for block in handoff.row_blocks(EMIT_BUDGET):
            append_lines(_constraint_lines(handoff, block.lo, block.hi, handoff.matrix_block(block.lo, block.hi)), f)
        for row, pairs in handoff.quadratic_blocks():
            append_lines(_quadratic_row_lines(handoff, row, pairs), f)

        f.write(b'\nbounds\n')
        append_lines(bounds, f)

        for domain, keyword in _LP_DOMAIN_SECTION.items():
            chosen = handoff.cols.lazy().with_row_index('col').filter(pl.col('vtype') == domain)
            if chosen.select(pl.len()).pipe(collected).item() == 0:
                continue
            f.write(f'\n{keyword}\n'.encode())
            append_lines(chosen.select(pl.concat_str(pl.lit('x'), digits(pl.col('col')))), f)

        if handoff.sos.height:
            f.write(b'\nsos\n')
            append_lines(_set_lines(handoff), f)

        f.write(b'\nend\n')


def _quadratic_row_lines(handoff: Handoff, row: int, pairs: pl.DataFrame) -> pl.LazyFrame:
    """One quadratic constraint, ``c7: +1 x0 + [ 2 x0 * x1 ] >= 4``.

    Not doubled: the format divides only the objective's bracket by two.
    """
    entries = handoff.matrix_block(row, row + 1)
    header = pl.LazyFrame({'line': [f'c{row}:']})
    linear = entries.lazy().sort('col').select(_term(pl.col('coeff'), pl.col('col')).alias('line'))
    opened = pl.LazyFrame({'line': ['+ [']})
    quadratic = pairs.lazy().select(_pair(pl.col('coeff')).alias('line'))
    closed = (
        handoff.rows.lazy().filter(pl.col('row') == row).select(pl.concat_str(pl.lit('] '), _footer()).alias('line'))
    )
    return pl.concat([header, linear, opened, quadratic, closed])


def _quadratic_terms(handoff: Handoff) -> pl.LazyFrame:
    """The objective's quadratic part, one ``+2 x3 * x7`` line per pair.

    The format divides the section by two, so every coefficient is doubled, on
    the diagonal and off it alike. A pair arrives ordered, summed and
    deduplicated, so nothing here sorts.
    """
    return handoff.quad.lazy().select(_pair(pl.col('coeff') * 2))


def _pair(coeff: pl.Expr) -> pl.Expr:
    """One quadratic pair as ``+2 x3 * x7``, or ``x3 ^ 2`` for a squared column: no parser accepts ``x3 * x3``."""
    return pl.concat_str(
        *_signed(coeff),
        pl.lit(' x'),
        digits(pl.col('col_l')),
        pl.when(pl.col('col_l') == pl.col('col_r'))
        .then(pl.lit(' ^ 2'))
        .otherwise(pl.concat_str(pl.lit(' * x'), digits(pl.col('col_r')))),
    )


def _set_lines(handoff: Handoff) -> pl.LazyFrame:
    """Each special-ordered set as one ``s0: S2 :: x3:1 x4:2`` line, in linopy's spelling.

    ``maintain_order`` keeps a set's line the same bytes twice.
    """
    return (
        handoff.sos.lazy()
        .group_by('set', maintain_order=True)
        .agg(
            pl.col('type').first(),
            pl.concat_str(pl.lit('x'), digits(pl.col('col')), pl.lit(':'), digits(pl.col('weight')))
            .str.join(' ')
            .alias('members'),
        )
        .select(
            pl.concat_str(
                pl.lit('s'),
                digits(pl.col('set')),
                pl.lit(': S'),
                digits(pl.col('type')),
                pl.lit(' :: '),
                pl.col('members'),
            )
        )
    )


def _constraint_lines(handoff: Handoff, lo: int, hi: int, entries: pl.DataFrame) -> pl.LazyFrame:
    """Every constraint line for rows ``[lo, hi)``, one sorted stream.

    A row's lines occupy ``slots`` consecutive keys — header, placeholder, each
    term at its column index, sense — so one sort settles both the row order and
    the order within a row. The anti-join gives a termless row the ``+0 x0`` a
    parser needs. The terms are sorted although they arrive sorted, so the union
    sort merges runs rather than permuting them.
    """
    slots = handoff.cols.height + 3

    def _key(within: pl.Expr) -> pl.Expr:
        return chunk_key(pl.col('row'), lo, slots, within)

    rows = handoff.rows.lazy().filter(pl.col('row').is_between(lo, hi, closed='left'))
    matrix = entries.lazy()
    header = rows.select(
        _key(pl.lit(0, dtype=pl.Int64)),
        pl.concat_str(pl.lit('c').alias('c'), digits(pl.col('row')), pl.lit(':').alias('colon')).alias('line'),
    )
    placeholder = rows.join(matrix.select('row'), on='row', how='anti').select(
        _key(pl.lit(1, dtype=pl.Int64)),
        pl.lit('+0 x0').alias('line'),
    )
    terms = matrix.sort('row', 'col').select(
        _key(pl.col('col').cast(pl.Int64) + 2),
        _term(pl.col('coeff'), pl.col('col')).alias('line'),
    )
    footer = rows.select(_key(pl.lit(slots - 1, dtype=pl.Int64)), _footer().alias('line'))
    return pl.concat([header, placeholder, terms, footer]).sort('key').select('line')


def _footer() -> pl.Expr:
    """A row's comparison and right-hand side, ``>= 4``, off its ``sense`` and ``rhs`` columns."""
    return pl.concat_str(
        pl.col('sense').replace_strict(_LP_SENSE, return_dtype=pl.String),
        pl.lit(' '),
        number(pl.col('rhs')),
    )


def _term(coeff: pl.Expr, col: pl.Expr) -> pl.Expr:
    """One ``+1.5 x7`` term."""
    return pl.concat_str(*_signed(coeff), pl.lit(' x'), digits(col))


def _signed(value: pl.Expr) -> tuple[pl.Expr, pl.Expr]:
    """A coefficient with its sign always explicit, as the LP format needs.

    Zero is spelled out because ``-0.0`` takes the ``+`` arm while the cast
    renders ``-0.0``, giving ``+-0.0``, which no LP parser accepts.
    """
    return (
        pl.when(value >= 0).then(pl.lit('+')).otherwise(pl.lit('')).alias('sign'),
        pl.when(value == 0).then(pl.lit('0.0')).otherwise(number(value)).alias('magnitude'),
    )


def _bound(value: pl.Expr, infinite: str) -> pl.Expr:
    """A bound, with the LP format's own spelling for an unbounded one."""
    return pl.when(value.is_infinite()).then(pl.lit(infinite)).otherwise(number(value))
