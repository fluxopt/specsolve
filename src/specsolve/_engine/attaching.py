"""What a model makes of its data before an entry is built: missing rows filled, conditions settled, assumptions held.

[`encoded`][specsolve._encoding.encoded] has put every table in its
dimensions' types and held it to what the spec declares; [`attach`][] is what
needs the program's expressions and conditions. Its result is the one way data
reaches a build or a read.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING

import polars as pl
from mathspec import program as program_module

from specsolve._attached import label_product
from specsolve._clock import untimed
from specsolve._collect import collected
from specsolve._engine.assumptions import check_assumptions
from specsolve._engine.coverage import divisors_of
from specsolve._engine.scope import reads_as_zero
from specsolve._engine.specialize import settle
from specsolve._messages import negative_window_message
from specsolve._names import VALUE
from specsolve.errors import DataError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mathspec import program

    from specsolve._attached import AttachedSources, Dimension
    from specsolve._clock import Lap


def attach(program: program.Program, data: AttachedSources, lap: Lap = untimed) -> AttachedSources:
    """*data* as *program* reads it: missing rows filled, the conditions the data decides settled, the assumptions held.

    A parameter that declares a value for a missing row has it written in
    where a divisor reads it; one [`encoded`][specsolve._encoding.encoded]
    kept from an earlier attach is whole already, and keeps the coordinates
    the data gave it. Every atom of a ``where`` is then read once
    ([`settle`][]), for the assumptions and the build both, and the
    assumptions are checked.

    Args:
        program: The compiled spec.
        data: Its sources, as [`encoded`][specsolve._encoding.encoded] returns them.
        lap: Ends a lap of ``attach`` after each step: ``fill``, ``settle``
            and ``assume``.

    Raises:
        DataError: An ``assumptions:`` entry the data does not hold.
    """
    parameters = dict(data.parameters)
    rows = dict(data.parameter_rows)
    given = dict(data.given)
    filled = filled_parameters(program)
    for name, p in program.parameters.items():
        if isinstance(p.missing, str) or p.missing is None or (reads_as_zero(p.missing) and name not in filled):
            continue
        if rows[name] < math.prod(data.dimensions[d].size for d in p.dims):
            table = parameters[name].pipe(collected)
            given[name] = table.select(*p.dims).lazy()
            whole = _filled(table, p.dims, p.missing, data.dimensions)
            parameters[name], rows[name] = whole.lazy(), whole.height
    lap('attach', 'fill', None)
    attached = replace(data, parameters=parameters, parameter_rows=rows, given=given)
    attached = replace(attached, settled=settle(program, attached))
    lap('attach', 'settle', None)
    check_assumptions(program, attached)
    lap('attach', 'assume', None)
    return attached


def filled_parameters(declared: program.Program) -> frozenset[str]:
    """The parameters read under a divisor anywhere in *declared*.

    A ``missing: 0`` row of one is filled rather than left sparse: a sparse
    zero is no contribution to a term, but under a divisor it is the number
    divided by, as in ``x / (1 + d)``. A bound fills its own read instead.
    """
    named = (e.expression for e in declared.expressions.values())
    return program_module.parameters_of(*divisors_of(*declared.roots, *named))


def refuse_a_negative_window(data: AttachedSources, name: str) -> None:
    """Refuse parameter *name*, the window of a ``sum_back``, where a row holds a negative width.

    Raises:
        DataError: Naming *name* and each negative width it holds.
    """
    negative = data.parameters[name].filter(pl.col(VALUE) < 0).select(pl.col(VALUE).unique().sort()).pipe(collected)
    if negative.height:
        raise DataError(negative_window_message(name, negative[VALUE].to_list()))


def _filled(
    table: pl.DataFrame, dims: tuple[str, ...], value: bool | float, dimensions: Mapping[str, Dimension]
) -> pl.DataFrame:
    """*table* with a row at every coordinate of *dims*, *value* where the data has none.

    The rows come back in label order, so a sum over them adds its terms in
    the same order at every build.
    """
    if not dims:
        return table if table.height else pl.DataFrame({VALUE: [value]}, schema={VALUE: table.schema[VALUE]})
    product = label_product({d: dimensions[d] for d in dims})
    filled = product.join(table.lazy(), on=list(dims), how='left', maintain_order='left').with_columns(
        pl.col(VALUE).fill_null(value)
    )
    return filled.select(*dims, pl.col(VALUE).cast(table.schema[VALUE])).pipe(collected)
