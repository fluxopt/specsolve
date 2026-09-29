"""A solve's answer as rows of a datarecord record, in a schema this package defines.

Needs the ``datarecord`` extra: ``pip install 'specsolve[datarecord]'``.

[`answer_schema`][] is what a solve of a spec answers with, as a record's
schema: every frame [`Result.save`][specsolve.relational.result.Result.save]
writes, keyed ``<kind>.<name>`` over the run axis and the declaration's own
dims, and every column of the [`Record`][specsolve.relational.parquet.Record]
and [`Metrics`][specsolve.relational.parquet.Metrics] as an attribute over the
run axis. [`write_answer`][] stages one solve's answer as one run of it. A
record holding many runs is a family of solves: a Monte Carlo campaign, a
sweep, the steps of an iterative scheme. The record that held the inputs is
another record, which ``input_node`` names.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals as nw
import polars as pl
from datarecord import AttributeSpec, Dimension, Schema
from mathspec import to_spec

from specsolve.relational.parquet import LAYOUT, METRICS_SCHEMA, RECORD_SCHEMA

if TYPE_CHECKING:
    from datarecord import WorkingRecord

    from specsolve.lanes import Buildable
    from specsolve.relational.parquet import Metrics
    from specsolve.relational.result import Result

__all__ = ['answer_schema', 'write_answer']

#: The dim whose labels are the answer's names, which a reason is keyed by.
QUANTITY = 'quantity'


def answer_schema(spec: Buildable, *, over: str = 'run') -> Schema:
    """What a solve of *spec* answers with, as the schema of a record of answers.

    One attribute per frame a solve answers with, named ``<kind>.<name>``:
    ``primal.<variable>``, ``dual.<constraint>``, ``activity.<constraint>`` and
    ``expression.<name>``, each over *over* and the dims its declaration
    names. One attribute per column of the solve's record and metrics, over
    *over* alone. ``reason`` says why a quantity is not there, over *over* and
    ``quantity``; ``input_node`` names the input record's node a run solved.

    A layer patches one run at a time: *over* is the one ``partial`` dim.

    Args:
        spec: As [`check`][specsolve.check] takes it.
        over: The dim a record of many solves is a family along.

    Raises:
        ValueError: If *over* or ``quantity`` is a dim the spec declares.
    """
    spec = to_spec(spec)
    taken = sorted({over, QUANTITY} & set(spec.dimensions))
    if taken:
        msg = f'{taken} is a dim of the spec, so it cannot also be the answer axis; pass another over='
        raise ValueError(msg)
    dims = Schema.from_mathspec(spec).dimensions | {
        over: Dimension(dtype=nw.String()),
        QUANTITY: Dimension(dtype=nw.String()),
    }
    blocks = [
        *(('primal', name, block.dims) for name, block in spec.variables.items()),
        *(('dual', name, block.dims) for name, block in spec.constraints.items()),
        *(('activity', name, block.dims) for name, block in spec.constraints.items()),
        *(('expression', name, block.dims) for name, block in spec.expressions.items()),
    ]
    attributes = {
        f'{kind}.{name}': AttributeSpec(dtype=nw.Float64(), dims=frozenset({over, *(block_dims or ())}))
        for kind, name, block_dims in blocks
    }
    attributes |= {
        column: AttributeSpec(dtype=dtype, dims=frozenset({over}))
        for column, dtype in _facts().items()
        if column != 'run'
    }
    attributes['reason'] = AttributeSpec(dtype=nw.String(), dims=frozenset({over, QUANTITY}))
    attributes['input_node'] = AttributeSpec(dtype=nw.String(), dims=frozenset({over}))
    return Schema(
        dimensions=dims,
        attributes=attributes,
        partial=frozenset({over}),
        meta={'producer': 'specsolve', 'layout': LAYOUT},
    )


def _facts() -> dict[str, nw.dtypes.DType]:
    """The record's and the metrics' columns, as narwhals types them.

    Read off the schemas the archive writes, so a column added to either
    reaches a record of answers without a second list to keep.
    """
    empty = pl.DataFrame(schema=RECORD_SCHEMA | METRICS_SCHEMA)
    # pyrefly: ignore[no-matching-overload]  narwhals' overloads name no polars frame pyrefly resolves; it runs
    return dict(nw.from_native(empty, eager_only=True).schema)


def write_answer(
    working: WorkingRecord,
    result: Result,
    *,
    run: str,
    metrics: Metrics | None = None,
    input_node: str | None = None,
    over: str = 'run',
) -> None:
    """Stage *result* as run *run* of a record of answers.

    *working* is a record whose schema came from [`answer_schema`][]. What
    is staged is what [`Result.save`][specsolve.relational.result.Result.save]
    writes, frame for frame: a solve that left no values stages its record
    and metrics alone. Commit *working* to keep it.

    Args:
        working: The record of answers to stage into.
        result: One solve's answer.
        run: The label of this solve along *over*.
        metrics: What the model took, as ``model.diagnostics().metrics()``
            reads it; its columns are null where it is not given.
        input_node: The revision id of the input record's node this solve
            answered, if the inputs live in a record.
        over: The dim the record is a family along, as [`answer_schema`][]
            was given it.

    Raises:
        SpecsolveError: *result* was closed.
    """
    measured: dict[str, object] = metrics._asdict() if metrics is not None else {}
    facts = result.record._asdict() | measured
    del facts['run']
    facts |= {over: run, 'input_node': input_node}
    working.add(over, pl.DataFrame([facts], schema_overrides=_polars_facts(over)))
    if not result.has_primal:
        return
    answered, no_expressions = result._answered(result._unclosed('the solution'))
    for kind, name, frame in answered:
        working.set(f'{kind}.{name}', frame.lazy().with_columns(pl.lit(run).alias(over)).collect())
    reasons = {f'expression.{name}': why for name, why in no_expressions.items()}
    if result._no_duals is not None:
        reasons['dual'] = result._no_duals
    if reasons:
        working.set(
            'reason',
            pl.DataFrame({over: run, QUANTITY: list(reasons), 'value': list(reasons.values())}),
        )


def _polars_facts(over: str) -> dict[str, pl.DataType | type[pl.DataType]]:
    """The facts' column types for one run's row, which may hold only nulls."""
    return {
        **{c: t for c, t in (RECORD_SCHEMA | METRICS_SCHEMA).items() if c != 'run'},
        over: pl.String,
        'input_node': pl.String,
    }
