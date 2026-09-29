"""The floor under the `highs` sink: a matrix, hand-written, into `highspy`.

Raw solver API, no modelling layer: the `highs` counterpart of
`gurobipy-matrix`, built from the same `bench/models/<case>/matrix.py`. It hands
the model over in one `addCols` and one `addRows`, unchunked.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Mapping

    from bench.arms import Counts

#: Where this arm can hand a model over.
SINKS = ('highs',)

#: What has to be importable — the CSR the formulations build; `highspy` is a hard dependency.
REQUIRES = ('scipy',)

#: Which formulation module in `bench/models/<case>/` this arm builds from.
DIALECT = 'highspy-matrix'


class Prepared(NamedTuple):
    """What the timed verbs need; the parquet is read inside the clock."""

    case_name: str
    paths: dict[str, str]


def prepare(case_name: str, size: str, paths: dict[str, str], options: Mapping[str, Any]) -> Prepared:
    del size, options
    return Prepared(case_name, dict(paths))


def _row_bounds(lp: Any) -> tuple[Any, Any]:
    """A sense per row as the pair of bounds HiGHS takes; a sense other than `'='` or `'<'` raises."""
    import highspy
    import numpy as np

    unknown = set(np.unique(lp.senses)) - {'=', '<'}
    if unknown:
        raise ValueError(f'{DIALECT} has no row bounds for sense(s) {sorted(unknown)} — only "=" and "<"')
    return np.where(lp.senses == '=', lp.rhs, -highspy.kHighsInf), lp.rhs


def _built(prepared: Prepared) -> Any:
    """A populated `highspy.Highs`, `run()` never called — the whole timed body."""
    import highspy
    import numpy as np
    import polars as pl

    from bench.models import formulation

    tables = {name: pl.read_parquet(path) for name, path in prepared.paths.items()}
    lp = formulation(prepared.case_name, DIALECT).build(tables)
    matrix = lp.matrix.tocsr()
    lower, upper = _row_bounds(lp)

    empty_index = np.empty(0, dtype=np.int32)
    empty_value = np.empty(0, dtype=np.float64)
    highs = highspy.Highs()
    highs.setOptionValue('output_flag', False)
    highs.addCols(len(lp.obj), lp.obj, lp.lower, lp.upper, 0, empty_index, empty_index, empty_value)
    highs.addRows(
        len(lp.rhs),
        lower,
        upper,
        matrix.nnz,
        matrix.indptr[:-1].astype(np.int32),
        matrix.indices.astype(np.int32),
        matrix.data,
    )
    return highs


def _counts(highs: Any) -> Counts:
    return {'columns': highs.getNumCol(), 'rows': highs.getNumRow(), 'nonzeros': highs.getNumNz()}


def build_and_emit(sink: str, prepared: Prepared) -> Counts:
    """Build the matrix and load it — for this arm those are one act."""
    del sink
    highs = _built(prepared)
    try:
        return _counts(highs)
    finally:
        highs.clear()


def build_only(prepared: Prepared) -> Counts:
    """The same work: this arm has no sink to leave out."""
    return build_and_emit('highs', prepared)


def objective(prepared: Prepared) -> float:
    """Solve, and return what the parity gate compares."""
    highs = _built(prepared)
    try:
        highs.run()
        status = highs.modelStatusToString(highs.getModelStatus())
        if status != 'Optimal':
            raise RuntimeError(f'HiGHS finished {status!r} on the {DIALECT} model, not optimal')
        return float(highs.getInfo().objective_function_value)
    finally:
        highs.clear()
