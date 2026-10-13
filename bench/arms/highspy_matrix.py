"""The floor under the `highs` sink: a finished matrix, handed to `highspy`.

Raw solver API, no modelling layer: the `highs` counterpart of
`gurobipy-matrix`, built from the same `bench/models/<case>/matrix.py`. The
published number is `load`: one `passModel` over arrays `load_setup` built
outside the clock, so it is what handing a model to HiGHS costs and nothing
before that. The other verbs read the parquet and build inside their own call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from bench.arms import Counts

#: Where this arm can hand a model over.
SINKS = ('highs',)

#: What has to be importable — the CSR the formulations build; `highspy` is a hard dependency.
REQUIRES = ('scipy',)

#: Which formulation module in `bench/models/<case>/` this arm builds from.
DIALECT = 'highspy-matrix'


class Prepared(NamedTuple):
    """Where the parquet is; every verb reads it, `load` excepted."""

    case_name: str
    paths: dict[str, str]


class Arrays(NamedTuple):
    """The model as `Highs.passModel` takes it: row-wise, int32 indices, one integrality flag per column."""

    cost: Any
    lower: Any
    upper: Any
    row_lower: Any
    row_upper: Any
    starts: Any
    index: Any
    value: Any
    integrality: Any


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


def _arrays(prepared: Prepared) -> Arrays:
    """The parquet, read and formulated into what `passModel` takes."""
    import numpy as np
    import polars as pl

    from bench.models import formulation

    tables = {name: pl.read_parquet(path) for name, path in prepared.paths.items()}
    lp = formulation(prepared.case_name, DIALECT).build(tables)
    matrix = lp.matrix.tocsr()
    row_lower, row_upper = _row_bounds(lp)
    return Arrays(
        cost=lp.obj,
        lower=lp.lower,
        upper=lp.upper,
        row_lower=row_lower,
        row_upper=row_upper,
        starts=matrix.indptr[:-1].astype(np.int32),
        index=matrix.indices.astype(np.int32),
        value=matrix.data,
        integrality=np.zeros(len(lp.obj), dtype=np.int32),
    )


def _loaded(arrays: Arrays) -> Any:
    """A populated `highspy.Highs`, `run()` never called, in one `passModel`.

    `passModel` rather than `addCols` and `addRows`, because it is the faster
    of the two and the one specsolve's own `Highs` loads through.
    """
    import highspy
    import numpy as np

    empty_index = np.empty(0, dtype=np.int32)
    highs = highspy.Highs()
    highs.setOptionValue('output_flag', False)
    status = highs.passModel(
        len(arrays.cost),
        len(arrays.row_lower),
        len(arrays.value),
        0,
        int(highspy.MatrixFormat.kRowwise),
        int(highspy.HessianFormat.kTriangular),
        int(highspy.ObjSense.kMinimize),
        0.0,
        arrays.cost,
        arrays.lower,
        arrays.upper,
        arrays.row_lower,
        arrays.row_upper,
        arrays.starts,
        arrays.index,
        arrays.value,
        empty_index,
        empty_index,
        np.empty(0, dtype=np.float64),
        arrays.integrality,
    )
    if status != highspy.HighsStatus.kOk:
        raise RuntimeError(f'HiGHS answered {status} to passModel on the {DIALECT} model')
    return highs


def _built(prepared: Prepared) -> Any:
    """A populated `highspy.Highs` from the parquet, building included."""
    return _loaded(_arrays(prepared))


def _counts(highs: Any) -> Counts:
    return {'columns': highs.getNumCol(), 'rows': highs.getNumRow(), 'nonzeros': highs.getNumNz()}


def load_setup(sink: str, prepared: Prepared) -> tuple[tuple[Arrays], dict[str, Any]]:
    """The finished arrays `load` hands over, built before the clock starts."""
    del sink
    return (_arrays(prepared),), {}


def load(arrays: Arrays) -> Counts:
    """Hand the finished arrays to HiGHS — the floor's whole timed body."""
    highs = _loaded(arrays)
    try:
        return _counts(highs)
    finally:
        highs.clear()


def build_and_emit(sink: str, prepared: Prepared) -> Counts:
    """Read, build and load: `load` with its arrays built inside the call."""
    del sink
    return load(_arrays(prepared))


def build_only(prepared: Prepared) -> Counts:
    """The same work: this arm has no sink to leave out."""
    return build_and_emit('highs', prepared)


def written(prepared: Prepared, path: Path) -> None:
    """Build the matrix and write the loaded model to *path* as an LP file."""
    highs = _built(prepared)
    try:
        highs.writeModel(str(path))
    finally:
        highs.clear()


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
