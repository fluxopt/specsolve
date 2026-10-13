"""gurobipy, both dialects — the model is hand-written, in `bench/models/<case>/`.

One runtime for two arms: `gurobipy-loop` and `gurobipy-matrix` differ only in
which formulation module they call.

The seam is `update()`, inside the clock, because gurobipy defers every
`addVar` and `addConstr` until the model is flushed; constructing `Gurobi` ends with
the same call. `OutputFlag` goes off at `Env` construction, so the licence
banner is never written. There is one sink.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Mapping

    from bench.arms import Counts

#: Where this arm can hand a model over.
SINKS = ('gurobi',)

#: What has to be importable for this arm to run; an absent library skips the cell.
REQUIRES = ('gurobipy',)


class Prepared(NamedTuple):
    """Where the parquet is; every verb reads it inside its call, the matrix floor's `load` excepted."""

    dialect: str
    case_name: str
    paths: dict[str, str]


def prepare(dialect: str, case_name: str, size: str, paths: dict[str, str], options: Mapping[str, Any]) -> Prepared:
    del size, options
    return Prepared(dialect, case_name, dict(paths))


class Arrays(NamedTuple):
    """The `gurobipy-matrix` model as `loadModel` takes it: the matrix by column, one sense byte per row."""

    obj: Any
    lb: Any
    ub: Any
    csc: tuple[Any, Any, Any]
    sense: Any
    rhs: Any


def matrix_arrays(case_name: str, tables: Mapping[str, Any]) -> Arrays:
    """The `gurobipy-matrix` formulation over *tables*, finished for `loadModel`.

    `loadModel` reads the matrix by column, so the CSR is transposed here.
    """
    import numpy as np

    from bench.models import formulation

    lp = formulation(case_name, 'gurobipy-matrix').build(tables)
    matrix = lp.matrix.tocsc()
    return Arrays(
        obj=lp.obj,
        lb=lp.lower,
        ub=lp.upper,
        csc=(matrix.data, matrix.indices, matrix.indptr),
        sense=np.asarray(lp.senses, dtype='S1'),
        rhs=lp.rhs,
    )


def matrix_loaded(env: Any, arrays: Arrays) -> Any:
    """A model holding *arrays*, in one `loadModel` call.

    The fastest bulk API gurobipy has, and the one specsolve's own `Gurobi`
    loads through.
    """
    import gurobipy as gp

    return gp.loadModel(
        env=env,
        numvars=len(arrays.obj),
        numconstrs=len(arrays.rhs),
        obj=arrays.obj,
        lb=arrays.lb,
        ub=arrays.ub,
        constr_csc=arrays.csc,
        sense=arrays.sense,
        rhs=arrays.rhs,
    )


def _model(env: Any, dialect: str, case_name: str, tables: Mapping[str, Any]) -> Any:
    """The model, however this dialect spells one: a `Model` from `gurobipy-loop`, `loadModel` for `gurobipy-matrix`."""
    from bench.models import formulation

    if dialect == 'gurobipy-loop':
        return formulation(case_name, dialect).build(env, tables)
    return matrix_loaded(env, matrix_arrays(case_name, tables))


def tables(prepared: Prepared) -> dict[str, Any]:
    """The parquet, read into polars."""
    import polars as pl

    return {name: pl.read_parquet(path) for name, path in prepared.paths.items()}


def _built(prepared: Prepared) -> tuple[Any, Any]:
    """The environment and a flushed model, the parquet read inside the call."""
    import gurobipy as gp

    read = tables(prepared)
    env = gp.Env(params={'OutputFlag': 0})
    model = _model(env, prepared.dialect, prepared.case_name, read)
    model.update()
    return env, model


def _counts(model: Any) -> Counts:
    return {'columns': model.NumVars, 'rows': model.NumConstrs, 'nonzeros': model.NumNZs}


def _release(env: Any, model: Any) -> None:
    """Both, or the next round's peak is this round's high-water mark as well."""
    model.dispose()
    env.dispose()


def matrix_load(arrays: Arrays) -> Counts:
    """*arrays* loaded into a new environment and flushed, then released — the `gurobipy-matrix` floor's timed body."""
    import gurobipy as gp

    env = gp.Env(params={'OutputFlag': 0})
    model = matrix_loaded(env, arrays)
    model.update()
    try:
        return _counts(model)
    finally:
        _release(env, model)


def build_and_emit(sink: str, prepared: Prepared) -> Counts:
    """Build the model and flush it — for this arm those are one act."""
    del sink
    env, model = _built(prepared)
    try:
        return _counts(model)
    finally:
        _release(env, model)


def build_only(prepared: Prepared) -> Counts:
    """The same work: this arm has no sink to leave out."""
    env, model = _built(prepared)
    try:
        return _counts(model)
    finally:
        _release(env, model)


def written(prepared: Prepared, path: Path) -> None:
    """Build and write the model to *path* as an LP file; writing, unlike solving, is not size-limited."""
    env, model = _built(prepared)
    try:
        model.write(str(path))
    finally:
        _release(env, model)


def objective(prepared: Prepared) -> float:
    """Write the model out and solve it with HiGHS — never with Gurobi.

    The `gurobipy` wheel's size-limited licence refuses `optimize()` above 2000
    columns; writing is not limited.
    """
    import highspy

    with tempfile.TemporaryDirectory(prefix='specsolve-bench-') as tmp:
        path = Path(tmp) / 'model.lp'
        written(prepared, path)
        highs = highspy.Highs()
        highs.setOptionValue('output_flag', False)
        highs.readModel(str(path))
        highs.run()
        status = highs.getModelStatus()
        if highs.modelStatusToString(status) != 'Optimal':
            raise RuntimeError(f'HiGHS finished {highs.modelStatusToString(status)!r} on the gurobipy model')
        return float(highs.getInfo().objective_function_value)
