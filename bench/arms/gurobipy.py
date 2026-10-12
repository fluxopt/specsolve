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
    """What the timed verbs need; the parquet is read inside the clock."""

    dialect: str
    case_name: str
    paths: dict[str, str]


def prepare(dialect: str, case_name: str, size: str, paths: dict[str, str], options: Mapping[str, Any]) -> Prepared:
    del size, options
    return Prepared(dialect, case_name, dict(paths))


def _model(env: Any, dialect: str, case_name: str, tables: Mapping[str, Any]) -> Any:
    """The model, however this dialect spells one.

    `gurobipy-loop` returns a `Model`. `gurobipy-matrix` returns the
    solver-neutral `Lp`, loaded here in one `loadModel` call, the fastest
    bulk API gurobipy has and the one specsolve's own `Gurobi` loads through.
    `loadModel` reads the matrix by column, so the CSR is transposed first.
    """
    import gurobipy as gp
    import numpy as np

    from bench.models import formulation

    build = formulation(case_name, dialect).build
    if dialect == 'gurobipy-loop':
        return build(env, tables)
    lp = build(tables)
    matrix = lp.matrix.tocsc()
    return gp.loadModel(
        env=env,
        numvars=len(lp.obj),
        numconstrs=len(lp.rhs),
        obj=lp.obj,
        lb=lp.lower,
        ub=lp.upper,
        constr_csc=(matrix.data, matrix.indices, matrix.indptr),
        sense=np.asarray(lp.senses, dtype='S1'),
        rhs=lp.rhs,
    )


def _built(prepared: Prepared) -> tuple[Any, Any]:
    """The environment and a flushed model — every timed verb's whole body."""
    import gurobipy as gp
    import polars as pl

    tables = {name: pl.read_parquet(path) for name, path in prepared.paths.items()}
    env = gp.Env(params={'OutputFlag': 0})
    model = _model(env, prepared.dialect, prepared.case_name, tables)
    model.update()
    return env, model


def _counts(model: Any) -> Counts:
    return {'columns': model.NumVars, 'rows': model.NumConstrs, 'nonzeros': model.NumNZs}


def _release(env: Any, model: Any) -> None:
    """Both, or the next round's peak is this round's high-water mark as well."""
    model.dispose()
    env.dispose()


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
