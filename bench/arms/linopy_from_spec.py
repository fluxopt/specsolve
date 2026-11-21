"""linopy master's `Model.from_spec`: the case's own spec.yaml, built by linopy.

A probe arm, not for merging: `from_spec` is on linopy master only. The same
parquet as every arm, read with pandas, each parameter's `value` column renamed
to the parameter, which is what `from_spec` reads. Sinks and defaults are the
hand-written arm's (`bench.arms.linopy`).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Mapping

    from bench.arms import Counts

SINKS = ('lp', 'mps', 'highs', 'gurobi')
REQUIRES = ('linopy',)
WINDOW_CHANGES = ('values', 'shape')


class Prepared(NamedTuple):
    spec: str
    paths: dict[str, str]


def prepare(case_name: str, size: str, paths: dict[str, str], options: Mapping[str, Any]) -> Prepared:
    from bench.cases import CASES

    del options
    case = CASES[case_name]
    return Prepared(str(case.spec_path(case.shape(size))), dict(paths))


def _sources(paths: Mapping[str, str]) -> dict[str, Any]:
    """Each file as `from_spec` reads it: a dimension as its labels, a parameter under its own name."""
    import pandas as pd

    out: dict[str, Any] = {}
    for name, path in paths.items():
        df = pd.read_parquet(path)
        if list(df.columns) == [name]:
            out[name] = df[name]
        elif 'value' in df.columns:
            out[name] = df.rename(columns={'value': name})
        else:
            out[name] = df
    return out


def _built(prepared: Prepared) -> Any:
    import warnings

    import linopy

    warnings.filterwarnings('ignore', category=linopy.EvolvingAPIWarning)  # the API is new, and this is its use

    if not hasattr(linopy.Model, 'from_spec'):
        raise RuntimeError(f'linopy {linopy.__version__} has no Model.from_spec')
    return linopy.Model.from_spec(Path(prepared.spec), _sources(prepared.paths), expressions=False, force_dim_names=True)


def _counts(m: Any) -> Counts:
    return {'columns': int(m.nvars), 'rows': int(m.ncons), 'nonzeros': None}


def build_and_emit(sink: str, prepared: Prepared) -> Counts:
    with tempfile.TemporaryDirectory(prefix='specsolve-bench-') as tmp:
        m = _built(prepared)
        if sink == 'lp':
            m.to_file(Path(tmp) / 'model.lp', io_api='lp-polars', progress=False)
        elif sink == 'mps':
            m.to_file(Path(tmp) / 'model.mps', progress=False)
        elif sink == 'gurobi':
            _handle = m.to_gurobipy(set_names=False)
        else:
            _handle = m.to_highspy(set_names=False)
        return _counts(m)


def window_setup(sink: str, prepared: Prepared, following: Prepared, change: str) -> tuple[tuple[Any, ...], dict[str, Any]]:
    del prepared, change
    return (sink, following), {}


def window(sink: str, prepared: Prepared) -> Counts:
    return build_and_emit(sink, prepared)


def build_only(prepared: Prepared) -> Counts:
    return _counts(_built(prepared))


def written(prepared: Prepared, path: Path) -> None:
    _built(prepared).to_file(path, io_api='lp-polars', progress=False)


def objective(prepared: Prepared) -> float:
    m = _built(prepared)
    m.solve(solver_name='highs', output_flag=False)
    if m.status != 'ok':
        raise RuntimeError(f'linopy solve finished {m.status!r}, not ok')
    return float(m.objective.value)
