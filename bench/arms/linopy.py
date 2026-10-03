"""linopy, hand-written.

The model is typed out per case in `bench/models/<case>/linopy.py`, the way
`examples/ports/references/linopy/` writes it, and read from parquet with
pandas.

Three defaults are switched off:

- `set_names=False` on both solver hand-offs, because specsolve's sinks name
  nothing; naming is 82% of linopy's HiGHS hand-off (0.11s against 0.02s at
  200k variables) and 35% of its Gurobi one.
- `progress=False` on the LP writer, whose tqdm bars cost ~7% of the write at
  10M variables and write to stderr.
- `io_api='lp-polars'`, its fastest writer rather than its default one.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Mapping

    from bench.arms import Counts

#: Every sink linopy can hand a model to — the same three as ours.
SINKS = ('lp', 'highs', 'gurobi')

#: What has to be importable for this arm to run; an absent library skips the cell.
REQUIRES = ('linopy',)

#: Which formulation module in `bench/models/<case>/` this arm builds from.
DIALECT = 'linopy'


class Prepared(NamedTuple):
    case_name: str
    paths: dict[str, str]


def prepare(case_name: str, size: str, paths: dict[str, str], options: Mapping[str, Any]) -> Prepared:
    del size, options
    return Prepared(case_name, dict(paths))


def _built(prepared: Prepared) -> Any:
    """The model, parquet read included — every timed verb starts here."""
    import pandas as pd

    from bench.models import formulation

    tables = {name: pd.read_parquet(path) for name, path in prepared.paths.items()}
    return formulation(prepared.case_name, DIALECT).build(tables)


def _counts(m: Any) -> Counts:
    return {'columns': int(m.nvars), 'rows': int(m.ncons), 'nonzeros': None}


def build_and_emit(sink: str, prepared: Prepared) -> Counts:
    """Build the model and hand it over — an LP file, or a populated solver."""
    with tempfile.TemporaryDirectory(prefix='specsolve-bench-') as tmp:
        m = _built(prepared)
        if sink == 'lp':
            m.to_file(Path(tmp) / 'model.lp', io_api='lp-polars', progress=False)
        elif sink == 'gurobi':
            _handle = m.to_gurobipy(set_names=False)
        else:
            _handle = m.to_highspy(set_names=False)
        return _counts(m)


#: The window changes this arm tells apart. Every window is a rebuild here, so
#: one parameter or a cold solver is the ``values`` window again.
WINDOW_CHANGES = ('values', 'shape')


def window_setup(
    sink: str, prepared: Prepared, following: Prepared, change: str
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Nothing to hold between windows, so the whole rebuild of *following* lands inside the measurement, whatever *change* is."""
    del prepared, change
    return (sink, following), {}


def window(sink: str, prepared: Prepared) -> Counts:
    """Every window is a fresh build, because linopy has no verb for a second one."""
    return build_and_emit(sink, prepared)


def build_only(prepared: Prepared) -> Counts:
    """Just the build — no sink, nothing to release."""
    return _counts(_built(prepared))


def objective(prepared: Prepared) -> float:
    """Solve, and return what the arms are checked against."""
    m = _built(prepared)
    m.solve(solver_name='highs', output_flag=False)
    if m.status != 'ok':
        raise RuntimeError(f'linopy solve finished {m.status!r}, not ok')
    return float(m.objective.value)
