#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["linopy==0.9.0", "pandas>=2.2", "xarray==2026.7.0", "highspy==1.15.1"]
# ///
"""Reference for ``dispatch``: the same LP, hand-written in linopy.

    pixi exec -s uv uv run --script examples/ports/references/linopy/dispatch.py

Unlike the ports, nothing here was published: the model is this project's own
smallest teaching example, so what verifies it is *agreement* — an independent
hand-written formulation on a different modelling stack reaching the same
objective and the same prices. ``references.json`` records what this script
printed, and ``tests/test_ports.py`` holds specsolve to it.

One difference: the YAML's ``where: p_max > 0`` gives the retired generator no
columns at all, where this script keeps them bounded to zero. Same polytope,
same objective, same duals.
"""

from __future__ import annotations

import json
from pathlib import Path

import linopy
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / 'data' / 'dispatch.json'


def load_tables() -> dict[str, pd.DataFrame]:
    """The instance, one frame per parameter — what a caller of either library holds."""
    return {k: pd.DataFrame(v) if isinstance(v, dict) else v for k, v in json.loads(DATA.read_text()).items()}


def build(tables: dict[str, pd.DataFrame]) -> linopy.Model:
    """The instance's tables as a linopy model, row for row.

    ``tables`` is the same mapping the specsolve call attaches as ``sources``.
    """
    p_max: pd.Series = tables['p_max'].set_index('generator')['value']
    cost: pd.Series = tables['cost'].set_index('generator')['value']
    load: pd.Series = tables['load'].set_index('snapshot')['value']

    m = linopy.Model()
    p = m.add_variables(lower=0, upper=p_max, coords=[load.index, p_max.index], name='p')
    m.add_constraints(p.sum('generator') == load, name='power_balance')
    m.add_objective((p * cost).sum())
    return m


def marginal_prices(m: linopy.Model) -> dict[str, list]:
    """The power-balance dual: the classic price signal.

    One price per snapshot — the cost of the marginal generator: a snapshot
    where wind covers the load prices at wind, the moment gas has to run the
    price jumps to gas.
    """
    dual = m.constraints['power_balance'].dual
    return {'snapshot': [int(v) for v in dual.indexes['snapshot']], 'value': [float(v) for v in dual.values]}


def main() -> float:
    m = build(load_tables())
    status, condition = m.solve(solver_name='highs')
    assert status == 'ok', f'{status}: {condition}'
    print(f'linopy {linopy.__version__}')
    print(f'objective {float(m.objective.value)!r}')
    print(f'duals {json.dumps({"power_balance": marginal_prices(m)})}')
    return float(m.objective.value)


if __name__ == '__main__':
    main()
