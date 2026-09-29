#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["linopy==0.9.0", "pandas>=2.2", "xarray==2026.7.0", "highspy==1.15.1"]
# ///
"""Reference for ``transport_dantzig``: the same LP, hand-written in linopy.

    pixi exec -s uv uv run --script examples/ports/references/linopy/transport_dantzig.py

**This is not what verifies the port.** The optimum came from the literature —
published with GAMS model library #1 — and that is what ``references.json``
records as primary. This script is a second, independent arrival at the same
number, from a different formulation in a different tool.

Run out of band with the pinned versions above: linopy is not a runtime
dependency of this project. Nothing here imports specsolve.
"""

from __future__ import annotations

import json
from pathlib import Path

import linopy
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / 'data' / 'transport_dantzig.json'


def load_tables() -> dict[str, pd.DataFrame]:
    """The instance, one frame per parameter — what a caller of either library holds."""
    return {k: pd.DataFrame(v) if isinstance(v, dict) else v for k, v in json.loads(DATA.read_text()).items()}


def build(tables: dict[str, pd.DataFrame]) -> linopy.Model:
    """The port's tables as a linopy model, term for term.

    ``tables`` is the same mapping the specsolve call attaches as ``sources``.
    """
    capacity: pd.Series = tables['capacity'].set_index('plant')['value']
    demand: pd.Series = tables['demand'].set_index('market')['value']
    distance: pd.DataFrame = (
        tables['distance']
        .pivot(index='plant', columns='market', values='value')
        .reindex(index=capacity.index)[demand.index]
    )
    cost: pd.DataFrame = distance * tables['freight'] / 1000

    m = linopy.Model()
    shipment = m.add_variables(lower=0, coords=[capacity.index, demand.index], name='shipment')
    m.add_constraints(shipment.sum('market') <= capacity, name='within_capacity')
    m.add_constraints(shipment.sum('plant') >= demand, name='meet_demand')
    m.add_objective((shipment * cost).sum())
    return m


def shadow_prices(m: linopy.Model, name: str, dim: str) -> dict[str, list]:
    """The dual of constraint *name*, tidy.

    Both of this model's constraints are inequalities: a capacity's shadow
    price and a demand's carry opposite signs.
    """
    dual = m.constraints[name].dual
    return {dim: [str(v) for v in dual.indexes[dim]], 'value': [float(v) for v in dual.values]}


def main() -> float:
    """Solve, and print what ``references.json`` records."""
    m = build(load_tables())
    status, condition = m.solve(solver_name='highs')
    assert status == 'ok', f'{status}: {condition}'
    print(f'linopy {linopy.__version__}')
    print(f'objective {float(m.objective.value)!r}')
    print(
        f'duals {json.dumps({"within_capacity": shadow_prices(m, "within_capacity", "plant"), "meet_demand": shadow_prices(m, "meet_demand", "market")})}'
    )
    return float(m.objective.value)


if __name__ == '__main__':
    main()
