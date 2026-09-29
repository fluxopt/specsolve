#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["linopy==0.9.0", "pandas>=2.2", "xarray==2026.7.0", "highspy==1.15.1"]
# ///
"""Reference for ``transport``: the same LP, hand-written in linopy.

    pixi exec -s uv uv run --script examples/ports/references/linopy/transport.py

A teaching model, so what verifies it is agreement with an independent
formulation, not a published figure — see ``dispatch.py`` next door.

The line worth comparing is the nodal balance. The YAML groups by relations it
declared over the dimensions — ``sum(p, by=gen_bus, over=generator, into=bus)``
— where this script builds the bus x generator and bus x line incidence
matrices and multiplies through them. Both say Kirchhoff's current law; one
says it as a relation, the other as linear algebra.

linopy's ``groupby`` (which ``monthly_budget.py`` uses) could carry the
generator half, but not the flows: a bus no line enters vanishes from the
grouped sum.
"""

from __future__ import annotations

import json
from pathlib import Path

import linopy
import pandas as pd
import xarray as xr

DATA = Path(__file__).resolve().parents[2] / 'data' / 'transport.json'


def load_tables() -> dict[str, pd.DataFrame]:
    """The instance, one frame per parameter — what a caller of either library holds."""
    return {k: pd.DataFrame(v) if isinstance(v, dict) else v for k, v in json.loads(DATA.read_text()).items()}


def build(tables: dict[str, pd.DataFrame]) -> linopy.Model:
    """The instance's tables as a linopy model, row for row.

    ``tables`` is the same mapping the specsolve call attaches as ``sources``.
    """
    p_max: pd.Series = tables['p_max'].set_index('generator')['value']
    cost: pd.Series = tables['cost'].set_index('generator')['value']
    cap: pd.Series = tables['cap'].set_index('line')['value']
    neg_cap: pd.Series = tables['neg_cap'].set_index('line')['value']
    load = xr.DataArray(tables['load'].pivot(index='snapshot', columns='bus', values='value'))
    snapshots, buses = load.indexes['snapshot'], load.indexes['bus']

    gen_at = pd.DataFrame(0.0, index=buses, columns=p_max.index)
    for gen, bus in zip(tables['gen_bus']['generator'], tables['gen_bus']['bus'], strict=True):
        gen_at.loc[bus, gen] = 1.0
    flow_in = pd.DataFrame(0.0, index=buses, columns=cap.index)
    for line, src, dst in zip(
        tables['line_from']['line'], tables['line_from']['bus'], tables['line_to']['bus'], strict=True
    ):
        flow_in.loc[dst, line] += 1.0
        flow_in.loc[src, line] -= 1.0

    m = linopy.Model()
    p = m.add_variables(lower=0, upper=p_max, coords=[snapshots, p_max.index], name='p')
    f = m.add_variables(lower=neg_cap, upper=cap, coords=[snapshots, cap.index], name='f')
    m.add_constraints(
        (p * xr.DataArray(gen_at)).sum('generator') + (f * xr.DataArray(flow_in)).sum('line') == load,
        name='balance',
    )
    m.add_objective((p * cost).sum())
    return m


def nodal_prices(m: linopy.Model) -> dict[str, list]:
    """The balance dual, tidy: one price per (snapshot, bus)."""
    dual = m.constraints['balance'].dual.transpose('snapshot', 'bus')
    return {
        'snapshot': [int(s) for s in dual.indexes['snapshot'] for _ in dual.indexes['bus']],
        'bus': [str(b) for _ in dual.indexes['snapshot'] for b in dual.indexes['bus']],
        'value': [float(v) for v in dual.values.ravel()],
    }


def main() -> float:
    m = build(load_tables())
    status, condition = m.solve(solver_name='highs')
    assert status == 'ok', f'{status}: {condition}'
    print(f'linopy {linopy.__version__}')
    print(f'objective {float(m.objective.value)!r}')
    print(f'duals {json.dumps({"balance": nodal_prices(m)})}')
    return float(m.objective.value)


if __name__ == '__main__':
    main()
