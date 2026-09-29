#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["linopy==0.9.0", "pandas>=2.2", "xarray==2026.7.0", "highspy==1.15.1"]
# ///
"""Reference for ``piecewise_conversion``: the same MILP, hand-written in linopy.

    pixi exec -s uv uv run --script examples/ports/references/linopy/piecewise_conversion.py

A teaching model, so what verifies it is agreement with an independent
formulation, not a published figure — here linopy's own
``add_piecewise_formulation``, which ties N expressions to one basis exactly as
the block does.

**The arity is written out, once per converter.** linopy's call takes the pairs
as arguments, so how many flows a converter ties is the length of an argument
list built in Python — two for the boiler, three for the CHP, and a loop over
the converters to build each. In the YAML the tie is one constraint over
``flow``, and a converter with a fourth flow is a row in a table.

Ragged curves need no padding on either side: each call carries that
converter's own breakpoints, three for one and four for the other.
"""

from __future__ import annotations

import json
from pathlib import Path

import linopy
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / 'data' / 'piecewise_conversion.json'


def load_tables() -> dict[str, pd.DataFrame]:
    """The instance, one frame per parameter — what a caller of either library holds."""
    return {k: pd.DataFrame(v) if isinstance(v, dict) else v for k, v in json.loads(DATA.read_text()).items()}


def curve_of(tables: dict[str, pd.DataFrame], flow: str, runs_to: int) -> pd.Series:
    """One flow's breakpoints, cut to the length its converter's curve runs."""
    values = tables['bp_rate'].set_index(['flow', 'bp'])['value']
    return values[flow].sort_index().iloc[:runs_to]


def build(tables: dict[str, pd.DataFrame]) -> linopy.Model:
    """The instance's tables as a linopy model, row for row.

    ``tables`` is the same mapping the specsolve call attaches as ``sources``.
    """
    flows: pd.DataFrame = tables['flow'].set_index('flow')
    times = pd.Index(tables['time']['time'], name='time')
    present = tables['bp_present'].set_index(['converter', 'bp'])['value']
    runs_to = {c: int(present[c].sum()) for c in present.index.get_level_values('converter').unique()}

    m = linopy.Model()
    rate = m.add_variables(lower=0, coords=[pd.Index(flows.index, name='flow'), times], name='rate')

    for converter, members in flows.groupby('converter_of'):
        pairs = [
            (rate.sel(flow=flow, drop=True), linopy.breakpoints(curve_of(tables, flow, runs_to[converter])))
            for flow in members.index
        ]
        m.add_piecewise_formulation(*pairs, name=f'curve_{converter}')

    for carrier, demand in (('is_heat', 'heat_demand'), ('is_power', 'power_demand')):
        weights = tables[carrier].set_index('flow')['value'].reindex(flows.index)
        wanted = tables[demand].set_index('time')['value'].reindex(times)
        m.add_constraints((rate * weights).sum('flow') == wanted, name=demand)

    price = tables['fuel_price'].set_index('flow')['value'].reindex(flows.index)
    m.add_objective((rate * price).sum())
    return m


def main() -> float:
    m = build(load_tables())
    status, condition = m.solve(solver_name='highs')
    assert status == 'ok', f'{status}: {condition}'
    print(f'linopy {linopy.__version__}')
    print(f'objective {float(m.objective.value)!r}')
    return float(m.objective.value)


if __name__ == '__main__':
    main()
