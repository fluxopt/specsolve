#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["linopy==0.9.0", "pandas>=2.2", "xarray==2026.7.0", "highspy==1.15.1"]
# ///
"""Reference for ``piecewise_ragged``: the same LP, hand-written in linopy.

    pixi exec -s uv uv run --script examples/ports/references/linopy/piecewise_ragged.py

A teaching model, so what verifies it is agreement with an independent
formulation, not a published figure — see ``piecewise.py`` next door, whose
curves are all the same length.

**A second formulation, not a second spelling.** The YAML states the hull as
weights over each curve's own breakpoints; this states it as the segment lines
themselves — ``op_cost >= slope * p + intercept``, one row per real segment,
which is exact for a convex curve under minimisation and needs no weights at
all.

``linopy.add_piecewise_formulation`` takes a rectangle, so a hydro unit with
two breakpoints in a four-breakpoint frame would repeat its last point — and
linopy then reads the zero-length segment as non-convex and reaches for SOS2, a
mixed-integer model with no duals.
"""

from __future__ import annotations

import json
from pathlib import Path

import linopy
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / 'data' / 'piecewise_ragged.json'


def load_tables() -> dict[str, pd.DataFrame]:
    """The instance, one frame per parameter — what a caller of either library holds."""
    return {k: pd.DataFrame(v) if isinstance(v, dict) else v for k, v in json.loads(DATA.read_text()).items()}


def segments(tables: dict[str, pd.DataFrame]) -> dict[str, list[tuple[float, float]]]:
    """Each generator's own curve as ``(slope, intercept)`` per segment.

    Read off the rows that exist, so a curve is as long as its data — which is
    what ``points: bp_x`` says on the other side.
    """
    x = tables['bp_x'].set_index(['generator', 'bp'])['value']
    y = tables['bp_y'].set_index(['generator', 'bp'])['value']
    lines: dict[str, list[tuple[float, float]]] = {}
    for g in x.index.get_level_values('generator').unique():
        xs, ys = x[g].sort_index(), y[g].sort_index()
        lines[g] = [
            (
                (ys.iloc[k + 1] - ys.iloc[k]) / (xs.iloc[k + 1] - xs.iloc[k]),
                ys.iloc[k] - xs.iloc[k] * (ys.iloc[k + 1] - ys.iloc[k]) / (xs.iloc[k + 1] - xs.iloc[k]),
            )
            for k in range(len(xs) - 1)
        ]
    return lines


def build(tables: dict[str, pd.DataFrame]) -> linopy.Model:
    """The instance's tables as a linopy model, row for row.

    ``tables`` is the same mapping the specsolve call attaches as ``sources``.
    """
    p_max: pd.Series = tables['p_max'].set_index('generator')['value']
    load: pd.Series = tables['load'].set_index('snapshot')['value']
    reach = tables['bp_x'].groupby('generator')['value'].max().reindex(p_max.index)

    m = linopy.Model()
    p = m.add_variables(lower=0, upper=reach, coords=[load.index, p_max.index], name='p')
    op_cost = m.add_variables(lower=0, coords=[load.index, p_max.index], name='op_cost')
    for g, lines in segments(tables).items():
        for k, (slope, intercept) in enumerate(lines):
            m.add_constraints(
                op_cost.sel(generator=g) - slope * p.sel(generator=g) >= intercept,
                name=f'chord_{g}_{k}',
            )
    m.add_constraints(p.sum('generator') == load, name='balance')
    m.add_objective(op_cost.sum())
    return m


def marginal_prices(m: linopy.Model) -> dict[str, list]:
    """The balance dual — the marginal cost read off each period's active segment."""
    dual = m.constraints['balance'].dual
    return {'snapshot': [int(v) for v in dual.indexes['snapshot']], 'value': [float(v) for v in dual.values]}


def main() -> float:
    m = build(load_tables())
    status, condition = m.solve(solver_name='highs')
    assert status == 'ok', f'{status}: {condition}'
    print(f'linopy {linopy.__version__}')
    print(f'objective {float(m.objective.value)!r}')
    print(f'duals {json.dumps({"balance": marginal_prices(m)})}')
    return float(m.objective.value)


if __name__ == '__main__':
    main()
