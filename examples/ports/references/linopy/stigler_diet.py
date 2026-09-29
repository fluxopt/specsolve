#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["linopy==0.9.0", "pandas>=2.2", "xarray==2026.7.0", "highspy==1.15.1"]
# ///
"""Reference for ``stigler_diet``: the same LP, hand-written in linopy.

    pixi exec -s uv uv run --script examples/ports/references/linopy/stigler_diet.py

**Two things verify this port, and they answer different questions.**

The *published* figure is $39.69 a year. Jack Laderman computed it by hand in
1947 at the National Bureau of Standards as the first serious test of the
simplex method, so it carries his rounding.

This script is what the same data gives to a modern solver: $39.6617 a year,
0.08% under Laderman's. The gap is his rounding, not a different model, and the
*composition* of the diet is the stronger corroboration — both arrive at the
same five foods.

Nothing here imports specsolve. The model is a covering LP with no network in
it at all.

Pinned above to the versions that produced the number in ``references.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import linopy
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / 'data' / 'stigler_diet.json'

#: Laderman (1947), in 1939 dollars. What the port is checked against loosely;
#: `references.json` records this run's exact value for the tight check.
PUBLISHED_ANNUAL = 39.69


def load_tables() -> dict[str, pd.DataFrame]:
    """The instance, one frame per parameter — what a caller of either library holds."""
    return {k: pd.DataFrame(v) if isinstance(v, dict) else v for k, v in json.loads(DATA.read_text()).items()}


def build(tables: dict[str, pd.DataFrame]) -> linopy.Model:
    """The port's tables as a linopy model, column for column.

    ``tables`` is the same mapping the specsolve call attaches as ``sources``.
    ``per_dollar`` is the sparse table filled back out: a missing
    (food, nutrient) pair means that food supplies none of that nutrient.
    """
    foods = pd.Index(tables['food']['food'], name='food')
    minimum: pd.Series = tables['daily_minimum'].set_index('nutrient')['value']
    per_dollar: pd.DataFrame = (
        tables['nutrient_per_dollar']
        .pivot(index='food', columns='nutrient', values='value')
        .reindex(index=foods, columns=minimum.index)
        .fillna(0.0)
    )

    m = linopy.Model()
    spend = m.add_variables(lower=0, coords=[foods], name='spend')
    m.add_constraints((spend * per_dollar).sum('food') >= minimum, name='meet_requirement')
    m.add_objective(spend.sum())
    return m


def shadow_prices(m: linopy.Model) -> dict[str, list]:
    """What one more unit of each nutrient per day would cost.

    The price of a binding nutrient; the nutrients that are *not* binding come
    back at zero because they arrive free alongside the ones that are.
    """
    dual = m.constraints['meet_requirement'].dual
    return {'nutrient': [str(v) for v in dual.indexes['nutrient']], 'value': [float(v) for v in dual.values]}


def main() -> float:
    m = build(load_tables())
    status, condition = m.solve(solver_name='highs')
    assert status == 'ok', f'{status}: {condition}'
    daily = float(m.objective.value)
    print(f'linopy {linopy.__version__}')
    print(f'objective {daily!r}')
    print(f'annual {daily * 365:.4f} vs published {PUBLISHED_ANNUAL}')
    print(f'duals {json.dumps({"meet_requirement": shadow_prices(m)})}')
    chosen = m.solution['spend'].to_series()
    print((chosen[chosen > 1e-9] * 365).round(2))
    return daily


if __name__ == '__main__':
    main()
