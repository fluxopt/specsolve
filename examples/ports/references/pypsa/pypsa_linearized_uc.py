#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pypsa==1.2.4", "linopy==0.9.0", "pandas>=2.2", "xarray==2026.7.0", "highspy==1.15.1"]
# ///
"""Reference for ``pypsa_linearized_uc``: PyPSA's own linearized unit commitment. See docs/examples/index.md.

    pixi exec -s uv uv run --script examples/ports/references/pypsa/pypsa_linearized_uc.py

Pinned above to the versions that produced the number in ``references.json``,
and run out of band — PyPSA is not a dependency of this project.

It reads the same instance the port attaches and builds the network with PyPSA's
own objects. Nothing here imports specsolve.

**The same rows, with the status continuous.**
``optimize(linearized_unit_commitment=True)`` declares the status
and the two transition variables in [0, 1] instead of {0, 1} and leaves every
constraint where it was. A unit may then be committed by a third.

``base`` carries unequal start-up and shut-down costs. PyPSA tightens the
relaxation with an extra dispatch-limit block wherever a generator's two costs
*match*, and that block reaches for the ramp-limit parameters, so ``base`` is
left untightened. ``peak`` is not, its two costs both being zero: PyPSA emits four
further blocks for it, every one of which collapses to a row the port already
holds, since ``p_min_pu`` is 0 and there are no ramp limits. Hence the same
objective and the same prices out of a model with more rows in it.

The relaxation is a bound, not an approximation: on this instance it is worth
less than half the integer answer, and ``main`` prints both.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pypsa

DATA = Path(__file__).resolve().parents[2] / 'data' / 'pypsa_linearized_uc.json'


def load_tables() -> dict[str, pd.DataFrame]:
    """The instance, one frame per parameter — what a caller of either library holds."""
    return {k: pd.DataFrame(v) if isinstance(v, dict) else v for k, v in json.loads(DATA.read_text()).items()}


def build(tables: dict[str, pd.DataFrame]) -> pypsa.Network:
    """The port's tables as a PyPSA network, column for column.

    ``tables`` is the same mapping the specsolve call attaches as ``sources``.

    Nothing here says the model is relaxed: ``committable=True`` is the same
    switch the integer model uses, and the mode is chosen at ``optimize`` time.
    Both ``*_time_before`` are 0, so no status is pinned by a prior horizon.
    """
    n = pypsa.Network()
    n.set_snapshots(tables['snapshot']['snapshot'])
    n.add('Bus', 'hub')

    generators: pd.DataFrame = tables['generator'].set_index('generator')
    n.add(
        'Generator',
        generators.index,
        bus='hub',
        committable=True,
        up_time_before=0,
        down_time_before=0,
        p_nom=tables['p_nom'].set_index('generator')['value'],
        p_min_pu=tables['p_min_pu'].set_index('generator')['value'],
        marginal_cost=tables['marginal_cost'].set_index('generator')['value'],
        start_up_cost=tables['start_up_cost'].set_index('generator')['value'],
        shut_down_cost=tables['shut_down_cost'].set_index('generator')['value'],
    )

    n.add('Load', 'l', bus='hub', p_set=tables['load'].set_index('snapshot')['value'])
    return n


def balance_duals(n: pypsa.Network) -> dict[str, list]:
    """The dual of the power balance per snapshot, tidy.

    A relaxed commitment is an LP, so it has a dual solution. Keyed by snapshot
    alone, as the port has no bus dimension.
    """
    dual = n.model.constraints['Bus-nodal_balance'].dual.to_series()
    return {
        'snapshot': [int(s) for s, _ in dual.index],
        'value': [float(v) for v in dual.to_numpy()],
    }


def main() -> float:
    n = build(load_tables())
    status, condition = n.optimize(solver_name='highs', linearized_unit_commitment=True)
    assert status == 'ok', f'{status}: {condition}'
    print(f'pypsa {pypsa.__version__}')
    print(f'objective {float(n.objective)!r}')
    print(f'duals {json.dumps({"power_balance": balance_duals(n)})}')
    print(n.generators_t.status)

    integer = build(load_tables())
    integer.optimize(solver_name='highs')
    print(f'the same instance as a MILP: {float(integer.objective)!r}')
    return float(n.objective)


if __name__ == '__main__':
    main()
