#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pypsa==1.2.4", "linopy==0.9.0", "pandas>=2.2", "xarray==2026.7.0", "highspy==1.15.1"]
# ///
"""Reference for ``pypsa_committable_extendable``: PyPSA's own big-M commitment. See docs/examples/index.md.

    pixi exec -s uv uv run --script examples/ports/references/pypsa/pypsa_committable_extendable.py

Pinned above to the versions that produced the number in ``references.json``,
and run out of band — PyPSA is not a dependency of this project.

It reads the same instance the port attaches and builds the network with PyPSA's
own objects. Nothing here imports specsolve.

**A committed unit whose capacity is also being built needs a big-M.**
``p_min_pu * p_nom * status`` is a product of two variables the moment ``p_nom``
is one, so PyPSA linearises it with a constant *M* and emits three rows instead
of two (``constraints.py:304``):

    p - min_pu * p_nom - M * status >= -M     the floor, only while committed
    p - M * status <= 0                        off means zero
    p - max_pu * p_nom <= 0                    the capacity limit, whatever the status

*M* is **not** free: PyPSA takes ``p_nom_max * p_max_pu`` where a maximum is
declared and otherwise infers a scale from the network
(``components.py:971``), which is what ``create_model(committable_big_m=...)``
exists to override. This instance declares ``p_nom_max``, so *M* is 100 exactly
on both sides — the port passes the same number as data rather than inferring
one.

``main`` also shows what PyPSA's own relaxation does to this intersection, which
is the finding of this model: see :func:`what_pypsa_relaxes`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pypsa

DATA = Path(__file__).resolve().parents[2] / 'data' / 'pypsa_committable_extendable.json'


def load_tables() -> dict[str, pd.DataFrame]:
    """The instance, one frame per parameter — what a caller of either library holds."""
    return {k: pd.DataFrame(v) if isinstance(v, dict) else v for k, v in json.loads(DATA.read_text()).items()}


def build(tables: dict[str, pd.DataFrame]) -> pypsa.Network:
    """The port's tables as a PyPSA network, column for column.

    ``tables`` is the same mapping the specsolve call attaches as ``sources``.

    ``flex`` is the one asset that is **both** ``committable`` and
    ``p_nom_extendable`` — the intersection this model is about. ``peak`` is
    extendable and not committable, so it is the unit the load falls back on
    when ``flex`` cannot run below its minimum. Both ``*_time_before`` are 0 and
    the start-up and shut-down costs stay at their default 0: the transitions are
    ported by ``pypsa_unit_commitment`` and would give a mismatch here a second
    thing to be about.
    """
    n = pypsa.Network()
    n.set_snapshots(tables['snapshot']['snapshot'])
    n.add('Bus', 'hub')

    generators: pd.DataFrame = tables['generator'].set_index('generator')
    committable = generators.index == 'flex'
    n.add(
        'Generator',
        generators.index,
        bus='hub',
        committable=committable,
        up_time_before=0,
        down_time_before=0,
        p_nom_extendable=True,
        p_min_pu=tables['p_min_pu'].set_index('generator')['value'],
        p_nom_max=tables['p_nom_max'].set_index('generator')['value'],
        marginal_cost=tables['marginal_cost'].set_index('generator')['value'],
        capital_cost=tables['capital_cost'].set_index('generator')['value'],
    )

    n.add('Load', 'l', bus='hub', p_set=tables['load'].set_index('snapshot')['value'])
    return n


def what_pypsa_relaxes() -> None:
    """What ``linearized_unit_commitment=True`` does to a committable extendable.

    PyPSA tightens its relaxation for units whose start-up and shut-down costs
    match (``constraints.py:523``) — both are 0 here — and the block is built
    from ``nominal``, which for an extendable asset is the ``p_nom`` *column*,
    not the variable. That column is 0 for something not yet built, so every
    coefficient in the block collapses and the rows read ``p <= 0``:

        [1, flex]: +1 Generator-p[1, flex] - 0 Generator-status[1, flex] <= -0.0

    The dispatch is pinned to zero from the second snapshot on, and the "relaxed"
    objective comes out **above** the integer one — 32100.0 against 21700.0. A
    relaxation that excludes the integer optimum is not a bound, so the honest
    relaxation of this model is taken on the port's own model instead (the same
    rows with the status in [0, 1]), which reaches 18200.0.
    """
    n = build(load_tables())
    n.optimize(solver_name='highs', linearized_unit_commitment=True)
    print("\nthe same instance in PyPSA's own relaxed mode:")
    print(f'  objective {float(n.objective)!r} — above the integer {21700.0!r}, so not a relaxation')
    print(f'  flex dispatch {[round(v, 4) for v in n.generators_t.p["flex"]]}')
    for name in ('Generator-com-p-current', 'Generator-com-partly-start-up'):
        print(f'  {name}: {str(n.model.constraints[name]).splitlines()[3]}')


def main() -> float:
    n = build(load_tables())
    status, condition = n.optimize(solver_name='highs')
    assert status == 'ok', f'{status}: {condition}'
    print(f'pypsa {pypsa.__version__}')
    print(f'objective {float(n.objective)!r}')
    print(
        f'the big-M PyPSA used: {dict(n.c.generators.get_committable_big_m_values(names=pd.Index(["flex"])).to_series())}'
    )
    print(n.generators[['p_nom_opt', 'p_min_pu', 'p_nom_max']])
    print(n.generators_t.p)
    print(n.generators_t.status)
    print('\nthe three rows PyPSA emits for the intersection:')
    for name in ('Generator-com-ext-p-lower', 'Generator-com-ext-p-upper-bigM', 'Generator-com-ext-p-upper-cap'):
        print(f'  {name}: {str(n.model.constraints[name]).splitlines()[3]}')

    what_pypsa_relaxes()
    return float(n.objective)


if __name__ == '__main__':
    main()
