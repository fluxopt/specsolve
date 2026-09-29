#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pypsa==1.2.4", "linopy==0.9.0", "pandas>=2.2", "xarray==2026.7.0", "highspy==1.15.1"]
# ///
"""Reference for ``pypsa_losses``: PyPSA's own tangent transmission losses. See docs/examples/index.md.

    pixi exec -s uv uv run --script examples/ports/references/pypsa/pypsa_losses.py

Pinned above to the versions that produced the number in ``references.json``,
and run out of band — PyPSA is not a dependency of this project.

It reads the same instance the port attaches and builds the network with PyPSA's
own objects. Nothing here imports specsolve.

**A quadratic loss, underestimated by its own tangents.** Loss on a passive
branch is ``r * s**2``. PyPSA approximates it from below with a fan of tangent
lines: for each segment *k* it takes the point ``p_k = k/segments * s_nom`` and
adds the tangent there, once for each sign of the flow. Each is a half-plane on
``(loss, s)`` and needs no auxiliary variable at all, which is why this is a
plain linear model rather than a piecewise one.

The loss is subtracted **half at each end** of the branch, which is PyPSA's
convention for where the energy goes.

Three of the six snapshots are quiet, so the flows reach every segment of the
fan.

The network is a **path**, b0—b1—b2—b3, radial so there is no Kirchhoff
voltage law to satisfy. ``x`` is carried in the instance all the same, because
it is what makes these lines passive branches.

The last line has **no resistance**: PyPSA gives every passive branch a loss
variable and lets ``r = 0`` pin it to nothing, while the port declares the
variable only where there is a curve to approximate. Its rating binds.

``r`` is 0.0003 because PyPSA's loss term is ``r_pu_eff * s**2`` with ``s`` in
MW; a per-unit resistance would make the loss exceed the flow. At this value
the losses run about 3% of throughput.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pypsa

DATA = Path(__file__).resolve().parents[2] / 'data' / 'pypsa_losses.json'

#: What ``n.optimize`` is asked for, and what the port's tangent columns encode.
SEGMENTS = 3

#: The tolerances PyPSA's default *secant* mode derives its breakpoints from; the
#: port's second instance encodes the coefficients they produce. ``atol`` is 0.01
#: rather than PyPSA's default 1, so the breakpoints need four segments and the
#: ``rtol`` half of the step rule ``max(k / (k - 1), rtol_step)`` takes effect.
SECANT_TOLERANCES = {'atol': 0.01, 'rtol': 0.1}


def load_tables() -> dict[str, pd.DataFrame]:
    """The instance, one frame per parameter — what a caller of either library holds."""
    return {k: pd.DataFrame(v) if isinstance(v, dict) else v for k, v in json.loads(DATA.read_text()).items()}


def build(tables: dict[str, pd.DataFrame]) -> pypsa.Network:
    """The port's tables as a PyPSA network, column for column.

    ``tables`` is the same mapping the specsolve call attaches as ``sources``.

    PyPSA is given ``r``, ``x`` and ``s_nom`` and derives the tangents itself.
    The port is given the tangents, because a slope of ``2 * r * p_k`` is
    arithmetic and the language's coefficients take a name or a number — the
    same reason ``pypsa_storage`` ships ``soc_max`` rather than a ratio. Both
    sides therefore describe one model from the same instance, and
    ``SEGMENTS`` is the one number that has to agree between them.
    """
    n = pypsa.Network()
    n.set_snapshots(tables['snapshot']['snapshot'])
    n.add('Bus', tables['bus']['bus'])

    lines: pd.DataFrame = tables['line'].set_index('line')
    n.add(
        'Line',
        lines.index,
        bus0=lines['from'],
        bus1=lines['to'],
        r=tables['r'].set_index('line')['value'],
        x=tables['x'].set_index('line')['value'],
        s_nom=tables['s_nom'].set_index('line')['value'],
    )

    generators: pd.DataFrame = tables['generator'].set_index('generator')
    n.add(
        'Generator',
        generators.index,
        bus=generators['gen_bus'],
        p_nom=tables['p_nom'].set_index('generator')['value'],
        marginal_cost=tables['marginal_cost'].set_index('generator')['value'],
    )

    load: pd.DataFrame = tables['load'].pivot(index='snapshot', columns='bus', values='value')
    for bus in load.columns:
        n.add('Load', f'load_{bus}', bus=bus, p_set=load[bus])
    return n


def nodal_duals(n: pypsa.Network) -> dict[str, list]:
    """The dual of the nodal balance per (snapshot, bus), tidy.

    Read off the model: ``buses_t.marginal_price`` divides it by the snapshot
    weighting.
    """
    dual = n.model.constraints['Bus-nodal_balance'].dual.to_series()
    return {
        'snapshot': [int(s) for s, _ in dual.index],
        'bus': [str(b) for _, b in dual.index],
        'value': [float(v) for v in dual.to_numpy()],
    }


def secant_coefficients(n: pypsa.Network) -> dict[str, list]:
    """PyPSA's secant half-planes, tidy — ``(line, segment, slope, offset)``.

    Read off the constraints it built rather than recomputed here. **How the
    breakpoints are chosen is PyPSA's business**: the first sits at
    ``2 * sqrt(atol / r)`` and each next steps by
    ``max(k / (k - 1), 1 + 2 * (rtol + sqrt(rtol + rtol**2)))`` until the
    rating is covered, so even the *number* of segments is an output of their
    heuristic.

    Lines with no resistance are left out: their coefficients are all zero, and
    the model gives them no rows at all.
    """
    constraint = n.model.constraints['Line-loss_secants-pos']
    slopes = constraint.coeffs.isel(snapshot=0, _term=1)
    offsets = constraint.rhs.isel(snapshot=0)
    line, segment, slope, offset = [], [], [], []
    for name in constraint.indexes['name']:
        column = slopes.sel(name=name).values
        if not column.any():
            continue
        for k, (m, c) in enumerate(zip(column, offsets.sel(name=name).values, strict=True), start=1):
            line.append(str(name))
            segment.append(k)
            slope.append(float(m))
            offset.append(float(c))
    return {'line': line, 'segment': segment, 'slope': slope, 'offset': offset}


def secant_objective() -> float:
    """The same network under PyPSA's *default* loss mode, for the second instance.

    Secants lie above a convex curve where tangents lie below, so this
    overestimates the losses the tangent instance underestimates and costs more.
    The rows are the same shape either way — one half-plane per segment per sign
    of the flow.
    """
    n = build(load_tables())
    status, condition = n.optimize(solver_name='highs', transmission_losses={'mode': 'secants', **SECANT_TOLERANCES})
    assert status == 'ok', f'{status}: {condition}'
    return float(n.objective)


def main() -> float:
    n = build(load_tables())
    status, condition = n.optimize(solver_name='highs', transmission_losses={'mode': 'tangents', 'segments': SEGMENTS})
    assert status == 'ok', f'{status}: {condition}'
    print(f'pypsa {pypsa.__version__}')
    print(f'objective {float(n.objective)!r}')
    print(f'objective, secant mode {secant_objective()!r}')
    secant = build(load_tables())
    secant.optimize.create_model(transmission_losses={'mode': 'secants', **SECANT_TOLERANCES})
    print(f'secant coefficients {json.dumps(secant_coefficients(secant))}')
    print(f'duals {json.dumps({"nodal_balance": nodal_duals(n)})}')
    print(n.lines_t.p0)
    print(n.lines_t.loss)
    print(n.generators_t.p)
    return float(n.objective)


if __name__ == '__main__':
    main()
