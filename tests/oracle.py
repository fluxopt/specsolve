"""Guarded access to the linopy lane, used as the differential oracle.

Importing this module skips the importing test module when the oracle's
dependencies (the ``dev`` group's linopy, and xarray and pandas) are absent.
Import linopy, xarray and pandas through here, so import ordering cannot bypass
the guard.

The oracle is linopy's v1 convention only: legacy fills every absent slot with
0 (PyPSA/linopy#712). A linopy without it raises rather than skips.
"""

from __future__ import annotations

import pytest

_REASON = 'needs the oracle (linopy from the dev group, xarray, pandas)'

linopy = pytest.importorskip('linopy', reason=_REASON)
xr = pytest.importorskip('xarray', reason=_REASON)
pd = pytest.importorskip('pandas', reason=_REASON)

if 'semantics' not in getattr(linopy.options, '_defaults', {}):
    raise RuntimeError(
        f'linopy {linopy.__version__} has no options["semantics"], so it cannot speak the v1 '
        f'arithmetic convention this package is written against. The oracle would silently '
        f'measure against the legacy convention instead. Install the pin in pyproject.toml '
        f'(the dev group: PyPSA/linopy@master) — `pixi install`.'
    )
from tests import linopy_lane as specsolve_linopy  # noqa: E402  — must follow the guard above
from tests.linopy_lane import builder, loader, operators, where  # noqa: E402

__all__ = [
    'builder',
    'linopy',
    'loader',
    'operators',
    'pd',
    'specsolve_linopy',
    'transport_linopy_objective',
    'where',
    'xr',
]


def transport_linopy_objective(gens, lines, load) -> float:
    gi = gens.set_index('generator')
    li = lines.set_index('line')
    snapshots = pd.Index(sorted(load['snapshot'].unique()), name='snapshot')
    buses = pd.Index(sorted(load['bus'].unique()), name='bus')

    load_da = xr.DataArray.from_series(load.set_index(['snapshot', 'bus'])['value'])
    p_max = xr.DataArray.from_series(gi['p_max'])
    cost = xr.DataArray.from_series(gi['cost'])
    cap = xr.DataArray.from_series(li['cap'])

    gen_at = xr.DataArray(
        (gi['bus'].to_numpy()[None, :] == buses.to_numpy()[:, None]).astype(float),
        coords={'bus': buses, 'generator': gi.index},
        dims=['bus', 'generator'],
    )
    line_in = xr.DataArray(
        (li['to_bus'].to_numpy()[None, :] == buses.to_numpy()[:, None]).astype(float),
        coords={'bus': buses, 'line': li.index},
        dims=['bus', 'line'],
    )
    line_out = xr.DataArray(
        (li['from_bus'].to_numpy()[None, :] == buses.to_numpy()[:, None]).astype(float),
        coords={'bus': buses, 'line': li.index},
        dims=['bus', 'line'],
    )

    m = linopy.Model()
    p = m.add_variables(lower=0, upper=p_max, coords=[snapshots, gi.index], name='p')
    f = m.add_variables(lower=-cap, upper=cap, coords=[snapshots, li.index], name='f')
    injection = (p * gen_at).sum('generator') + (f * line_in).sum('line') - (f * line_out).sum('line')
    m.add_constraints(injection == load_da, name='balance')
    m.add_objective((p * cost).sum())
    m.solve(solver_name='highs', output_flag=False)
    return float(m.objective.value)
