# SPDX-FileCopyrightText: mathspec Contributors
#
# SPDX-License-Identifier: MIT

"""The prep layer: a PyPSA network as the tables the example specs declare.

Every parameter the files mark "data prep" is computed here, beside the plain
renames — the prep half of how specsolve builds the corpus's specs, shown on
the ladder page beside the tables it produces. `parity.py` is the caller and
cuts the tables to what each spec declares; nothing here imports mathspec
or specsolve — the mapping is pure PyPSA-and-pandas, handed over as polars frames.

Sparseness is meaning: a table row left out is an absent value on the other
side, so the sparse tables here (`*_set` pins, ramp limits, weights) drop
their empty rows instead of shipping fills.

A table PyPSA keeps once for every scenario is written once, without a
``scenario`` column; `parity.py` spreads it over the scenarios a spec reads it
by. A plain run is one scenario and one period, each labelled here.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl
from pypsa.descriptors import get_switchable_as_dense

if TYPE_CHECKING:
    import pypsa


#: PyPSA component -> the dimension the file declares for it.
DIM = {
    'Generator': 'generator',
    'Link': 'link',
    'Process': 'process',
    'Load': 'load',
    'StorageUnit': 'storage_unit',
    'Store': 'store',
    'Line': 'line',
    'Transformer': 'transformer',
    'GlobalConstraint': 'global_constraint',
}

#: The one scenario of a network that declares none.
PLAIN_SCENARIO = 'base'

#: The one investment period of a network that declares none.
PLAIN_PERIOD = 0

#: The components whose build is a decision, with the nominal attribute PyPSA names it by.
NOMINAL = {
    'Generator': 'p_nom',
    'Link': 'p_nom',
    'Process': 'p_nom',
    'StorageUnit': 'p_nom',
    'Store': 'e_nom',
    'Line': 's_nom',
    'Transformer': 's_nom',
}

#: The components that commit, ramp and go into maintenance, each over its own power.
DISPATCHABLE = ('Generator', 'Link', 'Process')


def names(index: pd.Index) -> pd.Index:
    """A component index as its names — the ``name`` level once a network with scenarios stacks ``(scenario, name)``."""
    return index.get_level_values('name').unique() if index.nlevels > 1 else index


def keyed(index: pd.Index, dim: str) -> dict[str, object]:
    """The key columns a component index spells — *dim*, under a ``scenario`` column where the index carries one."""
    if index.nlevels > 1:
        return {'scenario': index.get_level_values('scenario'), dim: index.get_level_values('name').astype(str)}
    return {dim: index.astype(str)}


def timesteps(n: pypsa.Network) -> pd.Index:
    """The snapshots as the file's flat ``snapshot`` axis — the ``timestep`` level once a multi-period network stacks ``(period, timestep)``."""
    return n.snapshots.get_level_values('timestep') if n.snapshots.nlevels > 1 else n.snapshots


def scenario_labels(n: pypsa.Network) -> list[str]:
    return list(n.scenarios.astype(str)) if n.has_scenarios else [PLAIN_SCENARIO]


def period_labels(n: pypsa.Network) -> list[int]:
    return list(n.investment_periods) if n.snapshots.nlevels > 1 else [PLAIN_PERIOD]


def period_of(n: pypsa.Network) -> np.ndarray:
    """Each snapshot's investment period, in snapshot order."""
    if n.snapshots.nlevels > 1:
        return n.snapshots.get_level_values('period').to_numpy()
    return np.full(len(n.snapshots), PLAIN_PERIOD)


def first_scenario(table: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """A per-component table read once, from the first scenario — what PyPSA refuses to differ by scenario."""
    if table.index.nlevels > 1:
        return table.groupby(level='name', sort=False).first()
    return table


def static(n: pypsa.Network, component: str, attr: str, default: object = float('nan')) -> pd.DataFrame:
    """A static attribute as ``(dim, value)``, one row per component — per scenario where the network has them."""
    table = n.static(component)
    values = table[attr].to_numpy() if attr in table.columns else [default] * len(table)
    return pd.DataFrame(keyed(table.index, DIM[component]) | {'value': values})


def varying(n: pypsa.Network, component: str, attr: str) -> pd.DataFrame:
    """A time-varying attribute as ``(snapshot, dim, value)``, static values broadcast over the snapshots as PyPSA does."""
    dense = get_switchable_as_dense(n, component, attr).set_axis(timesteps(n), axis=0)
    dense.columns.names = ['scenario', DIM[component]] if dense.columns.nlevels > 1 else [DIM[component]]
    table = dense.melt(ignore_index=False).reset_index(names='snapshot')
    return table.astype({DIM[component]: str, 'value': float})


def relation(n: pypsa.Network, component: str, attr: str, into: str = 'bus') -> pd.DataFrame:
    """What a component's *attr* names, as the relation the file declares over it *into* a dimension; a blank names none."""
    table = first_scenario(n.static(component))
    named = table[attr].astype(str) if attr in table.columns else pd.Series('', index=table.index, dtype=str)
    out = pd.DataFrame({DIM[component]: table.index.astype(str), into: named.to_numpy()})
    return out[out[into] != '']


def per_component(component: str, values: pd.Series, dtype: object = float) -> pd.DataFrame:
    """One value per component, from a series indexed by name — or by ``(scenario, name)``."""
    return pd.DataFrame(keyed(values.index, DIM[component]) | {'value': values.to_numpy()}).astype({'value': dtype})


def long(component: str, dense: pd.DataFrame, dtype: object = float) -> pd.DataFrame:
    """A snapshot-by-component frame read long as ``(snapshot, dim, value)``."""
    frame = dense.copy()
    frame.index = pd.Index(frame.index, name='snapshot')
    frame.columns = pd.Index(frame.columns.astype(str), name=DIM[component])
    return frame.melt(ignore_index=False).reset_index().astype({'value': dtype})


# ---------------------------------------------------------------------------
# where an asset stands: investment periods
# ---------------------------------------------------------------------------


def active_by_period(n: pypsa.Network, component: str) -> pd.DataFrame:
    """Whether each asset stands in each investment period, names by periods — one column on a plain run."""
    c = n.components[component]
    if n.snapshots.nlevels > 1:
        table = pd.concat({p: c.get_active_assets(p) for p in n.investment_periods}, axis=1)
    else:
        table = c.get_active_assets().to_frame(PLAIN_PERIOD)
    return first_scenario(table).astype(bool)


def period_weights(n: pypsa.Network, column: str) -> pd.Series:
    if n.snapshots.nlevels > 1:
        return n.investment_period_weightings[column].loc[list(n.investment_periods)].astype(float)
    return pd.Series([1.0], index=[PLAIN_PERIOD])


def active_by_snapshot(n: pypsa.Network, component: str) -> pd.DataFrame:
    """Whether each asset stands at each snapshot — its period's flag, snapshots by names."""
    by_period = active_by_period(n, component)
    frame = by_period[period_of(n)].T
    frame.index = timesteps(n)
    return frame


def standing(n: pypsa.Network, component: str) -> dict[str, object]:
    """``active``, ``capital_weight`` and ``first_active`` of one component — where it stands, and what that weighs."""
    prefix = component
    by_period = active_by_period(n, component)
    weight = period_weights(n, 'objective')
    capital = (by_period.astype(float) * weight.reindex(by_period.columns).to_numpy()).sum(axis=1)
    earlier = by_period.astype(int).cumsum(axis=1).shift(1, axis=1).fillna(0) > 0
    first = (by_period & ~earlier).astype(float).T
    first.index = pd.Index(first.index, name='period')
    first.columns = pd.Index(first.columns.astype(str), name=DIM[component])
    return {
        f'{prefix}_active': long(component, active_by_snapshot(n, component), bool),
        f'{prefix}_capital_weight': per_component(component, capital),
        f'{prefix}_first_active': first.melt(ignore_index=False).reset_index().astype({'value': float}),
    }


def late_opening(n: pypsa.Network, component: str) -> dict[str, object]:
    """``opens_late`` and ``inactive_snapshots`` of a storage component — where it first stands past the first snapshot, and how long it does not."""
    active = active_by_snapshot(n, component)
    first = (active.astype(int).cumsum() == 1) & active
    first.iloc[0] = False
    return {
        f'{component}_opens_late': long(component, first, bool),
        f'{component}_inactive_snapshots': per_component(component, (~active).sum(), int),
    }


# ---------------------------------------------------------------------------
# ports: a link's outputs and a process's ports, read long
# ---------------------------------------------------------------------------


def _ports(n: pypsa.Network, component: str) -> list[tuple[str, str, str]]:
    """``(port, bus column, coefficient attribute)`` per port a multiport declares.

    A link's ports are its outputs, ``bus1`` on, each with ``efficiency``,
    ``efficiency2``, …; a process's are every bus it names, ``bus0`` on, each
    with ``rate0``, ``rate1``, …. PyPSA spells ``delay`` and ``cyclic_delay``
    the same way as the coefficient.
    """
    c = n.components[component]
    if component == 'Link':
        return [(p, f'bus{p}', 'efficiency' if p == '1' else f'efficiency{p}') for p in ['1', *c.additional_ports]]
    return [(str(p), f'bus{p}', f'rate{p}') for p in c.ports]


def _delay_attrs(component: str, coefficient: str) -> tuple[str, str]:
    suffix = coefficient.removeprefix('efficiency').removeprefix('rate')
    return f'delay{suffix}', f'cyclic_delay{suffix}'


def ports(n: pypsa.Network, component: str) -> dict[str, object]:
    """A multiport's ports read long — one label per port a component declares, its bus, its coefficient over the snapshots, its delay.

    PyPSA spells the ports across columns, and a component declares a port by
    naming a bus in one, so a component of any port count is as many labels
    here and one term in the balance. The label is the component and the
    column the port came from.
    """
    dim, port_dim = DIM[component], f'{DIM[component]}_output'
    table = n.static(component)
    blank = pd.Series('', index=table.index, dtype=str)
    statics, coefficients = [], []
    for port, bus_column, coefficient in _ports(n, component):
        delay_attr, cyclic_attr = _delay_attrs(component, coefficient)
        buses = table.get(bus_column, blank).astype(str)
        declared = buses.to_numpy() != ''
        delays = table.get(delay_attr, pd.Series(0, index=table.index)).fillna(0).astype(int)
        cyclic = table.get(cyclic_attr, pd.Series(True, index=table.index)).fillna(True).astype(bool)
        keys = keyed(table.index, dim)
        frame = pd.DataFrame(
            keys | {'bus': buses.to_numpy(), 'delay': delays.to_numpy(), 'cyclic_delay': cyclic.to_numpy()}
        )
        frame[port_dim] = frame[dim] + '_bus' + port
        statics.append(frame[declared])
        dense = get_switchable_as_dense(n, component, coefficient).set_axis(timesteps(n), axis=0)
        dense.columns.names = ['scenario', dim] if dense.columns.nlevels > 1 else [dim]
        melted = dense.melt(ignore_index=False).reset_index(names='snapshot').astype({dim: str, 'value': float})
        melted[port_dim] = melted[dim] + '_bus' + port
        named = set(frame.loc[declared, port_dim])
        coefficients.append(melted[melted[port_dim].isin(named)].drop(columns=dim))
    every = pd.concat(statics, ignore_index=True)
    keys = [k for k in ('scenario', port_dim) if k in every.columns]
    once = every.drop_duplicates(port_dim)
    prefix = component
    return {
        port_dim: pl.Series(port_dim, list(pd.unique(once[port_dim])), dtype=pl.String),
        f'{prefix}_output_{dim}': once[[port_dim, dim]].reset_index(drop=True),
        f'{prefix}_output_bus': once[[port_dim, 'bus']].reset_index(drop=True),
        f'{prefix}_output_delay': every[[*keys, 'delay']].rename(columns={'delay': 'value'}),
        f'{prefix}_output_cyclic_delay': every[[*keys, 'cyclic_delay']].rename(columns={'cyclic_delay': 'value'}),
        f'{prefix}_{"efficiency" if component == "Link" else "rate"}': pd.concat(coefficients, ignore_index=True),
    }


# ---------------------------------------------------------------------------
# commitment, modules and maintenance: Generator, Link and Process alike
# ---------------------------------------------------------------------------


def _remaining(n: pypsa.Network, component: str, before: str, minimum: str) -> pd.DataFrame:
    """True while the up or down time a unit brought into the horizon still binds — PyPSA's ``min - before`` snapshots, where ``before > 0``."""
    table = n.static(component)
    rows = []
    for key, unit in table.iterrows():
        if not unit['committable'] or unit[before] <= 0:
            continue
        remaining = int(min(max(unit[minimum] - unit[before], 0), len(n.snapshots)))
        scenario = {'scenario': key[0]} if table.index.nlevels > 1 else {}
        name = key[1] if table.index.nlevels > 1 else key
        rows.extend(
            scenario | {'snapshot': t, DIM[component]: str(name), 'value': True} for t in timesteps(n)[:remaining]
        )
    columns = [*(['scenario'] if table.index.nlevels > 1 else []), 'snapshot', DIM[component], 'value']
    return pd.DataFrame(rows, columns=columns).astype({'value': bool})


def _modules_installed(n: pypsa.Network, component: str) -> pd.DataFrame:
    """The whole modules a build has standing — ``p_nom / p_nom_mod`` where a fixed build is modular, one where it is not.

    PyPSA refuses a fixed modular build whose nominal power is not a whole number of modules, so the
    division is exact and left unrounded: a fraction here is a network this prep should not have taken.
    """
    table = n.static(component)
    modular = ~table['p_nom_extendable'] & (table.get('p_nom_mod', 0.0) > 0)
    counts = table['p_nom'].where(modular, 1.0) / table.get('p_nom_mod', 1.0).where(modular, 1.0)
    return per_component(component, counts)


def _big_m(n: pypsa.Network, component: str) -> pd.DataFrame:
    """PyPSA's own big-M for a committable extendable build — the build cap at full availability."""
    c = n.components[component]
    applies = c.committables.intersection(c.extendables)
    if applies.empty:
        return pd.DataFrame({DIM[component]: [], 'value': []}).astype({DIM[component]: str, 'value': float})
    values = c.get_committable_big_m_values(names=applies).to_series()
    return (
        per_component(component, values)
        if values.index.nlevels == 1
        else per_component(component, values.reorder_levels(['scenario', 'name']))
    )


def maintenance(n: pypsa.Network, component: str) -> dict[str, object]:
    """What maintenance a component takes: the events, their cover and where none may start.

    A start covers itself and the snapshots after it until their generator
    weightings reach the duration — PyPSA's first window end whose cumulative
    weighting meets ``maintenance_duration`` less a relative ``1e-6``. A start
    is blocked where its window runs past the horizon or into a snapshot the
    unit does not stand in.
    """
    dim = DIM[component]
    table = n.static(component)
    maintainable = table.get('maintainable', pd.Series(False, index=table.index)).astype(bool)
    weights = n.snapshot_weightings['generators'].to_numpy(dtype=float)
    cum = weights.cumsum()
    cum_prev = cum - weights
    stamps = list(timesteps(n))
    total = len(stamps)
    active = active_by_snapshot(n, component)
    covers, blocked = [], []
    for key, unit in table.iterrows():
        if not maintainable.loc[key]:
            continue
        scenario = {'scenario': key[0]} if table.index.nlevels > 1 else {}
        name = str(key[1] if table.index.nlevels > 1 else key)
        ends = np.searchsorted(cum, float(unit['maintenance_duration']) * (1 - 1e-6) + cum_prev)
        standing_ = active[name].to_numpy() if name in active.columns else np.ones(total, dtype=bool)
        for start, end in enumerate(ends):
            covers.extend(
                scenario | {dim: name, 'start': stamps[start], 'covered': stamps[t]}
                for t in range(start, min(end, total - 1) + 1)
            )
            inside = range(start, min(end, total - 1) + 1)
            valid = standing_[start] and end < total and all(standing_[t] for t in inside)
            blocked.append(
                scenario | {'snapshot': stamps[start], dim: name, 'value': bool(standing_[start] and not valid)}
            )
    keys = ['scenario'] if table.index.nlevels > 1 else []
    return {
        f'{component}_maintainable': per_component(component, first_scenario(maintainable), bool),
        f'{component}_maintenance_pu': static(n, component, 'maintenance_pu')[lambda f: f['value'].notna()],
        f'{component}_maintenance_events': static(n, component, 'maintenance_events', 0)
        .fillna({'value': 0})
        .astype({'value': int}),
        f'{component}_maintenance_duration': pd.DataFrame(
            keyed(table.index, dim)
            | {'value': table.get('maintenance_duration', pd.Series(np.nan, index=table.index)).to_numpy()}
        )[maintainable.to_numpy()],
        f'{component}_maintenance_cover': pd.DataFrame(covers, columns=[*keys, dim, 'start', 'covered']),
        f'{component}_maintenance_start_blocked': pd.DataFrame(
            blocked, columns=[*keys, 'snapshot', dim, 'value']
        ).astype({'value': bool}),
    }


def dispatchable(n: pypsa.Network, component: str) -> dict[str, object]:
    """Commitment, ramps, modules, maintenance and costs of a generator, link or process — PyPSA treats the three alike."""
    p = component
    table = n.static(component)
    up_before = table.get('up_time_before', pd.Series(0, index=table.index))
    p_init = table.get('p_init', pd.Series(np.nan, index=table.index))
    initial = per_component(component, (up_before > 0).astype(int), int)
    min_pu = get_switchable_as_dense(n, component, 'p_min_pu')
    return {
        f'{p}_p_nom': static(n, component, 'p_nom'),
        f'{p}_p_nom_extendable': static(n, component, 'p_nom_extendable'),
        f'{p}_p_min_pu': varying(n, component, 'p_min_pu'),
        f'{p}_p_max_pu': varying(n, component, 'p_max_pu'),
        f'{p}_marginal_cost': varying(n, component, 'marginal_cost'),
        f'{p}_marginal_cost_quadratic': varying(n, component, 'marginal_cost_quadratic'),
        f'{p}_p_set': varying(n, component, 'p_set').dropna(),
        f'{p}_p_nom_min': static(n, component, 'p_nom_min'),
        f'{p}_p_nom_max': static(n, component, 'p_nom_max'),
        f'{p}_capital_cost': per_component(component, n.components[component].periodized_cost.to_series())
        if not n.has_scenarios
        else _periodized(n, component),
        f'{p}_p_nom_set': static(n, component, 'p_nom_set').dropna(),
        f'{p}_committable': static(n, component, 'committable'),
        f'{p}_ramp_limit_up': varying(n, component, 'ramp_limit_up').dropna(),
        f'{p}_ramp_limit_down': varying(n, component, 'ramp_limit_down').dropna(),
        f'{p}_ramp_limit_start_up': static(n, component, 'ramp_limit_start_up').dropna(),
        f'{p}_ramp_limit_shut_down': static(n, component, 'ramp_limit_shut_down').dropna(),
        f'{p}_min_up_time': static(n, component, 'min_up_time', 0),
        f'{p}_min_down_time': static(n, component, 'min_down_time', 0),
        f'{p}_status_initial': initial,
        f'{p}_p_init': pd.DataFrame(
            keyed(table.index, DIM[component]) | {'value': p_init.where(up_before > 0, 0.0).to_numpy(dtype=float)}
        ).dropna(),
        f'{p}_must_stay_up': _remaining(n, component, 'up_time_before', 'min_up_time'),
        f'{p}_must_stay_down': _remaining(n, component, 'down_time_before', 'min_down_time'),
        f'{p}_start_up_cost': static(n, component, 'start_up_cost', 0.0),
        f'{p}_shut_down_cost': static(n, component, 'shut_down_cost', 0.0),
        f'{p}_stand_by_cost': varying(n, component, 'stand_by_cost'),
        f'{p}_p_nom_mod': static(n, component, 'p_nom_mod', 0.0).query('value > 0'),
        f'{p}_modules_installed': _modules_installed(n, component),
        f'{p}_big_m': _big_m(n, component),
        f'{p}_p_min_pu_nonneg': per_component(
            component,
            (min_pu >= 0).all().groupby(level='name').all() if min_pu.columns.nlevels > 1 else (min_pu >= 0).all(),
            bool,
        ),
        f'{p}_partly_tightened': per_component(
            component, first_scenario(table['start_up_cost'] == table['shut_down_cost']), bool
        ),
        **maintenance(n, component),
        **standing(n, component),
    }


def _periodized(n: pypsa.Network, component: str) -> pd.DataFrame:
    """PyPSA's periodized capital cost per scenario and component."""
    cost = n.components[component].periodized_cost.to_series()
    cost = cost.reorder_levels(['scenario', 'name']) if cost.index.nlevels > 1 else cost
    return per_component(component, cost)


def capital_cost(n: pypsa.Network, component: str) -> pd.DataFrame:
    return _periodized(n, component)


# ---------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------


def weighting(n: pypsa.Network, column: str) -> pd.DataFrame:
    return pd.DataFrame({'snapshot': timesteps(n), 'value': n.snapshot_weightings[column].to_numpy()})


def _retention(n: pypsa.Network, component: str) -> pd.DataFrame:
    """``(1 - standing_loss) ** elapsed hours`` per snapshot — the share kept over it."""
    lost = varying(n, component, 'standing_loss')
    hours = dict(zip(timesteps(n), n.snapshot_weightings['stores'].to_numpy(), strict=True))
    lost['value'] = (1.0 - lost['value']) ** lost['snapshot'].map(hours)
    return lost


def _per_period_flag(n: pypsa.Network, component: str, attr: str, multi: bool) -> pd.DataFrame:
    """A per-period storage flag — PyPSA reads it only under ``multi_investment_periods``, so false otherwise."""
    flags = static(n, component, attr, False).fillna({'value': False}).astype({'value': bool})
    if not multi:
        flags['value'] = False
    return flags


def storage(n: pypsa.Network, multi: bool) -> dict[str, object]:
    """What a storage unit and a store take, over and beyond their standing."""
    return {
        'StorageUnit_p_nom': static(n, 'StorageUnit', 'p_nom'),
        'StorageUnit_p_nom_extendable': static(n, 'StorageUnit', 'p_nom_extendable'),
        'StorageUnit_p_min_pu': varying(n, 'StorageUnit', 'p_min_pu'),
        'StorageUnit_p_max_pu': varying(n, 'StorageUnit', 'p_max_pu'),
        'StorageUnit_max_hours': static(n, 'StorageUnit', 'max_hours'),
        'StorageUnit_efficiency_store': varying(n, 'StorageUnit', 'efficiency_store'),
        'StorageUnit_efficiency_dispatch': varying(n, 'StorageUnit', 'efficiency_dispatch'),
        'StorageUnit_sign': per_component('StorageUnit', first_scenario(n.static('StorageUnit')['sign'])),
        'StorageUnit_retention': _retention(n, 'StorageUnit'),
        'StorageUnit_inflow': varying(n, 'StorageUnit', 'inflow'),
        'StorageUnit_state_of_charge_initial': static(n, 'StorageUnit', 'state_of_charge_initial'),
        'StorageUnit_cyclic_state_of_charge': static(n, 'StorageUnit', 'cyclic_state_of_charge'),
        'StorageUnit_cyclic_state_of_charge_per_period': _per_period_flag(
            n, 'StorageUnit', 'cyclic_state_of_charge_per_period', multi
        ),
        'StorageUnit_state_of_charge_initial_per_period': _per_period_flag(
            n, 'StorageUnit', 'state_of_charge_initial_per_period', multi
        ),
        'StorageUnit_marginal_cost': varying(n, 'StorageUnit', 'marginal_cost'),
        'StorageUnit_marginal_cost_quadratic': varying(n, 'StorageUnit', 'marginal_cost_quadratic'),
        'StorageUnit_marginal_cost_storage': varying(n, 'StorageUnit', 'marginal_cost_storage'),
        'StorageUnit_spill_cost': varying(n, 'StorageUnit', 'spill_cost'),
        'StorageUnit_p_set': varying(n, 'StorageUnit', 'p_set').dropna(),
        'StorageUnit_p_dispatch_set': varying(n, 'StorageUnit', 'p_dispatch_set').dropna(),
        'StorageUnit_p_store_set': varying(n, 'StorageUnit', 'p_store_set').dropna(),
        'StorageUnit_state_of_charge_set': varying(n, 'StorageUnit', 'state_of_charge_set').dropna(),
        'StorageUnit_p_nom_min': static(n, 'StorageUnit', 'p_nom_min'),
        'StorageUnit_p_nom_max': static(n, 'StorageUnit', 'p_nom_max'),
        'StorageUnit_capital_cost': capital_cost(n, 'StorageUnit'),
        'StorageUnit_p_nom_set': static(n, 'StorageUnit', 'p_nom_set').dropna(),
        **standing(n, 'StorageUnit'),
        **late_opening(n, 'StorageUnit'),
        'Store_e_nom': static(n, 'Store', 'e_nom'),
        'Store_e_nom_extendable': static(n, 'Store', 'e_nom_extendable'),
        'Store_e_min_pu': varying(n, 'Store', 'e_min_pu'),
        'Store_e_max_pu': varying(n, 'Store', 'e_max_pu'),
        'Store_sign': per_component('Store', first_scenario(n.static('Store')['sign'])),
        'Store_retention': _retention(n, 'Store'),
        'Store_e_initial': static(n, 'Store', 'e_initial'),
        'Store_e_cyclic': static(n, 'Store', 'e_cyclic'),
        'Store_e_cyclic_per_period': _per_period_flag(n, 'Store', 'e_cyclic_per_period', multi),
        'Store_e_initial_per_period': _per_period_flag(n, 'Store', 'e_initial_per_period', multi),
        'Store_marginal_cost': varying(n, 'Store', 'marginal_cost'),
        'Store_marginal_cost_quadratic': varying(n, 'Store', 'marginal_cost_quadratic'),
        'Store_marginal_cost_storage': varying(n, 'Store', 'marginal_cost_storage'),
        'Store_e_set': varying(n, 'Store', 'e_set').dropna(),
        'Store_p_set': varying(n, 'Store', 'p_set').dropna(),
        'Store_e_nom_min': static(n, 'Store', 'e_nom_min'),
        'Store_e_nom_max': static(n, 'Store', 'e_nom_max'),
        'Store_capital_cost': capital_cost(n, 'Store'),
        'Store_e_nom_set': static(n, 'Store', 'e_nom_set').dropna(),
        **standing(n, 'Store'),
        **late_opening(n, 'Store'),
    }


# ---------------------------------------------------------------------------
# passive branches: cycles, losses, outages
# ---------------------------------------------------------------------------


def _cycle_weights(n: pypsa.Network) -> dict[str, object]:
    """The KVL rows PyPSA itself writes, per branch — ``n.cycle_matrix(apply_weights=True)``, times the 1e5 PyPSA scales every cycle row by for conditioning.

    A branch whose weight is NaN — an extendable one at zero nominal rating,
    whose per-unit reactance is infinite — has no term: linopy drops it.

    A transformer's phase shift enters the same rows: a fixed one as a constant
    in radians at each snapshot, a varying one as the shift decision times its
    cycle sign and π/180. PyPSA builds the basis from the first scenario only.
    The constant is written at every snapshot, transformer and cycle, zero
    where nothing shifts, because a constant summed over transformers is owed
    each one.
    """
    n.determine_network_topology()
    n.calculate_dependent_values()
    weighted = n.cycle_matrix(apply_weights=True) * 1e5
    plain = n.cycle_matrix(apply_weights=False)
    rows: dict[str, list[dict]] = {'Line': [], 'Transformer': []}
    for (kind, name), weights in weighted.iterrows():
        rows.setdefault(kind, [])
        rows[kind].extend(
            {DIM[kind]: str(name), 'cycle': str(cycle), 'value': float(w)}
            for cycle, w in weights.items()
            if w and not math.isnan(w)
        )
    transformers = first_scenario(n.static('Transformer'))
    varying_ = transformers['phase_shift_min'] < transformers['phase_shift_max']
    shift_rows, decision_rows = [], []
    signs = plain.loc['Transformer'] if 'Transformer' in plain.index.get_level_values(0) else pd.DataFrame()
    dense = get_switchable_as_dense(n, 'Transformer', 'phase_shift').set_axis(timesteps(n), axis=0)
    if dense.columns.nlevels > 1:
        dense = dense.T.groupby(level='name').first().T
    for name in transformers.index:
        for cycle in plain.columns:
            sign = float(signs.at[name, cycle]) if name in signs.index else 0.0
            if varying_.get(name, False):
                if sign:
                    decision_rows.append(
                        {'transformer': str(name), 'cycle': str(cycle), 'value': sign * math.pi / 180 * 1e5}
                    )
                sign = 0.0
            shift_rows.extend(
                {
                    'snapshot': t,
                    'transformer': str(name),
                    'cycle': str(cycle),
                    'value': sign * float(shift) * math.pi / 180 * 1e5,
                }
                for t, shift in dense[name].items()
            )
    line = pd.DataFrame(rows['Line'], columns=['line', 'cycle', 'value']).astype({'value': float})
    transformer = pd.DataFrame(rows['Transformer'], columns=['transformer', 'cycle', 'value']).astype({'value': float})
    cycles = list(pd.unique(pd.concat([line['cycle'], transformer['cycle']])))
    return {
        'cycle': pl.Series('cycle', cycles, dtype=pl.String),
        'Line_cycle_weight': line,
        'Transformer_cycle_weight': transformer,
        'Transformer_phase_shift_weight': pd.DataFrame(
            shift_rows, columns=['snapshot', 'transformer', 'cycle', 'value']
        ).astype({'value': float}),
        'Transformer_phase_shift_cycle_weight': pd.DataFrame(
            decision_rows, columns=['transformer', 'cycle', 'value']
        ).astype({'value': float}),
        'Transformer_phase_shift_varying': per_component('Transformer', varying_, bool),
        'Transformer_phase_shift_min': per_component('Transformer', transformers['phase_shift_min']),
        'Transformer_phase_shift_max': per_component('Transformer', transformers['phase_shift_max']),
    }


def _loss_cuts(n: pypsa.Network, component: str, losses: dict) -> dict[str, object]:
    """The cuts PyPSA holds a branch's loss above — tangents, or secants placed by its tolerance loop.

    Per branch and snapshot: the loss at rating, ``r_pu_eff * (s_max_pu * s_nom_max)**2``.
    A tangent ``k`` of ``segments`` touches the curve at ``p_k = k/segments`` of the
    rating, slope ``2 * r_pu_eff * p_k`` and offset ``loss_k - slope_k * p_k``. A
    secant joins consecutive breakpoints ``a, b`` of PyPSA's
    `define_secant_loss_constraints`, slope ``2 * sqrt(atol * r) * (a + b)`` and
    offset ``-4 * atol * a * b`` in its factors; a branch without resistance
    takes every cut at zero.
    """
    dim = DIM[component]
    table = n.static(component)
    empty = pd.DataFrame({'snapshot': [], dim: [], 'value': []})
    if table.empty or not losses:
        return {'max': empty, 'slope': empty.assign(segment=[]), 'offset': empty.assign(segment=[]), 'segments': []}
    n.calculate_dependent_values()
    top = get_switchable_as_dense(n, component, 's_max_pu').set_axis(timesteps(n), axis=0)
    rating = table['s_nom_max'].where(table['s_nom_extendable'], table['s_nom'])
    if top.columns.nlevels > 1:
        rating.index = top.columns
    top = top * rating
    r = table['r_pu_eff'].copy()
    if top.columns.nlevels > 1:
        r.index = top.columns

    def melt(dense: pd.DataFrame) -> pd.DataFrame:
        frame = dense.copy()
        frame.columns.names = ['scenario', dim] if frame.columns.nlevels > 1 else [dim]
        return frame.melt(ignore_index=False).reset_index(names='snapshot').astype({dim: str, 'value': float})

    slopes, offsets = [], []
    if losses.get('mode') == 'secants':
        atol, rtol = float(losses.get('atol', 1)), float(losses.get('rtol', 0.1))
        max_segments = int(losses.get('max_segments', 20))
        lossy = r > 0
        p_1 = np.where(lossy, 2 * np.sqrt(atol / r.where(lossy, 1.0)), 0)
        target = top.where(lossy, 0).max() / pd.Series(np.where(lossy, p_1, 1.0), index=r.index)
        factors = [0.0, 1.0]
        while (factors[-1] < target.where(lossy, 0)).any():
            k = len(factors)
            factors.append(factors[-1] * max(k / (k - 1), 1 + 2 * (rtol + math.sqrt(rtol + rtol**2))))
            if k >= max_segments:
                break
        for k, (a, b) in enumerate(zip(factors[:-1], factors[1:], strict=True)):
            slope = (2 * np.sqrt(atol * r) * (a + b)).where(lossy, 0.0)
            offset = pd.Series(-4 * atol * a * b, index=r.index).where(lossy, 0.0)
            slopes.append(melt(top * 0 + slope).assign(segment=k))
            offsets.append(melt(top * 0 + offset).assign(segment=k))
        segments = list(range(len(factors) - 1))
    else:
        count = int(losses.get('segments', 0))
        for k in range(1, count + 1):
            p_k = k / count * top
            slopes.append(melt(2 * r * p_k).assign(segment=k))
            offsets.append(melt(r * p_k**2 - 2 * r * p_k * p_k).assign(segment=k))
        segments = list(range(1, count + 1))
    return {
        'max': melt(r * top**2),
        'slope': pd.concat(slopes, ignore_index=True) if slopes else empty.assign(segment=[]),
        'offset': pd.concat(offsets, ignore_index=True) if offsets else empty.assign(segment=[]),
        'segments': segments,
    }


def _loss_options(optimize: dict, outages: object) -> dict:
    """PyPSA's ``transmission_losses`` as a mapping — none in a security-constrained run, which PyPSA builds without it."""
    losses = optimize.get('transmission_losses', False)
    if outages is not None or not losses:
        return {}
    if losses is True:
        return {'mode': 'secants'}
    if isinstance(losses, int):
        return {'mode': 'tangents', 'segments': losses}
    return dict(losses)


def losses(n: pypsa.Network, optimize: dict, outages: object) -> dict[str, object]:
    options = _loss_options(optimize, outages)
    tables: dict[str, object] = {'transmission_losses': bool(options)}
    segments: list[int] = []
    for component in ('Line', 'Transformer'):
        cuts = _loss_cuts(n, component, options)
        tables |= {
            f'{component}_loss_max': cuts['max'],
            f'{component}_loss_slope': cuts['slope'],
            f'{component}_loss_offset': cuts['offset'],
        }
        segments = sorted(set(segments) | set(cuts['segments']))
    tables['segment'] = pl.Series('segment', segments, dtype=pl.Int64)
    return tables


def _outage_list(n: pypsa.Network, outages: object) -> list[tuple[str, str]]:
    """``(component, name)`` per branch a security-constrained run takes out — a plain list names lines."""
    if outages is None:
        return []
    if isinstance(outages, pd.MultiIndex):
        return [(str(c), str(name)) for c, name in outages]
    return [('Line', str(name)) for name in outages]


def outage_label(component: str, name: str) -> str:
    return f'{component}-{name}'


def security(n: pypsa.Network, outages: object) -> dict[str, object]:
    """The outages and each branch's share of an outaged branch's flow — PyPSA's BODF per sub-network."""
    taken = _outage_list(n, outages)
    rows: dict[str, list[dict]] = {'Line': [], 'Transformer': []}
    if taken:
        n.determine_network_topology()
        n.calculate_dependent_values()
        for sub_network in n.c.sub_networks.static.obj:
            branches = sub_network.branches_i()
            inside = [(c, name) for c, name in taken if (c, name) in branches]
            if not inside:
                continue
            sub_network.calculate_BODF()
            bodf = pd.DataFrame(sub_network.BODF, index=branches, columns=branches)
            for c_out, out in inside:
                for (c_aff, affected), share in bodf[(c_out, out)].items():
                    rows[c_aff].append(
                        {DIM[c_aff]: str(affected), 'outage': outage_label(c_out, out), 'value': float(share)}
                    )
    labels = [outage_label(c, name) for c, name in taken]
    return {
        'outage': pl.Series('outage', labels, dtype=pl.String),
        'Outage_line': pd.DataFrame(
            {
                'outage': [outage_label(c, x) for c, x in taken if c == 'Line'],
                'line': [x for c, x in taken if c == 'Line'],
            }
        ),
        'Outage_transformer': pd.DataFrame(
            {
                'outage': [outage_label(c, x) for c, x in taken if c == 'Transformer'],
                'transformer': [x for c, x in taken if c == 'Transformer'],
            }
        ),
        'Line_BODF': pd.DataFrame(rows['Line'], columns=['line', 'outage', 'value']).astype({'value': float}),
        'Transformer_BODF': pd.DataFrame(rows['Transformer'], columns=['transformer', 'outage', 'value']).astype(
            {'value': float}
        ),
    }


def branches(n: pypsa.Network) -> dict[str, object]:
    out: dict[str, object] = {}
    for component in ('Line', 'Transformer'):
        p = component
        out |= {
            f'{p}_s_nom': static(n, component, 's_nom'),
            f'{p}_s_nom_extendable': static(n, component, 's_nom_extendable'),
            f'{p}_s_max_pu': varying(n, component, 's_max_pu'),
            f'{p}_s_nom_min': static(n, component, 's_nom_min'),
            f'{p}_s_nom_max': static(n, component, 's_nom_max'),
            f'{p}_capital_cost': capital_cost(n, component),
            f'{p}_s_nom_set': static(n, component, 's_nom_set').dropna(),
            f'{p}_s_set': varying(n, component, 's_set').dropna(),
            f'{p}_bus0': relation(n, component, 'bus0'),
            f'{p}_bus1': relation(n, component, 'bus1'),
            **standing(n, component),
        }
    return out


# ---------------------------------------------------------------------------
# scenarios, periods, carriers
# ---------------------------------------------------------------------------


def scenarios(n: pypsa.Network) -> dict[str, object]:
    """The scenario dimension, its weights and the risk preference's two scalars — one scenario of weight one on a plain run, and no tail."""
    labels = scenario_labels(n)
    weights = n.scenario_weightings['weight'].to_numpy(dtype=float) if n.has_scenarios else [1.0]
    risk = n.risk_preference or {}
    return {
        'scenario': pl.Series('scenario', labels, dtype=pl.String),
        'scenario_weight': pd.DataFrame({'scenario': labels, 'value': weights}),
        'CVaR_omega': float(risk.get('omega', 0.0)),
        'CVaR_alpha': float(risk.get('alpha', 0.0)),
    }


def periods(n: pypsa.Network) -> dict[str, object]:
    """The investment periods, their weights and the period each snapshot falls in — one period of weight one on a plain run."""
    labels = period_labels(n)
    multi = n.snapshots.nlevels > 1
    return {
        'period': pl.Series('period', labels, dtype=pl.Int64),
        'period_weight_objective': pd.DataFrame({'period': labels, 'value': period_weights(n, 'objective').to_numpy()}),
        'period_weight_years': pd.DataFrame(
            {'period': labels, 'value': period_weights(n, 'years').to_numpy() if multi else [1.0]}
        ),
        'snapshot_period': pd.DataFrame({'snapshot': timesteps(n), 'period': period_of(n)}),
    }


def carriers(n: pypsa.Network, multi: bool) -> dict[str, object]:
    """The carriers, which one a component converts from, and the growth limits.

    An infinite `max_growth` is no limit and no row; with scenarios, the
    strictest limit, as PyPSA takes it. PyPSA reads a growth limit only under
    ``multi_investment_periods``, so there is none otherwise. A carrier a
    component names and the network does not declare is a label too.
    """
    table = n.carriers[['max_growth', 'max_relative_growth']]
    if table.index.nlevels > 1:
        table = table.groupby(level='name').min()
    relations = {
        f'{component}_carrier': relation(n, component, 'carrier', into='carrier')
        for component in ('Generator', 'Link', 'Process', 'StorageUnit', 'Store', 'Line')
    }
    named = set().union(*(set(frame['carrier']) for frame in relations.values()))
    labels = list(table.index.astype(str)) + sorted(named - set(table.index.astype(str)))
    growth = table['max_growth']
    bounded = growth[(growth < float('inf')) & multi] if multi else growth.iloc[:0]
    return {
        'carrier': pl.Series('carrier', labels, dtype=pl.String),
        **relations,
        'Carrier_max_growth': pd.DataFrame(
            {'carrier': bounded.index.astype(str), 'value': bounded.to_numpy(dtype=float)}
        ),
        'Carrier_max_relative_growth': pd.DataFrame(
            {'carrier': table.index.astype(str), 'value': table['max_relative_growth'].to_numpy(dtype=float)}
        ),
    }


# ---------------------------------------------------------------------------
# global constraints
# ---------------------------------------------------------------------------


def _gc_rows(n: pypsa.Network) -> list[tuple[str | None, str, pd.Series]]:
    """``(scenario, label, row)`` per global constraint, one per scenario where the network has them."""
    table = n.global_constraints
    if table.index.nlevels > 1:
        return [(str(s), str(name), row) for (s, name), row in table.iterrows()]
    return [(None, str(name), row) for name, row in table.iterrows()]


def _named_period(n: pypsa.Network, gc: pd.Series, multi: bool) -> object:
    """The period a row names: ``None`` for every period, ``False`` for one the run does not model."""
    period = gc.get('investment_period', np.nan)
    if period is None or (isinstance(period, float) and np.isnan(period)):
        return None
    if not multi or period not in list(n.investment_periods):
        return False
    return int(period)


def _members(n: pypsa.Network, component: str, scenario: str | None) -> pd.DataFrame:
    table = n.static(component)
    if scenario is not None and table.index.nlevels > 1 and scenario in table.index.get_level_values('scenario'):
        return table.xs(scenario, level='scenario')
    return first_scenario(table)


def _stands_in(n: pypsa.Network, component: str, period: object, multi: bool) -> pd.Series:
    """Whether each asset stands in *period* — every period where it names none, the ``active`` flag on a plain run."""
    c = n.components[component]
    if not multi:
        return first_scenario(c.get_active_assets()).astype(bool)
    chosen = list(n.investment_periods) if period is None else [period]
    return first_scenario(c.get_active_assets(chosen)).astype(bool)


def _emissions(n: pypsa.Network, gc: pd.Series, scenario: str | None) -> pd.Series:
    """The nonzero values of the carrier attribute a `primary_energy` row weighs."""
    carriers_ = n.carriers
    if carriers_.index.nlevels > 1:
        carriers_ = carriers_.xs(scenario, level='scenario') if scenario is not None else first_scenario(carriers_)
    values = carriers_[gc['carrier_attribute']]
    return values[values != 0]


def _carrier_list(gc: pd.Series) -> list[str]:
    return [c.strip().strip('[]()') for c in str(gc['carrier_attribute']).split(',')]


def global_constraints(n: pypsa.Network, multi: bool) -> dict[str, object]:
    """The rows' type, sense and constant, which snapshots each counts, and every component's weight in each.

    A `primary_energy` or `operational_limit` row counts what its non-cyclic
    storage draws down, so PyPSA adds the initial charge as a constant on the
    variable side; the file keeps the variables and moves it here — once, or
    times its period's years for each counted period where the storage reopens
    per period.
    """
    stamps = list(timesteps(n))
    periods_ = period_of(n)
    years = dict(zip(period_labels(n), period_weights(n, 'years').to_numpy() if multi else [1.0], strict=True))
    objective_weight = dict(zip(period_labels(n), period_weights(n, 'objective').to_numpy(), strict=True))
    gen_eff = get_switchable_as_dense(n, 'Generator', 'efficiency').set_axis(timesteps(n), axis=0)
    weights: dict[str, list[dict]] = {
        key: []
        for key in (
            'Generator_primary_energy_weight',
            'StorageUnit_primary_energy_weight',
            'Store_primary_energy_weight',
            'Generator_operational_limit_weight',
            'StorageUnit_operational_limit_weight',
            'Store_operational_limit_weight',
            'Line_volume_weight',
            'Link_volume_weight',
            'Line_expansion_cost_weight',
            'Link_expansion_cost_weight',
            'Generator_tech_capacity_weight',
            'Link_tech_capacity_weight',
            'Line_tech_capacity_weight',
            'StorageUnit_tech_capacity_weight',
            'Store_tech_capacity_weight',
            'Process_tech_capacity_weight',
        )
    }
    counts, constants, senses, types = [], [], [], []
    for scenario, label, gc in _gc_rows(n):
        key = {'scenario': scenario} if scenario is not None else {}
        period = _named_period(n, gc, multi)
        types.append({'global_constraint': label, 'value': str(gc['type'])})
        senses.append(key | {'global_constraint': label, 'value': str(gc['sense'])})
        constant = float(gc['constant'])
        if period is not False:
            counts.extend(
                key | {'global_constraint': label, 'snapshot': t, 'value': bool(period is None or p == period)}
                for t, p in zip(stamps, periods_, strict=True)
            )
        counted = [p for p in period_labels(n) if period is None or p == period]
        kind = gc['type']
        if kind in ('primary_energy', 'operational_limit'):
            if kind == 'primary_energy':
                emissions = _emissions(n, gc, scenario)

                def rate(carrier: str) -> float:
                    return float(emissions.get(carrier, 0.0))
            else:
                target = gc['carrier_attribute']

                def rate(carrier: str) -> float:
                    return float(carrier == target)

            for component, cyclic_attr, initial_attr, per_period_attr in (
                (
                    'StorageUnit',
                    'cyclic_state_of_charge',
                    'state_of_charge_initial',
                    'state_of_charge_initial_per_period',
                ),
                ('Store', 'e_cyclic', 'e_initial', 'e_initial_per_period'),
            ):
                members = _members(n, component, scenario)
                stands = _stands_in(n, component, period if kind == 'primary_energy' else None, multi)
                for name, unit in members.iterrows():
                    weight = rate(unit['carrier'])
                    if not weight or unit[cyclic_attr] or not stands.get(name, False):
                        continue
                    if kind == 'operational_limit' and not unit.get('active', True):
                        continue
                    weights[
                        f'{component}_{"primary_energy" if kind == "primary_energy" else "operational_limit"}_weight'
                    ].append(key | {'global_constraint': label, DIM[component]: str(name), 'value': weight})
                    reopens = multi and bool(unit.get(per_period_attr, False))
                    times = sum(years[p] for p in counted) if reopens else 1.0
                    constant -= weight * float(unit[initial_attr]) * times
            generators = _members(n, 'Generator', scenario)
            stands = _stands_in(n, 'Generator', period if kind == 'primary_energy' else None, multi)
            efficiency = (
                gen_eff.xs(scenario, axis=1, level='scenario')
                if gen_eff.columns.nlevels > 1 and scenario
                else (gen_eff.T.groupby(level='name').first().T if gen_eff.columns.nlevels > 1 else gen_eff)
            )
            for name, g in generators.iterrows():
                weight = rate(g['carrier'])
                if not weight or not stands.get(name, False):
                    continue
                if kind == 'operational_limit' and not g.get('active', True):
                    continue
                if kind == 'primary_energy':
                    weights['Generator_primary_energy_weight'].extend(
                        key
                        | {
                            'global_constraint': label,
                            'snapshot': t,
                            'generator': str(name),
                            'value': weight / float(e),
                        }
                        for t, e in efficiency[name].items()
                    )
                else:
                    weights['Generator_operational_limit_weight'].append(
                        key | {'global_constraint': label, 'generator': str(name), 'value': weight}
                    )
        elif kind in ('transmission_volume_expansion_limit', 'transmission_expansion_cost_limit'):
            carriers_ = _carrier_list(gc)
            for component in ('Line', 'Link'):
                members = _members(n, component, scenario)
                stands = _stands_in(n, component, period, multi)
                first_members = first_scenario(n.static(component))
                cost = n.components[component].capital_cost
                cost = first_scenario(cost) if isinstance(cost, pd.Series) else cost
                by_period = active_by_period(n, component)
                for name, unit in members.iterrows():
                    if not unit[f'{NOMINAL[component]}_extendable'] or unit['carrier'] not in carriers_:
                        continue
                    if not stands.get(name, False):
                        continue
                    if kind == 'transmission_volume_expansion_limit':
                        value = float(first_members.loc[name, 'length'])
                        table = f'{component}_volume_weight'
                    else:
                        share = 1.0
                        if multi and period is None:
                            share = float(sum(objective_weight[p] for p in by_period.columns if by_period.loc[name, p]))
                        value = float(cost.loc[name]) * share
                        table = f'{component}_expansion_cost_weight'
                    if value:
                        weights[table].append(
                            key | {'global_constraint': label, DIM[component]: str(name), 'value': value}
                        )
        elif kind == 'tech_capacity_expansion_limit':
            for component in NOMINAL:
                members = _members(n, component, scenario)
                if 'carrier' not in members.columns:
                    continue
                stands = _stands_in(n, component, period, multi)
                bus = 'bus0' if component in ('Line', 'Link', 'Transformer', 'Process') else 'bus'
                for name, unit in members.iterrows():
                    if not unit[f'{NOMINAL[component]}_extendable'] or unit['carrier'] != gc['carrier_attribute']:
                        continue
                    if gc.get('bus') and str(unit[bus]) != str(gc['bus']):
                        continue
                    if not stands.get(name, False):
                        continue
                    weights[f'{component}_tech_capacity_weight'].append(
                        {'global_constraint': label, DIM[component]: str(name), 'value': 1.0}
                    )
        constants.append(key | {'global_constraint': label, 'value': constant})
    keys = ['scenario'] if n.global_constraints.index.nlevels > 1 else []
    tables: dict[str, object] = {
        'GlobalConstraint_type': pd.DataFrame(types, columns=['global_constraint', 'value']).drop_duplicates(),
        'GlobalConstraint_sense': pd.DataFrame(senses, columns=[*keys, 'global_constraint', 'value']),
        'GlobalConstraint_constant': pd.DataFrame(constants, columns=[*keys, 'global_constraint', 'value']).astype(
            {'value': float}
        ),
        'GlobalConstraint_counts_snapshot': pd.DataFrame(
            counts, columns=[*keys, 'global_constraint', 'snapshot', 'value']
        ).astype({'value': bool}),
    }
    for name, rows in weights.items():
        frame = pd.DataFrame(rows)
        if frame.empty:
            dims = {
                'Generator_primary_energy_weight': [*keys, 'global_constraint', 'snapshot', 'generator'],
            }.get(name)
            component = name.split('_')[0]
            frame = pd.DataFrame(columns=dims or [*keys, 'global_constraint', DIM[component], 'value'])
            if 'value' not in frame.columns:
                frame['value'] = []
        tables[name] = frame.astype({'value': float})
    return tables


# ---------------------------------------------------------------------------
# the whole network
# ---------------------------------------------------------------------------


def sources(n: pypsa.Network, optimize: dict | None = None, outages: object = None) -> dict[str, object]:
    """Every table the example specs declare, from one PyPSA network.

    *optimize* is the rung's ``OPTIMIZE``: ``multi_investment_periods`` and
    ``transmission_losses`` decide tables PyPSA reads only under them.
    *outages* is the rung's ``BRANCH_OUTAGES``, ``None`` on a plain run.
    """
    optimize = optimize or {}
    multi = bool(optimize.get('multi_investment_periods')) and n.snapshots.nlevels > 1
    loads = n.loads
    tables: dict[str, object] = {
        'snapshot': pl.Series('snapshot', list(timesteps(n)), dtype=pl.Datetime('us')),
        'bus': pl.Series('bus', list(names(n.buses.index).astype(str)), dtype=pl.String),
        **{
            dim: pl.Series(dim, list(names(n.static(component).index).astype(str)), dtype=pl.String)
            for component, dim in DIM.items()
        },
        **scenarios(n),
        **periods(n),
        **carriers(n, multi),
        'Generator_bus': relation(n, 'Generator', 'bus'),
        'Generator_sign': per_component('Generator', first_scenario(n.generators['sign'])),
        'Link_bus0': relation(n, 'Link', 'bus0'),
        'Load_bus': relation(n, 'Load', 'bus'),
        'Load_sign': per_component('Load', first_scenario(loads['sign'])),
        'Load_active': per_component('Load', first_scenario(loads['active']), bool),
        'Load_p_set': varying(n, 'Load', 'p_set'),
        'StorageUnit_bus': relation(n, 'StorageUnit', 'bus'),
        'Store_bus': relation(n, 'Store', 'bus'),
        'snapshot_weightings_objective': weighting(n, 'objective'),
        'snapshot_weightings_stores': weighting(n, 'stores'),
        'snapshot_weightings_generators': weighting(n, 'generators'),
        'Generator_e_sum_min': static(n, 'Generator', 'e_sum_min'),
        'Generator_e_sum_max': static(n, 'Generator', 'e_sum_max'),
        **{name: table for component in DISPATCHABLE for name, table in dispatchable(n, component).items()},
        **ports(n, 'Link'),
        **ports(n, 'Process'),
        **storage(n, multi),
        **branches(n),
        **_cycle_weights(n),
        **losses(n, optimize, outages),
        **security(n, outages),
        **global_constraints(n, multi),
    }

    for name, table in tables.items():
        if isinstance(table, pd.DataFrame):
            tables[name] = pl.from_pandas(
                table.astype({column: _key_dtype(column) for column in table.columns if column != 'value'})
            )
    return tables


def _key_dtype(column: str) -> str:
    """The pandas dtype a key column takes whatever its rows — an empty table has none to infer it from."""
    if column in ('snapshot', 'start', 'covered'):
        return 'datetime64[us]'
    if column in ('segment', 'period'):
        return 'int64'
    return 'string'
