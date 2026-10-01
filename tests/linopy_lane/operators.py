"""Eager evaluation of the language's built-in operators, on xarray and linopy.

Each entry point takes an operand that is already a value — an ``xr.DataArray``
for a parameter, a linopy ``Variable`` or ``LinearExpression`` for anything
carrying one — and returns the same. Nothing here reads the schema, the plan or
the model: ``builder.py`` evaluates the operands and the keywords, and calls in.
Each entry point comes first and its own machinery after.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

from tests.linopy_lane import absence

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping

    import pandas as pd


def operator_sum(array: Any, over: str) -> Any:
    """Sum *array* over dimension *over*, which lowering has checked it carries.

    The one sum linopy cannot take — a term carrying a dimension the data left
    empty — is built directly (:func:`_empty_sum`).
    """
    if not isinstance(array, xr.DataArray) and any(not array.sizes[dim] for dim in array.coord_dims if dim != over):
        return _empty_sum(array, over)
    return array.sum(over)


def _empty_sum(array: Any, over: str) -> Any:
    """*over* reduced while a dimension beside it is empty — the sum linopy refuses to take.

    linopy's own ``sum`` stacks the summed dim into its term axis, and once any
    remaining dimension is zero-sized the reshape cannot infer the term count
    (``cannot reshape array of size 0``, PyPSA/linopy#906). Every coordinate of
    the result is empty, so it is built as the constant zero over the
    coordinates linopy's sum would keep.
    """
    from linopy.expressions import LinearExpression

    kept = [dim for dim in array.coord_dims if dim != over]
    zeros = xr.DataArray(
        np.zeros([array.sizes[dim] for dim in kept]),
        coords={dim: array.indexes[dim] for dim in kept},
        dims=kept,
    )
    return LinearExpression.from_constant(array.model, zeros)


def operator_grouped_sum(
    array: Any,
    mappings: tuple[Any, ...],
    *,
    into: tuple[str, ...],
    joined: tuple[str, ...] = (),
    labels: Mapping[str, pd.Index],
) -> Any:
    """Sum *array* through a declared relation, producing dimensions *into*.

    YAML: ``sum(p, by=gen_bus, over=generator, into=bus)``. *mappings* are the
    walked columns' values as arrays over the dimensions the relation's key
    names — one dimension for a plain map, and one more per condition where the
    key is several columns. The walked dimension is summed out and *into* holds
    the group labels, one dim per column walked to, so a walk to two columns is
    one grouping onto their pair.

    A null value says the key belongs to no group, so its terms contribute
    nowhere. linopy refuses to group by NaN at all, so the operand is masked
    to absent at those keys and a declared label stands in for them — which is
    what lets a key of several columns leave one combination out without
    dropping a whole label of any one dimension. A key missing *any* walked
    column belongs to no group.

    *joined* names the dimensions the key holds beside the one walked: they
    are group keys too, so a group is a (value, condition) pair rather than a
    value summed across conditions — which is what the whole two-dimensional
    label array would otherwise collapse into.

    *labels* holds each group dimension's declared index, and the result is
    reindexed onto them: a groupby yields only the labels some member actually
    points at, in xarray's sort order, and linopy v1 aligns on membership *and*
    order. A map's values are validated against their target's labels when they
    are loaded, so this only ever adds a label, never drops a term.
    """
    if any(not len(labels[target]) for target in into):
        return _into_nothing(array, mappings, into=into, joined=joined, labels=labels)
    mappings = _renamed(mappings, into)
    present = _present(mappings)
    if not bool(present.all()):
        array = array.where(present)
        mappings = tuple(m.fillna(labels[t][0]) for m, t in zip(mappings, into, strict=True))
    attached = array.assign_coords({t: (m.dims, m.to_numpy()) for t, m in zip(into, mappings, strict=True)})
    groups = (*into, *joined)
    summed = attached.groupby(list(groups)).sum()
    return _reindexed(summed, into=groups, labels=labels)


def _into_nothing(
    array: Any, mappings: tuple[Any, ...], *, into: tuple[str, ...], joined: tuple[str, ...], labels: Mapping[str, pd.Index]
) -> Any:
    """A grouped sum onto a dimension with no labels: zero over the dims kept, and none along the empty one.

    The fill a null group takes needs a label to stand in, and an empty
    dimension has none to offer.
    """
    from linopy.expressions import LinearExpression

    walked = {dim for mapping in mappings for dim in mapping.dims} - set(joined)
    dims = array.dims if isinstance(array, xr.DataArray) else array.coord_dims
    kept = [dim for dim in dims if dim not in walked]
    zeros = xr.DataArray(
        np.zeros([array.sizes[dim] for dim in kept] + [0] * len(into)),
        coords={**{dim: array.indexes[dim] for dim in kept}, **{target: labels[target] for target in into}},
        dims=[*kept, *into],
    )
    return zeros if isinstance(array, xr.DataArray) else LinearExpression.from_constant(array.model, zeros)


def operator_at(array: Any, mappings: tuple[Any, ...], *, into: tuple[str, ...]) -> Any:
    """Read *array* through a declared relation — the adjoint of a group.

    YAML: ``at(on, by=component, over=bus, into=line)``. *mappings* are the same
    arrays ``sum`` takes; grouping sums *along* them, this indexes *through*
    them, so the operand must carry every dim in ``into`` and the result carries
    the mappings' own dims. xarray's vectorised selection is the pullback exactly
    — one ``into`` label read once per fine key pointing at it, pointwise
    along a condition the operand already carries.

    A null value reads nothing and its row is absent, the same reading
    ``sum`` gives a null group. It cannot be selected, so a declared label
    stands in for it and the result is masked back out at exactly those
    coordinates, which leaves them holding the operand's own **absence**
    rather than a zero: absence propagates and takes the row with it, where a
    zero would leave a row asserting ``x <= 0`` at a coordinate the model said
    nothing about.

    **The coordinates come back as the fine labels**, which a selection
    through a map onto the fine dimension's own dimension would otherwise
    leave as the labels it read: a self-map reads ``snapshot`` at
    ``snapshot``, and the row the value lands at is the one that read it, not
    the one it read.
    """
    mappings = _renamed(mappings, into)
    present = _present(mappings)
    fine = {str(d): mappings[0].coords[d] for d in mappings[0].dims}
    if bool(present.all()):
        return array.sel(dict(zip(into, mappings, strict=True))).assign_coords(fine)

    stood_in = tuple(m.fillna(array.coords[t].to_numpy()[0]) for m, t in zip(mappings, into, strict=True))
    picked = array.sel(dict(zip(into, stood_in, strict=True))).assign_coords(fine)
    return picked.where(present)


@dataclass(frozen=True)
class _Edge:
    """One ``edge=`` policy: cyclic, a fill, or absent (``fill=None``, no wrap)."""

    wrap: bool
    fill: float | None


def operator_shift(array: Any, *, over: str, offset: Any, wrap: bool, fill: float | None, by: Any = None) -> Any:
    """Translate *array* along one dimension — the value at *t - offset*.

    YAML: ``shift(soc, along=snapshot, offset=1)``. *wrap* is cyclic and vacates
    nothing; *fill* is what the vacated positions contribute; neither leaves
    them **absent**, which propagates and drops the row — what linopy v1's own
    ``shift`` already answers.

    A DataArray shift always fills, absence not being representable in data, so
    lowering refuses a bare shift over a variable-free operand and that branch
    is only reached under a numeric fill.

    *offset* arrives as an array where the model named a parameter — an offset
    that differs per entity, which is a gather rather than a shift
    (:func:`_gather_by_offset`).
    """
    edge = _Edge(wrap, fill)
    if by is not None:
        groups = _grouped(over, np.asarray(array.indexes[over]), by)
        return _gather_in_groups(array, over, _per_group(offset, by), groups=groups, edge=edge)
    if isinstance(offset, xr.DataArray) and offset.ndim:
        return _gather_by_offset(array, over, offset, edge=edge)
    amount: dict[Hashable, int] = {over: int(offset)}
    if wrap:
        return array.roll(amount, roll_coords=False) if isinstance(array, xr.DataArray) else array.roll(amount)
    if isinstance(array, xr.DataArray):
        return array.shift(amount, fill_value=fill if fill is not None else np.nan)
    shifted = array.shift(amount)
    return shifted if fill is None else absence.vacated(shifted, array, over, _off_the_axis(array, over, offset), fill)


def operator_sum_back(array: Any, *, over: str, within: Any, wrap: bool, by: Any = None) -> Any:
    """Sum *array* over a trailing window along one dimension.

    YAML: ``sum_back(started, along=snapshot, window=min_up)``. The result at
    *t* is the sum from *t - within + 1* through *t*, so a width of 1 is the
    operand itself and *wrap* lets the window reach around the axis.

    Written as a sum of scalar gathers, one per position of the widest window
    the data asks for — a bound read from data, sound because it decides how
    many *terms* are added rather than what the plan does.

    A position the window cannot reach contributes a **zero**, never an
    absence: absence propagates, so an unreachable lag would annihilate the
    whole row, and a window at the first position is short, not empty. The
    window that reaches **nothing** is the exception — a zero there would
    build a row about constants alone — so the fill is paired with the
    positions any lag actually reached, and a window that reached none keeps
    no row. A width of zero everywhere is that window at every position.

    ``by=`` stops the window at each group's edge. A width declared over the
    group's own dim is read through the relation first (:func:`_per_group`).
    """
    within = _per_group(within, by) if by is not None else within
    asked = _widest(within)
    widest = max(1, min(asked, int(array.sizes[over])))
    probe = _Edge(wrap=wrap, fill=None)
    groups = _grouped(over, np.asarray(array.indexes[over]), by) if by is not None else None
    terms: list[Any] = []
    reached: list[Any] = []
    for lag in range(widest):
        lagged = (
            _gather_in_groups(array, over, lag, groups=groups, edge=probe)
            if groups is not None
            else _gather_by_offset(array, over, lag, edge=probe)
        )
        live, term = ~lagged.isnull(), absence.filled(lagged, 0.0)
        if isinstance(within, xr.DataArray):
            live, term = live & (within > lag), term * (within > lag).astype(float)
        terms.append(term)
        reached.append(live)
    return _merged(terms).where(reduce(operator.or_, reached))


def _widest(within: Any) -> int:
    """The widest window the data asks for, which bounds how many lags are gathered.

    A coordinate the relation maps nowhere asks for no window: a width read
    through one carries the operand's own absence there (:func:`_per_group`),
    so the maximum skips the holes rather than coming back absent itself, and
    a width no coordinate carries at all is a window of nothing. The lag loop
    takes it from there — ``within > lag`` is false at a hole, so the
    coordinate contributes no term and keeps no row.
    """
    if not isinstance(within, xr.DataArray):
        return int(within)
    widths = np.asarray(within, dtype=float)
    return 0 if bool(np.isnan(widths).all()) else int(np.nanmax(widths))


def _merged(terms: list[Any]) -> Any:
    """The sum of *terms*, which all share one set of coordinates."""
    if isinstance(terms[0], xr.DataArray):
        return reduce(operator.add, terms)
    from linopy import merge

    return merge(terms)


def _renamed(mappings: tuple[Any, ...], into: tuple[str, ...]) -> tuple[Any, ...]:
    """*mappings* renamed to the dims they target, so the group's own name is the dim that comes out."""
    return tuple(mapping.rename(target) for mapping, target in zip(mappings, into, strict=True))


def _present(mappings: tuple[Any, ...]) -> Any:
    """The members every mapping has a value for, as a boolean over their dim."""
    keep = mappings[0].notnull()
    for mapping in mappings[1:]:
        keep = keep & mapping.notnull()
    return keep


def _reindexed(summed: Any, *, into: tuple[str, ...], labels: Mapping[str, pd.Index]) -> Any:
    """*summed* over exactly the declared labels, empty groups filled with an empty sum.

    Reindexing onto a label no member reached creates an **absent** slot, and
    the unstacked multi-key groupby invents the combinations no member lands on
    the same way. Neither is an absence in this language: a group with no
    members holds the empty sum, which is 0. ``fillna`` reaches the constant of
    a ``LinearExpression`` and the value of the ``DataArray`` a grouped
    parameter is, so one call serves both, and it comes after the reindex
    because the invented combinations sit at *present* labels.
    """
    return summed.reindex({d: labels[d] for d in into}).fillna(0.0)


def _gather_by_offset(array: Any, over: str, offset: Any, *, edge: _Edge) -> Any:
    """Translate *array* along *over* by an offset that differs per entity.

    A scalar shift is one call; a per-entity one is a **gather**: every output
    position reads a source position of its own, so the index is an array over
    the offset's dims and *over* rather than a number.

    Selection is by *label* rather than by ordinal, because that is what linopy
    passes through to its own labels — which also keeps a non-integer axis (a
    datetime snapshot) working for free.

    Out-of-range positions are clipped so the gather stays on the axis, then
    emptied again by ``where``, so an edge means the same thing it does for a
    scalar shift: absent by default, and :func:`~tests.linopy_lane.absence.vacated`
    fills it where the model asked. Under ``wrap`` nothing is out of range and
    the modulo is the whole of it.
    """
    card = int(array.sizes[over])
    labels = np.asarray(array.indexes[over])
    ordinal = xr.DataArray(np.arange(card), coords={over: labels}, dims=[over])
    source = (ordinal - offset).astype(int)

    def gathered(ordinals: Any) -> Any:
        """Pick each position's source, then put the original labels back — the indexer carries no coordinate."""
        picked = array.sel({over: _labelled(labels, ordinals, over)})
        return picked.assign_coords({over: labels})

    if edge.wrap:
        return gathered(source % card)
    inside = ((source >= 0) & (source < card)).assign_coords({over: labels})
    moved = gathered(source.clip(0, card - 1)).where(inside)
    return moved if edge.fill is None else absence.vacated(moved, array, over, ~inside, edge.fill)


def _per_group(offset: Any, groups: Any) -> Any:
    """*offset* at every coordinate, where it is declared over the group's own dim.

    One lag per group — a lead time that differs by period, which a
    ``(period, timestep)`` model writes as an offset over ``period`` because
    ``period`` is not the axis it walks. The group *is* the relation's value, so
    the lag a coordinate moves by is its group's, read through that relation:
    the pullback ``at()`` already is. Every other offset is returned as it
    came.

    The group's label is dropped rather than ridden along: a pullback leaves
    what it read through as a coordinate, and a constraint built from one is
    then reported as carrying a dimension the language says a shift does not
    have.
    """
    target = getattr(groups, 'name', None)
    if not isinstance(offset, xr.DataArray) or target not in offset.dims:
        return offset
    return operator_at(offset, (groups,), into=(str(target),)).drop_vars(str(target))


@dataclass(frozen=True)
class _Groups:
    """How one relation partitions an axis, as the arrays every in-group gather reads.

    Computed once per operand and shared across a window's lags: the partition
    does not depend on the lag.

    Attributes:
        labels: The axis's own labels, in order.
        grouped: Whether each coordinate belongs to any group.
        belongs: The group ordinal of each coordinate, 0 where it has none.
        within: Each coordinate's position inside its group.
        size: Each coordinate's group size, 1 where it has none.
        roster: The label ordinal at each ``(group, position)``.
        names: Each group's key, by group ordinal.
        counts: Each group's member count, by group ordinal.
    """

    labels: np.ndarray
    grouped: xr.DataArray
    belongs: xr.DataArray
    within: xr.DataArray
    size: xr.DataArray
    roster: np.ndarray
    names: tuple[object, ...]
    counts: tuple[int, ...]


def _grouped(over: str, labels: np.ndarray, groups: Any) -> _Groups:
    """The partition *groups* makes of the axis *over* carrying *labels*.

    A coordinate the relation sends nowhere belongs to no group: its ``within``
    is 0, its ``size`` 1 and its ``grouped`` False, and every gather reads the
    last of those first.
    """
    keys = np.asarray(groups.sel({over: labels}).values, dtype=object)

    peers: dict[object, list[int]] = {}
    within = np.zeros(len(labels), dtype=int)
    grouped = np.zeros(len(labels), dtype=bool)
    for k, key in enumerate(keys):
        if absence.unmapped(key):
            continue
        grouped[k] = True
        beside = peers.setdefault(key, [])
        within[k] = len(beside)
        beside.append(k)

    order = {key: g for g, key in enumerate(peers)}
    widest = max((len(beside) for beside in peers.values()), default=1)
    roster = np.zeros((max(len(peers), 1), widest), dtype=int)
    for key, beside in peers.items():
        roster[order[key], : len(beside)] = beside
    belongs = np.array([order.get(key, 0) for key in keys], dtype=int)
    span = np.array([len(peers[key]) if held else 1 for key, held in zip(keys, grouped, strict=True)], dtype=int)

    def on_axis(values: np.ndarray) -> xr.DataArray:
        return xr.DataArray(values, coords={over: labels}, dims=[over])

    return _Groups(
        labels,
        on_axis(grouped),
        on_axis(belongs),
        on_axis(within),
        on_axis(span),
        roster,
        tuple(peers),
        tuple(len(beside) for beside in peers.values()),
    )


def _gather_in_groups(array: Any, over: str, offset: Any, *, groups: _Groups, edge: _Edge) -> Any:
    """Translate *array* inside each group *groups* makes, not along the axis.

    The neighbour of a coordinate is the one *offset* back among the coordinates
    sharing its group, so the gather is by a source ordinal computed per group:
    a position past the group's start is vacated where the axis edge would
    vacate, and under *wrap* it comes round to that group's own last.

    *offset* is a number, or an array where the model named a parameter — per
    entity, per group (:func:`_per_group`), or both — so the source ordinal is
    computed from the within-group position rather than looked up member by
    member, and carries the offset's own dims.

    A coordinate in no group reaches nothing — the null reading a partial
    relation gets everywhere else. That is not the same as reaching *off* a
    group's edge, which is what a policy speaks for, so the two are tracked
    apart and only the second is filled.
    """
    reached = groups.within - offset
    if edge.wrap:
        reached = reached % groups.size
    inside = groups.grouped & (reached >= 0) & (reached < groups.size)

    def peer(group: Any, position: Any) -> Any:
        """The label ordinal sitting at *position* of the group at *group*."""
        return groups.roster[group, position]

    source = xr.apply_ufunc(peer, groups.belongs, reached.where(inside, 0).astype(int))
    labels = groups.labels
    gathered = array.sel({over: _labelled(labels, source, over)}).assign_coords({over: labels}).where(inside)
    if edge.fill is None:
        return gathered
    return absence.vacated(gathered, array, over, groups.grouped & ~inside, edge.fill)


def _off_the_axis(array: Any, over: str, offset: float) -> Any:
    """Which positions along *over* a scalar shift of *offset* leaves vacated.

    The source a position reads, off both ends, so one expression covers a
    shift in either direction — the same verdict :func:`_gather_by_offset`
    reaches with ``inside`` negated.
    """
    labels = np.asarray(array.indexes[over])
    source = xr.DataArray(np.arange(len(labels)), coords={over: labels}, dims=[over]) - int(offset)
    return (source < 0) | (source >= len(labels))


def _labelled(labels: Any, ordinals: Any, over: str) -> Any:
    """*ordinals* as the coordinate labels they stand for, keeping their dims.

    Carries no coordinates of its own: an indexer that keeps the axis's own
    coordinate asserts the values it holds *are* that axis, and after a gather
    they are not — which xarray reports as a size conflict rather than a
    mislabelling.
    """
    return xr.DataArray(labels[ordinals.transpose(*ordinals.dims).values], dims=ordinals.dims)
