"""The crossing into pandas and xarray: the door's frames as this lane's arrays.

The language's relation is a table over any number of dimensions, keyed by
any number of its columns. What this lane builds of it is the **single-valued
map**: one value column read at the key, so the map is one dense array over
the key's dimensions and a walk is an ``assign_coords`` and a ``groupby``, or
a vectorised ``sel``. A relation may carry several value columns and each is
its own array, since a call names the column it walks.
:func:`refuse_relations_the_lane_does_not_build` turns the rest away at the
lane's door, naming the relational lane, which builds every shape the language
admits.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
from mathspec import program as _program

from specsolve._frames import to_pandas
from specsolve._messages import coordinate_text, missing_row_message
from specsolve.errors import DataError, SpecsolveError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from mathspec import program


class OracleCannotBuildError(SpecsolveError):
    """The language accepts the spec and this oracle cannot build it."""


def refuse_relations_the_lane_does_not_build(program: program.Program) -> None:
    """Refuse a relation shape this lane does not build, before any data is read.

    Raises:
        OracleCannotBuildError: A partition grouped by a map keyed on more than the
            dimension it walks or by more than one column.
    """
    for node in _partitioning(program):
        assert node.partition is not None, '_partitioning yields only the nodes that carry one'
        if node.partition.joined:
            raise OracleCannotBuildError(
                _relation_shape_message(
                    f"a partition by '{node.partition.name}' groups by a map keyed by {list(node.partition.relation.key)}, "
                    f'and this lane groups a shift, sum_back or position by a map keyed by the dimension it '
                    f'walks alone'
                )
            )
        if len(node.partition.grouped) != 1:
            raise OracleCannotBuildError(
                _relation_shape_message(
                    f"a partition by '{node.partition.name}' groups by {list(node.partition.grouped)}, and this "
                    f'lane groups a shift, sum_back or position by one column'
                )
            )


def _relation_shape_message(what: str) -> str:
    return (
        f'the linopy lane builds the single-valued map — one value column read at the key — '
        f'and {what}. The language accepts it and the relational '
        f'lane builds it, so this is a limit of the lane rather than of the spec. Build it with '
        f'sps.build()/sps.solve() instead.'
    )


def _partitioning(
    program: program.Program,
) -> Iterator[program.Translate | program.WindowSum | program.DimensionPosition]:
    """Every node that groups by a relation: an operator in an expression, and a ``position(by=)`` in a mask."""
    bodies = (*program.roots, *(e.expression for e in program.expressions.values()))
    for node in _program.walk(*bodies):
        if isinstance(node, (_program.Translate, _program.WindowSum)) and node.partition is not None:
            yield node
        if isinstance(node, _program.Cases):
            for region in node.regions:
                yield from _positions(region.when)
    for declaration in (*program.variables.values(), *program.constraints.values()):
        yield from _positions(declaration.where)


def _positions(mask: program.Mask | None) -> Iterator[program.DimensionPosition]:
    """The grouped positions one mask tests, or nothing."""
    if mask is None:
        return
    for atom in mask.atoms:
        if isinstance(atom, _program.DimensionPosition) and atom.partition is not None:
            yield atom


def dimension_coords(
    program: program.Program,
    tidy: Mapping[str, pl.LazyFrame],
) -> tuple[dict[str, pd.Index], dict[tuple[str, str], xr.DataArray]]:
    """Every dimension's labels, and each relation's value columns as arrays over the dimensions it is keyed by.

    *tidy* is :func:`~specsolve._sources.tidy_sources`' output, so every index and
    map has been read and checked; what happens here is the conversion.

    Returns:
        The master coordinates by dimension, and one array per value column,
        by the relation's name and the column's.

    Raises:
        DataError: A relation that refuses a missing row has none at some
            combination of its key's labels.
    """
    master = {d: pd.Index(to_pandas(tidy[d].select(d).collect())[d], name=d) for d in program.dimensions}
    for name, relation in program.relations.items():
        if relation.missing == 'refused':
            key = relation.key or relation.roles
            _refuse_a_missing_row('relation', name, tidy[name].select(*key).unique(), key, relation.dim, tidy)
    return master, _relation_arrays(program, tidy, master)


def _refuse_a_missing_row(
    kind: str,
    name: str,
    rows: pl.LazyFrame,
    columns: tuple[str, ...],
    dim: Callable[[str], str],
    tidy: Mapping[str, pl.LazyFrame],
) -> None:
    """Refuse *rows* that hold no row at some combination of *columns*' labels, naming the first.

    The first in the order of the labels' product, read off the tidy frames
    with the relational lane's own joins, so both lanes name one coordinate.
    """
    if not columns:
        return
    labels = [tidy[dim(column)].select(pl.col(dim(column)).alias(column)) for column in columns]
    product = labels[0]
    for frame in labels[1:]:
        product = product.join(frame, how='cross')
    schema = product.collect_schema()
    held = rows.select(pl.col(column).cast(schema[column]) for column in columns)
    gaps = product.join(held, on=list(columns), how='anti').collect()
    if gaps.height:
        raise DataError(missing_row_message(kind, name, coordinate_text(gaps.row(0, named=True)), gaps.height))


def _relation_arrays(
    program: program.Program,
    tidy: Mapping[str, pl.LazyFrame],
    master: Mapping[str, pd.Index],
) -> dict[tuple[str, str], xr.DataArray]:
    """Each value column as an array over the dimensions its relation's key names.

    A map arrives as the table it declares, holding rows only where it is
    defined. **The padding happens here**: an array is dense by construction,
    and linopy's ``groupby`` wants one aligned to the dimensions' coordinates
    — so a key the relation leaves out becomes a null, which every reader on
    this lane treats as "in no group". A key of several columns pads to their
    product the same way.
    """
    out: dict[tuple[str, str], xr.DataArray] = {}
    for name, relation in program.relations.items():
        dims = [relation.dim(role) for role in relation.key]
        frame = to_pandas(tidy[name].collect())
        keys = [frame[role].to_numpy() for role in relation.key]
        keyed = pd.Index(keys[0], name=dims[0]) if len(dims) == 1 else pd.MultiIndex.from_arrays(keys, names=dims)
        index = pd.MultiIndex.from_product([master[d] for d in dims], names=dims) if len(dims) > 1 else master[dims[0]]
        shape = tuple(len(master[d]) for d in dims)
        coords = {d: master[d] for d in dims}
        for value in relation.values:
            padded = pd.Series(frame[value].to_numpy(), index=keyed).reindex(index)
            out[name, value] = xr.DataArray(padded.to_numpy().reshape(shape), dims=dims, coords=coords, name=name)
        if not relation.values:
            out[name, ''] = _membership(relation, frame, master)
    return out


def _membership(relation: Any, frame: pd.DataFrame, master: Mapping[str, pd.Index]) -> xr.DataArray:
    """A bare relation as one where its rows are and zero elsewhere, over its key's *roles*.

    The roles name the axes because two of them may range over one dimension —
    an event's ``start`` and the snapshot it ``covered`` are both snapshots.
    """
    roles = list(relation.key)
    index = pd.MultiIndex.from_product([master[relation.dim(role)] for role in roles], names=roles)
    held = pd.MultiIndex.from_arrays([frame[role].to_numpy() for role in roles], names=roles)
    shape = tuple(len(master[relation.dim(role)]) for role in roles)
    coords = {role: master[relation.dim(role)].rename(role) for role in roles}
    return xr.DataArray(index.isin(held).astype(float).reshape(shape), dims=roles, coords=coords)


def load_parameters(
    program: program.Program,
    tidy: Mapping[str, pl.LazyFrame],
    master_coords: Mapping[str, pd.Index],
) -> xr.Dataset:
    """Every declared parameter as the dataset this lane builds against, its ``missing:`` applied.

    *tidy* is :func:`~specsolve._sources.tidy_sources`' output — one checked
    ``(dims…, value)`` frame per parameter — so what is left is the crossing
    into xarray, and what a missing row reads as. A value fills it; ``absent``
    and ``neutral`` leave it NaN, which each position reads in its own way
    (:mod:`~tests.linopy_lane.absence`); a ``bool`` short of a row reads false
    there unless its value says otherwise.

    Returns:
        One DataArray per parameter, reindexed onto the master coordinates.

    Raises:
        DataError: A parameter that refuses a missing row has none at some
            coordinate of its dims.
    """
    arrays: dict[str, xr.DataArray] = {}
    for pname, pdef in program.parameters.items():
        frame = tidy[pname].collect()
        if not pdef.dims and frame.is_empty():
            arrays[pname] = xr.DataArray(_fill(pdef, pdef.dtype == 'bool'))
            continue
        arr = _from_tidy(frame, pdef.dims)
        if pdef.dims:
            if pdef.missing == 'refused':
                _refuse_a_missing_row('parameter', pname, tidy[pname], pdef.dims, lambda d: d, tidy)
            onto = {d: master_coords[d] for d in pdef.dims}
            arr = arr.reindex(onto, fill_value=_fill(pdef, arr.dtype == bool))
        arrays[pname] = arr
    return xr.Dataset(arrays)


def given_rows(
    program: program.Program, tidy: Mapping[str, pl.LazyFrame], master_coords: Mapping[str, pd.Index]
) -> dict[str, xr.DataArray]:
    """Where the data gave a row, for each numeric parameter that a value fills where it gave none.

    What a bare numeric name in a ``where`` asks whatever the ``missing:``:
    the fill is a value to read, and not a row the data holds.
    """
    given: dict[str, xr.DataArray] = {}
    for pname, pdef in program.parameters.items():
        if not pdef.dims or pdef.dtype == 'bool' or isinstance(pdef.missing, str) or pdef.missing is None:
            continue
        product = math.prod(len(master_coords[d]) for d in pdef.dims)
        if tidy[pname].select(pl.len()).collect().item() < product:
            held = _from_tidy(tidy[pname].collect(), pdef.dims).reindex({d: master_coords[d] for d in pdef.dims})
            given[pname] = held.notnull()
    return given


def _fill(pdef: program.Parameter, flag: bool) -> Any:
    """What a reindex writes where the data has no row: the declared value, false for a flag, NaN otherwise."""
    if not isinstance(pdef.missing, str) and pdef.missing is not None:
        return pdef.missing
    return False if flag else np.nan


def _from_tidy(frame: pl.DataFrame, dims: Sequence[str]) -> xr.DataArray:
    """A tidy ``(dims…, value)`` frame as an array.

    A dims-less value keeps the dtype it arrived with, as a column does — a
    ``bool`` cast to ``0.0`` would read as *defined* under a bare ``where``.
    """
    if not dims:
        return xr.DataArray(frame['value'].to_numpy()[0])
    columns = [frame[d].to_numpy() for d in dims]
    index = (
        pd.Index(columns[0], name=dims[0]) if len(dims) == 1 else pd.MultiIndex.from_arrays(columns, names=list(dims))
    )
    return xr.DataArray.from_series(pd.Series(frame['value'].to_numpy(), index=index))
