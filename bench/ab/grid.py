"""The models an A/B times: one generator with an axis per thing a change may assume.

Every cell starts from ``BASELINE`` and moves one axis, so a cell that comes
back slower names the assumption that broke. ``WORST`` moves every axis at once.
The ladder's own cases at two sizes stand beside them, so the grid is checked
against the models the benchmark page publishes.

The main constraint is ``x0 + x1 + … >= demand`` over every dimension, with
each variable bounded below by 0, and the objective is a positive cost on every
variable. Every term ``x0`` keeps its coordinate under every mask, so every row
holds a variable and every cell solves.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl

from bench.cases import CASES, DEFAULT_CACHE

if TYPE_CHECKING:
    from pathlib import Path

#: The labels of every dimension past the first; the first takes the size.
OTHER_DIMS = {'d1': 4, 'd2': 3}

#: Total columns of the model at each size. ``below`` and ``above`` straddle 250,000 columns.
SIZES = {'tiny': 24, 's': 24_000, 'below': 240_000, 'above': 260_000, 'l': 1_200_000}

#: The ladder cases the grid is run beside, at each size of a run that is also a rung of theirs: ``s`` or ``l``.
LADDER = ('dispatch', 'nodal', 'transport', 'storage', 'fleet', 'profiled')


@dataclass(frozen=True)
class Shape:
    """One model of the grid, short of its size."""

    #: How many dimensions the main variables and the main constraint span.
    dims: int = 2
    #: How many variables over every dimension the main constraint adds.
    terms: int = 2
    #: Whether the main constraint adds a variable over the first dimension only.
    broadcast: bool = False
    #: Which rows or coordinates a ``where`` removes.
    mask: Literal['none', 'constraint-all', 'constraint-half', 'variable-half'] = 'none'
    #: The row order of every parameter table, against the order of its dimensions.
    order: Literal['sorted', 'shuffled', 'reversed'] = 'sorted'
    #: The order of the first dimension's own labels, which sets its ordinals.
    index: Literal['sorted', 'shuffled'] = 'sorted'
    #: The dtype of the first dimension's labels.
    labels: Literal['int', 'str'] = 'int'
    #: What bounds the variables from above: nothing, a literal, a table over every dimension, or over the first.
    bounds: Literal['none', 'scalar', 'every', 'first'] = 'every'
    #: Whether the main constraint adds a wrapped shift of ``x0`` along the first dimension.
    shift: bool = False
    #: Whether a second constraint is quadratic in ``x0``.
    quadratic: bool = False


BASELINE = Shape()
WORST = Shape(
    dims=3,
    terms=4,
    broadcast=True,
    mask='constraint-half',
    order='shuffled',
    index='shuffled',
    labels='str',
    bounds='first',
    shift=True,
    quadratic=True,
)

#: Every shape the grid times, by name: the baseline, each axis moved alone, and every axis moved at once.
SHAPES: dict[str, Shape] = {
    'baseline': BASELINE,
    'dims-1': replace(BASELINE, dims=1),
    'dims-3': replace(BASELINE, dims=3),
    'terms-1': replace(BASELINE, terms=1),
    'terms-4': replace(BASELINE, terms=4),
    'broadcast': replace(BASELINE, broadcast=True),
    'mask-constraint-all': replace(BASELINE, mask='constraint-all'),
    'mask-constraint-half': replace(BASELINE, mask='constraint-half'),
    'mask-variable-half': replace(BASELINE, mask='variable-half'),
    'order-shuffled': replace(BASELINE, order='shuffled'),
    'order-reversed': replace(BASELINE, order='reversed'),
    'index-shuffled': replace(BASELINE, index='shuffled'),
    'labels-str': replace(BASELINE, labels='str'),
    'bounds-none': replace(BASELINE, bounds='none'),
    'bounds-scalar': replace(BASELINE, bounds='scalar'),
    'bounds-first': replace(BASELINE, bounds='first'),
    'shift': replace(BASELINE, shift=True),
    'quadratic': replace(BASELINE, quadratic=True),
    'worst': WORST,
}


@dataclass(frozen=True)
class Cell:
    """One model at one size, named by its ``id``: ``grid-<shape>-<size>`` or ``case-<case>-<size>``."""

    id: str
    #: The spec, as ``sps.build`` takes it.
    spec: dict | Path
    #: Parameter and dimension names to parquet paths.
    sources: dict[str, str]
    #: Whether HiGHS can solve it; it takes no quadratic constraint.
    solvable: bool = True


def ids(sizes: list[str]) -> list[str]:
    """Every cell's id: each grid shape at each of *sizes*, then each ladder case at those of *sizes* it has."""
    rungs = {name: {shape.label for shape in CASES[name].ladder} for name in LADDER}
    return [f'grid-{name}-{size}' for name in SHAPES for size in sizes] + [
        f'case-{name}-{size}' for name in LADDER for size in sizes if size in rungs[name]
    ]


def cell(cell_id: str, cache: Path = DEFAULT_CACHE / 'ab') -> Cell:
    """The cell *cell_id* names, its tables written to parquet under *cache* on first use."""
    kind, rest = cell_id.split('-', 1)
    name, size = rest.rsplit('-', 1)
    if kind == 'case':
        case = CASES[name]
        rung = case.shape(size)
        return Cell(cell_id, case.spec_path(rung), case.data(rung))
    shape = SHAPES[name]
    key = cache / f'{name}-{size}'
    stamp = key / '.complete'
    if not stamp.exists():
        key.mkdir(parents=True, exist_ok=True)
        for source, frame in tables(shape, SIZES[size]).items():
            frame.write_parquet(key / f'{source}.parquet')
        stamp.write_text(repr(asdict(shape)))
    paths = {p.stem: str(p) for p in sorted(key.glob('*.parquet'))}
    return Cell(cell_id, spec(shape), paths, solvable=not shape.quadratic)


def _dims(shape: Shape) -> list[str]:
    return ['d0', *list(OTHER_DIMS)[: shape.dims - 1]]


def spec(shape: Shape) -> dict:
    """The model of *shape*, which no size changes."""
    dims = _dims(shape)
    parameters: dict[str, dict] = {'demand': {'dims': dims}, 'cost': {'dims': dims}}
    upper: object | None = {'none': None, 'scalar': 10, 'every': 'ub', 'first': 'ub'}[shape.bounds]
    if upper == 'ub':
        parameters['ub'] = {'dims': dims if shape.bounds == 'every' else ['d0']}
    bounds = {'lower': 0} if upper is None else {'lower': 0, 'upper': upper}

    names = [f'x{i}' for i in range(shape.terms)]
    variables: dict[str, dict] = {v: {'dims': dims, 'bounds': dict(bounds)} for v in names}
    if shape.mask == 'variable-half' and shape.terms > 1:
        parameters['gate'] = {'dims': dims, 'dtype': 'bool'}
        variables['x1']['where'] = 'gate'

    lhs = ' + '.join(names)
    objective = ' + '.join(f'sum({v} * cost)' for v in names)
    if shape.broadcast:
        variables['y'] = {'dims': ['d0'], 'bounds': {'lower': 0}}
        lhs += ' + y'
        objective += ' + sum(y)'
    if shape.shift:
        lhs += " - 0.5 * shift(x0, along=d0, offset=1, edge='wrap')"
    constraint: dict = {'dims': dims, 'expression': f'{lhs} >= demand'}
    if shape.mask == 'constraint-all':
        constraint['where'] = 'demand >= 0'
    elif shape.mask == 'constraint-half':
        constraint['where'] = 'demand >= 0.5'
    constraints = {'main': constraint}
    if shape.quadratic:
        constraints['curved'] = {'dims': dims, 'expression': 'x0 * x0 + x0 <= 200'}

    return {
        'dimensions': {d: {'dtype': 'int' if d == 'd0' and shape.labels == 'int' else 'str'} for d in dims},
        'parameters': parameters,
        'variables': variables,
        'constraints': constraints,
        'objective': {'sense': 'minimize', 'expression': objective},
    }


def tables(shape: Shape, columns: int) -> dict[str, pl.DataFrame]:
    """Every source [`spec`][] of *shape* reads, at about *columns* columns, from a fixed seed."""
    rng = np.random.default_rng(1931)
    dims = _dims(shape)
    rest = int(np.prod([OTHER_DIMS[d] for d in dims[1:]]))
    n0 = max(2, columns // (shape.terms * rest))

    first = np.arange(n0)
    index: dict[str, np.ndarray] = {
        'd0': first if shape.labels == 'int' else np.array([f'l{i:07d}' for i in first]),
        **{d: np.array([f'{d}_{i}' for i in range(OTHER_DIMS[d])]) for d in dims[1:]},
    }
    if shape.index == 'shuffled':
        index['d0'] = rng.permutation(index['d0'])
    product = _product(dims, index)
    n = product.height

    def table(values: np.ndarray, frame: pl.DataFrame = product) -> pl.DataFrame:
        out = frame.with_columns(value=pl.Series(values))
        if shape.order == 'shuffled':
            return out.sample(fraction=1.0, shuffle=True, seed=7)
        if shape.order == 'reversed':
            return out.reverse()
        return out

    frames = {d: pl.DataFrame({d: index[d]}) for d in dims}
    frames['demand'] = table(rng.uniform(0.0, 1.0, n))
    frames['cost'] = table(rng.uniform(1.0, 2.0, n))
    if shape.bounds == 'every':
        frames['ub'] = table(rng.uniform(5.0, 10.0, n))
    elif shape.bounds == 'first':
        firsts = pl.DataFrame({'d0': index['d0']})
        frames['ub'] = table(rng.uniform(5.0, 10.0, firsts.height), firsts)
    if shape.mask == 'variable-half' and shape.terms > 1:
        frames['gate'] = table(rng.random(n) < 0.5)
    return frames


def _product(dims: list[str], index: dict[str, np.ndarray]) -> pl.DataFrame:
    """Every coordinate of *dims*, row-major in each dimension's own label order."""
    frame = pl.DataFrame({dims[0]: index[dims[0]]})
    for d in dims[1:]:
        frame = frame.join(pl.DataFrame({d: index[d]}), how='cross')
    return frame
