"""The models we time, and the data that sizes them.

One case = one YAML model + a deterministic data generator + a size ladder.
Each case stresses a different SQL shape:

``dispatch``   pointwise bounds + one ``sum``; its ``where`` removes nothing.
``commitment`` dispatch with a binary commitment gating every generator — the
               MILP. Its bottom rung is tiny so branch and bound closes the gap
               exactly at the parity gate.
``nodal``      dispatch over (snapshot, node, tech) where a technology exists
               only at the nodes it is installed at.
``transport``  three ``sum(by=)`` joins per row — the mapping-table path.
``storage``    a cyclic ``shift`` recurrence, a term stream joined against
               itself on ``snapshot.ord - 1``, at ``dispatch``'s width and
               snapshot counts.
``sector``     ``nodal``'s sparse portfolio crossed with dense carriers.
``fleet``      the same variable total spread over many declarations.
``declarations`` ``fleet``'s question as a sweep over the declaration count,
               with its model YAML generated per rung (``_declarations_spec``).
``profiled``   ``nodal``'s ladder with no mask, and an availability table with a
               row per variable.

A rung label counts variables per snapshot across all of a case's
declarations. ``nominal_variables`` is the full coordinate product; what
survives a mask is measured (``live`` in the report).

Data is generated once per (case, shape) into a cache directory, and every arm
reads the same parquet files. Every generator is feasible by construction.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

BENCH_DIR = Path(__file__).resolve().parent
MODELS = BENCH_DIR / 'models'
DEFAULT_CACHE = BENCH_DIR / '.cache'


@dataclass(frozen=True)
class Shape:
    """One rung of a ladder: the dimension cardinalities, and how much survives.

    ``density`` is the fraction of the coordinate product a case's mask keeps;
    1.0 for cases with no mask. ``masked`` says whether each declaration
    carries a ``where:`` at all, whatever it keeps.
    """

    label: str
    sizes: dict[str, int]
    nominal_variables: int
    density: float = 1.0
    masked: bool = False

    @property
    def key(self) -> str:
        dims = '-'.join(f'{k}{v}' for k, v in sorted(self.sizes.items()))
        if self.density != 1.0:
            dims = f'{dims}-d{self.density:g}'
        return f'{dims}-masked' if self.masked else dims


@dataclass(frozen=True)
class Case:
    name: str
    ladder: tuple[Shape, ...]
    write: Callable[[Shape, Path], dict[str, str]]
    spec: Path | None = None
    generate_spec: Callable[[Shape], str] | None = None
    #: A parameter that multiplies a variable inside a constraint, so new values of it move the matrix.
    coefficient: str | None = None

    def spec_path(self, shape: Shape, cache: Path = DEFAULT_CACHE) -> Path:
        """The YAML *shape* builds — ``spec``, unless the case generates one per rung.

        A generated spec is cached beside the rung's data, under ``shape.key``.
        """
        if self.generate_spec is None:
            if self.spec is None:
                raise ValueError(f'{self.name}: neither a spec file nor a generator — nothing to build')
            return self.spec
        dest = cache / self.name / shape.key / 'spec.yaml'
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(self.generate_spec(shape))
        return dest

    def shape(self, label: str) -> Shape:
        for s in self.ladder:
            if s.label == label:
                return s
        known = ', '.join(s.label for s in self.ladder)
        raise KeyError(f"{self.name}: no size '{label}' (have: {known})")

    def data(self, shape: Shape, cache: Path = DEFAULT_CACHE) -> dict[str, str]:
        """Parquet paths for *shape*, generating them on first use."""
        out = cache / self.name / shape.key
        stamp = out / '.complete'
        if not stamp.exists():
            out.mkdir(parents=True, exist_ok=True)
            paths = self.write(shape, out)
            stamp.write_text('\n'.join(sorted(paths)))
            return paths
        return {p.stem: str(p) for p in sorted(out.glob('*.parquet'))}


def shortened(shape: Shape) -> Shape:
    """*shape* one snapshot shorter: the next window of a horizon whose length moved.

    Every rung carries ``snapshot``, and a build one snapshot shorter has
    other counts, so no loaded solver can take it by value.
    """
    snapshots = shape.sizes['snapshot']
    return replace(
        shape,
        sizes={**shape.sizes, 'snapshot': snapshots - 1},
        nominal_variables=shape.nominal_variables // snapshots * (snapshots - 1),
    )


def rescaled(path: str) -> pl.DataFrame:
    """The parameter at *path* with every value one percent larger: new coefficients on the same pattern.

    A zero stays zero, so no entry enters or leaves the matrix; only its numbers move.
    """
    return pl.read_parquet(path).with_columns(pl.col('value') * 1.01)


def _seed(shape: Shape) -> np.random.Generator:
    """Same shape, same numbers, in every process; ``hash()`` is salted per process."""
    digest = hashlib.blake2b(shape.key.encode(), digest_size=4).digest()
    return np.random.default_rng(int.from_bytes(digest, 'big'))


def _dump(frames: dict[str, pd.DataFrame], dest: Path) -> dict[str, str]:
    paths = {}
    for name, df in frames.items():
        path = (dest / f'{name}.parquet').absolute()
        df.to_parquet(path, index=False)
        paths[name] = str(path)
    return paths


def _installed_frame(nodes: list[str], techs: list[str], installed: np.ndarray, capacity: np.ndarray) -> pd.DataFrame:
    """The (node, tech) pairs that exist, tidy — the table *is* the sparsity."""
    live = installed.reshape(-1)
    return pd.DataFrame(
        {
            'node': np.repeat(nodes, len(techs))[live],
            'tech': np.tile(techs, len(nodes))[live],
            'value': capacity.reshape(-1)[live],
        }
    )


# --------------------------------------------------------------------------
# dispatch


def _dispatch_data(shape: Shape, dest: Path) -> dict[str, str]:
    """Parquet for one rung of the ``dispatch`` ladder.

    Load is drawn against the fleet total, so it is feasible.
    """
    rng = _seed(shape)
    n_snap, n_gen = shape.sizes['snapshot'], shape.sizes['generator']
    gens = [f'g{i:05d}' for i in range(n_gen)]

    p_max = rng.uniform(50.0, 150.0, n_gen)
    cost = rng.uniform(10.0, 100.0, n_gen)
    load = p_max.sum() * 0.6 * (0.8 + 0.4 * rng.random(n_snap))

    return _dump(
        {
            'p_max': pd.DataFrame({'generator': gens, 'value': p_max}),
            'cost': pd.DataFrame({'generator': gens, 'value': cost}),
            'load': pd.DataFrame({'snapshot': np.arange(n_snap), 'value': load}),
            'generator': pd.DataFrame({'generator': gens}),
            'snapshot': pd.DataFrame({'snapshot': np.arange(n_snap)}),
        },
        dest,
    )


# --------------------------------------------------------------------------
# commitment


def _commitment_data(shape: Shape, dest: Path) -> dict[str, str]:
    """Parquet for one rung of the ``commitment`` ladder.

    Load is ``dispatch``'s draw, so every snapshot is feasible with the whole
    fleet on. Fix costs are drawn wide, so the optimal commitment is not all-on.
    """
    rng = _seed(shape)
    n_snap, n_gen = shape.sizes['snapshot'], shape.sizes['generator']
    gens = [f'g{i:05d}' for i in range(n_gen)]

    p_max = rng.uniform(50.0, 150.0, n_gen)
    cost = rng.uniform(10.0, 100.0, n_gen)
    fix_cost = rng.uniform(100.0, 2000.0, n_gen)
    load = p_max.sum() * 0.6 * (0.8 + 0.4 * rng.random(n_snap))

    return _dump(
        {
            'p_max': pd.DataFrame({'generator': gens, 'value': p_max}),
            'cost': pd.DataFrame({'generator': gens, 'value': cost}),
            'fix_cost': pd.DataFrame({'generator': gens, 'value': fix_cost}),
            'load': pd.DataFrame({'snapshot': np.arange(n_snap), 'value': load}),
            'generator': pd.DataFrame({'generator': gens}),
            'snapshot': pd.DataFrame({'snapshot': np.arange(n_snap)}),
        },
        dest,
    )


# --------------------------------------------------------------------------
# nodal

#: Technologies a system might have. Which ones a given node *has* is the mask.
TECHNOLOGIES = (
    'onwind',
    'offwind',
    'solar',
    'hydro',
    'ror',
    'biomass',
    'geothermal',
    'ccgt',
    'ocgt',
    'coal',
    'nuclear',
    'oil',
)


def _portfolios(rng: np.random.Generator, n_node: int, n_tech: int, density: float) -> np.ndarray:
    """Which (node, tech) pairs exist — a boolean node x tech matrix.

    Every node gets at least one technology; the rest are drawn to the
    requested density.
    """
    per_node = max(1, round(density * n_tech))
    installed = np.zeros((n_node, n_tech), dtype=bool)
    for i in range(n_node):
        installed[i, rng.choice(n_tech, size=per_node, replace=False)] = True
    return installed


def _nodal_data(shape: Shape, dest: Path) -> dict[str, str]:
    """Parquet for one rung of the ``nodal`` ladder.

    Every node meets its own demand from its own portfolio. Only installed pairs
    are written.
    """
    rng = _seed(shape)
    n_snap, n_node = shape.sizes['snapshot'], shape.sizes['node']
    techs = list(TECHNOLOGIES[: shape.sizes['tech']])
    nodes = [f'n{i:04d}' for i in range(n_node)]

    installed = _portfolios(rng, n_node, len(techs), shape.density)
    capacity = installed * rng.uniform(200.0, 800.0, (n_node, len(techs)))
    cost = rng.uniform(10.0, 100.0, len(techs))

    at_node = capacity.sum(axis=1)
    demand = at_node[None, :] * 0.5 * (0.8 + 0.4 * rng.random((n_snap, n_node)))

    return _dump(
        {
            'installed': _installed_frame(nodes, techs, installed, capacity),
            'cost': pd.DataFrame({'tech': techs, 'value': cost}),
            'demand': pd.DataFrame(
                {
                    'snapshot': np.repeat(np.arange(n_snap), n_node),
                    'node': nodes * n_snap,
                    'value': demand.reshape(-1),
                }
            ),
            'node': pd.DataFrame({'node': nodes}),
            'tech': pd.DataFrame({'tech': techs}),
            'snapshot': pd.DataFrame({'snapshot': np.arange(n_snap)}),
        },
        dest,
    )


# --------------------------------------------------------------------------
# sector


#: What each technology's output arrives as, one carrier per technology.
CARRIERS = ('electricity', 'heat', 'hydrogen', 'gas', 'transport')


def _sector_data(shape: Shape, dest: Path) -> dict[str, str]:
    """Parquet for one rung of the ``sector`` ladder.

    ``reachable`` is what a node can deliver into a carrier. Demand exists only
    where that is nonzero, so the model is feasible on any draw.
    """
    rng = _seed(shape)
    n_snap, n_node = shape.sizes['snapshot'], shape.sizes['node']
    techs = list(TECHNOLOGIES[: shape.sizes['tech']])
    carriers = list(CARRIERS[: shape.sizes['carrier']])
    nodes = [f'n{i:04d}' for i in range(n_node)]

    installed = _portfolios(rng, n_node, len(techs), shape.density)
    capacity = installed * rng.uniform(200.0, 800.0, (n_node, len(techs)))
    serves = rng.integers(0, len(carriers), len(techs))
    efficiency = rng.uniform(0.3, 0.95, len(techs))

    reachable = np.zeros((n_node, len(carriers)))
    for t, c in enumerate(serves):
        reachable[:, c] += capacity[:, t] * efficiency[t]
    served = reachable > 0
    demand = reachable[None, :, :] * 0.6 * (0.8 + 0.4 * rng.random((n_snap, n_node, len(carriers))))
    live = np.broadcast_to(served, demand.shape).reshape(-1)

    return _dump(
        {
            'installed': _installed_frame(nodes, techs, installed, capacity),
            'produces': pd.DataFrame({'tech': techs, 'carrier': [carriers[c] for c in serves], 'value': efficiency}),
            'cost': pd.DataFrame({'tech': techs, 'value': rng.uniform(10.0, 100.0, len(techs))}),
            'demand': pd.DataFrame(
                {
                    'snapshot': np.repeat(np.arange(n_snap), n_node * len(carriers))[live],
                    'node': np.tile(np.repeat(nodes, len(carriers)), n_snap)[live],
                    'carrier': np.array(carriers * (n_snap * n_node))[live],
                    'value': demand.reshape(-1)[live],
                }
            ),
            'node': pd.DataFrame({'node': nodes}),
            'tech': pd.DataFrame({'tech': techs}),
            'carrier': pd.DataFrame({'carrier': carriers}),
            'snapshot': pd.DataFrame({'snapshot': np.arange(n_snap)}),
        },
        dest,
    )


# --------------------------------------------------------------------------
# transport


def _transport_data(shape: Shape, dest: Path) -> dict[str, str]:
    """Parquet for one rung of the ``transport`` ladder.

    Generation is dealt round-robin so every bus has some, and the network is a
    ring plus chords, so ``from != to``. Load is sized against a bus's own
    generators, so the model is feasible whatever the line capacities.
    """
    rng = _seed(shape)
    n_snap = shape.sizes['snapshot']
    n_gen, n_bus, n_line = shape.sizes['generator'], shape.sizes['bus'], shape.sizes['line']

    buses = [f'b{i:04d}' for i in range(n_bus)]
    gens = [f'g{i:05d}' for i in range(n_gen)]
    gen_bus = [buses[i % n_bus] for i in range(n_gen)]
    p_max = rng.uniform(50.0, 150.0, n_gen)

    lines = [f'l{i:05d}' for i in range(n_line)]
    frm = [buses[i % n_bus] for i in range(n_line)]
    to = [buses[(i % n_bus + 1 + i // n_bus) % n_bus] for i in range(n_line)]

    own = pd.Series(p_max, index=gen_bus).groupby(level=0).sum().reindex(buses).to_numpy()
    load = own[None, :] * 0.5 * (0.8 + 0.4 * rng.random((n_snap, n_bus)))
    snaps = np.repeat(np.arange(n_snap), n_bus)

    return _dump(
        {
            'p_max': pd.DataFrame({'generator': gens, 'value': p_max}),
            'cost': pd.DataFrame({'generator': gens, 'value': rng.uniform(10.0, 100.0, n_gen)}),
            'cap': pd.DataFrame({'line': lines, 'value': rng.uniform(20.0, 80.0, n_line)}),
            'neg_cap': pd.DataFrame({'line': lines, 'value': -rng.uniform(20.0, 80.0, n_line)}),
            'load': pd.DataFrame({'snapshot': snaps, 'bus': buses * n_snap, 'value': load.ravel()}),
            'generator': pd.DataFrame({'generator': gens}),
            'gen_bus': pd.DataFrame({'generator': gens, 'bus': gen_bus}),
            'line': pd.DataFrame({'line': lines}),
            'line_from': pd.DataFrame({'line': lines, 'bus': frm}),
            'line_to': pd.DataFrame({'line': lines, 'bus': to}),
            'bus': pd.DataFrame({'bus': buses}),
            'snapshot': pd.DataFrame({'snapshot': np.arange(n_snap)}),
        },
        dest,
    )


# --------------------------------------------------------------------------
# fleet


def _fleet_data(shape: Shape, dest: Path) -> dict[str, str]:
    """Parquet for one rung of the ``fleet`` ladder — and of ``declarations``,
    whose generated models read the same three tables.

    Demand can be met three ways, all three priced.
    """
    rng = _seed(shape)
    n_snap, n_unit = shape.sizes['snapshot'], shape.sizes['unit']
    units = [f'u{i:05d}' for i in range(n_unit)]

    p_max = rng.uniform(50.0, 150.0, n_unit)
    demand = p_max.sum() * 0.6 * (0.8 + 0.4 * rng.random(n_snap))

    return _dump(
        {
            'p_max': pd.DataFrame({'unit': units, 'value': p_max}),
            'cost': pd.DataFrame({'unit': units, 'value': rng.uniform(10.0, 100.0, n_unit)}),
            'demand': pd.DataFrame({'snapshot': np.arange(n_snap), 'value': demand}),
            'unit': pd.DataFrame({'unit': units}),
            'snapshot': pd.DataFrame({'snapshot': np.arange(n_snap)}),
        },
        dest,
    )


# --------------------------------------------------------------------------
# declarations


def _added(terms: Iterable[str]) -> str:
    """*terms* summed as a balanced tree rather than a left-leaning chain.

    A chain of N terms parses to a tree N deep, and the language admits 100
    levels; bracketing makes the depth log2(N).
    """
    items = list(terms)
    if len(items) == 1:
        return items[0]
    half = len(items) // 2
    return f'({_added(items[:half])} + {_added(items[half:])})'


def _declarations_spec(shape: Shape) -> str:
    """The ``declarations`` model at this rung's declaration count.

    Each declaration gets its own variable, capacity constraint and objective
    term, and one balance ties them to the load. On a ``masked`` rung every
    declaration also carries ``where: p_max > 0``, which ``_fleet_data`` draws
    strictly positive, so the paired rungs build the identical model.
    """
    names = [f'v{i:03d}' for i in range(shape.sizes['declaration'])]
    guard = ', where: "p_max > 0"' if shape.masked else ''
    variables = '\n'.join(
        f'  {v}: {{dims: [snapshot, unit], bounds: {{lower: 0, upper: p_max}}{guard}}}' for v in names
    )
    caps = '\n'.join(f'  cap_{v}: {{dims: [snapshot, unit], expression: "{v} <= p_max"{guard}}}' for v in names)
    balance = _added(f'sum({v}, over=unit)' for v in names)
    objective = _added(f'sum({v} * cost)' for v in names)
    return f"""# Generated by bench.cases._declarations_spec — one file per rung of the
# declaration sweep. fleet's mechanism with the declaration count as the axis.
dimensions:
  snapshot:
    dtype: int
  unit:
    dtype: str

parameters:
  p_max:
    dims: [unit]
  cost:
    dims: [unit]
  demand:
    dims: [snapshot]

variables:
{variables}

constraints:
{caps}
  balance:
    dims: [snapshot]
    expression: "{balance} == demand"

objective:
  sense: minimize
  expression: "{objective}"
"""


# --------------------------------------------------------------------------
# profiled


def _profiled_data(shape: Shape, dest: Path) -> dict[str, str]:
    """Parquet for one rung of the ``profiled`` ladder.

    ``availability`` has one row per variable and is never zero. Demand is half
    of what is available in that snapshot. The label columns are categorical so
    the upper rungs generate quickly; the parquet is dictionary-encoded either
    way.
    """
    rng = _seed(shape)
    n_snap, n_node = shape.sizes['snapshot'], shape.sizes['node']
    techs = list(TECHNOLOGIES[: shape.sizes['tech']])
    n_tech = len(techs)
    nodes = [f'n{i:04d}' for i in range(n_node)]

    capacity = rng.uniform(200.0, 800.0, (n_node, n_tech))
    availability = capacity[None, :, :] * (0.2 + 0.8 * rng.random((n_snap, n_node, n_tech)))
    demand = availability.sum(axis=2) * 0.5

    return _dump(
        {
            'availability': pd.DataFrame(
                {
                    'snapshot': np.repeat(np.arange(n_snap), n_node * n_tech),
                    'node': pd.Categorical.from_codes(
                        np.tile(np.repeat(np.arange(n_node), n_tech), n_snap), categories=pd.Index(nodes)
                    ),
                    'tech': pd.Categorical.from_codes(
                        np.tile(np.arange(n_tech), n_snap * n_node), categories=pd.Index(techs)
                    ),
                    'value': availability.reshape(-1),
                }
            ),
            'cost': pd.DataFrame({'tech': techs, 'value': rng.uniform(10.0, 100.0, n_tech)}),
            'demand': pd.DataFrame(
                {
                    'snapshot': np.repeat(np.arange(n_snap), n_node),
                    'node': nodes * n_snap,
                    'value': demand.reshape(-1),
                }
            ),
            'node': pd.DataFrame({'node': nodes}),
            'tech': pd.DataFrame({'tech': techs}),
            'snapshot': pd.DataFrame({'snapshot': np.arange(n_snap)}),
        },
        dest,
    )


# --------------------------------------------------------------------------
# storage


def _storage_data(shape: Shape, dest: Path) -> dict[str, str]:
    """Parquet for one rung of the ``storage`` ladder.

    Load is ``dispatch``'s, so the generators alone serve every snapshot.
    Generator costs spread over an order of magnitude, so the optimum still uses
    storage.
    """
    rng = _seed(shape)
    n_snap = shape.sizes['snapshot']
    n_gen, n_store = shape.sizes['generator'], shape.sizes['store']

    gens = [f'g{i:05d}' for i in range(n_gen)]
    stores = [f's{i:05d}' for i in range(n_store)]
    p_max = rng.uniform(50.0, 150.0, n_gen)
    load = p_max.sum() * 0.6 * (0.8 + 0.4 * rng.random(n_snap))
    p_store = rng.uniform(10.0, 40.0, n_store)

    return _dump(
        {
            'p_max': pd.DataFrame({'generator': gens, 'value': p_max}),
            'cost': pd.DataFrame({'generator': gens, 'value': rng.uniform(10.0, 100.0, n_gen)}),
            'load': pd.DataFrame({'snapshot': np.arange(n_snap), 'value': load}),
            'e_max': pd.DataFrame({'store': stores, 'value': p_store * 4.0}),
            'p_store': pd.DataFrame({'store': stores, 'value': p_store}),
            'eta': pd.DataFrame({'store': stores, 'value': rng.uniform(0.85, 0.95, n_store)}),
            'generator': pd.DataFrame({'generator': gens}),
            'store': pd.DataFrame({'store': stores}),
            'snapshot': pd.DataFrame({'snapshot': np.arange(n_snap)}),
        },
        dest,
    )


# --------------------------------------------------------------------------


def _ladder(
    sizes: dict[str, int],
    snapshots: Sequence[int],
    per_snapshot: int,
    density: float = 1.0,
) -> tuple[Shape, ...]:
    """A case's rungs, one per entry of *snapshots*, below them a tenth of the first.

    ``xs``..``l`` is the published ladder. ``xl`` and ``2xl`` test the claim in
    ``docs/about/benchmarks.md`` that a model whose dense build cannot fit on
    the machine still streams out under the budget. ``2xs`` is the size of one
    slice of a sweep or one solve of a model predictive controller, where a
    build's fixed cost per query outweighs its rows. It keeps two snapshots at
    least, so a window one snapshot shorter still has one.
    """
    labels = ('xs', 's', 'm', 'l', 'xl', '2xl')
    smallest = max(2, snapshots[0] // 10)
    return (
        Shape('2xs', {**sizes, 'snapshot': smallest}, smallest * per_snapshot, density),
        *(
            Shape(labels[i], {**sizes, 'snapshot': n}, n * per_snapshot, density)
            for i, n in enumerate(snapshots)
            if i < len(labels)
        ),
    )


def _width_ladder(
    entities: dict[str, int], snapshots: int, per_snapshot: int, multipliers: Sequence[int] = (1, 10, 100, 1000)
) -> tuple[Shape, ...]:
    """The size ladder's variable counts, grown sideways instead of forward.

    The entity counts grow and ``snapshot`` is fixed. Each rung matches the size
    ladder's variable count: ``w1`` is ``xs``, ``w1000`` is ``l``.
    """
    return tuple(
        Shape(
            f'w{m}',
            {name: count * m for name, count in entities.items()} | {'snapshot': snapshots},
            snapshots * per_snapshot * m,
        )
        for m in multipliers
    )


def _declaration_sweep(
    pool: int, snapshots: int, counts: Sequence[int], masked: Sequence[int] = ()
) -> tuple[Shape, ...]:
    """One model size, several declaration counts — rungs named ``n002``/``n008``/…

    The pool of units splits into N declarations of pool/N units each, so total
    variables, rows and snapshots are flat across the sweep. Counts in *masked*
    get a second rung, suffixed ``m``, whose declarations each carry a vacuous
    ``where:``.
    """
    for n in (*counts, *masked):
        if pool % n:
            raise ValueError(f'a pool of {pool} units does not split into {n} equal declarations')
    if not set(masked) <= set(counts):
        raise ValueError(f'masked rungs {sorted(set(masked) - set(counts))} have no dense twin to be read against')
    sizes = {n: {'declaration': n, 'unit': pool // n, 'snapshot': snapshots} for n in (*counts, *masked)}
    return (
        *(Shape(f'n{n:03d}', sizes[n], snapshots * pool) for n in counts),
        *(Shape(f'n{n:03d}m', sizes[n], snapshots * pool, masked=True) for n in masked),
    )


def _density_sweep(
    sizes: dict[str, int], snapshots: int, per_snapshot: int, densities: Sequence[float]
) -> tuple[Shape, ...]:
    """One model size, several mask densities — rungs named ``d100``/``d30``/…"""
    return tuple(
        Shape(f'd{round(d * 100):02d}', {**sizes, 'snapshot': snapshots}, snapshots * per_snapshot, d)
        for d in densities
    )


CASES: dict[str, Case] = {
    'dispatch': Case(
        name='dispatch',
        spec=MODELS / 'dispatch' / 'spec.yaml',
        ladder=_ladder({'generator': 100}, (100, 1_000, 10_000, 100_000, 400_000, 1_200_000), per_snapshot=100),
        write=_dispatch_data,
    ),
    'commitment': Case(
        name='commitment',
        spec=MODELS / 'commitment' / 'spec.yaml',
        ladder=_ladder({'generator': 50}, (10, 100, 1_000, 10_000, 40_000, 120_000), per_snapshot=100),
        write=_commitment_data,
        coefficient='p_max',
    ),
    'fleet': Case(
        name='fleet',
        spec=MODELS / 'fleet' / 'spec.yaml',
        ladder=_ladder({'unit': 50}, (20, 200, 2_000, 20_000, 80_000, 240_000), per_snapshot=600),
        write=_fleet_data,
    ),
    'declarations': Case(
        name='declarations',
        ladder=_declaration_sweep(pool=512, snapshots=2_000, counts=(2, 8, 32, 128), masked=(8, 128)),
        write=_fleet_data,
        generate_spec=_declarations_spec,
    ),
    'nodal': Case(
        name='nodal',
        spec=MODELS / 'nodal' / 'spec.yaml',
        ladder=(
            *_ladder(
                {'node': 50, 'tech': 12}, (20, 200, 2_000, 20_000, 80_000, 240_000), per_snapshot=600, density=0.25
            ),
            *_density_sweep({'node': 50, 'tech': 12}, 2_000, 600, (1.0, 0.5, 0.25, 0.083)),
        ),
        write=_nodal_data,
    ),
    'sector': Case(
        name='sector',
        spec=MODELS / 'sector' / 'spec.yaml',
        ladder=_ladder(
            {'node': 50, 'tech': 12, 'carrier': 5},
            (20, 200, 2_000, 20_000, 80_000, 240_000),
            per_snapshot=850,
            density=0.083,
        ),
        write=_sector_data,
        coefficient='produces',
    ),
    'profiled': Case(
        name='profiled',
        spec=MODELS / 'profiled' / 'spec.yaml',
        ladder=_ladder({'node': 50, 'tech': 12}, (20, 200, 2_000, 20_000, 80_000, 240_000), per_snapshot=600),
        write=_profiled_data,
    ),
    'transport': Case(
        name='transport',
        spec=MODELS / 'transport' / 'spec.yaml',
        ladder=(
            *_ladder(
                {'generator': 100, 'bus': 20, 'line': 40}, (70, 700, 7_000, 70_000, 280_000, 840_000), per_snapshot=140
            ),
            *_width_ladder({'generator': 100, 'bus': 20, 'line': 40}, snapshots=70, per_snapshot=140),
        ),
        write=_transport_data,
    ),
    'storage': Case(
        name='storage',
        spec=MODELS / 'storage' / 'spec.yaml',
        ladder=(
            *_ladder(
                {'generator': 40, 'store': 20}, (100, 1_000, 10_000, 100_000, 400_000, 1_200_000), per_snapshot=100
            ),
            *_width_ladder({'generator': 40, 'store': 20}, snapshots=100, per_snapshot=100),
        ),
        write=_storage_data,
        coefficient='eta',
    ),
}
