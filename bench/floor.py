"""The speed-of-light floor: one model, hand-written into a populated HiGHS.

    pixi run -e bench python -m bench.floor l
    pixi run -e bench python -m bench.floor xs --check

``transport`` built straight from the case's cached parquet into numpy arrays
and a CSR matrix, with no specsolve and no polars expression engine in the
path, ending at a populated ``highspy.Highs`` with ``run()`` never called. The
published floor is the ``highspy-matrix`` arm; this one tiles its CSR directly
rather than through ``scipy.kron``, to show where a floor's own time goes. It
is not an arm.

Phases print as minima over ``--rounds``, after one untimed warmup round that
pays the imports. The peak RSS is the process high-water mark over every round,
warmup included.

The columns are read in file order, as ``_transport_data`` writes them;
``--check`` fails on a permuted file.
"""

from __future__ import annotations

import argparse
import resource
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from bench import cases as bench_cases

CASE = 'transport'

#: The relative gap ``--check`` accepts between the floor's objective and
#: specsolve's — the parity gate's own tolerance (``bench/conftest.py``).
CHECK_RTOL = 1e-9


@dataclass(frozen=True)
class Raw:
    """The case's parquet, as numpy arrays in file order."""

    p_max: np.ndarray
    cost: np.ndarray
    cap: np.ndarray
    neg_cap: np.ndarray
    load: np.ndarray
    gen_bus: np.ndarray
    line_from: np.ndarray
    line_to: np.ndarray
    n_snap: int
    n_bus: int


@dataclass(frozen=True)
class Floor:
    """The model exactly as ``Highs.addCols``/``addRows`` take it.

    Columns are ``p`` snapshot-major then ``f`` snapshot-major; rows are the
    balance, snapshot-major with buses within — the same order the load table
    carries its values in, so ``rhs`` is that column verbatim.
    """

    cost: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    rhs: np.ndarray
    starts: np.ndarray
    cols: np.ndarray
    vals: np.ndarray

    @property
    def column_count(self) -> int:
        return len(self.cost)

    @property
    def row_count(self) -> int:
        return len(self.rhs)

    @property
    def nonzeros(self) -> int:
        return len(self.vals)


def read(paths: dict[str, str]) -> Raw:
    """The parquet into numpy, bus labels resolved to positions in the bus table's order."""
    import polars as pl

    def column(name: str, field: str = 'value') -> Any:
        return pl.read_parquet(paths[name])[field]

    buses = {b: i for i, b in enumerate(column('bus', 'bus').to_list())}
    positions = np.vectorize(buses.__getitem__, otypes=[np.int64])
    return Raw(
        p_max=column('p_max').to_numpy(),
        cost=column('cost').to_numpy(),
        cap=column('cap').to_numpy(),
        neg_cap=column('neg_cap').to_numpy(),
        load=column('load').to_numpy(),
        gen_bus=positions(column('gen_bus', 'bus').to_numpy()),
        line_from=positions(column('line_from', 'bus').to_numpy()),
        line_to=positions(column('line_to', 'bus').to_numpy()),
        n_snap=len(column('snapshot', 'snapshot')),
        n_bus=len(buses),
    )


def arrays(raw: Raw) -> Floor:
    """Cost, bounds and the CSR balance matrix, by tiling one snapshot.

    The pattern is built and sorted by bus once, then broadcast: a ``p`` entry
    advances by ``n_gen`` per snapshot, an ``f`` entry by ``n_line``.
    """
    n_gen, n_line, n_snap = len(raw.p_max), len(raw.cap), raw.n_snap
    f_block = n_snap * n_gen

    entry_bus = np.concatenate([raw.gen_bus, raw.line_to, raw.line_from])
    base_col = np.concatenate([np.arange(n_gen), f_block + np.arange(n_line), f_block + np.arange(n_line)])
    stride = np.concatenate([np.full(n_gen, n_gen), np.full(2 * n_line, n_line)])
    value = np.concatenate([np.ones(n_gen), np.ones(n_line), -np.ones(n_line)])

    order = np.argsort(entry_bus, kind='stable')
    per_bus = np.bincount(entry_bus, minlength=raw.n_bus)
    row_nnz = np.tile(per_bus, n_snap)
    shifted = base_col[order][None, :] + np.arange(n_snap)[:, None] * stride[order][None, :]

    return Floor(
        cost=np.concatenate([np.tile(raw.cost, n_snap), np.zeros(n_snap * n_line)]),
        lb=np.concatenate([np.zeros(n_snap * n_gen), np.tile(raw.neg_cap, n_snap)]),
        ub=np.concatenate([np.tile(raw.p_max, n_snap), np.tile(raw.cap, n_snap)]),
        rhs=raw.load,
        starts=np.concatenate(([0], np.cumsum(row_nnz)[:-1])).astype(np.int32),
        cols=shifted.ravel().astype(np.int32),
        vals=np.tile(value[order], n_snap),
    )


def handoff(model: Floor) -> Any:
    """A populated ``highspy.Highs``, ``run()`` never called, in one ``addCols`` and one ``addRows``."""
    import highspy

    h = highspy.Highs()
    h.setOptionValue('output_flag', False)
    empty_i = np.empty(0, dtype=np.int32)
    empty_f = np.empty(0, dtype=np.float64)
    h.addCols(model.column_count, model.cost, model.lb, model.ub, 0, empty_i, empty_i, empty_f)
    h.addRows(model.row_count, model.rhs, model.rhs, model.nonzeros, model.starts, model.cols, model.vals)
    return h


def check() -> tuple[float, float]:
    """Solve the ``xs`` rung both ways and return (floor, specsolve) objectives."""
    from bench.arms import solved

    case = bench_cases.CASES[CASE]
    rung = case.shape('xs')
    paths = case.data(rung)
    h = handoff(arrays(read(paths)))
    h.run()
    return float(h.getInfo().objective_function_value), solved('specsolve', CASE, rung.label, paths, {})


def _maxrss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if sys.platform == 'darwin' else peak / 1e3


def _line(label: str, times: list[float]) -> None:
    low = min(times)
    spread = (max(times) / low - 1) * 100
    print(f'  {label:28} {low * 1000:8.1f} ms   median {statistics.median(times) * 1000:7.1f}   spread {spread:4.1f}%')


def main(argv: list[str] | None = None) -> int:
    case = bench_cases.CASES[CASE]
    parser = argparse.ArgumentParser(prog='python -m bench.floor', description=__doc__)
    parser.add_argument('size', choices=[s.label for s in case.ladder])
    parser.add_argument('--rounds', type=int, default=9, help='builds; the minimum per phase is reported')
    parser.add_argument('--check', action='store_true', help='also solve the xs rung both ways and compare')
    args = parser.parse_args(argv)

    shape = case.shape(args.size)
    paths = case.data(shape)

    handoff(arrays(read(paths)))

    spent: dict[str, list[float]] = {'read': [], 'arrays': [], 'handoff': []}
    totals: list[float] = []
    model = None
    for _ in range(args.rounds):
        started = time.perf_counter()
        raw = read(paths)
        spent['read'].append(time.perf_counter() - started)
        model = arrays(raw)
        spent['arrays'].append(time.perf_counter() - started - spent['read'][-1])
        handoff(model)
        totals.append(time.perf_counter() - started)
        spent['handoff'].append(totals[-1] - spent['read'][-1] - spent['arrays'][-1])

    assert model is not None, '--rounds must be at least 1'
    print(f'\n{CASE}/{args.size} floor: {args.rounds} rounds, minimum reported')
    print(f'  {model.column_count} columns, {model.row_count} rows, {model.nonzeros} nonzeros\n')
    for phase, times in spent.items():
        _line(phase, times)
    _line('total', totals)
    print(f'\n  ru_maxrss {_maxrss_mb():.0f} MB (process peak over all rounds)')

    if args.check:
        ours, specsolve = check()
        gap = abs(ours - specsolve) / max(abs(specsolve), 1e-12)
        verdict = 'agree' if gap <= CHECK_RTOL else 'DISAGREE'
        print(f'\n  check (xs): floor {ours!r}, specsolve {specsolve!r} — {verdict} ({gap:.1e} relative)')
        if gap > CHECK_RTOL:
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
