"""Is the new code faster than the old? An A/B of two git refs over every cell of the grid.

    pixi run -e bench python -m bench.ab.compare <base-ref> <head-ref> [--sizes tiny s below above]
        [--rounds 6] [--op build|solve] [--focus module:Class.method] [--memory] [-k <substring>] [--out ab.md]
        [--raw ab.parquet]

Each ref is checked out with ``git worktree add --detach``; nothing is
installed. Each round starts one worker process per side, with that checkout's
``src`` first on ``PYTHONPATH``, and measures every cell in it, warm. The sides
alternate ABBA so a drift in the machine falls on both. With ``--memory`` each
measurement is a fresh process that runs the operation once, and the verdict is
on its peak resident memory. The first round also saves what each side
produced: the built model's numbers, or a solve's status and objective. Where the two differ beyond
the last bits, the cell *differs* and fails however fast it is. Where floats
differ only within ``LAST_BITS``, the sums ran in another order: the cell keeps
its verdict, marked *last bits*.

A cell's verdict is a two-sided sign test at 5% over its paired rounds: head
is *faster* or *slower* where enough rounds agree, which takes at least six
rounds, and *no change* otherwise. The change printed is the median of the
paired ratios. The exit status is 1 where any cell is slower (or higher, with ``--memory``), differs or fails.

The table goes to stdout and ``--out`` in markdown, ready for a PR's
``<details>``. ``--raw`` writes every round's numbers as one long parquet
table (`raw`), from which the medians are taken. Measure on an idle machine:
the load averages before and after are printed in its header.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from math import comb
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import pytest

from bench.ab.grid import SIZES, cell, ids
from bench.conftest import BENCH_LOCK, refuse_unless_idle, take_lock

if TYPE_CHECKING:
    from collections.abc import Iterator

ROOT = Path(__file__).resolve().parents[2]

#: The two-sided significance of the sign test a verdict rests on.
ALPHA = 0.05

#: The relative difference within which two floats are one number summed in another order.
LAST_BITS = 1e-12

#: The numbers about a whole cell that [`raw`][] keeps from each measurement, beside each phase's seconds.
RAW_METRICS = ('seconds', 'total_seconds', 'peak_mb', 'calls', 'columns', 'rows')

#: The raw table's columns, so a run that measured nothing still writes a table that pools.
RAW_SCHEMA = {
    'base': pl.String,
    'head': pl.String,
    'op': pl.String,
    'cell': pl.String,
    'side': pl.String,
    'round': pl.Int64,
    'phase': pl.String,
    'metric': pl.String,
    'value': pl.Float64,
}


@dataclass
class Row:
    """One cell's measurements on both sides, and what they say."""

    cell: str
    base: list[dict[str, Any]] = field(default_factory=list)
    head: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    #: What the verdict is on: ``seconds``, or ``peak_mb`` for a claim about memory.
    metric: str = 'seconds'
    #: Whether both sides built one model: ``same``, ``last bits`` where some numbers differ within [`LAST_BITS`][], or ``different``.
    model: str = 'same'

    @property
    def ratios(self) -> list[float]:
        m = self.metric
        return [h[m] / b[m] for b, h in zip(self.base, self.head, strict=True) if b[m] > 0]

    @property
    def phases(self) -> list[str]:
        """The phases either side clocked, in the order head ran them, then any only base has."""
        seen = [*self.head[0].get('phases', {}), *self.base[0].get('phases', {})] if self.base and self.head else []
        return list(dict.fromkeys(seen))

    def phase_ratios(self, phase: str) -> list[float]:
        """Head over base for *phase*, per paired round, where both sides clocked it above zero."""
        pairs = zip(self.base, self.head, strict=True)
        return [
            h['phases'][phase] / b['phases'][phase]
            for b, h in pairs
            if b.get('phases', {}).get(phase, 0) > 0 and phase in h.get('phases', {})
        ]

    @property
    def verdict(self) -> str:
        if self.error:
            return 'error'
        if self.model == 'different':
            return 'differs'
        if self.base[0]['calls'] == 0 and self.head[0]['calls'] == 0:
            return 'not reached'
        said = verdict(self.ratios)
        return {'faster': 'lower', 'slower': 'higher'}.get(said, said) if self.metric == 'peak_mb' else said


def verdict(ratios: list[float]) -> str:
    """*faster* where head beats base in enough paired rounds to pass a two-sided sign test, *slower* the other way."""
    n = len(ratios)
    wins = sum(r < 1 for r in ratios)
    losses = sum(r > 1 for r in ratios)
    if _tail(n, max(wins, losses)) * 2 > ALPHA:
        return 'no change'
    return 'faster' if wins > losses else 'slower'


def chance(n: int) -> float:
    """How often a cell of *n* rounds reads *faster* or *slower* when nothing changed."""
    k = next((k for k in range(n + 1) if _tail(n, k) * 2 <= ALPHA), n + 1)
    return _tail(n, k) * 2 if k <= n else 0.0


def _tail(n: int, k: int) -> float:
    """The chance of *k* or more heads in *n* fair tosses."""
    return sum(comb(n, i) for i in range(k, n + 1)) / 2**n


@contextmanager
def checkouts(*refs: str) -> Iterator[list[tuple[str, Path]]]:
    """Each of *refs* as its commit and a detached worktree, removed on exit."""
    out = []
    with tempfile.TemporaryDirectory(prefix='ab-') as tmp:
        try:
            for i, ref in enumerate(refs):
                sha = _git('rev-parse', '--verify', f'{ref}^{{commit}}')
                path = Path(tmp) / f'{i}-{sha[:8]}'
                _git('worktree', 'add', '--detach', '--quiet', str(path), sha)
                out.append((sha, path))
            yield out
        finally:
            for _, path in out:
                _git('worktree', 'remove', '--force', str(path))


def _git(*args: str) -> str:
    return subprocess.run(['git', *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


class Worker:
    """A worker process measuring cells against the ``specsolve`` in *tree*, until closed."""

    def __init__(self, tree: Path, *, op: str, focus: str | None, memory: bool = False) -> None:
        command = [sys.executable, '-m', 'bench.ab.worker', '--tree', str(tree), '--op', op]
        if focus:
            command += ['--focus', focus]
        if memory:
            command += ['--memory']
        env = {**os.environ, 'PYTHONPATH': os.pathsep.join([str(tree / 'src'), str(ROOT)])}
        self.stderr = tempfile.TemporaryFile(mode='w+')  # noqa: SIM115 — closed in close(), with the process
        self.process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr, text=True
        )

    def ask(self, cell_id: str, *, fingerprint: Path | None) -> dict[str, Any]:
        """One measurement of *cell_id*; raises with the cell's error, or the worker's last words where it died."""
        assert self.process.stdin and self.process.stdout
        try:
            request = {'cell': cell_id, 'fingerprint': None if fingerprint is None else str(fingerprint)}
            self.process.stdin.write(json.dumps(request) + '\n')
            self.process.stdin.flush()
        except BrokenPipeError:
            pass
        line = self.process.stdout.readline()
        if not line:
            self.process.wait()
            self.stderr.seek(0)
            said = self.stderr.read().strip().splitlines()
            raise RuntimeError(said[-1] if said else f'the worker exited {self.process.returncode}')
        answer = json.loads(line)
        if 'error' in answer:
            raise RuntimeError(answer['error'])
        return answer

    @property
    def alive(self) -> bool:
        return self.process.poll() is None

    def close(self) -> None:
        with contextlib.suppress(BrokenPipeError):
            self.process.communicate()
        self.stderr.close()


def compare(
    cells: list[str],
    base: Path,
    head: Path,
    *,
    rounds: int,
    op: str = 'build',
    focus: str | None = None,
    memory: bool = False,
) -> list[Row]:
    """Every cell of *cells* measured *rounds* times on each side, ABBA, the first round fingerprinted.

    Each round starts one fresh worker per side and runs every cell in it, so
    what a process brings to its timings falls in one round and the rounds stay
    independent, as the sign test needs. With *memory*, every measurement gets
    a fresh process instead, which runs the cell once, and the verdict is on
    its peak resident memory.
    """
    trees = {'base': base, 'head': head}
    rows = {c: Row(c, metric='peak_mb' if memory else 'seconds') for c in cells}
    with tempfile.TemporaryDirectory(prefix='ab-numbers-') as numbers:
        _rounds(rows, trees, Path(numbers), rounds=rounds, op=op, focus=focus, memory=memory)
    return list(rows.values())


def _rounds(
    rows: dict[str, Row],
    trees: dict[str, Path],
    numbers: Path,
    *,
    rounds: int,
    op: str,
    focus: str | None,
    memory: bool,
) -> None:
    """[`compare`][]'s rounds, the first saving each side's numbers under *numbers* and judging the model by them."""
    for i in range(rounds):
        workers = {} if memory else {side: Worker(tree, op=op, focus=focus) for side, tree in trees.items()}
        try:
            for row in rows.values():
                if row.error:
                    continue
                saved = {side: numbers / f'{row.cell}-{side}.npz' for side in trees} if i == 0 else {}
                try:
                    for side in ('base', 'head') if i % 2 == 0 else ('head', 'base'):
                        getattr(row, side).append(_ask(workers, trees, side, row.cell, op, focus, saved.get(side)))
                except RuntimeError as e:
                    row.error = str(e)
                if saved and not row.error and row.base[0]['fingerprint'] != row.head[0]['fingerprint']:
                    row.model = 'last bits' if same_model(saved['base'], saved['head']) else 'different'
                if i == rounds - 1:
                    print(f'{row.cell:40} {row.verdict}', file=sys.stderr)
        finally:
            for w in workers.values():
                w.close()


def same_model(base: Path, head: Path) -> bool:
    """Whether two saved sets of numbers are one model: every name and shape alike, floats within [`LAST_BITS`][], the rest equal."""
    with np.load(base) as a, np.load(head) as b:
        if sorted(a.files) != sorted(b.files):
            return False
        for name in a.files:
            x, y = a[name], b[name]
            if x.shape != y.shape:
                return False
            if np.issubdtype(x.dtype, np.floating):
                if not np.allclose(x, y, rtol=LAST_BITS, atol=0.0, equal_nan=True):
                    return False
            elif not np.array_equal(x, y):
                return False
    return True


def _ask(
    workers: dict[str, Worker],
    trees: dict[str, Path],
    side: str,
    cell_id: str,
    op: str,
    focus: str | None,
    fingerprint: Path | None,
) -> dict[str, Any]:
    """One measurement on *side*: in its worker for the round, or in a process of its own where there is none.

    A process of its own is a memory run's, and runs the cell once. A worker
    that died on a cell is replaced, so the cells after it are still measured.
    """
    if side not in workers:
        own = Worker(trees[side], op=op, focus=focus, memory=True)
        try:
            return own.ask(cell_id, fingerprint=fingerprint)
        finally:
            own.close()
    if not workers[side].alive:
        workers[side].close()
        workers[side] = Worker(trees[side], op=op, focus=focus)
    return workers[side].ask(cell_id, fingerprint=fingerprint)


def raw(rows: list[Row], *, base: str, head: str, op: str) -> pl.DataFrame:
    """Every number each round measured, one row per number, long as `bench.tidy`'s table is.

    The medians [`table`][] prints are taken from these, so pooling two runs of
    one pair of commits is a concat. ``round`` pairs a base number with the head
    number measured beside it. ``phase`` is empty for a number about the whole
    cell. A number a measurement did not produce is an absent row, never a null.
    """
    out = []
    for r in rows:
        for side in ('base', 'head'):
            for i, x in enumerate(getattr(r, side)):
                key = {'base': base, 'head': head, 'op': op, 'cell': r.cell, 'side': side, 'round': i}
                out += [{**key, 'phase': '', 'metric': m, 'value': x[m]} for m in RAW_METRICS if x.get(m) is not None]
                out += [{**key, 'phase': p, 'metric': 'seconds', 'value': s} for p, s in x.get('phases', {}).items()]
    return pl.DataFrame(out, schema=RAW_SCHEMA)


def table(rows: list[Row], header: list[str], rounds: int, *, memory: bool = False) -> str:
    """The rows as markdown, with *header* lines above, and below them a count of verdicts and how many chance alone gives."""
    unit, scale = ('MB', 1.0) if memory else ('ms', 1e3)
    lines = [*header, '', f'| cell | columns | rows | base {unit} | head {unit} | change | head lower in | verdict |']
    lines.append('|---|--:|--:|--:|--:|--:|--:|---|')
    for r in rows:
        if r.error:
            lines.append(f'| `{r.cell}` | | | | | | | error: {r.error} |')
            continue
        b = r.base[0]
        change = statistics.median(r.ratios) - 1 if r.ratios else 0.0
        lines.append(
            f'| `{r.cell}` | {b["columns"] or ""} | {b["rows"] or ""} '
            f'| {statistics.median(x[r.metric] for x in r.base) * scale:.3g} '
            f'| {statistics.median(x[r.metric] for x in r.head) * scale:.3g} '
            f'| {change:+.1%} | {sum(x < 1 for x in r.ratios)}/{len(r.ratios)} | {r.verdict}'
            + (' · last bits |' if r.model == 'last bits' else ' |')
        )
    counts = {v: sum(r.verdict == v for r in rows) for v in dict.fromkeys(r.verdict for r in rows)}
    rate = chance(rounds)
    lines += [
        '',
        ', '.join(f'{n} {v}' for v, n in counts.items())
        + f'. With nothing changed, a cell reads {"lower or higher" if memory else "faster or slower"} {rate:.1%} of the time at {rounds} rounds: '
        f'about {rate * len(rows):.1f} of these {len(rows)}. Re-run a lone verdict with `-k` and more rounds.',
    ]
    if not memory:
        lines += phase_table(rows)
    return '\n'.join(lines)


def phase_table(rows: list[Row]) -> list[str]:
    """Each cell's seconds per phase on both sides, so a slower cell says which phase moved.

    A phase one side never clocked, as a base older than a phase has none of
    it, reads ``—`` there and takes no verdict. The verdicts here are for
    reading: the exit status is the totals'.
    """
    measured = [r for r in rows if not r.error and r.phases]
    if not measured:
        return []
    lines = [
        '',
        "Where the time went: each phase's median ms, its median paired ratio and its verdict, as above.",
        '',
        '| cell | phase | base ms | head ms | change | head lower in | verdict |',
        '|---|---|--:|--:|--:|--:|---|',
    ]
    said: list[tuple[str, int]] = []
    for r in measured:
        for phase in r.phases:
            base, head = _phase_median(r.base, phase), _phase_median(r.head, phase)
            ratios = r.phase_ratios(phase)
            if base is None or head is None or not ratios:
                lines.append(f'| `{r.cell}` | {phase} | {_ms(base)} | {_ms(head)} | | | — |')
                continue
            said.append((verdict(ratios), len(ratios)))
            lines.append(
                f'| `{r.cell}` | {phase} | {_ms(base)} | {_ms(head)} | {statistics.median(ratios) - 1:+.1%} '
                f'| {sum(x < 1 for x in ratios)}/{len(ratios)} | {said[-1][0]} |'
            )
    return lines + _chance_line(said)


def _chance_line(said: list[tuple[str, int]]) -> list[str]:
    """The phase verdicts counted, beside how many of them chance alone gives, so a lone verdict among hundreds is not read as a finding."""
    if not said:
        return []
    counts = {v: sum(s == v for s, _ in said) for v in dict.fromkeys(s for s, _ in said)}
    expected = sum(chance(n) for _, n in said)
    return [
        '',
        ', '.join(f'{n} {v}' for v, n in counts.items())
        + f'. With nothing changed, about {expected:.1f} of these {len(said)} phases read faster or slower.',
    ]


def _phase_median(runs: list[dict[str, Any]], phase: str) -> float | None:
    """The median seconds *runs* spent in *phase*, or None where none of them clocked it."""
    spent = [run['phases'][phase] for run in runs if phase in run.get('phases', {})]
    return statistics.median(spent) if spent else None


def _ms(seconds: float | None) -> str:
    return '—' if seconds is None else f'{seconds * 1e3:.3g}'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('base')
    parser.add_argument('head')
    parser.add_argument('--sizes', nargs='+', default=['tiny', 's', 'below', 'above'], choices=list(SIZES))
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--op', choices=('build', 'solve'), default='build')
    parser.add_argument('--focus', help='module:Qual.name of one function, to time it alone')
    parser.add_argument(
        '--memory', action='store_true', help='the verdict on peak memory, a fresh process and one run per measurement'
    )
    parser.add_argument('-k', dest='select', help='only the cells whose id contains this')
    parser.add_argument('--out', type=Path, help='the report as markdown, for the PR')
    parser.add_argument('--raw', type=Path, help="every round's numbers as one long parquet table")
    parser.add_argument('--i-know-another-is-running', dest='shared', action='store_true')
    args = parser.parse_args()
    if not chance(args.rounds):
        parser.error(
            f'{args.rounds} rounds can never pass the sign test at {ALPHA:.0%}, so every cell would read no change'
        )
    if not args.shared:
        try:
            take_lock(BENCH_LOCK)
            refuse_unless_idle(os.getloadavg()[0], os.cpu_count() or 1)
        except pytest.UsageError as e:
            raise SystemExit(str(e)) from None
    try:
        _run(args)
    finally:
        if not args.shared:
            BENCH_LOCK.unlink(missing_ok=True)


def _run(args: argparse.Namespace) -> None:
    selected = [c for c in ids(args.sizes) if not args.select or args.select in c]
    if args.op == 'solve':
        selected = [c for c in selected if cell(c).solvable]
    for c in selected:
        cell(c)
    before = os.getloadavg()
    with checkouts(args.base, args.head) as ((base_sha, base), (head_sha, head)):
        rows = compare(selected, base, head, rounds=args.rounds, op=args.op, focus=args.focus, memory=args.memory)
    after = os.getloadavg()

    if args.memory:
        counted = f'the peak resident memory of a fresh process that ran `sps.{args.op}` once'
    elif args.focus:
        counted = f'`{args.focus}`, summed over its calls, warm, a fresh process per side per round'
    else:
        counted = f'`sps.{args.op}` wall time, warm, a fresh process per side per round'
    header = [
        f'A/B: base `{args.base}` (`{base_sha[:8]}`) against head `{args.head}` (`{head_sha[:8]}`).',
        f'What is counted: {counted}; {args.rounds} rounds per side, ABBA. '
        'The values are medians; the change is the median paired ratio.',
        f'Machine: {os.cpu_count()} cores, {platform.python_version()}, polars {pl.__version__}; '
        f'load average {before[0]:.2f} before, {after[0]:.2f} after.',
    ]
    report = table(rows, header, args.rounds, memory=args.memory)
    print(report)
    if args.out:
        args.out.write_text(report + '\n')
    if args.raw:
        raw(rows, base=base_sha, head=head_sha, op=args.op).write_parquet(args.raw)
    sys.exit(any(r.verdict in ('slower', 'higher', 'differs', 'error') for r in rows))


if __name__ == '__main__':
    main()
