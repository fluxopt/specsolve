"""Is the new code faster than the old? An A/B of two git refs over every cell of the grid.

    pixi run -e bench python -m bench.ab.compare <base-ref> <head-ref> [--sizes tiny s below above]
        [--rounds 6] [--op build|solve] [--focus module:Class.method] [--memory] [-k <substring>] [--out ab.md]

Each ref is checked out with ``git worktree add --detach``; nothing is
installed. Each round starts one worker process per side, with that checkout's
``src`` first on ``PYTHONPATH``, and measures every cell in it, warm. The sides
alternate ABBA so a drift in the machine falls on both. With ``--memory`` each
measurement is a fresh process instead, so its peak resident memory is the
cell's own. The first round also fingerprints what each side produced: a digest
of the built model to the last bit, or a solve's status and objective. Two different
fingerprints fail the cell however fast it is.

A cell's verdict is a two-sided sign test at 5% over its paired rounds: head
is *faster* or *slower* where enough rounds agree, which takes at least six
rounds, and *no change* otherwise. The change printed is the median of the
paired ratios. The exit status is 1 where any cell is slower, differs or fails.

The table goes to stdout and ``--out`` in markdown, ready for a PR's
``<details>``. Measure on an idle machine: the load averages before and after
are printed in its header.
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

import polars as pl
import pytest

from bench.ab.grid import SIZES, cell, ids
from bench.conftest import BENCH_LOCK, refuse_unless_idle, take_lock

if TYPE_CHECKING:
    from collections.abc import Iterator

ROOT = Path(__file__).resolve().parents[2]

#: The two-sided significance of the sign test a verdict rests on.
ALPHA = 0.05


@dataclass
class Row:
    """One cell's measurements on both sides, and what they say."""

    cell: str
    base: list[dict[str, Any]] = field(default_factory=list)
    head: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    @property
    def ratios(self) -> list[float]:
        return [h['seconds'] / b['seconds'] for b, h in zip(self.base, self.head, strict=True) if b['seconds'] > 0]

    @property
    def verdict(self) -> str:
        if self.error:
            return 'error'
        if self.base[0]['fingerprint'] != self.head[0]['fingerprint']:
            return 'differs'
        if self.base[0]['calls'] == 0 and self.head[0]['calls'] == 0:
            return 'not reached'
        return verdict(self.ratios)


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

    def __init__(self, tree: Path, *, op: str, focus: str | None) -> None:
        command = [sys.executable, '-m', 'bench.ab.worker', '--tree', str(tree), '--op', op]
        if focus:
            command += ['--focus', focus]
        env = {**os.environ, 'PYTHONPATH': os.pathsep.join([str(tree / 'src'), str(ROOT)])}
        self.stderr = tempfile.TemporaryFile(mode='w+')  # noqa: SIM115 — closed in close(), with the process
        self.process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr, text=True
        )

    def ask(self, cell_id: str, *, fingerprint: bool) -> dict[str, Any]:
        """One measurement of *cell_id*; raises with the cell's error, or the worker's last words where it died."""
        assert self.process.stdin and self.process.stdout
        try:
            self.process.stdin.write(json.dumps({'cell': cell_id, 'fingerprint': fingerprint}) + '\n')
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
    a fresh process instead, so that its peak is the cell's own.
    """
    trees = {'base': base, 'head': head}
    rows = {c: Row(c) for c in cells}
    for i in range(rounds):
        workers = {} if memory else {side: Worker(tree, op=op, focus=focus) for side, tree in trees.items()}
        try:
            for row in rows.values():
                if row.error:
                    continue
                try:
                    for side in ('base', 'head') if i % 2 == 0 else ('head', 'base'):
                        getattr(row, side).append(_ask(workers, trees, side, row.cell, op, focus, first=i == 0))
                except RuntimeError as e:
                    row.error = str(e)
                if i == rounds - 1:
                    print(f'{row.cell:40} {row.verdict}', file=sys.stderr)
        finally:
            for w in workers.values():
                w.close()
    return list(rows.values())


def _ask(
    workers: dict[str, Worker],
    trees: dict[str, Path],
    side: str,
    cell_id: str,
    op: str,
    focus: str | None,
    *,
    first: bool,
) -> dict[str, Any]:
    """One measurement on *side*: in its worker for the round, or in a process of its own where there is none.

    A worker that died on a cell is replaced, so the cells after it are still measured.
    """
    if side not in workers:
        own = Worker(trees[side], op=op, focus=focus)
        try:
            return own.ask(cell_id, fingerprint=first)
        finally:
            own.close()
    if not workers[side].alive:
        workers[side].close()
        workers[side] = Worker(trees[side], op=op, focus=focus)
    return workers[side].ask(cell_id, fingerprint=first)


def table(rows: list[Row], header: list[str], rounds: int, *, memory: bool = False) -> str:
    """The rows as markdown, with *header* lines above, and below them a count of verdicts and how many chance alone gives."""
    lines = [*header, '', '| cell | columns | rows | base ms | head ms | change | head faster in | verdict | peak MB |']
    lines.append('|---|--:|--:|--:|--:|--:|--:|---|--:|')
    for r in rows:
        if r.error:
            lines.append(f'| `{r.cell}` | | | | | | | error: {r.error} | |')
            continue
        b, h = r.base[0], r.head[0]
        change = statistics.median(r.ratios) - 1 if r.ratios else 0.0
        lines.append(
            f'| `{r.cell}` | {b["columns"] or ""} | {b["rows"] or ""} '
            f'| {statistics.median(x["seconds"] for x in r.base) * 1e3:.1f} '
            f'| {statistics.median(x["seconds"] for x in r.head) * 1e3:.1f} '
            f'| {change:+.1%} | {sum(x < 1 for x in r.ratios)}/{len(r.ratios)} | {r.verdict} '
            + (f'| {b["peak_mb"]:.0f} → {h["peak_mb"]:.0f} |' if memory else '| |')
        )
    counts = {v: sum(r.verdict == v for r in rows) for v in dict.fromkeys(r.verdict for r in rows)}
    rate = chance(rounds)
    lines += [
        '',
        ', '.join(f'{n} {v}' for v, n in counts.items())
        + f'. With nothing changed, a cell reads faster or slower {rate:.1%} of the time at {rounds} rounds: '
        f'about {rate * len(rows):.1f} of these {len(rows)}. Re-run a lone verdict with `-k` and more rounds.',
    ]
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('base')
    parser.add_argument('head')
    parser.add_argument('--sizes', nargs='+', default=['tiny', 's', 'below', 'above'], choices=list(SIZES))
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--op', choices=('build', 'solve'), default='build')
    parser.add_argument('--focus', help='module:Qual.name of one function, to time it alone')
    parser.add_argument('--memory', action='store_true', help='a fresh process per measurement, for its peak')
    parser.add_argument('-k', dest='select', help='only the cells whose id contains this')
    parser.add_argument('--out', type=Path)
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

    clock = f'`{args.focus}`, summed over its calls' if args.focus else f'`sps.{args.op}` wall time'
    header = [
        f'A/B: base `{args.base}` (`{base_sha[:8]}`) against head `{args.head}` (`{head_sha[:8]}`).',
        f'What is counted: {clock}, warm, '
        + ('a fresh process per measurement, ' if args.memory else 'a fresh process per side per round, ')
        + f'{args.rounds} rounds per side, ABBA. Times are medians; the change is the median paired ratio.',
        f'Machine: {os.cpu_count()} cores, {platform.python_version()}, polars {pl.__version__}; '
        f'load average {before[0]:.2f} before, {after[0]:.2f} after.',
    ]
    report = table(rows, header, args.rounds, memory=args.memory)
    print(report)
    if args.out:
        args.out.write_text(report + '\n')
    sys.exit(any(r.verdict in ('slower', 'differs', 'error') for r in rows))


if __name__ == '__main__':
    main()
