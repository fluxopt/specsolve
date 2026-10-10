"""A process that measures cells for one checkout, one request per line on stdin.

    python -m bench.ab.worker --tree <checkout> [--op solve] [--focus module:Class.method]

Each request is a JSON line ``{"cell": <id>, "fingerprint": <bool>}``, and each
answer one JSON line: the seconds, the peak resident memory of the process, the
model's size and, where asked, what the build or the solve produced, to compare
across the two checkouts. The first request for a cell builds it once off the
clock, so every timed build is a warm one.

Runs under whichever ``specsolve`` is first on ``PYTHONPATH``, which is how
[`bench.ab.compare`][] points it at one checkout or the other, and refuses to
start where that is not the checkout named by ``--tree``.

``--focus`` puts the clock on one function instead of the whole build: the
seconds spent inside it, summed over its calls and counting nested calls once.
It is patched where it is defined, so a caller that imported the name before
the patch is not seen. Work a function only plans, a ``LazyFrame`` it returns,
is counted where the plan is collected, not inside it.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib
import json
import resource
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import specsolve as sps
from bench.ab.grid import Cell, cell


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--tree', type=Path, required=True)
    parser.add_argument('--op', choices=('build', 'solve'), default='build')
    parser.add_argument('--focus')
    args = parser.parse_args()

    loaded = Path(sps.__file__).resolve()
    if not loaded.is_relative_to(args.tree.resolve()):
        raise SystemExit(f'specsolve was imported from {loaded}, not from the checkout under test {args.tree}')

    run = _build if args.op == 'build' else _solve
    clock = _Clock(args.focus) if args.focus else None
    warm: set[str] = set()
    for line in sys.stdin:
        request = json.loads(line)
        try:
            record = _measure(cell(request['cell']), run, clock, warm, fingerprint=request['fingerprint'])
        except Exception as e:  # one cell that fails must not end the run of every cell after it
            record = {'error': f'{type(e).__name__}: {e}'.splitlines()[0]}
        print(json.dumps(record), flush=True)


def _measure(target: Cell, run: Any, clock: _Clock | None, warm: set[str], *, fingerprint: bool) -> dict[str, Any]:
    """One timed run of *target*, after a run off the clock the first time this process sees it."""
    if target.id not in warm:
        run(target)
        warm.add(target.id)
    if clock:
        clock.reset()
    start = time.perf_counter()
    out = run(target)
    seconds = time.perf_counter() - start
    return {
        'seconds': clock.seconds if clock else seconds,
        'calls': clock.calls if clock else None,
        'total_seconds': seconds,
        'peak_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        **out.sizes,
        'fingerprint': out.fingerprint() if fingerprint else None,
    }


class _Clock:
    """The seconds spent inside the function *dotted* names (``module:Qual.name``), over every call."""

    def __init__(self, dotted: str) -> None:
        module_name, _, qualname = dotted.partition(':')
        owner: Any = importlib.import_module(module_name)
        *path, name = qualname.split('.')
        for part in path:
            owner = getattr(owner, part)
        raw = vars(owner).get(name) if isinstance(owner, type) else getattr(owner, name, None)
        if raw is None:
            raise SystemExit(f'{dotted}: no such function in this checkout')
        func = raw.__func__ if isinstance(raw, (staticmethod, classmethod)) else raw
        self.depth = 0
        self.reset()

        @functools.wraps(func)
        def timed(*a: Any, **kw: Any) -> Any:
            self.depth += 1
            start = time.perf_counter()
            try:
                return func(*a, **kw)
            finally:
                self.depth -= 1
                if self.depth == 0:
                    self.seconds += time.perf_counter() - start
                    self.calls += 1

        setattr(owner, name, type(raw)(timed) if isinstance(raw, (staticmethod, classmethod)) else timed)

    def reset(self) -> None:
        self.seconds = 0.0
        self.calls = 0


class _Built:
    """A built model's size, and the sha256 of the LP file it writes as its fingerprint."""

    def __init__(self, model: sps.Model) -> None:
        self.model = model
        diagnostics = model.diagnostics()
        self.sizes = {'columns': diagnostics.columns, 'rows': diagnostics.rows}

    def fingerprint(self) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.lp'
            self.model.write(path)
            return hashlib.sha256(path.read_bytes()).hexdigest()


class _Solved:
    """A solve's status and objective, to nine significant digits, as its fingerprint."""

    def __init__(self, result: sps.Result) -> None:
        self.result = result
        self.sizes = {'columns': None, 'rows': None}

    def fingerprint(self) -> str:
        return f'{self.result.status} {self.result.objective:.9g}'


def _build(target: Cell) -> _Built:
    return _Built(sps.build(target.spec, target.sources))


def _solve(target: Cell) -> _Solved:
    return _Solved(sps.solve(target.spec, target.sources, 'highs'))


if __name__ == '__main__':
    main()
