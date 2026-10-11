"""The result files a published run writes, and the rungs each one holds.

    python -m bench.published <sink> <case> <rung>...

prints one line per file the case writes through that sink: the path, then its
rungs. `ladder-ci` runs each line in a process of its own, because a process
the memory watchdog kills writes no file at all, and a kill on one ladder must
not take the rungs another ladder already measured.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from bench.cases import CASES

#: How a rung label says which ladder it belongs to.
_LADDERS = (
    ('width', re.compile(r'w\d+$')),
    ('density', re.compile(r'd\d+$')),
    ('declarations', re.compile(r'n\d+$')),
)


def ladder_of(size: str) -> str:
    """Which ladder a rung belongs to — its own, or the size one; the budget decides each separately."""
    return next((name for name, pattern in _LADDERS if pattern.match(size)), 'size')


def files(sink: str, case: str, sizes: list[str]) -> dict[Path, list[str]]:
    """Each file *case* writes through *sink*, to the rungs of *sizes* it holds.

    The size ladder keeps the case's own name, ``latest-<sink>-<case>.json``;
    another ladder adds its name. A rung the case does not have is dropped,
    and a ladder left with none writes no file.
    """
    labels = {s.label for s in CASES[case].ladder}
    out: dict[Path, list[str]] = {}
    for size in sizes:
        if size in labels:
            ladder = ladder_of(size)
            stem = f'latest-{sink}-{case}' if ladder == 'size' else f'latest-{sink}-{case}-{ladder}'
            out.setdefault(Path('bench/results') / f'{stem}.json', []).append(size)
    return out


if __name__ == '__main__':
    sink, case, *sizes = sys.argv[1:]
    for path, rungs in files(sink, case, sizes).items():
        print(path, *rungs)
