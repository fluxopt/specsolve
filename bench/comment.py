"""The pull request comment for a published run: each library against specsolve, on the base branch and in the PR.

    pixi run -e bench python -m bench.comment <pr-results> <base-results> [--run-url URL] [--out comment.md]

It reads whatever the two directories hold and fails on none of it. A case,
rung, sink or library that one side lacks prints `—`, and so does a number
that is not finite, and a file that does not read is named under the tables.
A change to what the benchmark measures — a new model, other rungs, a library
added or dropped — changes the comment, not the exit status.

The ratios are what carries from one run to another: two runs on the benchmark
server are two machines, but a library and specsolve measured side by side in
one run share one. So each cell is the committed ratio, then this run's.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import NamedTuple

from bench import report
from bench import results as bench_results

#: Where the workflow finds its own comment again, so a second run edits the first.
MARKER = '<!-- published-benchmark -->'

#: GitHub refuses a comment over 65,536 characters; the full tables are dropped before that.
LIMIT = 60_000

#: How far a ratio moves before the comment calls it moved. Two runs are two servers, and between two
#: published runs a ratio of two unchanged libraries moved 16.5% at the median (#2030), so a moved
#: cell is a lead to check rather than a verdict.
MOVED = 0.05

#: Rows of the moved table; the rest are counted, and the full tables hold them.
SHOWN = 15


class Side(NamedTuple):
    """One directory of results: the best timing per cell, its runs, its ceilings, and the files that did not read."""

    rows: dict[report.Key, report.Row]
    runs: list[report.Row]
    ceilings: list[report.Row]
    problems: list[str]


def read(directory: Path | None) -> Side:
    """Every result file under *directory*; a missing directory is a side with nothing in it."""
    report.CEILINGS.clear()
    timings: list[report.Row] = []
    runs: list[report.Row] = []
    problems: list[str] = []
    paths = bench_results.files(directory) if directory and directory.is_dir() else []
    for path in paths:
        try:
            run, _, one, _, _ = report.load(path)
        except Exception as e:  # any file that does not read is named in the comment, never raised
            problems.append(f'`{path.name}`: {type(e).__name__}: {e}')
            continue
        if run:
            runs.append(run)
        timings += [t for t in one if t.get('case') and t.get('size') and t.get('arm')]
    return Side(report.best(timings), runs, list(report.CEILINGS), problems)


def _wall(side: Side, key: report.Key) -> float | None:
    """The cell's wall time, or ``None`` where it is absent, zero or not finite."""
    value = (side.rows.get(key) or {}).get('wall_seconds')
    return value if isinstance(value, (int, float)) and math.isfinite(value) and value > 0 else None


def _ratio(side: Side, case: str, size: str, sink: str, arm: str) -> float | None:
    """specsolve's wall over *arm*'s, as the published tables draw it; ``None`` where either is missing."""
    ours, theirs = _wall(side, (case, size, sink, report.BASELINE)), _wall(side, (case, size, sink, arm))
    return ours / theirs if ours is not None and theirs is not None else None


def _x(ratio: float) -> str:
    """A ratio to two decimals, or to two figures below a tenth, where two decimals would print zero."""
    return f'{ratio:.2f}x' if ratio >= 0.1 else f'{ratio:.2g}x'


def _shown(side: Side, case: str, size: str, sink: str, arm: str) -> str:
    """One side's cell: the ratio, marked where its rounds spread too wide; else the budget that stopped *arm*, else `—`."""
    ratio = _ratio(side, case, size, sink, arm)
    if ratio is not None:
        noisy = any(report.suspect(side.rows.get((case, size, sink, a))) for a in (report.BASELINE, arm))
        return _x(ratio) + (report.MARK if noisy else '')
    report.CEILINGS[:] = side.ceilings
    try:
        return report.over_budget(case, size, sink, arm) or '—'
    except Exception:  # a case the ladder no longer knows has no ceiling to show
        return '—'


class Move(NamedTuple):
    """A cell whose ratio moved by `MOVED` or more between the committed run and this one."""

    case: str
    sink: str
    size: str
    arm: str
    before: float
    after: float


def _moved(before: float, after: float) -> bool:
    return abs(after / before - 1) >= MOVED


def _mark(before: float, after: float) -> str:
    """🔴 where specsolve lost ground to the library, 🟢 where it gained."""
    return '🔴' if after > before else '🟢'


def moves(pr: Side, base: Side) -> list[Move]:
    """Every cell both runs measured whose ratio moved by `MOVED` or more, the largest move first."""
    pairs = ((key, _ratio(base, *key), _ratio(pr, *key)) for key in pr.rows if key[3] != report.BASELINE)
    found = [
        Move(case, sink, size, arm, before, after)
        for (case, size, sink, arm), before, after in pairs
        if before is not None and after is not None and _moved(before, after)
    ]
    return sorted(found, key=lambda m: (-abs(math.log(m.after / m.before)), m))


def _cell(pr: Side, base: Side, case: str, size: str, sink: str, arm: str) -> str:
    """This run's cell, with the committed one beside it only where they differ enough to matter."""
    before, after = _ratio(base, case, size, sink, arm), _ratio(pr, case, size, sink, arm)
    was, now = _shown(base, case, size, sink, arm), _shown(pr, case, size, sink, arm)
    if before is not None and after is not None:
        return f'{_mark(before, after)} {was} → **{now}**' if _moved(before, after) else now
    return now if was == now else f'{was} → {now}'


def _rungs(case: str, sink: str, *sides: Side) -> list[str]:
    """Every rung either side measured, smallest model first; a rung with no width sorts last."""
    width: dict[str, float] = {}
    for side in sides:
        for (c, size, k, _), row in side.rows.items():
            if (c, k) != (case, sink):
                continue
            columns = (row.get('counts') or {}).get('columns')
            known = isinstance(columns, (int, float)) and math.isfinite(columns)
            width[size] = min(width.get(size, math.inf), columns if known else math.inf)
    return sorted(width, key=lambda s: (width[s], s))


def versus(case: str, sink: str, pr: Side, base: Side) -> str:
    """One case through one sink: each library's ratio against specsolve at every rung."""
    arms = report.arms_in(k for k in (*pr.rows, *base.rows) if k[0] == case and k[2] == sink)
    others = [a for a in arms if a != report.BASELINE]
    if not others:
        return f'**{case} — {sink}**: only specsolve was measured, so there is no ratio to show.'
    rows = (
        f'| {size} | ' + ' | '.join(_cell(pr, base, case, size, sink, a) for a in others) + ' |'
        for size in _rungs(case, sink, pr, base)
    )
    return '\n'.join(
        [f'**{case} — {sink}**', '', '| rung | ' + ' | '.join(others) + ' |', '|---|' + '---|' * len(others), *rows]
    )


def _summary(moved: list[Move]) -> str:
    """The moved cells as one table, the largest first, cut at `SHOWN` rows."""
    if not moved:
        return f'**No cell moved by {MOVED:.0%} or more** against the committed results.'
    lines = [
        f'**{len(moved)} {"cell" if len(moved) == 1 else "cells"} moved by {MOVED:.0%} or more** against the '
        'committed results: 🔴 specsolve lost ground to the library, 🟢 specsolve gained.',
        '',
        '| | model | solver | rung | library | committed | this run |',
        '|---|---|---|---|---|---|---|',
        *(
            f'| {_mark(m.before, m.after)} | {m.case} | {m.sink} | {m.size} | {m.arm} | {_x(m.before)} | **{_x(m.after)}** |'
            for m in moved[:SHOWN]
        ),
    ]
    if len(moved) > SHOWN:
        lines += ['', f'And {len(moved) - SHOWN} more, in the tables below.']
    return '\n'.join(lines)


def _listed(side: str, problems: list[str]) -> list[str]:
    """The files of one side that did not read, under a heading; nothing where every file read."""
    return [f'**{side}: files that did not read**\n\n' + '\n'.join(f'- {p}' for p in problems)] if problems else []


def render(pr_dir: Path | None, base_dir: Path | None, run_url: str = '') -> str:
    """The comment's markdown for the results in *pr_dir* against those in *base_dir*."""
    base, pr = read(base_dir), read(pr_dir)
    link = f' ([run]({run_url}))' if run_url else ''
    heading = f'{MARKER}\n## Published benchmark{link}'
    problems = [*_listed('This run', pr.problems), *_listed('Committed', base.problems)]
    if not pr.rows:
        return '\n\n'.join(
            [heading, 'This run left no results to compare. The run log says where it stopped.', *problems]
        )

    ratio = "Each number is specsolve's wall time over the library's, so below 1.00x specsolve is faster."
    gone = sorted({(c, k) for c, _, k, _ in base.rows} - {(c, k) for c, _, k, _ in pr.rows})
    missing = [f'⚠️ **Committed but not in this run:** {", ".join(f"{c} — {k}" for c, k in gone)}'] if gone else []
    legend = (
        f'A cell within {MOVED:.0%} of the committed one shows this run only. The two runs ran on two servers, '
        'so a moved cell is a lead to check, not a verdict. `—` is a cell that side did not '
        'measure, `>N s` or `>N GB` the budget that stopped the library below that rung, and '
        f'`{report.MARK}` a number whose rounds spread too wide to quote.'
    )
    machines = (
        f'This run: {report.provenance(pr.runs)}\n'
        f'Committed: {report.provenance(base.runs) if base.runs else "no committed results to compare against."}'
    )
    pairs = sorted({(c, k) for c, _, k, _ in pr.rows})
    tables = [versus(c, k, pr, base) for c, k in pairs]
    details = '\n\n'.join(['<details><summary>Every model, library and rung</summary>', legend, *tables, '</details>'])

    top = [heading, _summary(moves(pr, base)), ratio, *missing]
    text = '\n\n'.join([*top, details, machines, *problems])
    if len(text) <= LIMIT:
        return text
    cut = "The tables of every rung do not fit in a comment; the run's `published-benchmark` artifact holds them."
    return '\n\n'.join([*top, cut, machines, *problems])[:LIMIT]


def main(argv: list[str] | None = None) -> int:
    """Write the comment for a published run; exit 0 whatever the results hold."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('pr', type=Path, help="this run's results directory")
    ap.add_argument('base', type=Path, help="the base branch's committed results directory")
    ap.add_argument('--run-url', default='', help='the workflow run, linked from the heading')
    ap.add_argument('--out', type=Path, help='write here instead of printing')
    opts = ap.parse_args(argv)
    text = render(opts.pr, opts.base, opts.run_url)
    if opts.out:
        opts.out.write_text(text + '\n')
    else:
        print(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
