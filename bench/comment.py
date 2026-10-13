"""The pull request comment for a published run: each library against specsolve, on the base branch and in the PR.

    pixi run -e bench python -m bench.comment <pr-results> <base-results> [--run-url URL] [--out comment.md]

It reads whatever the two directories hold and fails on none of it. A case,
rung, sink or library that one side lacks prints `—`, and so does a number
that is not finite. A file that does not read is named under the tables, and a
table that does not render is replaced by its error. A change to what the
benchmark measures — a new model, other rungs, a library added or dropped —
changes the comment, not the exit status.

The ratios are what carries from one run to another: two runs on the benchmark
server are two machines, but a library and specsolve measured side by side in
one run share one. So each cell is the committed ratio, then this run's.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from bench import report
from bench import results as bench_results

if TYPE_CHECKING:
    from collections.abc import Callable

#: Where the workflow finds its own comment again, so a second run edits the first.
MARKER = '<!-- published-benchmark -->'

#: GitHub refuses a comment over 65,536 characters; the full tables are dropped before that.
LIMIT = 60_000


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


def _cell(side: Side, case: str, size: str, sink: str, arm: str) -> str:
    """specsolve's wall over *arm*'s, as the published tables print it; else the budget that stopped *arm*, else `—`."""
    ours, theirs = _wall(side, (case, size, sink, report.BASELINE)), _wall(side, (case, size, sink, arm))
    if ours is not None and theirs is not None:
        noisy = report.suspect(side.rows.get((case, size, sink, report.BASELINE))) or report.suspect(
            side.rows.get((case, size, sink, arm))
        )
        ratio = ours / theirs
        return (f'{ratio:.2f}x' if ratio >= 0.1 else f'{ratio:.2g}x') + (report.MARK if noisy else '')
    report.CEILINGS[:] = side.ceilings
    try:
        return report.over_budget(case, size, sink, arm) or '—'
    except Exception:  # a case the ladder no longer knows has no ceiling to show
        return '—'


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
    """One case through one sink: each library's ratio against specsolve, committed then this run."""
    arms = report.arms_in(k for k in (*pr.rows, *base.rows) if k[0] == case and k[2] == sink)
    others = [a for a in arms if a != report.BASELINE]
    if not others:
        return f'**{case} — {sink}**: only specsolve was measured, so there is no ratio to show.'
    lines = [f'**{case} — {sink}**', '', '| rung | ' + ' | '.join(others) + ' |', '|---|' + '---|' * len(others)]
    for size in _rungs(case, sink, pr, base):
        cells = (f'{_cell(base, case, size, sink, a)} → {_cell(pr, case, size, sink, a)}' for a in others)
        lines.append(f'| {size} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def _guarded(label: str, render: Callable[..., str], *args: Any) -> str:
    """*render*'s table, or its error in its place, so one table cannot take the comment down."""
    try:
        return render(*args)
    except Exception as e:  # the comment is posted whatever one table does
        return f'**{label}** did not render: `{type(e).__name__}: {e}`'


def _listed(side: str, problems: list[str]) -> list[str]:
    """The files of one side that did not read, under a heading; nothing where every file read."""
    return [f'**{side}: files that did not read**\n\n' + '\n'.join(f'- {p}' for p in problems)] if problems else []


def render(pr_dir: Path | None, base_dir: Path | None, run_url: str = '') -> str:
    """The comment's markdown for the results in *pr_dir* against those in *base_dir*."""
    base, pr = read(base_dir), read(pr_dir)
    link = f' ([run]({run_url}))' if run_url else ''
    heading = f'{MARKER}\n## Published benchmark{link}'
    problems = [*_listed('This run', pr.problems), *_listed('Committed', base.problems)]
    gone = sorted({(c, k) for c, _, k, _ in base.rows} - {(c, k) for c, _, k, _ in pr.rows})
    missing = [f'**Committed but not in this run**: {", ".join(f"{c} — {k}" for c, k in gone)}'] if gone else []
    if not pr.rows:
        return '\n\n'.join(
            [heading, 'This run left no results to compare. The run log says where it stopped.', *problems]
        )

    legend = (
        "Each cell is specsolve's wall time over the library's, so below 1.00x specsolve is faster. "
        'The first number is the committed result on the base branch, the second is this run. '
        '`—` is a cell that side did not measure, `>N s` or `>N GB` is the budget that stopped the library '
        f'below that rung, and `{report.MARK}` marks a number whose rounds spread too wide to quote.'
    )
    machines = (
        f'This run: {report.provenance(pr.runs)}\n'
        f'Committed: {report.provenance(base.runs) if base.runs else "no committed results to compare against."}'
    )
    pairs = sorted({(c, k) for c, _, k, _ in pr.rows})
    tables = [_guarded(f'{c} — {k}', versus, c, k, pr, base) for c, k in pairs]
    report.CEILINGS[:] = pr.ceilings
    full = [_guarded(f'{c} — {k}', report.table, c, pr.rows, k) for c, k in pairs]
    details = '<details><summary>Every rung of this run, wall time and peak memory</summary>\n\n' + '\n\n'.join(full)

    text = '\n\n'.join([heading, legend, machines, *tables, *missing, details + '\n\n</details>', *problems])
    if len(text) <= LIMIT:
        return text
    cut = "The full tables do not fit in a comment; they are in the run's `published-benchmark` artifact."
    return '\n\n'.join([heading, legend, machines, *tables, *missing, cut, *problems])[:LIMIT]


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
