"""The pull request comment for a published run: specsolve's time and peak, committed against this run.

    pixi run -e bench python -m bench.comment <pr-results> <base-results> [--run-url URL] [--out comment.md]

It reads whatever the two directories hold and fails on none of it. A case,
rung or sink that one side lacks prints `—`, and so does a number that is not
finite, and a file that does not read is named under the tables. A change to
what the benchmark measures — a new model, other rungs, a library added or
dropped — changes the comment, not the exit status.

Only specsolve's own cells are compared. The other libraries do not change with
a pull request, so what they moved between the two runs is the difference
between the two servers, and the comment gives it one sentence.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from bench import report
from bench import results as bench_results

if TYPE_CHECKING:
    from collections.abc import Callable

#: Where the workflow finds its own comment again, so a second run edits the first.
MARKER = '<!-- published-benchmark -->'

#: GitHub refuses a comment over 65,536 characters; the full tables are dropped before that.
LIMIT = 60_000

#: How far specsolve's time or peak moves before the comment calls it moved. Between the
#: committed run and #2022's (#2033), the other libraries' wall times moved by 0.92x to 1.05x in
#: the middle 80% of cells, so a 5% move is a lead to check against that sentence, not a verdict.
MOVED = 0.05

#: Rows of the moved table; the rest are counted, and the full tables hold them.
SHOWN = 15

#: specsolve's own arms, which a pull request can change; every other arm says how far the servers differ.
OURS = ('specsolve', 'specsolve-in-memory')


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


def _number(side: Side, key: report.Key, field: str) -> float | None:
    """The cell's *field*, or ``None`` where it is absent, zero or not finite."""
    value = (side.rows.get(key) or {}).get(field)
    return value if isinstance(value, (int, float)) and math.isfinite(value) and value > 0 else None


def _seconds(value: float) -> str:
    """A wall time in milliseconds below a second, else in seconds."""
    return f'{value * 1000:.3g} ms' if value < 1 else f'{value:.3g} s'


def _gb(value: float) -> str:
    return f'{value / 1e9:.3g} GB'


#: The two numbers compared per cell, each with how it prints.
MEASURES = (('wall_seconds', _seconds), ('peak_rss_bytes', _gb))


def _x(factor: float) -> str:
    return f'{factor:.2f}x'


def _moved(before: float, after: float) -> bool:
    return abs(after / before - 1) >= MOVED


def _mark(before: float, after: float) -> str:
    """🔴 where specsolve took more time or memory than the committed run, 🟢 where it took less."""
    return '🔴' if after > before else '🟢'


def _missing(side: Side, key: report.Key) -> str:
    """Why a side has no number for a cell: the budget that stopped specsolve below it, else `—`."""
    report.CEILINGS[:] = side.ceilings
    try:
        return report.over_budget(*key) or '—'
    except Exception:  # a case the ladder no longer knows has no ceiling to show
        return '—'


def _cell(pr: Side, base: Side, key: report.Key, field: str, shown: Callable[[float], str]) -> str:
    """Committed, then this run, then the factor between them; a moved cell carries its mark."""
    before, after = _number(base, key, field), _number(pr, key, field)
    if before is None or after is None:
        was = shown(before) if before is not None else _missing(base, key)
        now = shown(after) if after is not None else _missing(pr, key)
        return f'{was} → {now}'
    noisy = field == 'wall_seconds' and any(report.suspect(s.rows.get(key)) for s in (pr, base))
    text = f'{shown(before)} → {shown(after)} ({_x(after / before)}){report.MARK if noisy else ""}'
    return f'{_mark(before, after)} {text}' if _moved(before, after) else text


class Move(NamedTuple):
    """A specsolve cell whose wall time or peak moved by `MOVED` or more between the committed run and this one."""

    size: float
    key: report.Key
    noisy: bool


def moves(pr: Side, base: Side) -> list[Move]:
    """Every specsolve cell both runs measured that moved by `MOVED` or more, the largest move first."""
    found = []
    for key in pr.rows:
        if key[3] != report.BASELINE:
            continue
        factors = [
            after / before
            for field, _ in MEASURES
            if (before := _number(base, key, field)) is not None and (after := _number(pr, key, field)) is not None
        ]
        if any(abs(f - 1) >= MOVED for f in factors):
            noisy = any(report.suspect(s.rows.get(key)) for s in (pr, base))
            found.append(Move(max(abs(math.log(f)) for f in factors), key, noisy))
    return sorted(found, key=lambda m: (-m.size, m.key))


def servers(pr: Side, base: Side) -> str:
    """One sentence on how far the libraries a pull request does not change moved between the two runs."""
    factors = sorted(
        after / before
        for key in pr.rows
        if key[3] not in OURS
        and (before := _number(base, key, 'wall_seconds')) is not None
        and (after := _number(pr, key, 'wall_seconds')) is not None
    )
    if not factors:
        return 'No other library ran on both sides, so nothing here says how far the two servers differ.'
    middle = factors[len(factors) // 10 : len(factors) - len(factors) // 10] or factors
    return (
        f'The other libraries, which this pull request does not change, took {_x(statistics.median(factors))} '
        f'their committed time at the median cell, and {_x(middle[0])} to {_x(middle[-1])} in the middle '
        f'{len(middle)} of {len(factors)} cells. That is how far the two servers differ.'
    )


def _variables(pr: Side, base: Side, key: report.Key) -> str:
    """The columns the model built at this cell, from whichever side has a count."""
    for side in (pr, base):
        columns = ((side.rows.get(key) or {}).get('counts') or {}).get('columns')
        if isinstance(columns, (int, float)) and math.isfinite(columns):
            return report._si(columns)
    return '—'


def _row(pr: Side, base: Side, key: report.Key, *lead: str) -> str:
    cells = [*lead, _variables(pr, base, key), *(_cell(pr, base, key, f, shown) for f, shown in MEASURES)]
    return '| ' + ' | '.join(cells) + ' |'


_COLUMNS = ['variables', 'wall: committed → this run', 'peak: committed → this run']


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


def table(case: str, sink: str, pr: Side, base: Side) -> str:
    """One case through one sink: specsolve at every rung, committed against this run."""
    rows = (_row(pr, base, (case, size, sink, report.BASELINE), size) for size in _rungs(case, sink, pr, base))
    head = ['rung', *_COLUMNS]
    return '\n'.join([f'**{case} — {sink}**', '', '| ' + ' | '.join(head) + ' |', '|---' * len(head) + '|', *rows])


def _summary(pr: Side, base: Side, moved: list[Move]) -> str:
    """The moved cells whose rounds can be quoted, as one table, the largest first, cut at `SHOWN` rows."""
    clean = [m for m in moved if not m.noisy]
    noisy = len(moved) - len(clean)
    if not moved:
        return f'**No specsolve cell moved by {MOVED:.0%} or more** in wall time or peak against the committed run.'
    lines = [
        f'**{len(moved)} specsolve {"cell" if len(moved) == 1 else "cells"} moved by {MOVED:.0%} or more** in wall '
        'time or peak: 🟢 this run took less than the committed one, 🔴 more.'
    ]
    if clean:
        head = ['model', 'solver', 'rung', *_COLUMNS]
        lines += [
            '',
            '| ' + ' | '.join(head) + ' |',
            '|---' * len(head) + '|',
            *(_row(pr, base, m.key, m.key[0], m.key[2], m.key[1]) for m in clean[:SHOWN]),
        ]
    rest = [
        *([f'{len(clean) - SHOWN} more'] if len(clean) > SHOWN else []),
        *([f'{noisy} whose rounds spread too wide to quote, marked `{report.MARK}`'] if noisy else []),
    ]
    if rest:
        lines += ['', f'The tables below hold {", and ".join(rest)}.']
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
    if not any(k[3] == report.BASELINE for k in pr.rows):
        return '\n\n'.join(
            [heading, 'This run left no specsolve results to compare. The run log says where it stopped.', *problems]
        )

    gone = sorted({(c, k) for c, _, k, _ in base.rows} - {(c, k) for c, _, k, _ in pr.rows})
    missing = [f'⚠️ **Committed but not in this run:** {", ".join(f"{c} — {k}" for c, k in gone)}'] if gone else []
    legend = (
        f'A factor below 1.00x is less time or memory than the committed run. A cell that moved by {MOVED:.0%} '
        'or more carries 🟢 or 🔴. `—` is a cell that side did not measure, `>N s` or `>N GB` the budget that '
        f'stopped specsolve below that rung, and `{report.MARK}` a wall time whose rounds spread too wide to quote.'
    )
    machines = (
        f'This run: {report.provenance(pr.runs)}\n'
        f'Committed: {report.provenance(base.runs) if base.runs else "no committed results to compare against."}'
    )
    pairs = sorted({(c, k) for c, _, k, a in pr.rows if a == report.BASELINE})
    tables = [table(c, k, pr, base) for c, k in pairs]
    details = '\n\n'.join(
        ['<details><summary>specsolve at every model and rung</summary>', legend, *tables, '</details>']
    )

    top = [heading, _summary(pr, base, moves(pr, base)), servers(pr, base), *missing]
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
