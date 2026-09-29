"""Refresh the chart page's numbers.

    pixi run -e bench python -m bench.plot

The page is hand-edited; this rewrites only its ``const DATA = {...};`` line.
One panel per model and sink, one line per library, log on both axes. The band
around each line is that measurement's own rounds.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from bench import results as bench_results

#: What the page calls each library; anything unlisted keeps its harness name.
NAME = {'specsolve': 'polars'}

#: The rungs the page plots, per ladder and in order. The two ladders are never
#: one curve: `w10` and `s` are the same size through different shapes.
LADDERS = {'length': ('xs', 's', 'm', 'l'), 'width': ('w1', 'w10', 'w100', 'w1000')}

#: Which ladder a rung belongs to, for the filters below.
LADDER_OF = {rung: name for name, rungs in LADDERS.items() for rung in rungs}
_DATA = re.compile(r'^const DATA = .*;$', re.MULTILINE)


def measurements() -> list[Path]:
    """Every results file under ``bench/results``, one per sink, the way `bench.report` reads them."""
    found = bench_results.files(Path('bench/results'))
    if not found:
        raise SystemExit('no results under bench/results — run the ladder first (bench/README.md)')
    return found


def series(*paths: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    """``(case, sink, arm) -> rung -> what one panel line needs at that rung``.

    ``wall`` is the median the tables publish, and the band is the first to the
    third quartile. A measurement taken without `isolate=True` has no peak and
    is dropped.
    """
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in (r for p in paths for r in bench_results.records(p)):
        if record.get('record') != 'timing' or record.get('phase', 'emit') != 'emit' or 'error' in record:
            continue
        if record.get('peak_rss_bytes') is None or record['size'] not in LADDER_OF:
            continue
        key = (record['case'], record.get('sink', 'lp'), record['arm'])
        out.setdefault(key, {})[record['size']] = {
            'wall': record['wall_seconds'],
            'lo': record.get('q1_seconds') or record['wall_seconds'],
            'hi': record.get('q3_seconds') or record['wall_seconds'],
            'peak': record['peak_rss_bytes'] / 1e9,
            'vars': (record.get('counts') or {}).get('columns') or record.get('nominal_variables'),
        }
    return out


def panels(taken: dict[tuple[str, str, str], dict[str, Any]], ceilings: list[dict[str, Any]]) -> dict[str, Any]:
    """One panel per (case, sink): a shared rung axis, and a line per library.

    ``null`` marks a rung with no measurement; ``bound`` carries the time-budget
    label where a ceiling stopped the library. A library that cannot reach a
    sink is absent from the panel. A ceiling applies only to the panels of its
    own ladder.
    """
    out: dict[str, Any] = {}
    for (case, sink, arm), rungs in sorted(taken.items()):
        for ladder, order in LADDERS.items():
            reached = {r: v for r, v in rungs.items() if r in order}
            if not reached:
                continue
            panel = out.setdefault(
                f'{case} — {sink} — {ladder}',
                {'case': case, 'sink': sink, 'ladder': ladder, 'series': {}, 'rungs': []},
            )
            for rung in order:
                if rung in reached and rung not in panel['rungs']:
                    panel['rungs'].append(rung)
            panel['series'][NAME.get(arm, arm)] = {'arm': arm, 'at': reached}

    stopped = {(c['case'], c['sink'], c['arm'], LADDER_OF[c['size']]): c for c in ceilings if c['size'] in LADDER_OF}
    for panel in out.values():
        order = [r for r in LADDERS[panel['ladder']] if r in panel['rungs']]
        panel['rungs'] = order
        panel['vars'] = [next(s['at'][r]['vars'] for s in panel['series'].values() if r in s['at']) for r in order]
        for line in panel['series'].values():
            at = line.pop('at')
            ceiling = stopped.get((panel['case'], panel['sink'], line.pop('arm'), panel['ladder']))
            for key in ('wall', 'lo', 'hi', 'peak'):
                line[key] = [round(at[r][key], 4) if r in at else None for r in order]
            stops_after = order.index(ceiling['size']) if ceiling and ceiling['size'] in order else None
            over_budget = bench_results.bound_label(ceiling) if ceiling else None
            line['bound'] = [
                over_budget if stops_after is not None and i > stops_after and r not in at else None
                for i, r in enumerate(order)
            ]
    return out


def main() -> int:
    paths = measurements()
    taken = series(*paths)
    ceilings = [r for p in paths for r in bench_results.records(p) if r.get('record') == 'ceiling']
    if not taken:
        raise SystemExit(
            f'{[str(p) for p in paths]} has no plottable measurement — was it run with --benchmark-memory?'
        )
    data = {'panels': panels(taken, ceilings), 'ladders': {k: list(v) for k, v in LADDERS.items()}}

    page = Path('docs/about/benchmarks-scaling.html')
    text = page.read_text()
    if not _DATA.search(text):
        raise SystemExit(f'{page} has no `const DATA = ...;` line — keep the literal on one line of its own')
    page.write_text(_DATA.sub(lambda _: 'const DATA = ' + json.dumps(data) + ';', text, count=1))
    print(f'{page} refreshed: {len(data["panels"])} panels')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
