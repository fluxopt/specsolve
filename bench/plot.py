"""Refresh the chart page's data.

    pixi run -e bench python -m bench.plot

Writes ``docs/about/benchmarks.json``: one row per model, sink, ladder,
rung and library, which the page's marks read by field name. The page itself is
hand-edited and this never touches it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bench import results as bench_results

#: Where the page fetches its rows from.
PAGE_DATA = Path('docs/about/benchmarks.json')

#: The rungs the page plots, per ladder and in order. The two ladders are never
#: one curve: `w10` and `s` are the same size through different shapes.
LADDERS = {'length': ('xs', 's', 'm', 'l'), 'width': ('w1', 'w10', 'w100', 'w1000')}

#: Which ladder a rung belongs to.
LADDER_OF = {rung: name for name, rungs in LADDERS.items() for rung in rungs}


def measurements() -> list[Path]:
    """Every results file under ``bench/results``, one per sink, the way `bench.report` reads them."""
    found = bench_results.files(Path('bench/results'))
    if not found:
        raise SystemExit('no results under bench/results — run the ladder first (bench/README.md)')
    return found


def _cell(record: dict[str, Any], rung: str) -> dict[str, Any]:
    """The columns that say which cell a row is, shared by a measurement and a refusal."""
    return {
        'model': record['case'],
        'sink': record.get('sink', 'lp'),
        'ladder': LADDER_OF[rung],
        'rung': rung,
        'variables': bench_results.nominal(record['case'], rung),
        'library': record['arm'],
    }


def rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per plotted cell: a measurement, or the budget that refused it.

    A measurement carries ``wall_s`` (the median the tables publish),
    ``wall_q1_s`` and ``wall_q3_s`` (the band) and ``peak_gb``; one taken
    without `isolate=True` has no peak and is dropped. A refusal carries
    ``refused``, the budget label, at every rung of its ladder past the one the
    library stopped at, and no numbers. A refusal at a size no library in its
    panel measured is left out, since the axis does not carry it.
    """
    measured = [
        {
            **_cell(r, r['size']),
            'wall_s': round(r['wall_seconds'], 4),
            'wall_q1_s': round(r.get('q1_seconds') or r['wall_seconds'], 4),
            'wall_q3_s': round(r.get('q3_seconds') or r['wall_seconds'], 4),
            'peak_gb': round(r['peak_rss_bytes'] / 1e9, 4),
        }
        for r in records
        if r.get('record') == 'timing'
        and r.get('phase', 'emit') == 'emit'
        and r.get('peak_rss_bytes') is not None
        and r['size'] in LADDER_OF
    ]
    taken = {(m['model'], m['sink'], m['library'], m['rung']) for m in measured}
    sizes = {(m['model'], m['sink'], m['rung']) for m in measured}
    refused = []
    for c in (r for r in records if r.get('record') == 'ceiling' and r['size'] in LADDER_OF):
        ladder = LADDERS[LADDER_OF[c['size']]]
        for rung in ladder[ladder.index(c['size']) + 1 :]:
            row = {**_cell(c, rung), 'refused': bench_results.bound_label(c)}
            panel = (row['model'], row['sink'])
            if (*panel, rung) in sizes and (*panel, row['library'], rung) not in taken:
                refused.append(row)
    order = {rung: i for rungs in LADDERS.values() for i, rung in enumerate(rungs)}
    return sorted(
        measured + refused, key=lambda r: (r['model'], r['sink'], r['ladder'], r['library'], order[r['rung']])
    )


def main() -> int:
    records = [r for p in measurements() for r in bench_results.records(p)]
    out = rows(records)
    if not any('wall_s' in r for r in out):
        raise SystemExit('bench/results has no plottable measurement — was it run with --benchmark-memory?')
    PAGE_DATA.write_text('[\n' + ',\n'.join(json.dumps(r) for r in out) + '\n]\n')
    print(f'{PAGE_DATA} refreshed: {len(out)} rows')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
