"""pytest-benchmark JSON in, flat measurement records out.

The record shape `bench/report.py` and `bench/plot.py` read:

    {'record': 'timing', 'case', 'size', 'arm', 'sink', 'phase',
     'wall_seconds', 'fastest_seconds', 'q1_seconds', 'q3_seconds', 'iqr', 'median', 'rounds',
     'peak_rss_bytes', 'peak_bytes', 'allocations',
     'counts': {...}, 'live_fraction'}
    {'record': 'loop',   'case', 'size', 'arm',
     'first_build_seconds', 'steady_build_seconds'}
    {'record': 'run',    'platform', 'machine', 'cpu', 'cores', 'python', 'versions', 'commits'}
    {'record': 'ceiling','case', 'size', 'sink', 'arm', 'ladder', 'budget', 'memory_budget',
     'stopped_by', 'reason'}

``wall_seconds`` is pytest-benchmark's ``median``, because the rounds per cell
differ and the tail belongs to the machine; ``fastest_seconds`` is the minimum.
``q1_seconds`` and ``q3_seconds`` are the middle half of the same rounds, drawn
as the chart's band. ``iqr``, ``median`` and ``rounds`` let the report flag a
doubtful number (#797). ``peak_rss_bytes`` is the minimum of pytest-benchmem's
``rss_bytes`` series under ``benchmem(isolate=True)``. ``first`` and ``steady``
are round 0 and the minimum of the rest.

``cpu`` and ``cores`` name the machine, because the sinks measure in separate
jobs (#1315) on a mixed runner pool.

A library the time budget stopped leaves no benchmark entry, so its ceiling
comes back from a `.ceilings.json` sidecar as a `ceiling` record. The readers
render an ``error`` record, but nothing produces one: under pytest a dead pass
is an error, not a result.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from bench.cases import CASES

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from pathlib import Path

#: pytest-benchmem's blob inside pytest-benchmark's ``extra_info``.
BENCHMEM = 'benchmem'


def _nominal(case: str | None, size: str | None) -> int | None:
    """The rung's declared width, looked up from (case, rung) rather than read from the run."""
    try:
        return CASES[case].shape(size).nominal_variables  # pyrefly: ignore[bad-argument-type]
    except (KeyError, TypeError):
        return None


def _commit(info: dict[str, Any]) -> str | None:
    head = (info.get('id') or '')[:7]
    if not head:
        return None
    return f'{head}-dirty' if info.get('dirty') else head


def _phase(name: str, params: dict[str, Any]) -> str:
    """Which rung a timing record came off: `test_emit` and both `test_window` changes measure the same cell (#1617)."""
    if not name.startswith('test_window'):
        return 'emit'
    return 'window-reshaped' if params.get('change') == 'shape' else 'window'


def _benchmem(extra: dict[str, Any], field: str) -> float | None:
    """One pytest-benchmem series, reduced to its minimum across repeats. ``None`` without `isolate=True`."""
    series = (extra.get(BENCHMEM) or {}).get(field)
    return min(float(v) for v in series) if series else None


def _counts(extra: dict[str, Any]) -> dict[str, Any]:
    return {k: extra.get(k) for k in ('columns', 'rows', 'nonzeros')}


def records(path: Path) -> Iterator[dict[str, Any]]:
    """Every measurement in *path*, in the shape the report and the plot read.

    ``.jsonl`` is one record per line, already in this shape, and is read
    verbatim. `.json` is pytest-benchmark's document. ``nominal_variables`` is
    the x of every scaling table; ``columns`` is what survived the mask.
    """
    if path.suffix == '.jsonl':
        for line in path.read_text().splitlines():
            if line.strip():
                yield json.loads(line)
        return

    ceilings = path.with_suffix('.ceilings.json')
    if ceilings.exists():
        yield from json.loads(ceilings.read_text())

    doc = json.loads(path.read_text())
    machine, commit = doc.get('machine_info', {}), doc.get('commit_info', {})
    yield {
        'record': 'run',
        'platform': machine.get('system', '') + ' ' + machine.get('release', ''),
        'machine': machine.get('machine'),
        'cpu': (machine.get('cpu') or {}).get('brand_raw') or machine.get('processor'),
        'cores': (machine.get('cpu') or {}).get('count'),
        'python': machine.get('python_version'),
        'versions': machine.get('versions', {}),
        'commits': {'specsolve': _commit(commit)},
    }

    for b in doc.get('benchmarks', []):
        params, extra, stats = b.get('params') or {}, b.get('extra_info') or {}, b.get('stats') or {}
        common = {
            'case': params.get('case_name'),
            'size': params.get('size'),
            'arm': params.get('arm'),
            'nominal_variables': _nominal(params.get('case_name'), params.get('size')),
        }
        if b['name'].startswith('test_rebuild'):
            series = stats.get('data') or []
            yield {
                **common,
                'record': 'loop',
                'first_build_seconds': series[0] if series else None,
                'steady_build_seconds': min(series[1:]) if len(series) > 1 else None,
                'counts': _counts(extra),
            }
            continue
        yield {
            **common,
            'record': 'timing',
            'phase': _phase(b['name'], params),
            'sink': params.get('sink'),
            'wall_seconds': stats.get('median'),
            'fastest_seconds': stats.get('min'),
            'q1_seconds': stats.get('q1'),
            'q3_seconds': stats.get('q3'),
            'iqr': stats.get('iqr'),
            'median': stats.get('median'),
            'rounds': stats.get('rounds'),
            'peak_rss_bytes': _benchmem(extra, 'rss_bytes'),
            'peak_bytes': _benchmem(extra, 'peak_bytes'),
            'allocations': _benchmem(extra, 'allocations'),
            'counts': _counts(extra),
            'live_fraction': extra.get('live_fraction'),
        }


def bound_label(ceiling: Mapping[str, Any]) -> str:
    """What a cell above *ceiling* prints — the budget that stopped the climb, in its own unit.

    A sidecar with no `stopped_by` falls back to seconds.
    """
    if ceiling.get('stopped_by') == 'memory':
        return f'>{ceiling["memory_budget"]:g} GB'
    return f'>{ceiling["budget"]:g} s'


def files(target: Path) -> list[Path]:
    """Every result file under *target*, or *target* itself when it is one.

    ``.jsonl`` first, as the older measurements. ``.ceilings.json`` sidecars and
    the watchdog's ``casualties.json`` are lists, not run documents, and are
    left out.
    """
    if target.is_dir():
        found = sorted(target.glob('*.jsonl')) + sorted(target.glob('*.json'))
        return [p for p in found if not p.name.endswith(('.ceilings.json', 'casualties.json'))]
    return [target]


def load(*paths: Path) -> list[dict[str, Any]]:
    """Flatten several result files, newest last — the readers take as many as given."""
    return [r for p in paths for f in files(p) for r in records(f)]
