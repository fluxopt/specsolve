"""Every measurement as one long table — a row per number, and nothing wide.

    pixi run -e bench python -m bench.tidy bench/results/latest.json > measurements.csv
    pixi run -e bench python -m bench.tidy bench/results/latest.json --runs > runs.csv

Dims in columns, one metric per row; whatever reads it does its own pivot.

    run,case,size,sink,arm,phase,variables,metric,value
    latest,dispatch,l,highs,specsolve,emit,2000000,wall_seconds,0.83

A missing number is an absent row, never a null, so every value column is
complete. ``phase`` is ``emit`` for build-and-emit, ``window`` for a later
window of a rolling horizon, and ``first`` and ``steady`` for the two halves of
the rebuild loop; only here is ``window`` rendered.

The fingerprint (`--runs`) is long too: one row per fact.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bench import results as bench_results

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

#: The dims every measurement is keyed by. The rebuild loop has no `sink`, and writes an empty string.
DIMS = ('run', 'case', 'size', 'sink', 'arm', 'phase', 'variables')

#: What a `test_emit` record carries, in the order the file writes them. memray's
#: `peak_bytes` is renamed so it cannot be read as the RSS peak.
EMIT_METRICS = (
    ('wall_seconds', 'wall_seconds'),
    ('peak_rss_bytes', 'peak_rss_bytes'),
    ('peak_bytes', 'memray_peak_bytes'),
    ('allocations', 'allocations'),
    ('iqr', 'iqr_seconds'),
    ('median', 'median_seconds'),
    ('rounds', 'rounds'),
    ('live_fraction', 'live_fraction'),
)


def _row(record: dict[str, Any], run: str, phase: str, metric: str, value: Any) -> dict[str, Any] | None:
    """One long row, or None where the run produced no number for it."""
    if value is None:
        return None
    return {
        'run': run,
        'case': record.get('case') or '',
        'size': record.get('size') or '',
        'sink': record.get('sink') or '',
        'arm': record.get('arm') or '',
        'phase': phase,
        'variables': record.get('nominal_variables') or '',
        'metric': metric,
        'value': value,
    }


def _counts(record: dict[str, Any], run: str, phase: str) -> Iterator[dict[str, Any]]:
    """The model's own dims — the proof that two arms measured one model."""
    for name, value in (record.get('counts') or {}).items():
        row = _row(record, run, phase, name, value)
        if row is not None:
            yield row


def measurements(records: Iterable[dict[str, Any]], run: str) -> Iterator[dict[str, Any]]:
    """Every timing and loop record in *records*, one row per number."""
    for record in records:
        kind = record.get('record')
        if kind == 'timing':
            phase = record.get('phase', 'emit')
            for field, metric in EMIT_METRICS:
                row = _row(record, run, phase, metric, record.get(field))
                if row is not None:
                    yield row
            yield from _counts(record, run, phase)
        elif kind == 'loop':
            for field, phase in (('first_build_seconds', 'first'), ('steady_build_seconds', 'steady')):
                row = _row(record, run, phase, 'wall_seconds', record.get(field))
                if row is not None:
                    yield row
            yield from _counts(record, run, 'first')


def fingerprint(records: Iterable[dict[str, Any]], run: str) -> Iterator[dict[str, Any]]:
    """What was installed and what ran it, as ``run,key,value`` rows; an unresolved version writes no row."""
    for record in records:
        if record.get('record') != 'run':
            continue
        for key in ('platform', 'machine', 'cpu', 'cores', 'python'):
            if record.get(key):
                yield {'run': run, 'key': key, 'value': record[key]}
        for package, version in (record.get('versions') or {}).items():
            if version:
                yield {'run': run, 'key': f'version:{package}', 'value': version}
        for package, commit in (record.get('commits') or {}).items():
            if commit:
                yield {'run': run, 'key': f'commit:{package}', 'value': commit}


def write(rows: Iterable[dict[str, Any]], header: tuple[str, ...], out: Any) -> None:
    writer = csv.DictWriter(out, fieldnames=header, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='python -m bench.tidy')
    parser.add_argument(
        'files', nargs='*', type=Path, default=[Path('bench/results')], help='result files, or a directory of them'
    )
    parser.add_argument('--runs', action='store_true', help='the fingerprint table instead of the measurements')
    args = parser.parse_args(argv)

    rows: list[dict[str, Any]] = []
    for path in [f for target in args.files for f in bench_results.files(target)]:
        records = list(bench_results.records(path))
        run = path.stem
        rows += list(fingerprint(records, run) if args.runs else measurements(records, run))

    header = ('run', 'key', 'value') if args.runs else (*DIMS, 'metric', 'value')
    write(rows, header, sys.stdout)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
