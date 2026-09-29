"""Turn a results JSONL into the markdown that goes in docs/about/benchmarks.md.

    pixi run -e bench python -m bench.report bench/results/latest.json

Nothing here recomputes or smooths anything. A cell whose spread exceeds
`SPREAD_BUDGET` is marked, because a run whose rounds all ran slow looks like a
clean one (#797).
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bench import results as bench_results

if TYPE_CHECKING:
    from collections.abc import Iterable

#: What every ratio is drawn against: ours ÷ theirs, so below 1.00 is a win for specsolve.
BASELINE = 'specsolve'

#: Ceilings from every results file read, in the order they were read. A cell
#: with no measurement is looked up here before it renders as absent.
CEILINGS: list[dict[str, Any]] = []


def over_budget(case: str, size: str, sink: str, arm: str) -> str | None:
    """The bound where a budget stopped this arm below this rung, else None; every rung above is covered."""
    from bench.cases import CASES

    for ceiling in CEILINGS:
        if (ceiling['case'], ceiling['sink'], ceiling['arm']) != (case, sink, arm):
            continue
        labels = [s.label for s in CASES[case].ladder]
        if size in labels and ceiling['size'] in labels and labels.index(size) > labels.index(ceiling['size']):
            return bench_results.bound_label(ceiling)
    return None


#: How far a cell's IQR may reach, as a fraction of its own median, before the
#: number is marked. It sits just above the p95 of 24.1% over a full ladder, and
#: catches the one cell verified as contaminated, at 53.8% (#797).
SPREAD_BUDGET = 0.25

#: Appended to a marked number.
MARK = '~'

_SPREAD_NOTE = (
    f'`{MARK}` marks a measurement whose rounds spread wider than '
    f'{SPREAD_BUDGET:.0%} of their own median. Every round was slow, so the '
    'minimum printed for it has no clean round behind it and may be '
    'contaminated: **do not quote a marked number, or a ratio drawn from one** '
    '— re-take the cell on an idle machine.'
)


def machine(run: Row) -> str:
    """One run's box, as the string two runs are compared by; every key is optional."""
    cores = f', {run["cores"]} cores' if run.get('cores') else ''
    where = (run.get('platform') or '?').strip() or '?'
    return f'{run.get("cpu") or "?"}{cores} ({where})'


def provenance(runs: list[Row]) -> str:
    """The line above the tables: what measured them, and on how many machines (#1315)."""
    if not runs:
        return 'No run record — provenance unknown.'
    first = runs[0]
    versions = ', '.join(f'{k} {v}' for k, v in (first.get('versions') or {}).items() if v)
    python = first.get('python') or '?'
    boxes = sorted({machine(run) for run in runs})
    line = f'{machine(first)}, python {python} — {versions}.'
    if len(boxes) == 1:
        return line
    return (
        f'python {python} — {versions}. **Taken on {len(boxes)} machines**: '
        + '; '.join(boxes)
        + '. A rung and its arms share a file and a machine, so the ratios hold; '
        'do not read a trend across rungs as one machine getting slower.'
    )


def load(
    path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    records = bench_results.load(path)
    run = next((r for r in records if r.get('record') == 'run'), {})
    gates = [r for r in records if r.get('record') == 'gate']
    timings = [r for r in records if r.get('record') == 'timing' and r.get('phase', 'emit') == 'emit']
    loop = [r for r in records if r.get('record') == 'loop']
    CEILINGS.extend(r for r in records if r.get('record') == 'ceiling')
    return run, gates, timings, loop


Row = dict[str, Any]
Key = tuple[str, str, str, str]
#: One rung's arms. A missing arm is ``None``, so a half-measured rung still renders.
Arms = dict[str, Row | None]


def _key(r: Row) -> Key:
    return (r['case'], r['size'], r.get('sink', 'lp'), r['arm'])


def best(timings: list[Row]) -> dict[Key, Row]:
    """(case, size, sink, arm) -> the run of that cell with the lowest median."""
    out: dict[Key, Row] = {}
    for r in timings:
        if 'error' in r:
            continue
        key = _key(r)
        if key not in out or r['wall_seconds'] < out[key]['wall_seconds']:
            out[key] = r
    return out


def failures(timings: list[Row]) -> dict[Key, str]:
    """A run that died is a measurement, and the report renders it as one."""
    return {_key(r): r['error'] for r in timings if 'error' in r}


def _si(n: float) -> str:
    for unit, scale in (('M', 1e6), ('k', 1e3)):
        if n >= scale:
            return f'{n / scale:.6g}{unit}'
    return f'{n:.0f}'


def _gb(n: float) -> str:
    return f'{n / 1e9:.2f}'


def _ratio(a: float | None, b: float | None) -> str:
    return f'{a / b:.2f}x' if a and b else '—'


def suspect(row: Row | None) -> bool:
    """Whether *row*'s rounds spread too wide to be quoted, as IQR over median.

    A single wild round inflates stddev, not the IQR; interference across every
    round spreads the IQR (#797). A record with no `iqr` is never marked.
    """
    if not row:
        return False
    iqr, median = row.get('iqr'), row.get('median')
    return iqr is not None and bool(median) and iqr / median > SPREAD_BUDGET


def _marked(cell: str, *, noisy: bool) -> str:
    """The cell with the noise mark, unless there is no number there to doubt."""
    return cell + MARK if noisy and cell != '—' else cell


def _note(lines: list[str], *, marked: bool) -> list[str]:
    """The rendered table, with the note under it when it marked at least one cell."""
    return [*lines, '', _SPREAD_NOTE] if marked else lines


def arms_in(keys: Iterable[Key]) -> tuple[str, ...]:
    """Whichever arms appear in *keys*, `BASELINE` first, then in order of first appearance.

    Pass the keys of one table, so a library that cannot reach its sink gets no
    column.
    """
    seen = list(dict.fromkeys(a for *_, a in keys))
    return tuple(([BASELINE] if BASELINE in seen else []) + [a for a in seen if a != BASELINE])


def _grid(
    heading: list[str],
    leading: tuple[str, ...],
    body: Iterable[tuple[list[str], Arms, tuple[str, str, str]]],
    arms: tuple[str, ...],
    *,
    ratios: bool = True,
) -> str:
    """A comparison table: cells that identify a row, then every arm and its ratio.

    Each arm's wall and peak, its ratio against `BASELINE`, and `MARK` on any
    wall cell whose rounds spread past `SPREAD_BUDGET`. A run of the baseline
    alone, or ``ratios=False``, renders no ratio columns.
    """
    against = [a for a in arms if a != BASELINE] if ratios else []
    head = [
        *leading,
        *(f'wall: {a}' for a in arms),
        *(f'wall ÷ {a}' for a in against),
        *(f'peak: {a}' for a in arms),
        *(f'peak ÷ {a}' for a in against),
    ]
    lines = [*heading, '', '| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    marked = False
    for cells, arms, cells_key in body:
        wall = {a: (r['wall_seconds'] if r else None) for a, r in arms.items()}
        peak = {a: (r['peak_rss_bytes'] if r else None) for a, r in arms.items()}
        noisy = {a: suspect(r) for a, r in arms.items()}
        over = {a: over_budget(*cells_key, a) for a in arms}
        marked = marked or any(noisy.values())
        lines.append(
            '| '
            + ' | '.join(
                [
                    *cells,
                    *(_marked(f'{wall[a]:.2f} s' if wall[a] else (over[a] or '—'), noisy=noisy[a]) for a in arms),
                    *(
                        _marked(_ratio(wall.get(BASELINE), wall[a]), noisy=noisy.get(BASELINE, False) or noisy[a])
                        for a in against
                    ),
                    *(f'{_gb(peak[a])} GB' if peak[a] else '—' for a in arms),
                    *(_ratio(peak.get(BASELINE), peak[a]) for a in against),
                ]
            )
            + ' |'
        )
    return '\n'.join(_note(lines, marked=marked))


def _at_rung(rows: dict[Key, Row], case: str, size: str, sink: str, arms: tuple[str, ...]) -> tuple[Arms, Row | None]:
    """Every arm at one rung, and any one of them to read the model's counts off; ``None`` if none ran it."""
    at_rung = {a: rows.get((case, size, sink, a)) for a in arms}
    return at_rung, next((r for r in at_rung.values() if r), None)


_DENSITY_RUNG = re.compile(r'd\d+$')
_DECLARATION_RUNG = re.compile(r'n\d+m?$')
#: Rungs that grow the model sideways — entity counts x N, snapshots fixed.
_WIDTH_RUNG = re.compile(r'w\d+$')
_SWEEPS = (_DENSITY_RUNG, _DECLARATION_RUNG, _WIDTH_RUNG)


def _rung_value(size: str) -> str:
    """The swept value a rung label carries; a trailing ``m`` prints as the masked twin."""
    count = int(size[1:].removesuffix('m'))
    return f'{count} masked' if size.endswith('m') else str(count)


def _sweep_of(size: str) -> re.Pattern[str] | None:
    """The sweep a rung label belongs to — ``None`` is the size ladder."""
    return next((p for p in _SWEEPS if p.match(size)), None)


def sizes_of(case: str, rows: dict[Key, Row], sink: str = 'lp', *, sweep: re.Pattern[str] | None = None) -> list[str]:
    """Rung labels for *case* on one axis, smallest model first, the label breaking a tie."""
    seen = {
        s: r['counts']['columns']
        for (c, s, k, _), r in rows.items()
        if c == case and k == sink and _sweep_of(s) is sweep
    }
    return sorted(seen, key=lambda s: (seen[s], s))


#: Where a reader goes to trace a curve.
_CHART_PAGE = '*The same runs with a cursor: [the chart page](benchmarks-scaling.html).*'


#: What every arm in a sink's table has ended up holding.
_SEAM = {
    'lp': 'Each arm has written the LP file, through whichever writer it has.',
    'highs': (
        'Each arm ends holding a populated `highspy.Highs` with `run()` never '
        'called — specsolve through `build_highs`. The simplex is the same work '
        'whoever filled the model, so timing it would say nothing about the '
        'lane that filled it.'
    ),
    'gurobi': (
        'Each arm ends holding a populated `gurobipy.Model` with `optimize()` '
        'never called — specsolve through `build_gurobi`, and gurobipy through '
        '`update()`, which is where its own deferred writes land. Opt-in: it '
        'needs the `[gurobi]` extra.'
    ),
}


def table(case: str, rows: dict[Key, Row], sink: str = 'lp') -> str:
    """One case's rungs through one sink, as a markdown table under a bold caption."""
    arms = arms_in(k for k in rows if k[0] == case and k[2] == sink)
    return _grid(
        [f'**{case} — {sink} sink**', '', _SEAM[sink]],
        ('variables', 'live', 'rows'),
        (
            ([_si(ref['counts']['columns']), _live(ref), _si(ref['counts']['rows'])], at_rung, (case, size, sink))
            for size in sizes_of(case, rows, sink)
            for at_rung, ref in [_at_rung(rows, case, size, sink, arms)]
            if ref
        ),
        arms,
        ratios=False,
    )


def _live(r: Row) -> str:
    """What fraction of the coordinate product survived the mask."""
    frac = r.get('live_fraction')
    return '—' if frac is None else f'{frac * 100:.0f}%'


def _settling(best: dict[tuple[str, str, str], Row], seen: list[tuple[str, str]], arms: tuple[str, ...]) -> str:
    """How far the first recorded round sits from steady state, per arm.

    Rendered from the results file (#619). Neither end carries the one-time
    import cost.
    """
    per_arm = []
    for arm in arms:
        deltas = sorted(
            (best[(case, size, arm)]['first_build_seconds'] - best[(case, size, arm)]['steady_build_seconds']) * 1000
            for case, size in seen
            if (case, size, arm) in best and best[(case, size, arm)].get('first_build_seconds') is not None
        )
        if deltas:
            per_arm.append(f'{statistics.median(deltas):+.1f} ms on {arm}')
    if not per_arm:
        return ''
    return (
        'The harness warms up before it records, so neither column carries the '
        'one-time import cost: the median gap between them is ' + ' and '.join(per_arm) + '.'
    )


def _ms(row: Row | None, half: str, *, bold: bool) -> str:
    """One half of the first-vs-steady pair, or a dash where that arm has none; the baseline's steady is bold."""
    if row is None or row.get(f'{half}_build_seconds') is None:
        return '—'
    cell = f'{row[f"{half}_build_seconds"] * 1000:.1f} ms'
    return f'**{cell}**' if bold and half == 'steady' else cell


def marginal(loop_rows: list[Row]) -> str:
    """First model in a process, against every model after it.

    No ratio columns: the libraries do not put the same work in the build. The
    sweep rungs have their own tables and are skipped.
    """
    best: dict[tuple[str, str, str], Row] = {}
    for r in loop_rows:
        if 'error' in r or r.get('steady_build_seconds') is None:
            continue
        if _sweep_of(r['size']) is not None:
            continue
        key = (r['case'], r['size'], r['arm'])
        if key not in best or r['first_build_seconds'] < best[key]['first_build_seconds']:
            best[key] = r
    if not best:
        return ''
    arms = arms_in(best)

    def order(key: tuple[str, str]) -> tuple[float, str, str]:
        """Widest model last, then by name — a total order, off whichever arm ran, since widths tie."""
        widths = (best[(*key, a)].get('nominal_variables') for a in arms if (*key, a) in best)
        return (next((w for w in widths if w is not None), 0), *key)

    seen = sorted({(c, s) for c, s, _ in best}, key=order)
    lines = [
        '### Marginal cost per model',
        '',
        'Build only, repeated in one process. **first** is the first recorded round '
        'and **steady** the best of the rounds after it, so the pair is what a '
        'rolling horizon pays for its second window against its first. ' + _settling(best, seen, arms),
        '',
        '**Read down a column, not across the row.** The build is not the same '
        'work in every library — one that defers materialising its coefficients '
        'to its writer spends almost nothing here and pays it at the seam — so '
        'these columns carry no ratios. The tables above measure to a common '
        'artifact and are where a comparison belongs.',
        '',
        '| ' + ' | '.join(['case', 'vars', *(f'{a}: {half}' for a in arms for half in ('first', 'steady'))]) + ' |',
        '|' + '---|' * (2 + 2 * len(arms)),
    ]
    for case, size in seen:
        at_rung = {a: best.get((case, size, a)) for a in arms}
        ref = next((r for r in at_rung.values() if r), None)
        if ref is None:
            continue
        lines.append(
            '| '
            + ' | '.join(
                [
                    case,
                    _si(ref.get('nominal_variables', 0)),
                    *(_ms(at_rung[a], half, bold=a == BASELINE) for a in arms for half in ('first', 'steady')),
                ]
            )
            + ' |'
        )
    return '\n'.join(lines)


def _sweep(
    rows: dict[Key, Row],
    rung: re.Pattern[str],
    heading: list[str],
    second: str,
    newest_first: bool,
    sink: str = 'lp',
) -> str:
    """A sweep table: one model size, one axis varied, every case that has it.

    *second* names the column for what the sweep varies, read off the rung
    label.
    """
    cases = [c for c in sorted({c for c, _, _, _ in rows}) if sizes_of(c, rows, sink, sweep=rung)]
    if not cases:
        return ''

    arms = arms_in(k for k in rows if rung.match(k[1]) and k[2] == sink)

    def body() -> Iterable[tuple[list[str], Arms, tuple[str, str, str]]]:
        for case in cases:
            sizes = sizes_of(case, rows, sink, sweep=rung)
            for size in reversed(sizes) if newest_first else sorted(sizes):
                at_rung, ref = _at_rung(rows, case, size, sink, arms)
                if ref:
                    label = _live(ref) if second == 'live' else _rung_value(size)
                    yield [case, label, _si(ref['counts']['columns'])], at_rung, (case, size, sink)

    return _grid(heading, ('case', second, 'variables'), body(), arms)


def density(rows: dict[Key, Row]) -> str:
    """One model size, four mask densities."""
    return _sweep(
        rows,
        _DENSITY_RUNG,
        [
            '### The mask sweep',
            '',
            'One model size, through the `lp` sink. For `nodal`, `live` is how many '
            'of the 12 technologies each node has installed: 12 / 6 / 3 / 1.',
        ],
        second='live',
        newest_first=True,
    )


def width(rows: dict[Key, Row]) -> str:
    """The same variable counts as the size ladder, reached by widening: `w10` is `s`, `w1000` is `l`."""
    return _sweep(
        rows,
        _WIDTH_RUNG,
        [
            '### The width ladder',
            '',
            'Entity counts x N with the snapshot count held fixed, through the `highs` '
            'sink. Each rung matches one of the size ladder rungs above variable for '
            'variable — `w10` is `s`, `w1000` is `l` — so the pair reads as one model '
            'at one size in two shapes.',
        ],
        second='entities x',
        newest_first=False,
        sink='highs',
    )


def declarations(rows: dict[Key, Row]) -> str:
    """One model size, several declaration counts, total variables and rows flat."""
    return _sweep(
        rows,
        _DECLARATION_RUNG,
        [
            '### The declaration sweep',
            '',
            'One model size, through the `lp` sink. A fixed pool of units split into '
            'N declarations of pool/N units each, so total variables and rows are '
            'flat and only the declaration count moves. A `masked` row is the same '
            'rung with a `where:` on every declaration — the same model, so the '
            'difference is what the mask costs per declaration.',
        ],
        second='declarations',
        newest_first=False,
    )


#: The published page, and the only file `--write` touches.
PAGE = Path('docs/about/benchmarks.md')


def _fenced(name: str) -> tuple[str, str]:
    return f'<!-- bench:{name} -->', f'<!-- bench:/{name} -->'


def splice(text: str, fragments: dict[str, str]) -> tuple[str, list[str]]:
    """Replace each fenced block in *text* with the fragment of the same name.

    A fragment the page has no fence for is skipped and named.

    Returns:
        The rewritten text, and the fragments the page had nowhere to put.

    Raises:
        SystemExit: If a fence is half-written or out of order, or a fragment
            came out empty. Each of those publishes less than the run measured.
    """
    skipped: list[str] = []
    for name, body in fragments.items():
        opening, closing = _fenced(name)
        start, end = text.find(opening), text.find(closing)
        if start < 0 and end < 0:
            skipped.append(name)
            continue
        if start < 0 or end < 0:
            raise SystemExit(f'{PAGE} has half a `{name}` fence — one of the pair is missing')
        if end < start:
            raise SystemExit(f'{PAGE} closes the {name} fence before it opens it')
        if not body.strip():
            raise SystemExit(f'{name} rendered empty from these results — refusing to blank the page')
        text = text[: start + len(opening)] + '\n\n' + body.strip() + '\n\n' + text[end:]
    return text, skipped


def main(argv: list[str] | None = None) -> int:
    """Print the published page for the given result files, or write it with ``--write``.

    Older files carry parity-gate records, which are printed; per-case tables
    are collapsed in ``<details>``.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        'results', type=Path, nargs='*', default=[Path('bench/results')], help='result files, or a directory of them'
    )
    ap.add_argument(
        '--write',
        action='store_true',
        help=f'write each table into its fence in {PAGE} instead of printing, the way bench.plot writes the chart',
    )
    opts = ap.parse_args(argv)

    runs: list[Row] = []
    gates: list[Row] = []
    timings: list[Row] = []
    loop: list[Row] = []
    for path in [f for target in opts.results for f in bench_results.files(target)]:
        one_run, one_gates, one_timings, one_loop = load(path)
        if one_run:
            runs.append(one_run)
        gates += one_gates
        timings += one_timings
        loop += one_loop
    rows = best(timings)
    failed = failures(timings)

    print(provenance(runs))
    print()
    if gates:
        print(
            'Parity gate: '
            + '; '.join(
                f'{g["case"]} objectives agree to {g["relative_gap"]:.1e} relative'
                if g['passed']
                else f'{g["case"]} FAILED'
                for g in gates
            )
            + '.'
        )
    else:
        print('Parity gate: enforced at measurement time — every arm below built the same model.')
    results = [_CHART_PAGE]
    for case in sorted({c for c, _, _, _ in rows}):
        sinks = [k for k in sorted({k for c, _, k, _ in rows if c == case}) if sizes_of(case, rows, k)]
        if not sinks:
            continue
        results.append(f'\n<details markdown="1">\n<summary><b>{case}</b> — every rung, every sink</summary>\n')
        results += [table(case, rows, sink) + '\n' for sink in sinks]
        results.append('</details>')
    for key, error in sorted(failed.items()):
        results.append(f'<!-- {" ".join(k for k in key if k)}: {error} -->')

    fragments = {
        'results': '\n'.join(results),
        'marginal': marginal(loop),
        'sweeps': '\n\n'.join(t for t in (width(rows), density(rows), declarations(rows)) if t),
    }
    fragments = {name: body for name, body in fragments.items() if body.strip()}

    if opts.write:
        written, skipped = splice(PAGE.read_text(), fragments)
        PAGE.write_text(written)
        wrote = [name for name in fragments if name not in skipped]
        print(f'{PAGE} refreshed: {", ".join(wrote) or "nothing"}')
        if skipped:
            print(f'  no fence for: {", ".join(skipped)} — rendered elsewhere, or the page does not want them')
        return 0

    for body in fragments.values():
        print()
        print(body)
    return 0


if __name__ == '__main__':
    sys.exit(main())
