"""The PyPSA ladder pages: one per rung, in the gallery's shape, from what the parity runner committed.

    pixi run python -m tools.ladder           # rewrite docs/examples/pypsa_ladder.md and docs/examples/pypsa_ladder/*.md
    pixi run python -m tools.ladder --check   # fail if any has drifted

A rung's page is its projected spec as math, then `specsolve` beside `PyPSA`,
then the tables the rung is the first to declare and the comparison verdict.
Every fence is a committed file under ``differential/pypsa/`` or a slice of
``prep.py``. Nothing here runs pypsa.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml
from mathspec import SymbolTable, to_markdown

ROOT = Path(__file__).resolve().parent.parent
LADDER = ROOT / 'differential' / 'pypsa'
RUNGS = LADDER / 'rungs'
PAGES = ROOT / 'docs' / 'examples' / 'pypsa_ladder'
INDEX = ROOT / 'docs' / 'examples' / 'pypsa_ladder.md'
CORPUS_SITE = 'https://mathspec.readthedocs.io/en/latest/examples/'
CORPUS_PAGE = f'{CORPUS_SITE}pypsa/'

#: Where each rung sits on mathspec's PyPSA pages. The anchors are that site's
#: headings, which name a rung in mathspec's words rather than this ladder's;
#: a secant-loss rung sits under the heading of the tangent rung it follows.
CORPUS_RUNGS = {
    1: 'pypsa/#rung-1--transport',
    2: 'pypsa/#rung-2--storage',
    3: 'pypsa/#rung-3--expansion',
    4: 'pypsa/#rung-4--ramps',
    5: 'pypsa/#rung-5--global-constraints',
    6: 'pypsa/#rung-6--kvl',
    7: 'pypsa/#rung-7--commitment',
    8: 'pypsa/#rung-8--modular-and-big-m',
    9: 'pypsa/#rung-9--multi-link',
    10: 'pypsa/#rung-10--quadratic-costs',
    11: 'pypsa/#rung-11--ac-dc-meshed',
    12: 'pypsa_linearized_uc/#rung-12--linearized-unit-commitment',
    13: 'pypsa/#rung-13--transmission-losses',
    14: 'pypsa/#rung-14--two-stage-stochastic',
    15: 'pypsa/#rung-15--investment-periods',
    16: 'pypsa/#rung-16--link-delay',
    17: 'pypsa/#rung-17--process',
    18: 'pypsa/#rung-18--transformer',
    19: 'pypsa/#rung-13--transmission-losses',
    20: 'pypsa/#rung-20--phase-shifter',
    21: 'pypsa/#rung-21--carrier-growth',
    22: 'pypsa/#rung-22--transformer-losses',
    23: 'pypsa/#rung-22--transformer-losses',
    24: 'pypsa/#rung-24--must-stay-down',
    25: 'pypsa/#rung-25--committable-links',
    26: 'pypsa/#rung-26--committable-processes',
    27: 'pypsa/#rung-27--modular-ramps',
    28: 'pypsa/#rung-28--a-start-up-ramp-alone',
    29: 'pypsa/#rung-29--storage-per-investment-period',
    30: 'pypsa/#rung-30--security-constrained',
    32: 'pypsa/#rung-32--storage-that-stands-in-one-period-only',
    33: 'pypsa/#rung-33--maintenance',
    34: 'pypsa/#rung-34--maintenance-of-committable-units',
    35: 'pypsa/#rung-35--a-global-constraint-for-one-investment-period',
    36: 'pypsa/#rung-36--quadratic-costs-on-a-process-and-on-storage',
    37: 'pypsa/#rung-37--storage-dispatch-pinned-to-a-schedule',
    38: 'pypsa/#rung-38--delays-per-investment-period',
    39: 'pypsa/#rung-39--a-negative-relative-growth',
    40: 'pypsa/#rung-40--a-global-constraint-per-scenario',
    41: 'pypsa/#rung-41--operating-data-per-scenario',
    42: 'pypsa/#rung-42--first-stage-data-per-scenario',
    43: 'pypsa/#rung-43--a-components-sign',
    44: 'pypsa_linearized_uc/#rung-44--the-integer-files-commitment-rows-relaxed',
    45: 'pypsa/#rung-45--ramp-limits-per-snapshot',
    46: 'pypsa/#rung-46--the-output-brought-in',
    47: 'pypsa_linearized_uc/#rung-47--ramps-and-signs-as-the-integer-file-states-them',
    48: 'pypsa/#rung-48--a-start-and-a-stop-unweighted',
    49: 'pypsa/#rung-49--a-growth-limit-in-one-period',
    50: 'pypsa/#rung-50--a-load-that-is-not-active',
    51: 'pypsa/#rung-51--a-growth-limit-after-an-asset-retires',
    52: 'pypsa/#rung-52--a-transmission-cost-limit-per-scenario',
    53: 'pypsa/#rung-53--a-transmission-volume-limit-per-scenario-and-period',
    54: 'pypsa/#rung-54--a-delay-per-scenario',
    55: 'pypsa/#rung-55--a-transformer-cycle-per-scenario',
    56: 'pypsa/#rung-56--a-security-constrained-run-per-scenario',
    57: 'pypsa/#rung-57--a-fixed-build-per-scenario',
    58: 'pypsa/#rung-58--a-committable-unit-per-scenario',
    60: 'pypsa/#rung-60--efficiencies-per-snapshot',
}


def stems() -> list[str]:
    return sorted(path.stem for path in RUNGS.glob('rung_*.yaml') if not path.stem.endswith('.symbols'))


def _indent(text: str) -> str:
    return '\n'.join(f'    {line}' if line else '' for line in text.splitlines())


def _title(stem: str) -> str:
    """The rung's own first docstring line — ``Rung 2: storage — …``."""
    first = next(line for line in (RUNGS / f'{stem}.py').read_text().splitlines() if line.startswith('"""'))
    return first.strip('"').rstrip('.')


def prep_slice(declared: list[str]) -> str:
    """The lines of ``prep.sources`` that make *declared*, with the helpers they call — the rung's prep."""
    text = (LADDER / 'prep.py').read_text()
    body = text[text.index('def sources(') :]
    entries = dict(re.findall(r"^        '(\w+)': (.+?),\n(?=        '|    \})", body, flags=re.DOTALL | re.MULTILINE))
    tail = re.findall(r"^    tables\['(\w+)'\] = (.+)$", body, flags=re.MULTILINE)
    entries.update(tail)
    helpers = {
        m.group(1): m.group(0).rstrip()
        for m in re.finditer(r'^def (_\w+)\(.*?(?=^def |\Z)', text, flags=re.DOTALL | re.MULTILINE)
    }
    lines = [f'    {name!r}: {entries[name]},' for name in declared if name in entries]
    used = sorted({h for h in helpers if any(f'{h}(' in line for line in lines)})
    closure = set(used)
    for h in used:
        closure |= {g for g in helpers if f'{g}(' in helpers[h]}
    public = sorted(
        name for name in ('relation', 'static', 'varying', 'weighting') if any(f'{name}(' in line for line in lines)
    )
    imported = f'from differential.pypsa.prep import {", ".join(public)}\n\n\n' if public else ''
    return (
        imported
        + '\n\n\n'.join(helpers[h] for h in sorted(closure))
        + ('\n\n\n' if closure else '')
        + 'n = build()  # the network from the PyPSA tab\n\nsources = {\n'
        + '\n'.join(lines)
        + '\n}'
    )


def specsolve_tab(stem: str, projection: dict, record: dict) -> str:
    declared = [*projection['dimensions'], *projection.get('relations', {}), *projection['parameters']]
    spec = (RUNGS / f'{stem}.yaml').read_text().rstrip()
    prep = prep_slice(declared)
    call = (
        f'{prep}\n\n'
        f"with sps.solve('differential/pypsa/rungs/{stem}.yaml', sources) as solution:\n"
        f'    solution.objective  # {record["parity"]["specsolve_objective"]!r}'
    )
    return (
        '=== "specsolve"\n\n'
        f'{_indent(f"The spec, `differential/pypsa/rungs/{stem}.yaml` — the file projected onto what this rung builds:")}\n\n'
        f'{_indent(f"```yaml{chr(10)}{spec}{chr(10)}```")}\n\n'
        f'{_indent("The prep — every table the spec declares, from the network — and the solve:")}\n\n'
        f'{_indent(f"```python{chr(10)}{call}{chr(10)}```")}\n'
    )


def pypsa_tab(stem: str, record: dict) -> str:
    script = (RUNGS / f'{stem}.py').read_text().rstrip()
    said = (
        f'wrong in pypsa 1.3.0, PyPSA/PyPSA#{record["parity"]["diverges"]}'
        if 'diverges' in record['parity']
        else repr(record['pypsa_objective'])
    )
    solve = f"n = build()\nn.optimize(solver_name='highs')\nn.objective  # {said}"
    return (
        '=== "PyPSA"\n\n'
        f'{_indent(f"The network, `{stem}.py` in the corpus — the spine plus what this rung adds:")}\n\n'
        f'{_indent(f"```python{chr(10)}{script}{chr(10)}```")}\n\n'
        f'{_indent(f"```python{chr(10)}{solve}{chr(10)}```")}\n'
    )


def _tables(stem: str) -> str:
    files = sorted((LADDER / 'tables' / stem).glob('*.csv'))
    if not files:
        return (
            'Every table this spec declares was first declared by a lower rung; its values here are in the prep above.'
        )
    return f'The tables this rung is the first to declare ({len(files)}), as the prep produced them:\n\n' + '\n\n'.join(
        f'`{p.name}`\n\n```csv\n{p.read_text().rstrip()}\n```' for p in files
    )


def _verdict(record: dict) -> str:
    parity, structural = record['parity'], record['structural']
    if 'diverges' in parity:
        return (
            f'> {"✔" if parity["matches"] else "✘"} Verified against pypsa 1.3.0 — objective'
            f' **{parity["specsolve_objective"]}**, held to the corpus oracle, because PyPSA 1.3.0 solves this network'
            f' wrongly ({_issue(parity)}); no model is compared.'
        )
    priced = _cell_duals(parity)
    proof = (
        f'**model for model**: {len(structural["equal"])} blocks equal, {len(structural["region"])} documented splits'
        + (f', {len(structural["recorded"])} recorded deviations' if structural.get('recorded') else '')
        if 'equal' in structural
        else f'objective only — the linopy oracle (`tests/linopy_lane`) stops at `{structural["error"]}`'
    )
    shape = parity['structure']['per_name']
    differing = parity['structure']['differences']
    rows = '\n'.join(
        f'| `{name}` | {c["pypsa"]} | {"≠ " if name in differing else ""}{_counts(c["specsolve"])} |'
        for name, c in shape['rows'].items()
    )
    columns = '\n'.join(
        f'| `{name}` | {c["pypsa"]} | {"≠ " if name in differing else ""}{_counts(c["specsolve"])} |'
        for name, c in shape['columns'].items()
    )
    return (
        f'> {"✔" if parity["matches"] else "✘"} Verified against pypsa 1.3.0 — objective **{parity["specsolve_objective"]}**'
        f' on both sides; structure {_cell_structure(parity)}; size {_cell_size(parity)}; duals {priced}; {proof}.\n\n'
        '<details markdown="1">\n<summary>Rows and columns, PyPSA against specsolve, name for name</summary>\n\n'
        f'| row | PyPSA | specsolve |\n| --- | ---: | ---: |\n{rows}\n\n'
        f'| column | PyPSA | specsolve |\n| --- | ---: | ---: |\n{columns}\n\n</details>'
    )


def _symbols(spec: str, projection: dict) -> SymbolTable | None:
    """The file's symbol table cut to what the projection declares — a table naming a dropped name is refused."""
    path = RUNGS / f'{Path(spec).stem}.symbols.yaml'
    if not path.exists():
        return None
    raw = yaml.safe_load(path.read_text())
    declared = {
        *projection['dimensions'],
        *projection.get('relations', {}),
        *projection['parameters'],
        *projection['variables'],
    }
    return SymbolTable.load(
        {
            'notation': raw['notation'],
            'dimensions': {d: v for d, v in raw.get('dimensions', {}).items() if d in projection['dimensions']},
            'names': {n: v for n, v in raw.get('names', {}).items() if n in declared},
        }
    )


def page(stem: str, record: dict) -> str:
    projection = yaml.safe_load((RUNGS / f'{stem}.yaml').read_text())
    math = to_markdown(str(RUNGS / f'{stem}.yaml'), symbols=_symbols(record['parity']['spec'], projection), legend=True)
    number = int(stem[5:7])
    return (
        f'# {_title(stem)}\n\n'
        f'<!-- generated by tools/ladder.py from differential/pypsa — do not edit -->\n\n'
        f'One rung of [the PyPSA corpus]({CORPUS_SITE}{CORPUS_RUNGS[number]}): the file `pypsa.yaml` projected onto what'
        f' this network builds, attached to that network, and held to what PyPSA solves it to.\n\n'
        f'{_verdict(record)}\n\n'
        f'## The model\n\n<details markdown="1">\n<summary>The same model, as math</summary>\n\n{math}\n</details>\n\n'
        f'{specsolve_tab(stem, projection, record)}\n'
        f'{pypsa_tab(stem, record)}\n'
        f'## The data\n\n{_tables(stem)}\n'
    )


def _short(stem: str) -> str:
    """``Rung 2 — storage``: the number and the one word before the dash."""
    number, _, rest = _title(stem).partition(': ')
    return f'{number} — {rest.split(" — ")[0]}'


def _issue(parity: dict) -> str:
    return f'[PyPSA/PyPSA#{parity["diverges"]}](https://github.com/PyPSA/PyPSA/issues/{parity["diverges"]})'


def _cell_objective(parity: dict) -> str:
    return f'{"✔" if parity["matches"] else "✘"} `{parity["specsolve_objective"]}`'


def _cell_duals(parity: dict) -> str:
    duals = parity['duals']
    if 'diverges' in parity:
        return '—'
    if not duals['compared']:
        return '— integer model, no duals'
    negated = f', {len(duals["negated"])} negated' if duals['negated'] else ''
    if not duals['differences']:
        return f'✔ {duals["compared"]} rows{negated}'
    reasons = '; '.join(
        f'`{n}` off by {d["max_abs_diff"]} — {d["reason"] or "UNEXPLAINED"}' for n, d in duals['differences'].items()
    )
    return f'≠ {duals["compared"]} rows{negated}, {reasons}'


def _counts(blocks: dict[str, int]) -> str:
    """A name's built blocks as the pages print them — ``3+1+4``, one figure per block, never a sum."""
    return '+'.join(str(count) for count in blocks.values()) or '0'


def _cell_size(parity: dict) -> str:
    if 'diverges' in parity:
        return '—'
    theirs, ours = parity['structure']['solver']['pypsa'], parity['structure']['solver']['specsolve']
    return ' · '.join(
        f'✔ {theirs[k]} {k}' if theirs[k] == ours[k] else f'≠ {theirs[k]} vs {ours[k]} {k}'
        for k in ('rows', 'columns', 'nonzeros')
    )


def _cell_structure(parity: dict) -> str:
    if 'diverges' in parity:
        return f'— PyPSA 1.3.0 solves this network wrongly, {_issue(parity)}; the objective is held to the oracle'
    shape = parity['structure']
    names = {n: d for n, d in shape['differences'].items() if d.get('kind') != 'solver'}
    if not names:
        return f'✔ {len(shape["per_name"]["rows"])} constraints · {len(shape["per_name"]["columns"])} variables, name for name'
    reasons = '; '.join(
        f'`{n}` {d["pypsa"]} vs {_counts(d["specsolve"])} — {d["reason"] or "UNEXPLAINED"}' for n, d in names.items()
    )
    return f'≠ {reasons}'


def _deviations(stamped: dict) -> str:
    seen: dict[str, tuple[str, list[str]]] = {}
    for stem in stems():
        for kind in ('structure', 'duals'):
            for name, d in stamped[stem]['parity'][kind]['differences'].items():
                seen.setdefault(f'{name} ({kind})', (d['reason'] or 'UNEXPLAINED', []))[1].append(_short(stem))
        for name, reason in stamped[stem]['parity']['duals']['negated'].items():
            seen.setdefault(f'{name} (duals, negated)', (reason, []))[1].append(_short(stem))
        for name, reason in stamped[stem]['structural'].get('recorded', {}).items():
            seen.setdefault(f'{name} (linopy lane)', (reason, []))[1].append(_short(stem))
    if not seen:
        return 'None recorded.'
    rows = '\n'.join(f'| `{n}` | {r} | {", ".join(dict.fromkeys(rungs))} |' for n, (r, rungs) in sorted(seen.items()))
    return f'| PyPSA name (comparison) | why specsolve differs | on rungs |\n| --- | --- | --- |\n{rows}'


def _cell_lane(structural: dict) -> str:
    if 'equal' in structural:
        split = f', {len(structural["region"])} split' if structural.get('region') else ''
        recorded = f' · ≠ {len(structural["recorded"])} recorded' if structural.get('recorded') else ''
        return f'✔ {len(structural["equal"])} equal{split}{recorded}'
    return f'◌ cannot build yet: `{structural["error"].split(":")[0]}`'


def _row(stem: str, record: dict) -> str:
    parity = record['parity']
    if 'unattached' in parity:
        number = int(stem[5:7])
        return (
            f'| Rung {number} — {stem[8:].replace("_", " ")} | ◌ prep cannot prepare `{parity["spec"]}` yet:'
            f' `{parity["unattached"]}` | | | | |'
        )
    return (
        f'| [{_short(stem)}](pypsa_ladder/{stem}.md) | {_cell_objective(parity)} | {_cell_structure(parity)} |'
        f' {_cell_size(parity)} | {_cell_duals(parity)} | {_cell_lane(record["structural"])} |'
    )


def index(stamped: dict) -> str:
    rows = '\n'.join(_row(stem, stamped[stem]) for stem in sorted(stamped))
    return (
        '# The PyPSA ladder\n\n'
        '<!-- generated by tools/ladder.py from differential/pypsa — do not edit -->\n\n'
        f'[mathspec states PyPSA in one file]({CORPUS_PAGE}), grown a rung at a time. Each rung here is that file'
        ' projected onto what its network builds, shown as specsolve builds it beside the PyPSA code that builds the'
        " same network, and compared with PyPSA four ways. The `PyPSA parity` workflow regenerates every page's"
        ' sources from the pinned mathspec on each run and fails on a diff.\n\n'
        '**objective** — one number, both solves · **structure** — the same constraint and variable names, one block each · **size** — the same solver rows, columns and nonzeros'
        " · **duals** — every constraint's dual, per row · **linopy lane** — the two linopy models, label for label."
        ' ✔ identical · ≠ differs, with the recorded reason · ◌ not comparable yet.\n\n'
        '| rung | objective | structure | size | duals | linopy lane |\n| --- | --- | --- | --- | --- | --- |\n'
        f'{rows}\n\n'
        '## The four comparisons\n\n'
        "Both sides start from one object, the network the rung's script builds. PyPSA solves it directly;"
        ' specsolve solves the file attached to the tables `prep.py` makes of it.\n\n'
        '| column | specsolve | PyPSA | identical means |\n'
        '| --- | --- | --- | --- |\n'
        '| **objective** | `result.objective` | `n.objective + n.objective_constant` | equal, relative 1e-9 |\n'
        '| **structure** | `len(result.activity(block))`, `len(result.primal(variable))` | rows and columns of'
        ' `n.model` per name, masked labels excluded | one block per PyPSA name, equal count — a split counts as a'
        ' difference |\n'
        '| **size** | `diagnostics()` rows, columns, nonzeros | `n.model.solver_model` rows, columns, nonzeros |'
        ' the model handed to HiGHS is the same size on both sides |\n'
        "| **duals** | `result.dual(block)` | `n.model.constraints[name].dual` | every row's dual equal, absolute"
        ' 1e-6 — against the negative where the file writes the row negated; an integer model has none |\n'
        '| **linopy lane** | the test oracle, `tests/linopy_lane` | `n.optimize.create_model()` | label for label:'
        ' coefficients, sense, right-hand side, bounds, integrality, objective terms |\n\n'
        "Both sides solve one object, the network the rung's script builds — PyPSA directly, specsolve through the"
        " rung's own spec attached to the tables `prep.py` makes of it."
        ' A difference in structure, duals or the linopy lane is allowed only'
        ' with a reason in `differential/pypsa/deviations.yaml`; the runner fails on one recorded nowhere and on a reason'
        ' no rung needs. A rung the linopy lane cannot build yet names the blocker instead. Not compared: primals'
        ' (an optimum need not be unique).\n\n'
        'One kind of reason is checked rather than excused. Where the file states a row as PyPSA writes it negated —'
        ' a storage balance with the charge on the left, a ramp written the other way about — the dual is the'
        " negative of PyPSA's, exactly, so the runner compares it against the negative at the same tolerance and the"
        ' claim is under test. Those names are marked `negated` below; a difference surviving the negation is red.\n\n'
        f'## Recorded deviations\n\n{_deviations(stamped)}\n\n'
        'Not compared, deliberately: primals — an optimum need not be unique. Counted rather than compared: the rows built per block, on'
        " each rung's page, and over the whole ladder that every block is built by some rung, every mask is"
        ' partially true somewhere and every parameter is fed somewhere. The runner fails on a gap not listed in'
        ' `differential/pypsa/untested.txt`, and on a listed one a rung now closes; the rungs that would close them'
        ' are [mathspec#830](https://github.com/energy-models/mathspec/issues/830).\n\n'
        "Each rung's own model is the file cut to what the rung builds. The runner builds the whole file on the"
        " rung's network, cuts it to the rows and columns that build produced, and proves the cut builds the same"
        ' model, coefficient for coefficient, before anything solves; every comparison above solves the cut.\n'
    )


def rendered() -> dict[Path, str]:
    stamped = json.loads((LADDER / 'references.json').read_text())
    for s in stems():
        stamped[s]['pypsa_objective'] = stamped[s]['parity']['specsolve_objective']
    return {INDEX: index(stamped), **{PAGES / f'{s}.md': page(s, stamped[s]) for s in stems() if s in stamped}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='fail if a committed page has drifted')
    args = ap.parse_args(argv)
    stale = []
    for path, text in rendered().items():
        if args.check:
            if not path.exists() or path.read_text() != text:
                stale.append(path.relative_to(ROOT))
        else:
            path.parent.mkdir(exist_ok=True)
            path.write_text(text)
    if stale:
        print(f'stale: {", ".join(map(str, stale))} — pixi run python -m tools.ladder', file=sys.stderr)
        return 1
    if not args.check:
        print(f'wrote {INDEX.relative_to(ROOT)} and {len(stems())} rung pages')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
