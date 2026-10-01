"""The parity gate: every rung of the PyPSA corpus, as deep as the engines allow.

    python differential/pypsa/parity.py <mathspec checkout>

The corpus is mathspec's — `examples/pypsa.yaml` and its quadratic sibling,
and one `rung_*.py` per rung whose `build()` returns the PyPSA network with
its data inline. `prep.py` beside this file is the prep: a network becomes
the tables the file declares, every "data prep" parameter computed there.
This file is the rest of the engine side — prepare, build, solve, compare — and
it needs a checkout of that repository at the tag `pypsa-parity.yml` names. No
pixi environment carries pypsa, so run it locally with the workflow's own line:

    pixi exec -s uv uv run --with-editable . \
        --with "$(grep -o 'linopy @ git+[^"]*' pyproject.toml)" \
        --with "pypsa==1.3.0" --with "highspy==1.15.1" --with "polars>=1.30" \
        python differential/pypsa/parity.py ../mathspec

Per rung, from the same network:

1. **Spec against model** — PyPSA's ``n.optimize.create_model()`` and
   the oracle's ``tests.linopy_lane.build``, label for label: coefficients, sense, right-hand
   side, bounds, integrality, objective terms. No solver, so it covers MIP
   and QP alike. The verdict speaks the index table's words: ``equal`` is
   the one block PyPSA builds — **done**; ``region`` is the same rows from
   several ``where:`` blocks — **split**; a difference the file states on
   purpose carries a ``blocks`` reason in ``deviations.yaml`` and comes back
   **recorded**; ``mismatch`` fails the run. A rung whose file
   `tests.linopy_lane` cannot build yet stamps the error instead.
2. **One solved objective across the fence** — PyPSA's solve against
   `specsolve.relational`'s, both HiGHS, rtol 1e-9 on the generic spine.
3. **Coverage** — what the relational lane built per block, each
   dimension's size, the tables attached non-empty; and, over the ladder, that
   every block is built, every mask is partially true and every parameter is
   fed by some rung.
4. **Prices across the fence** — PyPSA's ``buses_t.marginal_price`` against
   the relational lane's ``Bus_nodal_balance`` duals, per unit of the
   snapshot's objective weighting, which is how PyPSA reports them. A
   mixed-integer rung has no duals on our side and stamps why instead.
5. **Structure** — PyPSA's rows and columns per name, masked labels
   excluded, against what the relational lane built per block — never
   summed. A difference is allowed only with a reason in ``deviations.yaml``;
   one recorded nowhere reds the run, and so does a reason no rung needs.

Primals are not compared — an optimum need not be unique.

The comparison canonicalises linopy's ``.flat`` export, because the two
builders lay the same model out differently. Importing `tests.linopy_lane` sets
linopy's global ``semantics`` option to ``v1`` and PyPSA speaks ``legacy``, so
the option is reset around each PyPSA build.

The stamps are rewritten into `references.json` on every run; the workflow
fails on a diff.
"""

from __future__ import annotations

import importlib
import json
import math
import re
import shutil
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import polars as pl
import polars.selectors as cs

CORPUS = Path(sys.argv[1] if len(sys.argv) > 1 else 'corpus').resolve()
RUNGS = CORPUS / 'examples' / 'references' / 'pypsa'
HERE = Path(__file__).resolve().parent
RECORDS = HERE / 'references.json'
TABLES = HERE / 'tables'
PROJECTIONS = HERE / 'rungs'
DEVIATIONS = HERE / 'deviations.yaml'
sys.path.insert(0, str(RUNGS))
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE.parents[1]))  # the repository root, for the linopy oracle in tests/

import linopy  # noqa: E402
import mathspec  # noqa: E402
import prep  # noqa: E402  the prep, beside this file
import projection  # noqa: E402
import yaml  # noqa: E402
from sweep import untested_conjuncts  # noqa: E402  the pure half, so a test needs no pypsa

import specsolve as sps  # noqa: E402
from specsolve.relational.engines.polars.predicates import masked  # noqa: E402
from specsolve.relational.engines.polars.scope import Scope  # noqa: E402
from specsolve.sources import tidy_sources  # noqa: E402


def rungs() -> list[str]:
    """Every rung, in ladder order — the scripts beside the corpus's spine."""
    return sorted(path.stem for path in RUNGS.glob('rung_*.py'))


def network(stem: str):
    """The rung's PyPSA network, built by its own script."""
    return importlib.import_module(stem).build()


def keywords(stem: str) -> dict:
    """What the rung's `n.optimize` takes beyond the solver — the script's ``OPTIMIZE``, if it names any."""
    return dict(getattr(importlib.import_module(stem), 'OPTIMIZE', {}))


def outages(stem: str) -> object:
    """The branches a security-constrained rung takes out — the script's ``BRANCH_OUTAGES``, ``None`` on a plain run."""
    return getattr(importlib.import_module(stem), 'BRANCH_OUTAGES', None)


def issue(stem: str) -> int | None:
    """The PyPSA bug a rung records — the script's ``ISSUE``, whose ``oracle()`` gives the objective PyPSA should reach."""
    return getattr(importlib.import_module(stem), 'ISSUE', None)


def solved(stem: str, n):
    """*n* solved as the rung asks — security-constrained over its outages where it names any."""
    with legacy():
        if outages(stem) is None:
            status, condition = n.optimize(solver_name='highs', **keywords(stem))
        else:
            status, condition = n.optimize.optimize_security_constrained(
                solver_name='highs', branch_outages=outages(stem), **keywords(stem)
            )
    assert status == 'ok', f'{stem}: pypsa did not solve — {status} / {condition}'
    return n


def intended(stem: str) -> float:
    """The objective a rung that records a PyPSA bug should reach — its oracle's networks, each at its weight."""
    oracle = importlib.import_module(stem).oracle()
    return sum(
        weight * (float(m.objective) + float(m.objective_constant))
        for weight, m in ((w, solved(stem, x)) for w, x in oracle)
    )


def spec_of(stem: str) -> Path:
    """The spec file the rung names: ``MODEL`` in its script where it names one, ``pypsa.yaml`` otherwise."""
    return CORPUS / 'examples' / getattr(importlib.import_module(stem), 'MODEL', 'pypsa.yaml')


def stands_for(description: str | None, model=None) -> str:
    """The PyPSA name a declaration's description opens with, in backticks — the declared pages' convention.

    A description may open with several, comma-separated, where PyPSA names
    one block differently by mode (tangent rows or stacked secants); the one
    *model* has is chosen, else the first.
    """
    opening = re.match(r'((?:`[^`]+`(?:, )?)+)', description or '').group(1)
    names = re.findall(r'`([^`]+)`', opening)
    if model is None or len(names) == 1:
        return names[0]
    present = [*model.constraints, *model.variables]
    for name in names:
        pattern = templated(name)
        if any(pattern.match(m) if pattern else m == name for m in present):
            return name
    return names[0]


def templated(name: str) -> re.Pattern | None:
    """The pattern a name with a ``{…}`` placeholder stands for — PyPSA writes one row family per label, the file one block over the dimension."""
    if not re.search(r'\{\w+\}', name):
        return None
    return re.compile('^' + re.sub(r'\\\{\w+\\\}', '([^-]+)', re.escape(name)) + '$')


def template_dims(declared, model) -> dict[str, str]:
    """Block name -> the dimension PyPSA spells into the constraint's name — the one declared dim its constraint has no axis for (a component dim rides its ``name`` axis)."""
    out = {}
    for name, block in declared.constraints.items():
        pattern = templated(stands_for(block.description, model))
        if pattern is None or pattern.groups != 1:
            continue
        their = next((c for c in model.constraints if pattern.match(c)), None)
        if their is None:
            continue
        axes = set(model.constraints[their].coords.dims)
        component = {*prep.DIM.values(), 'bus'} if 'name' in axes else set()
        (out[name],) = [
            d for d in block.dims if d not in axes and d not in component and not (PLAIN[0] and d == 'scenario')
        ]
    return out


def template_axis(block, dim: str) -> int:
    """Where *dim* sits in a row key — keys are ordered snapshot first, then by name, a plain run's scenario dropped."""
    dims = sorted((d for d in block.dims if not (PLAIN[0] and d == 'scenario')), key=lambda d: (d != 'snapshot', d))
    return dims.index(dim)


def flattened(name: str, table: object, dims: list[str]) -> object:
    """A table prep spreads over the scenarios, cut to the dims the file declares.

    The file states a value once where PyPSA carries it per scenario, so the
    scenario column goes and the rows collapse; a value that differed between
    scenarios would be a different model, and is refused.
    """
    if not isinstance(table, pl.DataFrame) or 'scenario' not in table.columns or 'scenario' in dims:
        return table
    cut = table.drop('scenario').unique()
    assert not cut.select(dims).is_duplicated().any(), (
        f'{name} differs between scenarios, and the file declares it over {dims}'
    )
    return cut


def spread(table: object, dims: list[str], scenarios: object) -> object:
    """A table PyPSA keeps once for every scenario, repeated in each where the file reads it by scenario."""
    if not isinstance(table, pl.DataFrame) or 'scenario' not in dims or 'scenario' in table.columns:
        return table
    return pl.DataFrame({'scenario': scenarios}).join(table, how='cross')


def prepared(spec: Path, n, stem: str | None = None) -> dict[str, object]:
    """`prep.sources` cut to what *spec* declares — specsolve refuses a key the spec does not take; *stem* names the rung whose `OPTIMIZE` and outages prep reads."""
    declared = mathspec.to_spec(spec)
    names = {*declared.dimensions, *declared.parameters, *declared.relations}
    dims = {name: p.dims for name, p in declared.parameters.items()} | {
        name: list(relation.key_roles) for name, relation in declared.relations.items()
    }
    tables = prep.sources(n, keywords(stem) if stem else {}, outages(stem) if stem else None)
    return {
        name: spread(flattened(name, table, dims.get(name, [])), dims.get(name, []), tables['scenario'])
        for name, table in tables.items()
        if name in names
    }


#: Spec file -> the tables some lower rung already committed; a table is written once, under the rung that first feeds it.
FIRST: dict[str, set[str]] = defaultdict(set)


def rung_spec(stem: str, spec: Path, built_rows: dict, built_columns: dict, fed: list[str], zero: set[str]) -> Path:
    """Write the rung's own spec: *spec* cut to what the full build produced, the rung's script and symbols beside it.

    *zero* names the parameters this network sets to zero everywhere, whose
    products the cut drops too, so a square no unit pays for leaves the file.
    """
    raw = yaml.safe_load(spec.read_text())
    cut = projection.project(
        raw, {'built_rows': built_rows, 'built_columns': built_columns, 'attached_nonempty': fed}, zero=zero
    )
    path = PROJECTIONS / f'{stem}.yaml'
    path.parent.mkdir(exist_ok=True)
    path.write_text(projection.dump(cut))
    shutil.copy(RUNGS / f'{stem}.py', PROJECTIONS / f'{stem}.py')
    symbols = spec.parent / 'symbols' / spec.name
    if symbols.exists():
        shutil.copy(symbols, PROJECTIONS / f'{stem}.symbols.yaml')
    return path


def built_counts(model) -> tuple[dict[str, int], dict[str, int]]:
    """Rows per constraint block and columns per variable block of one build, before any solve."""
    built = model._engine._model
    return (
        {name: held.height for name, held in built.constraints.items()},
        {name: held.height for name, held in built.variables.items()},
    )


def fed(sources: dict[str, object]) -> list[str]:
    """The tables a build was handed with at least one row, or a scalar."""
    return sorted(name for name, table in sources.items() if not hasattr(table, '__len__') or len(table))


def zero_parameters(sources: dict[str, object], declared) -> set[str]:
    """The numeric parameters this network sets to zero at every row it gives."""
    zero = set()
    for name, p in declared.parameters.items():
        table = sources.get(name)
        if getattr(p, 'dtype', None) in ('bool', 'str') or table is None:
            continue
        if isinstance(table, pl.DataFrame):
            if len(table) and table['value'].dtype.is_numeric() and (table['value'] == 0).all():
                zero.add(name)
        elif isinstance(table, (int, float)) and not isinstance(table, bool) and table == 0:
            zero.add(name)
    return zero


def _keyed_model(model) -> dict[str, object]:
    """One build as label-free sets — every column, row, coefficient and objective term keyed by its declaration and coordinate."""
    built = model._engine._model
    handoff = built.handoff

    def keys(held: dict) -> dict[int, tuple]:
        out = {}
        for name, labelled in held.items():
            frame = labelled.frame.collect()
            label = 'var_label' if 'var_label' in frame.columns else 'row'
            dims = [c for c in frame.columns if c != label]
            for row in frame.iter_rows(named=True):
                out[row[label]] = (name, *(str(row[d]) for d in dims))
        return out

    columns, rows = keys(built.variables), keys(built.constraints)
    starts = list(handoff.row_starts)
    matrix = handoff.matrix
    terms = {}
    for r, (sense, rhs) in enumerate(zip(handoff.rows['sense'], handoff.rows['rhs'], strict=True)):
        span = matrix.slice(starts[r], starts[r + 1] - starts[r])
        pairs = frozenset((columns[c], round(v, 9)) for c, v in zip(span['col'], span['coeff'], strict=True) if v)
        terms[rows[handoff.rows['row'][r]]] = (str(sense), round(float(rhs), 9), pairs)
    return {
        'columns': {
            columns[i]: (float(lb), float(ub), str(vtype)) for i, (lb, ub, vtype) in enumerate(handoff.cols.iter_rows())
        },
        'rows': terms,
        'quadratic rows': frozenset(
            (rows[r], columns[a], columns[b], round(v, 9)) for r, a, b, v in handoff.qmatrix.iter_rows() if v
        ),
        'objective': frozenset((columns[c], round(v, 9)) for c, v in handoff.obj.iter_rows() if v),
        'quadratic objective': frozenset(
            (frozenset((columns[a], columns[b])), round(v, 9)) for a, b, v in handoff.quad.iter_rows() if v
        ),
        'sense': handoff.objective_sense,
        'constant': round(handoff.objective_constant, 9),
    }


def same_model(stem: str, full, cut) -> None:
    """The rung's spec builds the model the whole file builds on the same network, coefficient for coefficient.

    This is what makes the cut a projection rather than a second model.
    """
    ours, whole = _keyed_model(cut), _keyed_model(full)
    for part, held in whole.items():
        if ours[part] == held:
            continue
        if isinstance(held, dict):
            missing = sorted(set(held) - set(ours[part]), key=str)[:3]
            extra = sorted(set(ours[part]) - set(held), key=str)[:3]
            changed = sorted((k for k in set(held) & set(ours[part]) if held[k] != ours[part][k]), key=str)[:3]
            detail = f'only in the file {missing}, only in the cut {extra}, differing {changed}'
        else:
            detail = f'file {sorted(map(str, held))[:3]} … cut {sorted(map(str, ours[part]))[:3]}'
        raise AssertionError(f'{stem}: the rung spec builds other {part} than the file — {detail}')


def committed(stem: str, spec: str, declared, sources: dict[str, object]) -> None:
    """Write the tables this rung is the first to feed as CSV, rows sorted — the tables the page shows under it.

    Written through :func:`tidy_sources`, so a file holds exactly the tidy
    frame `sps.solve` received, floats rounded to twelve places because a
    ``pow`` differs by an ulp between libms and the gate is a byte diff. Once
    per table, under the lowest rung that feeds it.
    """
    folder = TABLES / stem
    folder.mkdir(parents=True)
    for name, source in tidy_sources(declared.program, sources).items():
        frame = source.collect() if hasattr(source, 'collect') else source
        if len(frame) and name not in FIRST[spec]:
            frame.sort(frame.columns).with_columns(cs.float().round(12)).write_csv(folder / f'{name}.csv')
            FIRST[spec].add(name)


def conjunct_verdicts(built_model, program) -> dict[str, str]:
    """Per masked block, what each conjunct of its ``where:`` did on this rung — one character each.

    ``t`` it held at every coordinate, ``f`` at none, ``b`` at some of them,
    and ``-`` where the rung builds no frame for the block at all. Positional:
    :attr:`mathspec.program.Mask.conjuncts` is deterministic for a program.
    """
    model = built_model._engine._model
    scope = Scope(model.program, model.attached, model.variables)
    verdicts: dict[str, str] = {}
    for name, block in {**program.constraints, **program.variables}.items():
        where = getattr(block, 'where', None)
        if where is None:
            continue
        dims = tuple(getattr(block, 'dims', ()) or ())
        whole = masked(scope, dims, None).select(pl.len()).collect().item()
        held = [
            masked(scope, dims, mathspec.program.Mask(conjunct)).select(pl.len()).collect().item()
            for conjunct in where.conjuncts
        ]
        verdicts[name] = ''.join(
            '-' if not whole else 'f' if not count else 't' if count == whole else 'b' for count in held
        )
    return verdicts


def built_by_label(result, templates: dict[str, str]) -> dict[str, dict[str, int]]:
    """Rows built per label of a templated block's dimension — what each PyPSA row family is held to."""
    out = {}
    for name, dim in templates.items():
        counts = result.activity(name).to_pandas().groupby(dim).size()
        out[name] = {str(k): int(c) for k, c in counts.items()}
    return out


def _gc_dual(dual, label: str) -> pd.DataFrame:
    """One global constraint's dual as rows — one, or one per scenario where the network has scenarios."""
    if not dual.ndim:
        return pd.DataFrame({'name': [label], 'dual': [float(dual)]})
    return dual.to_dataframe('dual').reset_index().assign(name=label)


def duals(result, n, declared, gc_kinds: dict[str, str], reasons: dict) -> dict[str, object]:
    """Every constraint's dual, per coordinate: PyPSA's own linopy model against `result.dual`, raw on both sides.

    A block's rows are joined to the PyPSA constraint its description names
    on (snapshot, name) — PyPSA calls every component coordinate ``name`` —
    and a global-constraint block on the row's label. An integer variable
    leaves the lane without duals, and the stamp says so.

    A name the file writes as PyPSA's row negated carries a ``negated:`` reason
    in ``deviations.yaml`` and is compared against the negative. A ``duals:``
    reason excuses a difference no comparison can express.
    """
    try:
        result.dual(next(iter(declared.constraints)))
    except sps.SpecsolveError as error:
        return {
            'compared': 0,
            'skipped': str(error).splitlines()[0][:120],
            'negated': {},
            'per_name': {},
            'differences': {},
        }
    per_name: dict[str, dict] = {}
    templates = template_dims(declared, n.model)
    pairs = []
    for block_name, block in declared.constraints.items():
        stands = stands_for(block.description, n.model)
        pattern = templated(stands)
        if pattern is None:
            pairs.append((block_name, stands, None))
        else:
            pairs.extend(
                (block_name, their_name, found.group(1) if pattern.groups == 1 else None)
                for their_name in n.model.constraints
                if (found := pattern.match(their_name))
            )
    for block_name, their_name, k in pairs:
        ours = result.dual(block_name).to_pandas().rename(columns={'value': 'ours'})
        if PLAIN[0] and 'scenario' in ours.columns:
            ours = ours.drop(columns='scenario')
        if k is not None:
            dim = templates[block_name]
            ours = ours[ours[dim].astype(str) == k].drop(columns=dim)
        if ours.empty:
            continue
        if their_name[0].isupper():
            if their_name not in n.model.constraints:
                continue
            dual = n.model.constraints[their_name].dual
            theirs = dual.to_dataframe('dual').reset_index() if dual.ndim else pd.DataFrame({'dual': [float(dual)]})
            if 'timestep' in theirs.columns:
                theirs = theirs.drop(columns=['period', 'snapshot'], errors='ignore')
            theirs = theirs.rename(columns=AXES)
            theirs = theirs.rename(columns={c: 'name' for c in theirs.columns if c.endswith('_i')})
            for column in [c for c in theirs.columns if c.endswith('-outage')]:
                component = column.removesuffix('-outage')
                theirs['outage'] = [prep.outage_label(component, v) for v in theirs.pop(column)]
            ours = ours.rename(columns={c: 'name' for c in ours.columns if c in prep.DIM.values() or c == 'bus'})
        else:
            labels = [label for label, kind in gc_kinds.items() if kind == their_name]
            theirs = pd.concat(
                [_gc_dual(n.model.constraints[f'GlobalConstraint-{label}'].dual, label) for label in labels]
                or [pd.DataFrame({'name': [], 'dual': []})]
            )
            ours = ours.rename(columns={'global_constraint': 'name'})
        keys = [c for c in ours.columns if c != 'ours']
        entry = per_name.setdefault(
            their_name, {'rows': 0, 'max_abs_diff': 0.0, 'negated': bool(reasons.get(their_name, {}).get('negated'))}
        )
        missing = [k for k in keys if k not in theirs.columns]
        if missing:
            entry['note'] = f'PyPSA has no {missing} coordinate on {their_name}'
            print(f'  {block_name}: {entry["note"]} (theirs: {list(theirs.columns)})', file=sys.stderr)
            continue
        casts = {k: ours[k].dtype for k in keys if k != 'name'} | ({'name': str} if 'name' in keys else {})
        joined = ours.merge(theirs.astype(casts), on=keys, how='inner') if keys else ours.merge(theirs, how='cross')
        gaps = (joined['ours'] + joined['dual']).abs() if entry['negated'] else (joined['ours'] - joined['dual']).abs()
        if len(gaps) and gaps.max() > 1e-6:
            off = joined[gaps > 1e-6].head(3)
            print(f'  {block_name} vs {their_name}:\n{off.to_string(index=False)}', file=sys.stderr)
        entry['rows'] += len(joined)
        entry['max_abs_diff'] = round(max(entry['max_abs_diff'], float(gaps.max()) if len(gaps) else 0.0), 9)
    differences = {}
    for name, entry in per_name.items():
        entry['matches'] = entry['max_abs_diff'] <= 1e-6 and 'note' not in entry
        if not entry['matches']:
            differences[name] = {
                'max_abs_diff': entry['max_abs_diff'],
                **({'note': entry['note']} if 'note' in entry else {}),
                **({'compared': 'negated'} if entry['negated'] else {}),
                'reason': reasons.get(name, {}).get('duals'),
            }
    return {
        'compared': sum(e['rows'] for e in per_name.values()),
        'negated': {
            name: reasons[name]['negated'] for name, e in sorted(per_name.items()) if e['negated'] and e['matches']
        },
        'per_name': per_name,
        'differences': differences,
    }


@contextmanager
def legacy():
    """linopy's ``legacy`` semantics, which PyPSA speaks — a NaN coefficient is a zero, not a dropped row — for the span of one PyPSA build."""
    linopy.options['semantics'] = 'legacy'
    try:
        yield
    finally:
        linopy.options['semantics'] = 'v1'


def pypsa_model(stem: str):
    """The network's own linopy model, as PyPSA builds it."""
    with legacy():
        return network(stem).optimize.create_model(**keywords(stem))


#: Whether the rung under comparison is a plain run, whose one scenario PyPSA spells no axis for.
PLAIN = [True]


#: PyPSA's axis names for the file's dimensions — a multi-period model keys by ``(period, timestep)`` where the file has one snapshot, and the growth limit by ``Carrier`` and ``periods``.
AXES = {'timestep': 'snapshot', 'Carrier': 'carrier', 'periods': 'period', 'secant': 'segment'}


def _keyed(labels) -> dict:
    """label per coordinate key — dim names dropped, ``snapshot`` first and ``outage`` last, so the two spellings align.

    Key components are strings, because a dimension's labels can be ints on
    one side and text on the other. A dimensionless array is its one label at
    the empty key.
    """
    if not labels.ndim:
        return {(): int(labels.item())}
    series = labels.to_series()
    if 'timestep' in series.index.names:
        series = series.droplevel('period')
    if PLAIN[0] and 'scenario' in series.index.names:
        if series.index.nlevels == 1:
            return {(): int(series.iloc[0])}
        series = series.droplevel('scenario')
    for level in [name for name in series.index.names if str(name).endswith('-outage')]:
        component = level.removesuffix('-outage')
        series = series.rename(lambda v, c=component: prep.outage_label(c, v), level=level)
        series.index = series.index.set_names('outage', level=level)
    series.index = series.index.set_names([AXES.get(name, name) for name in series.index.names])
    index = series.index
    if index.nlevels > 1:
        order = sorted(index.names, key=lambda name: (name != 'snapshot', name == 'outage', name))
        series = series.reorder_levels(order).sort_index()
        return {tuple(str(part) for part in key): int(label) for key, label in series.items()}
    return {str(key): int(label) for key, label in series.items()}


def _label_map(theirs, ours, pairs: dict[str, list[str]]) -> dict[int, int]:
    """Our variable labels to theirs, matched by name pair and coordinate key."""
    mapping: dict[int, int] = {}
    for pypsa_name, our_names in pairs.items():
        if pypsa_name not in theirs.variables:
            continue
        their = _keyed(theirs.variables[pypsa_name].labels)
        for our_name in our_names:
            for key, our_label in _keyed(ours.variables[our_name].labels).items():
                if int(our_label) == -1 or key not in their:
                    continue
                their_label = int(their[key])
                if their_label != -1:
                    mapping[int(our_label)] = their_label
    return mapping


def _without(key, axis: int, label: str):
    """*key* with its *axis* component dropped where that component is *label*, else None — a one-part key is spelled bare on both sides."""
    parts = key if isinstance(key, tuple) else (key,)
    if parts[axis] != label:
        return None
    rest = parts[:axis] + parts[axis + 1 :]
    return rest[0] if len(rest) == 1 else rest


def _rows(flat: pd.DataFrame, labels, relabel) -> dict:
    """Constraint rows by coordinate key: (sign, rhs, sorted (variable, coefficient) pairs).

    The pairs are sorted, and a row whose first nonzero coefficient is negative
    is flipped whole, its sense with it. Coefficients and constants are
    rounded to nine places, as the builders may differ in the last ulp.
    """
    terms = defaultdict(list)
    meta = {}
    for row in flat.itertuples():
        terms[row.labels].append((relabel(int(row.vars)), round(float(row.coeffs), 9)))
        meta[row.labels] = (row.sign, round(float(row.rhs), 9))
    rows = {}
    flipped = {'<=': '>=', '>=': '<=', '=': '='}
    for key, label in _keyed(labels).items():
        if int(label) == -1:
            continue
        sign, rhs = meta[int(label)]
        pairs = tuple(sorted(terms[int(label)]))
        lead = next((coeff for _, coeff in pairs if coeff), 0.0)
        if lead < 0:
            sign, rhs, pairs = flipped[sign], -rhs, tuple((var, -coeff) for var, coeff in pairs)
        rows[key] = (sign, rhs, pairs)
    return rows


def _objective(model, relabel) -> tuple:
    """The objective as a sorted term tuple — quadratic pairs unordered."""
    flat = model.objective.expression.flat
    terms = []
    for row in flat.itertuples():
        if hasattr(row, 'vars1'):
            pair = tuple(sorted((relabel(int(row.vars1)), relabel(int(row.vars2)))))
        else:
            pair = (relabel(int(row.vars)),)
        terms.append((pair, round(float(row.coeffs), 9)))
    return tuple(sorted(terms))


def structure(
    theirs, declared, gc_kinds: dict[str, str], built_rows: dict, built_columns: dict, by_label: dict
) -> dict:
    """Row and column counts per PyPSA name, PyPSA's model against what specsolve built — the shape, before the labels.

    PyPSA's counts come off its own linopy model, masked labels excluded;
    ours are the rows and columns built per block, keyed by the PyPSA name the
    block's description opens with and never summed. A global-constraint row
    is matched through its recorded type.
    """
    theirs_rows = {name: int((c.labels != -1).sum()) for name, c in theirs.constraints.items()}
    theirs_columns = {name: int((v.labels != -1).sum()) for name, v in theirs.variables.items()}
    for label, kind in gc_kinds.items():
        theirs_rows[kind] = theirs_rows.get(kind, 0) + theirs_rows.pop(f'GlobalConstraint-{label}', 0)
    ours_rows: dict[str, dict[str, int]] = defaultdict(dict)
    for name, block in declared.constraints.items():
        stands = stands_for(block.description, theirs)
        pattern = templated(stands)
        if pattern is not None and pattern.groups > 1:
            family = [their_name for their_name in theirs_rows if pattern.match(their_name)]
            theirs_rows[stands] = theirs_rows.get(stands, 0) + sum(theirs_rows.pop(t) for t in family)
            if built_rows.get(name, 0):
                ours_rows[stands][name] = built_rows[name]
        elif pattern is not None:
            for their_name in theirs_rows:
                if (found := pattern.match(their_name)) and by_label.get(name, {}).get(found.group(1)):
                    ours_rows[their_name][name] = by_label[name][found.group(1)]
        elif built_rows.get(name, 0):
            ours_rows[stands][name] = built_rows[name]
    ours_columns: dict[str, dict[str, int]] = defaultdict(dict)
    for name, block in declared.variables.items():
        if built_columns.get(name, 0):
            ours_columns[stands_for(block.description, theirs)][name] = built_columns[name]

    def table(theirs_side: dict, ours_side: dict) -> dict[str, dict]:
        names = {n for n, c in theirs_side.items() if c} | set(ours_side)
        return {n: {'pypsa': theirs_side.get(n, 0), 'specsolve': ours_side.get(n, {})} for n in sorted(names)}

    return {'rows': table(theirs_rows, ours_rows), 'columns': table(theirs_columns, ours_columns)}


def matched(counts: dict) -> bool:
    """One PyPSA name, one block, one equal count — anything else is a difference."""
    return len(counts['specsolve']) == 1 and next(iter(counts['specsolve'].values())) == counts['pypsa']


def shown(blocks: dict[str, int]) -> str:
    """A block breakdown as the pages and messages print it — ``3+1+4``, never a sum."""
    return '+'.join(str(count) for count in blocks.values()) or '0'


def solver_size(n, built_model) -> dict[str, dict[str, int]]:
    """Rows, columns and nonzeros of the model each side handed HiGHS — the size that is actually optimised.

    PyPSA's from the ``highspy.Highs`` handle linopy keeps after the solve;
    ours from :meth:`Model.diagnostics`. Naming-independent, so it catches a
    padded term or a helper row one side adds.
    """
    theirs = n.model.solver_model
    ours = built_model.diagnostics()
    return {
        'pypsa': {'rows': theirs.getNumRow(), 'columns': theirs.getNumCol(), 'nonzeros': theirs.getNumNz()},
        'specsolve': {'rows': ours.rows, 'columns': ours.columns, 'nonzeros': ours.nonzeros},
    }


def explained(stem: str, shape: dict, reasons: dict) -> tuple[dict, list[str]]:
    """Every name whose count is not one block equal to PyPSA's, with its recorded reason — and those with none.

    ``deviations.yaml`` maps a PyPSA name to ``{structure: reason}``.
    """
    differences, unexplained = {}, []
    solver = shape['solver']
    if solver['pypsa'] != solver['specsolve']:
        reason = reasons.get('solver model', {}).get('structure')
        differences['solver model'] = {**solver, 'kind': 'solver', 'reason': reason}
        if not reason:
            unexplained.append(
                f'{stem}: the solver models differ — pypsa {solver["pypsa"]}, specsolve {solver["specsolve"]}'
            )
    for kind in ('rows', 'columns'):
        for name, counts in shape[kind].items():
            if matched(counts):
                continue
            reason = reasons.get(name, {}).get('structure')
            differences[name] = {**counts, 'kind': kind, 'reason': reason}
            if not reason:
                unexplained.append(
                    f'{stem}: {kind} of {name} — pypsa {counts["pypsa"]}, specsolve {shown(counts["specsolve"])}'
                )
    return differences, unexplained


def _gc_key(their_name: str, key: object) -> object:
    """A global-constraint row's key as ours spells it — its label, then the scenario where it has one."""
    label = their_name.removeprefix('GlobalConstraint-')
    if key == ():
        return label
    return (label, *((key,) if isinstance(key, str) else key))


def compare(theirs, ours, declared, gc_kinds: dict[str, str]) -> dict[str, object]:
    """Verdicts: which PyPSA names are model-equal, which are the same region in several blocks, which differ.

    A name empty on both sides lands in no bucket. A mismatch with a ``blocks``
    reason in ``deviations.yaml`` comes back under ``recorded``.
    """
    rows = defaultdict(list)
    for name, block in declared.constraints.items():
        stands = stands_for(block.description, theirs)
        pattern = templated(stands)
        if pattern is None or pattern.groups > 1:
            rows[stands].append((name, None))
        else:
            for their_name in theirs.constraints:
                if found := pattern.match(their_name):
                    rows[their_name].append((name, found.group(1)))
    columns = defaultdict(list)
    for name, block in declared.variables.items():
        columns[stands_for(block.description, theirs)].append(name)

    ours_to_theirs = _label_map(theirs, ours, columns)
    templates = template_dims(declared, theirs)

    def relabel(label: int) -> int:
        if label == -1:
            return -1
        return ours_to_theirs.get(label, -label - 1000)

    verdict: dict[str, list[str]] = {'equal': [], 'region': [], 'mismatch': []}
    for pypsa_name, our_names in columns.items():
        bounds_ours = {}
        for our_name in our_names:
            for r in ours.variables[our_name].flat.itertuples():
                bounds_ours[relabel(int(r.labels))] = (r.lower, r.upper)
        if pypsa_name not in theirs.variables:
            if bounds_ours:
                verdict['mismatch'].append(pypsa_name)
            continue
        bounds_theirs = {int(r.labels): (r.lower, r.upper) for r in theirs.variables[pypsa_name].flat.itertuples()}
        if not bounds_ours and not bounds_theirs:
            continue
        their_kind = pypsa_name in [*theirs.integers, *theirs.binaries]
        ok = all((our_name in [*ours.integers, *ours.binaries]) == their_kind for our_name in our_names)
        if bounds_ours != bounds_theirs:
            ok = False
        bucket = 'mismatch' if not ok else ('equal' if len(our_names) == 1 else 'region')
        verdict[bucket].append(pypsa_name)

    for pypsa_name, our_names in rows.items():
        family = templated(pypsa_name)
        their_names = (
            [n for n in theirs.constraints if n.startswith('GlobalConstraint-')]
            if not pypsa_name[0].isupper()
            else [n for n in theirs.constraints if family.match(n)]
            if family is not None
            else ([pypsa_name] if pypsa_name in theirs.constraints else [])
        )
        their_rows: dict = {}
        for their_name in their_names:
            constraint = theirs.constraints[their_name]
            for key, row in _rows(constraint.flat, constraint.labels, lambda x: x).items():
                their_rows[key if pypsa_name[0].isupper() else _gc_key(their_name, key)] = row
        our_rows: dict = {}
        for our_name, k in our_names:
            if our_name not in ours.constraints:
                continue
            constraint = ours.constraints[our_name]
            found = _rows(constraint.flat, constraint.labels, relabel)
            if k is not None:
                axis = template_axis(declared.constraints[our_name], templates[our_name])
                found = {rest: row for key, row in found.items() if (rest := _without(key, axis, k)) is not None}
            our_rows |= found
        if not pypsa_name[0].isupper():
            typed = {label for label, gc in gc_kinds.items() if gc == pypsa_name}
            their_rows = {
                key: row for key, row in their_rows.items() if (key if isinstance(key, str) else key[0]) in typed
            }
        if not our_rows and not their_rows:
            continue
        if our_rows == their_rows:
            verdict['equal' if len(our_names) == 1 else 'region'].append(pypsa_name)
        else:
            verdict['mismatch'].append(pypsa_name)
            for key in sorted({*our_rows, *their_rows}, key=str):
                if our_rows.get(key) != their_rows.get(key):
                    print(
                        f'  {pypsa_name}[{key}]:\n    ours   {our_rows.get(key)}\n    theirs {their_rows.get(key)}',
                        file=sys.stderr,
                    )

    if _objective(ours, relabel) == _objective(theirs, lambda x: x):
        verdict['equal'].append('objective')
    else:
        verdict['mismatch'].append('objective')
    recorded = {name: REASONS[name]['blocks'] for name in verdict['mismatch'] if REASONS.get(name, {}).get('blocks')}
    verdict['mismatch'] = [name for name in verdict['mismatch'] if name not in recorded]
    return {'recorded': dict(sorted(recorded.items()))} | {kind: sorted(names) for kind, names in verdict.items()}


def lanes(stem: str) -> tuple[dict[str, object], dict[str, object], bool]:
    """One rung through everything: the objective across the fence, the model against the model, the coverage.

    A rung that records a PyPSA bug is held to its oracle's objective alone:
    PyPSA's own model of it is the one that is wrong, so there is no model to
    compare against.
    """
    from tests import linopy_lane as lpl

    bug = issue(stem)
    n = network(stem)
    PLAIN[0] = not n.has_scenarios
    gc_kinds = {str(label): str(gc['type']) for label, gc in prep.first_scenario(n.global_constraints).iterrows()}
    if bug is None:
        theirs = pypsa_model(stem) if outages(stem) is None else None
        n = solved(stem, n)
        theirs = n.model if theirs is None else theirs
        target = float(n.objective) + float(n.objective_constant)
    else:
        target = intended(stem)
    spec = spec_of(stem)
    whole = mathspec.to_spec(spec)
    try:
        sources = prepared(spec, network(stem), stem)
        full = sps.build(spec, sources)
    except (
        sps.DataError,
        TypeError,
        KeyError,
        ValueError,
        IndexError,
    ) as error:  # prep has not learnt this network yet
        note = f'{type(error).__name__}: {error}'.splitlines()[0][:160]
        print(f'{stem}: prep cannot prepare {spec.name} yet — {note}', file=sys.stderr)
        return {'spec': spec.name, 'unattached': note}, {'error': 'not attached'}, True
    file_rows, file_columns = built_counts(full)
    cut = rung_spec(stem, spec, file_rows, file_columns, fed(sources), zero_parameters(sources, whole))
    cut_sources = prepared(cut, network(stem), stem)
    model = sps.build(cut, cut_sources)
    same_model(stem, full, model)
    declared = mathspec.to_spec(cut)
    committed(stem, spec.name, declared, cut_sources)
    result = model.solve(solver_name='highs')
    assert result.is_ok, f'{stem}: specsolve did not solve — {result.termination_condition}'
    stamps = {
        'spec': spec.name,
        'built_rows': file_rows,
        'built_columns': file_columns,
        'dims': {name: len(table) for name, table in sources.items() if name in whole.dimensions},
        'attached_nonempty': fed(sources),
        'conjuncts': conjunct_verdicts(full, whole.program),
    }
    if bug is not None:
        return diverging(bug, stamps, result, target)
    solver = solver_size(n, model)
    built_rows, built_columns = built_counts(model)
    by_label = built_by_label(result, template_dims(declared, n.model))
    shape = structure(theirs, declared, gc_kinds, built_rows, built_columns, by_label) | {'solver': solver}
    differences, unexplained = explained(stem, shape, REASONS)
    for line in unexplained:
        print(line, file=sys.stderr)
    parity = {
        'specsolve_objective': round(float(result.objective), 6),
        'matches': math.isclose(float(result.objective), target, rel_tol=1e-9, abs_tol=1e-6),
        **stamps,
        'duals': duals(result, n, declared, gc_kinds, REASONS),
        'structure': {
            'rows': [
                sum(c['pypsa'] for c in shape['rows'].values()),
                sum(sum(c['specsolve'].values()) for c in shape['rows'].values()),
            ],
            'columns': [
                sum(c['pypsa'] for c in shape['columns'].values()),
                sum(sum(c['specsolve'].values()) for c in shape['columns'].values()),
            ],
            'solver': solver,
            'per_name': {kind: shape[kind] for kind in ('rows', 'columns')},
            'differences': differences,
        },
    }
    try:
        ours = lpl.build(cut, cut_sources)
    except Exception as error:
        note = f'{type(error).__name__}: {error}'.splitlines()[0][:200]
        return parity, {'error': note}, parity['matches'] and priced(parity) and shaped(parity)
    verdict = compare(theirs, ours, declared, gc_kinds)
    return parity, verdict, parity['matches'] and priced(parity) and shaped(parity) and not verdict['mismatch']


def diverging(bug: int, stamps: dict, result, target: float) -> tuple:
    """The stamps of a rung that records a PyPSA bug: its objective against the oracle's, and what it built."""
    reason = f'PyPSA/PyPSA#{bug} — PyPSA 1.3.0 gets this network wrong, so the objective is held to the oracle and no model is compared'
    parity = {
        'specsolve_objective': round(float(result.objective), 6),
        'matches': math.isclose(float(result.objective), target, rel_tol=1e-9, abs_tol=1e-6),
        'diverges': bug,
        **stamps,
        'duals': {'compared': 0, 'skipped': reason, 'negated': {}, 'per_name': {}, 'differences': {}},
        'structure': {
            'rows': [None, sum(stamps['built_rows'].values())],
            'columns': [None, sum(stamps['built_columns'].values())],
            'solver': None,
            'per_name': {'rows': {}, 'columns': {}},
            'differences': {},
        },
    }
    return parity, {'error': reason}, parity['matches']


def priced(parity: dict) -> bool:
    """Every dual agrees or has a reason, or the lane had none to offer."""
    return all(d['reason'] for d in parity['duals']['differences'].values())


def shaped(parity: dict) -> bool:
    """Every count that differs has a reason on record."""
    return all(d['reason'] for d in parity['structure']['differences'].values())


REASONS: dict = yaml.safe_load(DEVIATIONS.read_text()) or {} if DEVIATIONS.exists() else {}


def settled(committed: object, fresh: object) -> object:
    """*fresh*, with every float the committed certificate already agrees on left as it stands.

    HiGHS re-solving the same model does not return the same bits, and the gate
    is a byte diff. Ints are never settled.
    """
    if isinstance(committed, dict) and isinstance(fresh, dict):
        return {key: settled(committed.get(key), value) for key, value in fresh.items()}
    if isinstance(committed, list) and isinstance(fresh, list) and len(committed) == len(fresh):
        return [settled(was, now) for was, now in zip(committed, fresh, strict=True)]
    if _is_float(committed) and _is_float(fresh):
        return committed if math.isclose(float(committed), float(fresh), rel_tol=1e-9, abs_tol=1e-12) else fresh
    return fresh


def _is_float(value: object) -> bool:
    return isinstance(value, float) and not isinstance(value, bool)


def coverage(stamped: dict[str, dict]) -> list[str]:
    """What the ladder as a whole leaves untested — empty when every block, mask and parameter is exercised."""
    gaps = []
    by_file: dict[str, list[dict]] = defaultdict(list)
    for stem in sorted(stamped):
        if 'unattached' in stamped[stem]['parity']:
            continue
        by_file[stamped[stem]['parity']['spec']].append(stamped[stem]['parity'])
    for name, stamps in by_file.items():
        declared = mathspec.to_spec(CORPUS / 'examples' / name)
        for kind, blocks in (('built_rows', declared.constraints), ('built_columns', declared.variables)):
            for block_name, block in blocks.items():
                counts = [stamp[kind][block_name] for stamp in stamps]
                if not sum(counts):
                    gaps.append(f'{name}: no rung builds {block_name}')
                elif block.where and not any(
                    0 < c < math.prod(stamp['dims'][d] for d in block.dims)
                    for c, stamp in zip(counts, stamps, strict=True)
                ):
                    gaps.append(f'{name}: {block_name} is always all-or-nothing, so its mask is untested')
        fed = set().union(*(stamp['attached_nonempty'] for stamp in stamps))
        gaps.extend(
            f'{name}: no rung feeds {unfed}' for unfed in sorted({*declared.parameters, *declared.relations} - fed)
        )
        gaps.extend(untested_conjuncts(name, mathspec.to_spec(CORPUS / 'examples' / name).program, stamps))
    return gaps


def main() -> int:
    ladder = rungs()
    assert ladder, f'no rung scripts under {RUNGS} — is {CORPUS} a mathspec checkout?'
    committed = json.loads(RECORDS.read_text()) if RECORDS.exists() else {}
    for folder in (TABLES, PROJECTIONS):
        if folder.exists():
            shutil.rmtree(folder)
    stamped: dict[str, dict] = {}
    broken = []
    for stem in ladder:
        parity, structural, good = lanes(stem)
        if 'unattached' in parity:
            stamped[stem] = {'parity': parity, 'structural': structural}
            print(f'{stem}: UNATTACHED · prep cannot prepare {parity["spec"]} yet')
            continue
        was = committed.get(stem, {})
        stamped[stem] = {
            'parity': settled(was.get('parity'), parity),
            'structural': settled(was.get('structural'), structural),
        }
        proof = (
            f'{len(structural["equal"])} equal · {len(structural["region"])} region'
            + (f' · MISMATCH {structural["mismatch"]}' if structural.get('mismatch') else '')
            + (f' · {len(structural["recorded"])} recorded' if structural.get('recorded') else '')
            if 'equal' in structural
            else f'objective only — {structural["error"]}'
        )
        duals_ = parity['duals']
        priced_ = (
            f'duals on {duals_["compared"]} rows'
            + (f', {len(duals_["negated"])} negated' if duals_['negated'] else '')
            + (f', {len(duals_["differences"])} names differ' if duals_['differences'] else '')
            if duals_['compared']
            else f'no duals — {duals_["skipped"]}'
        )
        for name, d in duals_['differences'].items():
            if not d['reason']:
                print(f'{stem}: duals of {name} differ by {d["max_abs_diff"]} with no reason', file=sys.stderr)
        shape_ = parity['structure']
        shaped_ = (
            f'held to the oracle, PyPSA/PyPSA#{parity["diverges"]}'
            if 'diverges' in parity
            else f'{shape_["rows"][0]} rows, {shape_["columns"][0]} columns'
            if not shape_['differences']
            else f'{len(shape_["differences"])} of {len(shape_["per_name"]["rows"]) + len(shape_["per_name"]["columns"])} names differ'
        )
        print(f'{stem}: {"MATCH" if parity["matches"] else "DIFFER"} · {shaped_} · {priced_} · {proof}')
        if not good:
            broken.append(stem)
    # the conjunct verdicts feed `coverage` below and are not committed
    recorded = {
        stem: record | {'parity': {key: v for key, v in record['parity'].items() if key != 'conjuncts'}}
        for stem, record in stamped.items()
    }
    RECORDS.write_text(json.dumps(recorded, indent=2, sort_keys=True) + '\n')
    attached_stamps = [st for st in stamped.values() if 'unattached' not in st['parity']]
    used_structure = {
        name for st in attached_stamps for name, d in st['parity']['structure']['differences'].items() if d['reason']
    }
    used_duals = {
        name for st in attached_stamps for name, d in st['parity']['duals']['differences'].items() if d['reason']
    }
    used_negated = {name for st in attached_stamps for name in st['parity']['duals'].get('negated', ())}
    used_blocks = {name for st in attached_stamps for name in st['structural'].get('recorded', {})}
    stale = sorted(
        f'{name}.{key}'
        for name, entry in REASONS.items()
        for key, used in (
            ('structure', used_structure),
            ('duals', used_duals),
            ('negated', used_negated),
            ('blocks', used_blocks),
        )
        if key in entry and name not in used
    )
    gaps = coverage(stamped) + [f'deviations.yaml: {name} records a reason no rung needs' for name in stale]
    for gap in gaps:
        print(gap, file=sys.stderr)
    if broken or gaps:
        print(f'{len(broken)} rung(s) differ, {len(gaps)} coverage gap(s)', file=sys.stderr)
        return 1
    print('every rung matches PyPSA as deep as the engines allow, and says how deep that is')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
