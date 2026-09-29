"""Every rung page shows the files that ran, byte for byte, and is current.

The projection, the script and the tables under `differential/pypsa/` are what
the parity runner wrote from the pinned mathspec; the `PyPSA parity` workflow
holds them to the corpus. Here, with no pypsa on the install, what is held is
the page: each fence is one of those files verbatim, and every page is what
`tools.ladder` prints from them and the certificate.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import yaml

from tests.test_models_gallery import _fences
from tools import ladder

STEMS = ladder.stems()


def test_the_ladder_pages_are_current():
    assert ladder.main(['--check']) == 0, 'stale ladder pages — pixi run python -m tools.ladder'


def test_every_stamped_rung_has_a_page_and_a_projection():
    stamped = json.loads((ladder.LADDER / 'references.json').read_text())
    attached = {s for s, r in stamped.items() if 'unattached' not in r['parity']}
    assert attached == set(STEMS), 'a certified rung without a projection, or a projection no run certifies'
    missing = [s for s in STEMS if not (ladder.PAGES / f'{s}.md').exists()]
    assert not missing, f'rungs without a page: {missing}'
    index = ladder.INDEX.read_text()
    unlisted = [s for s in stamped if s not in attached and 'prep cannot prepare' not in index]
    assert not unlisted, f'unattached rungs the index does not list: {unlisted}'


@pytest.mark.parametrize('stem', STEMS, ids=STEMS)
def test_the_specsolve_tab_shows_the_projection_that_solves(stem: str):
    fences = _fences((ladder.PAGES / f'{stem}.md').read_text(), 'yaml')
    assert (ladder.RUNGS / f'{stem}.yaml').read_text().rstrip() + '\n' in fences, 'the projected model has drifted'


@pytest.mark.parametrize('stem', STEMS, ids=STEMS)
def test_the_pypsa_tab_shows_the_script_that_builds_the_network(stem: str):
    fences = _fences((ladder.PAGES / f'{stem}.md').read_text(), 'python')
    assert (ladder.RUNGS / f'{stem}.py').read_text().rstrip() + '\n' in fences, 'the rung script has drifted'


@pytest.mark.parametrize('stem', STEMS, ids=STEMS)
def test_the_data_section_shows_the_tables_the_binding_produced(stem: str):
    fences = _fences((ladder.PAGES / f'{stem}.md').read_text(), 'csv')
    for path in sorted((ladder.LADDER / 'tables' / stem).glob('*.csv')):
        assert path.read_text().rstrip() + '\n' in fences, f'{path.name} is not embedded byte for byte'


def test_every_structure_difference_has_a_reason_and_is_on_the_index():
    stamped = json.loads((ladder.LADDER / 'references.json').read_text())
    reasons = yaml.safe_load((ladder.LADDER / 'deviations.yaml').read_text()) or {}
    index = ladder.INDEX.read_text()
    for stem in STEMS:
        for kind in ('structure', 'duals'):
            for name, d in stamped[stem]['parity'][kind]['differences'].items():
                assert d['reason'] and reasons.get(name, {}).get(kind) == d['reason'], (
                    f'{stem}: {kind} of {name} differ without a reason'
                )
                assert f'`{name}' in index and d['reason'] in index, f'{name} and its reason are not on the index'
        for name, reason in stamped[stem]['structural'].get('recorded', {}).items():
            assert reason and reasons.get(name, {}).get('blocks') == reason, (
                f'{stem}: the linopy-lane comparison records {name} without a reason'
            )
            assert f'`{name}' in index and reason in index, f'{name} and its reason are not on the index'


def test_every_rung_page_is_in_the_nav():
    """The nav is written by hand, and the build does not refuse a page it leaves out."""
    nav = (ladder.ROOT / 'mkdocs.yml').read_text()
    unlisted = [s for s in STEMS if f'examples/pypsa_ladder/{s}.md' not in nav]
    assert not unlisted, f'rung pages missing from the nav in mkdocs.yml: {unlisted}'


def test_the_reasons_file_names_each_pypsa_name_once():
    """A YAML loader keeps the last of two mappings for one key, so a second entry silently drops the first's reasons."""
    import re

    keys = re.findall(r'^([^\s#][^:\n]*):$', (ladder.LADDER / 'deviations.yaml').read_text(), flags=re.MULTILINE)
    doubled = sorted({k for k in keys if keys.count(k) > 1})
    assert not doubled, f'deviations.yaml names these more than once: {doubled}'


def test_the_index_lists_every_rung():
    text = ladder.INDEX.read_text()
    missing = [s for s in STEMS if f'pypsa_ladder/{s}.md' not in text]
    assert not missing, f'rungs the index does not list: {missing}'


def test_every_negated_dual_is_checked_rather_than_excused():
    """A ``negated:`` reason is a claim under test, not an exemption.

    It says the file states PyPSA's row negated, so the runner compares our
    dual against the negative of theirs and the rung is red if that fails. The
    ``duals:`` key beside it is the opposite — a difference no comparison can
    reach — so one name carrying both would let the exemption swallow the
    claim.
    """
    stamped = json.loads((ladder.LADDER / 'references.json').read_text())
    reasons = yaml.safe_load((ladder.LADDER / 'deviations.yaml').read_text()) or {}
    index = ladder.INDEX.read_text()
    recorded = {name for name, entry in reasons.items() if 'negated' in entry}
    assert not [n for n in recorded if 'duals' in reasons[n]], (
        'a name recorded as negated and also excused by a `duals:` reason, which would hide a difference'
    )
    checked = set()
    for stem in STEMS:
        duals = stamped[stem]['parity']['duals']
        for name, reason in duals['negated'].items():
            assert reasons.get(name, {}).get('negated') == reason, f'{stem}: {name} negated for an unrecorded reason'
            entry = duals['per_name'][name]
            assert entry['matches'] and entry['max_abs_diff'] <= 1e-6, (
                f"{stem}: {name} is recorded negated but its dual is not the negative of PyPSA's, "
                f'off by {entry["max_abs_diff"]} at the tolerance the runner compares at'
            )
            assert f'`{name}' in index and reason in index, f'{name} and why it is negated are not on the index'
            checked.add(name)
        assert not recorded & set(duals['differences']), (
            f'{stem}: {sorted(recorded & set(duals["differences"]))} differ after being negated'
        )
    assert checked == recorded, f'`negated:` reasons no rung checks: {sorted(recorded - checked)}'


#: A file whose named expressions read one another, so the projection has to walk them and
#: could reach for a set while doing it. `eta` reads `zeta`, so one pass is not enough.
_ORDERING_FIXTURE = {
    'dimensions': {'t': {'dtype': 'int'}, 'g': {}},
    'parameters': {n: {'dims': ['g']} for n in ('alpha', 'beta', 'gamma', 'delta', 'epsilon')},
    'variables': {'x': {'dims': ['t', 'g'], 'bounds': {'lower': 0}}},
    'expressions': {
        'zeta': {'dims': ['t', 'g'], 'cases': {'a': {'when': 'alpha', 'expression': 'beta'}}, 'otherwise': 1},
        'eta': {'dims': ['t', 'g'], 'cases': {'a': {'when': 'gamma', 'expression': 'delta'}}, 'otherwise': 'zeta'},
        'theta': {'dims': ['t', 'g'], 'expression': 'epsilon'},
    },
    'constraints': {'c': {'dims': ['t', 'g'], 'expression': 'x >= zeta + eta + theta'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(x)'},
}

_EMIT = """
import json, sys
from differential.pypsa import projection
raw = json.load(open(sys.argv[1]))
record = {'built_columns': {'x': 4}, 'built_rows': {'c': 4}, 'attached_nonempty': sorted(raw['parameters'])}
out = projection.project(raw, record)
print(json.dumps([list(out['expressions']), list(out['parameters']), list(out['dimensions'])]))
"""


def test_a_projection_orders_its_names_the_same_whatever_the_hash_seed(tmp_path):
    """The projection is diff-gated, so nothing in it may be *ordered* by a set.

    Python seeds string hashing per process, so a set that decides emit order
    writes a different file on every run, and the committed projection goes red
    on a tree nobody touched (#1466). Subprocesses, because one interpreter has
    one seed.
    """
    import subprocess
    import sys

    (tmp_path / 'raw.json').write_text(json.dumps(_ORDERING_FIXTURE))
    (tmp_path / 'emit.py').write_text(_EMIT)
    orders = {
        subprocess.run(
            [sys.executable, str(tmp_path / 'emit.py'), str(tmp_path / 'raw.json')],
            capture_output=True,
            text=True,
            check=True,
            env={'PYTHONHASHSEED': seed, 'PYTHONPATH': str(ladder.ROOT)},
        ).stdout
        for seed in ('1', '2', '3', '4')
    }
    assert len(orders) == 1, f'the projection emitted {len(orders)} different name orders across four hash seeds'
    expressions, parameters, _ = json.loads(orders.pop())
    assert expressions == ['zeta', 'eta', 'theta'], "and the order is the file's own, not any set's"
    assert parameters == ['alpha', 'beta', 'gamma', 'delta', 'epsilon'], 'parameters likewise'


def _conjunct_stamps(marks: str) -> list[dict]:
    """One stamp per rung, each recording what a block's single conjunct did there."""
    return [{'conjuncts': {'block': mark}} for mark in marks]


@pytest.mark.parametrize(
    ('marks', 'expected'),
    [
        pytest.param('bt', None, id='a rung has it true of some coordinates and not others'),
        pytest.param('ff', 'never true', id='no rung makes it true'),
        pytest.param('tt', 'never false', id='no rung makes it false'),
        pytest.param('tf', None, id='true on one rung and false on another'),
        pytest.param('-t', 'never false', id='a rung that builds no frame says nothing'),
    ],
)
def test_a_conjunct_no_rung_varies_is_reported(marks, expected):
    """The half the block-level sweep cannot reach — mathspec#312.

    A mask of `a AND b` is exercised as a whole the moment `a` varies, so a `b`
    true at every coordinate of every rung passes unnoticed and a regime it
    alone guards could be missing. ``tf`` is the case a per-rung check gets
    wrong: neither rung is partial, and the conjunct is exercised anyway
    because the two disagree.
    """
    import sys

    if str(ladder.LADDER) not in sys.path:
        sys.path.insert(0, str(ladder.LADDER))
    import sweep

    mask = SimpleNamespace(conjuncts=('the conjunct',))
    program = SimpleNamespace(constraints={'block': SimpleNamespace(where=mask, dims=('t',))}, variables={})

    gaps = sweep.untested_conjuncts('pypsa.yaml', program, _conjunct_stamps(marks))

    if expected is None:
        assert gaps == [], 'a conjunct the ladder varies is not a gap'
    else:
        assert len(gaps) == 1 and expected in gaps[0], f'expected a {expected!r} gap, got {gaps}'
