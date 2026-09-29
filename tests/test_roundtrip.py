"""``Spec.to_yaml`` gives back the same model — held over the whole corpus.

A field gaining a default, a validator normalising a value, an alias, a dict
that stops preserving order: each turns the dumped file into a *different*
model while every test that builds from the original still passes. The property
is checked against every example and every ported model, which between them
exercise every construct the language has.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import yaml as pyyaml
from mathspec import to_spec

from tests.conftest import SPEC_PATHS

if TYPE_CHECKING:
    from pathlib import Path


def test_the_corpus_is_not_empty():
    """The parametrised tests below pass vacuously if ``constructs.models()`` finds nothing."""
    assert len(SPEC_PATHS) >= 10, f'the model corpus looks wrong: {SPEC_PATHS}'


@pytest.mark.parametrize('path', SPEC_PATHS, ids=lambda p: p.stem)
def test_a_model_survives_a_round_trip(path: Path):
    """`load -> to_yaml -> load` is the same model, field for field."""
    original = to_spec(path)
    dumped = original.to_yaml()
    reloaded = to_spec(pyyaml.safe_load(dumped))

    assert reloaded.model_dump() == original.model_dump(), f'{path} does not survive a round trip'


@pytest.mark.parametrize('path', SPEC_PATHS, ids=lambda p: p.stem)
def test_the_two_out_forms_agree(path: Path):
    """`to_dict`, `model_dump` and what `to_yaml` writes agree, so a caller cannot get two answers."""
    spec = to_spec(path)
    assert pyyaml.safe_load(spec.to_yaml()) == spec.to_dict()
    assert spec.model_dump() == spec.to_dict(), "pydantic's own dump has to agree too"
    assert to_spec(spec.to_dict()).model_dump() == spec.model_dump()


@pytest.mark.parametrize('path', SPEC_PATHS, ids=lambda p: p.stem)
def test_the_dump_is_stable(path: Path):
    """Dumping twice gives the same bytes, so a model emitted for review diffs cleanly."""
    once = to_spec(path).to_yaml()
    twice = to_spec(pyyaml.safe_load(once)).to_yaml()
    assert once == twice, f'{path} dumps differently the second time'


def test_a_dict_built_model_gets_a_file():
    """A model built as a dict gets a reviewable file, and it is the same spec."""
    built = {
        'dimensions': {'t': {'dtype': 'int'}},
        'parameters': {'cost': {'dims': ['t']}},
        'variables': {'x': {'dims': ['t'], 'bounds': {'lower': 0, 'upper': 10}}},
        'constraints': {'cap': {'dims': ['t'], 'expression': 'x <= 4'}},
        'objective': {'sense': 'maximize', 'expression': 'sum(x * cost)'},
    }
    text = to_spec(built).to_yaml()

    assert to_spec(pyyaml.safe_load(text)).model_dump() == to_spec(built).model_dump()
    assert text.startswith('version: 0\n'), 'a generated file should say which surface it targets'
    assert 'dimensions:' in text
    assert 'piecewise' not in text, 'an absent section is absence, not a value'


def test_a_declared_version_survives():
    """`version:` is not a default when the file states it — a dumped model has
    to keep saying which surface it targets (#67)."""
    text = to_spec({'version': 0, 'dimensions': {'t': {'dtype': 'int'}}}).to_yaml()
    assert 'version: 0' in text


@pytest.mark.parametrize('path', SPEC_PATHS, ids=lambda p: p.stem)
def test_the_review_copy_states_the_objective_sense(path: Path):
    """`sense` is emitted even at its default — the one word a reviewer must
    not have to infer."""
    spec = to_spec(path)
    if spec.objective is None:
        pytest.skip('no objective to state')
    text = spec.to_yaml()
    assert f'sense: {spec.objective.sense}' in text, f'{path}: the objective lost its direction'


def test_absence_is_dropped_and_values_are_kept():
    """The whole rule, both halves: absence is dropped, every value is kept, default or not."""
    text = to_spec(
        {
            'dimensions': {'t': {'dtype': 'int'}},
            'variables': {'x': {'dims': ['t']}},
            'objective': {'sense': 'minimize', 'expression': 'sum(x)'},
        }
    ).to_yaml()

    for absent in ('where: null', 'coords: {}', 'macros:', 'piecewise:'):
        assert absent not in text, f'{absent!r} is absence and should not be written'
    for stated in ('dtype: int', 'sense: minimize', 'domain: continuous', 'version: 0'):
        assert stated in text, f'{stated!r} is a value and should be written'


def test_json_carries_a_model_too():
    """`model_dump_json` round-trips, because an open bound is `null` and not an infinity.

    JSON has no infinity. An open side is absent, so the serializer drops it
    and JSON and `to_dict` agree.
    """
    spec = to_spec(
        {
            'dimensions': {'t': {'dtype': 'int'}},
            'variables': {'x': {'dims': ['t'], 'bounds': {'lower': 0}}, 'y': {'dims': ['t']}},
            'objective': {'sense': 'minimize', 'expression': 'sum(x) + sum(y)'},
        }
    )
    assert json.loads(spec.model_dump_json()) == spec.to_dict()
    assert to_spec(json.loads(spec.model_dump_json())).to_dict() == spec.to_dict()

    out = spec.to_dict()['variables']
    assert out['x']['bounds'] == {'lower': 0.0}, 'a stated bound stays, its open partner does not'
    assert 'bounds' not in out['y'], 'unbounded on both sides is no bounds block at all'
    assert to_spec(spec.to_dict()).variables['y'].bounds.lower is None
