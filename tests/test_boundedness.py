"""A model no data can make bounded is named by `check`, not by the solver (#229)."""

from __future__ import annotations

import warnings

import pytest
import yaml
from mathspec import advice, to_spec

import specsolve as sps
from specsolve.errors import SpecsolveWarning
from tests.conftest import EXAMPLES_DIR

#: The issue's variant 1, as a mapping the cases below vary one key of:
#: ``slack`` is unbounded below, is in the objective, and no constraint names it.
FREE_SLACK = {
    'dimensions': {'t': {'dtype': 'int'}},
    'parameters': {'cap': {'dims': ['t']}, 'cost': {'dims': ['t']}},
    'variables': {
        'x': {'dims': ['t'], 'bounds': {'lower': 0, 'upper': 'cap'}},
        'slack': {'dims': ['t']},
    },
    'constraints': {'limit': {'dims': ['t'], 'expression': 'x <= cap'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(x + slack, over=t)'},
}


def _check(**overrides):
    return sps.check({**FREE_SLACK, **overrides})


def test_the_note_reaches_the_caller_as_a_warning_off_check():
    """``check`` hands each note to the caller whole; which models earn one is mathspec's."""
    with pytest.warns(SpecsolveWarning) as record:
        _check()
    message = '\n'.join(str(w.message) for w in record)
    assert "Variable 'slack'" in message, 'the note names the variable, which the solver answer does not'
    assert 'bounds.lower' in message, 'the note names the open side'
    assert 'no constraint names it' in message, 'the note gives the other half of the conjunction'
    assert 'unbounded' in message, 'the note uses the word the solve would have answered with'


def test_the_note_closes_no_door():
    """Every verb but `check` is silent, and the unbounded model still builds."""
    data = {'t': [0, 1, 2], 'cap': [1.0, 1.0, 1.0], 'cost': [1.0, 1.0, 1.0]}
    with warnings.catch_warnings():
        warnings.simplefilter('error', SpecsolveWarning)
        sps.build(FREE_SLACK, data)

    assert sps.solve(FREE_SLACK, data).termination_condition == 'unbounded', (
        'the solve is left to answer as it always did — the note is advice, not a gate'
    )


def test_check_reads_the_notes_off_the_expanded_curve():
    """`piecewise:` bounds its variables through the constraints it expands into."""
    raw = yaml.safe_load((EXAMPLES_DIR / 'piecewise.yaml').read_text())
    del raw['variables']['op_cost']['bounds']
    written_out = to_spec(raw).expand('piecewise')

    assert not advice(written_out), 'the curve bounds op_cost, so there is nothing to advise about'
    with warnings.catch_warnings():
        warnings.simplefilter('error', SpecsolveWarning)
        sps.check(written_out)
