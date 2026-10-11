"""The streaming language boundary: out-of-subset constructs are load errors.

There is no runtime fallback — the streaming subset IS the language
(docs/about/architecture.md), and both lanes are inside it: `tests.linopy_lane`
builds the same file through the same `inputs.lowered` gate.
Errors must carry the construct and its context, verbatim.
"""

from __future__ import annotations

import pytest

import specsolve as sps
from specsolve.errors import LanguageError
from tests.conftest import EXAMPLES_DIR, SPEC_PATHS, expanded, schema_of

DISPATCH = EXAMPLES_DIR / 'dispatch.yaml'


def _objective(expression: str) -> dict:
    return {'objective.expression': expression}


@pytest.mark.parametrize('path', SPEC_PATHS, ids=lambda p: p.name)
def test_every_shipped_example_is_inside_the_language(path):
    """Every dim rule, over the corpus this repository ships, and then lowering.

    The language sweeps the rules over its own probes; this is the same sweep
    over the gallery and the ports. Loading a ``Spec`` runs every dim rule, and
    the result must then lower.
    """
    expanded(path, 'piecewise')


@pytest.mark.parametrize(
    'patch',
    [
        pytest.param({'variables.p.domain': 'binary', 'variables.p.bounds': {}}, id='binary-variable'),
        pytest.param({'variables.p.where': 'snapshot > 2'}, id='where-on-a-dimension-roadmap-5b'),
        pytest.param(_objective('sum(p * cost)'), id='affine-product'),
        pytest.param(_objective('sum(p * p)'), id='degree-two-in-the-objective'),
        pytest.param(
            {'constraints.power_balance.expression': 'sum(p * p, over=generator) == load'},
            id='degree-two-in-a-constraint',
        ),
    ],
)
def test_inside_the_language(patch):
    """Each of these lowers, so both lanes accept it."""
    schema_of(DISPATCH, **patch)


@pytest.mark.parametrize(
    ('patch', 'match'),
    [
        pytest.param(_objective('sum(cost / p)'), 'divisor contains variables', id='an-expression-the-file-writes'),
        pytest.param(
            {
                'expressions': {'price': 'dual(power_balance)'},
                'objective': {'sense': 'minimize', 'expression': 'sum(p * cost) + sum(price)'},
            },
            'a dual exists only after a solve',
            id='an-expression-the-math-reads-is-checked-where-it-is-read',
        ),
    ],
)
def test_outside_the_language_is_a_load_error(patch, match):
    """The refusal reaches the caller through ``sps.check``, with no data attached.

    Two rows, one per position the verb has to reach; the rules themselves are
    swept in mathspec's ``test_degree.py``. A named expression is checked where
    the math reads it, so a dual smuggled into the objective through one is
    refused there, and one the math never reads is held to nothing.
    """
    with pytest.raises(LanguageError, match=match):
        sps.check(schema_of(DISPATCH, **patch))


def test_an_unknown_operator_names_its_context_and_teaches_the_rewrite():
    """The message is the whole test: an error that pointed at another lane
    would be telling the user to leave the language rather than restate it."""
    patch = {'constraints.power_balance.expression': 'my_helper(p, over=generator) == load'}
    with pytest.raises(LanguageError, match='my_helper') as exc:
        schema_of(DISPATCH, **patch)

    reason = str(exc.value)
    assert 'power_balance' in reason, 'the reason carries its context'
    assert 'macro' in reason, 'and the rewrite, rather than a pointer to another lane'
    assert 'linopy' not in reason.lower()
