"""`check(spec, sink=...)`: the second axis, asked with no data attached.

Bare `check` stays silent about sinks, the answer needs no data and no
installed solver, and a refusal names the sinks that would have taken it.
"""

from __future__ import annotations

import re
import warnings

import pytest
from mathspec import to_spec

import specsolve as sps
from specsolve.errors import SpecsolveError, SpecsolveWarning
from specsolve.relational import sinks
from specsolve.relational.sinks.capabilities import Capabilities
from specsolve.relational.sinks.solvers import SOLVERS
from specsolve.relational.sinks.writers import WRITERS

#: A pure LP: every sink takes it whole.
PLAIN = {
    'dimensions': {'g': {'dtype': 'str'}},
    'parameters': {'cost': {'dims': ['g']}},
    'variables': {'p': {'dims': ['g'], 'bounds': {'lower': 0, 'upper': 10}}},
    'constraints': {'total': {'dims': [], 'expression': 'sum(p, over=g) <= 5'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(p * cost, over=g)'},
}

#: The same model with a set on it, which HiGHS has no concept of.
WITH_A_SET = PLAIN | {'sos': {'pick': {'variable': 'p', 'along': 'g', 'type': 1}}}

#: The same model at degree 2, in each of the two positions the language takes it.
WITH_A_QUADRATIC_OBJECTIVE = PLAIN | {'objective': {'sense': 'minimize', 'expression': 'sum(p * p, over=g)'}}
WITH_A_QUADRATIC_ROW = PLAIN | {
    'constraints': {**PLAIN['constraints'], 'ball': {'dims': ['g'], 'expression': 'p * p <= 9'}}
}


def _warnings(spec, **kwargs) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        sps.check(spec, **kwargs)
    return [str(w.message) for w in caught if issubclass(w.category, SpecsolveWarning)]


@pytest.mark.parametrize('sink', ['highs', 'gurobi', '.lp'])
def test_a_plain_lp_is_silent_on_every_sink(sink):
    assert _warnings(PLAIN, sink=sink) == [], 'a model inside the common subset must say nothing about portability'


def test_bare_check_says_nothing_about_portability():
    """The default stays silent about a sink nobody named."""
    assert _warnings(WITH_A_SET) == []


def test_a_set_on_highs_is_refused_naming_the_expansion():
    """Nothing is rewritten at the hand-off, so the refusal names the language's own way out."""
    with pytest.raises(SpecsolveError, match="'highs' sink cannot take special-ordered sets") as refusal:
        sps.check(WITH_A_SET, sink='highs')
    assert 'gurobi' in str(refusal.value), 'the sinks that take a set are named'
    assert 'expand()' in str(refusal.value), 'and so is the expansion that writes one out for the sinks that do not'


@pytest.mark.parametrize('sink', ['gurobi', '.lp'])
def test_a_set_is_silent_on_the_sinks_that_carry_one(sink):
    """Gurobi branches on a set and LP text writes it, so neither says anything."""
    assert _warnings(WITH_A_SET, sink=sink) == []


def test_an_unknown_sink_names_the_ones_there_are():
    with pytest.raises(SpecsolveError, match='unknown sink') as refused:
        sps.check(PLAIN, sink='cplex')
    assert re.search(r'\.lp, \.mps, gurobi, highs, xpress', str(refused.value)), (
        'the refusal lists every sink there is, so a reader picks one instead of guessing'
    )


def test_the_question_needs_no_solver_installed(monkeypatch):
    """`check(sink=...)` answers for a solver that is not installed; `solver()` refuses it."""
    monkeypatch.setattr(SOLVERS['gurobi'], 'is_available', classmethod(lambda cls: False))
    assert _warnings(WITH_A_SET, sink='gurobi') == []
    with pytest.raises(ModuleNotFoundError):
        sinks.solver('gurobi')


def _refusing(spec) -> set[str]:
    """Every shipped sink that would turn *model* away."""
    return {name for name in (*SOLVERS, *WRITERS) if sinks.refusal(_program(spec), name) is not None}


def test_which_shipped_sinks_refuse_what_the_language_can_now_say():
    """Which shipped sink refuses a set or degree 2, named individually."""
    assert _refusing(WITH_A_SET) == {'highs'}, 'the one shipped sink with no SOS concept, and nothing rewrites for it'
    assert _refusing(WITH_A_QUADRATIC_OBJECTIVE) == {'xpress', '.mps'}, (
        'the two with no path for a Hessian: xpress ships one and this package hands it none, '
        'and MPS spells it in a section this writer does not write'
    )
    assert _refusing(WITH_A_QUADRATIC_ROW) == {'highs', 'xpress', '.mps'}, (
        'and a quadratic row, which HiGHS has no entry point for at all'
    )


def test_a_sink_that_takes_nothing_is_refused_by_name_and_offered_the_others(monkeypatch):
    """The refusal path itself, which no shipped sink reaches, through a stub."""

    class Stub:
        capabilities = Capabilities(supports={})

    monkeypatch.setitem(SOLVERS, 'stub', Stub)
    message = sinks.refusal(_program(WITH_A_SET), 'stub')
    assert message is not None
    assert "'stub'" in message, 'the refusal names the sink'
    assert 'special-ordered sets' in message, "the refusal names the construct, in the modeller's own words"
    assert 'gurobi' in message and '.lp' in message and 'highs' not in message, 'and the sinks that do take it'


def test_a_sink_excluding_a_pair_says_so_rather_than_denying_the_half(monkeypatch):
    """A sink that takes two constructs apart and refuses them together says so."""

    class Stub:
        capabilities = Capabilities(
            supports={'integrality': 'native', 'sos': 'native'},
            excludes=(frozenset({'sos', 'integrality'}),),
        )

    monkeypatch.setitem(SOLVERS, 'stub', Stub)
    integral = WITH_A_SET | {
        'variables': {'p': {'dims': ['g'], 'domain': 'integer', 'bounds': {'lower': 0, 'upper': 10}}}
    }
    message = sinks.refusal(_program(integral), 'stub')
    assert message is not None
    assert 'separately and refuses them together' in message
    assert 'binary or integer variables' in message and 'special-ordered sets' in message


def test_a_suffix_is_a_sink_however_the_path_spelled_it():
    """``check(spec, sink=path.suffix)`` takes ``.LP`` as ``write`` does."""
    assert sinks.sink_capabilities('.LP') is sinks.sink_capabilities('.lp')


def test_a_refusal_does_not_swallow_the_solver_independent_advice(recwarn):
    """Naming a sink keeps the solver-independent advice."""
    unused = PLAIN | {'dimensions': PLAIN['dimensions'] | {'spare': {'dtype': 'str'}}}

    class Stub:
        capabilities = Capabilities(supports={})

    bare = _warnings(unused)
    assert bare, 'the premise: this model has something to say without any sink named'

    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(SOLVERS, 'stub', Stub)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with pytest.raises(SpecsolveError):
                sps.check(unused | {'sos': {'pick': {'variable': 'p', 'along': 'g', 'type': 1}}}, sink='stub')
        assert [str(w.message) for w in caught] == bare, 'the advice a bare check gives is issued before the raise'


def _program(spec):
    """The lowered plan a capability question is asked of."""
    return to_spec(spec).program
