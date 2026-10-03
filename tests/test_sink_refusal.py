"""Whether a sink takes a model is asked of the model that was built, not of the file.

`Model.check(sink)` asks it with no solve. The answer needs no installed
solver — it is read off a declared table — and a refusal names the sinks that
would have taken it. The file is an upper
bound: a square the data prices at zero, an integer variable no column is
built for and a set whose variable has no column ask for nothing.
"""

from __future__ import annotations

import re

import polars as pl
import pytest

import specsolve as sps
from specsolve.errors import SpecsolveError
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

SOURCES = {'g': ['a', 'b'], 'cost': pl.DataFrame({'g': ['a', 'b'], 'value': [1.0, 3.0]})}

#: The same model with a set on it, which HiGHS has no concept of.
WITH_A_SET = PLAIN | {'sos': {'pick': {'variable': 'p', 'along': 'g', 'type': 1}}}

#: The same model at degree 2, in each of the two positions the language takes it.
WITH_A_QUADRATIC_OBJECTIVE = PLAIN | {'objective': {'sense': 'minimize', 'expression': 'sum(p * p, over=g)'}}
WITH_A_QUADRATIC_ROW = PLAIN | {
    'constraints': {**PLAIN['constraints'], 'ball': {'dims': ['g'], 'expression': 'p * p <= 9'}}
}

#: The same model with an integer decision.
INTEGRAL = PLAIN | {'variables': {'p': {'dims': ['g'], 'domain': 'integer', 'bounds': {'lower': 0, 'upper': 10}}}}


def _handoff(spec, sources=SOURCES):
    """The built model a capability question is asked of, as a stub sink is asked it."""
    with sps.build(spec, dict(sources)) as model:
        return model._engine._model.handoff


def _refusal(spec, sink: str, sources=SOURCES) -> str | None:
    """What `Model.check` says of *sink*, or ``None`` where it takes the build."""
    with sps.build(spec, dict(sources)) as model:
        try:
            model.check(sink)
        except SpecsolveError as refused:
            return str(refused)
    return None


def _refusing(spec, sources=SOURCES) -> set[str]:
    """Every shipped sink that would turn the built model away."""
    return {name for name in (*SOLVERS, *WRITERS) if _refusal(spec, name, sources) is not None}


def test_a_plain_lp_is_taken_by_every_sink():
    assert _refusing(PLAIN) == set(), 'a model inside the common subset is turned away nowhere'


def test_a_set_on_highs_is_refused_before_the_load_naming_the_expansion():
    """Nothing is rewritten at the hand-off, so the refusal names the language's own way out."""
    with pytest.raises(SpecsolveError, match="'highs' sink cannot take special-ordered sets") as refusal:
        sps.solve(WITH_A_SET, SOURCES)
    assert 'gurobi' in str(refusal.value), 'the sinks that take a set are named'
    assert 'expand()' in str(refusal.value), 'and so is the expansion that writes one out for the sinks that do not'


def test_an_unknown_sink_names_the_ones_there_are():
    with sps.build(PLAIN, dict(SOURCES)) as model, pytest.raises(SpecsolveError, match='unknown sink') as refused:
        model.check('cplex')
    assert re.search(r'\.lp, \.mps, gurobi, highs, linopy, pyomo, xpress', str(refused.value)), (
        'the refusal lists every sink there is, so a reader picks one instead of guessing'
    )


def test_the_question_needs_no_solver_installed(monkeypatch):
    """The table answers for a solver that is not installed; `solver()` refuses it."""
    monkeypatch.setattr(SOLVERS['gurobi'], 'is_available', classmethod(lambda cls: False))
    assert _refusal(WITH_A_SET, 'gurobi') is None
    with pytest.raises(ModuleNotFoundError):
        sinks.solver('gurobi')


def test_which_shipped_sinks_refuse_what_the_language_can_say():
    """Which shipped sink refuses a set or degree 2, named individually."""
    assert _refusing(WITH_A_SET) == {'highs'}, 'the one shipped sink with no SOS concept, and nothing rewrites for it'
    assert _refusing(WITH_A_QUADRATIC_OBJECTIVE) == {'xpress', '.mps'}, (
        'the two with no path for a Hessian: xpress ships one and this package hands it none, '
        'and MPS spells it in a section this writer does not write'
    )
    assert _refusing(WITH_A_QUADRATIC_ROW) == {'highs', 'xpress', '.mps'}, (
        'and a quadratic row, which HiGHS has no entry point for at all'
    )


@pytest.mark.parametrize(
    ('spec', 'sources'),
    [
        pytest.param(
            INTEGRAL
            | {'objective': {'sense': 'minimize', 'expression': 'sum(p * cost, over=g) + sum(p * p * cost, over=g)'}},
            SOURCES | {'cost': pl.DataFrame({'g': ['a', 'b'], 'value': [0.0, 0.0]})},
            id='a-square-the-data-prices-at-zero-beside-integrality',
        ),
        pytest.param(
            PLAIN
            | {'constraints': {**PLAIN['constraints'], 'ball': {'dims': ['g'], 'expression': 'p * p * cost <= 9'}}},
            SOURCES | {'cost': pl.DataFrame({'g': ['a', 'b'], 'value': [0.0, 0.0]})},
            id='a-quadratic-row-whose-squares-are-all-zero',
        ),
        pytest.param(
            INTEGRAL
            | {
                'variables': {
                    'p': {'dims': ['g'], 'bounds': {'lower': 0, 'upper': 10}},
                    'n': {'dims': ['g'], 'domain': 'integer', 'where': 'cost > 100', 'bounds': {'lower': 0}},
                },
                'objective': {'sense': 'minimize', 'expression': 'sum(p * cost, over=g) + sum(p * p, over=g)'},
            },
            SOURCES,
            id='an-integer-variable-no-column-is-built-for-beside-a-square',
        ),
        pytest.param(
            PLAIN
            | {
                'variables': {
                    **PLAIN['variables'],
                    's': {'dims': ['g'], 'where': 'cost > 100', 'bounds': {'lower': 0, 'upper': 1}},
                },
                'sos': {'pick': {'variable': 's', 'along': 'g', 'type': 1}},
            },
            SOURCES,
            id='a-set-on-a-variable-no-column-is-built-for',
        ),
    ],
)
def test_what_the_file_declares_and_the_data_never_builds_asks_highs_for_nothing(spec, sources):
    """The file is an upper bound on what a build produces, and the sink is asked about the build.

    Each case declares a construct HiGHS refuses — a Hessian beside
    integrality, a quadratic row, a set — and feeds data under which the build
    produces none of it. The built model is an LP, and HiGHS solves it.
    """
    assert _refusal(spec, 'highs', sources) is None, 'the built model asks for nothing highs lacks'
    with sps.solve(spec, dict(sources)) as result:
        assert result.is_ok


@pytest.mark.parametrize(
    ('spec', 'sink', 'verb'),
    [
        pytest.param(WITH_A_SET, 'highs', lambda model, tmp: model.solve('highs'), id='solve'),
        pytest.param(WITH_A_QUADRATIC_ROW, '.mps', lambda model, tmp: model.write(tmp / 'm.mps'), id='write'),
    ],
)
def test_check_refuses_exactly_what_the_verb_refuses(spec, sink, verb, tmp_path):
    """One path, one message: a CI job that checks instead of solving learns the same thing."""
    with sps.build(spec, dict(SOURCES)) as model:
        with pytest.raises(SpecsolveError) as checked:
            model.check(sink)
        with pytest.raises(SpecsolveError) as acted:
            verb(model, tmp_path)
    assert str(checked.value) == str(acted.value)


def test_a_sink_that_takes_nothing_is_refused_by_name_and_offered_the_others(monkeypatch):
    """The refusal path itself, which no shipped sink reaches, through a stub."""

    class Stub:
        capabilities = Capabilities(supports=frozenset({}))

    monkeypatch.setitem(SOLVERS, 'stub', Stub)
    message = sinks.refusal(_handoff(WITH_A_SET), 'stub')
    assert message is not None
    assert "'stub'" in message, 'the refusal names the sink'
    assert 'special-ordered sets' in message, "the refusal names the construct, in the modeller's own words"
    assert 'gurobi' in message and '.lp' in message and 'highs' not in message, 'and the sinks that do take it'


def test_a_sink_excluding_a_pair_says_so_rather_than_denying_the_half(monkeypatch):
    """A sink that takes two constructs apart and refuses them together says so."""

    class Stub:
        capabilities = Capabilities(
            supports=frozenset({'integrality', 'sos'}),
            excludes=(frozenset({'sos', 'integrality'}),),
        )

    monkeypatch.setitem(SOLVERS, 'stub', Stub)
    message = sinks.refusal(_handoff(INTEGRAL | {'sos': WITH_A_SET['sos']}), 'stub')
    assert message is not None
    assert 'separately and refuses them together' in message
    assert 'binary or integer variables' in message and 'special-ordered sets' in message


def test_a_suffix_is_a_sink_however_the_path_spelled_it():
    """``write`` takes ``.LP`` as ``.lp``, and so does the table."""
    assert sinks.sink_capabilities('.LP') is sinks.sink_capabilities('.lp')
