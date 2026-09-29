"""What an objective sums, when its terms do not carry the same dims.

An objective is scalar or the file does not load, so every reduction in it is
one somebody wrote. ``sum(a) + sum(b)`` totals each term over its own dims,
while ``sum(a + b)`` broadcasts and counts each once per coordinate of the
other. The two differ only when the terms' dims differ, as in a sparse
``(snapshot, node, tech)`` variable beside a dense ``(snapshot, node,
carrier)`` one (#197).
"""

from __future__ import annotations

import pytest
from mathspec import to_spec

from tests.differential import differential
from tests.oracle import pd  # through the guard: a bare import would beat it

#: Two variables pinned to 1 on disjoint dims, so the objective is arithmetic
#: with no optimisation left in it: whatever comes out is what was summed.
DISJOINT_SPEC = {
    'dimensions': {
        'i': {'dtype': 'int'},
        'j': {'dtype': 'int'},
        'k': {'dtype': 'int'},
    },
    'parameters': {'a': {'dims': ['i']}, 'b': {'dims': ['j']}, 'c': {'dims': ['k']}},
    'variables': {
        'x': {'dims': ['i'], 'bounds': {'lower': 1, 'upper': 1}},
        'y': {'dims': ['j'], 'bounds': {'lower': 1, 'upper': 1}},
    },
    'constraints': {'floor': {'dims': ['i'], 'expression': 'x >= 0'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(x * a) + sum(y * b)'},
}


@pytest.fixture
def data():
    """``sum(x * a) == 2``, ``sum(y * b) == 30``, ``sum(c) == 200``.

    Distinct enough that a broadcast shows up as a different number rather than
    a coincidence: broadcasting the first two gives 66, not 32.
    """
    return {
        'i': [0, 1],
        'j': [0, 1, 2],
        'k': [0, 1],
        'a': pd.Series([1.0, 1.0], index=pd.Index([0, 1], name='i')),
        'b': pd.Series([10.0, 10.0, 10.0], index=pd.Index([0, 1, 2], name='j')),
        'c': pd.Series([100.0, 100.0], index=pd.Index([0, 1], name='k')),
    }


@pytest.mark.parametrize(
    ('expression', 'expected'),
    [
        pytest.param('sum(x * a) + sum(y * b)', 32.0, id='a-sum-per-term'),
        pytest.param('sum(x * a + y * b)', 66.0, id='one-sum-around-both-broadcasts'),
        pytest.param('sum(x * a) - sum(y * b)', -28.0, id='a-difference'),
        pytest.param('-(sum(x * a) + sum(y * b))', -32.0, id='negated'),
        pytest.param('sum(x * a * c) + sum(y * b * c)', 6400.0, id='a-factor-in-each-term'),
        pytest.param('sum((x * a + y * b) * c)', 13200.0, id='a-factor-over-the-broadcast-group'),
        pytest.param('(sum(x * a) + sum(y * b)) / 2', 16.0, id='a-divisor-applied-to-the-group'),
        pytest.param('sum(x * a, over=i) + sum(y * b, over=j)', 32.0, id='the-dims-named-one-at-a-time'),
    ],
)
def test_where_the_sum_is_written_decides_what_it_counts(data, expression, expected):
    """Two readings, both sayable, and the bracket is what picks.

    ``differential`` asserts the two lanes agree; these pairs pin the right
    number. 32 against 66, and 6400 against 13200, differ only in where the
    sum's bracket closes.
    """
    spec = {**DISJOINT_SPEC, 'objective': {'sense': 'minimize', 'expression': expression}}
    with differential(spec, data) as run:
        assert run.oracle == pytest.approx(expected)


#: #1046's model: a bracketed addition under a product, its branches on
#: different dims. The two readings differ by a factor of |j| on the first
#: term.
BRACKETED_SPEC = {
    'dimensions': {'i': {'dtype': 'int'}, 'j': {'dtype': 'int'}},
    'parameters': {'c': {'dims': ['i']}},
    'variables': {
        'x': {'dims': ['i'], 'bounds': {'lower': 1, 'upper': 1}},
        'y': {'dims': ['j'], 'bounds': {'lower': 1, 'upper': 1}},
    },
    'constraints': {'floor': {'dims': ['i'], 'expression': 'x >= 0'}},
}


@pytest.mark.parametrize(
    ('expression', 'expected'),
    [
        pytest.param('sum(c * (x + y))', 660.0, id='the-bracket-broadcasts-and-the-page-says-so'),
        pytest.param('sum(c * x) + sum(c * y)', 440.0, id='distributed-by-hand-is-a-different-model'),
    ],
)
def test_a_bracketed_addition_under_a_product_means_what_it_prints(expression, expected):
    """#1046: ``c * (x + y)`` carrying dims is refused, so both readings are written down.

    30 per unit of ``x[0]`` where the sum closes outside the bracket, 10 where
    it closes around each term.
    """
    data = {'i': [0, 1], 'j': [0, 1, 2], 'c': pd.Series([10.0, 100.0], index=pd.Index([0, 1], name='i'))}
    spec = {**BRACKETED_SPEC, 'objective': {'sense': 'minimize', 'expression': expression}}
    with differential(spec, data, lp=True) as run:
        assert run.oracle == pytest.approx(expected)


def test_an_objective_carrying_dims_is_refused_with_the_wrapper_named():
    """An objective that carries dims is a load error that asks for a `sum` around each term."""
    from specsolve.errors import DimensionError

    spec = {**DISJOINT_SPEC, 'objective': {'sense': 'minimize', 'expression': 'x * a + y * b'}}
    with pytest.raises(DimensionError, match=r"carries dims \['i', 'j'\].*Wrap each additive term"):
        to_spec(spec)


#: No `objective:` at all — the constraints are the whole question, and the
#: answer is whether they can be met. `need` sits inside the caps, so they can.
FEASIBILITY_SPEC = {
    'dimensions': {'g': {'dtype': 'str'}},
    'parameters': {'cap': {'dims': ['g']}, 'need': {'dims': []}},
    'variables': {'x': {'dims': ['g'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
    'constraints': {'meet': {'dims': [], 'expression': 'sum(x, over=g) >= need'}},
}


def test_a_model_with_no_objective_is_a_feasibility_problem(tmp_path):
    """Both lanes build it, and the answer is a point rather than an optimum (#845).

    Nothing optimises, so the objective value is the zero the sink was handed.
    """
    import yaml as pyyaml

    import specsolve as sps
    from tests.oracle import specsolve_linopy

    sources = {'g': ['wind', 'gas'], 'cap': {'wind': 40.0, 'gas': 100.0}, 'need': 90.0}

    path = tmp_path / 'feasibility.yaml'
    path.write_text(pyyaml.safe_dump(FEASIBILITY_SPEC))
    linopy_lane = specsolve_linopy.build(path, sources)
    assert 'meet' in linopy_lane.constraints, 'the linopy lane built the same file'

    with sps.solve(FEASIBILITY_SPEC, sources) as result:
        assert result.is_ok, 'the constraints can be met, so this is not a failed solve'
        assert result.objective == 0.0, 'nothing was optimised, so the objective is the zero it was given'
        served = result.primal('x')['value'].sum()
        assert served == pytest.approx(90.0), 'the constraint is the whole model, so it binds'

    lp = sps.write(FEASIBILITY_SPEC, sources, tmp_path / 'feasibility.lp')
    assert 'obj:\n\ns.t.' in lp.read_text(), 'the objective section is written, and is empty'


def test_a_model_with_no_objective_still_says_when_it_cannot_be_met():
    """The answer a feasibility problem exists to give."""
    import specsolve as sps

    sources = {'g': ['wind', 'gas'], 'cap': {'wind': 40.0, 'gas': 10.0}, 'need': 90.0}
    with sps.solve(FEASIBILITY_SPEC, sources) as result:
        assert not result.is_ok
        assert result.termination_condition == 'infeasible'
