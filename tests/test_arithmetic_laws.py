"""The algebraic laws the language promises — and the ones it deliberately breaks.

**Laws** are spellings that must produce the same model, each solved through
``differential``. **Non-laws** are spellings equal in ordinary algebra and not
equal here, because absence is a state (#311); they are asserted to differ,
with the values written down.

The fixture keeps one masked variable (``y``, absent at ``f=b``) and one total
one (``x``). The wide cases at the end vary one thing at a time: the
reduction, the number of masks, where the mask sits, and where the absence
comes from.
"""

from __future__ import annotations

import polars as pl
import pytest

import specsolve as sps
from specsolve.errors import DataError
from tests.conftest import law_data, law_spec, override, schema_of
from tests.differential import RTOL, both_lanes_refuse, differential
from tests.oracle import pd, specsolve_linopy

# ---------------------------------------------------------------------------
# the fixture: `x` total, `y` absent at f=b, `w` a dense coefficient
# ---------------------------------------------------------------------------

DATA = law_data()


def _spec(
    expression: str,
    *,
    objective: str = 'sum(x)',
    dims: list[str] | None = None,
    also: dict | None = None,
) -> dict:
    """The shared model, over ``t`` unless the case says otherwise."""
    return law_spec(
        expression,
        dims=dims if dims is not None else ['t'],
        objective=objective,
        also=also,
    )


def _objective_of(
    expression: str,
    objective: str = 'sum(x)',
    dims: list[str] | None = None,
    also: dict | None = None,
) -> float:
    """Solve *expression* on both lanes and the LP file; return the agreed value."""
    with differential(_spec(expression, objective=objective, dims=dims, also=also), DATA, lp=True) as run:
        return float(run.result.objective)


# ---------------------------------------------------------------------------
# laws — these must hold
# ---------------------------------------------------------------------------

#: Rewrites that must build the same model. ``reduction-is-linear`` holds only
#: while nothing is absent; ``commutative-add-under-absence`` holds because both
#: spellings carry the same absence.
LAWS = [
    pytest.param(
        'sum(x + w * x, over=f) <= 120',
        'sum(w * x + x, over=f) <= 120',
        id='commutative-add',
    ),
    pytest.param(
        'sum((x + w * x) + x, over=f) <= 120',
        'sum(x + (w * x + x), over=f) <= 120',
        id='associative-add',
    ),
    pytest.param(
        'sum(x - w * x, over=f) <= 120',
        'sum(x + (-1) * w * x, over=f) <= 120',
        id='subtraction-is-negated-addition',
    ),
    pytest.param(
        'sum(w * (x + x), over=f) <= 120',
        'sum(w * x + w * x, over=f) <= 120',
        id='distributive-over-a-variable-free-factor',
    ),
    pytest.param(
        'sum((x + x) / w, over=f) <= 120',
        'sum(x / w + x / w, over=f) <= 120',
        id='distributive-over-a-divisor',
    ),
    pytest.param(
        'sum(x + w * x, over=f) <= 120',
        'sum(x, over=f) + sum(w * x, over=f) <= 120',
        id='reduction-is-linear-when-every-operand-is-total',
    ),
    pytest.param(
        "sum(shift(shift(x, along=t, offset=1, edge='wrap'), along=t, offset=-1, edge='wrap'), over=f) <= 120",
        'sum(x, over=f) <= 120',
        id='cyclic-shift-is-invertible',
    ),
    pytest.param(
        'sum(y + w * y, over=f) <= 120',
        'sum(w * y + y, over=f) <= 120',
        id='commutative-add-under-absence',
    ),
]


@pytest.mark.parametrize(('left', 'right'), LAWS)
def test_the_two_spellings_build_the_same_model(left, right):
    assert _objective_of(left) == pytest.approx(_objective_of(right), rel=RTOL)


# ---------------------------------------------------------------------------
# non-laws — equal in ordinary algebra, deliberately unequal here
# ---------------------------------------------------------------------------


def test_a_reduction_does_not_distribute_over_addition_when_an_operand_is_absent():
    """``sum(x + y)`` and ``sum(x) + sum(y)`` answer different questions (#311).

    ``y`` is absent at ``f=b``, so the summand ``x + y`` is absent there and the
    reduction skips that slot, ``x[b]`` with it. Summing each operand
    separately keeps it.
    """
    together = _objective_of('sum(x + y, over=f) <= 120')
    apart = _objective_of('sum(x, over=f) + sum(y, over=f) <= 120')

    assert together == pytest.approx(400.0, rel=RTOL), 'together binds only at f=a, so x[b] is free to its bound'
    assert apart == pytest.approx(240.0, rel=RTOL), 'apart keeps x[b] in the row, so the cap binds the total'
    assert together != pytest.approx(apart, rel=RTOL), 'the two questions must stay distinguishable'


def test_a_sum_skips_a_parameter_where_a_masked_operand_beside_it_is_absent():
    """The same rule when the present operand is a parameter over fewer dimensions.

    ``w`` is over ``f`` alone and ``y`` over ``f`` and ``t``, absent at ``f=b``.
    The summand ``y + w`` is absent at ``f=b`` for every ``t``, so the row keeps
    ``w[a]`` alone and ``y[a, t]`` may reach ``10 - 2``. The relational lane kept
    ``w[b]`` too, because it restricted only an operand that carried every
    dimension ``y``'s presence is keyed by (#1782).
    """
    assert _objective_of('sum(y + w, over=f) <= 10', objective='sum(y)') == pytest.approx(16.0, rel=RTOL), (
        'each of the two rows caps y[a, t] at 8, not at the 5 that w[b] would leave'
    )


def test_a_term_whose_variable_is_absent_is_not_a_term_worth_zero():
    """``x + y >= k`` is no constraint where ``y`` is absent — not ``x >= k``.

    Compared against the zero-fill reading, spelled as two constraints under
    complementary ``where`` clauses.
    """
    minimise_x = 'sum((-1) * x)'
    propagated = _objective_of('x + y >= 60', objective=minimise_x, dims=['f', 't'])
    zero_filled = _objective_of(
        'x + y >= 60',
        objective=minimise_x,
        dims=['f', 't'],
        also={'c_unsized': {'dims': ['f', 't'], 'where': 'NOT y', 'expression': 'x >= 60'}},
    )

    assert propagated == pytest.approx(-(10.0 + 10.0), rel=RTOL), (
        'f=a: y covers 50 of the 60, so x is pushed to 10. f=b: no row at all, so x falls to 0'
    )
    assert zero_filled == pytest.approx(-(10.0 + 10.0 + 60.0 + 60.0), rel=RTOL), (
        'asking for zero-fill explicitly puts the requirement back at f=b'
    )


def test_absence_zero_says_at_the_declaration_what_two_blocks_said_at_the_rows():
    """``absence: zero`` builds the model the two complementary ``where`` blocks build."""
    minimise_x = 'sum((-1) * x)'
    two_blocks = _objective_of(
        'x + y >= 60',
        objective=minimise_x,
        dims=['f', 't'],
        also={'c_unsized': {'dims': ['f', 't'], 'where': 'NOT y', 'expression': 'x >= 60'}},
    )

    spec = _spec('x + y >= 60', objective=minimise_x, dims=['f', 't'])
    spec['variables']['y']['absence'] = 'zero'
    with differential(spec, DATA, lp=True) as run:
        declared = float(run.result.objective)

    assert declared == pytest.approx(two_blocks, rel=RTOL), (
        'absence: zero builds the model the two complementary blocks build'
    )
    assert declared == pytest.approx(-(10.0 + 10.0 + 60.0 + 60.0), rel=RTOL), (
        'f=b keeps its row, and with y worth zero there x carries the whole 60 itself'
    )


def test_absence_zero_does_not_disturb_a_reduction():
    """A reduction sums the y's that exist, so ``absence: zero`` changes nothing there."""
    total = 'sum(y, over=f) <= 40'
    undefined = _objective_of(total, objective='sum(y)')

    spec = _spec(total, objective='sum(y)')
    spec['variables']['y']['absence'] = 'zero'
    with differential(spec, DATA, lp=True) as run:
        zero = float(run.result.objective)

    assert zero == pytest.approx(undefined, rel=RTOL), (
        'a reduction sums what exists under either reading — absence: zero adds zeros to it'
    )


def test_shift_and_a_filled_shift_are_different_operators():
    """Bare, the vacated slot is absent and the row goes with it (#289); filled, the row survives."""
    bare = _objective_of('sum(x - shift(x, along=t, offset=1), over=f) <= 10')
    filled = _objective_of('sum(x - shift(x, along=t, offset=1, edge=0), over=f) <= 10')

    assert bare != pytest.approx(filled, rel=RTOL), (
        'a bare shift drops the first row; a filled one keeps it, so these cannot agree'
    )


# ---------------------------------------------------------------------------
# the same rules under wider shapes
# ---------------------------------------------------------------------------

#: ``y`` is absent at ``d``; ``v`` at ``b`` and ``d``, so the two masks do not nest.
WIDE_DATA = {
    'gate': pd.Series([True, True, True], index=pd.Index(['a', 'b', 'c'], name='f')),
    'gate2': pd.Series([True, True], index=pd.Index(['a', 'c'], name='f')),
    'w': pd.Series([2.0, 3.0, 4.0, 5.0], index=pd.Index(['a', 'b', 'c', 'd'], name='f')),
}

WIDE_COORDS = {
    'f': pd.DataFrame({'f': ['a', 'b', 'c', 'd']}),
    'grp': pd.DataFrame({'f': ['a', 'b', 'c', 'd'], 'g': ['g0', 'g0', 'g1', 'g1']}),
    'g': pd.Index(['g0', 'g1'], name='g'),
    't': pd.Index([0, 1], name='t'),
}

PLAIN_COORDS = {'f': pd.Index(['a', 'b', 'c', 'd'], name='f'), 't': pd.Index([0, 1], name='t')}


def _wide_objective_of(expression: str, *, dims: list[str]) -> float:
    """The wide fixture solved through both lanes; ``g`` and ``grp`` only for the grouped cases (#488)."""
    grouped = 'g' in dims
    dimensions = {'g': {}, 'f': {}, 't': {'dtype': 'int'}} if grouped else {'f': {}, 't': {'dtype': 'int'}}
    spec = {
        'dimensions': dimensions,
        **({'relations': {'grp': {'key': 'f', 'values': 'g'}}} if grouped else {}),
        'parameters': {
            'gate': {'dims': ['f'], 'dtype': 'bool'},
            'gate2': {'dims': ['f'], 'dtype': 'bool'},
            'w': {'dims': ['f']},
        },
        'variables': {
            'x': {'dims': ['f', 't'], 'bounds': {'lower': 0, 'upper': 100}},
            'y': {'dims': ['f', 't'], 'where': 'gate', 'bounds': {'lower': 0, 'upper': 50}},
            'v': {'dims': ['f', 't'], 'where': 'gate2', 'bounds': {'lower': 0, 'upper': 50}},
        },
        'constraints': {'c': {'dims': dims, 'expression': expression}},
        'objective': {'sense': 'maximize', 'expression': 'sum(x)'},
    }
    with differential(spec, WIDE_DATA | (WIDE_COORDS if grouped else PLAIN_COORDS), lp=True) as run:
        return float(run.result.objective)


def test_sum_does_not_distribute_over_addition_either():
    """A grouped `sum` is a reduction, so the non-law applies to it unchanged (#314)."""
    together = _wide_objective_of('sum(x + y, by=grp, over=f, into=g) <= 120', dims=['g', 't'])
    apart = _wide_objective_of(
        'sum(x, by=grp, over=f, into=g) + sum(y, by=grp, over=f, into=g) <= 120', dims=['g', 't']
    )

    assert together == pytest.approx(640.0, rel=RTOL)
    assert apart == pytest.approx(480.0, rel=RTOL)
    assert together != pytest.approx(apart, rel=RTOL)


def test_two_masks_intersect_rather_than_applying_one_at_a_time():
    """Three operands, two different masks — the summand needs *all* of them.

    `y` is absent at `d`, `v` at `b` and `d`, so the summand exists only at
    `a` and `c`.
    """
    together = _wide_objective_of('sum(x + y + v, over=f) <= 120', dims=['t'])
    apart = _wide_objective_of('sum(x, over=f) + sum(y, over=f) + sum(v, over=f) <= 120', dims=['t'])

    assert together == pytest.approx(640.0, rel=RTOL)
    assert apart == pytest.approx(240.0, rel=RTOL)


def test_a_broadcast_coefficient_does_not_move_where_the_summand_exists():
    """A coefficient is not absence, so the separation comes from `y` alone."""
    together = _wide_objective_of('sum(w * x + y, over=f) <= 120', dims=['t'])
    apart = _wide_objective_of('sum(w * x, over=f) + sum(y, over=f) <= 120', dims=['t'])

    assert together == pytest.approx(320.0, rel=RTOL)
    assert apart == pytest.approx(120.0, rel=RTOL)


def test_a_mask_on_a_dim_the_reduction_does_not_touch_still_propagates():
    """Absence does not have to live on the summed dim to reach the summand.

    The mask is on `t` and the reduction is over `f`: at `t=1` the whole
    summand is absent for every `f`.
    """
    spec = {
        'dimensions': {'f': {}, 't': {'dtype': 'int'}},
        'parameters': {'tgate': {'dims': ['t'], 'dtype': 'bool'}},
        'variables': {
            'x': {'dims': ['f', 't'], 'bounds': {'lower': 0, 'upper': 100}},
            'y': {'dims': ['f', 't'], 'where': 'tgate', 'bounds': {'lower': 0, 'upper': 50}},
        },
        'constraints': {'c': {'dims': ['t'], 'expression': 'sum(x + y, over=f) <= 120'}},
        'objective': {'sense': 'maximize', 'expression': 'sum(x)'},
    }
    data = {'tgate': pd.Series([True], index=pd.Index([0], name='t'))}
    index = {'f': pd.Index(['a', 'b'], name='f'), 't': pd.Index([0, 1], name='t')}

    with differential(spec, data | index, lp=True) as run:
        assert float(run.result.objective) == pytest.approx(320.0, rel=RTOL), (
            't=0 the row binds; t=1 the summand is absent everywhere, so both x are free'
        )


def test_shift_created_absence_reaches_a_reduction_like_any_other():
    """A bare `shift`'s vacated slot is absent inside a reduction, as a mask's is (#289/#291).

    ``v`` appears only under the shift, so ``t=0`` is the only place the rule
    can show; shifting ``x`` itself would let the ``t=1`` row mask the result.
    """
    spec = {
        'dimensions': {'f': {}, 't': {'dtype': 'int'}},
        'parameters': {},
        'variables': {
            'x': {'dims': ['f', 't'], 'bounds': {'lower': 0, 'upper': 100}},
            'v': {'dims': ['f', 't'], 'bounds': {'lower': 0, 'upper': 100}},
        },
        'constraints': {'c': {'dims': ['t'], 'expression': 'sum(x + shift(v, along=t, offset=1), over=f) <= 120'}},
        'objective': {'sense': 'maximize', 'expression': 'sum(x)'},
    }
    index = {'f': pd.Index(['a', 'b'], name='f'), 't': pd.Index([0, 1], name='t')}

    with differential(spec, {} | index, lp=True) as run:
        assert float(run.result.objective) == pytest.approx(320.0, rel=RTOL), (
            't=0: the shifted operand vacates, so both x[.,0] stay at 100; t=1: the row binds'
        )


#: The divisor fixture, varied per test by :func:`override`.
DIVISOR_SPEC = {
    'dimensions': {'f': {'dtype': 'str'}},
    'parameters': {'d': {'dims': ['f']}},
    'variables': {'x': {'dims': ['f'], 'bounds': {'lower': 0, 'upper': 100}}},
    'constraints': {'c': {'dims': ['f'], 'expression': 'x / d <= 10'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(x)'},
}

#: ``d`` covers ``a`` and not ``b`` — the gap every case below turns on.
SPARSE_D = {'f': ['a', 'b'], 'd': pd.Series([2.0], index=pd.Index(['a'], name='f'))}


def test_a_sparse_divisor_is_refused_rather_than_read_as_zero():
    """A divisor has no defensible fill (#312).

    0 divides by zero, 1 silently rescales, and dropping the term rewrites
    what the constraint asserts.
    """
    with pytest.raises(DataError, match='used as a divisor'), differential(DIVISOR_SPEC, SPARSE_D) as run:
        _ = run.result.objective

    dense = {'f': ['a', 'b'], 'd': pd.Series([2.0, 5.0], index=pd.Index(['a', 'b'], name='f'))}
    with differential(DIVISOR_SPEC, dense, lp=True) as run:
        assert float(run.result.objective) == pytest.approx(70.0, rel=RTOL), (
            'covered, the same model builds and the row binds on both lanes'
        )


def test_a_sparse_divisor_written_as_a_power_is_named_and_refused_on_both_lanes():
    """The same gap under `**`: both lanes name `d` and refuse in one sentence (#1536)."""
    spec = override(DIVISOR_SPEC, **{'constraints.c.expression': 'x / (d ** 2) <= 10'})
    both_lanes_refuse(spec, SPARSE_D, match="parameter 'd' is used as a divisor")


def test_a_divisor_that_adds_is_added_up_before_it_divides():
    """`x / (d - 1)` divides by one number per coordinate, on both lanes.

    Taken term by term it would read `x / d - x / 1`, a different row.
    """
    spec = override(DIVISOR_SPEC, **{'constraints.c.expression': 'x / (d - 1) <= 10'})
    dense = {'f': ['a', 'b'], 'd': pd.Series([2.0, 5.0], index=pd.Index(['a', 'b'], name='f'))}
    with differential(spec, dense, lp=True) as run:
        assert float(run.result.objective) == pytest.approx(50.0, rel=RTOL), (
            'f=a: x <= 10 * 1; f=b: x <= 10 * 4, both under the bound of 100'
        )


def test_a_sparse_divisor_that_adds_is_named_and_refused_on_both_lanes():
    """`1 + d` has no value where `d` has no row: the constant beside it does not fill the gap."""
    spec = override(DIVISOR_SPEC, **{'constraints.c.expression': 'x / (1 + d) <= 10'})
    both_lanes_refuse(spec, SPARSE_D, match="parameter 'd' is used as a divisor")


def test_a_sparse_divisor_in_the_objective_is_refused_too():
    """The refusal holds in the objective, the one declaration with no rows to mask."""
    spec = override(
        DIVISOR_SPEC,
        **{
            'variables.x.bounds.lower': 1,
            'constraints.c.expression': 'x <= 10',
            'objective.sense': 'minimize',
            'objective.expression': 'sum(x / d, over=f)',
        },
    )
    with pytest.raises(DataError, match='used as a divisor'), differential(spec, SPARSE_D) as run:
        _ = run.result.objective


def test_a_sparse_divisor_on_a_constant_side_is_refused_too():
    """`x <= h / d` divides where no variable stands, and is refused the same (#1465)."""
    spec = override(
        DIVISOR_SPEC,
        **{'parameters.h': {'dims': ['f']}, 'constraints.c.expression': 'x <= h / d'},
    )
    data = SPARSE_D | {'h': pd.Series([10.0, 10.0], index=pd.Index(['a', 'b'], name='f'))}
    both_lanes_refuse(spec, data, match="parameter 'd' is used as a divisor")


def test_a_sparse_divisor_on_a_constant_side_has_the_same_escape():
    """On the constant side too, the refusal is keyed to the rows built."""
    spec = override(
        DIVISOR_SPEC,
        **{'parameters.h': {'dims': ['f']}, 'constraints.c.expression': 'x <= h / d', 'constraints.c.where': 'd'},
    )
    data = SPARSE_D | {'h': pd.Series([10.0, 10.0], index=pd.Index(['a', 'b'], name='f'))}
    with differential(spec, data, lp=True) as run:
        assert float(run.result.objective) == pytest.approx(105.0, rel=RTOL), (
            'f=a: the row binds at x <= 5. f=b: masked out, so x runs to its bound'
        )


def test_a_divisor_may_be_sparse_where_the_row_is_masked_out():
    """The check is keyed to the rows built, not the coordinate product."""
    spec = override(
        DIVISOR_SPEC,
        **{
            'parameters.active': {'dims': ['f'], 'dtype': 'bool'},
            'constraints.c.where': 'active',
        },
    )
    data = SPARSE_D | {'active': pd.Series([True], index=pd.Index(['a'], name='f'))}
    with differential(spec, data, lp=True) as run:
        assert float(run.result.objective) == pytest.approx(120.0, rel=RTOL), (
            'f=a: the row binds at x <= 20. f=b: masked out, so x runs to its bound'
        )


@pytest.mark.parametrize(
    ('patch', 'expected'),
    [
        pytest.param({'constraints.c.where': 'd'}, 120.0, id='mask-the-row'),
        pytest.param({'variables.x.where': 'd'}, 20.0, id='mask-the-variable'),
    ],
)
def test_a_sparse_divisor_has_an_escape(patch, expected):
    """Masking the row or the variable lifts the refusal."""
    with differential(override(DIVISOR_SPEC, **patch), SPARSE_D, lp=True) as run:
        assert float(run.result.objective) == pytest.approx(expected, rel=RTOL), (
            'either spelling of "this coordinate has no row" lifts the refusal'
        )


#: ``d`` has a row at every coordinate, and the row at ``b`` is zero.
ZERO_D = {'f': ['a', 'b'], 'd': pd.Series([2.0, 0.0], index=pd.Index(['a', 'b'], name='f'))}
#: ``h`` for the cases that divide on the constant side: 10 everywhere, or 0 at ``b``.
H = {'h': pd.Series([10.0, 10.0], index=pd.Index(['a', 'b'], name='f'))}
H_ZERO_AT_B = {'h': pd.Series([10.0, 0.0], index=pd.Index(['a', 'b'], name='f'))}
WITH_H = {'parameters.h': {'dims': ['f']}}
#: ``r`` has no dimensions, and is zero.
SCALAR_R = {'parameters.r': {'dims': []}}
ZERO_R = {'f': ['a', 'b'], 'd': pd.Series([2.0, 5.0], index=pd.Index(['a', 'b'], name='f')), 'r': 0.0}


@pytest.mark.parametrize(
    'expression',
    [
        pytest.param('x / d <= 10', id='a-term'),
        pytest.param('x * d / d <= 10', id='a-term-zero-over-zero'),
        pytest.param('x / (2 - d) <= 10', id='a-divisor-that-adds-up-to-zero'),
    ],
)
def test_a_zero_divisor_under_a_variable_is_refused_on_both_lanes(expression):
    """A zero divisor under a variable leaves no coefficient, so it is refused, not solved (#1892).

    Both lanes handed the solver an infinite or NaN coefficient with no error:
    `x * r / r >= 1` reported `infeasible`.
    """
    spec = override(DIVISOR_SPEC, **{'constraints.c.expression': expression})
    message = both_lanes_refuse(spec, ZERO_D, match="where the model divides by 'd'")
    assert '1 coefficient(s) are not finite' in message, 'the message counts the one coordinate the zero reaches'


def test_a_zero_divisor_in_the_objective_is_refused_on_both_lanes():
    """The report's own model: `sum(x * d / d)` with `d` zero at `b` solved `optimal` with a NaN objective (#1892)."""
    spec = override(
        DIVISOR_SPEC,
        **{
            'variables.x.bounds.lower': 1,
            'constraints.c.expression': 'x <= 10',
            'objective.sense': 'minimize',
            'objective.expression': 'sum(x * d / d, over=f)',
        },
    )
    with pytest.raises(DataError, match=r"1 coefficient\(s\) are not finite where the model divides by 'd'"):
        sps.build(spec, ZERO_D).close()
    with pytest.raises(DataError, match=r"1 coefficient\(s\) are not finite where the model divides by 'd'"):
        specsolve_linopy.build(schema_of(spec).expand(), ZERO_D)


def test_an_infinite_value_as_a_coefficient_is_refused():
    """A cost of ``inf`` reached the solver as a coefficient, and the solve reported optimal at 0.0.

    Only the relational lane is asked: the linopy lane checks a zero divisor alone.
    """
    spec = override(DIVISOR_SPEC, **{'objective.expression': 'sum(x * d, over=f)'})
    data = {'f': ['a', 'b'], 'd': pd.Series([2.0, float('inf')], index=pd.Index(['a', 'b'], name='f'))}
    with pytest.raises(DataError, match=r'objective: 1 coefficient\(s\) are not finite: a divisor is zero there'):
        sps.build(spec, data).close()


@pytest.mark.parametrize(
    ('patch', 'data'),
    [
        pytest.param({**WITH_H, 'constraints.c.expression': 'x <= h / d'}, ZERO_D | H_ZERO_AT_B, id='zero-over-zero'),
        pytest.param({**SCALAR_R, 'constraints.c.expression': 'x <= 0 / r'}, ZERO_R, id='no-dimensions'),
    ],
)
def test_a_constant_that_is_nan_is_refused_on_both_lanes(patch, data):
    """`0 / 0` on a constant side is NaN, which is no number at all, so the row is refused (#1892)."""
    message = both_lanes_refuse(override(DIVISOR_SPEC, **patch), data, match='constant value\\(s\\) are NaN')
    assert message.startswith("constraint 'c'"), 'the refusal names the constraint'


@pytest.mark.parametrize(
    ('patch', 'data', 'expected'),
    [
        pytest.param(
            {**WITH_H, 'constraints.c.expression': 'x <= h / d'}, ZERO_D | H, 105.0, id='a-constant-over-zero'
        ),
        pytest.param({**SCALAR_R, 'constraints.c.expression': 'x <= 10 / r'}, ZERO_R, 200.0, id='no-dimensions'),
        pytest.param(
            {**WITH_H, 'constraints.c.expression': 'x <= h'},
            ZERO_D | {'h': pd.Series([5.0, float('inf')], index=pd.Index(['a', 'b'], name='f'))},
            105.0,
            id='an-infinite-value',
        ),
    ],
)
def test_an_infinite_constant_is_a_limit_that_never_binds_on_both_lanes(patch, data, expected):
    """`x <= inf` holds for every `x`: data such as PyPSA's `e_sum_max` uses it to mean "no limit".

    Each lane is solved on its own rather than through `differential`: the
    linopy lane drops a row whose limit is infinite and the relational lane
    keeps it, so their row counts differ while their answers agree.
    """
    spec = override(DIVISOR_SPEC, **patch)
    with sps.solve(spec, data) as result:
        assert result.objective == pytest.approx(expected, rel=RTOL), (
            'the infinite row never binds on the relational lane'
        )
    linopy_model = specsolve_linopy.build(schema_of(spec).expand(), data)
    linopy_model.solve(solver_name='highs', output_flag=False)
    assert float(linopy_model.objective.value) == pytest.approx(expected, rel=RTOL), 'nor on the linopy lane'


def test_an_objective_constant_that_is_nan_is_refused():
    """`sum(x) + 0 / r` would report a NaN objective. The linopy lane refuses any objective constant (#894)."""
    spec = override(DIVISOR_SPEC, **SCALAR_R, **{'objective.expression': 'sum(x, over=f) + 0 / r'})
    with pytest.raises(DataError, match=r'objective: 1 constant value\(s\) are NaN'):
        sps.build(spec, ZERO_R).close()


@pytest.mark.parametrize(
    'patch',
    [
        pytest.param({'constraints.c.where': 'd != 0'}, id='mask-the-row'),
        pytest.param({'variables.x.where': 'd != 0'}, id='mask-the-variable'),
    ],
)
def test_a_zero_divisor_has_the_escape_a_sparse_one_has(patch):
    """The refusal is keyed to the quotients built, so a mask that removes the zero lifts it."""
    with differential(override(DIVISOR_SPEC, **patch), ZERO_D, lp=True) as run:
        assert run.oracle > 0, 'the masked model builds and solves on both lanes'


#: `sum(w, over=g)` is 3, and no single summand is: a divisor, base or exponent
#: taken summand by summand reads 1/2 + 1/1, 2² + 1² or 2² + 2¹ instead.
WHOLE_SUM = {
    'dimensions': {'g': {'dtype': 'str'}, 't': {'dtype': 'int'}},
    'parameters': {'w': {'dims': ['g']}},
    'variables': {'x': {'dims': ['t'], 'bounds': {'lower': 0, 'upper': 100}}},
    'objective': {'sense': 'minimize', 'expression': 'sum(x, over=t)'},
}
WHOLE_SUM_DATA = {'g': ['a', 'b'], 't': [0], 'w': pl.DataFrame({'g': ['a', 'b'], 'value': [2.0, 1.0]})}


@pytest.mark.parametrize(
    ('constraint', 'expected'),
    [
        pytest.param('x / sum(w, over=g) >= 1', 3.0, id='a-divisor'),
        pytest.param('x >= 6 / sum(w, over=g)', 2.0, id='a-divisor-on-the-constant-side'),
        pytest.param('x * sum(w, over=g) ** 2 >= 18', 2.0, id='a-base'),
        pytest.param('x * 2 ** sum(w, over=g) >= 16', 2.0, id='an-exponent'),
    ],
)
def test_an_operand_that_is_a_sum_is_taken_whole(constraint, expected):
    """Division and a power do not distribute over a sum, so the sum is added up before either reads it.

    `*` distributes, which is why a sum can stay one row per summand until the
    row is assembled; `/` and `**` do not. Before #1777 the relational lane
    divided by, or raised, each summand and added the results, and solved a
    different model with no error: 0.667 for the divisor, 3.6 for the base.
    """
    spec = override(WHOLE_SUM, **{'constraints.c': {'dims': ['t'], 'expression': constraint}})
    with differential(spec, WHOLE_SUM_DATA, lp=True) as run:
        assert run.oracle == pytest.approx(expected, rel=RTOL), 'the optimum reads the sum as its total, 3'


def test_a_sparse_divisor_under_a_summed_divisor_is_still_refused():
    """A null summand is a divisor's hole, so adding the sum up keeps it null rather than skipping it.

    `w / d` has no value at `b`. Added up by skipping nulls, the outer divisor
    read 2 instead of refusing, and the build solved `x / 2 >= 1` with no error.
    """
    spec = override(
        WHOLE_SUM,
        **{
            'parameters.d': {'dims': ['g']},
            'constraints.c': {'dims': ['t'], 'expression': 'x / sum(w / d, over=g) >= 1'},
        },
    )
    data = WHOLE_SUM_DATA | {'d': pl.DataFrame({'g': ['a'], 'value': [1.0]})}
    with pytest.raises(DataError, match='used as a divisor'):
        sps.build(spec, data).close()
