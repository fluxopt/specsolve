"""The algebraic laws the language promises — and the ones it deliberately breaks.

linopy's v1 convention ships formal law tests for the same reason this file
exists: an arithmetic convention is a set of *equalities between spellings*, and
nothing else in a test suite checks those. A model can build, solve, and agree
across both lanes while `a + b` and `b + a` quietly mean different things.

Two kinds of case here, and the second is the point:

**Laws** — spellings that must produce the same model. Each is solved through
``differential`` (linopy lane, relational lane, and the LP file re-solve), so a
law holding is six numbers agreeing rather than two.

**Non-laws** — spellings that are equal in ordinary algebra and are *not* equal
here, because absence is a first-class state. These are asserted to differ, with
the values written down. They are the ones worth having: ``sum(a + b)`` versus
``sum(a) + sum(b)`` diverged silently by 40% on one lane for as long as the
oracle was blind to it (#311), and no law-shaped test would have caught it —
only a test that says *these two are supposed to disagree, and by exactly this
much*.

The fixture keeps one masked variable (``y``, absent at ``f=b``) and one total
one (``x``), because every interesting law is conditional on whether absence is
in play. Laws are therefore checked twice where it matters: once over ``x``
alone, where they hold, and once over ``y``, where some of them stop.

The wide cases at the end vary one thing at a time against that fixture: the
*reduction* (a plain sum rather than a grouped one), the *number* of masks,
where the mask sits relative to the dim being reduced, and where the absence
comes from. The narrow cases reduce over ``f`` with one mask on ``f``, which is
the smallest arrangement that shows a rule — and small enough that it could be
passing for the wrong reason. The grouped form is the one that had to be here:
#314 routed it through the same propagation as a plain sum and nothing covered
it, so its behaviour was a claim rather than a result.
"""

from __future__ import annotations

import polars as pl
import pytest

from specsolve.errors import DataError
from tests.conftest import law_data, law_spec, override
from tests.differential import RTOL, both_lanes_refuse, differential
from tests.oracle import pd

# ---------------------------------------------------------------------------
# the fixture: `x` total, `y` absent at f=b, `w` a dense coefficient. It is
# `conftest.law_spec`, shared with the sweep that has to be over one model.
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
    """Solve *expression* on both lanes and the LP file; return the agreed value.

    ``differential`` raises if the three disagree, so a number coming back out
    of here is already a statement that the lanes concur about this spelling.
    """
    with differential(_spec(expression, objective=objective, dims=dims, also=also), DATA, lp=True) as run:
        return float(run.result.objective)


# ---------------------------------------------------------------------------
# laws — these must hold
# ---------------------------------------------------------------------------

#: Rewrites that must build the same model. ``reduction-is-linear`` holds only
#: while nothing is absent, which is why it is stated over ``x`` and not ``y``.
#: ``commutative-add-under-absence`` is the one law that *is* allowed absence:
#: both spellings carry the same absence, so it survives — which is what makes
#: the non-laws below meaningful rather than "anything with a mask behaves
#: oddly".
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
    """The defect behind #311, pinned as the semantics it turned out to be.

    ``y`` is absent at ``f=b``, so the summand ``x + y`` is absent there too
    (absence propagates, the absence rules) and the reduction skips that slot — taking the
    perfectly present ``x[b]`` with it. Summing each operand separately keeps
    it, because each is reduced over its own domain.

    Both are correct answers to *different questions*: "the total of the net,
    where the net is defined" against "the total in, minus the total out". The
    language declines to guess which was meant, which is the whole content of
    the v1 convention — distributing one into the other would read the absent
    ``y`` as a zero.

    The relational lane used to distribute, so it answered the second question
    for both spellings and disagreed with linopy, silently.
    """
    together = _objective_of('sum(x + y, over=f) <= 120')
    apart = _objective_of('sum(x, over=f) + sum(y, over=f) <= 120')

    assert together == pytest.approx(400.0, rel=RTOL), 'together binds only at f=a, so x[b] is free to its bound'
    assert apart == pytest.approx(240.0, rel=RTOL), 'apart keeps x[b] in the row, so the cap binds the total'
    assert together != pytest.approx(apart, rel=RTOL), 'the two questions must stay distinguishable'


def test_a_term_whose_variable_is_absent_is_not_a_term_worth_zero():
    """Row absence, the other half of the same rule.

    ``x + y >= k`` is *no constraint* where ``y`` is absent — not ``x >= k``.
    Compared against the spelling the absence rules points at for the other reading — two
    constraints under complementary ``where`` clauses — so the test states the
    *difference between the two intents* rather than the behaviour alone.
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
    """The rewrite the absence rules point at, moved to where the quantity is declared.

    For *the row kept, the missing term read as zero* that page names two
    constraints under complementary ``where`` clauses. A variable whose absence
    **is** zero should reach that model from one key — and reaching it is the
    claim, not merely matching its number: ``differential`` has already made
    both lanes and the LP file agree before either figure comes back.
    """
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
    """Out of a reduction absence never propagated, so the key changes nothing there.

    ``sum(y, over=f)`` was always *the total over the y's that exist* — a
    reduction is defined when only some of its set is. Declaring the absent
    coordinates worth zero adds zero to that total, so the two readings have to
    agree here even though they differ in a bare term.
    """
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
    """``fill=`` is not decoration: it decides whether the row exists at all.

    Bare, the vacated slot is absent and the row goes with it (#289). Filled, it
    contributes the identity of the position it sits in and the row survives.
    """
    bare = _objective_of('sum(x - shift(x, along=t, offset=1), over=f) <= 10')
    filled = _objective_of('sum(x - shift(x, along=t, offset=1, edge=0), over=f) <= 10')

    assert bare != pytest.approx(filled, rel=RTOL), (
        'a bare shift drops the first row; a filled one keeps it, so these cannot agree'
    )


# ---------------------------------------------------------------------------
# the same rules under wider shapes
# ---------------------------------------------------------------------------

#: ``y`` is absent at ``d``; ``v`` at ``b`` *and* ``d``, so the two masks do
#: not nest.
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
    """The wide fixture solved through both lanes, for one expression.

    ``g`` and the relation that reaches it exist only for the grouped cases:
    the plain fixture passes no ``g`` index, and a target with no index of its
    own is refused rather than carried as a dangling relation (#488).
    """
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
    """`sum` is a reduction, so the non-law applies to it unchanged.

    #314 routed it through the same absence propagation as `sum` on the argument
    that a group *is* a sum. Nothing tested that, so this is the assertion the
    change was made on: the two spellings separate, and both lanes agree about
    where they land.
    """
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
    `a` and `c`. Worth its own case because the implementation collects one
    restriction per fragment and applies every one to every fragment; a version
    that stopped at the first, or that composed them pairwise down the addition
    tree, would still pass every single-mask test above.
    """
    together = _wide_objective_of('sum(x + y + v, over=f) <= 120', dims=['t'])
    apart = _wide_objective_of('sum(x, over=f) + sum(y, over=f) + sum(v, over=f) <= 120', dims=['t'])

    assert together == pytest.approx(640.0, rel=RTOL)
    assert apart == pytest.approx(240.0, rel=RTOL)


def test_a_broadcast_coefficient_does_not_move_where_the_summand_exists():
    """A sparse *parameter* is not absence, so it must not restrict anything.

    `w * x` is a term whose coefficient varies over `f`; multiplying by it
    changes the arithmetic and nothing about presence (the absence rules: absence is a
    property of variables). The separation here must therefore come from `y`
    alone, exactly as in the un-weighted case.
    """
    together = _wide_objective_of('sum(w * x + y, over=f) <= 120', dims=['t'])
    apart = _wide_objective_of('sum(w * x, over=f) + sum(y, over=f) <= 120', dims=['t'])

    assert together == pytest.approx(320.0, rel=RTOL)
    assert apart == pytest.approx(120.0, rel=RTOL)


def test_a_mask_on_a_dim_the_reduction_does_not_touch_still_propagates():
    """Absence does not have to live on the summed dim to reach the summand.

    Here the mask is on `t` and the reduction is over `f`: at `t=1` the whole
    summand is absent for *every* `f`, so the row sums nothing rather than
    summing the surviving operand. The restriction is keyed by the dims the
    presence actually names, which is what makes this work — an implementation
    keying it by the reduced dim would silently do nothing here.
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
    """The two absence sources have to agree, or `shift` is a second rule.

    A bare `shift` vacates its first coordinate into absence (#289/#291), and
    that absence must behave inside a reduction exactly like a mask's. If it did
    not, the language would have two kinds of "not here" and the absence rules would be
    describing only one of them.

    The shifted operand is a **separate** variable from the one the objective
    maximises, and that is what makes the case discriminating. Written as
    ``sum(x + shift(x, along=t, offset=1), over=f)`` it is not: the ``t=1`` row bounds the
    same ``x[.,0]`` that a missing restriction would bound at ``t=0``, so it
    dominates and the objective reads 120 either way. Verified by disabling the
    propagation — that spelling still passed while five other cases failed.

    Here ``v`` appears only under the shift, so ``t=0`` is the only place the
    rule can show: propagated, the summand is absent there and ``x[.,0]`` is
    free to its bounds; without it the row survives as ``sum(x, over=f) <= 120``
    and caps them at 120, giving 240.
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


#: The divisor fixture: every test below is this model but for one thing, and
#: :func:`override` states the one thing.
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
    """The one position with no defensible fill (#312).

    Everywhere else a missing parameter row is a zero coefficient (the absence rules), and
    a zeroed term still leaves a row that says something. A divisor has no such
    identity: 0 divides by zero, 1 silently rescales, and dropping the term
    rewrites what the constraint asserts. Both lanes used to take that last
    option and *agree* about it — `x / d <= 10` became vacuous at the uncovered
    coordinate and `x` ran to its bound, objective 120 where the constraint
    reads as 20.

    Agreement is why the differential harness could not catch it: this is the
    shape of defect that needs a test saying what the answer *should* be, not
    that the lanes concur.
    """
    with pytest.raises(DataError, match='used as a divisor'), differential(DIVISOR_SPEC, SPARSE_D) as run:
        _ = run.result.objective

    dense = {'f': ['a', 'b'], 'd': pd.Series([2.0, 5.0], index=pd.Index(['a', 'b'], name='f'))}
    with differential(DIVISOR_SPEC, dense, lp=True) as run:
        assert float(run.result.objective) == pytest.approx(70.0, rel=RTOL), (
            'covered, the same model builds and the row binds on both lanes'
        )


def test_a_sparse_divisor_written_as_a_power_is_named_and_refused_on_both_lanes():
    """The same gap under `**`, where the walk used to see nothing (#1536).

    `program.children` had no branch for a power, so `d ** 2` hid `d` from every
    walk this package asks the language for a divisor's names. The relational
    lane named the divisor `''`, and the linopy lane found no divisor to check,
    filled the missing row with a zero coefficient and built `inf` into the
    matrix with no refusal at all — the two lanes disagreed, and neither answer
    was usable. energy-models/mathspec#403 fixed the walk upstream; this pins
    what is owed here, that both lanes name `d` and refuse in one sentence.
    """
    spec = override(DIVISOR_SPEC, **{'constraints.c.expression': 'x / (d ** 2) <= 10'})
    both_lanes_refuse(spec, SPARSE_D, match="parameter 'd' is used as a divisor")


def test_a_sparse_divisor_in_the_objective_is_refused_too():
    """The refusal holds in the one declaration with no rows to mask.

    The objective aggregates its terms per column, and a sum reads a null
    coefficient as zero — the exact silent answer #312 exists to refuse. A
    check only on constraints would let the same quotient through here, and
    the optimum would simply be missing terms.
    """
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
    """The one side whose null never reaches the matrix to be counted there.

    `x <= h / d` divides where no variable stands, so the quotient is a
    constant piece rather than a term and the null coefficient the check above
    reads is not in the matrix to read. The piece is summed per coordinate on
    its way to the row and a sum reads a null as zero, so the gap is counted
    before that or not at all — it used to be not at all, and the row bound `x`
    by the half of the quotient the data covered (#1465).
    """
    spec = override(
        DIVISOR_SPEC,
        **{'parameters.h': {'dims': ['f']}, 'constraints.c.expression': 'x <= h / d'},
    )
    data = SPARSE_D | {'h': pd.Series([10.0, 10.0], index=pd.Index(['a', 'b'], name='f'))}
    both_lanes_refuse(spec, data, match="parameter 'd' is used as a divisor")


def test_a_sparse_divisor_on_a_constant_side_has_the_same_escape():
    """And the refusal is keyed to the rows built there too.

    The mask answers the question on the constant side for the reason it
    answers it under a term: the coordinate the divisor says nothing about is
    one the model never builds a row at.
    """
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
    """The check is keyed to the rows built, not the coordinate product.

    Supplying a divisor only where the constraint exists is the ordinary idiom,
    and the first cut of this check refused it — the gap sits at coordinates the
    model already decided not to build. Kept as its own case because a check
    that ignores the mask still passes every test above: it only ever refuses
    *more*, and nothing else here asks it to accept something.
    """
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
    """Sparse data is the ordinary case, so the refusal must be escapable.

    Both spellings say the same thing in different places — *this coordinate has
    no row* — and either is enough, because the check asks where the model
    actually divides rather than whether the divisor is dense. A refusal with no
    way out would be worse than the silent answer it replaced.
    """
    with differential(override(DIVISOR_SPEC, **patch), SPARSE_D, lp=True) as run:
        assert float(run.result.objective) == pytest.approx(expected, rel=RTOL), (
            'either spelling of "this coordinate has no row" lifts the refusal'
        )


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
