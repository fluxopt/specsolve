"""`result.evaluate`: a declared name (#562) or an expression the file never named, at the solution.

The relational lane only — the differential half, both lanes agreeing on the
same values, lives in ``test_linopy_lane.py`` with the rest of the oracle
comparisons. What is pinned here for a declared name: the value is the one the
primal implies, an expression no constraint references still reads, the
frame's dims are the ones it survives over, laziness (a build compiles no
expression; a read compiles that one), and the unknown-name refusal. For an
undeclared expression, below: a name and the body it stands for read one
value, both written forms are taken, and what it refuses — a name the model
does not declare, and an answer read back off disk, which carries no model to
lower against.
"""

from __future__ import annotations

import polars as pl
import pytest

import specsolve as sps
from specsolve import expressions
from specsolve.errors import DataError, LanguageError, SpecsolveError
from specsolve.relational.engines.polars.compiler import PolarsCompiler
from tests.fixtures import override

SPEC = {
    'dimensions': {
        'snapshot': {'dtype': 'int'},
        'generator': {'dtype': 'str'},
    },
    'parameters': {
        'p_max': {'dims': ['generator']},
        'cost': {'dims': ['generator']},
        'load': {'dims': ['snapshot']},
    },
    'variables': {
        'p': {'dims': ['snapshot', 'generator'], 'bounds': {'lower': 0, 'upper': 'p_max'}},
    },
    'expressions': {
        'total_gen': 'sum(p, over=generator)',
        'spend': 'sum(p * cost, over=generator)',
        'answer': '21 * 2',
        'total_cost': 'sum(sum(p * cost, over=generator), over=snapshot)',
        'squared': 'sum(p * p, over=generator)',
        'price': 'dual(balance)',
        'weighted': 'dual(balance) * total_gen',
        'rational': '1 / (1 + total_gen)',
    },
    'constraints': {
        'balance': {'dims': ['snapshot'], 'expression': 'total_gen == load'},
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(sum(p * cost, over=generator), over=snapshot)'},
}


def sources() -> dict[str, pl.DataFrame]:
    return {
        'snapshot': [0, 1, 2],
        'generator': ['g1', 'g2'],
        'p_max': pl.DataFrame({'generator': ['g1', 'g2'], 'value': [100.0, 100.0]}),
        'cost': pl.DataFrame({'generator': ['g1', 'g2'], 'value': [10.0, 20.0]}),
        'load': pl.DataFrame({'snapshot': [0, 1, 2], 'value': [50.0, 120.0, 80.0]}),
    }


@pytest.fixture(scope='module')
def result():
    """One solve for the whole module — `sps.solve` closes the model, so
    every read below also proves the readers outlive it."""
    return sps.solve(SPEC, sources())


def test_a_referenced_expression_reads_the_value_its_constraint_pinned(result):
    frame = result.evaluate('total_gen')
    assert frame.columns == ['snapshot', 'value'], 'an expression frame is (dims…, value), dims in declaration order'
    got = dict(zip(frame['snapshot'], frame['value'], strict=True))
    assert got == pytest.approx({0: 50.0, 1: 120.0, 2: 80.0}), (
        'balance pins total_gen to load, so the reader must hand back exactly the load values'
    )


def test_an_expression_nothing_references_reads_the_value_the_primal_implies(result):
    external = (
        result.primal('p')
        .join(sources()['cost'].rename({'value': 'cost'}), on='generator')
        .group_by('snapshot')
        .agg((pl.col('value') * pl.col('cost')).sum().alias('value'))
        .sort('snapshot')
    )
    frame = result.evaluate('spend')
    assert frame.sort('snapshot').equals(external), (
        'spend is referenced by nothing, and must still equal sum(p * cost) computed from the primal by hand'
    )


def test_a_scalar_expression_is_one_row_matching_the_objective(result):
    frame = result.evaluate('total_cost')
    assert frame.columns == ['value'] and frame.height == 1, 'an expression with no dims is a single value row'
    assert frame.item() == pytest.approx(result.objective), (
        'total_cost restates the objective, so the two numbers must agree'
    )


def test_a_variable_free_expression_is_legal_and_reads_its_constant(result):
    assert result.evaluate('answer').item() == pytest.approx(42.0), (
        'the grammar admits a variable-free named expression, and its value is the constant it spells'
    )


@pytest.mark.parametrize(
    ('name', 'dims'),
    [
        pytest.param('total_gen', {'snapshot'}, id='summed-over-one-of-two'),
        pytest.param('spend', {'snapshot'}, id='a-product-summed-over-one-of-two'),
        pytest.param('answer', set(), id='a-constant'),
        pytest.param('total_cost', set(), id='summed-over-both'),
    ],
)
def test_the_frame_carries_exactly_the_dims_the_expression_survives_over(result, name, dims):
    frame = result.evaluate(name)
    assert set(frame.columns) - {'value'} == dims, (
        'the returned frame answers over the dims the expression still ranges over after its sums'
    )


def test_an_unknown_name_is_refused_as_a_name_the_model_does_not_declare(result):
    with pytest.raises(LanguageError, match='nope'):
        result.evaluate('nope')


def test_a_masked_coordinate_has_no_row():
    masked = {
        **SPEC,
        'variables': {
            'p': {
                'dims': ['snapshot', 'generator'],
                'bounds': {'lower': 0, 'upper': 'p_max'},
                'where': 'p_max > 0',
            }
        },
        'expressions': {'scaled': 'p * cost'},
        'constraints': {'balance': {'dims': ['snapshot'], 'expression': 'sum(p, over=generator) == load'}},
    }
    data = sources() | {
        'p_max': pl.DataFrame({'generator': ['g1', 'g2'], 'value': [200.0, 0.0]}),
        'load': pl.DataFrame({'snapshot': [0, 1, 2], 'value': [50.0, 120.0, 80.0]}),
    }
    frame = sps.solve(masked, data).evaluate('scaled')
    assert frame['generator'].unique().to_list() == ['g1'], (
        'absence propagates into a reader the way it does into a constraint (the operator rules): the masked-out '
        "generator's coordinates have no rows rather than zeros"
    )
    assert frame.height == 3, 'the surviving generator keeps one row per snapshot'


def test_an_entry_of_degree_two_reads_the_primal_squared(result):
    """The language holds an entry the math never reads to no degree, and a read needs none: every variable is a number by then."""
    primal = result.primal('p')
    want = primal.with_columns(pl.col('value') ** 2).group_by('snapshot').agg(pl.col('value').sum()).sort('snapshot')
    got = result.evaluate('squared')
    assert got['snapshot'].to_list() == want['snapshot'].to_list(), 'one row per snapshot, in label order'
    assert got['value'].to_list() == pytest.approx(want['value'].to_list()), (
        'p * p at the solution is each primal squared, summed over generators'
    )


def test_an_entry_reads_a_constraints_dual(result):
    assert result.evaluate('price').equals(result.dual('balance')), (
        "dual(balance) is the constraint's own dual frame, row for row"
    )


def test_a_dual_multiplies_like_any_number(result):
    dual = result.dual('balance')
    load = sources()['load']
    want = [d * v for d, v in zip(dual['value'], load['value'], strict=True)]
    assert result.evaluate('weighted')['value'].to_list() == pytest.approx(want), (
        'balance pins total_gen to load, so dual(balance) * total_gen is the dual times the load'
    )


def test_a_divisor_that_adds_is_added_up_before_it_divides(result):
    want = [1 / (1 + v) for v in sources()['load']['value']]
    assert result.evaluate('rational')['value'].to_list() == pytest.approx(want), (
        'no degree rule holds an entry the math never reads, so 1 / (1 + total_gen) is one value per snapshot'
    )


def test_a_dual_on_a_solve_that_left_none_is_refused_by_name():
    """An integer variable makes duals undefined; the entry reading one is refused with `Result.dual`'s own sentence, and every other entry still reads."""
    result = sps.solve(override(SPEC, **{'variables.p.domain': 'integer'}), sources())
    with pytest.raises(SpecsolveError, match='duals are undefined'):
        result.evaluate('price')
    assert result.evaluate('spend').height == 3, 'the refusal is per entry, not per result'


def test_a_divisor_that_adds_keeps_its_hole():
    """Adding up a divisor must not invent a zero: a coordinate no piece covers stays null, so the division reports it rather than dividing by it."""
    spec = override(
        SPEC,
        **{
            'parameters.scale': {'dims': ['snapshot']},
            'parameters.other': {'dims': ['snapshot']},
            'expressions.holed': '1 / (scale + other)',
        },
    )
    covered = pl.DataFrame({'snapshot': [0, 1], 'value': [2.0, 3.0]})
    result = sps.solve(spec, sources() | {'scale': covered, 'other': covered})
    with pytest.raises(DataError, match='used as a divisor but has no row at 1 of the coordinates'):
        result.evaluate('holed')


SIZED = {
    'dimensions': {'g': {'dtype': 'str'}},
    'parameters': {
        'fixed': {'dims': ['g']},
        'sizable': {'dims': ['g'], 'dtype': 'bool'},
        'out': {'dims': ['g']},
    },
    'variables': {'chosen': {'dims': ['g'], 'where': 'sizable', 'bounds': {'lower': 0, 'upper': 10}}},
    'expressions': {
        'size': {'dims': ['g'], 'cases': {'given': {'when': 'fixed', 'expression': 'fixed'}}, 'otherwise': 'chosen'},
    },
    'constraints': {'enough': {'dims': ['g'], 'where': 'sizable', 'expression': 'chosen >= out'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(chosen, over=g)'},
}


def sized(fixed: dict[str, float], out: dict[str, float]) -> dict[str, pl.DataFrame]:
    """`chosen` exists at `b` alone and solves to `out[b]`; `size` is `fixed` where it is given, else `chosen`."""
    return {
        'g': pl.DataFrame({'g': ['a', 'b', 'c']}),
        'fixed': pl.DataFrame({'g': list(fixed), 'value': list(fixed.values())}),
        'sizable': pl.DataFrame({'g': ['b'], 'value': [True]}),
        'out': pl.DataFrame({'g': list(out), 'value': list(out.values())}),
    }


ABSENT_AT_C = sized({'a': 4.0}, {'a': 2.0, 'b': 5.0, 'c': 1.0})
GIVEN_ZERO_AT_C = sized({'a': 4.0, 'c': 0.0}, {'a': 2.0, 'b': 5.0, 'c': 1.0})
SOLVED_ZERO_AT_B = sized({'a': 4.0, 'c': 2.0}, {'a': 2.0, 'b': 0.0, 'c': 1.0})


@pytest.mark.parametrize(
    ('data', 'expression', 'expected'),
    [
        pytest.param(ABSENT_AT_C, 'out / size', {'a': 0.5, 'b': 1.0}, id='absent-through-cases'),
        pytest.param(ABSENT_AT_C, 'out / chosen', {'b': 1.0}, id='absent-variable'),
        pytest.param(ABSENT_AT_C, '1 + out / size', {'a': 1.5, 'b': 2.0}, id='absent-beside-a-constant'),
        pytest.param(GIVEN_ZERO_AT_C, 'out / size', {'a': 0.5, 'b': 1.0}, id='zero-in-the-data'),
        pytest.param(GIVEN_ZERO_AT_C, '1 + out / size', {'a': 1.5, 'b': 2.0}, id='zero-beside-a-constant'),
        pytest.param(SOLVED_ZERO_AT_B, 'out / size', {'a': 0.5, 'c': 0.5}, id='zero-from-the-solve'),
    ],
)
def test_a_quotient_is_absent_where_its_divisor_is_absent_or_zero(data, expression, expected):
    """A reported quotient has no row where its divisor is absent, or is zero however it got there.

    Before #1775, an absent divisor was refused as if a parameter were short of
    a row, and a zero one read `inf`, or `nan` over a zero numerator.
    """
    frame = sps.solve(override(SIZED, **{'expressions.q': expression}), data).evaluate('q')
    got = dict(zip(frame['g'], frame['value'], strict=True))
    assert got == pytest.approx(expected), 'the coordinate whose divisor is absent or zero has no row, not inf'


def test_a_divisor_that_is_a_sum_divides_as_its_total():
    """`sum(fixed, over=g)` is 4 + 0, so every coordinate divides by 4, and the zero summand makes nothing absent.

    Before #1775 the numerator was divided by each summand and the quotients
    added, so the zero summand read `inf` at every coordinate.
    """
    spec = override(SIZED, **{'expressions.q': 'out / sum(fixed, over=g)'})
    frame = sps.solve(spec, GIVEN_ZERO_AT_C).evaluate('q')
    got = dict(zip(frame['g'], frame['value'], strict=True))
    assert got == pytest.approx({'a': 0.5, 'b': 1.25, 'c': 0.25}), 'out divided by the total 4, at every coordinate'


@pytest.mark.parametrize('data', [pytest.param(ABSENT_AT_C, id='absent'), pytest.param(GIVEN_ZERO_AT_C, id='zero')])
def test_a_sum_of_quotients_skips_the_one_whose_divisor_is_absent_or_zero(data):
    """A summand with no value is one summand fewer, so the sum reads `a` and `b` and stands.

    Before #1775 the absent summand was refused as a short parameter, and the
    zero one made the whole sum `inf`.
    """
    frame = sps.solve(override(SIZED, **{'expressions.q': 'sum(out / size, over=g)'}), data).evaluate('q')
    assert frame['value'].to_list() == pytest.approx([1.5]), 'the sum reads 2/4 + 5/5 and skips c'


def test_a_divisor_parameter_short_of_a_row_is_refused_in_a_reported_expressions_words():
    """A missing parameter row is not absence, so the refusal stands; its advice fits an entry that constrains nothing and takes no `where`.

    Before #1775 it read a constraint's message: that the constraint would stop
    constraining, and to mask the coordinates out with a `where`.
    """
    spec = override(SIZED, **{'expressions.q': 'out / fixed'})
    with pytest.raises(DataError, match="named expression 'q': parameter 'fixed'") as refused:
        sps.solve(spec, ABSENT_AT_C).evaluate('q')
    assert 'where' not in str(refused.value), 'a named expression takes no where, so the message cannot advise one'
    assert 'constrain' not in str(refused.value), 'a reported entry constrains nothing'


@pytest.mark.parametrize('crossed', [pytest.param('p * r', id='a-product'), pytest.param('p ** r', id='a-power')])
def test_a_product_is_absent_where_either_factor_is(crossed):
    """Presence travels out of a product, and a power, from both sides — as it does out of a quadratic term at a build.

    `r` is masked out at `g2` and is 1 where it exists, so `p * r` and `p ** r`
    both read `p`; `bonus`, a constant over the same dims, is owed only where
    the crossed term exists, so the sum over generators reads it at `g1` alone.
    Losing `r`'s presence would add `bonus` at `g2` back in.
    """
    spec = override(
        SPEC,
        **{
            'parameters.r_max': {'dims': ['generator']},
            'parameters.bonus': {'dims': ['snapshot', 'generator']},
            'variables.r': {
                'dims': ['snapshot', 'generator'],
                'bounds': {'lower': 1, 'upper': 1},
                'where': 'r_max > 0',
            },
            'expressions.summed_with': f'sum({crossed} + bonus, over=generator)',
        },
    )
    bonus = pl.DataFrame({'snapshot': [0, 0, 1, 1, 2, 2], 'generator': ['g1', 'g2'] * 3, 'value': [10.0] * 6})
    data = sources() | {'r_max': pl.DataFrame({'generator': ['g1', 'g2'], 'value': [1.0, 0.0]}), 'bonus': bonus}
    result = sps.solve(spec, data)
    p = result.primal('p').filter(pl.col('generator') == 'g1').sort('snapshot')
    want = [v * 1.0 + 10.0 for v in p['value']]
    assert result.evaluate('summed_with')['value'].to_list() == pytest.approx(want), (
        f'the sum reads {crossed} and bonus at g1 only, since r is absent at g2'
    )


def test_a_variable_declared_zero_is_zero_under_a_nonlinear_read():
    """`absence: zero` says the quantity *is* zero where the variable has no row.

    Affine arithmetic cannot tell no row from a zero, and neither can a sum a
    constant reaches — `1 + p` lands the 1 on every coordinate. `0.5 ** p`
    can: the absent generator contributes `0.5 ** 0`, which is 1, rather
    than nothing, and the present one a value near zero.
    """
    spec = override(
        SPEC,
        **{
            'variables.p.where': 'p_max > 0',
            'variables.p.absence': 'zero',
            'expressions.grown': 'sum(0.5 ** p, over=generator)',
        },
    )
    data = sources() | {'p_max': pl.DataFrame({'generator': ['g1', 'g2'], 'value': [200.0, 0.0]})}
    result = sps.solve(spec, data)
    p = result.primal('p').sort('snapshot')
    assert p['generator'].unique().to_list() == ['g1'], 'g2 is masked out, so only g1 has a primal'
    want = [0.5**v + 1.0 for v in p['value']]
    assert result.evaluate('grown')['value'].to_list() == pytest.approx(want), (
        'the absent generator is a zero under absence: zero, so 0.5 ** 0 counts as 1 in the sum'
    )


def test_a_build_compiles_no_expression_and_a_read_compiles_exactly_one(monkeypatch):
    compiled = []
    original = PolarsCompiler.expression

    def counting(self, expr, context, **kwargs):
        compiled.append(context)
        return original(self, expr, context, **kwargs)

    monkeypatch.setattr(PolarsCompiler, 'expression', counting)
    with sps.build(SPEC, sources()) as model:
        named = [c for c in compiled if c.startswith('named expression')]
        assert named == [], 'a build lowers no named expression — fifty declared and none read must cost none'
        assert len(compiled) == 2 * len(SPEC['constraints']) + 1, (
            'a model declaring expressions compiles exactly what one without them compiles: '
            'each constraint side, and the objective'
        )
        outcome = model.solve()
        named = [c for c in compiled if c.startswith('named expression')]
        assert named == [], 'a solve lowers none either — the readers it hands out are thunks'
        outcome.evaluate('spend')
        named = [c for c in compiled if c.startswith('named expression')]
        assert named == ["named expression 'spend'"], 'reading one expression compiles that one expression'


def test_a_closed_result_refuses_an_expression_read():
    with sps.build(SPEC, sources()) as model:
        outcome = model.solve()
    outcome.close()
    with pytest.raises(SpecsolveError, match='closed'):
        outcome.evaluate('spend')


# ---------------------------------------------------------------------------
# evaluate: an expression the file never named
# ---------------------------------------------------------------------------


def test_a_declared_name_and_the_body_it_stands_for_read_one_value(result):
    """`evaluate` takes a name because the language takes one: it substitutes a declared name where it stands, so the two spellings are one expression."""
    declared = result.evaluate('total_gen')
    written = result.evaluate('sum(p, over=generator)')
    assert declared.equals(result.evaluate('total_gen')), 'a declared name is served by the reader that holds it'
    assert written.equals(declared), "the body reads what the name reads, the name being the body's own spelling"


def test_an_expression_the_file_never_declared_reads_what_the_primal_implies(result):
    external = (
        result.primal('p')
        .join(sources()['cost'].rename({'value': 'cost'}), on='generator')
        .group_by('snapshot')
        .agg((pl.col('value') * pl.col('cost') * 2).sum().alias('value'))
        .sort('snapshot')
    )
    frame = result.evaluate('sum(p * cost * 2, over=generator)')
    assert frame.columns == ['snapshot', 'value'], 'an evaluated frame is (dims…, value), like a declared one'
    assert frame.sort('snapshot').equals(external), (
        'an expression nothing declared is evaluated at the same primal a declared one is'
    )


def test_a_mapping_is_the_other_form_the_language_writes_an_expression_in(result):
    """A bare string and a mapping are `ExpressionBlock`'s two written forms, so `evaluate` takes both."""
    assert result.evaluate({'expression': 'sum(p, over=generator)'}).equals(result.evaluate('total_gen'))


def test_a_mapping_carries_the_cases_a_string_cannot_say():
    """`cases:` is why the mapping form is not sugar: a region-varying quantity has no spelling as one string."""
    spec = {
        **SPEC,
        'parameters': {**SPEC['parameters'], 'peak': {'dims': ['snapshot'], 'dtype': 'bool'}},
    }
    data = sources() | {'peak': pl.DataFrame({'snapshot': [0, 1, 2], 'value': [False, True, False]})}
    frame = sps.solve(spec, data).evaluate(
        {
            'dims': ['snapshot'],
            'cases': {'busy': {'when': 'peak', 'expression': 'total_gen'}},
            'otherwise': 0,
        }
    )
    got = dict(zip(frame['snapshot'], frame['value'], strict=True))
    assert got == pytest.approx({0: 0.0, 1: 120.0, 2: 0.0}), (
        'the case holds only where peak does, and otherwise supplies the rest'
    )


def test_an_expression_may_read_a_dual_the_file_never_priced(result):
    assert result.evaluate('dual(balance) * 2')['value'].to_list() == pytest.approx(
        [v * 2 for v in result.dual('balance')['value']]
    ), 'the math reads nothing evaluated, so a dual stands in it exactly as it stands in a declared entry'


def test_a_name_the_model_does_not_declare_is_refused_rather_than_read_as_a_gap(result):
    """A read reaches the solved model's own declarations and no further, so a new parameter is named as missing rather than read as an absence — supplying one is a build."""
    with pytest.raises(LanguageError, match='co2_rate'):
        result.evaluate('sum(p * co2_rate, over=generator)')


def test_the_splice_steps_over_a_declaration_of_its_own_name():
    """Names share one flat namespace, so the spliced entry must not land on a declared one and shadow what the expression reads.

    Read through an expression that *references* the collision rather than
    naming it: naming it is served by the declared reader, and never splices.
    """
    spec = {**SPEC, 'expressions': {**SPEC['expressions'], '_evaluated': 'sum(p, over=generator) * 3'}}
    assert sps.solve(spec, sources()).evaluate('_evaluated * 2')['value'].to_list() == pytest.approx(
        [300.0, 720.0, 480.0]
    ), "the splice lands beside the declaration, so the expression still reads the model's own entry"


def test_an_answer_read_back_off_disk_says_why_it_cannot_evaluate(result, tmp_path):
    """An answer carries the values and not the model, and a name the file never wrote needs the model."""
    read_back = sps.load_result(result.save(tmp_path))
    with pytest.raises(SpecsolveError, match='no model behind it'):
        read_back.evaluate('sum(p, over=generator)')
    assert read_back.evaluate('total_gen').equals(result.evaluate('total_gen')), (
        'a declared name is readable either way — it was written, so nothing needs lowering'
    )


@pytest.fixture
def archived(tmp_path):
    """A solve written to an archive, which carries the spec and sources back."""
    with sps.build(SPEC, sources()) as model:
        model.solve(archive=tmp_path / 'run.zip')
    return tmp_path / 'run.zip'


@pytest.mark.parametrize(
    'expression',
    [
        pytest.param('sum(p * p_max, over=generator)', id='over-a-parameter'),
        pytest.param('sum(p * p, over=generator)', id='nonlinear'),
        pytest.param('dual(balance)', id='a-dual'),
        pytest.param('dual(balance) * total_gen', id='a-dual-times-an-expression'),
    ],
)
def test_evaluate_off_a_loaded_archive_reads_the_archived_solution(result, archived, expression):
    """A quantity the file never named reads off a loaded archive, at the values the solve left — no re-solve."""
    read_back = sps.load_archive(archived).answer.evaluate(expression)
    live = result.evaluate(expression)
    keys = live.columns[:-1]
    assert read_back.sort(keys).equals(live.sort(keys)), (
        'an archived evaluate rebuilds the model and reads the archived primal, so it matches the live answer'
    )


def test_evaluate_off_a_scanned_archive_reads_the_same(result, archived, tmp_path):
    """`scan_archive` leaves the frames on disk, and evaluate rebuilds against them just the same."""
    read_back = sps.scan_archive(archived, into=tmp_path / 'unpacked').answer.evaluate('sum(p * p_max, over=generator)')
    live = result.evaluate('sum(p * p_max, over=generator)')
    keys = live.columns[:-1]
    assert read_back.sort(keys).equals(live.sort(keys)), 'a scanned archive evaluates against the frames left on disk'


def test_a_declared_name_off_an_archive_is_served_from_disk_not_lowered(archived, monkeypatch):
    """A declared name was written, so it reads back without a rebuild — only a name outside them reaches the reader."""
    monkeypatch.setattr(expressions, 'lower', lambda *a, **k: pytest.fail('a declared name must not lower'))
    frame = sps.load_archive(archived).answer.evaluate('total_gen')
    assert frame.columns == ['snapshot', 'value'], 'a declared name off an archive reads its written frame'


def test_a_closed_result_refuses_to_evaluate():
    result = sps.solve(SPEC, sources())
    result.close()
    with pytest.raises(SpecsolveError, match='was closed'):
        result.evaluate('sum(p, over=generator)')


def test_an_evaluated_expression_names_nothing_and_so_is_not_a_kind(result, tmp_path):
    """It is not written, spilled or enumerated: a quantity worth keeping across runs is worth declaring."""
    written = {p.stem for p in (result.save(tmp_path) / 'expression').glob('*.parquet')}
    assert written == set(SPEC['expressions']), 'save writes the declared names, and evaluate adds none'
