"""``Model.to_linopy`` hands over the model the engine built, over its declared dims.

Two halves, as for the pyomo export. The model is the same one: every fixture
and every port reaches the engine's or the published optimum through linopy.
And each entry is the right one: the row at ``balance`` for ``snapshot=1``
holds the terms [`Model.row`][] reads at that coordinate, which an entry placed
one position off would break while leaving every objective intact.

The export is not the oracle: ``tests/linopy_lane`` rebuilds the program on its
own, and nothing there reads this module.
"""

from __future__ import annotations

import datetime
import itertools
import sys
from typing import Any

import polars as pl
import pytest

import specsolve as sps
from specsolve.errors import SpecsolveError
from tests.conftest import DISPATCH_SPEC, PORT_REFERENCES, expanded, port_sources, port_spec
from tests.test_mps_text import COMMITMENT, COMMITMENT_DATA, DISPATCH_DATA, FREE_DATA, FREE_SPEC, QUADRATIC_ROW_SPEC
from tests.test_quadratic_objective import SOURCES as QUADRATIC_DATA
from tests.test_quadratic_objective import SPEC as QUADRATIC_OBJECTIVE_SPEC
from tests.test_sos import CAP, SITES, SIZES, VALUE, best
from tests.test_sos import DATA as SOS_DATA
from tests.test_sos import spec as sos_spec

linopy = pytest.importorskip('linopy', reason='to_linopy needs the [linopy] extra')

#: A masked variable, a scalar beside it, and a constraint whose middle row has
#: no term and is not built, so the rows after it renumber.
MASKED_SPEC: dict[str, Any] = {
    'dimensions': {'i': {'dtype': 'int'}, 'j': {'dtype': 'int'}},
    'parameters': {'a': {'dims': ['i', 'j']}, 'cap': {'dims': ['j']}},
    'variables': {
        'x': {'dims': ['j'], 'where': 'cap > 0', 'bounds': {'lower': 0, 'upper': 10}},
        'total': {'dims': [], 'bounds': {'lower': 0}},
    },
    'constraints': {
        'c': {'dims': ['i'], 'expression': 'sum(a * x, over=j) >= 4'},
        'counted': {'dims': [], 'expression': 'total == sum(x, over=j)'},
        'capped': {'dims': ['j'], 'expression': 'x <= cap'},
    },
    'objective': {'sense': 'minimize', 'expression': 'total'},
}

MASKED_DATA = {
    'i': [0, 1, 2],
    'j': [0, 1, 2],
    'cap': pl.DataFrame({'j': [0, 1, 2], 'value': [5.0, 0.0, 7.0]}),
    'a': pl.DataFrame({'i': [0, 0, 2], 'j': [0, 2, 2], 'value': [1.0, 2.0, 3.0]}),
}

LINEAR = [
    pytest.param(DISPATCH_SPEC, DISPATCH_DATA, id='lp'),
    pytest.param(COMMITMENT, COMMITMENT_DATA, id='milp'),
    pytest.param(FREE_SPEC, FREE_DATA, id='free-and-bounded-columns'),
    pytest.param(MASKED_SPEC, MASKED_DATA, id='masked-and-dropped-rows'),
]


def _solved(m: Any, solver: str = 'highs') -> float:
    status, condition = m.solve(solver_name=solver)
    assert (status, condition) == ('ok', 'optimal'), f'linopy reports {status}/{condition}, not an optimum'
    return float(m.objective.value)


def _exported(spec: Any, data: Any) -> tuple[Any, Any]:
    """The linopy model, and the engine's own answer to the same build."""
    with sps.build(spec, data) as model:
        m = model.to_linopy()
        return m, model.solve()


@pytest.mark.parametrize(('spec', 'data'), LINEAR)
def test_the_exported_model_reaches_the_optimum_the_engine_reaches(spec: Any, data: Any) -> None:
    m, answer = _exported(spec, data)
    assert _solved(m) == pytest.approx(answer.objective), (
        'linopy solved a different model from the one the engine built'
    )


def test_a_quadratic_objective_reaches_the_engine_optimum() -> None:
    m, answer = _exported(QUADRATIC_OBJECTIVE_SPEC, QUADRATIC_DATA)
    assert _solved(m) == pytest.approx(answer.objective), 'the quadratic pairs changed on the way to linopy'


def test_a_quadratic_constraint_is_refused_naming_the_sinks_that_take_it() -> None:
    """linopy has no quadratic constraint, so its table refuses one as every sink's does."""
    with (
        sps.build(QUADRATIC_ROW_SPEC, QUADRATIC_DATA) as model,
        pytest.raises(
            SpecsolveError,
            match=r"the 'linopy' sink cannot take a quadratic constraint.*Sinks that do take it: \.lp, gurobi, pyomo\.",
        ),
    ):
        model.to_linopy()


#: Ports whose objective carries a constant, which linopy's objective cannot (#894).
WITH_AN_OBJECTIVE_CONSTANT = {'osemosys_utopia'}


@pytest.mark.parametrize('name', sorted(PORT_REFERENCES.keys() - WITH_AN_OBJECTIVE_CONSTANT), ids=str)
def test_every_referenced_model_reaches_its_optimum_through_linopy(name: str) -> None:
    """The corpus at its own sizes, each formulation written out so HiGHS reads it."""
    with sps.build(expanded(port_spec(name)), port_sources(name)) as model:
        m = model.to_linopy()
    assert _solved(m) == pytest.approx(PORT_REFERENCES[name]['objective'], rel=1e-6), (
        'the linopy model misses the published optimum to 1e-6'
    )


@pytest.mark.parametrize('name', sorted(WITH_AN_OBJECTIVE_CONSTANT), ids=str)
def test_an_objective_constant_is_refused_rather_than_dropped(name: str) -> None:
    """Dropped, the constant would leave linopy's optimum off by its value."""
    with sps.build(expanded(port_spec(name)), port_sources(name)) as model:
        assert model._engine._model.handoff.objective_constant, f'{name} no longer has an objective constant'
        with pytest.raises(SpecsolveError, match="linopy's objective holds none"):
            model.to_linopy()


def _row(
    m: Any, constraint: str, coordinate: dict[str, Any]
) -> tuple[dict[tuple[str, tuple[Any, ...]], float], Any, str]:
    """One linopy row as ``{(variable, coordinate): coefficient}``, its right-hand side and its sign."""
    held = m.constraints[constraint]
    at = held.data.sel(coordinate) if coordinate else held.data
    terms = {}
    for label, coefficient in zip(at['vars'].values.ravel(), at['coeffs'].values.ravel(), strict=True):
        if label == -1:
            continue
        name, position = m.variables.get_label_position(int(label))
        terms[name, tuple(position.values())] = float(coefficient)
    return terms, float(at['rhs']), str(at['sign'].item())


@pytest.mark.parametrize(('spec', 'data'), LINEAR)
def test_every_row_holds_the_terms_the_built_row_holds(spec: Any, data: Any) -> None:
    """Rows against [`Model.row`][], which finds a row by its coordinate and not by position.

    Every row the build made is checked, and the masked count shows nothing
    else is in the constraint, so a row placed one off, or one the build
    dropped, fails here.
    """
    signs = {'<=': '<=', '>=': '>=', '==': '='}
    with sps.build(spec, data) as model:
        m = model.to_linopy()
        result = model.solve()
        for constraint, declared in model._program.constraints.items():
            coordinates = (
                list(result.activity(constraint).drop('value').iter_rows(named=True)) if declared.dims else [{}]
            )
            held = m.constraints[constraint]
            built = int((held.labels != -1).sum())
            assert built == len(coordinates), f"'{constraint}' has a row the build did not make"
            for coordinate in coordinates:
                row = model.row(constraint, **coordinate)
                expected = {
                    (
                        t['variable'],
                        tuple(_label(v) for v in t['coordinate'].split(', ')) if t['coordinate'] else (),
                    ): t['coefficient']
                    for t in row.terms.iter_rows(named=True)
                }
                terms, rhs, sign = _row(m, constraint, coordinate)
                assert terms == expected, f'{constraint}{coordinate} holds another row'
                assert rhs == pytest.approx(row.rhs), f'{constraint}{coordinate} compares against another bound'
                assert sign == signs[row.sense], f'{constraint}{coordinate} changed its sense'


def _label(text: str) -> Any:
    """A coordinate label as [`Model.row`][] prints it, read back as the int the fixtures use where it is one."""
    return int(text) if text.lstrip('-').isdigit() else text


def test_each_variable_reads_back_the_value_the_engine_solved() -> None:
    """The dispatch optimum is unique, so linopy's value at a coordinate is the engine's at that coordinate."""
    m, answer = _exported(DISPATCH_SPEC, DISPATCH_DATA)
    _solved(m)
    solution = m.variables['p'].solution
    for row in answer.primal('p').iter_rows(named=True):
        at = solution.sel(snapshot=row['snapshot'], generator=row['generator']).item()
        assert at == pytest.approx(row['value']), f'p at {row} reads another coordinate'


def test_a_masked_coordinate_is_masked_out_and_every_label_is_a_coordinate() -> None:
    m, _ = _exported(MASKED_SPEC, MASKED_DATA)
    x = m.variables['x']
    assert list(x.labels.coords['j'].values) == [0, 1, 2], 'every label of the dim is a coordinate'
    assert int(x.labels.sel(j=1)) == -1, 'the coordinate cap removed is masked out'
    assert int(m.constraints['c'].labels.sel(i=1)) == -1, 'the row with no term is not built'
    assert m.variables['total'].labels.dims == (), 'a dimensionless variable is a scalar'


def test_a_label_keeps_its_own_type() -> None:
    spec = {
        'dimensions': {'hour': {'dtype': 'datetime'}},
        'variables': {'p': {'dims': ['hour'], 'bounds': {'lower': 1, 'upper': 5}}},
        'objective': {'sense': 'minimize', 'expression': 'sum(p, over=hour)'},
    }
    hour = datetime.datetime(2030, 1, 1, 6)
    m, _ = _exported(spec, {'hour': [hour]})
    assert m.variables['p'].labels.coords['hour'].values[0] == hour, 'the coordinate is the datetime itself'


def _termless(sense: str, floor: float) -> tuple[dict[str, Any], dict[str, Any]]:
    """A constraint whose one row the build keeps with no term: the solver sees ``0 <sense> floor``."""
    spec = {
        'dimensions': {'j': {'dtype': 'int'}},
        'parameters': {'a': {'dims': ['j']}, 'floor': {'dims': []}},
        'variables': {'x': {'dims': ['j'], 'bounds': {'lower': 0, 'upper': 10}}},
        'constraints': {'c': {'dims': [], 'expression': f'sum(a * x, over=j) {sense} floor'}},
        'objective': {'sense': 'minimize', 'expression': 'sum(x, over=j)'},
    }
    data = {
        'j': [0, 1],
        'a': pl.DataFrame({'j': [0, 1], 'value': [0.0, 0.0]}),
        'floor': pl.DataFrame({'value': [floor]}),
    }
    return spec, data


#: A termless row no point meets, one per comparison.
UNMET = [
    pytest.param('>=', 10.0, id='at-least-a-positive'),
    pytest.param('<=', -10.0, id='at-most-a-negative'),
    pytest.param('==', 3.0, id='equal-to-a-nonzero'),
]

#: A termless row every point meets, one per comparison.
MET = [
    pytest.param('>=', -5.0, id='at-least-a-negative'),
    pytest.param('<=', 5.0, id='at-most-a-positive'),
    pytest.param('==', 0.0, id='equal-to-zero'),
]


@pytest.mark.parametrize(('sense', 'floor'), UNMET)
def test_a_row_no_point_meets_is_refused_rather_than_dropped(sense: str, floor: float) -> None:
    """linopy drops a row with no term, so ``0 >= 10`` would vanish and the model would solve."""
    spec, data = _termless(sense, floor)
    with sps.build(spec, data) as model:
        assert model.solve().termination_condition == 'infeasible', 'the engine keeps the row'
        with pytest.raises(
            SpecsolveError, match=rf"constraint 'c' at \{{\}} has no term left and reads 0 {sense} {floor}"
        ):
            model.to_linopy()


#: A model of each kind linopy would lose, beside the refusal it reads.
LOST = [
    pytest.param(QUADRATIC_ROW_SPEC, QUADRATIC_DATA, 'cannot take a quadratic constraint', id='a-quadratic-constraint'),
    pytest.param(*_termless('>=', 10.0), 'has no term left', id='a-row-no-point-meets'),
    pytest.param(
        {
            'variables': {'x': {'dims': [], 'bounds': {'lower': 1}}},
            'objective': {'sense': 'minimize', 'expression': 'x + 5'},
        },
        {},
        "linopy's objective holds none",
        id='an-objective-constant',
    ),
]


@pytest.mark.parametrize(('spec', 'data', 'refused'), LOST)
def test_check_refuses_what_the_export_refuses_in_its_words(spec: Any, data: Any, refused: str) -> None:
    with sps.build(spec, data) as model:
        with pytest.raises(SpecsolveError, match=refused) as checked:
            model.check('linopy')
        with pytest.raises(SpecsolveError) as exported:
            model.to_linopy()
    assert str(checked.value) == str(exported.value), 'check and to_linopy refuse in different words'


def test_a_model_linopy_takes_checks_clean() -> None:
    with sps.build(DISPATCH_SPEC, DISPATCH_DATA) as model:
        model.check('linopy')


@pytest.mark.parametrize(('sense', 'floor'), MET)
def test_a_row_every_point_meets_drops_without_changing_the_answer(sense: str, floor: float) -> None:
    m, answer = _exported(*_termless(sense, floor))
    assert 'c' not in m.constraints, 'a constraint with no term at all is left out'
    assert _solved(m) == pytest.approx(answer.objective), f'leaving out 0 {sense} {floor} changed the optimum'


@pytest.mark.parametrize('sos_type', [1, 2])
def test_a_set_restricts_what_the_declaration_does(sos_type: int) -> None:
    """HiGHS takes no SOS, so Gurobi solves it, as ``test_sos`` does."""
    pytest.importorskip('gurobipy', reason='no solver here takes an SOS without it')
    with sps.build(sos_spec(sos_type), SOS_DATA) as model:
        m = model.to_linopy()
    assert _solved(m, 'gurobi') == pytest.approx(best(sos_type)), 'the exported sets do not restrict what they should'


@pytest.mark.parametrize('sos_type', [1, 2])
def test_a_set_over_a_variable_with_no_other_dim_is_one_set(sos_type: int) -> None:
    pytest.importorskip('gurobipy', reason='no solver here takes an SOS without it')
    site = SITES[0]
    spec = {
        'dimensions': {'size': {'dtype': 'int'}},
        'parameters': {'value': {'dims': ['size']}, 'cap': {'dims': ['size']}},
        'variables': {'take': {'dims': ['size'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
        'objective': {'sense': 'maximize', 'expression': 'sum(take * value, over=size)'},
        'sos': {'pick': {'variable': 'take', 'along': 'size', 'type': sos_type}},
    }
    data = {
        'size': SIZES,
        'value': pl.DataFrame({'size': SIZES, 'value': [VALUE[site, s] for s in SIZES]}),
        'cap': pl.DataFrame({'size': SIZES, 'value': [CAP[site, s] for s in SIZES]}),
    }
    patterns = [(s,) for s in SIZES] + (list(itertools.pairwise(SIZES)) if sos_type == 2 else [])
    optimum = max(sum(VALUE[site, s] * CAP[site, s] for s in pattern) for pattern in patterns)
    with sps.build(spec, data) as model:
        m = model.to_linopy()
    assert _solved(m, 'gurobi') == pytest.approx(optimum), 'the one set does not restrict what it should'


def test_a_variable_and_a_constraint_keep_one_name() -> None:
    """linopy holds variables and constraints apart, so a name both carry needs no suffix."""
    spec = {
        'variables': {'x': {'dims': [], 'bounds': {'lower': 0}}},
        'constraints': {'x': {'dims': [], 'expression': 'x >= 2'}},
        'objective': {'sense': 'minimize', 'expression': 'x'},
    }
    m, _ = _exported(spec, {})
    assert 'x' in m.variables, 'the variable keeps its name'
    assert 'x' in m.constraints, 'and so does the constraint'
    assert _solved(m) == pytest.approx(2.0), 'the constraint binds the variable of its name'


def test_without_linopy_the_export_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    with sps.build(DISPATCH_SPEC, DISPATCH_DATA) as model:
        monkeypatch.setitem(sys.modules, 'linopy', None)
        with pytest.raises(SpecsolveError, match=r'pip install "specsolve\[linopy\]"'):
            model.to_linopy()
