"""``Model.to_pyomo`` hands over the model the engine built, indexed by its labels.

Two halves, as for a written file. The model is the same one: every fixture
and every port reaches the engine's or the published optimum through pyomo.
And each component entry is the right one: the row at ``m.balance[1]`` holds
the terms [`Model.row`][] reads at that coordinate, which an index shifted by
one row would break while leaving every objective intact.
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

pyo = pytest.importorskip('pyomo.environ', reason='to_pyomo needs the [pyomo] extra')

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


def _solved(m: Any, solver: str = 'appsi_highs') -> Any:
    outcome = pyo.SolverFactory(solver).solve(m)
    assert outcome.solver.termination_condition == pyo.TerminationCondition.optimal
    return pyo.value(m.objective)


def _exported(spec: Any, data: Any) -> tuple[Any, Any]:
    """The pyomo model, and the engine's own answer to the same build."""
    with sps.build(spec, data) as model:
        m = model.to_pyomo()
        return m, model.solve()


@pytest.mark.parametrize(('spec', 'data'), LINEAR)
def test_the_exported_model_reaches_the_optimum_the_engine_reaches(spec: Any, data: Any) -> None:
    m, answer = _exported(spec, data)
    assert _solved(m) == pytest.approx(answer.objective), 'pyomo solved a different model from the one the engine built'


@pytest.mark.parametrize(
    'spec',
    [
        pytest.param(QUADRATIC_OBJECTIVE_SPEC, id='a-quadratic-objective'),
        pytest.param(QUADRATIC_ROW_SPEC, id='a-quadratic-constraint'),
    ],
)
def test_a_quadratic_model_reaches_the_engine_optimum(spec: Any) -> None:
    """appsi's HiGHS interface is linear only, so Gurobi solves both sides."""
    pytest.importorskip('gurobipy', reason='no pyomo interface here takes a quadratic model without it')
    with sps.build(spec, QUADRATIC_DATA) as model:
        m = model.to_pyomo()
        direct = model.solve(solver_name='gurobi').objective
    assert _solved(m, 'gurobi_direct') == pytest.approx(direct), 'the quadratic terms changed on the way to pyomo'


@pytest.mark.parametrize('name', sorted(PORT_REFERENCES), ids=str)
def test_every_referenced_model_reaches_its_optimum_through_pyomo(name: str) -> None:
    """The corpus at its own sizes, each formulation written out so HiGHS reads it."""
    with sps.build(expanded(port_spec(name)), port_sources(name)) as model:
        m = model.to_pyomo()
    assert _solved(m) == pytest.approx(PORT_REFERENCES[name]['objective'], rel=1e-6), (
        'the pyomo model misses the published optimum to 1e-6'
    )


def _named(m: Any, name: str, kind: Any, suffix: str) -> Any:
    """The component a declaration became: under its own name, or suffixed where that was taken."""
    held = m.component(name)
    return held if isinstance(held, kind) else m.component(f'{name}_{suffix}')


def _terms(body: Any) -> dict[str, float]:
    from pyomo.repn import generate_standard_repn

    repn = generate_standard_repn(body, compute_values=False)
    return {v.name: c for v, c in zip(repn.linear_vars, repn.linear_coefs, strict=True)}


@pytest.mark.parametrize(('spec', 'data'), LINEAR)
def test_every_row_holds_the_terms_the_built_row_holds(spec: Any, data: Any) -> None:
    """Rows against [`Model.row`][], which finds a row by its coordinate and not by position.

    Every row the build made is checked, and nothing else is in the component,
    so a row shifted by one, or one the build dropped, fails here.
    """
    with sps.build(spec, data) as model:
        m = model.to_pyomo()
        result = model.solve()
        for constraint, declared in model._program.constraints.items():
            built = result.activity(constraint).drop('value')
            coordinates = list(built.iter_rows()) if declared.dims else [()]
            component = _named(m, constraint, pyo.Constraint, 'constraint')
            assert len(component) == len(coordinates), f"'{constraint}' has a row the build did not make"
            for coordinate in coordinates:
                row = model.row(constraint, **dict(zip(declared.dims, coordinate, strict=True)))
                expected = {
                    f'{_named(m, t["variable"], pyo.Var, "variable").name}[{t["coordinate"].replace(", ", ",")}]'
                    if t['coordinate']
                    else _named(m, t['variable'], pyo.Var, 'variable').name: t['coefficient']
                    for t in row.terms.iter_rows(named=True)
                }
                held = component[coordinate[0] if len(coordinate) == 1 else coordinate] if coordinate else component
                bound = pyo.value(held.lower if row.sense == '>=' else held.upper)
                assert _terms(held.body) == expected, f'{constraint}{coordinate} holds another row'
                assert bound == pytest.approx(row.rhs), f'{constraint}{coordinate} compares against another bound'
                assert held.equality == (row.sense == '=='), f'{constraint}{coordinate} changed its sense'


def test_each_variable_reads_back_the_value_the_engine_solved() -> None:
    """The dispatch optimum is unique, so pyomo's value at an index is the engine's at that coordinate."""
    m, answer = _exported(DISPATCH_SPEC, DISPATCH_DATA)
    _solved(m)
    expected = {(row['snapshot'], row['generator']): row['value'] for row in answer.primal('p').iter_rows(named=True)}
    assert {index: m.p[index].value for index in m.p} == pytest.approx(expected), 'an index labels another coordinate'


def test_a_masked_coordinate_has_no_entry() -> None:
    m, _ = _exported(MASKED_SPEC, MASKED_DATA)
    assert list(m.x) == [0, 2], 'the coordinate cap removed is absent, and one dim indexes by the bare label'
    assert list(m.c) == [0, 2], 'the row with no term is not built'
    assert not m.total.is_indexed(), 'a dimensionless variable is a scalar'


#: Every column of ``x`` masked, so anything summing over it has no term.
EMPTIED_DATA = {'j': [0, 1], 'cap': pl.DataFrame({'j': [0, 1], 'value': [0.0, 0.0]})}


def test_a_scalar_constraint_the_build_dropped_has_no_row() -> None:
    """The build drops a row with no variable term, and a scalar constraint then has none.

    The scalar's rule read its one row unconditionally, so the export raised
    ``KeyError: ()`` on a model the engine solves.
    """
    spec = {
        'dimensions': {'j': {'dtype': 'int'}},
        'parameters': {'cap': {'dims': ['j']}},
        'variables': {
            'x': {'dims': ['j'], 'where': 'cap > 0', 'bounds': {'lower': 0, 'upper': 10}},
            'y': {'dims': [], 'bounds': {'lower': 1, 'upper': 2}},
        },
        'constraints': {'c': {'dims': [], 'expression': 'sum(x, over=j) <= 5'}},
        'objective': {'sense': 'minimize', 'expression': 'y'},
    }
    m, answer = _exported(spec, EMPTIED_DATA)
    assert len(m.c) == 0, 'the row the build dropped is not in the component'
    assert _solved(m) == pytest.approx(answer.objective), 'the model without the row is the one the engine solved'


def test_a_scalar_set_over_a_fully_masked_variable_has_no_set() -> None:
    """Ordered along its only dim, a variable with every column masked leaves the one set empty.

    The scalar's rule read that set unconditionally, so the export raised
    ``KeyError: ()``.
    """
    spec = {
        'dimensions': {'j': {'dtype': 'int'}},
        'parameters': {'cap': {'dims': ['j']}},
        'variables': {
            'x': {'dims': ['j'], 'where': 'cap > 0', 'bounds': {'lower': 0, 'upper': 10}},
            'y': {'dims': [], 'bounds': {'lower': 1, 'upper': 2}},
        },
        'objective': {'sense': 'minimize', 'expression': 'y'},
        'sos': {'pick': {'variable': 'x', 'along': 'j', 'type': 1}},
    }
    with sps.build(spec, EMPTIED_DATA) as model:
        m = model.to_pyomo()
    assert len(m.pick) == 0, 'a set with no member is not built'


def test_a_row_of_only_zeros_stays_infeasible() -> None:
    """``0 >= 10`` has no terms, and pyomo would drop it as a trivial relation if it were a number."""
    spec = {
        'dimensions': {'j': {'dtype': 'int'}},
        'parameters': {'a': {'dims': ['j']}},
        'variables': {'x': {'dims': ['j'], 'bounds': {'lower': 0, 'upper': 10}}},
        'constraints': {'c': {'dims': [], 'expression': 'sum(a * x, over=j) >= 10'}},
        'objective': {'sense': 'minimize', 'expression': 'sum(x, over=j)'},
    }
    with sps.build(spec, {'j': [0, 1], 'a': pl.DataFrame({'j': [0, 1], 'value': [0.0, 0.0]})}) as model:
        m = model.to_pyomo()
    outcome = pyo.SolverFactory('appsi_highs').solve(m, load_solutions=False)
    assert outcome.solver.termination_condition == pyo.TerminationCondition.infeasible


def test_a_label_keeps_its_own_type() -> None:
    spec = {
        'dimensions': {'hour': {'dtype': 'datetime'}},
        'variables': {'p': {'dims': ['hour'], 'bounds': {'lower': 1, 'upper': 5}}},
        'objective': {'sense': 'minimize', 'expression': 'sum(p, over=hour)'},
    }
    hour = datetime.datetime(2030, 1, 1, 6)
    m, _ = _exported(spec, {'hour': [hour]})
    assert list(m.p) == [hour], 'the index is the datetime itself, not its string'


def test_a_model_with_no_objective_exports_none() -> None:
    spec = {k: v for k, v in DISPATCH_SPEC.items() if k != 'objective'}
    with sps.build(spec, DISPATCH_DATA) as model:
        m = model.to_pyomo()
    assert not hasattr(m, 'objective'), 'a feasibility problem has no objective to add'


@pytest.mark.parametrize('sos_type', [1, 2])
def test_a_set_is_indexed_by_its_members_coordinate_without_the_ordering_dim(sos_type: int) -> None:
    """HiGHS takes no SOS, so Gurobi solves it, as ``test_sos`` does."""
    pytest.importorskip('gurobipy', reason='no solver here takes an SOS without it')
    spec = sos_spec(sos_type)
    with sps.build(spec, SOS_DATA) as model:
        m = model.to_pyomo()
    (name, declared), *_ = spec['sos'].items()
    component = getattr(m, name)
    assert all(len(list(component[i].get_variables())) > 1 for i in component), 'every set orders more than one member'
    assert _solved(m, 'gurobi_direct') == pytest.approx(best(sos_type)), (
        f'the exported {declared} sets do not restrict what they should'
    )


@pytest.mark.parametrize('sos_type', [1, 2])
def test_a_set_over_a_variable_with_no_other_dim_is_one_scalar_set(sos_type: int) -> None:
    """With the ordering dim the variable's only one, nothing is left to index the set by, and there is one set."""
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
        m = model.to_pyomo()
    assert not m.pick.is_indexed(), 'one set, with nothing left to index it by'
    assert len(list(m.pick.get_variables())) == len(SIZES), 'the set orders every member'
    assert _solved(m, 'gurobi_direct') == pytest.approx(optimum), 'the one set does not restrict what it should'


@pytest.mark.parametrize('name', ['write', 'name', 'component'], ids=str)
def test_a_declaration_named_like_a_model_attribute_is_refused(name: str) -> None:
    spec = {
        'variables': {name: {'dims': [], 'bounds': {'lower': 1}}},
        'objective': {'sense': 'minimize', 'expression': name},
    }
    with sps.build(spec, {}) as model, pytest.raises(SpecsolveError, match=rf"'{name}' is already an attribute"):
        model.to_pyomo()


def test_a_variable_named_like_a_result_suffix_is_suffixed_with_its_kind() -> None:
    """A component called ``slack`` is where pyomo's solvers write slacks, so a solve would fail."""
    m, answer = _exported(FREE_SPEC, FREE_DATA)
    assert isinstance(m.slack_variable, pyo.Var), 'the variable takes the name with its kind appended'
    assert not hasattr(m, 'slack'), 'nothing is left where a solver looks for its result suffix'
    assert _solved(m) == pytest.approx(answer.objective), 'the renamed model still solves'


def test_a_constraint_named_like_a_variable_is_suffixed_with_its_kind() -> None:
    spec = {
        'variables': {'x': {'dims': [], 'bounds': {'lower': 0}}},
        'constraints': {'x': {'dims': [], 'expression': 'x >= 2'}},
        'objective': {'sense': 'minimize', 'expression': 'x'},
    }
    m, _ = _exported(spec, {})
    assert isinstance(m.x, pyo.Var), 'the variable keeps the name'
    assert m.x_constraint.lower == 2, 'the constraint takes the name with its kind appended'


def test_without_pyomo_the_export_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    with sps.build(DISPATCH_SPEC, DISPATCH_DATA) as model:
        monkeypatch.setitem(sys.modules, 'pyomo.environ', None)
        with pytest.raises(SpecsolveError, match=r'pip install "specsolve\[pyomo\]"'):
            model.to_pyomo()
