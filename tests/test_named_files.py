"""``names=True`` writes each column and row by its declaration and coordinate.

The claim has two halves. The file is the same model: every fixture and every
port reaches the optimum the numbered file reaches. And each name is the right
one: a row read back by name holds the terms [`Model.row`][] reads out of the
built model at that coordinate, which a name list off by one position would
break while leaving every objective intact.
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any

import polars as pl
import pytest

import specsolve as sps
from specsolve.errors import SpecsolveError
from tests.conftest import DISPATCH_SPEC, PORT_REFERENCES, expanded, port_sources, port_spec, solve_written_file
from tests.test_mps_text import (
    COMMITMENT,
    COMMITMENT_DATA,
    DISPATCH_DATA,
    FREE_DATA,
    FREE_SPEC,
    QUADRATIC_ROW_SPEC,
)
from tests.test_quadratic_objective import SOURCES as QUADRATIC_DATA
from tests.test_quadratic_objective import SPEC as QUADRATIC_OBJECTIVE_SPEC
from tests.test_sos import DATA as SOS_DATA
from tests.test_sos import best
from tests.test_sos import spec as sos_spec

if TYPE_CHECKING:
    from pathlib import Path

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


def _highs(path: Path) -> Any:
    import highspy

    h = highspy.Highs()
    h.setOptionValue('output_flag', False)
    h.readModel(str(path))
    return h


def _rows_by_name(path: Path) -> dict[str, dict[str, float]]:
    """Each row HiGHS reads back, as ``{column name: coefficient}`` under its own name."""
    lp = _highs(path).getLp()
    rows: dict[str, dict[str, float]] = {name: {} for name in lp.row_names_}
    matrix = lp.a_matrix_
    for col, col_name in enumerate(lp.col_names_):
        for k in range(matrix.start_[col], matrix.start_[col + 1]):
            rows[lp.row_names_[matrix.index_[k]]][col_name] = matrix.value_[k]
    return rows


@pytest.mark.parametrize('suffix', ['.lp', '.mps'])
@pytest.mark.parametrize(('spec', 'data'), LINEAR)
def test_a_named_file_reaches_the_optimum_the_numbered_one_does(
    spec: Any, data: Any, suffix: str, tmp_path: Path
) -> None:
    numbered = sps.write(spec, data, tmp_path / f'numbered{suffix}')
    named = sps.write(spec, data, tmp_path / f'named{suffix}', names=True)
    assert solve_written_file(named) == pytest.approx(solve_written_file(numbered)), (
        'naming the columns and rows changed the model the file describes'
    )


def test_a_named_quadratic_objective_reaches_the_numbered_optimum(tmp_path: Path) -> None:
    """LP only: MPS refuses a quadratic model before it reaches a name."""
    numbered = sps.write(QUADRATIC_OBJECTIVE_SPEC, QUADRATIC_DATA, tmp_path / 'numbered.lp')
    named = sps.write(QUADRATIC_OBJECTIVE_SPEC, QUADRATIC_DATA, tmp_path / 'named.lp', names=True)
    assert solve_written_file(named) == pytest.approx(solve_written_file(numbered)), (
        'naming the columns changed the quadratic objective'
    )


def test_a_named_quadratic_constraint_reaches_the_numbered_optimum(tmp_path: Path) -> None:
    """HiGHS reads no quadratic row, so Gurobi does, as ``test_quadratic_constraint`` does."""
    gp = pytest.importorskip('gurobipy', reason='no reader here takes a quadratic row without it')
    objectives = []
    for path, names in ((tmp_path / 'numbered.lp', False), (tmp_path / 'named.lp', True)):
        sps.write(QUADRATIC_ROW_SPEC, QUADRATIC_DATA, path, names=names)
        with gp.Env(params={'OutputFlag': 0}) as env, gp.read(str(path), env) as model:
            model.optimize()
            objectives.append(model.ObjVal)
    assert objectives[1] == pytest.approx(objectives[0]), 'naming the rows changed the quadratic model'


@pytest.mark.parametrize('suffix', ['.lp', '.mps'])
@pytest.mark.parametrize('name', sorted(PORT_REFERENCES), ids=str)
def test_every_referenced_model_reaches_its_optimum_through_a_named_file(
    name: str, suffix: str, tmp_path: Path
) -> None:
    """The corpus at its own sizes, where a collision or a keyword would first turn up."""
    path = sps.write(expanded(port_spec(name)), port_sources(name), tmp_path / f'{name}{suffix}', names=True)
    assert solve_written_file(path) == pytest.approx(PORT_REFERENCES[name]['objective'], rel=1e-6), (
        'the named file misses the published optimum to 1e-6'
    )


@pytest.mark.parametrize('suffix', ['.lp', '.mps'])
@pytest.mark.parametrize(('spec', 'data'), LINEAR)
def test_every_named_row_holds_the_terms_the_built_row_holds(spec: Any, data: Any, suffix: str, tmp_path: Path) -> None:
    """Names against [`Model.row`][], which finds a row by its coordinate and not by position.

    Every row the build made is checked, so a name list shifted by one, or one
    that kept a row the build dropped, names some row's terms wrongly.
    """
    with sps.build(spec, data) as model:
        model.write(tmp_path / f'model{suffix}', names=True)
        result = model.solve()
        expected = {}
        for constraint, declared in model._program.constraints.items():
            built_at = result.activity(constraint).drop('value')
            for coordinate in built_at.iter_rows(named=True) if declared.dims else [{}]:
                built = model.row(constraint, **coordinate)
                label = ','.join(str(coordinate[d]) for d in declared.dims)
                expected[f'{constraint}({label})'] = {
                    f'{t["variable"]}({t["coordinate"].replace(", ", ",")})': t['coefficient']
                    for t in built.terms.iter_rows(named=True)
                }
    assert _rows_by_name(tmp_path / f'model{suffix}') == expected, 'a row name labels another row, or its terms'


def test_a_quadratic_constraint_declared_first_keeps_its_names(tmp_path: Path) -> None:
    """A quadratic row is built after every linear one, whatever the declaration order.

    So the names follow the build, not the file: had they followed the file,
    the quadratic rows would carry the linear constraint's names.
    """
    gp = pytest.importorskip('gurobipy', reason='no reader here takes a quadratic row without it')
    spec = {
        **QUADRATIC_ROW_SPEC,
        'constraints': {
            'coupled': QUADRATIC_ROW_SPEC['constraints']['coupled'],
            **QUADRATIC_OBJECTIVE_SPEC['constraints'],
        },
    }
    path = sps.write(spec, QUADRATIC_DATA, tmp_path / 'model.lp', names=True)
    with gp.Env(params={'OutputFlag': 0}) as env, gp.read(str(path), env) as model:
        quadratic = {q.QCName.split('(')[0] for q in model.getQConstrs()}
        linear = {c.ConstrName.split('(')[0] for c in model.getConstrs()}
    assert quadratic == {'coupled'}, "a quadratic row carries a linear constraint's name"
    assert 'coupled' not in linear, "a linear row carries the quadratic constraint's name"


def test_each_named_column_reads_back_the_value_the_engine_solved(tmp_path: Path) -> None:
    """The dispatch optimum is unique, so HiGHS's value under a name is the engine's value at that coordinate."""
    with sps.build(DISPATCH_SPEC, DISPATCH_DATA) as model:
        model.write(tmp_path / 'model.lp', names=True)
        primal = model.solve().primal('p')
    expected = {f'p({row["snapshot"]},{row["generator"]})': row['value'] for row in primal.iter_rows(named=True)}

    h = _highs(tmp_path / 'model.lp')
    h.run()
    read = dict(zip(h.getLp().col_names_, h.getSolution().col_value, strict=True))
    assert read == pytest.approx(expected), 'a column name labels another coordinate'


#: One dimension whose labels need cleaning, each beside the name it writes.
CLEANED = [
    pytest.param('north sea', 'p(north_sea)', id='a-space'),
    pytest.param('H2+CH4', 'p(H2_CH4)', id='an-operator'),
    pytest.param('a,b', 'p(a_b)', id='the-separator'),
    pytest.param('f(x)', 'p(f_x_)', id='the-brackets'),
    pytest.param('Zürich', 'p(Zürich)', id='a-letter-outside-ascii'),
    pytest.param('#1@x.y', 'p(#1@x.y)', id='punctuation-the-readers-take'),
]


@pytest.mark.parametrize('suffix', ['.lp', '.mps'])
@pytest.mark.parametrize(('label', 'written'), CLEANED)
def test_a_label_the_format_cannot_hold_is_written_with_an_underscore(
    label: str, written: str, suffix: str, tmp_path: Path
) -> None:
    spec = {
        'dimensions': {'g': {'dtype': 'str'}},
        'variables': {'p': {'dims': ['g'], 'bounds': {'lower': 1, 'upper': 5}}},
        'objective': {'sense': 'minimize', 'expression': 'sum(p, over=g)'},
    }
    path = sps.write(spec, {'g': [label, 'other']}, tmp_path / f'model{suffix}', names=True)
    h = _highs(path)
    assert h.getLp().col_names_ == [written, 'p(other)'], f'{label!r} did not come back as {written!r}'


def test_a_datetime_label_is_written_in_its_iso_spelling(tmp_path: Path) -> None:
    spec = {
        'dimensions': {'hour': {'dtype': 'datetime'}},
        'variables': {'p': {'dims': ['hour'], 'bounds': {'lower': 1, 'upper': 5}}},
        'objective': {'sense': 'minimize', 'expression': 'sum(p, over=hour)'},
    }
    path = sps.write(spec, {'hour': [datetime.datetime(2030, 1, 1, 6)]}, tmp_path / 'model.lp', names=True)
    assert _highs(path).getLp().col_names_ == ['p(2030_01_01_06_00_00.000000)'], (
        'the dashes, colons and space a reader refuses become underscores'
    )


@pytest.mark.parametrize('suffix', ['.lp', '.mps'])
def test_a_declaration_named_like_a_keyword_still_reads_back(suffix: str, tmp_path: Path) -> None:
    """``end`` and ``st`` close and open LP sections, so a bare name would end the file early.

    The parentheses a dimensionless declaration carries are what keep them names.
    """
    spec = {
        'variables': {'end': {'dims': [], 'bounds': {'lower': 0}}},
        'constraints': {'st': {'dims': [], 'expression': 'end >= 3'}},
        'objective': {'sense': 'minimize', 'expression': 'end'},
    }
    path = sps.write(spec, {}, tmp_path / f'model{suffix}', names=True)
    assert solve_written_file(path) == pytest.approx(3.0), 'the reader took a keyword for the end of a section'
    assert _rows_by_name(path) == {'st()': {'end()': 1.0}}, 'the scalar names are spelled with their parentheses'


#: Two labels that clean to one name, on a variable and on a constraint alone.
SHARED = [
    pytest.param(
        {'p': {'dims': ['g'], 'bounds': {'lower': 1, 'upper': 5}}},
        {},
        'sum(p)',
        "variable 'p'",
        id='a-variable',
    ),
    pytest.param(
        {'q': {'dims': [], 'bounds': {'lower': 0}}},
        {'floor': {'dims': ['g'], 'expression': 'q >= lo'}},
        'q',
        "constraint 'floor'",
        id='a-constraint',
    ),
]


@pytest.mark.parametrize(('variables', 'constraints', 'objective', 'named'), SHARED)
def test_two_labels_that_write_as_one_name_are_refused(
    variables: dict[str, Any], constraints: dict[str, Any], objective: str, named: str, tmp_path: Path
) -> None:
    """A reader takes one name twice as one column, or refuses the file.

    Either is worse than a refusal here that names both labels.
    """
    with pytest.raises(SpecsolveError, match=rf"{named} writes the coordinates \('a b',\) and \('a_b',\)"):
        sps.write(_sharing(variables, constraints, objective), SHARED_DATA, tmp_path / 'model.lp', names=True)
    assert not (tmp_path / 'model.lp').exists(), 'a refused write leaves no file behind'


@pytest.mark.parametrize('suffix', ['.lp', '.mps'])
@pytest.mark.parametrize(('variables', 'constraints', 'objective', 'named'), SHARED)
def test_check_with_names_refuses_what_a_named_write_refuses(
    variables: dict[str, Any], constraints: dict[str, Any], objective: str, named: str, suffix: str, tmp_path: Path
) -> None:
    """Without *names* the same model checks clean: a numbered file has no label to share."""
    with sps.build(_sharing(variables, constraints, objective), SHARED_DATA) as model:
        model.check(suffix)
        with pytest.raises(SpecsolveError, match=named) as checked:
            model.check(suffix, names=True)
        with pytest.raises(SpecsolveError) as written:
            model.write(tmp_path / f'model{suffix}', names=True)
    assert str(checked.value) == str(written.value), 'check and write refuse a shared name in different words'


def test_check_refuses_names_for_a_sink_that_writes_no_file() -> None:
    """A solver reads no names, so the question ``names=`` asks has no answer there."""
    with (
        sps.build(DISPATCH_SPEC, DISPATCH_DATA) as model,
        pytest.raises(SpecsolveError, match=r"names= is read by a written file \(\.lp, \.mps\), not by 'highs'"),
    ):
        model.check('highs', names=True)


#: The labels ``a b`` and ``a_b``, which clean to one name.
SHARED_DATA = {'g': ['a b', 'a_b'], 'lo': {'a b': 1.0, 'a_b': 2.0}}


def _sharing(variables: dict[str, Any], constraints: dict[str, Any], objective: str) -> dict[str, Any]:
    return {
        'dimensions': {'g': {'dtype': 'str'}},
        'parameters': {'lo': {'dims': ['g']}},
        'variables': variables,
        'constraints': constraints,
        'objective': {'sense': 'minimize', 'expression': objective},
    }


@pytest.mark.parametrize('suffix', ['.lp', '.mps'])
@pytest.mark.parametrize('sos_type', [1, 2])
def test_a_named_set_survives_the_file(sos_type: int, suffix: str, tmp_path: Path) -> None:
    """HiGHS refuses an SOS section, so Gurobi reads it, as ``test_sos`` does."""
    pytest.importorskip('gurobipy', reason='no reader here takes an SOS section without it')
    import gurobipy as gp

    path = sps.write(sos_spec(sos_type), SOS_DATA, tmp_path / f'model{suffix}', names=True)
    with gp.Env(params={'OutputFlag': 0}) as env, gp.read(str(path), env) as model:
        model.optimize()
        assert model.ObjVal == pytest.approx(best(sos_type)), 'the named sets do not restrict what they should'
