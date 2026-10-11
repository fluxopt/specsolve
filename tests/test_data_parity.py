"""Malformed *data*, checked for the same verdict on both lanes.

`test_resolution_parity.py` does this for the language (hard rule 3). The
table is the contract #351 preserves. Each case carries data twice, once per
lane's preferred shape, so two representations of the same mistake get the
same answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import polars as pl
import pytest
import yaml as pyyaml

import specsolve as sps
from specsolve.errors import DataError
from tests.differential import both_lanes_refuse, differential
from tests.oracle import pd, specsolve_linopy  # skips the module without the oracle

if TYPE_CHECKING:
    from pathlib import Path

#: One dimension, one variable, a coefficient and a bound.
SPEC = {
    'dimensions': {'f': {'dtype': 'str'}},
    'parameters': {'cost': {'dims': ['f']}, 'cap': {'dims': ['f']}},
    'variables': {'x': {'dims': ['f'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
    'constraints': {'k': {'dims': ['f'], 'expression': 'x <= cap'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(x * cost)'},
}

ACCEPTED = 'accepted'


def _tidy(**cols: list[Any]) -> pl.DataFrame:
    return pl.DataFrame(cols)


def _written(tmp_path: Path, spec: dict) -> Path:
    """*spec* on disk, because the linopy lane only takes a path."""
    path = tmp_path / 'spec.yaml'
    path.write_text(pyyaml.safe_dump(spec))
    return path


@dataclass(frozen=True)
class Case:
    """One malformed (or valid) attachment, in both representations."""

    label: str
    relational: dict[str, Any]
    linopy_lane: dict[str, Any]
    #: `DataError` for a refusal both lanes owe the caller, or `ACCEPTED`.
    verdict: type[Exception] | str


def _cases() -> list[Case]:
    index = {'f': ['a', 'b']}
    good_r = {**index, 'cost': _tidy(f=['a', 'b'], value=[1.0, 2.0]), 'cap': _tidy(f=['a', 'b'], value=[5.0, 5.0])}
    good_e = {**index, 'cost': pd.Series({'a': 1.0, 'b': 2.0}), 'cap': pd.Series({'a': 5.0, 'b': 5.0})}
    return [
        Case('valid', good_r, good_e, ACCEPTED),
        Case(
            'parameter missing entirely',
            {**index, 'cost': good_r['cost']},
            {**index, 'cost': good_e['cost']},
            DataError,
        ),
        Case(
            'bound parameter sparse',
            {**good_r, 'cap': _tidy(f=['a'], value=[5.0])},
            {**good_e, 'cap': pd.Series({'a': 5.0})},
            DataError,  # a missing bound has no reading, so law 8 refuses rather than guessing
        ),
        Case(
            'coefficient sparse',
            {**good_r, 'cost': _tidy(f=['a'], value=[1.0])},
            {**good_e, 'cost': pd.Series({'a': 1.0})},
            # The ordinary case: a missing row is a zero coefficient (the data-attachment rules).
            ACCEPTED,
        ),
        Case(
            'duplicated coordinate row',
            {**good_r, 'cost': _tidy(f=['a', 'a', 'b'], value=[1.0, 9.0, 2.0])},
            {**good_e, 'cost': pd.Series([1.0, 9.0, 2.0], index=pd.Index(['a', 'a', 'b'], name='f'))},
            # Which value applies is undefined, so neither lane may pick one.
            DataError,
        ),
        Case(
            'label the dimension does not have',
            {**good_r, 'cost': _tidy(f=['a', 'zz'], value=[1.0, 2.0])},
            {**good_e, 'cost': pd.Series({'a': 1.0, 'zz': 2.0})},
            # Present and unaddressable is a typo, not sparsity (#350).
            DataError,
        ),
        Case(
            'a null value',
            {**good_r, 'cost': _tidy(f=['a', 'b'], value=[1.0, None])},
            {**good_e, 'cost': pd.Series({'a': 1.0, 'b': None})},
            # A row claiming the coordinate while its value denies it says both at once.
            DataError,
        ),
        Case(
            'a NaN value',
            {**good_r, 'cost': _tidy(f=['a', 'b'], value=[1.0, float('nan')])},
            {**good_e, 'cost': pd.Series({'a': 1.0, 'b': float('nan')})},
            # The same hole, in the only spelling pandas has for one.
            DataError,
        ),
        Case(
            'an index holding a label twice',
            {**good_r, 'f': ['a', 'b', 'a']},
            {**good_e, 'f': ['a', 'b', 'a']},
            # A label's row is its position, so a second row gives it two.
            DataError,
        ),
        Case(
            'a hole in a bound',
            {**good_r, 'cap': _tidy(f=['a', 'b'], value=[5.0, None])},
            {**good_e, 'cap': pd.Series({'a': 5.0, 'b': None})},
            DataError,
        ),
    ]


CASES = _cases()


@pytest.fixture(scope='module')
def spec_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The linopy lane only takes a path, so the model has to hit disk once."""
    path = tmp_path_factory.mktemp('data-parity') / 'm.yaml'
    path.write_text(pyyaml.safe_dump(SPEC))
    return path


def _verdict_relational(path: Path, data: dict[str, Any]) -> type[Exception] | str:
    try:
        sps.solve(path, data)
    except DataError:
        return DataError
    return ACCEPTED


def _verdict_linopy(path: Path, data: dict[str, Any]) -> type[Exception] | str:
    try:
        m = specsolve_linopy.build(path, data)
        m.solve(solver_name='highs', output_flag=False)
    except DataError:
        return DataError
    return ACCEPTED


@pytest.mark.parametrize('case', CASES, ids=lambda c: c.label)
def test_both_lanes_reach_the_same_verdict(case: Case, spec_path: Path):
    """And it is the verdict the table names, not merely the same one."""
    relational = _verdict_relational(spec_path, case.relational)
    linopy_lane = _verdict_linopy(spec_path, case.linopy_lane)

    assert relational == case.verdict, f'{case.label}: relational lane'
    assert linopy_lane == case.verdict, f'{case.label}: linopy lane'


def test_the_table_covers_both_verdicts():
    """A table drifted to one verdict would still pass every assertion above."""
    verdicts = {c.verdict for c in CASES}
    assert verdicts == {ACCEPTED, DataError}, f'expected both verdicts to be exercised; got {verdicts}'


def test_a_hole_is_named_where_it_sits_rather_than_as_a_divisor(spec_path: Path):
    """`x * cost` divides by nothing, so a hole in `cost` is refused at attach as a hole."""
    index = {'f': ['a', 'b']}
    holed = {**index, 'cost': _tidy(f=['a', 'b'], value=[1.0, None]), 'cap': _tidy(f=['a', 'b'], value=[5.0, 5.0])}
    linopy_lane = {**index, 'cost': pd.Series({'a': 1.0, 'b': None}), 'cap': pd.Series({'a': 5.0, 'b': 5.0})}

    with pytest.raises(DataError, match="parameter 'cost'") as relational_error:
        sps.build(spec_path, holed).close()
    with pytest.raises(DataError, match="parameter 'cost'") as linopy_error:
        specsolve_linopy.build(spec_path, linopy_lane)

    assert 'divisor' not in str(relational_error.value), (
        'the message names the hole, not a divisor the model has not got'
    )
    assert 'f=b' in str(relational_error.value), 'and names the coordinate the hole sits at'
    assert str(relational_error.value) == str(linopy_error.value), 'one defect, one sentence'


@pytest.mark.parametrize(
    'holed',
    [
        pytest.param(float('nan'), id='a-scalar'),
        pytest.param([1.0, float('nan')], id='a-sequence'),
        pytest.param({'a': 1.0, 'b': None}, id='a-dict'),
        pytest.param(_tidy(f=['a', 'b'], value=[1.0, None]), id='a-tidy-frame'),
    ],
)
def test_a_hole_is_refused_in_every_shape_a_source_arrives_in(spec_path: Path, holed: Any):
    """One source object, both lanes — the linopy lane asks at a different site per shape."""
    sources = {'f': ['a', 'b'], 'cost': holed, 'cap': _tidy(f=['a', 'b'], value=[5.0, 5.0])}

    both_lanes_refuse(spec_path, sources, match='no value')


def test_a_hole_in_a_scalar_parameter_is_refused_on_both_lanes(tmp_path: Path):
    """`dims: []` attaches one value, and one value that is a hole is still a hole."""
    spec = {
        'dimensions': {'f': {'dtype': 'str'}},
        'parameters': {'rate': {'dims': []}},
        'variables': {'x': {'dims': ['f'], 'bounds': {'lower': 0, 'upper': 1}}},
        'constraints': {'k': {'dims': ['f'], 'expression': 'x <= 1'}},
        'objective': {'sense': 'maximize', 'expression': 'sum(x * rate)'},
    }
    path = _written(tmp_path, spec)
    sources = {'f': ['a', 'b'], 'rate': _tidy(value=[None])}

    both_lanes_refuse(path, sources, match='no value')


#: A model reading a parameter as a position.
POSITION_SPEC = {
    'dimensions': {'g': {'dtype': 'str'}, 't': {'dtype': 'int'}},
    'parameters': {'lead': {'dims': ['g'], 'dtype': 'int'}, 'demand': {'dims': ['g', 't']}},
    'variables': {'x': {'dims': ['g', 't'], 'bounds': {'lower': 0}}},
    'constraints': {'c': {'dims': ['g', 't'], 'expression': 'shift(x, along=t, offset=lead, edge=0) >= demand'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(x)'},
}

_DEMAND = _tidy(g=['a', 'a', 'a'], t=[0, 1, 2], value=[1.0, 2.0, 3.0])


def _position_sources(lead: Any) -> dict[str, Any]:
    return {'t': [0, 1, 2], 'g': ['a'], 'lead': _tidy(g=['a'], value=[lead]), 'demand': _DEMAND}


def test_an_int_declaration_takes_no_float_column_so_a_fraction_cannot_arrive(tmp_path: Path):
    """An `int` declaration takes an integer column, which has no fraction to hold."""
    path = _written(tmp_path, POSITION_SPEC)

    both_lanes_refuse(path, _position_sources(1.5), match="declared 'int'")
    with sps.solve(path, _position_sources(1)) as run:
        assert run.is_ok, 'and an integer column is the ordinary case'


def test_whole_numbers_serve_a_float_declaration(tmp_path: Path):
    """An integer column serves a `float` declaration: the one widening."""
    spec = {
        'dimensions': {'g': {'dtype': 'str'}},
        'parameters': {'cost': {'dims': ['g'], 'dtype': 'float'}},
        'variables': {'x': {'dims': ['g'], 'bounds': {'lower': 0, 'upper': 1}}},
        'constraints': {'k': {'dims': [], 'expression': 'sum(x, over=g) <= 9'}},
        'objective': {'sense': 'minimize', 'expression': 'sum(x * cost)'},
    }
    path = _written(tmp_path, spec)
    integral = {'g': ['a', 'b'], 'cost': _tidy(g=['a', 'b'], value=[1, 2])}

    with sps.solve(path, integral) as run:
        assert run.is_ok, 'an integer column serves a float declaration'
    assert specsolve_linopy.build(path, integral) is not None, 'and does so on both lanes'


#: A flag. Only a boolean column satisfies `dtype: bool`.
FLAG_SPEC = {
    'dimensions': {'g': {'dtype': 'str'}},
    'parameters': {'active': {'dims': ['g'], 'dtype': 'bool'}},
    'variables': {'x': {'dims': ['g'], 'where': 'active', 'bounds': {'lower': 0, 'upper': 1}}},
    'constraints': {'k': {'dims': [], 'expression': 'sum(x, over=g) <= 9'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(x)'},
}


@pytest.mark.parametrize(
    ('spelling', 'column'),
    [
        pytest.param('boolean', _tidy(g=['a', 'b'], value=[True, False]), id='a-boolean-column'),
        pytest.param('1/0 ints', _tidy(g=['a', 'b'], value=[1, 0]), id='a-1-0-int-column'),
        pytest.param('1.0/0.0 floats', _tidy(g=['a', 'b'], value=[1.0, 0.0]), id='a-1-0-float-column'),
    ],
)
def test_a_flag_masks_by_its_declaration_rather_than_by_its_storage(tmp_path: Path, spelling: str, column: Any):
    """A column that is not what it declares does not attach, so a 1/0 flag cannot mask nothing."""
    path = _written(tmp_path, FLAG_SPEC)
    sources = {'g': ['a', 'b'], 'active': column}

    if spelling == 'boolean':
        with sps.solve(path, sources) as run:
            assert run.objective == pytest.approx(1.0), 'the inactive column is masked away'
        return
    both_lanes_refuse(path, sources, match="declared 'bool'")


def test_a_bare_where_on_a_string_parameter_asks_whether_it_has_a_row(tmp_path: Path):
    """A string parameter is defined wherever it has a row."""
    spec = {
        'dimensions': {'g': {'dtype': 'str'}},
        'parameters': {'fuel': {'dims': ['g'], 'dtype': 'str'}},
        'variables': {'x': {'dims': ['g'], 'where': 'fuel', 'bounds': {'lower': 0, 'upper': 1}}},
        'constraints': {'k': {'dims': [], 'expression': 'sum(x, over=g) <= 9'}},
        'objective': {'sense': 'maximize', 'expression': 'sum(x)'},
    }
    path = _written(tmp_path, spec)
    sources = {'g': ['a', 'b'], 'fuel': _tidy(g=['a'], value=['gas'])}

    with sps.solve(path, sources) as run:
        assert run.objective == pytest.approx(1.0), 'defined is having a row, and only `a` has one'


#: A relation-carrying dimension.
RELATION_SPEC = {
    'dimensions': {'g': {}, 'b': {'dtype': 'str'}},
    'relations': {'gen_bus': {'key': 'g', 'values': 'b'}},
    'parameters': {'p_max': {'dims': ['g']}},
    'variables': {'x': {'dims': ['g'], 'bounds': {'lower': 0, 'upper': 'p_max'}}},
    'constraints': {'k': {'dims': ['b'], 'expression': 'sum(x, by=gen_bus, over=g, into=b) <= 10'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(x)'},
}

_P_MAX = {'p_max': _tidy(g=['w', 's'], value=[5.0, 5.0])}
_INDEX = {'g': _tidy(g=['w', 's']), 'b': _tidy(b=['n', 'e'])}
_MAP = {'gen_bus': _tidy(g=['w', 's'], b=['n', 'e'])}


@pytest.mark.parametrize(
    ('sources', 'match'),
    [
        pytest.param({**_P_MAX, **_MAP}, 'has its maps', id='a-map-and-no-labels'),
        pytest.param({**_P_MAX, **_INDEX}, 'no data provided for relation', id='an-index-and-no-map'),
        pytest.param(
            {**_P_MAX, **_INDEX, 'gen_bus': _tidy(g=['w', 's'], gen_bus=['n', 'e'])},
            r"must carry a column per column it declares, \['g', 'b'\]",
            id='a-map-named-after-itself-and-not-its-target',
        ),
        pytest.param(
            {**_P_MAX, **_MAP, 'g': _tidy(gg=['w', 's'])},
            "without a 'g' column",
            id='an-index-without-the-label-column',
        ),
        pytest.param(
            {**_P_MAX, **_INDEX, 'gen_bus': _tidy(g=['w', 's'], b=['n', 'zz'])},
            'not .b. labels',
            id='a-relation-value-that-is-no-label-of-its-target',
        ),
        pytest.param(
            {**_P_MAX, **_INDEX, 'gen_bus': _tidy(g=['w', 'w', 's'], b=['n', 'e', 'e'])},
            'more than once',
            id='a-relation-with-two-values-for-one-label',
        ),
        pytest.param(
            {**_P_MAX, **_INDEX, 'gen_bus': _tidy(g=['w', 's'], b=[None, 'e'])},
            "null in 'b'",
            id='a-relation-mapping-a-label-to-nothing',
        ),
        pytest.param(
            {**_P_MAX, **_MAP, 'g': _tidy(g=['w', 's'], gen_bus=['n', 'e'])},
            "is a relation with a column over 'g'",
            id='a-map-carried-on-the-index-it-runs-over',
        ),
    ],
)
def test_a_relation_defect_reads_the_same_on_both_lanes(tmp_path, sources, match):
    """One wording, not two, from the one door both lanes enter."""
    path = _written(tmp_path, RELATION_SPEC)

    both_lanes_refuse(path, sources, match=match)


def test_an_index_a_declared_map_is_read_against_is_checked_before_the_read(tmp_path):
    """The front door, not the engine, refuses an index without its label column."""
    spec = {**RELATION_SPEC, 'relations': {'gen_bus': {'key': 'g', 'values': 'b'}}}
    path = _written(tmp_path, spec)
    sources = {**_P_MAX, **_MAP, 'b': _tidy(b=['n', 'e']), 'g': _tidy(gg=['w', 's'])}

    both_lanes_refuse(path, sources, match="without a 'g' column")


def test_a_relation_a_label_holds_twice_is_refused_before_it_can_drop_a_row(tmp_path):
    """A label holding a null and a value is two values, on both lanes.

    pandas `nunique()` skips nulls where polars `n_unique()` counts them.
    """
    spec = {
        **RELATION_SPEC,
        'dimensions': {'g': {}, 'b': {'dtype': 'str'}},
        'constraints': {'k': {'dims': ['b'], 'expression': 'sum(x, by=gen_bus, over=g, into=b) <= 3'}},
    }
    path = _written(tmp_path, spec)
    clean = {**_P_MAX, **_INDEX, 'gen_bus': _tidy(g=['w', 's'], b=['n', 'n'])}
    holed = {**_P_MAX, **_INDEX, 'gen_bus': _tidy(g=['w', 'w', 's'], b=[None, 'n', 'n'])}

    with sps.solve(path, clean) as run:
        assert run.objective == pytest.approx(3.0), 'both members are on the bus, and the bus caps them'
    built = specsolve_linopy.build(path, clean)
    built.solve(solver_name='highs', output_flag=False)
    assert float(built.objective.value) == pytest.approx(3.0), 'and the linopy lane agrees where the index is clean'

    both_lanes_refuse(path, holed, match="null in 'b'")


def test_a_dimension_index_is_a_table_on_both_lanes(tmp_path):
    """A polars index under the dimension's own key reaches both lanes."""
    path = _written(tmp_path, RELATION_SPEC)
    sources = {**_P_MAX, **_INDEX, **_MAP}

    with sps.solve(path, sources) as relational:
        assert relational.is_ok
    built = specsolve_linopy.build(path, sources)
    assert set(built.variables['x'].coords['g'].to_numpy()) == {'w', 's'}, 'the linopy lane read the same index'


def test_a_dimension_index_may_be_a_parquet_path_without_pyarrow(tmp_path, monkeypatch):
    """The oracle brings pandas and xarray, and nothing says it brings pyarrow."""
    import sys

    path = _written(tmp_path, RELATION_SPEC)
    index = tmp_path / 'g.parquet'
    _tidy(g=['w', 's']).write_parquet(index)
    sources = {**_P_MAX, **_MAP, 'b': _tidy(b=['n', 'e']), 'g': str(index)}

    monkeypatch.setitem(sys.modules, 'pyarrow', None)
    with sps.solve(path, sources) as relational:
        assert relational.is_ok
    built = specsolve_linopy.build(path, sources)
    assert set(built.variables['x'].coords['g'].to_numpy()) == {'w', 's'}, 'the linopy lane read the same path'


#: A relation whose target is a temporal dimension.
TEMPORAL_RELATION_SPEC = {
    'dimensions': {'g': {}, 'd': {'dtype': 'datetime'}},
    'relations': {'day_of': {'key': 'g', 'values': 'd'}},
    'parameters': {'p_max': {'dims': ['g']}, 'cap': {'dims': ['d']}},
    'variables': {'x': {'dims': ['g'], 'bounds': {'lower': 0, 'upper': 'p_max'}}},
    'constraints': {'k': {'dims': ['d'], 'expression': 'sum(x, by=day_of, over=g, into=d) <= cap'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(x, over=g)'},
}


def _days() -> list[Any]:
    import datetime

    return [datetime.date(2030, 1, 1), datetime.date(2030, 1, 2)]


def test_a_date_under_a_dimension_not_declared_datetime_stays_a_date_on_both_lanes(tmp_path):
    """Only a ``datetime`` dimension reads a date as the instant its day starts on; another keeps what it was given."""
    days = _days()
    spec = {**TEMPORAL_RELATION_SPEC, 'dimensions': {'g': {}, 'd': {}}}
    sources = {
        **_P_MAX,
        'd': days,
        'g': ['w', 's'],
        'cap': _tidy(d=days, value=[3.0, 7.0]),
        'day_of': _tidy(g=['w', 's'], d=[days[0], days[0]]),
    }
    with differential(_written(tmp_path, spec), sources) as run:
        assert run.oracle == pytest.approx(3.0), 'one day, one cap, both members under it, on both lanes'
        assert run.result.dual('k')['d'].to_list() == days[:1], 'the day with members comes back as the date it was'


def _spelled(spelling: str, columns: dict[str, list[Any]], path: Path) -> Any:
    """*columns*, their ``d`` column of dates in *spelling*: the table a caller holds for the same instants."""
    if spelling == 'pandas date':
        return pd.DataFrame(columns)
    if spelling == 'pandas datetime64':
        return pd.DataFrame({**columns, 'd': pd.to_datetime(columns['d'])})
    table = pl.DataFrame(columns)
    if spelling == 'polars Date':
        return table
    nanoseconds = table.with_columns(pl.col('d').cast(pl.Datetime('ns')))
    if spelling == 'polars Datetime ns':
        return nanoseconds
    nanoseconds.write_parquet(path)
    return str(path)


#: Five spellings of one day, as a caller holds them.
_SPELLINGS = {
    'pandas date': 'pandas-date',
    'pandas datetime64': 'pandas-datetime64',
    'polars Date': 'polars-date',
    'polars Datetime ns': 'polars-datetime-ns',
    'a parquet path': 'parquet-datetime-ns',
}


@pytest.mark.parametrize('column', list(_SPELLINGS), ids=[f'column-{i}' for i in _SPELLINGS.values()])
@pytest.mark.parametrize('index', list(_SPELLINGS), ids=[f'index-{i}' for i in _SPELLINGS.values()])
def test_a_relation_into_a_temporal_dimension_is_one_instant_on_both_lanes(tmp_path, index, column):
    """A relation's values and a parameter's labels are labels of the dimension, canonicalised as those are.

    Both members map to the same day, so that day's cap binds them together.
    A date and a midnight datetime are one instant, whichever spells the index
    and whichever the columns (#1630). A date was compared with a datetime and
    matched nothing, so it was refused as not a label of its dimension; and the
    seconds `pd.to_datetime` gives a date reached polars, which reads no such
    unit, as a bare `ValueError` naming nothing.
    """
    days = _days()
    sources = {
        **_P_MAX,
        'd': _spelled(index, {'d': days}, tmp_path / 'd.parquet'),
        'g': ['w', 's'],
        'cap': _spelled(column, {'d': days, 'value': [3.0, 7.0]}, tmp_path / 'cap.parquet'),
        'day_of': _spelled(column, {'g': ['w', 's'], 'd': [days[0], days[0]]}, tmp_path / 'day_of.parquet'),
    }
    with differential(_written(tmp_path, TEMPORAL_RELATION_SPEC), sources) as run:
        assert run.oracle == pytest.approx(3.0), 'one day, one cap, both members under it, on both lanes'


def test_a_stray_relation_value_reads_the_same_over_an_int_labelled_target(tmp_path):
    """One sentence, and the labels in it spelled as the caller wrote them."""
    spec = {
        **RELATION_SPEC,
        'dimensions': {'g': {}, 'b': {'dtype': 'int'}},
    }
    path = _written(tmp_path, spec)
    sources = {**_P_MAX, **_INDEX, 'gen_bus': _tidy(g=['w', 's'], b=[1, 99])}

    sentence = both_lanes_refuse(path, sources, match=r'not .b. labels')
    assert '99.' in sentence, 'the label as the caller wrote it, not as numpy holds it'


def test_a_multi_indexed_series_is_refused_on_both_lanes(tmp_path):
    """A MultiIndex restates the declared dims and can contradict them, so it is refused."""
    path = _written(tmp_path, RELATION_SPEC)
    deep = pd.MultiIndex.from_tuples([('w', 0), ('s', 0)], names=['g', 'k'])
    sources = {'p_max': pd.Series([5.0, 5.0], index=deep), **_INDEX, **_MAP}

    sentence = both_lanes_refuse(path, sources, match='MultiIndex is not a source')
    assert "['g', 'value']" in sentence, 'and it names the tidy frame the caller should pass'


def test_a_series_shallower_than_the_declared_dims_is_refused_on_both_lanes(tmp_path):
    """A Series is one level deep, so a declaration of two dims is refused as for a dict.

    The index is unnamed: a named one attaches by its own name and is reported
    as a missing column.
    """
    spec = {
        'dimensions': {'g': {'dtype': 'str'}, 'b': {'dtype': 'str'}},
        'parameters': {'p_max': {'dims': ['g', 'b']}},
        'variables': {'x': {'dims': ['g'], 'bounds': {'lower': 0, 'upper': 1}}},
        'objective': {'sense': 'maximize', 'expression': 'sum(x)'},
    }
    path = _written(tmp_path, spec)
    sources = {'p_max': pd.Series([5.0, 5.0], index=pd.Index(['w', 's']))}

    sentence = both_lanes_refuse(path, sources, match='runs along one dimension')
    assert "['g', 'b', 'value']" in sentence, 'and it names the table that carries both dims'


def test_a_source_key_the_model_does_not_declare_is_refused_on_both_lanes(tmp_path):
    """An undeclared source key is a typo, not something to ignore."""
    path = _written(tmp_path, SPEC)
    good = {'cost': _tidy(f=['a', 'b'], value=[1.0, 2.0]), 'cap': _tidy(f=['a', 'b'], value=[5.0, 5.0])}
    typo = {**good, 'csot': good['cost']}

    both_lanes_refuse(path, typo, match="Did you mean 'cost'")


def test_an_entity_table_is_a_dimension_index_columns_and_all(tmp_path):
    """An undeclared column is ignored where an undeclared key is not.

    A misspelled required column is a missing one. A column named after a
    relation over the dimension is refused, naming the key it belongs under.
    """
    spec = {
        'dimensions': {'g': {}, 'b': {'dtype': 'str'}},
        'relations': {'gen_bus': {'key': 'g', 'values': 'b'}},
        'parameters': {'cap': {'dims': ['g']}},
        'variables': {'x': {'dims': ['g'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
        'constraints': {'k': {'dims': ['b'], 'expression': 'sum(x, by=gen_bus, over=g, into=b) <= 100'}},
        'objective': {'sense': 'maximize', 'expression': 'sum(x)'},
    }
    path = _written(tmp_path, spec)
    generators = _tidy(g=['w', 's'], cap=[10.0, 20.0], note=['a', 'b'])
    sources = {
        'g': generators,
        'b': _tidy(b=['n', 'e']),
        'gen_bus': _tidy(g=['w', 's'], b=['n', 'e']),
        'cap': _tidy(g=['w', 's'], value=[10.0, 20.0]),
    }

    with sps.solve(path, sources) as result:
        assert result.objective == pytest.approx(30.0)

    with pytest.raises(DataError, match=r"missing columns \['g'\]"):
        sps.build(path, {**sources, 'cap': _tidy(gg=['w', 's'], value=[10.0, 20.0])}).close()

    carried = _tidy(g=['w', 's'], cap=[10.0, 20.0], gen_bus=['n', 'e'])
    with pytest.raises(DataError, match=r"index for dimension 'g' carries a 'gen_bus' column"):
        sps.build(path, {**sources, 'g': carried}).close()


@pytest.mark.parametrize(
    ('index', 'rewrite'),
    [
        pytest.param(['a', 'b', 'a', 'c', 'b'], 'list(dict.fromkeys(labels))', id='a-bare-sequence'),
        pytest.param(
            _tidy(f=['a', 'b', 'a', 'c', 'b'], colour=list('vwxyz')),
            "table.select('f').unique(maintain_order=True)",
            id='a-table-with-another-column',
        ),
    ],
)
def test_an_index_holding_a_label_twice_names_the_labels_and_the_fix(spec_path: Path, index: Any, rewrite: str):
    """Two rows give one label two positions, so `shift` would read one of them as the other.

    The rewrite is in the shape the index came in: a bare sequence was once told
    to call `.select` on a table it does not have.
    """
    sources = {
        'f': index,
        'cost': _tidy(f=['a', 'b', 'c'], value=[1.0, 2.0, 3.0]),
        'cap': _tidy(f=['a', 'b', 'c'], value=[5.0, 5.0, 5.0]),
    }
    with pytest.raises(DataError) as refused:
        sps.build(spec_path, sources).close()
    message = str(refused.value)
    assert "index for dimension 'f' holds 2 label(s) more than once: 'a', 'b'" in message, (
        'the dimension and the repeated labels, in the order they first repeat'
    )
    assert rewrite in message, 'and the rewrite that keeps the first of each, for the shape that was given'
    assert ('.select(' in message) == ('.select(' in rewrite), 'and no rewrite for a shape that was not given'


@pytest.mark.parametrize(
    'index',
    [
        pytest.param([None, 'a', 'c'], id='a-bare-sequence-leading-with-a-null'),
        pytest.param(_tidy(f=['a', None, 'c']), id='a-table-with-a-null-inside'),
    ],
)
def test_an_index_holding_a_null_label_is_refused(spec_path: Path, index: Any):
    """A null in an index is refused at load, before it can become a coordinate.

    It was read as a label: a string dimension failed with polars' own ``TypeError``
    on the null category, and an integer one built a coordinate labelled null.
    """
    sources = {
        'f': index,
        'cost': _tidy(f=['a', 'c'], value=[1.0, 3.0]),
        'cap': _tidy(f=['a', 'c'], value=[5.0, 5.0]),
    }
    with pytest.raises(DataError, match="index for dimension 'f' holds a null label"):
        sps.build(spec_path, sources).close()


#: A temporal dimension whose index bounds a variable.
TEMPORAL_BOUND_SPEC = {
    'dimensions': {'t': {'dtype': 'datetime'}},
    'parameters': {'cap': {'dims': ['t']}},
    'variables': {'x': {'dims': ['t'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
    'objective': {'sense': 'maximize', 'expression': 'sum(x)'},
}


def _instants(unit: str, time_zone: str | None = None, *, nanoseconds: int = 0) -> pl.Series:
    """Two instants a day apart, held in *unit*, the first *nanoseconds* past midnight."""
    import datetime

    midnight = pl.Series([datetime.datetime(2030, 1, 1), datetime.datetime(2030, 1, 2)]).cast(pl.Datetime('ns'))
    instants = midnight + pl.Series([nanoseconds, 0]).cast(pl.Duration('ns'))
    return instants.dt.replace_time_zone(time_zone).dt.cast_time_unit(unit)


@pytest.mark.parametrize(
    ('index', 'column', 'zone'),
    [
        pytest.param('ns', 'ns', None, id='nanoseconds-both'),
        pytest.param('ms', 'ms', None, id='milliseconds-both'),
        pytest.param('ns', 'us', None, id='nanosecond-index-microsecond-column'),
        pytest.param('us', 'ns', None, id='microsecond-index-nanosecond-column'),
        pytest.param('ns', 'ms', 'Europe/Berlin', id='zone-aware-both'),
    ],
)
def test_a_datetime_index_in_any_unit_bounds_a_variable_on_both_lanes(tmp_path, index, column, zone):
    """An index in nanoseconds, pandas' default, crashed the bound attach: the labels it met were microseconds."""
    sources = {
        't': pl.DataFrame({'t': _instants(index, zone)}),
        'cap': pl.DataFrame({'t': _instants(column, zone), 'value': [3.0, 4.0]}),
    }
    with differential(_written(tmp_path, TEMPORAL_BOUND_SPEC), sources) as run:
        assert run.oracle == pytest.approx(7.0), 'each instant takes its own cap, on both lanes'


def _temporal_sources(*, index: pl.Series, cap: pl.Series, day_of: pl.Series) -> dict[str, Any]:
    """The temporal relation model's sources, its three datetime columns given."""
    return {
        **_P_MAX,
        'cap': pl.DataFrame({'d': cap, 'value': [3.0, 7.0]}),
        'd': pl.DataFrame({'d': index}),
        'g': ['w', 's'],
        'day_of': pl.DataFrame({'g': ['w', 's'], 'd': day_of}),
    }


_FINE = _instants('ns', nanoseconds=1)
_EVEN = _instants('ns')
_UTC = _instants('us', 'UTC')
_DATES = _EVEN.cast(pl.Date)


@pytest.mark.parametrize(
    ('sources', 'owner', 'match'),
    [
        pytest.param(
            _temporal_sources(index=_FINE, cap=_FINE, day_of=_FINE),
            "index for dimension 'd'",
            'finer',
            id='index-finer',
        ),
        pytest.param(
            _temporal_sources(index=_EVEN, cap=_FINE, day_of=_EVEN), "parameter 'cap'", 'finer', id='parameter-finer'
        ),
        pytest.param(
            _temporal_sources(index=_EVEN, cap=_EVEN, day_of=_FINE), "relation 'day_of'", 'finer', id='relation-finer'
        ),
        pytest.param(
            _temporal_sources(index=_DATES, cap=_DATES, day_of=_instants('ns', nanoseconds=3_600_000_000_000)),
            "relation 'day_of'",
            "not 'd' labels",
            id='relation-past-midnight-of-a-date',
        ),
        pytest.param(
            _temporal_sources(index=_UTC, cap=_EVEN, day_of=_UTC),
            "parameter 'cap'",
            'convert_time_zone',
            id='parameter-zone',
        ),
        pytest.param(
            _temporal_sources(
                index=_instants('us', 'Europe/Berlin'), cap=_instants('us', 'Europe/Berlin'), day_of=_UTC
            ),
            "relation 'day_of'",
            'convert_time_zone',
            id='relation-other-zone',
        ),
    ],
)
def test_a_datetime_label_the_index_cannot_hold_is_refused_on_both_lanes(tmp_path, sources, owner, match):
    """A cast that drops a nanosecond, or a compare across two clocks, is refused rather than made."""
    sentence = both_lanes_refuse(_written(tmp_path, TEMPORAL_RELATION_SPEC), sources, match=match)
    assert sentence.startswith(owner), 'the refusal names what carried the label'
