"""An entry that holds no row is not queried, and the model is the one without it.

PyPSA's spec declares every component, and a network has rows for few of them:
most entries span a dimension without a label or a `where` that admits
nothing. Querying one costs every query its plan needs and builds nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

import specsolve as sps
from specsolve import _assumptions
from specsolve._engine import conditions, indices, specialize
from specsolve._engine import queries as queries_module
from specsolve._names import ROW

if TYPE_CHECKING:
    from pathlib import Path

#: `e` has no label: `y`, `over_nothing` and `sized` span it. `cap` never
#: exceeds 100, so `admits_nothing`'s `where` admits no row.
SPEC = {
    'dimensions': {'g': {}, 'e': {}},
    'parameters': {'cap': {'dims': ['g']}, 'size': {'dims': ['e']}},
    'variables': {
        'x': {'dims': ['g'], 'bounds': {'lower': 0, 'upper': 'cap'}},
        'y': {'dims': ['e'], 'bounds': {'lower': 0, 'upper': 'size'}},
    },
    'constraints': {
        'live': {'dims': ['g'], 'expression': 'x + sum(y, over=e) <= 1'},
        'over_nothing': {'dims': ['e'], 'expression': 'y <= size'},
        'admits_nothing': {'dims': ['g'], 'where': 'cap > 100', 'expression': 'x >= 1'},
    },
    'assumptions': {'sized': {'holds': 'size > 0'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(x)'},
}

SOURCES = {
    'g': ['a', 'b'],
    'e': [],
    'cap': pl.DataFrame({'g': ['a', 'b'], 'value': [2.0, 3.0]}),
    'size': pl.DataFrame(schema={'e': pl.String, 'value': pl.Float64}),
}


def _read(monkeypatch: pytest.MonkeyPatch) -> dict[str, set[str]]:
    """Which constraints were queried, which variables had their bounds queried, which dimensions were indexed by a query, and which an assumption was read over."""
    seen: dict[str, set] = {'constraint': set(), 'variable': set(), 'indexed': set(), 'assumption': set()}
    expression, bounds, table, narrowed = (
        queries_module.QueryBuilder.expression,
        queries_module.QueryBuilder.bounds,
        indices.table,
        _assumptions.narrowed,
    )

    def spied_expression(self, e, what, **kwargs):
        if what.startswith("constraint '"):
            seen['constraint'].add(what.split("'")[1])
        return expression(self, e, what, **kwargs)

    def spied_bounds(self, table, name, v):
        seen['variable'].add(name)
        return bounds(self, table, name, v)

    def spied_table(scope, dims, *args, **kwargs):
        seen['indexed'].add(tuple(dims))
        return table(scope, dims, *args, **kwargs)

    def spied_narrowed(scope, dims, where):
        seen['assumption'].update(dims)
        return narrowed(scope, dims, where)

    monkeypatch.setattr(queries_module.QueryBuilder, 'expression', spied_expression)
    monkeypatch.setattr(queries_module.QueryBuilder, 'bounds', spied_bounds)
    monkeypatch.setattr(indices, 'table', spied_table)
    monkeypatch.setattr(_assumptions, 'narrowed', spied_narrowed)
    sps.build(SPEC, SOURCES).close()
    return seen


@pytest.mark.parametrize(
    ('kind', 'name'),
    [
        pytest.param('constraint', 'over_nothing', id='a-constraint-over-a-dimension-without-a-label'),
        pytest.param('constraint', 'admits_nothing', id='a-constraint-whose-where-admits-no-row'),
        pytest.param('variable', 'y', id='a-variable-over-a-dimension-without-a-label'),
        pytest.param('indexed', ('e',), id='an-entry-over-a-dimension-without-a-label-is-indexed-without-a-query'),
        pytest.param('assumption', 'e', id='an-assumption-over-a-dimension-without-a-label'),
    ],
)
def test_an_entry_that_holds_no_row_is_not_queried(monkeypatch, kind, name):
    seen = _read(monkeypatch)
    assert name not in seen[kind], f'{name} holds no row, yet the build queried it'


def test_an_entry_that_holds_a_row_is_still_queried(monkeypatch):
    seen = _read(monkeypatch)
    assert seen['constraint'] == {'live'}, 'only `live` holds a row, and it has to be queried'
    assert seen['variable'] == {'x'}, 'only `x` holds a column, and its bounds have to be queried'


def test_a_model_with_entries_that_hold_no_row_is_the_model_without_them(tmp_path: Path):
    without = {
        **SPEC,
        'constraints': {'live': SPEC['constraints']['live']},
        'assumptions': {},
    }
    sps.write(SPEC, SOURCES, tmp_path / 'with.lp')
    sps.write(without, SOURCES, tmp_path / 'without.lp')
    assert (tmp_path / 'with.lp').read_text() == (tmp_path / 'without.lp').read_text(), (
        'an entry without a row changed the written model'
    )
    assert sps.solve(SPEC, SOURCES).objective == pytest.approx(2.0), 'each x is capped at 1 by `live` alone'


#: `limit` has a row at one of the four coordinates of `g` and `h`, so `capped` holds
#: one row and has no `where` to admit it early.
SPARSE_SPEC = {
    'dimensions': {'g': {}, 'h': {}},
    'parameters': {'limit': {'dims': ['g', 'h'], 'missing': 'absent'}},
    'variables': {'z': {'dims': ['g', 'h'], 'bounds': {'lower': 0}}},
    'constraints': {'capped': {'dims': ['g', 'h'], 'expression': 'z <= limit'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(z)'},
}

SPARSE_SOURCES = {
    'g': ['a', 'b'],
    'h': ['p', 'q'],
    'limit': pl.DataFrame({'g': ['a'], 'h': ['p'], 'value': [4.0]}),
}


def test_a_constraint_without_a_where_is_indexed_only_inside_its_presences(monkeypatch):
    """A sparse absence narrows such a constraint, and indexing it before its presences would hold the full product of its dimensions."""
    table, unnarrowed = indices.table, []

    def spied_table(scope, dims, where, index, start, presences=()):
        if index == ROW and where is None and not presences:
            unnarrowed.append(tuple(dims))
        return table(scope, dims, where, index, start, presences)

    monkeypatch.setattr(indices, 'table', spied_table)
    with sps.build(SPARSE_SPEC, SPARSE_SOURCES) as model:
        assert model.metrics().rows == 1, '`limit` has one row, so `capped` keeps one'
    assert not unnarrowed, (
        f'`capped` was indexed over its full product {unnarrowed} before the presences that narrow it'
    )


#: `e` has no label, so `count(on, over=e)` is 0 at every `g`.
COUNTED_SPEC = {
    'dimensions': {'g': {}, 'e': {}},
    'parameters': {'on': {'dims': ['g', 'e'], 'dtype': 'bool'}},
    'variables': {'x': {'dims': ['g'], 'bounds': {'lower': 0, 'upper': 1}}},
    'objective': {'sense': 'maximize', 'expression': 'sum(x)'},
}

COUNTED_SOURCES = {
    'g': ['a', 'b', 'c'],
    'e': [],
    'on': pl.DataFrame(schema={'g': pl.String, 'e': pl.String, 'value': pl.Boolean}),
}


@pytest.mark.parametrize(
    ('where', 'columns'),
    [
        pytest.param('count(on, over=e) > 0', 0, id='more-than-none-admits-no-row'),
        pytest.param('count(on, over=e) == 0', 3, id='exactly-none-admits-every-row'),
        pytest.param('count(on, over=e) < 1', 3, id='fewer-than-one-admits-every-row'),
    ],
)
def test_a_count_over_a_dimension_without_a_label_is_zero_without_a_query(monkeypatch, where, columns):
    """`breakpoint` is empty for most of PyPSA's components, and each curve's assumptions count along it."""
    counted = []
    real = conditions._counted

    def spied(scope, p, alias):
        counted.append(p.over)
        return real(scope, p, alias)

    monkeypatch.setattr(conditions, '_counted', spied)
    spec = {**COUNTED_SPEC, 'variables': {'x': {**COUNTED_SPEC['variables']['x'], 'where': where}}}
    with sps.build(spec, COUNTED_SOURCES) as model:
        assert model.metrics().columns == columns, f'a count over nothing is 0, so `{where}` holds at {columns} of 3'
    assert not counted, f'counted over {counted}, a dimension without a label, with a query'


#: `e` with labels and `size` over them, so `y`, `over_nothing` and `sized` hold rows.
GROWN = {'e': ['p', 'q'], 'size': pl.DataFrame({'e': ['p', 'q'], 'value': [0.5, 0.25]})}


@pytest.mark.parametrize(
    ('first', 'then'),
    [
        pytest.param(SOURCES, SOURCES | GROWN, id='a-dimension-gains-labels'),
        pytest.param(SOURCES | GROWN, SOURCES, id='a-dimension-loses-its-labels'),
        pytest.param(
            SOURCES, SOURCES | {'cap': pl.DataFrame({'g': ['a', 'b'], 'value': [200.0, 3.0]})}, id='a-where-gains-a-row'
        ),
    ],
)
def test_an_update_across_what_holds_no_row_is_the_model_a_fresh_build_gives(tmp_path: Path, first, then):
    """A skip is decided on the data a build reads, so `update` decides it again.

    `cap > 100` admits no row of `admits_nothing` until the update moves `cap`'s
    numbers alone: no dimension changes, so it is the case a splice could keep.
    """
    with sps.build(SPEC, first) as model:
        model.update(then)
        model.write(tmp_path / 'updated.lp')
    sps.write(SPEC, then, tmp_path / 'fresh.lp')
    assert (tmp_path / 'updated.lp').read_text() == (tmp_path / 'fresh.lp').read_text(), (
        'the update kept a skip that its new data undoes'
    )


#: `flag` has no row and `cap` none above 3, so a `where` that needs either admits
#: nothing; each is over `g` alone, a sixth of the product an entry over
#: `g` and `t` spans. `e` has no label, so a count along it is 0.
TABLED_SPEC = {
    'dimensions': {'g': {}, 't': {'dtype': 'int'}, 'e': {}},
    'parameters': {
        'flag': {'dims': ['g'], 'dtype': 'bool', 'missing': 'absent'},
        'cap': {'dims': ['g']},
        'on': {'dims': ['g', 'e'], 'dtype': 'bool', 'missing': 'absent'},
    },
    'conditions': {'flagged': {'condition': 'flag'}},
    'variables': {'x': {'dims': ['g', 't'], 'bounds': {'lower': 0, 'upper': 1}}},
    'objective': {'sense': 'maximize', 'expression': 'sum(x)'},
}

TABLED_SOURCES = {
    'g': ['a', 'b'],
    't': [1, 2, 3],
    'flag': pl.DataFrame(schema={'g': pl.String, 'value': pl.Boolean}),
    'cap': pl.DataFrame({'g': ['a', 'b'], 'value': [2.0, 3.0]}),
    'e': [],
    'on': pl.DataFrame(schema={'g': pl.String, 'e': pl.String, 'value': pl.Boolean}),
}


def _indexed_wheres(monkeypatch: pytest.MonkeyPatch, spec: dict) -> tuple[int, int, list]:
    """The model's columns and rows, and every `where` the build indexed over the product of its entry's dimensions."""
    table, indexed = indices.table, []

    def spied_table(scope, dims, where, *args, **kwargs):
        if where is not None:
            indexed.append(where)
        return table(scope, dims, where, *args, **kwargs)

    monkeypatch.setattr(indices, 'table', spied_table)
    with sps.build(spec, TABLED_SOURCES) as model:
        metrics = model.metrics()
        return metrics.columns, metrics.rows, indexed


@pytest.mark.parametrize(
    ('where', 'rows'),
    [
        pytest.param('flag', 0, id='a-parameter-without-a-row'),
        pytest.param('cap > 3', 0, id='a-comparison-no-row-passes'),
        pytest.param('flagged', 0, id='a-named-condition-read-as-its-body'),
        pytest.param('cap > 1 AND flag', 0, id='one-condition-of-an-and'),
        pytest.param('flag OR cap > 3', 0, id='both-sides-of-an-or'),
        pytest.param('count(on, over=e) > 0', 0, id='a-count-along-a-dimension-without-a-label'),
        pytest.param('NOT flag', 6, id='a-negation-of-a-parameter-without-a-row-admits-every-row'),
        pytest.param('flag OR cap > 2', 3, id='an-or-one-side-of-which-admits-rows'),
        pytest.param('count(on, over=e) == 0', 6, id='a-count-along-a-dimension-without-a-label-that-holds'),
        pytest.param('cap > 2', 3, id='a-comparison-one-label-passes-admits-its-rows'),
    ],
)
def test_a_where_that_admits_nothing_is_decided_without_its_product(monkeypatch, where, rows):
    """PyPSA's spec gates most unit-by-snapshot entries on a per-unit flag no unit of a network sets, or on a curve no unit has.

    The flag's table, or the empty dimension the curve counts along, answers
    whether the entry holds a row; indexing the product of its dimensions to
    find none read the whole product.
    """
    spec = {**TABLED_SPEC, 'constraints': {'probe': {'dims': ['g', 't'], 'where': where, 'expression': 'x >= 0'}}}
    _, built, indexed = _indexed_wheres(monkeypatch, spec)
    assert built == rows, f'`{where}` admits {rows} of the 6 rows of `probe`'
    if not rows:
        assert not indexed, f'`{where}` admits no row, yet `probe` was indexed over its product'


def test_a_variable_whose_where_one_parameter_empties_is_not_indexed(monkeypatch):
    variables = {**TABLED_SPEC['variables'], 'gated': {'dims': ['g', 't'], 'where': 'flag', 'bounds': {'lower': 0}}}
    columns, _, indexed = _indexed_wheres(monkeypatch, {**TABLED_SPEC, 'variables': variables})
    assert columns == 6, '`gated` holds no column, so the model has the 6 of `x` alone'
    assert not indexed, '`flag` has no row, yet `gated` was indexed over its product'


@pytest.mark.parametrize(
    ('where', 'refused'),
    [
        pytest.param('flag', False, id='a-where-that-admits-nothing-is-not-read'),
        pytest.param('cap > 2', True, id='a-where-that-admits-a-unit-is-still-checked-there'),
    ],
)
def test_an_assumption_is_read_only_where_its_where_admits_a_coordinate(monkeypatch, where, refused):
    """`cap > 10` fails at every unit, so the data is refused exactly where the `where` admits one."""
    spec = {**TABLED_SPEC, 'assumptions': {'large': {'holds': 'cap > 10', 'where': where}}}
    real, read = _assumptions.narrowed, []

    def spied(scope, dims, where):
        read.append(tuple(dims))
        return real(scope, dims, where)

    monkeypatch.setattr(_assumptions, 'narrowed', spied)
    if refused:
        with pytest.raises(sps.errors.DataError, match='large'):
            sps.build(spec, TABLED_SOURCES)
        return
    sps.build(spec, TABLED_SOURCES).close()
    assert not read, f'`{where}` admits no unit, yet `large` was read over {read}'


def test_a_build_with_assumptions_attaches_its_sources_once(monkeypatch):
    """The assumptions are read against an attach, and the build reuses it rather than attaching the same sources again."""
    from specsolve import _api

    calls = []
    real = _api.attach

    def counted(program, sources):
        calls.append(1)
        return real(program, sources)

    monkeypatch.setattr(_api, 'attach', counted)
    monkeypatch.setattr(_assumptions, 'attach', counted)
    spec = {**TABLED_SPEC, 'assumptions': {'capped': {'holds': 'cap > 0'}}}
    sps.build(spec, TABLED_SOURCES).close()
    assert len(calls) == 1, f'the sources were attached {len(calls)} times in one build'


def test_an_atom_the_assumptions_and_the_build_share_is_read_once(monkeypatch):
    """The assumptions and the entries specialize their `where` to one read of each atom, made at attach.

    Each read its own before, so a model gated on `cap > 2` in both read `cap`'s table twice.
    """
    reads, real = [], specialize.passing

    def counted(scope, atom):
        reads.append(atom)
        return real(scope, atom)

    monkeypatch.setattr(specialize, 'passing', counted)
    spec = {
        **TABLED_SPEC,
        'constraints': {'probe': {'dims': ['g', 't'], 'where': 'cap > 2', 'expression': 'x >= 0'}},
        'assumptions': {'positive': {'holds': 'cap > 0', 'where': 'cap > 2'}},
    }
    sps.build(spec, TABLED_SOURCES).close()
    assert len(reads) == 1, f'`cap > 2` was read {len(reads)} times for one build'


def test_a_build_with_assumptions_reports_the_time_its_attach_took():
    """The assumptions attach the sources before the engine runs, and the engine timed only an attach of its own.

    A model with an assumption so reported no attach time at all, and an
    archive of it recorded zero.
    """
    spec = {**TABLED_SPEC, 'assumptions': {'capped': {'holds': 'cap > 0'}}}
    with sps.build(spec, TABLED_SOURCES) as model:
        attach = model.metrics().attach_seconds
    assert attach > 0, 'the one attach of a build with assumptions is timed, whoever asked for it'


#: `active` is true at every `(g, t)`, as PyPSA's is on a network without
#: investment periods; `level` is given at `(a, 1)` alone and filled with 0.
EVERYWHERE_SOURCES = TABLED_SOURCES | {
    'active': pl.DataFrame({'g': ['a', 'a', 'a', 'b', 'b', 'b'], 't': [1, 2, 3] * 2, 'value': [True] * 6}),
    'level': pl.DataFrame({'g': ['a'], 't': [1], 'value': [5.0]}),
}
EVERYWHERE_PARAMETERS = {
    **TABLED_SPEC['parameters'],
    'active': {'dims': ['g', 't'], 'dtype': 'bool'},
    'level': {'dims': ['g', 't'], 'missing': 0.0},
}


@pytest.mark.parametrize(
    ('where', 'active', 'rows', 'read'),
    [
        pytest.param('active', [True] * 6, 6, [], id='a-flag-true-everywhere-is-dropped'),
        pytest.param('active', [True] * 5 + [False], 5, [('g', 't')], id='a-flag-false-somewhere-is-kept'),
        pytest.param('level', [True] * 6, 1, [('g', 't')], id='a-filled-parameter-holds-only-where-given'),
        pytest.param('cap > 2 AND active', [True] * 6, 3, [('g',)], id='the-rest-of-an-and-is-kept'),
    ],
)
def test_a_condition_that_holds_everywhere_is_dropped_from_its_where(monkeypatch, where, active, rows, read):
    """PyPSA gates nearly every entry on `active`, which holds everywhere without investment periods.

    Kept, it reads every entry at unit-by-snapshot and keeps it off the
    path of an entry without a `where`.
    """
    spec = {
        **TABLED_SPEC,
        'parameters': EVERYWHERE_PARAMETERS,
        'constraints': {'probe': {'dims': ['g', 't'], 'where': where, 'expression': 'x >= 0'}},
    }
    sources = EVERYWHERE_SOURCES | {'active': EVERYWHERE_SOURCES['active'].with_columns(value=pl.Series(active))}
    table, indexed = indices.table, []

    def spied_table(scope, dims, where, *args, **kwargs):
        if where is not None:
            indexed.append(tuple(sorted(where.dims)))
        return table(scope, dims, where, *args, **kwargs)

    monkeypatch.setattr(indices, 'table', spied_table)
    with sps.build(spec, sources) as model:
        assert model.metrics().rows == rows, f'`{where}` admits {rows} of the 6 rows of `probe`'
    assert indexed == read, f'`probe` was indexed over {indexed}, where what is left of `{where}` reads {read}'


@pytest.mark.parametrize(
    ('where', 'joins'),
    [
        pytest.param('cap > 2', 0, id='a-comparison'),
        pytest.param('active', 0, id='a-flag'),
        pytest.param('filled', 1, id='a-filled-parameter-joins-the-rows-it-was-given'),
    ],
)
def test_an_atom_is_counted_over_its_parameter_without_joining_it_onto_itself(monkeypatch, where, joins):
    """Attach counts each atom of a `where` over its parameter's own table.

    The count joined the parameter back onto that table's keys: 14 ms of the
    24 ms a half mask's `attach` spends counting at 1.2M columns (#2132).
    """
    plans, real = [], specialize.passing

    def planned(scope, atom):
        query = real(scope, atom)
        plans.append(query.explain(optimized=False).count('JOIN:'))
        return query

    monkeypatch.setattr(specialize, 'passing', planned)
    spec = {
        **TABLED_SPEC,
        'parameters': {**EVERYWHERE_PARAMETERS, 'filled': {'dims': ['g', 't'], 'missing': 3.0}},
        'constraints': {'probe': {'dims': ['g', 't'], 'where': where, 'expression': 'x >= 0'}},
    }
    sps.build(spec, EVERYWHERE_SOURCES | {'filled': EVERYWHERE_SOURCES['level']}).close()
    assert plans == [joins], f'counting `{where}` took {plans} joins, where it needs {joins}'


def test_a_where_that_holds_everywhere_builds_the_model_without_it(tmp_path: Path):
    spec = {**TABLED_SPEC, 'parameters': EVERYWHERE_PARAMETERS}
    gated = {**spec, 'constraints': {'probe': {'dims': ['g', 't'], 'where': 'active', 'expression': 'x <= 0.5'}}}
    plain = {**spec, 'constraints': {'probe': {'dims': ['g', 't'], 'expression': 'x <= 0.5'}}}
    sps.write(gated, EVERYWHERE_SOURCES, tmp_path / 'gated.lp')
    sps.write(plain, EVERYWHERE_SOURCES, tmp_path / 'plain.lp')
    assert (tmp_path / 'gated.lp').read_text() == (tmp_path / 'plain.lp').read_text(), (
        'a where that holds everywhere changed the written model'
    )


@pytest.mark.parametrize(
    ('where', 'rows'),
    [
        pytest.param('unset != 1', 6, id='a-parameter-without-a-row-whose-missing-value-passes'),
        pytest.param('level < 1', 5, id='a-parameter-whose-rows-all-fail-and-whose-missing-value-passes'),
    ],
)
def test_a_where_holds_where_its_parameter_has_no_row_and_the_missing_value_passes(where, rows):
    """Fuzz seeds 441 and 884: `p != 1` under `missing: 0`, with no row of `p`, admits every coordinate.

    The data settled such a `where` false: a parameter with no row, or one
    whose rows all failed, was read as failing at every coordinate, though a
    coordinate without a row reads the missing value, and here that passes.
    """
    spec = {
        **TABLED_SPEC,
        'parameters': {**EVERYWHERE_PARAMETERS, 'unset': {'dims': ['g', 't'], 'missing': 0.0}},
        'constraints': {'probe': {'dims': ['g', 't'], 'where': where, 'expression': 'x >= 0'}},
    }
    sources = EVERYWHERE_SOURCES | {'unset': pl.DataFrame(schema={'g': pl.String, 't': pl.Int64, 'value': pl.Float64})}
    with sps.build(spec, sources) as model:
        assert model.metrics().rows == rows, f'`{where}` admits {rows} of the 6 rows of `probe`'
