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
from specsolve._engine import conditions, indices
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


def test_a_constraint_without_a_where_is_indexed_only_inside_its_restrictions(monkeypatch):
    """A sparse absence narrows such a constraint, and indexing it unrestricted first would hold the full product of its dimensions."""
    table, unrestricted = indices.table, []

    def spied_table(scope, dims, where, index, start, restrictions=()):
        if index == ROW and where is None and not restrictions:
            unrestricted.append(tuple(dims))
        return table(scope, dims, where, index, start, restrictions)

    monkeypatch.setattr(indices, 'table', spied_table)
    with sps.build(SPARSE_SPEC, SPARSE_SOURCES) as model:
        assert model.diagnostics().rows == 1, '`limit` has one row, so `capped` keeps one'
    assert not unrestricted, f'`capped` was indexed over its full product {unrestricted} before its restriction'


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
        assert model.diagnostics().columns == columns, (
            f'a count over nothing is 0, so `{where}` holds at {columns} of 3'
        )
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
    numbers alone: no dimension changes, so it is the case a refill could keep.
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
    'dimensions': {'g': {}, 't': {}, 'e': {}},
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


def test_a_build_with_assumptions_attaches_its_sources_once(monkeypatch):
    """The assumptions are read against an attach, and the build reuses it rather than attaching the same sources again."""
    from specsolve._engine import engine as engine_module

    calls = []
    real = engine_module.attach

    def counted(program, sources):
        calls.append(1)
        return real(program, sources)

    monkeypatch.setattr(engine_module, 'attach', counted)
    monkeypatch.setattr(_assumptions, 'attach', counted)
    spec = {**TABLED_SPEC, 'assumptions': {'capped': {'holds': 'cap > 0'}}}
    sps.build(spec, TABLED_SOURCES).close()
    assert len(calls) == 1, f'the sources were attached {len(calls)} times in one build'


def test_a_build_with_assumptions_reports_the_time_its_attach_took():
    """The assumptions attach the sources before the engine runs, and the engine timed only an attach of its own.

    A model with an assumption so reported no attach time at all, and an
    archive of it recorded zero.
    """
    spec = {**TABLED_SPEC, 'assumptions': {'capped': {'holds': 'cap > 0'}}}
    with sps.build(spec, TABLED_SOURCES) as model:
        seconds = model.diagnostics().seconds
    assert seconds.get('attach', 0.0) > 0, 'the one attach of a build with assumptions is timed, whoever asked for it'
