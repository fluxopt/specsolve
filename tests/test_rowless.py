"""A declaration that holds no row is not compiled, and the model is the one without it.

PyPSA's spec declares every component, and a network has rows for few of them:
most declarations span a dimension without a label or a `where` that admits
nothing. Compiling one costs every query its plan needs and builds nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

import specsolve as sps
from specsolve import _assumptions
from specsolve._engine import compiler as compiler_module
from specsolve._engine import labels, predicates

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
    """Which constraints were compiled, which variables had their bounds compiled, which dims were labelled by a query, and which an assumption was read over."""
    seen: dict[str, set] = {'constraint': set(), 'variable': set(), 'labelled': set(), 'assumption': set()}
    expression, bounds, frame, masked = (
        compiler_module.Compiler.expression,
        compiler_module.Compiler.bounds,
        labels.frame,
        _assumptions.masked,
    )

    def spied_expression(self, e, what, **kwargs):
        if what.startswith("constraint '"):
            seen['constraint'].add(what.split("'")[1])
        return expression(self, e, what, **kwargs)

    def spied_bounds(self, frame, name, v):
        seen['variable'].add(name)
        return bounds(self, frame, name, v)

    def spied_frame(scope, dims, *args, **kwargs):
        seen['labelled'].add(tuple(dims))
        return frame(scope, dims, *args, **kwargs)

    def spied_masked(scope, dims, where):
        seen['assumption'].update(dims)
        return masked(scope, dims, where)

    monkeypatch.setattr(compiler_module.Compiler, 'expression', spied_expression)
    monkeypatch.setattr(compiler_module.Compiler, 'bounds', spied_bounds)
    monkeypatch.setattr(labels, 'frame', spied_frame)
    monkeypatch.setattr(_assumptions, 'masked', spied_masked)
    sps.build(SPEC, SOURCES).close()
    return seen


@pytest.mark.parametrize(
    ('kind', 'name'),
    [
        pytest.param('constraint', 'over_nothing', id='a-constraint-over-a-dimension-without-a-label'),
        pytest.param('constraint', 'admits_nothing', id='a-constraint-whose-where-admits-no-row'),
        pytest.param('variable', 'y', id='a-variable-over-a-dimension-without-a-label'),
        pytest.param(
            'labelled', ('e',), id='a-declaration-over-a-dimension-without-a-label-is-labelled-without-a-query'
        ),
        pytest.param('assumption', 'e', id='an-assumption-over-a-dimension-without-a-label'),
    ],
)
def test_a_declaration_that_holds_no_row_is_not_compiled(monkeypatch, kind, name):
    seen = _read(monkeypatch)
    assert name not in seen[kind], f'{name} holds no row, yet the build compiled it'


def test_a_declaration_that_holds_a_row_is_still_compiled(monkeypatch):
    seen = _read(monkeypatch)
    assert seen['constraint'] == {'live'}, 'only `live` holds a row, and it has to be compiled'
    assert seen['variable'] == {'x'}, 'only `x` holds a column, and its bounds have to be compiled'


def test_a_model_with_declarations_that_hold_no_row_is_the_model_without_them(tmp_path: Path):
    without = {
        **SPEC,
        'constraints': {'live': SPEC['constraints']['live']},
        'assumptions': {},
    }
    sps.write(SPEC, SOURCES, tmp_path / 'with.lp')
    sps.write(without, SOURCES, tmp_path / 'without.lp')
    assert (tmp_path / 'with.lp').read_text() == (tmp_path / 'without.lp').read_text(), (
        'a declaration without a row changed the written model'
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


def test_a_constraint_without_a_where_is_labelled_only_inside_its_restrictions(monkeypatch):
    """A sparse absence narrows such a constraint, and labelling it unrestricted first would hold the full product of its dims."""
    frame, unrestricted = labels.frame, []

    def spied_frame(scope, dims, where, label, start, restrictions=()):
        if label == 'row' and where is None and not restrictions:
            unrestricted.append(tuple(dims))
        return frame(scope, dims, where, label, start, restrictions)

    monkeypatch.setattr(labels, 'frame', spied_frame)
    with sps.build(SPARSE_SPEC, SPARSE_SOURCES) as model:
        assert model.diagnostics().rows == 1, '`limit` has one row, so `capped` keeps one'
    assert not unrestricted, f'`capped` was labelled over its full product {unrestricted} before its restriction'


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
    real = predicates._counted

    def spied(scope, p, alias):
        counted.append(p.over)
        return real(scope, p, alias)

    monkeypatch.setattr(predicates, '_counted', spied)
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
