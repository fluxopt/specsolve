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
from specsolve._engine import labels

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
