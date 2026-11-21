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


def test_a_constraint_with_a_where_and_restrictions_reads_its_where_once(monkeypatch):
    """The `where` labels the rows, and the restrictions only cut them: reading it again for the cut cost the whole read twice."""
    spec = {
        **SPARSE_SPEC,
        'parameters': {**SPARSE_SPEC['parameters'], 'on': {'dims': ['g'], 'dtype': 'bool'}},
        'constraints': {'capped': {**SPARSE_SPEC['constraints']['capped'], 'where': 'on'}},
    }
    sources = SPARSE_SOURCES | {'on': pl.DataFrame({'g': ['a', 'b'], 'value': [True, False]})}
    frame, read = labels.frame, []

    def spied_frame(scope, dims, where, *args, **kwargs):
        if where is not None:
            read.append(tuple(dims))
        return frame(scope, dims, where, *args, **kwargs)

    monkeypatch.setattr(labels, 'frame', spied_frame)
    with sps.build(spec, sources) as model:
        assert model.diagnostics().rows == 1, '`limit` has one row, so `capped` keeps one'
    assert read == [('g', 'h')], f'`capped` read its `where` {len(read)} times, where once labels it'


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


#: `flag` has no row and `cap` none above 3, so a `where` that needs either admits
#: nothing; each is over `g` alone, a sixth of the product a declaration over
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


def _labelled_wheres(monkeypatch: pytest.MonkeyPatch, spec: dict) -> tuple[int, int, list]:
    """The model's columns and rows, and every `where` the build labelled over the product of its declaration's dims."""
    frame, labelled = labels.frame, []

    def spied_frame(scope, dims, where, *args, **kwargs):
        if where is not None:
            labelled.append(where)
        return frame(scope, dims, where, *args, **kwargs)

    monkeypatch.setattr(labels, 'frame', spied_frame)
    with sps.build(spec, TABLED_SOURCES) as model:
        diagnostics = model.diagnostics()
        return diagnostics.columns, diagnostics.rows, labelled


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
    """PyPSA's spec gates most unit-by-snapshot declarations on a per-unit flag no unit of a network sets, or on a curve no unit has.

    The flag's table, or the empty dimension the curve counts along, answers
    whether the declaration holds a row; labelling the product of its dims to
    find none read the whole product.
    """
    spec = {**TABLED_SPEC, 'constraints': {'probe': {'dims': ['g', 't'], 'where': where, 'expression': 'x >= 0'}}}
    _, built, labelled = _labelled_wheres(monkeypatch, spec)
    assert built == rows, f'`{where}` admits {rows} of the 6 rows of `probe`'
    if not rows:
        assert not labelled, f'`{where}` admits no row, yet `probe` was labelled over its product'


def test_a_variable_whose_where_one_parameter_empties_is_not_labelled(monkeypatch):
    variables = {**TABLED_SPEC['variables'], 'gated': {'dims': ['g', 't'], 'where': 'flag', 'bounds': {'lower': 0}}}
    columns, _, labelled = _labelled_wheres(monkeypatch, {**TABLED_SPEC, 'variables': variables})
    assert columns == 6, '`gated` holds no column, so the model has the 6 of `x` alone'
    assert not labelled, '`flag` has no row, yet `gated` was labelled over its product'


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
    real, read = _assumptions.masked, []

    def spied(scope, dims, mask):
        read.append(tuple(dims))
        return real(scope, dims, mask)

    monkeypatch.setattr(_assumptions, 'masked', spied)
    if refused:
        with pytest.raises(sps.errors.DataError, match='large'):
            sps.build(spec, TABLED_SOURCES)
        return
    sps.build(spec, TABLED_SOURCES).close()
    assert not read, f'`{where}` admits no unit, yet `large` was read over {read}'


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


def test_a_test_the_assumptions_and_the_build_share_is_read_once(monkeypatch):
    """The assumptions and the declarations fold their `where` over one read of each test, made at attach.

    Each read its own before, so a model gated on `cap > 2` in both read `cap`'s table twice.
    """
    reads, real = [], predicates._passing

    def counted(scope, atom):
        reads.append(atom)
        return real(scope, atom)

    monkeypatch.setattr(predicates, '_passing', counted)
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
        seconds = model.diagnostics().seconds
    assert seconds.get('attach', 0.0) > 0, 'the one attach of a build with assumptions is timed, whoever asked for it'


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
    """PyPSA gates nearly every declaration on `active`, which holds everywhere without investment periods.

    Kept, it reads every declaration at unit-by-snapshot and keeps it off the
    path of a declaration without a `where`.
    """
    spec = {
        **TABLED_SPEC,
        'parameters': EVERYWHERE_PARAMETERS,
        'constraints': {'probe': {'dims': ['g', 't'], 'where': where, 'expression': 'x >= 0'}},
    }
    sources = EVERYWHERE_SOURCES | {'active': EVERYWHERE_SOURCES['active'].with_columns(value=pl.Series(active))}
    frame, labelled = labels.frame, []

    def spied_frame(scope, dims, mask, *args, **kwargs):
        if mask is not None:
            labelled.append(tuple(sorted(mask.dims)))
        return frame(scope, dims, mask, *args, **kwargs)

    monkeypatch.setattr(labels, 'frame', spied_frame)
    with sps.build(spec, sources) as model:
        assert model.diagnostics().rows == rows, f'`{where}` admits {rows} of the 6 rows of `probe`'
    assert labelled == read, f'`probe` was labelled over {labelled}, where what is left of `{where}` reads {read}'


def test_a_where_that_holds_everywhere_builds_the_model_without_it(tmp_path: Path):
    spec = {**TABLED_SPEC, 'parameters': EVERYWHERE_PARAMETERS}
    gated = {**spec, 'constraints': {'probe': {'dims': ['g', 't'], 'where': 'active', 'expression': 'x <= 0.5'}}}
    plain = {**spec, 'constraints': {'probe': {'dims': ['g', 't'], 'expression': 'x <= 0.5'}}}
    sps.write(gated, EVERYWHERE_SOURCES, tmp_path / 'gated.lp')
    sps.write(plain, EVERYWHERE_SOURCES, tmp_path / 'plain.lp')
    assert (tmp_path / 'gated.lp').read_text() == (tmp_path / 'plain.lp').read_text(), (
        'a where that holds everywhere changed the written model'
    )


#: `square` is zero at every unit, as PyPSA's quadratic costs are on most networks.
ZERO_SPEC = {
    'dimensions': {'g': {}},
    'parameters': {'square': {'dims': ['g']}, 'cost': {'dims': ['g']}},
    'variables': {'x': {'dims': ['g'], 'bounds': {'lower': 0, 'upper': 1}}},
    'objective': {'sense': 'minimize', 'expression': 'sum(x * x * square) - sum(x * cost)'},
}


@pytest.mark.parametrize(
    ('square', 'quadratic', 'objective'),
    [
        pytest.param([0.0, 0.0], 0, -2.0, id='a-cost-zero-everywhere-builds-no-term'),
        pytest.param([0.0, 2.0], 1, -1.125, id='a-cost-nonzero-somewhere-builds-its-term'),
    ],
)
def test_a_variable_times_a_parameter_zero_everywhere_adds_nothing_to_the_objective(
    monkeypatch, square, quadratic, objective
):
    """PyPSA prices the square of every unit's dispatch, and most networks price it at zero."""
    from specsolve._engine import assembly as assembly_module

    compiled, real = [], assembly_module.Assembly._objective_quadratic

    def spied(self, terms, expression):
        compiled.append(len(terms))
        return real(self, terms, expression)

    monkeypatch.setattr(assembly_module.Assembly, '_objective_quadratic', spied)
    sources = {
        'g': ['a', 'b'],
        'square': pl.DataFrame({'g': ['a', 'b'], 'value': square}),
        'cost': pl.DataFrame({'g': ['a', 'b'], 'value': [1.0, 1.0]}),
    }
    with sps.build(ZERO_SPEC, sources) as model:
        assert model.solve().objective == pytest.approx(objective), 'b settles at x = 0.25 where its square is priced'
    assert compiled == [quadratic], f'the objective compiled {compiled} quadratic terms, where {quadratic} are needed'


def test_a_constraint_times_a_parameter_zero_everywhere_keeps_its_rows():
    """A zero coefficient keeps its row in a constraint: only the objective, which only sums, drops it."""
    spec = {**ZERO_SPEC, 'constraints': {'scaled': {'dims': ['g'], 'expression': 'x * square <= 1'}}}
    sources = {
        'g': ['a', 'b'],
        'square': pl.DataFrame({'g': ['a', 'b'], 'value': [0.0, 0.0]}),
        'cost': pl.DataFrame({'g': ['a', 'b'], 'value': [1.0, 1.0]}),
    }
    with sps.build(spec, sources) as model:
        assert model.diagnostics().rows == 2, 'both rows of `scaled` are built, their coefficient zero'
