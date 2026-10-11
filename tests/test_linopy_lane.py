"""The linopy lane: its verbs, its loader, its where evaluator, its notes.

Everything here needs the oracle's linopy. The lane holds no state: one call,
one model, nothing retained, and nothing put on ``linopy.Model``.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from mathspec import program

from specsolve.errors import DataError, LanguageError, SpecsolveError
from specsolve.sources import tidy_sources
from tests.conftest import EXAMPLES_DIR, expanded, schema_of
from tests.differential import differential
from tests.oracle import builder, linopy, loader, pd, specsolve_linopy, where, xr
from tests.piecewise_models import curve_frame

if TYPE_CHECKING:
    from mathspec import Spec


@pytest.fixture
def yaml_file(tmp_path):
    """Write YAML text to a file — the only shape the linopy lane accepts."""

    def write(text: str, name: str = 'm.yaml'):
        path = tmp_path / name
        path.write_text(textwrap.dedent(text).lstrip())
        return path

    return write


# ---------------------------------------------------------------------------
# the lane is a pure producer
# ---------------------------------------------------------------------------


def test_a_side_the_program_leaves_open_builds_as_linopy_s_open_bound():
    """A program says an open side is `None`, and linopy takes an infinity there."""
    declared = program.Program(
        parameters={},
        variables={'p': program.VariableDeclaration(('snapshot',), lower=None, upper=None)},
        constraints={},
        objective=None,
        dimensions={'snapshot': program.DimensionDeclaration()},
    )
    model = linopy.Model()
    builder.build_model(model, declared, xr.Dataset(), {'snapshot': pd.Index([0, 1], name='snapshot')}, {})
    p = model.variables['p']
    assert (float(p.lower.min()), float(p.upper.max())) == (-np.inf, np.inf), (
        'an open side is the infinity on that side'
    )


def test_nothing_is_patched_onto_linopy_model():
    """Importing specsolve_linopy must not touch linopy.Model."""
    assert not hasattr(linopy.Model, 'from_yaml')
    assert not hasattr(linopy.Model, 'yaml')


def test_the_lane_takes_sources_as_the_one_type():
    """The lane's two verbs annotate ``sources`` the way every door in ``api.py`` does."""
    from tests.test_architecture import sources_annotations

    doors = {'build': specsolve_linopy.build, 'evaluate': specsolve_linopy.evaluate}
    assert sources_annotations(doors) == {'Mapping[str, Source]'}, (
        f'both lane verbs take sources as Mapping[str, Source], and these do not: {sources_annotations(doors)}'
    )


# ---------------------------------------------------------------------------
# loader: master coords and parameter coercion
# ---------------------------------------------------------------------------


def _schema(dims=None, params=None) -> Spec:
    raw = {}
    if dims:
        raw['dimensions'] = dims
    if params:
        raw['parameters'] = params
    return schema_of(raw)


def _program(schema: Spec):
    """The plan the loader reads its declarations off, as a build makes one."""

    return schema.program


def _master_coords(schema: Spec, sources=None) -> dict:
    """The labels, through the front door both lanes enter by."""
    return loader.dimension_coords(_program(schema), tidy_sources(_program(schema), sources or {}))[0]


class TestMasterCoords:
    def test_labels_come_from_the_source_under_the_dimensions_own_key(self):
        assert list(_master_coords(_schema(dims={'x': {'dtype': 'int'}}), {'x': [10, 20]})['x']) == [10, 20]

    def test_a_dimension_with_no_index_is_refused(self):
        """The index is the authority, and both lanes refuse a dimension without one in the same sentence.

        Labels read out of the parameters would *be* the definition, so a
        mistyped one could not be told from a new one.
        """
        schema = _schema(dims={'x': {}}, params={'a': {'dims': ['x']}})

        with pytest.raises(ValueError, match="dimension 'x' has no index"):
            _master_coords(schema, {'a': {'wind': 1.0}})

    def test_the_labels_keep_the_order_they_arrived_in(self):
        """A caller's index is read in its own order on both lanes.

        The ordinals a translation moves by are positions in this list, so a
        sort here would move `shift` somewhere else than it moves relationally.
        """
        schema = _schema(dims={'x': {}}, params={'a': {'dims': ['x']}})
        labels = _master_coords(schema, {'x': ['z', 'a', 'm'], 'a': {'z': 1.0}})['x']

        assert list(labels) == ['z', 'a', 'm'], 'source order, not sorted'

    @pytest.mark.parametrize('index', ['pandas', 'polars', 'a bare list'])
    def test_a_temporal_axis_is_the_same_instant_whichever_library_brought_it(self, index):
        """`datetime.date` out of pandas and `datetime64` out of polars are one label.

        They compare unequal, so the declaration makes the axis canonical past
        the read and the source library stops being visible.
        """
        import datetime

        days = [datetime.date(2030, 1, d) for d in (1, 2, 3)]
        sources = {
            'pandas': pd.Index(days, name='t'),
            'polars': pl.DataFrame({'t': days}),
            'a bare list': days,
        }
        schema = _schema(dims={'t': {'dtype': 'datetime'}}, params={'a': {'dims': ['t']}})

        labels = _master_coords(schema, {'t': sources[index], 'a': {days[0]: 1.0}})['t']

        assert labels.dtype.kind == 'M', f'a {index} index reads as the instants it holds'
        assert list(labels) == list(pd.DatetimeIndex(days)), 'and as the same three of them'


class TestLoadParameters:
    """Every shape a user may hand a parameter, coerced onto the master coords."""

    @pytest.mark.parametrize(
        ('values', 'data', 'select', 'expected'),
        [
            pytest.param([1, 2], 5.0, {'x': 1}, 5.0, id='scalar-broadcasts'),
            pytest.param(['a', 'b'], {'a': 100, 'b': 60}, {'x': 'a'}, 100.0, id='dict'),
            pytest.param(
                ['a', 'b'],
                pd.Series([1.0, 2.0], index=pd.Index(['a', 'b'], name='x')),
                {'x': 'b'},
                2.0,
                id='series',
            ),
            pytest.param(
                [0, 1],
                pd.DataFrame({'x': [0, 1], 'value': [10.0, 20.0]}),
                {'x': 1},
                20.0,
                id='tidy-frame',
            ),
            pytest.param([0, 1], [10.0, 20.0], {'x': 1}, 20.0, id='sequence'),
            pytest.param(
                [0, 1],
                pl.DataFrame({'x': [0, 1], 'value': [10.0, 20.0]}),
                {'x': 1},
                20.0,
                id='a-polars-frame',
            ),
            pytest.param(
                [0, 1],
                pl.LazyFrame({'x': [0, 1], 'value': [10.0, 20.0]}),
                {'x': 1},
                20.0,
                id='a-polars-scan',
            ),
        ],
    )
    def test_accepted_shapes(self, values, data, select, expected):
        dtype = 'int' if isinstance(values[0], int) else 'str'
        s = _schema(dims={'x': {'dtype': dtype}}, params={'a': {'dims': ['x']}})
        tidy = tidy_sources(_program(s), {'x': values, 'a': data})
        ds = loader.load_parameters(_program(s), tidy, loader.dimension_coords(_program(s), tidy)[0])
        assert float(ds['a'].sel(**select)) == expected

    @pytest.mark.parametrize(
        ('dtype', 'value', 'kind'),
        [
            pytest.param('bool', False, 'b', id='bool'),
            pytest.param('str', 'a', 'U', id='str'),
            pytest.param('int', 3, 'i', id='int'),
            pytest.param('float', 2.5, 'f', id='float'),
        ],
    )
    def test_a_dims_less_parameter_keeps_the_dtype_it_declares(self, dtype, value, kind):
        s = _schema(params={'a': {'dims': [], 'dtype': dtype}})
        tidy = tidy_sources(_program(s), {'a': pd.DataFrame({'value': [value]})})
        ds = loader.load_parameters(_program(s), tidy, loader.dimension_coords(_program(s), tidy)[0])
        assert ds['a'].dtype.kind == kind, f'declared {dtype}, loaded as {ds["a"].dtype}'
        assert ds['a'].item() == value

    @pytest.mark.parametrize(
        ('dims', 'params', 'data', 'match'),
        [
            pytest.param(
                {'x': {'dtype': 'int'}},
                {'a': {'dims': ['x']}},
                {'x': [1]},
                'no data provided',
                id='missing-required',
            ),
            pytest.param(
                {'x': {'dtype': 'int'}, 'y': {'dtype': 'int'}},
                {'a': {'dims': ['x']}},
                {
                    'x': [1],
                    'y': [2],
                    'a': pd.DataFrame({'x': [1], 'y': [2], 'value': [1.0]}).set_index(['x', 'y'])['value'],
                },
                'a pandas Series with a MultiIndex is not a source',
                id='a-multi-indexed-series',
            ),
            pytest.param(
                {'x': {'dtype': 'int'}},
                {'a': {'dims': ['x']}},
                {'x': [1], 'a': xr.DataArray([1], dims=['x'], coords={'x': [1]})},
                'not a source',
                id='a-dense-array',
            ),
            pytest.param(
                {'g': {'dtype': 'str'}},
                {'p': {'dims': ['g']}},
                {'g': ['a', 'b'], 'p': pd.Series([1.0], index=pd.Index(['z'], name='g'))},
                'that are not coordinates',
                id='unknown-coord',
            ),
        ],
    )
    def test_refused_shapes(self, dims, params, data, match):
        s = _schema(dims=dims, params=params)
        with pytest.raises(ValueError, match=match):
            tidy = tidy_sources(_program(s), data)
            loader.load_parameters(_program(s), tidy, loader.dimension_coords(_program(s), tidy)[0])


# ---------------------------------------------------------------------------
# where.evaluate_where: the linopy reading of a lowered predicate
# ---------------------------------------------------------------------------


@pytest.fixture
def gens():
    """A dataset and its master coords, with one generator masked out by p_max."""
    ds = xr.Dataset({'p_max': xr.DataArray([100, 0, 50], dims=['g'], coords={'g': ['wind', 'solar', 'gas']})})
    return ds, {'g': pd.Index(['wind', 'solar', 'gas'], name='g')}


def _lowered(text, parameters=('p_max',), dimensions=('g',)):
    """The program a model with predicate *text* lowers to, and that predicate.

    Through a whole model, because the resolver is the language's.
    """
    from mathspec import to_spec

    spec = {
        'dimensions': {d: {'dtype': 'int' if d == 't' else 'str'} for d in dimensions},
        'parameters': {name: {'dims': list(dimensions)} for name in parameters},
        'variables': {'x': {'dims': list(dimensions), 'where': text, 'bounds': {'lower': 0, 'upper': 1}}},
        'objective': {'sense': 'minimize', 'expression': 'sum(x)'},
    }
    program = to_spec(spec).program
    return program, program.variables['x'].where


def _context(program, dataset, master_coords):
    return where.EvaluationContext(dataset, master_coords, linopy.Model(), {}, program)


def test_no_where_is_a_scalar_true(gens):
    program, _ = _lowered('p_max')
    mask = where.evaluate_where(None, _context(program, *gens))
    assert mask.ndim == 0
    assert bool(mask) is True


def test_a_bare_parameter_name_is_an_existence_check(gens):
    program, node = _lowered('p_max')
    assert where.evaluate_where(node, _context(program, *gens)).all()


def test_a_comparison_masks_per_coordinate(gens):
    program, node = _lowered('p_max > 0')
    mask = where.evaluate_where(node, _context(program, *gens))
    assert [bool(mask.sel(g=g)) for g in ('wind', 'solar', 'gas')] == [True, False, True]


def test_a_dimension_comparison_masks_on_the_coordinate_itself():
    program, node = _lowered('t > 0', parameters=(), dimensions=('t',))
    mask = where.evaluate_where(node, _context(program, xr.Dataset(), {'t': pd.Index([0, 1, 2], name='t')}))
    assert [bool(mask.sel(t=t)) for t in (0, 1, 2)] == [False, True, True]


def test_a_missing_parameter_is_a_load_error():
    """An undeclared name is a load error in both lanes, not a silently empty model."""
    with pytest.raises(LanguageError, match="'nonexistent' not found"):
        _lowered('nonexistent')


# ---------------------------------------------------------------------------
# error notes: the context add_note() carries out of build
# ---------------------------------------------------------------------------

_MINIMAL = """
    dimensions:
      g: {dtype: str}
    variables:
      p:
        dims: [g]
"""


def _has_note(exc: BaseException, substring: str) -> bool:
    return any(substring in n for n in getattr(exc, '__notes__', []))


#: A constant side the data does not cover, which only the *build* can see: the
#: rows exist, the parameter has no row at one of them, and the fill would be
#: the bound. The declaration it names is reached through `note()` rather than
#: written into the message.
_UNCOVERED_BOUND = "parameters:\n  cap: {dims: [g]}\nconstraints:\n  c:\n    dims: [g]\n    expression: 'p <= cap'\n"
_NO_ROWS = {'g': ['a'], 'cap': pd.Series([], index=pd.Index([], name='g', dtype='object'), dtype='float64')}


@pytest.mark.parametrize(
    ('tail', 'data', 'error', 'match', 'context'),
    [
        pytest.param(
            "    where: '<<<'\n",
            {},
            ValueError,
            'Failed to parse where string',
            "Variable 'p'",
            id='malformed-where',
        ),
        pytest.param(
            "constraints:\n  c:\n    dims: [g]\n    expression: 'p + 1'\n",
            {},
            ValueError,
            'exactly one comparison',
            "Constraint 'c'",
            id='constraint-without-comparison',
        ),
        pytest.param(
            "objective:\n  expression: 'p == 1'\n",
            {},
            ValueError,
            'must not contain a comparison',
            'The objective',
            id='objective-with-comparison',
        ),
        pytest.param(
            "constraints:\n  c:\n    dims: []\n    expression: '1 <= 2'\n",
            {},
            ValueError,
            'decides nothing',
            "Constraint 'c'",
            id='a-comparison-with-no-variable-in-it',
        ),
        pytest.param(
            _UNCOVERED_BOUND,
            _NO_ROWS,
            DataError,
            'fewer coordinates than the rows built here',
            "while building constraint 'c'",
            id='a-constant-side-the-data-misses-so-only-the-build-sees-it',
        ),
    ],
)
def test_a_failure_names_the_declaration_and_the_file(yaml_file, tail, data, error, match, context):
    bad = yaml_file(textwrap.dedent(_MINIMAL).lstrip() + tail, 'bad.yaml')

    with pytest.raises(error, match=match) as ei:
        specsolve_linopy.build(bad, {'g': ['a'], **data})

    assert context in str(ei.value) or _has_note(ei.value, context)
    assert _has_note(ei.value, f"while loading YAML '{bad}'")


# --------------------------------------------------------------------------
# the convention this lane speaks
# --------------------------------------------------------------------------


def test_importing_the_lane_selects_the_v1_convention():
    """In a *fresh* process, because ``tests/oracle.py`` sets ``semantics = 'v1'`` at import.

    linopy's default is ``legacy``, which fills an absent slot with 0 rather
    than dropping the row.
    """
    probe = 'import linopy, tests.linopy_lane; print(linopy.options["semantics"])'
    out = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == 'v1', f'the lane must select v1 on import, got {out.stdout.strip()!r}'


def test_the_two_lanes_agree_about_a_masked_variable_without_the_harness(tmp_path):
    """Both lanes, driven from a subprocess with nothing but the package imported.

    Not a ``differential()`` case: that helper runs inside the suite, where the
    convention is already set.
    """
    spec = tmp_path / 'masked.yaml'
    spec.write_text(
        textwrap.dedent("""
            dimensions: {f: {dtype: str}}
            parameters:
              gate: {dims: [f], dtype: bool}
              relmax: {dims: [f]}
            variables:
              x: {dims: [f], bounds: {lower: 0, upper: 100}}
              size: {dims: [f], where: gate, bounds: {lower: 0, upper: 50}}
            constraints:
              env:
                dims: [f]
                expression: "x - relmax * size <= 0"
            objective:
              sense: maximize
              expression: "sum(x)"
        """).lstrip()
    )
    probe = textwrap.dedent(f"""
        import warnings; warnings.simplefilter('ignore')
        import pandas as pd, polars as pl
        import specsolve as sps
        from tests import linopy_lane as fkl
        data = {{'f': ['a', 'b'], 'gate': pd.Series({{'a': True}}), 'relmax': pd.Series({{'a': 0.5, 'b': 0.5}})}}
        m = fkl.build({str(spec)!r}, data)
        m.solve(solver_name='highs', output_flag=False)
        native = sps.solve({str(spec)!r}, {{
            'f': ['a', 'b'],
            'gate': pl.DataFrame({{'f': ['a'], 'value': [True]}}),
            'relmax': pl.DataFrame({{'f': ['a', 'b'], 'value': [0.5, 0.5]}}),
        }})
        print(float(m.objective.value), native.objective)
    """)
    out = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True, check=True)
    linopy_lane, native = (float(v) for v in out.stdout.split())
    assert linopy_lane == pytest.approx(native), f'lanes disagree outside the harness: {linopy_lane} vs {native}'
    assert native == pytest.approx(125.0), 'the masked row should be dropped, leaving x[b] at its bound'


#: A scalar switch gates one variable; the other keeps the model non-empty
#: whichever way the switch is thrown.
SCALAR_SWITCH = {
    'dimensions': {'i': {'dtype': 'int'}},
    'parameters': {'on': {'dims': [], 'dtype': 'bool'}},
    'variables': {
        'x': {'dims': ['i'], 'bounds': {'lower': 1, 'upper': 5}, 'where': 'on'},
        'y': {'dims': ['i'], 'bounds': {'lower': 2, 'upper': 5}},
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(x) + sum(y)'},
}


@pytest.mark.parametrize(('on', 'expected'), [pytest.param(True, 6.0, id='on'), pytest.param(False, 4.0, id='off')])
def test_a_where_on_a_scalar_bool_agrees_on_both_lanes(on, expected):
    """A dims-less ``False`` is a false mask on both lanes, not a defined ``0.0``."""
    with differential(SCALAR_SWITCH, {'i': [1, 2], 'on': on}) as agreed:
        assert agreed.oracle == pytest.approx(expected), 'x is built only where the switch is on'


def test_a_missing_bound_is_refused_at_build_with_the_native_lane_s_message(yaml_file):
    """A bound the data does not cover is a ``DataError`` at build, as on the native lane (#313).

    A coordinate the variable does not occupy needs no bound, so the gap is
    refused and the masked one accepted.
    """
    spec = yaml_file("""
        dimensions:
          f: {dtype: str}
        parameters:
          ub: {dims: [f]}
          live: {dims: [f], dtype: bool}
        variables:
          x: {dims: [f], bounds: {lower: 0, upper: ub}}
        constraints:
          c:
            dims: [f]
            expression: x <= 100
        objective:
          sense: maximize
          expression: sum(x)
        """)
    data = {
        'f': ['a', 'b'],
        'ub': pd.Series([10.0], index=pd.Index(['a'], name='f')),
        'live': pd.Series([True], index=pd.Index(['a'], name='f')),
    }

    with pytest.raises(DataError, match='NULL bounds'):
        specsolve_linopy.build(spec, data)

    masked = yaml_file(
        spec.read_text().replace('{dims: [f], bounds:', '{dims: [f], where: live, bounds:'),
        'masked.yaml',
    )
    built = specsolve_linopy.build(masked, data)
    assert 'x' in built.variables


# ---------------------------------------------------------------------------
# named expressions: one reader per lane, one answer (#562)
# ---------------------------------------------------------------------------

EXPRESSION_YAML = """
dimensions:
  snapshot: {dtype: int}
  generator: {dtype: str}
parameters:
  p_max: {dims: [generator]}
  cost: {dims: [generator]}
  load: {dims: [snapshot]}
variables:
  p:
    dims: [snapshot, generator]
    bounds: {lower: 0, upper: p_max}
expressions:
  total_gen: sum(p, over=generator)
  spend: sum(p * cost, over=generator)
  squared: sum(p * p, over=generator)
  price: dual(balance)
constraints:
  balance:
    dims: [snapshot]
    expression: total_gen == load
objective:
  sense: minimize
  expression: sum(sum(p * cost, over=generator), over=snapshot)
"""

#: Distinct costs and a load exceeding the cheap generator's capacity make the
#: dispatch unique, so the two lanes' expression values are comparable exactly
#: rather than up to an alternative optimum. No load sits at a capacity, so the
#: duals are unique too, which is what lets ``price`` be compared lane to lane.
EXPRESSION_DATA = {
    'snapshot': [0, 1, 2],
    'generator': ['g1', 'g2'],
    'p_max': pd.Series({'g1': 100.0, 'g2': 100.0}),
    'cost': pd.Series({'g1': 10.0, 'g2': 20.0}),
    'load': pd.Series({0: 50.0, 1: 120.0, 2: 80.0}),
}


#: ``EXPRESSION_YAML`` with ``p`` masked out at the generator whose ``p_max`` is
#: zero and declared ``absence: zero``, plus one entry only a nonlinear read can
#: tell apart from an affine one.
ZERO_ABSENCE_YAML = EXPRESSION_YAML.replace(
    '    bounds: {lower: 0, upper: p_max}\n',
    '    bounds: {lower: 0, upper: p_max}\n    where: p_max > 0\n    absence: zero\n',
).replace(
    '  spend: sum(p * cost, over=generator)\n',
    '  spend: sum(p * cost, over=generator)\n  grown: sum(0.5 ** p, over=generator)\n',
)

ZERO_ABSENCE_DATA = {**EXPRESSION_DATA, 'p_max': pd.Series({'g1': 200.0, 'g2': 0.0})}


def test_the_two_lanes_agree_on_an_absent_slot_declared_zero_under_a_nonlinear_read(yaml_file):
    """`absence: zero` is a zero on both lanes, so `0.5 ** p` reads `0.5 ** 0`, which is 1, at the masked generator on each."""
    path = yaml_file(ZERO_ABSENCE_YAML, 'zero_absence.yaml')
    with differential(path, ZERO_ABSENCE_DATA) as run:
        tidy = run.result.evaluate('grown')
        linopy_lane = specsolve_linopy.evaluate(run.model, path, 'grown', dict(ZERO_ABSENCE_DATA))
        got = {int(k): v for k, v in zip(tidy['snapshot'], tidy['value'], strict=True)}
        want = {int(k): float(v) for k, v in linopy_lane.to_series().items()}
        assert got == pytest.approx(want), 'the two lanes disagree about an absent slot declared zero'
        assert all(v == pytest.approx(1.0) for v in want.values()), (
            'each snapshot reads 1, the absent generator as 0.5 ** 0, plus a term below double precision from the present one'
        )


def test_a_dual_on_a_solve_that_left_none_is_refused_on_this_lane_too(yaml_file):
    """An integer variable makes duals undefined; linopy stores HiGHS's zeros for a MIP, so the read refuses by the declaration rather than reading a number that means nothing."""
    path = yaml_file(
        EXPRESSION_YAML.replace(
            '    dims: [snapshot, generator]\n', '    dims: [snapshot, generator]\n    domain: integer\n'
        ),
        'integer.yaml',
    )
    built = specsolve_linopy.build(path, dict(EXPRESSION_DATA))
    built.solve(solver_name='highs')
    with pytest.raises(SpecsolveError, match='duals are undefined'):
        specsolve_linopy.evaluate(built, path, 'price', dict(EXPRESSION_DATA))
    assert float(specsolve_linopy.evaluate(built, path, 'spend', dict(EXPRESSION_DATA)).sum()) > 0, (
        'the refusal is per entry: the affine one still reads'
    )


@pytest.mark.parametrize(
    'name',
    [
        pytest.param('total_gen', id='referenced-by-a-constraint'),
        pytest.param('spend', id='declared-but-never-referenced'),
        pytest.param('squared', id='degree-two-in-an-entry-the-math-never-reads'),
        pytest.param('price', id='reading-a-dual'),
    ],
)
def test_the_two_lanes_agree_on_a_named_expression(yaml_file, name):
    """`result.evaluate(name)` and the lane's `evaluate` read one value.

    Including a never-referenced expression (#562). The linopy lane evaluates it
    at the solved model's `.solution` and `.dual` arrays, so an entry of any
    degree, and one reading a dual, reads on both lanes.
    """
    path = yaml_file(EXPRESSION_YAML, 'expressions.yaml')
    with differential(path, EXPRESSION_DATA) as run:
        tidy = run.result.evaluate(name)
        linopy_lane = specsolve_linopy.evaluate(run.model, path, name, dict(EXPRESSION_DATA))
        got = {int(k): v for k, v in zip(tidy['snapshot'], tidy['value'], strict=True)}
        want = {int(k): float(v) for k, v in linopy_lane.to_series().items()}
        assert got == pytest.approx(want), f"the two lanes disagree about named expression '{name}'"


#: A curve masked by ``points:``, so the expansion declares a parameter the file
#: does not — ``cost_curve_points``, derived from ``bp_x``'s own rows. Ragged on
#: purpose: hydro states two breakpoints where the axis has four.
MASKED_CURVE_YAML = """
dimensions:
  snapshot: {dtype: int}
  generator: {dtype: str}
  bp: {dtype: int}
parameters:
  p_max: {dims: [generator]}
  load: {dims: [snapshot]}
  bp_x: {dims: [generator, bp]}
  bp_y: {dims: [generator, bp]}
variables:
  p:
    dims: [snapshot, generator]
    bounds: {lower: 0, upper: p_max}
  op_cost:
    dims: [snapshot, generator]
    bounds: {lower: 0}
piecewise:
  cost_curve:
    over: bp
    points: bp_x
    links:
      - [p, bp_x]
      - [op_cost, bp_y, ">="]
    method: convex
expressions:
  spend: sum(op_cost, over=generator)
constraints:
  balance:
    dims: [snapshot]
    expression: sum(p, over=generator) == load
objective:
  sense: minimize
  expression: sum(sum(op_cost, over=generator), over=snapshot)
"""


MASKED_CURVE_DATA = {
    'snapshot': [0],
    'generator': ['hydro', 'gas'],
    'bp': [0, 1, 2, 3],
    'p_max': pd.Series({'hydro': 40.0, 'gas': 80.0}),
    'load': pd.Series([50.0], index=pd.RangeIndex(1, name='snapshot')),
    'bp_x': curve_frame(
        {('hydro', 0): 0.0, ('hydro', 1): 40.0, ('gas', 0): 0.0, ('gas', 1): 20.0, ('gas', 2): 50.0, ('gas', 3): 80.0}
    ),
    'bp_y': curve_frame(
        {
            ('hydro', 0): 0.0,
            ('hydro', 1): 200.0,
            ('gas', 0): 0.0,
            ('gas', 1): 150.0,
            ('gas', 2): 450.0,
            ('gas', 3): 900.0,
        }
    ),
}


def test_a_named_expression_reads_off_a_masked_curve(yaml_file):
    """Both lanes value one named expression the same where a curve is masked.

    `points:` names a values parameter, and the mask is that parameter read as
    a bare name in a `where` — true wherever it has a row. So the weights the
    expression sums over exist at some breakpoints and not others, and a lane
    that read the unmasked product would answer a longer curve.
    """
    path = yaml_file(MASKED_CURVE_YAML, 'masked_curve.yaml')
    with differential(path, MASKED_CURVE_DATA) as run:
        tidy = run.result.evaluate('spend')
        linopy_lane = specsolve_linopy.evaluate(run.model, expanded(path), 'spend', dict(MASKED_CURVE_DATA))
        got = {int(k): v for k, v in zip(tidy['snapshot'], tidy['value'], strict=True)}
        want = {int(k): float(v) for k, v in linopy_lane.to_series().items()}
        assert got == pytest.approx(want), 'the two lanes disagree about a named expression over a masked curve'


def test_the_lane_values_an_expression_the_file_never_declared(yaml_file):
    """An expression string is what `evaluate` takes, alongside a name the file declares.

    Both spellings reach the same node, so `total_gen`'s own body is the check.
    """
    path = yaml_file(EXPRESSION_YAML, 'expressions.yaml')
    m = specsolve_linopy.build(path, dict(EXPRESSION_DATA))
    m.solve(solver_name='highs')
    written = specsolve_linopy.evaluate(m, path, 'sum(p, over=generator)', dict(EXPRESSION_DATA))
    declared = specsolve_linopy.evaluate(m, path, 'total_gen', dict(EXPRESSION_DATA))
    assert float(written.sum()) == pytest.approx(float(declared.sum())), (
        'the body and the name it is declared under are one expression, so they read one value'
    )


def test_the_lane_refuses_an_expression_against_a_lowered_program(yaml_file):
    """A Program is what a model lowered to, and lowering does not run backwards."""
    from mathspec import to_spec

    path = yaml_file(EXPRESSION_YAML, 'expressions.yaml')
    m = specsolve_linopy.build(path, dict(EXPRESSION_DATA))
    m.solve(solver_name='highs')
    with pytest.raises(SpecsolveError, match='lowered Program'):
        specsolve_linopy.evaluate(m, to_spec(path).program, 'total_gen', dict(EXPRESSION_DATA))


def test_one_set_of_tables_reaches_both_lanes(dispatch_yaml, dispatch_frame_inputs, tmp_path):
    """One `sources` mapping of polars frames and a parquet path reaches either lane unchanged (#60)."""
    frames = dispatch_frame_inputs
    path = tmp_path / 'load.parquet'
    frames['load'].write_parquet(path)
    sources = {**frames, 'load': path}

    with differential(dispatch_yaml, sources) as run:
        assert run.result.primal('p').height, 'the relational lane built no rows'
        assert float(run.model.variables['p'].labels.count()), 'the linopy lane built no variables'


@pytest.mark.parametrize(
    'as_spec',
    [
        pytest.param(lambda raw, path: path, id='a-path'),
        pytest.param(lambda raw, path: raw, id='a-mapping'),
        pytest.param(lambda raw, path: schema_of(raw), id='a-loaded-model'),
    ],
)
def test_the_lane_takes_a_model_the_same_three_ways_the_runner_does(tmp_path, as_spec):
    """`sps.build` and this take the same first argument, so neither decides the lane (#845)."""
    import yaml as pyyaml

    raw = {
        'dimensions': {'g': {'dtype': 'str'}},
        'parameters': {'cap': {'dims': ['g']}},
        'variables': {'x': {'dims': ['g'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
        'objective': {'sense': 'maximize', 'expression': 'sum(x)'},
    }
    path = tmp_path / 'm.yaml'
    path.write_text(pyyaml.safe_dump(raw))

    built = specsolve_linopy.build(as_spec(raw, path), {'g': ['wind', 'gas'], 'cap': {'wind': 40.0, 'gas': 100.0}})
    assert 'x' in built.variables, 'the same file, whichever way it was handed over'


#: A shift over a variable-free expression: the vacated positions have no
#: value, and inventing one silently pins a bound to zero.
_BARE_SHIFT = {
    'dimensions': {'t': {'dtype': 'int'}},
    'parameters': {'eff': {'dims': ['t']}},
    'variables': {'x': {'dims': ['t'], 'bounds': {'lower': 0, 'upper': 5}}},
    'constraints': {'c': {'dims': ['t'], 'expression': 'x <= shift(eff, along=t, offset=1)'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(x)'},
}


def test_a_construct_the_streaming_lane_refuses_is_refused_here_too():
    """One gate, both lanes: the lowering pass's refusals fire on this lane too."""
    import specsolve as sps

    with pytest.raises(LanguageError, match='vacated positions') as native:
        sps.check(_BARE_SHIFT)
    with pytest.raises(LanguageError, match='vacated positions') as linopy_lane:
        specsolve_linopy.build(_BARE_SHIFT, {'eff': {0: 1.0, 1: 2.0, 2: 3.0}})

    assert str(native.value) == str(linopy_lane.value), 'one refusal, one wording, whichever lane was asked'


#: The one construct this lane accepts and cannot build: a bare parameter term
#: in the objective, which linopy has no slot for. `osemosys_utopia` owes one
#: as the fixed cost of capacity that already stood in 1990.
OBJECTIVE_CONSTANT = {
    'dimensions': {'t': {'dtype': 'int'}},
    'parameters': {'standing': {'dims': []}},
    'variables': {'x': {'dims': ['t'], 'bounds': {'lower': 0, 'upper': 1}}},
    'objective': {'sense': 'minimize', 'expression': 'sum(x) + standing'},
}


def test_a_construct_this_lane_cannot_build_is_refused_in_its_own_words():
    """The mirror of the test above: the streaming lane builds this one.

    The model is sayable, lowers and solves relationally, so the refusal says
    the wall is this lane's, which linopy's `Constant values in objective
    function not supported.` cannot say (#894).
    """
    import specsolve as sps

    assert sps.solve(OBJECTIVE_CONSTANT, {'t': [0, 1], 'standing': 5.0}).objective == pytest.approx(5.0), (
        'the streaming lane builds it, so the model is not the problem'
    )
    with pytest.raises(loader.OracleCannotBuildError) as refusal:
        specsolve_linopy.build(OBJECTIVE_CONSTANT, {'t': [0, 1], 'standing': 5.0})

    assert str(refusal.value) == builder.OBJECTIVE_CONSTANT_IS_A_LANE_GAP, (
        "the sentence is the lane's own, which is the whole of the fix"
    )


#: The terms on the right of the comparison. The language puts them on neither
#: side — `ConstraintDeclaration` says which side a consumer gathers them onto
#: is its own arrangement — and linopy takes them only on the left.
TERM_ON_THE_RIGHT = {
    'dimensions': {'g': {'dtype': 'str'}},
    'parameters': {'cap': {'dims': ['g']}, 'cost': {'dims': ['g']}},
    'variables': {'p': {'dims': ['g'], 'bounds': {'lower': 0, 'upper': 100}}},
    'constraints': {'limit': {'dims': ['g'], 'expression': 'cap >= p'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(cost * p, over=g)'},
}

TERM_ON_THE_RIGHT_DATA = {'g': ['a', 'b'], 'cap': {'a': 10.0, 'b': 20.0}, 'cost': {'a': 1.0, 'b': 1.0}}


def test_a_constraint_carrying_its_terms_on_the_right_builds_on_both_lanes():
    """`cap >= p` builds on both lanes, though linopy takes a term only on the left (#1534).

    The swap flips the sense with it, which the objective tells: read as
    `p >= cap` both variables would run to their bound of 100.
    """
    with differential(TERM_ON_THE_RIGHT, TERM_ON_THE_RIGHT_DATA) as agreed:
        assert agreed.oracle == pytest.approx(30.0), 'each generator is capped by its own row, at 10 and at 20'
        assert set(np.unique(agreed.model.constraints['limit'].sign.values)) == {'<='}, (
            'the swap flips the sense with it: `cap >= p` is built as `p <= cap`'
        )


def test_a_file_that_declares_no_labels_at_all_is_refused_on_both_lanes():
    """The index is what says which labels exist, on either lane.

    Neither `values:` nor a table under `g` says what it holds, so a mistyped
    label in `cost` would define a generator rather than fail. Both lanes
    refuse, in the same sentence.
    """
    import specsolve as sps
    from specsolve.errors import DataError

    spec = {
        'dimensions': {'g': {}},
        'parameters': {'cap': {'dims': ['g']}, 'cost': {'dims': ['g']}},
        'variables': {'x': {'dims': ['g'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
        'objective': {'sense': 'maximize', 'expression': 'sum(x * cost)'},
    }
    sources = {
        'cap': pd.Series({'wind': 40.0, 'gas': 100.0}).rename_axis('g'),
        'cost': pd.Series({'wind': 3.0, 'gas': 1.0}).rename_axis('g'),
    }

    with pytest.raises(DataError, match="dimension 'g' has no index") as native:
        sps.build(spec, sources).close()
    with pytest.raises(DataError, match="dimension 'g' has no index") as linopy_lane:
        specsolve_linopy.build(spec, sources)
    assert str(native.value) == str(linopy_lane.value), 'one refusal, one wording'

    indexed = {**sources, 'g': pd.DataFrame({'g': ['wind', 'gas']})}
    assert 'x' in specsolve_linopy.build(spec, indexed).variables


def test_from_yaml_fails_before_data_validation(tmp_path):
    """A typo in an expression errors even when data= is absent."""
    f = tmp_path / 'm.yaml'
    f.write_text(
        'dimensions:\n'
        '  g: {dtype: str}\n'
        'variables:\n'
        '  p:\n'
        '    dims: [g]\n'
        'constraints:\n'
        '  cap:\n'
        '    dims: [g]\n'
        '    expression: pp <= 100\n'
    )
    with pytest.raises(ValueError, match="'pp' not found"):
        specsolve_linopy.build(f, {})


def test_dispatch_yaml_agrees_variable_by_variable(dispatch_inputs):
    """The two lanes agree variable by variable, not only in total.

    An objective can agree while the dispatch behind it differs, which is what
    this rules out.
    """
    data = dispatch_inputs

    with differential(EXAMPLES_DIR / 'dispatch.yaml', data, lp=True) as run:
        linopy_p = run.model.solution['p'].to_dataframe(name='value').reset_index()
        rel_p = run.result.to_pandas('p')
        merged = linopy_p.merge(rel_p, on=['snapshot', 'generator'], suffixes=('_linopy', '_rel'))
        assert len(merged) == len(rel_p), 'nothing is masked here, so the rows align 1:1'
        assert np.allclose(merged['value_linopy'], merged['value_rel'], atol=1e-6)
