"""Native API: YAML → streaming engine → solver, with linopy never imported.

The linopy-free guarantee is asserted in a subprocess so the suite's own
oracle imports cannot pollute the check. This module is pandas-free; the tests
of the bridges out say so with an ``importorskip``.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import textwrap
from dataclasses import replace
from unittest import mock

import numpy as np
import polars as pl
import pytest
from mathspec import Spec, to_spec

import specsolve as sps
from specsolve.errors import DimensionError
from tests.conftest import (
    CASES,
    DISPATCH_COST,
    DISPATCH_GENERATORS,
    DISPATCH_P_MAX,
    EXAMPLES_DIR,
    _dispatch_load,
    override,
    raw_of,
    solve_written_file,
)


@pytest.fixture
def dispatch_solution(dispatch_yaml, dispatch_frame_inputs):
    """The dispatch model solved on the native lane, closed after the test."""
    sources = dispatch_frame_inputs
    with sps.solve(dispatch_yaml, sources) as result:
        yield result


def test_solve(dispatch_solution, dispatch_frame_inputs):
    sources = dispatch_frame_inputs
    assert dispatch_solution.is_ok
    assert np.isfinite(dispatch_solution.objective)
    balance = dispatch_solution.primal('p').group_by('snapshot').agg(pl.col('value').sum()).sort('snapshot')
    assert np.allclose(balance['value'], sources['load']['value'])


def test_build_context_manager_and_write(dispatch_yaml, dispatch_frame_inputs, tmp_path):
    sources = dispatch_frame_inputs
    with sps.build(dispatch_yaml, sources) as model:
        result = model.solve()
        assert result.is_ok
        objective_direct = result.objective

    lp = sps.write(dispatch_yaml, sources, tmp_path / 'm.lp')
    assert solve_written_file(lp) == pytest.approx(objective_direct, rel=1e-9)


def test_a_points_parameter_supplied_as_a_parquet_path_keeps_its_own_curve_length(tmp_path):
    """A ``points:`` curve supplied as a file keeps its own length, as a frame does."""
    from tests.conftest import expanded, port_sources, port_spec

    frames = port_sources('piecewise_ragged')
    paths = {}
    for name, frame in frames.items():
        frame.write_parquet(tmp_path / f'{name}.parquet')
        paths[name] = str(tmp_path / f'{name}.parquet')

    with (
        sps.solve(expanded(port_spec('piecewise_ragged')), paths) as from_paths,
        sps.solve(expanded(port_spec('piecewise_ragged')), frames) as from_frames,
    ):
        assert from_paths.objective == pytest.approx(from_frames.objective, rel=1e-9), (
            'a curve read from a file is the curve read from a frame'
        )


def test_a_string_is_a_parquet_path_at_every_door(tmp_path):
    """Every reader of a source goes through `as_frame`, so a path attaches the same everywhere."""
    from specsolve.frames import as_frame

    frame = pl.DataFrame({'snapshot': [0, 1], 'value': [1.0, 2.0]})
    frame.write_parquet(tmp_path / 'load.parquet')
    assert as_frame(str(tmp_path / 'load.parquet')).collect().equals(frame), 'a str is scanned as parquet'
    assert as_frame(tmp_path / 'load.parquet').collect().equals(frame), 'and so is a Path'
    assert as_frame(frame).collect().equals(frame), 'a table is normalised as before'
    assert as_frame(2.0) is None, 'a number is not a table, and the caller says what it is'


def test_parquet_path_sources(dispatch_yaml, dispatch_frame_inputs, tmp_path):
    sources = dispatch_frame_inputs
    paths = {}
    for name, frame in sources.items():
        p = tmp_path / f'{name}.parquet'
        frame.write_parquet(p)
        paths[name] = str(p)

    with sps.solve(dispatch_yaml, paths) as result:
        assert result.is_ok
        objective = result.objective

    with sps.solve(dispatch_yaml, sources) as ref:
        assert objective == pytest.approx(ref.objective, rel=1e-9)


#: ``examples/dispatch.yaml``'s numbers as plain Python rather than tables.
_PLAIN = {
    'dict': {
        'p_max': dict(zip(DISPATCH_GENERATORS, DISPATCH_P_MAX, strict=True)),
        'cost': dict(zip(DISPATCH_GENERATORS, DISPATCH_COST, strict=True)),
        'load': dict(enumerate(_dispatch_load())),
    },
    'sequence': {
        'p_max': list(DISPATCH_P_MAX),
        'cost': list(DISPATCH_COST),
        'load': _dispatch_load(),
    },
}


@pytest.mark.parametrize('shape', sorted(_PLAIN), ids=sorted(_PLAIN))
def test_plain_python_sources_reach_the_same_answer_as_tables(dispatch_yaml, dispatch_frame_inputs, shape):
    """A dict and a sequence are sources, and mean what the tables mean.

    A dict carries its own labels; a sequence is positional against the index.
    """
    frames = dispatch_frame_inputs
    with sps.solve(dispatch_yaml, frames) as tables:
        expected = tables.objective

    index = {'snapshot': frames['snapshot'], 'generator': frames['generator']}
    with sps.solve(dispatch_yaml, _PLAIN[shape] | index) as plain:
        assert plain.objective == pytest.approx(expected, rel=1e-9)


def test_one_number_stands_for_every_coordinate(dispatch_yaml, dispatch_frame_inputs):
    """A scalar covers the dims the parameter declares, not just a 0-D one."""
    frames = dispatch_frame_inputs
    flat = {**frames, 'cost': 7.0}
    spelled = {**frames, 'cost': pl.DataFrame({'generator': list(DISPATCH_GENERATORS), 'value': [7.0] * 3})}

    with (
        sps.solve(dispatch_yaml, flat) as broadcast,
        sps.solve(dispatch_yaml, spelled) as written,
    ):
        assert broadcast.objective == pytest.approx(written.objective, rel=1e-9)


@pytest.mark.parametrize(
    ('sources', 'match'),
    [
        pytest.param({'p_max': [100.0, 60.0]}, 'one entry per label', id='a-sequence-of-the-wrong-length'),
        pytest.param({'p_max': object()}, 'cannot adapt', id='nothing-table-shaped-at-all'),
        pytest.param({'snapshot': 3.0}, 'cannot read labels out of float', id='an-index-that-is-one-number'),
    ],
)
def test_a_plain_python_source_that_does_not_fit_is_refused(dispatch_yaml, dispatch_frame_inputs, sources, match):
    frames = dispatch_frame_inputs
    with pytest.raises(sps.errors.DataError, match=match):
        sps.build(dispatch_yaml, {**frames, **sources}).close()


#: One parameter over two dims — what a dict and a sequence cannot cover.
_TWO_DIMS = {
    'dimensions': {'g': {'dtype': 'str'}, 't': {'dtype': 'int'}},
    'parameters': {'cap': {'dims': ['g', 't']}},
    'variables': {'x': {'dims': ['g', 't'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
    'objective': {'sense': 'maximize', 'expression': 'sum(x)'},
}


@pytest.mark.parametrize(
    ('source', 'match'),
    [
        pytest.param({('wind', 0): 1.0}, 'a dict maps one label to one value', id='a-dict'),
        pytest.param([1.0, 2.0, 3.0, 4.0], 'a sequence runs along one dimension', id='a-sequence'),
    ],
)
def test_a_flat_shape_cannot_cover_two_dimensions(source, match):
    """Both carry one axis, and the rewrite is the table that carries both."""
    with pytest.raises(sps.errors.DataError, match=match):
        sps.build(_TWO_DIMS, {'cap': source}).close()


def test_a_one_level_series_cannot_cover_two_dimensions():
    """A pandas Series is a sequence with its index along: one axis, declined the same way."""
    pandas = pytest.importorskip('pandas')
    series = pandas.Series([1.0, 2.0], index=pandas.Index(['wind', 'gas'], name='g'))
    with pytest.raises(sps.errors.DataError, match='a sequence runs along one dimension'):
        sps.build(_TWO_DIMS, {'cap': series}).close()


def test_a_positional_source_needs_the_labels_it_is_written_against():
    """A sequence says what the values are and not what they are labelled, and no
    lane reads labels off the parameters."""
    spec = {
        'dimensions': {'g': {}},
        'parameters': {'cap': {'dims': ['g']}},
        'variables': {'x': {'dims': ['g'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
        'objective': {'sense': 'maximize', 'expression': 'sum(x, over=g)'},
    }
    with pytest.raises(sps.errors.DataError, match='nothing else supplies an index'):
        sps.build(spec, {'cap': [1.0, 2.0]}).close()


def test_runtime_is_linopy_free(dispatch_yaml):
    """Import the package, build and solve on Arrow sources — linopy, xarray, pandas and pyarrow never load.

    Weaker than the claim that they need not be installed, which the
    bare-install CI job proves.
    """
    absent = ('linopy', 'xarray', 'pandas', 'pyarrow')
    script = textwrap.dedent(f"""
        import sys
        assert "linopy" not in sys.modules

        import polars as pl
        import specsolve as sps
        for lib in {absent!r}:
            assert lib not in sys.modules, f"package import pulled in {{lib}}"

        result = sps.solve(
            {str(dispatch_yaml)!r},
            {{
                "p_max": pl.DataFrame({{"generator": ["wind", "solar", "gas"],
                                       "value": [100.0, 60.0, 200.0]}}),
                "cost": pl.DataFrame({{"generator": ["wind", "solar", "gas"],
                                      "value": [1.0, 2.0, 50.0]}}),
                "load": pl.DataFrame({{"snapshot": [0, 1, 2],
                                      "value": [80.0, 120.0, 150.0]}}),
                "snapshot": range(3),
                "generator": ["wind", "solar", "gas"],
            }},
        )
        assert result.is_ok
        assert isinstance(result.primal("p"), pl.DataFrame), "no dataframe on either side"
        assert result.primal("p").height == 9
        result.close()
        for lib in {absent!r}:
            assert lib not in sys.modules, f"solve pulled in {{lib}}"
        print("LINOPY_FREE_OK")
    """)
    out = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert 'LINOPY_FREE_OK' in out.stdout


@pytest.mark.parametrize(
    'form',
    ['path', 'str', 'dict', 'spec'],
)
def test_every_verb_opens_a_model_the_way_the_language_does(dispatch_yaml, dispatch_frame_inputs, tmp_path, form):
    """One first argument across the five verbs, and it is `to_spec`'s own.

    Asserted per verb: each has its own door.
    """
    spec = {
        'path': dispatch_yaml,
        'str': str(dispatch_yaml),
        'dict': to_spec(dispatch_yaml).to_dict(),
        'spec': to_spec(dispatch_yaml),
    }[form]
    with sps.solve(dispatch_yaml, dispatch_frame_inputs) as reference:
        expected = reference.objective

    assert sps.check(spec).variables['p'].dims == ('snapshot', 'generator'), (
        'check lowers it, and the plan is the same one whichever form the model arrived as'
    )
    with sps.build(spec, dispatch_frame_inputs) as model:
        assert model.solve('highs').objective == pytest.approx(expected, rel=1e-9), 'build takes it'
    with sps.solve(spec, dispatch_frame_inputs) as result:
        assert result.objective == pytest.approx(expected, rel=1e-9), 'and so does solve'
    assert sps.write(spec, dispatch_frame_inputs, tmp_path / f'{form}.lp').exists(), 'and write'

    runs = sps.solve_over(spec, dispatch_frame_inputs, [(0, dict(dispatch_frame_inputs))], key_name='draw')
    assert runs.keys == [0], 'a hand-built axis of one slice still runs, whatever the model arrived as'
    assert runs.record['objective'].to_list() == pytest.approx([expected], rel=1e-9), (
        'and the sweep reaches the answer the one-shot verbs do'
    )


def test_check_and_the_spec_program_need_no_data(dispatch_yaml):
    """The plan is read from the model alone, with no data."""
    for program in (sps.check(dispatch_yaml), to_spec(dispatch_yaml).program):
        assert program.variables['p'].dims == ('snapshot', 'generator')
        assert program.parameters['load'].dims == ('snapshot',)


@pytest.mark.parametrize(
    ('expression', 'match'),
    [
        pytest.param('sum(p ** 2)', 'over variables', id='a-power-over-a-variable'),
        pytest.param(
            'sum(p) * sum(p)',
            'sums of more than one term',
            id='two-reductions-multiplied-caught-with-no-data-bound',
        ),
    ],
)
def test_check_reports_language_errors_before_any_data_is_bound(
    dispatch_yaml, dispatch_frame_inputs, expression, match
):
    """The CI verb enforces the ceiling with no data attached (mathspec's docs/about/limits.md).

    The refusal is the language's; both ``check`` and ``build`` surface it.
    """
    raw = {**to_spec(dispatch_yaml).model_dump(), 'objective': {'sense': 'minimize', 'expression': expression}}

    with pytest.raises(sps.errors.LanguageError, match=match):
        sps.check(raw)
    sources = dispatch_frame_inputs
    with pytest.raises(sps.errors.LanguageError, match=match):
        sps.build(raw, sources)


def test_error_hierarchy_is_one_catchable_tree():
    """One ``except`` covers the package, and the model/run split is real."""
    for cls in (sps.errors.LanguageError, sps.errors.DataError):
        assert issubclass(cls, sps.errors.SpecsolveError)
    for cls in (sps.errors.SchemaError, sps.errors.DimensionError):
        assert issubclass(cls, sps.errors.LanguageError)
    assert not issubclass(sps.errors.DataError, sps.errors.LanguageError)
    assert issubclass(sps.errors.SpecsolveError, ValueError)


def test_an_unknown_solver_is_refused_with_the_alternatives(dispatch_yaml, dispatch_frame_inputs):
    """The set of solvers is closed, and a name outside it never falls back to the default."""
    from specsolve.relational.sinks import SOLVERS

    sources = dispatch_frame_inputs
    with pytest.raises(sps.errors.SpecsolveError, match='unknown solver'):
        sps.solve(dispatch_yaml, sources, solver_name='cplex')
    assert set(SOLVERS) == {'highs', 'gurobi', 'xpress'}


def test_a_solver_this_environment_cannot_run_is_refused_before_the_build(
    dispatch_yaml, dispatch_frame_inputs, monkeypatch
):
    """A name in the closed set is not a promise the package is installed.

    Faked by naming a package nothing has, so the real probe runs wherever the
    suite does.
    """
    from specsolve import api
    from specsolve.relational.sinks import SOLVERS

    sources = dispatch_frame_inputs
    monkeypatch.setattr(SOLVERS['gurobi'], 'requires', ('a_package_no_environment_has',))
    monkeypatch.setattr(api.Engine, 'build', lambda *_a, **_k: pytest.fail('the model was built before the refusal'))

    with pytest.raises(ModuleNotFoundError, match=r'not installed here.*\[gurobi\] extra'):
        sps.solve(dispatch_yaml, sources, solver_name='gurobi')


def test_a_list_of_models_is_refused(dispatch_yaml):
    """Composition is merging declarations, not passing several models."""
    with pytest.raises(sps.errors.LanguageError, match='merge the declarations'):
        sps.check([dispatch_yaml, dispatch_yaml])


#: A model over `t` with one variable, one constraint and one named expression,
#: so a case pair can be introduced into any namespace in turn.
def _named(**declared) -> dict:
    spec = {
        'dimensions': {'t': {'dtype': 'int'}},
        'parameters': {'load': {'dims': ['t']}},
        'variables': {'p': {'dims': ['t'], 'bounds': {'lower': 0, 'upper': 10}}},
        'constraints': {'meet': {'dims': ['t'], 'expression': 'p >= load'}},
        'expressions': {'spend': '2 * p'},
        'objective': {'sense': 'minimize', 'expression': 'sum(p)'},
    }
    for section, added in declared.items():
        spec[section] = {**spec[section], **added}
    return spec


@pytest.mark.parametrize(
    'spec',
    [
        pytest.param(
            _named(variables={'P': {'dims': ['t'], 'bounds': {'lower': 0, 'upper': 10}}}),
            id='two variables',
        ),
        pytest.param(_named(parameters={'P': {'dims': ['t']}}), id='a parameter beside a variable'),
        pytest.param(_named(dimensions={'T': {'dtype': 'int'}}), id='two dimensions'),
        pytest.param(_named(expressions={'SPEND': '3 * p'}), id='two named expressions'),
        pytest.param(_named(expressions={'P': '3 * p'}), id='a named expression beside a variable'),
        pytest.param(
            _named(constraints={'MEET': {'dims': ['t'], 'expression': 'p >= load'}}),
            id='two constraints',
        ),
    ],
)
def test_two_names_in_one_namespace_differing_only_by_case_are_refused(spec):
    """`p` beside `P` is refused: on a case-insensitive filesystem their files fold into one."""
    with pytest.raises(sps.errors.SpecsolveError, match='differ only by case'):
        sps.check(spec)


def test_a_case_pair_across_two_namespaces_is_allowed():
    """The rule is per namespace: a constraint is written under `dual/`, a variable under `primal/`."""
    spec = _named(constraints={'P': {'dims': ['t'], 'expression': 'p >= load'}})
    assert 'P' in sps.check(spec).constraints, "a constraint named like a variable is the language's to allow"


@pytest.mark.parametrize(
    ('spec', 'named'),
    [
        pytest.param(_named(dimensions={'specsolve_t': {'dtype': 'int'}}), "dimension 'specsolve_t'", id='a dimension'),
        pytest.param(
            _named(dimensions={'specsolve_position': {'dtype': 'int'}}),
            "dimension 'specsolve_position'",
            id='a dimension named as the column that numbers its labels',
        ),
        pytest.param(
            _named(parameters={'specsolve_load': {'dims': ['t']}}), "parameter 'specsolve_load'", id='a parameter'
        ),
        pytest.param(
            _named(variables={'specsolve_p': {'dims': ['t'], 'bounds': {'lower': 0, 'upper': 10}}}),
            "variable 'specsolve_p'",
            id='a variable',
        ),
        pytest.param(
            _named(constraints={'specsolve_meet': {'dims': ['t'], 'expression': 'p >= load'}}),
            "constraint 'specsolve_meet'",
            id='a constraint',
        ),
        pytest.param(
            _named(expressions={'specsolve_spend': '3 * p'}),
            "named expression 'specsolve_spend'",
            id='a named expression',
        ),
        pytest.param(
            {**_named(), 'relations': {'specsolve_next': {'key': {'here': 't'}, 'values': {'there': 't'}}}},
            "relation 'specsolve_next'",
            id='a relation',
        ),
        pytest.param(
            {**_named(), 'relations': {'next': {'key': {'specsolve_here': 't'}, 'values': {'there': 't'}}}},
            "column 'specsolve_here' of relation 'next'",
            id='a column of a relation',
        ),
        pytest.param(
            {**_named(), 'sos': {'specsolve_pick': {'variable': 'p', 'along': 't', 'type': 1}}},
            "sos set 'specsolve_pick'",
            id='an sos set',
        ),
        pytest.param(
            {**_named(), 'assumptions': {'specsolve_positive': 'load >= 0'}},
            "assumption 'specsolve_positive'",
            id='an assumption',
        ),
    ],
)
def test_a_name_that_starts_with_the_reserved_prefix_is_refused(spec, named):
    """Every column specsolve adds starts with `specsolve_`, so a declared name that does could collide with one."""
    with pytest.raises(sps.errors.SpecsolveError, match=rf"^{named} starts with 'specsolve_', which is reserved"):
        sps.check(spec)


@pytest.mark.parametrize('name', ['Specsolve_run', 'SPECSOLVE_RUN', 'specSolve_p'], ids=str)
def test_the_reserved_prefix_is_refused_in_any_letter_case(name):
    """A capital passed the reserved prefix, but SQL, DuckDB and Power BI read column names without case.

    So a variable `Specsolve_run` and the stamped `specsolve_run` were one
    column to every query engine an archive is read with.
    """
    spec = _named(variables={name: {'dims': ['t'], 'bounds': {'lower': 0, 'upper': 10}}})
    with pytest.raises(
        sps.errors.SpecsolveError, match=rf"^variable '{name}' starts with 'specsolve_', which is reserved"
    ):
        sps.check(spec)


@pytest.mark.parametrize('name', ['value', 'Value', 'VALUE'], ids=str)
def test_a_dimension_named_value_is_refused_at_load(name):
    """A parameter's table and every answer frame hold a dimension's labels beside a column named `value`.

    So a dimension of that name cannot be told apart from the numbers. Query
    engines read column names without case, so `Value` collides as well.
    """
    spec = {
        'dimensions': {name: {'dtype': 'str'}},
        'parameters': {'need': {'dims': [name]}},
        'variables': {'p': {'dims': [name], 'bounds': {'lower': 0, 'upper': 10}}},
        'constraints': {'meet': {'dims': [name], 'expression': 'p >= need'}},
        'objective': {'sense': 'minimize', 'expression': 'sum(p)'},
    }
    with pytest.raises(sps.errors.SpecsolveError, match=rf"^dimension '{name}' has the name of the column 'value'"):
        sps.check(spec)


@pytest.mark.parametrize('door', ['check', 'build', 'solve', 'archive'], ids=str)
def test_every_door_refuses_a_case_pair_rather_than_only_the_front_one(door, tmp_path):
    """A rule only `check` enforced is one `solve` walks past."""
    spec = _named(variables={'P': {'dims': ['t'], 'bounds': {'lower': 0, 'upper': 10}}})
    sources = {'t': range(2), 'load': [1.0, 2.0]}
    call = {
        'check': lambda: sps.check(spec),
        'build': lambda: sps.build(spec, sources),
        'solve': lambda: sps.solve(spec, sources),
        'archive': lambda: sps.solve(spec, sources, archive=tmp_path / 'case.zip'),
    }[door]
    with pytest.raises(sps.errors.SpecsolveError, match='differ only by case'):
        call()
    assert not (tmp_path / 'case.zip').exists(), 'and a refused archive leaves no file behind'


def test_write_suffix_dispatch(dispatch_yaml, dispatch_frame_inputs, tmp_path):
    sources = dispatch_frame_inputs
    out = sps.write(dispatch_yaml, sources, tmp_path / 'm.lp')
    assert out.stat().st_size > 0
    with pytest.raises(ValueError, match='unknown output format'):
        sps.write(dispatch_yaml, sources, tmp_path / 'm.nc')


def test_a_solution_saves_every_kind_it_answered_with(dispatch_solution, dispatch_yaml, tmp_path):
    """Every kind the solve answered with, tidy, as `<kind>/<name>.parquet`."""
    assert dispatch_solution.is_ok
    out = dispatch_solution.save(tmp_path / 'solution')
    assert out == tmp_path / 'solution'
    frame = pl.read_parquet(out / 'primal' / 'p.parquet')
    assert set(frame.columns) == {'snapshot', 'generator', 'value'}
    assert frame.height == dispatch_solution.primal('p').height
    assert {p.stem for p in (out / 'dual').iterdir()} == set(sps.check(dispatch_yaml).constraints), (
        'one dual file per constraint'
    )


def test_a_saved_solution_says_how_it_terminated(dispatch_solution, tmp_path):
    """The record beside the frames, as the row a sweep writes per slice."""
    out = dispatch_solution.save(tmp_path / 'solution')
    record = pl.read_parquet(out / 'record.parquet')
    assert record.columns == [
        'status',
        'termination_condition',
        'objective',
        'has_primal',
        'spec_digest',
        'solved_at',
        'specsolve_run',
        'model_digest',
        'slice_axis',
        'slice',
        'solver',
        'solver_version',
        'solver_options',
        'specsolve_version',
        'mathspec_version',
    ], 'the columns a sweep writes per slice, the slice null'
    assert record.height == 1, 'one solve, one row'
    assert record.row(0, named=True) == {
        'status': dispatch_solution.status,
        'termination_condition': dispatch_solution.termination_condition,
        'objective': dispatch_solution.objective,
        'has_primal': dispatch_solution.has_primal,
        'spec_digest': dispatch_solution.spec_digest,
        'solved_at': dispatch_solution.solved_at,
        'specsolve_run': None,
        'model_digest': dispatch_solution.model_digest(),
        'slice_axis': None,
        'slice': None,
        **dispatch_solution.provenance._asdict(),
    }, 'the row carries what the result itself reports, not a second reading of the solve'


def test_an_export_writes_the_kinds_the_solve_answered_with(tmp_path):
    """An integer variable leaves the duals undefined and the export leaves
    them out; an expression that cannot be evaluated on this data is left out
    the same way, and `expression()` still says why."""
    spec = {
        'dimensions': {'t': {'dtype': 'int'}},
        'parameters': {'load': {'dims': ['t']}, 'scale': {'dims': ['t']}},
        'variables': {'p': {'dims': ['t'], 'bounds': {'lower': 0}, 'domain': 'integer'}},
        'constraints': {'meet': {'dims': ['t'], 'expression': 'p >= load'}},
        'expressions': {'twice': '2 * p', 'ratio': 'p / scale'},
        'objective': {'sense': 'minimize', 'expression': 'sum(p)'},
    }
    sources = {'t': range(2), 'load': [1.5, 2.5], 'scale': pl.DataFrame({'t': [0], 'value': [2.0]})}
    with sps.solve(spec, sources) as result:
        out = result.save(tmp_path)
        with pytest.raises(sps.errors.SpecsolveError):
            result.evaluate('ratio')
        with pytest.raises(sps.errors.SpecsolveError, match='integer'):
            result.to_dataset(kind='dual')
    assert sorted(p.name for p in out.iterdir()) == [
        'expression',
        'format.json',
        'primal',
        'reasons.parquet',
        'record.parquet',
    ], 'no dual/ — there are none to write, and reasons.parquet says so; no activity/, which was not asked for'
    assert [p.name for p in (out / 'expression').iterdir()] == ['twice.parquet'], 'the one that evaluated'
    assert pl.read_parquet(out / 'expression' / 'twice.parquet')['value'].to_list() == [4.0, 6.0], (
        'twice the integer dispatch that meets 1.5 and 2.5'
    )


def test_a_saved_solution_carries_the_activities_it_was_asked_for(dispatch_yaml, dispatch_frame_inputs, tmp_path):
    """An answer asked for its activities saves each row's left-hand side, and loads it back."""
    with sps.solve(dispatch_yaml, dispatch_frame_inputs, outputs={'activity'}) as solution:
        out = solution.save(tmp_path / 'solution')
        constraints = set(sps.check(dispatch_yaml).constraints)
        assert {p.stem for p in (out / 'activity').iterdir()} == constraints, 'one activity file per constraint'
        loaded = sps.load_result(out)
        for name in constraints:
            assert pl.read_parquet(out / 'activity' / f'{name}.parquet').equals(solution.activity(name))
            assert loaded.activity(name).equals(solution.activity(name))


def test_a_saved_solution_says_why_a_kind_is_absent(tmp_path):
    """An absence is a fact about the answer, so it is written down with its reason."""
    spec = {
        'dimensions': {'t': {'dtype': 'int'}},
        'parameters': {'load': {'dims': ['t']}, 'scale': {'dims': ['t']}},
        'variables': {'p': {'dims': ['t'], 'bounds': {'lower': 0}, 'domain': 'integer'}},
        'constraints': {'meet': {'dims': ['t'], 'expression': 'p >= load'}},
        'expressions': {'twice': '2 * p', 'ratio': 'p / scale'},
        'objective': {'sense': 'minimize', 'expression': 'sum(p)'},
    }
    sources = {'t': range(2), 'load': [1.5, 2.5], 'scale': pl.DataFrame({'t': [0], 'value': [2.0]})}
    with sps.solve(spec, sources) as result:
        out = result.save(tmp_path)
        with pytest.raises(sps.errors.SpecsolveError) as no_dual:
            result.dual('meet')
        with pytest.raises(sps.errors.SpecsolveError) as no_ratio:
            result.evaluate('ratio')

    absent = pl.read_parquet(out / 'reasons.parquet')
    assert absent.columns == ['kind', 'name', 'reason'], 'the kind, what is missing under it, and why'
    assert absent.sort('kind', 'name').rows() == [
        ('dual', '', str(no_dual.value)),
        ('expression', 'ratio', str(no_ratio.value)),
    ], 'the whole kind for the duals, one name for the expression, each with the sentence the reader gives'


def test_a_saved_solution_loads_back_as_the_result_it_was(dispatch_solution, dispatch_yaml, tmp_path):
    """Every reader answers what it answered, off the directory rather than a session."""
    loaded = sps.load_result(dispatch_solution.save(tmp_path / 'solution'))

    assert (loaded.status, loaded.termination_condition) == (
        dispatch_solution.status,
        dispatch_solution.termination_condition,
    ), 'the outcome as recorded, on both axes'
    assert loaded.objective == dispatch_solution.objective
    assert loaded.has_primal
    program = sps.check(dispatch_yaml)
    for name in program.variables:
        assert loaded.primal(name).equals(dispatch_solution.primal(name))
    for name in program.constraints:
        assert loaded.dual(name).equals(dispatch_solution.dual(name))


def test_a_loaded_result_gives_the_reason_the_solve_gave(tmp_path):
    """An absence loads back as the sentence, not as an unknown name."""
    spec = {
        'dimensions': {'t': {'dtype': 'int'}},
        'parameters': {'load': {'dims': ['t']}, 'scale': {'dims': ['t']}},
        'variables': {'p': {'dims': ['t'], 'bounds': {'lower': 0}, 'domain': 'integer'}},
        'constraints': {'meet': {'dims': ['t'], 'expression': 'p >= load'}},
        'expressions': {'twice': '2 * p', 'ratio': 'p / scale'},
        'objective': {'sense': 'minimize', 'expression': 'sum(p)'},
    }
    sources = {'t': range(2), 'load': [1.5, 2.5], 'scale': pl.DataFrame({'t': [0], 'value': [2.0]})}
    with sps.solve(spec, sources) as result:
        loaded = sps.load_result(result.save(tmp_path))
        with pytest.raises(sps.errors.SpecsolveError) as no_dual:
            result.dual('meet')
        with pytest.raises(sps.errors.SpecsolveError) as no_ratio:
            result.evaluate('ratio')

    assert loaded.evaluate('twice').equals(pl.DataFrame({'t': [0, 1], 'value': [4.0, 6.0]}))
    with pytest.raises(sps.errors.SpecsolveError, match='integer'):
        loaded.dual('meet')
    with pytest.raises(sps.errors.SpecsolveError) as loaded_no_ratio:
        loaded.evaluate('ratio')
    assert (str(loaded_no_ratio.value), str(no_ratio.value)) == (str(no_ratio.value), str(no_ratio.value)), (
        'the expression names the same reason it named in the process that solved'
    )
    assert 'integer' in str(no_dual.value), 'and the dual refuses for the reason it refused there'


def test_a_solve_that_left_no_values_loads_back_and_still_has_none(tmp_path):
    """A run that did not solve is an answer, and reads back as that answer."""
    with sps.solve(*CASES['INFEASIBLE']) as solution:
        loaded = sps.load_result(solution.save(tmp_path / 'infeasible'))
    assert loaded.termination_condition == 'infeasible'
    assert not loaded.has_primal, 'the record says the solve produced none, so no reader is offered any'
    assert loaded.objective != loaded.objective, 'nan, as the solve reported it'
    with pytest.raises(sps.errors.NoSolutionError, match='infeasible'):
        loaded.primal('p')


def test_a_directory_that_is_not_a_saved_answer_is_refused(tmp_path):
    empty = tmp_path / 'nothing'
    empty.mkdir()
    with pytest.raises(sps.errors.LayoutError, match=r'record\.parquet'):
        sps.load_result(empty)


def test_a_loaded_answer_outlives_the_directory_and_a_scanned_one_does_not(dispatch_solution, tmp_path):
    """`load_result` holds the values when it returns; `scan_result` reads each frame when asked."""
    saved = dispatch_solution.save(tmp_path / 'solution')
    loaded = sps.load_result(saved)
    scanned = sps.scan_result(saved)
    expected = dispatch_solution.primal('p')
    assert scanned.primal('p').equals(expected), 'both read the same answer while the directory is there'
    shutil.rmtree(saved)

    assert loaded.primal('p').equals(expected), 'the loaded answer was in memory before the files went'
    with pytest.raises(FileNotFoundError):
        scanned.primal('p')


def test_a_scanned_answer_reads_its_frames_at_the_call_that_asks(dispatch_solution, tmp_path):
    """A scan returns what the file holds at the read; a load, what it held at the load."""
    saved = dispatch_solution.save(tmp_path / 'solution')
    loaded = sps.load_result(saved)
    scanned = sps.scan_result(saved)
    was = dispatch_solution.primal('p')
    was.with_columns(pl.col('value') * 2).write_parquet(saved / 'primal' / 'p.parquet')

    assert scanned.primal('p')['value'].to_list() == (was['value'] * 2).to_list(), 'the scan reads the file it finds'
    assert loaded.primal('p').equals(was), 'and the loaded answer is the one the load read'


def test_read_back_is_in_label_order_and_stays_there(dispatch_yaml, dispatch_frame_inputs, tmp_path):
    """A read is a join, and a join settles no order — so the read states one.

    Label order is row-major over the coordinate product: `snapshot` varies
    slowest, and within it `generator` follows the declared order.
    """
    sources = dispatch_frame_inputs
    generators = list(sources['p_max']['generator'])
    with sps.solve(dispatch_yaml, sources) as result:
        first = result.primal('p')
        assert first.equals(result.primal('p')), 'a second read agrees, to the row'

        by_declaration = first.with_columns(
            pl.col('generator').replace_strict(generators, range(len(generators))).alias('ord')
        )
        assert by_declaration.equals(by_declaration.sort('snapshot', 'ord'))

        written = [(result.save(tmp_path / f'solution{i}') / 'primal' / 'p.parquet').read_bytes() for i in range(3)]
        assert len(set(written)) == 1, 'the same solution writes the same bytes'


SCALAR_SPEC = {
    'dimensions': {'f': {'dtype': 'str'}},
    'parameters': {'cost': {'dims': ['f']}, 'budget': {'dims': []}},
    'variables': {
        'x': {'dims': ['f'], 'bounds': {'lower': 0, 'upper': 100}},
        'slack': {'dims': [], 'bounds': {'lower': 0, 'upper': 10}},
    },
    'constraints': {'budget_row': {'dims': [], 'expression': 'sum(x, over=f) - slack <= budget'}},
    'expressions': {'price': {'dims': [], 'expression': 'dual(budget_row)'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(x * cost)'},
}


@pytest.mark.parametrize(
    ('read', 'expected'),
    [
        pytest.param(lambda r: r.primal('slack'), 10.0, id='primal'),
        pytest.param(lambda r: r.dual('budget_row'), 2.0, id='dual'),
        pytest.param(lambda r: r.activity('budget_row'), 120.0, id='activity'),
        pytest.param(lambda r: r.evaluate('price'), 2.0, id='an-expression-reading-the-dual'),
    ],
)
def test_a_declaration_with_no_dimensions_reads_back_its_one_value(read, expected):
    """A product over no dimensions has one coordinate, so every reader returns one row.

    On polars 2.0 each came back empty and the expression read 0.0: the reader
    selected the declaration's dims first, which with none is a frame with no
    columns, and polars gives that no rows.
    """
    sources = {'f': ['a', 'b', 'c'], 'cost': {'a': 1.0, 'b': 2.0, 'c': 3.0}, 'budget': 120.0}
    with sps.solve(SCALAR_SPEC, sources, outputs={'activity'}) as result:
        assert read(result).to_dicts() == [{'value': expected}], 'one row, holding the solver value'


def test_a_result_stays_readable_until_it_is_closed(dispatch_yaml, dispatch_frame_inputs):
    """Reading is valid until `close()`, and nothing after it."""
    sources = dispatch_frame_inputs
    result = sps.solve(dispatch_yaml, sources)
    height = result.primal('p').height
    assert height > 0
    assert result.primal('p').height == height, 'still readable, with no close in sight'

    result.close()
    with pytest.raises(sps.errors.SpecsolveError, match='this result was closed'):
        result.primal('p')


def test_a_second_solve_does_not_rewrite_the_first_result(dispatch_yaml, dispatch_frame_inputs):
    """A result reports its own solve, not the engine's latest."""
    key = ['snapshot', 'generator']
    sources = dispatch_frame_inputs
    with sps.build(dispatch_yaml, sources) as model:
        first = model.solve()
        before = first.primal('p').sort(key)
        assert first.is_ok

        built = model._engine._model
        negated = replace(built.handoff, obj=built.handoff.obj.with_columns(-pl.col('coeff')))
        model._engine._built = replace(built, handoff=negated)
        second = model.solve()

        assert not second.primal('p').sort(key).equals(before), 'the second solve really moved'
        assert first.primal('p').sort(key).equals(before), 'and the first still reports its own'
        assert first.objective != pytest.approx(second.objective)


def test_primal_is_a_frame_and_to_pandas_is_the_bridge(dispatch_solution):
    """A frame is the shape results come in; `to_pandas` converts the same table."""
    frame = dispatch_solution.primal('p')
    assert isinstance(frame, pl.DataFrame)
    assert frame.columns == ['snapshot', 'generator', 'value']

    pandas = pytest.importorskip('pandas')
    converted = dispatch_solution.to_pandas('p')
    assert isinstance(converted, pandas.DataFrame)
    assert list(converted.columns) == frame.columns
    assert len(converted) == frame.height
    assert frame['value'].sum() == pytest.approx(converted['value'].sum())


@pytest.mark.parametrize(
    ('absent', 'bridge'),
    [
        pytest.param('pandas', 'to_pandas', id='to_pandas-without-pandas'),
        pytest.param('xarray', 'to_dataarray', id='to_dataarray-without-xarray'),
        pytest.param('xarray', 'to_dataset', id='to_dataset-without-xarray'),
    ],
)
def test_a_bridge_out_names_the_package_to_install(dispatch_solution, absent, bridge):
    """A bridge out of a bare install says which package to add.

    The message says the package is the caller's to install. On an install
    with neither package, `to_dataarray` names pandas, which it reads first.
    """
    named = absent if importlib.util.find_spec('pandas') is not None else 'pandas'
    with (
        mock.patch.dict(sys.modules, {absent: None}),
        pytest.raises(ModuleNotFoundError, match=f'pip install {named}'),
    ):
        getattr(dispatch_solution, bridge)('p')


def test_no_operator_registry_on_this_package():
    """The operator set is closed, so both lanes accept the same language.

    See docs/about/architecture.md, "The expressive ceiling".
    """
    assert not hasattr(sps, 'register')


def test_solution_to_dataarray(dispatch_solution):
    """`to_dataarray` is the bridge to labelled array math."""
    pytest.importorskip('xarray')
    arr = dispatch_solution.to_dataarray('p')
    tidy = dispatch_solution.to_pandas('p')

    assert arr.name == 'p', "named for the variable, not 'value' — the tidy column it came from"
    assert sorted(arr.dims) == ['generator', 'snapshot']
    assert arr.sizes['generator'] == 3
    wind_0 = tidy.query("generator == 'wind' and snapshot == 0")['value'].iloc[0]
    assert float(arr.sel(generator='wind', snapshot=0)) == pytest.approx(wind_0), (
        'the labelled form is the tidy form, indexed'
    )


def test_solution_to_dataset(dispatch_solution):
    """Several variables at once, each keeping its own dims."""
    pytest.importorskip('xarray')
    ds = dispatch_solution.to_dataset('p')
    tidy = dispatch_solution.to_pandas('p')

    assert list(ds.data_vars) == ['p']
    assert sorted(ds['p'].dims) == ['generator', 'snapshot']
    first = tidy.iloc[0]
    assert float(ds['p'].sel(snapshot=first['snapshot'], generator=first['generator'])) == pytest.approx(first['value'])


def test_every_bridge_takes_a_kind(dispatch_solution, dispatch_yaml):
    """`to_pandas`, `to_dataarray` and `to_dataset` take `kind=`, one per call, `primal` by default."""
    pytest.importorskip('xarray')
    constraint = next(iter(sps.check(dispatch_yaml).constraints))
    tidy = dispatch_solution.to_pandas(constraint, 'dual')
    assert tidy['value'].tolist() == dispatch_solution.dual(constraint)['value'].to_list()
    array = dispatch_solution.to_dataarray(constraint, 'dual')
    assert array.name == constraint
    assert set(dispatch_solution.to_dataset(kind='dual').data_vars) == set(sps.check(dispatch_yaml).constraints), (
        'all of one kind by default, as to_dataset() is all of the variables'
    )
    with pytest.raises(sps.errors.SpecsolveError, match='primal, dual, expression'):
        dispatch_solution.to_pandas('p', 'objective')


def test_a_dataset_of_expressions_holds_every_one_this_data_evaluates():
    """`to_dataset(kind='expression')` is every declared expression, each over
    its own dims, and one that fails on this data fails the call the way
    `evaluate` does rather than being left out silently."""
    pytest.importorskip('xarray')
    spec = {**TWO_VARIABLE_SPEC, 'expressions': {'shed_twice': '2 * shed', 'total': 'sum(p, over=generator)'}}
    n = 4
    sources = {
        'p_max': pl.DataFrame({'generator': ['wind', 'gas'], 'value': [100.0, 200.0]}),
        'load': pl.DataFrame({'snapshot': list(range(n)), 'value': np.full(n, 90.0)}),
        'snapshot': range(n),
        'generator': ['wind', 'gas'],
    }
    with sps.solve(spec, sources) as result:
        ds = result.to_dataset(kind='expression')
        assert set(ds.data_vars) == {'shed_twice', 'total'}, 'every declared expression, none named'
        assert list(ds['total'].dims) == ['snapshot'], 'each over its own dims'
        assert set(result.to_dataset('total', kind='expression').data_vars) == {'total'}, 'named ones only'


TWO_VARIABLE_SPEC = {
    'dimensions': {'snapshot': {'dtype': 'int'}, 'generator': {'dtype': 'str'}},
    'parameters': {'p_max': {'dims': ['generator']}, 'load': {'dims': ['snapshot']}},
    'variables': {
        'p': {'dims': ['snapshot', 'generator'], 'bounds': {'lower': 0, 'upper': 'p_max'}},
        'shed': {'dims': ['snapshot'], 'bounds': {'lower': 0}},
    },
    'constraints': {
        'balance': {
            'dims': ['snapshot'],
            'expression': 'sum(p, over=generator) + shed == load',
        }
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(shed)'},
}


def test_to_dataset_defaults_to_every_variable():
    """`to_dataset()` with no names is every variable."""
    pytest.importorskip('xarray')
    n = 4
    sources = {
        'p_max': pl.DataFrame({'generator': ['wind', 'gas'], 'value': [100.0, 200.0]}),
        'load': pl.DataFrame({'snapshot': list(range(n)), 'value': np.full(n, 90.0)}),
    }

    with sps.solve(TWO_VARIABLE_SPEC, sources | {'snapshot': range(n), 'generator': ['wind', 'gas']}) as result:
        ds = result.to_dataset()
        subset = result.to_dataset('shed')

    assert set(ds.data_vars) == {'p', 'shed'}
    assert sorted(ds['p'].dims) == ['generator', 'snapshot']
    assert list(ds['shed'].dims) == ['snapshot'], 'each variable keeps its own dims'
    assert set(subset.data_vars) == {'shed'}


@pytest.mark.parametrize(
    'raw',
    [
        pytest.param({'dimensionz': {}}, id='unknown-key'),
        pytest.param({'dimensions': {'g': {'dtype': 'complex'}}}, id='bad-dtype'),
        pytest.param({'version': 99}, id='unknown-version'),
        pytest.param(
            {
                'dimensions': {'g': {'dtype': 'str'}},
                'constraints': {'c': {'dims': ['g'], 'expression': 'nope <= 1'}},
            },
            id='undeclared-name',
        ),
    ],
)
def test_a_wrong_model_raises_one_tree(raw: dict[str, object], tmp_path):
    """Every documented door answers with `SpecsolveError` (#527).

    `Spec.__init__` is not a door: defining one would run every after-validator
    twice, the first time with no context.
    """
    doors = {
        'to_spec': lambda: to_spec(raw),
        'sps.check': lambda: sps.check(raw),
        'sps.solve': lambda: sps.solve(raw, {}),
        'sps.write': lambda: sps.write(raw, {}, str(tmp_path / 'm.lp')),
        'Spec.model_validate': lambda: Spec.model_validate(raw),
    }
    for door, call in doors.items():
        with pytest.raises(sps.errors.SpecsolveError) as ei:
            call()
        assert 'errors.pydantic.dev' not in str(ei.value), f"{door} leaks pydantic's envelope"


def test_a_closed_result_says_it_was_closed(dispatch_yaml, dispatch_frame_inputs):
    """After `close` the readers say so; frames read before it stay valid."""
    sources = dispatch_frame_inputs
    sol = sps.solve(dispatch_yaml, sources)
    frame = sol.primal('p')
    objective = sol.objective
    sol.close()

    assert frame.height > 0, 'a frame read before the close is its own data'
    assert sol.objective == objective, 'and the outcome needs no model to report'
    for read in (lambda: sol.primal('p'), lambda: sol.dual('power_balance')):
        with pytest.raises(sps.errors.SpecsolveError, match='this result was closed'):
            read()


def test_check_catches_a_dim_error_with_no_sources_bound():
    """`check` reaches a dim rule without a byte of data."""
    raw = override(
        raw_of(EXAMPLES_DIR / 'dispatch.yaml'),
        **{'constraints.stray': {'dims': ['snapshot'], 'expression': 'p <= p_max'}},
    )
    with pytest.raises(DimensionError):
        sps.check(raw)


# ---------------------------------------------------------------------------
# tidy
# ---------------------------------------------------------------------------


def test_tidy_returns_one_table_per_declared_name_in_its_canonical_columns(dispatch_yaml, dispatch_frame_inputs):
    """A dimension is its labels and their positions, a parameter its dims and `value`."""
    generators = ['gas', 'wind', 'solar']
    sources = {
        **dispatch_frame_inputs,
        'generator': pl.DataFrame({'generator': generators, 'colour': ['a', 'b', 'c']}),
        'p_max': dict(zip(generators, DISPATCH_P_MAX, strict=True)),
        'cost': pl.DataFrame({'generator': generators, 'value': DISPATCH_COST, 'note': ['x', 'y', 'z']}),
    }
    tables = sps.tidy(dispatch_yaml, sources)

    assert sorted(tables) == ['cost', 'generator', 'load', 'p_max', 'snapshot'], 'one table per declared name'
    assert tables['generator'].columns == ['generator', 'specsolve_position'], 'the extra column is gone'
    assert tables['generator']['generator'].to_list() == generators, 'the labels keep the order they arrived in'
    assert tables['generator']['specsolve_position'].to_list() == [0, 1, 2], 'positions count from 0 in that order'
    assert tables['generator'].schema['specsolve_position'] == pl.Int64, 'the position is an Int64'
    assert tables['cost'].columns == ['generator', 'value'], 'a parameter is its dims and value, nothing else'
    assert tables['p_max'].columns == ['generator', 'value'], 'a plain-Python shape comes back as the same table'


def test_what_tidy_returns_solves_as_the_sources_did(dispatch_yaml, dispatch_frame_inputs):
    """The tidy tables are sources too, so an archive of them asks the same question."""
    with (
        sps.solve(dispatch_yaml, dispatch_frame_inputs) as direct,
        sps.solve(dispatch_yaml, sps.tidy(dispatch_yaml, dispatch_frame_inputs)) as tidied,
    ):
        assert tidied.objective == pytest.approx(direct.objective, rel=1e-9), 'the tidy tables build the same model'
