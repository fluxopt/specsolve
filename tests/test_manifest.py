"""A manifest: specsolve's arguments for several runs, read from one YAML file."""

from __future__ import annotations

import inspect
import shutil
import textwrap
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields
from pathlib import Path

import polars as pl
import pytest

import specsolve as sps
from specsolve import manifest as manifest_module
from specsolve.errors import DataError, SpecsolveError
from specsolve.manifest import AXES, OWN_KEYS, RUN_KEYS, SWEEP_ONLY, TOP_KEYS, read_sources
from specsolve.strategy import solve_over

REPO = Path(__file__).resolve().parent.parent
EXAMPLE = REPO / 'examples' / 'manifest'
GENERATORS = ['wind', 'solar', 'gas']


def _written(directory: Path, text: str, name: str = 'specsolve.yaml') -> Path:
    path = directory / name
    path.write_text(textwrap.dedent(text))
    return path


@pytest.fixture
def data(tmp_path: Path) -> Path:
    """The dispatch model's sources as parquet and CSV files, under ``tmp_path/data``."""
    shutil.copy(REPO / 'examples' / 'dispatch.yaml', tmp_path / 'model.yaml')
    base = tmp_path / 'data'
    base.mkdir()
    pl.DataFrame({'generator': GENERATORS}).write_parquet(base / 'generator.parquet')
    pl.DataFrame({'generator': GENERATORS, 'value': [80.0, 40.0, 200.0]}).write_parquet(base / 'p_max.parquet')
    pl.DataFrame({'generator': GENERATORS, 'value': [10.0, 25.0, 50.0]}).write_parquet(base / 'cost.parquet')
    pl.DataFrame({'snapshot': [0, 1, 2, 3]}).write_csv(base / 'snapshot.csv')
    pl.DataFrame({'snapshot': [0, 1, 2, 3], 'value': [60.0, 110.0, 170.0, 90.0]}).write_csv(base / 'load.csv')
    (tmp_path / 'dear').mkdir()
    dear = tmp_path / 'dear' / 'cost.parquet'
    pl.DataFrame({'generator': GENERATORS, 'value': [10.0, 25.0, 90.0]}).write_parquet(dear)
    pl.DataFrame(
        {'scenario': ['low'] * 4 + ['high'] * 4, 'snapshot': [0, 1, 2, 3] * 2, 'value': [60.0, 110.0, 170.0, 90.0] * 2}
    ).write_parquet(tmp_path / 'load.parquet')
    return tmp_path


def test_top_level_keys_are_every_runs_defaults_and_a_runs_own_key_replaces_one(data: Path) -> None:
    loaded = sps.load_manifest(
        _written(
            data,
            """
            manifest: 1
            spec: model.yaml
            sources: [data/]
            solver_options: {time_limit: 600}
            runs:
              base: {}
              quick:
                solver_options: {time_limit: 5}
            """,
        )
    )
    assert list(loaded.runs) == ['base', 'quick'], 'runs keep the order the file lists them in'
    assert loaded.runs['base'].options == {'solver_options': {'time_limit': 600}}, 'the default reaches a run'
    assert loaded.runs['quick'].options == {'solver_options': {'time_limit': 5}}, 'the run replaces the whole mapping'
    assert loaded.runs['base'].spec == data / 'model.yaml'


def test_from_inherits_another_runs_resolved_settings_and_a_runs_sources_replace_them(data: Path) -> None:
    loaded = sps.load_manifest(
        _written(
            data,
            """
            manifest: 1
            spec: model.yaml
            sources: [data/]
            runs:
              dear:
                sources: [data/, dear/cost.parquet]
                record_options: [duals]
              scenarios:
                from: dear
                sources: [data/, load.parquet]
                axis: {EachCoordinate: {dim: scenario}}
              inherits:
                from: dear
            """,
        )
    )
    scenarios = loaded.runs['scenarios']
    assert scenarios.sources == (data / 'data', data / 'load.parquet'), (
        "the run's own locations, and none of its parent's or the defaults'"
    )
    assert loaded.runs['inherits'].sources == (data / 'data', data / 'dear' / 'cost.parquet'), (
        "a run that sets no sources reads its parent's"
    )
    assert scenarios.options == {'record_options': ['duals']}, 'what the parent set reaches the run'
    assert scenarios.axis == sps.EachCoordinate('scenario')


def test_a_later_location_replaces_an_earlier_table_of_the_same_name(data: Path) -> None:
    sources = read_sources((data / 'data', data / 'dear' / 'cost.parquet'))
    assert sources['cost'] == data / 'dear' / 'cost.parquet', 'the run location replaced the default table'
    assert sorted(sources) == ['cost', 'generator', 'load', 'p_max', 'snapshot'], 'a directory holds one table per file'
    assert sources['p_max'] == data / 'data' / 'p_max.parquet', 'a parquet file stays a path'
    assert isinstance(sources['load'], pl.DataFrame)


def test_paths_are_relative_to_the_manifest_rather_than_the_working_directory(
    data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nested = data / 'studies'
    nested.mkdir()
    path = _written(
        nested,
        """
        manifest: 1
        spec: ../model.yaml
        sources: [../data/]
        archive: runs/
        runs:
          base: {}
        """,
    )
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    run = sps.load_manifest(path).runs['base']
    assert run.spec == nested / '../model.yaml'
    assert run.archive == nested / 'runs' / 'base', 'each run archives into a directory named after it'
    assert run.solve().termination_condition == 'optimal'


def test_arguments_are_exactly_the_call_written_by_hand(data: Path) -> None:
    path = _written(
        data,
        """
        manifest: 1
        spec: model.yaml
        archive: runs/
        runs:
          scenarios:
            sources: [data/p_max.parquet, data/cost.parquet, data/generator.parquet, data/snapshot.csv, load.parquet]
            axis: {EachCoordinate: {dim: scenario}}
            keep: nothing
        """,
    )
    run = sps.load_manifest(path).runs['scenarios']
    arguments = run.arguments
    snapshot, load = arguments['sources'].pop('snapshot'), arguments['sources'].pop('load')
    assert arguments == {
        'spec': data / 'model.yaml',
        'sources': {
            'p_max': data / 'data' / 'p_max.parquet',
            'cost': data / 'data' / 'cost.parquet',
            'generator': data / 'data' / 'generator.parquet',
        },
        'axis': sps.EachCoordinate('scenario'),
        'archive': str(data / 'runs' / 'scenarios'),
        'keep': 'nothing',
    }, 'the arguments are the keyword arguments of solve_over, and nothing else'
    assert pl.read_csv(data / 'data' / 'snapshot.csv').equals(snapshot)
    assert load == data / 'load.parquet'

    by_hand = solve_over(
        data / 'model.yaml',
        {
            **{name: data / 'data' / f'{name}.parquet' for name in ('p_max', 'cost', 'generator')},
            'snapshot': [0, 1, 2, 3],
            'load': data / 'load.parquet',
        },
        sps.EachCoordinate('scenario'),
        keep='nothing',
    )
    from_the_manifest = run.solve(archive=None)
    assert from_the_manifest.record['objective'].to_list() == by_hand.record['objective'].to_list()


def test_workers_is_a_spawn_pool_that_solve_shuts_down(data: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _written(
        data,
        """
        manifest: 1
        spec: model.yaml
        sources: [data/, load.parquet]
        runs:
          scenarios:
            axis: {EachCoordinate: {dim: scenario}}
            workers: 2
        """,
    )
    run = sps.load_manifest(path).runs['scenarios']
    pool = run.arguments['executor']
    assert isinstance(pool, ProcessPoolExecutor)
    assert pool._mp_context.get_start_method() == 'spawn', 'a forked worker hangs'

    pools: list[ProcessPoolExecutor] = []
    real = manifest_module.ProcessPoolExecutor

    def recorded(*args: object, **kwargs: object) -> ProcessPoolExecutor:
        pools.append(real(*args, **kwargs))
        return pools[-1]

    monkeypatch.setattr(manifest_module, 'ProcessPoolExecutor', recorded)
    sweep = run.solve()
    assert sweep.record['termination_condition'].to_list() == ['optimal', 'optimal'], 'both slices solved in the pool'
    assert pools[0]._shutdown_thread, 'the pool the run started is shut down'


def test_a_workbook_is_one_table_per_sheet() -> None:
    pytest.importorskip('fastexcel')
    sources = read_sources((EXAMPLE / 'data' / 'base.xlsx',))
    assert sorted(sources) == ['cost', 'generator', 'p_max'], 'each sheet is a table named after it'


def test_a_workbook_without_fastexcel_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    real = manifest_module.importlib.util.find_spec
    monkeypatch.setattr(
        manifest_module.importlib.util, 'find_spec', lambda name: None if name == 'fastexcel' else real(name)
    )
    with pytest.raises(SpecsolveError, match=r"pip install 'specsolve\[cli\]'"):
        read_sources((EXAMPLE / 'data' / 'base.xlsx',))


@pytest.mark.parametrize(
    ('make', 'match'),
    [
        pytest.param(lambda d: d / 'absent.parquet', 'does not exist', id='missing-location'),
        pytest.param(lambda d: d / 'model.yaml', 'not a source location', id='unknown-form'),
        pytest.param(
            lambda d: (
                pl.DataFrame({'snapshot': [0]}).write_parquet(d / 'data' / 'snapshot.parquet'),
                d / 'data',
            )[1],
            'holds snapshot twice',
            id='one-name-twice-in-a-directory',
        ),
    ],
)
def test_a_location_no_reader_takes_is_refused(data: Path, make, match: str) -> None:
    with pytest.raises(DataError, match=match):
        read_sources((make(data),))


BASE = """
manifest: 1
spec: model.yaml
sources: [data/]
"""


@pytest.mark.parametrize(
    ('text', 'match'),
    [
        pytest.param(
            BASE + 'solver: highs\nruns: {base: {}}\n',
            r"unknown key 'solver' in the top level\. Did you mean 'solver_name'\? Valid keys: archive, axis",
            id='unknown-top-level-key',
        ),
        pytest.param(
            BASE + 'runs:\n  base: {scenario: high}\n',
            r"unknown key 'scenario' in runs\.base\. Valid keys: archive, axis, carry, from",
            id='unknown-run-key',
        ),
        pytest.param(BASE + 'runs:\n  base: {}\n  base: {}\n', 'written twice', id='duplicate-key'),
        pytest.param('spec: model.yaml\nruns: {base: {}}\n', r'manifest: Field required', id='no-format-version'),
        pytest.param(
            'manifest: 2\nspec: model.yaml\nruns: {base: {}}\n', r'manifest: Input should be 1', id='format-2'
        ),
        pytest.param(
            BASE + 'runs:\n  a: {from: c}\n', r"from: 'c' names no run\. Runs: a\.|Did you mean", id='from-unknown'
        ),
        pytest.param(
            BASE + 'runs:\n  a: {from: b}\n  b: {from: a}\n', r'from makes a cycle, a -> b -> a', id='from-cycle'
        ),
        pytest.param('manifest: 1\nruns: {base: {}}\n', r'runs\.base: no spec', id='no-spec'),
        pytest.param(
            BASE + 'runs:\n  base: {keep_windows: true}\n',
            r'runs\.base: keep_windows only solve_over takes',
            id='sweep-key-without-axis',
        ),
        pytest.param(
            BASE + 'runs:\n  base: {carry: {soc_initial: soc}}\n',
            r'runs\.base: carry only solve_over takes',
            id='carry-without-axis',
        ),
        pytest.param(
            BASE + 'runs:\n  base: {workers: 4}\n',
            r'runs\.base: workers only solve_over takes',
            id='workers-without-axis',
        ),
        pytest.param(
            BASE + 'runs:\n  base: {workers: 0}\n', r'workers: Input should be greater than 0', id='no-workers'
        ),
        pytest.param(
            BASE + 'runs:\n  base: {axis: {EachScenario: {dim: s}}}\n',
            r'axis takes one class and its arguments',
            id='unknown-axis-class',
        ),
        pytest.param(
            BASE + 'runs:\n  base: {axis: {EachCoordinate: {dim: s, steps: 2}}}\n',
            r'axis EachCoordinate takes dim, not steps',
            id='unknown-axis-argument',
        ),
        pytest.param(
            BASE + 'runs:\n  base: {axis: {EachWindow: {dim: snapshot, steps: 2}}}\n',
            r'axis EachWindow needs into, lookahead',
            id='missing-axis-argument',
        ),
        pytest.param(
            BASE + 'runs:\n  base: {axis: {EachWindow: {dim: snapshot, steps: 2, lookahead: -1, into: t}}}\n',
            r'axis EachWindow: lookahead=-1 is negative',
            id='axis-refuses-its-value',
        ),
        pytest.param(BASE + 'runs:\n  a/b: {}\n', r'cannot hold a path separator', id='run-name-with-a-separator'),
    ],
)
def test_a_manifest_the_rules_refuse_is_refused_at_load_naming_the_file(data: Path, text: str, match: str) -> None:
    path = _written(data, text)
    with pytest.raises(SpecsolveError, match=match) as raised:
        sps.load_manifest(path)
    assert str(path) in str(raised.value), 'the error names the file'


def test_every_manifest_key_is_an_argument_of_solve_solve_over_or_the_manifests_own() -> None:
    """A rename in specsolve fails here rather than leaving the manifest a key that reaches nothing."""
    verbs = set(inspect.signature(sps.solve).parameters) | set(inspect.signature(solve_over).parameters)
    keys = set(RUN_KEYS) | set(TOP_KEYS)
    assert not keys - verbs - set(OWN_KEYS), 'every key a verb does not take is one of the manifest-own keys'
    assert set(OWN_KEYS) <= keys, 'every manifest-own key is a key of the file'
    assert {'axis', 'carry', 'key_name', 'keep', 'keep_windows', 'executor'} <= SWEEP_ONLY, (
        'what only solve_over takes is read off the two signatures'
    )


@pytest.mark.parametrize('name', sorted(AXES))
def test_an_axis_takes_its_constructor_arguments_by_name(name: str) -> None:
    cls = AXES[name]
    assert cls.__name__ == name and getattr(sps, name) is cls
    assert set(inspect.signature(cls).parameters) == {held.name for held in fields(cls)}, (
        'the manifest names the fields, so they have to be what the constructor takes'
    )


def test_the_example_manifest_loads() -> None:
    loaded = sps.load_manifest(EXAMPLE / 'specsolve.yaml')
    assert {name: run.is_sweep for name, run in loaded.runs.items()} == {
        'base': False,
        'high_gas': False,
        'scenarios': True,
    }, 'the example has two solves and a sweep'
    assert loaded.runs['scenarios'].workers == 2
