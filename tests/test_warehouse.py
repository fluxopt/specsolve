"""A folder of archives is one warehouse: every table globs to one schema, and the warehouse joins hold.

The folder holds what ``docs/howto/warehouse.md`` writes into it: two single
solves of one spec over different data, a scenario sweep, and a rolling
horizon archived with and without its windows. The spec carries one relation
of each kind a warehouse joins through. Every claim the page makes is asserted
here, and every block on the page runs against this folder.
"""

from __future__ import annotations

import contextlib
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
import pytest

import specsolve as sps
from specsolve.archive_layout import _with_run

if TYPE_CHECKING:
    from collections.abc import Iterator

PAGE = Path(__file__).resolve().parent.parent / 'docs' / 'howto' / 'warehouse.md'

#: One relation of each kind: keyed by one dimension (``sited``), keyed by a
#: pair (``zone_of``), bare (``connection``), and two roles over one dimension
#: (``ends``). ``reach`` sums through the bare relation, so a generator
#: connected to two buses counts at both.
SPEC: dict[str, Any] = {
    'dimensions': {
        't': {'dtype': 'int'},
        'generator': {'dtype': 'str'},
        'bus': {'dtype': 'str'},
        'line': {'dtype': 'str'},
        'zone': {'dtype': 'str'},
    },
    'relations': {
        'sited': {'key': 'generator', 'values': 'bus'},
        'zone_of': {'key': ['generator', 't'], 'values': 'zone'},
        'connection': {'key': ['generator', 'bus']},
        'ends': {'key': 'line', 'values': {'bus0': 'bus', 'bus1': 'bus'}},
    },
    'parameters': {
        'p_max': {'dims': ['generator']},
        'cost': {'dims': ['generator']},
        'load': {'dims': ['t', 'bus']},
        'zone_cap': {'dims': ['zone']},
    },
    'variables': {
        'p': {'dims': ['t', 'generator'], 'bounds': {'lower': 0, 'upper': 'p_max'}},
        'f': {'dims': ['t', 'line'], 'bounds': {'lower': -50, 'upper': 50}},
    },
    'constraints': {
        'balance': {
            'dims': ['t', 'bus'],
            'expression': 'sum(p, by=sited, over=generator, into=bus)'
            ' + sum(f, by=ends, over=line, into=bus1) - sum(f, by=ends, over=line, into=bus0) == load',
        },
        'zonal': {'dims': ['t', 'zone'], 'expression': 'sum(p, by=zone_of, over=generator, into=zone) <= zone_cap'},
    },
    'expressions': {'reach': {'dims': ['t', 'bus'], 'expression': 'sum(p, by=connection, over=generator, into=bus)'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(p * cost)'},
}

GENERATORS = ['wind', 'gas', 'coal']
BUSES = ['north', 'south']


def _sources(time: str = 't', periods: int = 4, scale: float = 1.0) -> dict[str, object]:
    """``SPEC``'s data over *periods* steps of *time*, the load scaled by *scale*.

    ``wind`` is sited at ``north`` and connected to both buses; ``coal``
    changes zone every step.
    """
    steps = range(periods)
    sources: dict[str, object] = {
        'generator': GENERATORS,
        'bus': BUSES,
        'line': ['l1'],
        'zone': ['z1', 'z2'],
        'sited': pl.DataFrame({'generator': GENERATORS, 'bus': ['north', 'south', 'south']}),
        'zone_of': pl.DataFrame(
            {
                'generator': GENERATORS * periods,
                time: [step for step in steps for _ in GENERATORS],
                'zone': [zone for step in steps for zone in ('z1', 'z2', 'z1' if step % 2 else 'z2')],
            }
        ),
        'connection': pl.DataFrame({'generator': ['wind', 'wind', 'gas', 'coal'], 'bus': ['north', 'south'] * 2}),
        'ends': pl.DataFrame({'line': ['l1'], 'bus0': ['north'], 'bus1': ['south']}),
        'p_max': pl.DataFrame({'generator': GENERATORS, 'value': [15.0, 40.0, 40.0]}),
        'cost': pl.DataFrame({'generator': GENERATORS, 'value': [0.0, 50.0, 30.0]}),
        'load': pl.DataFrame(
            {
                time: [step for step in steps for _ in BUSES],
                'bus': BUSES * periods,
                'value': [scale * (10.0 + step) for step in steps for _ in BUSES],
            }
        ),
        'zone_cap': pl.DataFrame({'zone': ['z1', 'z2'], 'value': [100.0, 100.0]}),
    }
    if time == 't':
        sources['t'] = list(steps)
    return sources


def _by_scenario() -> dict[str, object]:
    """``_sources`` with a ``scenario`` column on ``load``, which the spec does not declare."""
    scaled = [
        pl.DataFrame(_sources(scale=scale)['load']).with_columns(scenario=pl.lit(name))
        for name, scale in (('low', 0.5), ('high', 1.5))
    ]
    return {**_sources(), 'load': pl.concat(scaled)}


#: What the page's first block takes from its reader: a spec, and data for each run.
GIVEN = {
    'spec': SPEC,
    'sources': _sources(),
    'peak': _sources(scale=1.5),
    'by_scenario': _by_scenario(),
    'by_snapshot': _sources('snapshot', periods=8),
}

#: The run names the page writes, in the order a sorted glob returns them.
RUNS = ['base', 'peak', 'rolling', 'rolling-kept', 'scenarios']

#: A fence the page shows and the suite does not run, with the reason on the
#: line before it.
_UNRUN = '<!-- warehouse: not run'

_FENCE = re.compile(
    r'^(?P<marker><!--[^\n]*-->\n)?```(?P<lang>python|sql)\n(?P<code>.*?)^```$', re.DOTALL | re.MULTILINE
)


def _blocks() -> list[tuple[str, str, bool]]:
    """``(language, code, runs)`` for each python and sql fence on the page, in page order."""
    return [
        (match['lang'], match['code'], not (match['marker'] or '').startswith(_UNRUN))
        for match in _FENCE.finditer(PAGE.read_text())
    ]


def _run_page(under: Path) -> dict[str, Any]:
    """Every block on the page that runs, in order, in one namespace, from *under*.

    A python block runs as written. A sql block runs through polars' own SQL
    engine, which reads the same ``read_parquet('<glob>')`` a DuckDB query does.
    """
    namespace: dict[str, Any] = {'__name__': '__warehouse__', **GIVEN}
    with contextlib.chdir(under):
        for language, code, runs in _blocks():
            if not runs:
                continue
            if language == 'python':
                exec(compile(code, str(PAGE), 'exec'), namespace)
            else:
                namespace.setdefault('sql_results', []).append(pl.SQLContext().execute(code, eager=True))
    return namespace


@pytest.fixture(scope='module')
def page(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Path, dict[str, Any]]]:
    """The folder the page writes, under a fresh directory, and the namespace its blocks leave."""
    under = tmp_path_factory.mktemp('warehouse')
    yield under / 'runs', _run_page(under)


@pytest.fixture(scope='module')
def runs(page: tuple[Path, dict[str, Any]]) -> Path:
    return page[0]


def test_the_page_writes_one_folder_of_every_kind_of_run(runs: Path) -> None:
    assert sorted(path.name for path in runs.iterdir()) == RUNS, (
        'two single solves, a scenario sweep, and a rolling horizon with and without its windows'
    )


def test_every_block_on_the_page_runs_but_the_ones_it_says_do_not(page: tuple[Path, dict[str, Any]]) -> None:
    unrun = [code.splitlines()[0] for _, code, runs in _blocks() if not runs]
    assert len(unrun) == 2, f'only the DuckDB-only and Delta Lake blocks are shown and not run: {unrun}'


def test_the_page_names_the_input_that_moved_between_two_runs(page: tuple[Path, dict[str, Any]]) -> None:
    _, namespace = page
    assert namespace['moved']['source'].to_list() == ['load'], 'base and peak differ in their load alone'
    assert namespace['sql_results'], 'the sql blocks ran'


# ---------------------------------------------------------------------------
# one glob, one schema
# ---------------------------------------------------------------------------

#: Every table a warehouse globs that has one schema across every kind of run.
STRICT = {
    'answer/record.parquet': {
        'status': pl.String,
        'termination_condition': pl.String,
        'objective': pl.Float64,
        'has_primal': pl.Boolean,
        'spec_digest': pl.String,
        'solved_at': pl.Datetime('us', 'UTC'),
        'specsolve_run': pl.String,
        'model_digest': pl.String,
        'slice_axis': pl.String,
        'slice': pl.String,
        'solver': pl.String,
        'solver_version': pl.String,
        'solver_options': pl.String,
        'specsolve_version': pl.String,
        'mathspec_version': pl.String,
    },
    'answer/metrics.parquet': {
        'columns': pl.Int64,
        'rows': pl.Int64,
        'nonzeros': pl.Int64,
        'solves': pl.Int64,
        'loads': pl.Int64,
        'attach_seconds': pl.Float64,
        'build_seconds': pl.Float64,
        'handoff_seconds': pl.Float64,
        'solve_seconds': pl.Float64,
        'write_seconds': pl.Float64,
        'specsolve_run': pl.String,
        'slice_axis': pl.String,
        'slice': pl.String,
    },
    'sources.parquet': {'specsolve_run': pl.String, 'source': pl.String, 'digest': pl.String},
    'catalog.parquet': {
        'specsolve_run': pl.String,
        'path': pl.String,
        'name': pl.String,
        'kind': pl.String,
        'description': pl.String,
        'dtype': pl.String,
        'column': pl.String,
        'dim': pl.String,
    },
    'sources/generator.parquet': {'generator': pl.String, 'specsolve_position': pl.Int64, 'specsolve_run': pl.String},
    'sources/p_max.parquet': {'generator': pl.String, 'value': pl.Float64, 'specsolve_run': pl.String},
    'sources/sited.parquet': {'generator': pl.String, 'bus': pl.String, 'specsolve_run': pl.String},
    'sources/connection.parquet': {'generator': pl.String, 'bus': pl.String, 'specsolve_run': pl.String},
    'sources/ends.parquet': {'line': pl.String, 'bus0': pl.String, 'bus1': pl.String, 'specsolve_run': pl.String},
}


@pytest.mark.parametrize('member', sorted(STRICT), ids=str)
def test_one_glob_reads_every_run_under_one_strict_schema(member: str, runs: Path) -> None:
    """A column added, dropped or retyped in one kind of run fails the read, as it fails a warehouse."""
    table = pl.scan_parquet(runs / '*' / member, schema=STRICT[member]).collect()
    assert table['specsolve_run'].unique().sort().to_list() == RUNS, 'every run is in the one table'


#: ``answer/primal/p.parquet`` across the folder: the scenario sweep adds its
#: key, and the rolling horizon writes the swept ``snapshot`` where a single
#: solve writes ``t``.
P_UNION = {
    'scenario': pl.String,
    't': pl.Int64,
    'snapshot': pl.Int64,
    'generator': pl.String,
    'value': pl.Float64,
    'specsolve_run': pl.String,
}


def _value_table(runs: Path, name: str = 'p') -> pl.DataFrame:
    """One answer table across every run, unioned by column name."""
    files = sorted(runs.glob(f'*/answer/primal/{name}.parquet'))
    return pl.concat([pl.read_parquet(file) for file in files], how='diagonal')


def test_a_value_table_globs_only_by_name_across_kinds_of_run(runs: Path) -> None:
    """A strict glob of a value table is refused, and a union by name has exactly the columns the runs add."""
    with pytest.raises(pl.exceptions.SchemaError, match='extra column'):
        pl.read_parquet(runs / '*' / 'answer' / 'primal' / 'p.parquet')
    union = _value_table(runs)
    assert union.select(sorted(union.columns)).schema == pl.Schema(dict(sorted(P_UNION.items()))), (
        'the union of p is t, generator and value, plus scenario from the sweep and snapshot from the rolling horizon'
    )


# ---------------------------------------------------------------------------
# the warehouse joins
# ---------------------------------------------------------------------------


def _stacked(runs: Path, member: str) -> pl.DataFrame:
    return pl.read_parquet(runs / '*' / member)


def test_a_value_table_joins_its_dimension_tables_on_run_and_label(runs: Path) -> None:
    """Every row of every run finds its label in its own run's table, and only there."""
    p = _value_table(runs)
    generators = _stacked(runs, 'sources/generator.parquet')
    assert generators.select('specsolve_run', 'generator').is_duplicated().sum() == 0, 'run and label is a key'
    assert generators['generator'].is_duplicated().any(), 'and the label alone is not one across runs'
    stray = p.join(generators, on=['specsolve_run', 'generator'], how='anti')
    assert stray.is_empty(), 'every generator of the answer is a label of its run'

    held = sorted(runs.glob('*/sources/t.parquet'))
    steps = pl.concat([pl.read_parquet(file) for file in held])
    over_t = p.filter(pl.col('t').is_not_null())
    assert over_t.join(steps, on=['specsolve_run', 't'], how='anti').is_empty(), 'every step is a label of its run'
    assert sorted(file.parent.parent.name for file in held) == ['base', 'peak', 'scenarios'], (
        'a rolling horizon holds no index of t, which each window numbers for itself'
    )


def test_a_value_table_joins_the_record_on_the_run(runs: Path) -> None:
    """A run's objective reaches each of its rows, and a sweep's through its slice."""
    record = _stacked(runs, 'answer/record.parquet')
    single = record.filter(pl.col('slice').is_null())
    joined = _value_table(runs).join(single.select('specsolve_run', 'objective'), on='specsolve_run')
    assert sorted(joined['specsolve_run'].unique()) == ['base', 'peak'], 'a single solve has one record row'
    sweeps = record.filter(pl.col('slice').is_not_null()).group_by('specsolve_run').agg(pl.col('slice').sort())
    assert sorted(sweeps.rows()) == [
        ('rolling', ['0', '4']),
        ('rolling-kept', ['0', '4']),
        ('scenarios', ['high', 'low']),
    ], 'a sweep has one record row per slice, named in slice as text'


def test_a_relation_keyed_by_one_dimension_joins_on_run_and_its_key(runs: Path) -> None:
    """Output per bus, through ``sited``, balances the load less what the line moves."""
    sited = _stacked(runs, 'sources/sited.parquet')
    p = pl.read_parquet(runs / 'base' / 'answer' / 'primal' / 'p.parquet')
    per_bus = p.join(sited, on=['specsolve_run', 'generator']).group_by('t', 'bus').agg(pl.col('value').sum())
    flow = pl.read_parquet(runs / 'base' / 'answer' / 'primal' / 'f.parquet')
    load = pl.read_parquet(runs / 'base' / 'sources' / 'load.parquet')
    balance = (
        load.join(per_bus, on=['t', 'bus'], suffix='_made')
        .join(flow.select('t', pl.col('value').alias('flow')), on='t')
        .with_columns(moved=pl.when(pl.col('bus') == 'north').then(-pl.col('flow')).otherwise(pl.col('flow')))
    )
    assert (balance['value_made'] + balance['moved'] - balance['value']).abs().max() < 1e-9, (
        'the join through sited reproduces the balance the model held, to solver precision'
    )


def test_a_total_through_a_bare_relation_counts_a_member_once_per_row_it_is_related_to(runs: Path) -> None:
    """``reach`` gives each bus every connected generator's output, so its total counts ``wind`` twice."""
    p = pl.read_parquet(runs / 'base' / 'answer' / 'primal' / 'p.parquet')
    reach = pl.read_parquet(runs / 'base' / 'answer' / 'expression' / 'reach.parquet')
    wind = p.filter(pl.col('generator') == 'wind')['value'].sum()
    assert wind > 0, 'wind runs, so counting it twice shows'
    assert reach['value'].sum() == pytest.approx(p['value'].sum() + wind, rel=1e-9), (
        'the sum of reach over buses is the total output plus wind once more, wind being connected to both buses'
    )


# ---------------------------------------------------------------------------
# the types every reader agrees on
# ---------------------------------------------------------------------------


def _disagreed_on(dtype: pl.DataType) -> bool:
    """Whether readers disagree on *dtype*: an unsigned integer, a nanosecond or non-UTC timestamp, or a nested type."""
    if dtype.is_unsigned_integer() or dtype.is_nested():
        return True
    return isinstance(dtype, pl.Datetime) and (dtype.time_unit == 'ns' or dtype.time_zone not in (None, 'UTC'))


def _disagreements(under: Path) -> list[str]:
    """``member: column dtype`` for every column of every parquet file under *under* that readers disagree on."""
    return [
        f'{file.relative_to(under)}: {column} {dtype}'
        for file in sorted(under.rglob('*.parquet'))
        for column, dtype in pl.read_parquet_schema(file).items()
        if _disagreed_on(dtype)
    ]


def test_every_member_of_the_folder_is_a_type_every_reader_agrees_on(runs: Path) -> None:
    assert not _disagreements(runs), 'a column of a type readers disagree on'


REPRO: dict[str, Any] = {
    'dimensions': {'t': {'dtype': 'int'}},
    'parameters': {'load': {'dims': ['t']}, 'weight': {'dims': ['t'], 'dtype': 'int'}},
    'variables': {'p': {'dims': ['t'], 'bounds': {'lower': 0}}},
    'constraints': {'covered': {'dims': ['t'], 'expression': 'p >= load'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(p * weight)'},
}

HOURS = [datetime(2026, 1, 1, hour) for hour in range(3)]


@pytest.mark.parametrize(
    ('labels', 'weight', 'written'),
    [
        pytest.param(pl.Series([0, 1, 2], dtype=pl.UInt32), pl.Int64, pl.Int64, id='unsigned-labels'),
        pytest.param(pl.Series([0, 1, 2]), pl.UInt8, pl.Int64, id='an-unsigned-value'),
        pytest.param(pl.Series(HOURS, dtype=pl.Datetime('ns')), pl.Int64, pl.Datetime('us'), id='nanosecond-labels'),
        pytest.param(
            pl.Series(HOURS, dtype=pl.Datetime('us', 'Europe/Berlin')),
            pl.Int64,
            pl.Datetime('us', 'UTC'),
            id='labels-in-a-time-zone',
        ),
    ],
)
def test_an_archive_writes_a_source_of_a_type_readers_disagree_on_as_one_they_agree_on(
    labels: pl.Series, weight: pl.DataType, written: pl.DataType, tmp_path: Path
) -> None:
    """Each of these reached ``sources/`` and ``answer/`` as it arrived."""
    dtype = 'datetime' if labels.dtype.is_temporal() else 'int'
    spec = {**REPRO, 'dimensions': {'t': {'dtype': dtype}}}
    steps = labels.alias('t')
    sources = {
        't': steps.to_frame(),
        'load': pl.DataFrame({'t': steps, 'value': [1.0, 2.0, 3.0]}),
        'weight': pl.DataFrame({'t': steps, 'value': pl.Series([1, 2, 3], dtype=weight)}),
    }
    with sps.solve(spec, sources, archive=tmp_path / 'run') as result:
        live = result.primal('p')
    assert not _disagreements(tmp_path / 'run'), 'a column of a type readers disagree on'
    archived = pl.read_parquet(tmp_path / 'run' / 'answer' / 'primal' / 'p.parquet')
    assert archived['t'].dtype == written
    assert archived['t'].to_list() == live['t'].cast(written).to_list(), 'the same labels, as the same instants'


def test_an_unsigned_integer_int64_cannot_hold_is_archived_as_it_arrived() -> None:
    frame = _with_run(pl.DataFrame({'id': pl.Series([2**64 - 1], dtype=pl.UInt64)}), 'run')
    assert frame['id'].to_list() == [2**64 - 1], 'a UInt64 past Int64 keeps its value rather than failing the archive'
