"""A solve's answer as runs of a datarecord record, beside the archive's files."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

pytest.importorskip('datarecord')

from datarecord import Revision, duck
from datarecord.layered.resolve import write_schema
from datarecord.mutable import NewChild, WorkingRecord

import specsolve as sps
from specsolve.record import answer_schema, write_answer


@pytest.fixture
def record_base(tmp_path, monkeypatch):
    """A directory the records of this test live under, and a connection to it."""
    base = str(tmp_path / 'records') + '/'
    monkeypatch.setattr(duck, 'DEFAULT_BASE_URI', base)
    con = duck.connect(base_uri=base)
    yield base, con
    con.close()


@pytest.fixture
def two_runs(dispatch_yaml, dispatch_frame_inputs, record_base, tmp_path):
    """The dispatch example solved at its load and at a higher one, each written as a run."""
    base, con = record_base
    high = dispatch_frame_inputs | {'load': dispatch_frame_inputs['load'].with_columns(pl.col('value') * 1.2)}
    write_schema(answer_schema(dispatch_yaml), base)
    root = Revision.create(con)
    working = WorkingRecord(root.record, con)
    results = {}
    for run, sources in {'base': dispatch_frame_inputs, 'high': high}.items():
        with sps.build(dispatch_yaml, sources) as model:
            results[run] = model.solve()
            metrics = model.diagnostics().metrics()
            write_answer(working, results[run], run=run, metrics=metrics, input_node=f'node-{run}')
            results[run] = (results[run], metrics)
    return working.commit(NewChild(root)).record, results


def _run(record, name: str, run: str) -> pl.DataFrame:
    frame = pl.from_arrow(record.attributes[name].collect().to_arrow())
    columns = [c for c in frame.columns if c not in ('run', 'attribute', 'breakpoint')]
    return frame.filter(pl.col('run') == run).select(columns).sort(columns[:-1])


def test_each_run_holds_its_own_primal(two_runs):
    record, results = two_runs
    for run, (result, _) in results.items():
        expected = result.primal('p').sort(['snapshot', 'generator'])
        assert _run(record, 'primal.p', run).equals(expected), f'run {run!r} keeps its own dispatch'


def test_the_solve_facts_are_attributes_of_the_run(two_runs):
    """Objective, termination and every metrics column are one row per run."""
    record, results = two_runs
    runs = pl.from_arrow(record.dims['run'].collect().to_arrow())
    by_run = {row['run']: row for row in runs.to_dicts()}
    for run, (result, metrics) in results.items():
        assert by_run[run]['objective'] == pytest.approx(result.objective), f'run {run!r} keeps its objective'
        assert by_run[run]['termination_condition'] == result.termination_condition
        assert by_run[run]['rows'] == metrics.rows
        assert by_run[run]['input_node'] == f'node-{run}'
    assert by_run['high']['objective'] > by_run['base']['objective'], (
        'more load costs more, so the two runs hold different answers'
    )


def test_a_record_holds_what_the_archive_holds(two_runs, tmp_path):
    """Every frame `save` writes is the same rows in the record, and nothing more.

    The two are one enumeration (`Result._answered`), so a kind or a name one
    of them gains the other gains too.
    """
    record, results = two_runs
    saved = results['base'][0].save(tmp_path / 'answer')
    files = sorted(saved.glob('*/*.parquet'))
    names = {f'{path.parent.name}.{path.stem}' for path in files}
    assert names, 'the archive holds at least one frame'
    stored = {n for n in record.attributes if '.' in n}
    assert stored == names, 'the record holds a frame for exactly the files the archive writes'
    for path in files:
        on_disk = pl.read_parquet(path)
        keys = [c for c in on_disk.columns if c != 'value']
        assert _run(record, f'{path.parent.name}.{path.stem}', 'base').equals(on_disk.sort(keys)), path.name


def test_an_answer_axis_the_spec_declares_is_refused(dispatch_yaml):
    with pytest.raises(ValueError, match=r"\['snapshot'\] is a dim of the spec"):
        answer_schema(dispatch_yaml, over='snapshot')


def test_the_schema_names_a_quantity_per_declaration(dispatch_yaml):
    schema = answer_schema(dispatch_yaml)
    quantities = sorted(n for n in schema.attributes if '.' in n)
    assert quantities == ['activity.power_balance', 'dual.power_balance', 'primal.p'], (
        'one per variable, and a dual and an activity per constraint'
    )
    assert schema.attributes['primal.p'].dims == {'run', 'snapshot', 'generator'}, (
        'a quantity varies over the run and its own dims'
    )
    assert Path(dispatch_yaml).exists()
