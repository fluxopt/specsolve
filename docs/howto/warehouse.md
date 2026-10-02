# Reading a directory of runs

How to read many archives at once: which runs terminated how, what each cost,
and which input changed between them. One archive is
[archiving a solve](archiving.md); this is the directory they pile up in.

## The three tables

An archive is a tree of parquet files, so a directory of them is a table per
glob. Nothing is loaded and no schema is maintained:

```python
import polars as pl

answers = pl.read_parquet('runs/*/answer/record.parquet')
metrics = pl.read_parquet('runs/*/answer/metrics.parquet')
inputs = pl.read_parquet('runs/*/sources.parquet')
```

| glob | one row per | says |
|---|---|---|
| `answer/record.parquet` | solve, or sweep slice | how it terminated, what it reached, when, under what name |
| `answer/metrics.parquet` | the same | what the build and its solves spent, and how big the model was |
| `sources.parquet` | source per archive | what each input's table digests to |

**Every row says which archive it came from.** `run` is the archive's own
name: `runs/nightly-2026-09-10.zip` writes `nightly-2026-09-10`. It is on all
three tables, so nothing has to read the paths.

**A directory holding both solves and sweeps does not glob.** A sweep's record
carries the dimension its axis cut on, and its metrics carry different columns
from a solve's. polars refuses the mismatch:

```text
SchemaError: extra column in file outside of expected schema: scenario
```

Union by name instead. The columns one side lacks come back null:

```python
from glob import glob

answers = pl.concat(
    [pl.read_parquet(file) for file in sorted(glob('runs/*/answer/record.parquet'))],
    how='diagonal',
)
```

`sources.parquet` has the same three columns whoever wrote it, so it globs
either way.

**Use `load_archive` and `scan_archive` for one archive, not for many.** They
give back a spec, its sources and an answer, which is what re-running a case
needs. A warehouse question is a query over the parquet.

## The run on a value frame

`answer/primal/p.parquet` holds the model's own dimension columns and `value`.
Nothing in it says which archive it came from, so only the three tables above
answer that.

**Name the directory `run=<name>` and every frame carries the run.** That is
the hive layout. A query engine reads it as a column, and no file stores it:

```python
sps.solve('dispatch.yaml', sources, archive='runs/run=nightly-2026-09-10/')

pl.read_parquet('runs/*/answer/primal/p.parquet', hive_partitioning=True)
# snapshot  generator  value  run
# 0         wind       90.5   nightly-2026-09-10
# 0         solar      0.0    nightly-2026-09-10
```

**The three tables carry the same name.** The archive is named
`nightly-2026-09-10`. The stamp drops the `run=`, so a join on `run` matches
whichever side a column came from.

**A zip cannot use the layout.** No query engine reads inside one, so a
warehouse queried where it lies is a directory of directories.

## Compare cases solved apart

`solved_at` is when the solver returned, so runs solved on different machines
still order:

```python
table = pl.read_parquet('runs/*/answer/record.parquet')
table.sort('solved_at').select('run', 'status', 'objective')
```

**Check the digests before you read the numbers.** `spec_digest` is a digest of
the spec an answer came back from. One distinct value across the table is the
claim that every row answered the same document, and a null is a row that
named none, so ask for both:

```python
assert table['spec_digest'].n_unique() == 1, 'one spec, or this compares nothing'
assert table['spec_digest'].null_count() == 0, 'and every row named the document it answered'
```

## Find which input changed between two runs

`spec_digest` says two runs answered the same document and nothing about the
numbers, so two runs of one spec over different data carry the same one.
`sources.parquet` separates them, and names the input that moved:

```python
import specsolve as sps

base = sps.load_archive('runs/base/')
other = sps.load_archive('runs/halved/')

moved = base.source_digests.join(other.source_digests, on='source', suffix='_other').filter(
    pl.col('digest') != pl.col('digest_other')
)
moved['source'].to_list()  # ['load']
```

**Across a whole directory it is a window rather than a join**, `run` being on
every row:

```python
inputs = pl.read_parquet('runs/*/sources.parquet')
inputs.sort('run').with_columns(before=pl.col('digest').shift().over('source')).filter(
    pl.col('before').is_not_null() & (pl.col('before') != pl.col('digest'))
)
```

**The digest is of the bytes the archive holds**, so hashing
`sources/load.parquet` gives the row back. That file is the table the solve
read, not the file it was given. Two archives of the same data
written by different versions of polars can differ, and reading an archive
does not verify the digests
([the rule](../reference/api.md#specsolve.SolveArchive)).

## See what the runs cost

`answer/metrics.parquet` carries the model's size beside the clocks, so one
query says which cases are growing and where the time goes:

```python
metrics.select('run', 'rows', 'nonzeros', 'build_seconds', 'solve_seconds').sort(
    pl.col('build_seconds') + pl.col('solve_seconds'), descending=True
)
```

The columns are [the metrics](../reference/api.md#specsolve.relational.parquet.Metrics). A sweep
records a `SliceMetrics` per slice instead, keyed by the axis and stamped
with `run` like any other row
([reading a sweep](../reference/sweeps.md#reading-a-sweep)).

## Query it from a database

A directory archive is parquet where it lies, so a query engine reads it
without polars in the way. In DuckDB, `union_by_name` takes the two kinds of
archive together:

```sql
select run, rows, nonzeros, build_seconds, solve_seconds
from read_parquet('runs/*/answer/metrics.parquet', union_by_name = true)
order by build_seconds + solve_seconds desc;
```

**A value frame carries `run` where the directory is named for it.** DuckDB
reads the same hive layout, and the query says nothing about it:

```sql
select run, snapshot, generator, value
from read_parquet('runs/*/answer/primal/p.parquet', hive_partitioning = true)
order by run, snapshot;
```

Inside one archive every frame is tidy, so the values join to the sources they
were solved from on the coordinates both carry:

```sql
select p.snapshot, p.generator, p.value, load.value as load
from 'runs/base/answer/primal/p.parquet' p
join 'runs/base/sources/load.parquet' load using (snapshot);
```

**A zip has to be unpacked first**, because no query engine reads inside one.
Give either reader an `into=` and query what lands there.
