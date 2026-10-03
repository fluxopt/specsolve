# Reading a directory of runs

How to read many archives at once: which runs terminated how, what each cost,
and which input changed between them. One archive is
[archiving a solve](archiving.md); this is the directory they pile up in.

## The three tables

An archive is a tree of parquet files, so a directory of them is a table per
glob. Nothing is loaded and no schema is maintained. The same globs work in
any tool that reads a folder of parquet files, such as DuckDB, Spark or Power
BI's "Combine files":

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

**Every row says which archive it came from.** `specsolve_run` is the
archive's own name: `runs/nightly-2026-09-10.zip` writes `nightly-2026-09-10`.
Every table in an archive carries it, so nothing has to read the paths.

**Every file has the same columns, whoever wrote it.** A single solve, an
`EachCoordinate` sweep and an `EachWindow` sweep write one schema, so a
directory that holds all three globs as one table. A sweep writes one row per
slice and names it in two text columns. `slice_axis` is the key name, such as
`scenario` or `snapshot_start`, and `slice` is the key as text. Both are null
for a single solve:

```python
answers.select('specsolve_run', 'slice_axis', 'slice')
# specsolve_run  slice_axis      slice
# base           null            null
# scenarios      scenario        high
# scenarios      scenario        low
# rolling        snapshot_start  0
# rolling        snapshot_start  4
```

**Use `load_archive` and `scan_archive` for one archive, not for many.** They
give back a spec, its sources and an answer, which is what re-running a case
needs. A warehouse question is a query over the parquet.

## The run on a value frame

`answer/primal/p.parquet` holds the model's own dimension columns, `value`,
and `specsolve_run`. A tool that combines files and drops their paths, such as
Power BI's "Combine files", still gets the run:

```python
sps.solve('dispatch.yaml', sources, archive='runs/nightly-2026-09-10/')

pl.read_parquet('runs/*/answer/primal/p.parquet')
# snapshot  generator  value  specsolve_run
# 0         wind       90.5   nightly-2026-09-10
# 0         solar      0.0    nightly-2026-09-10
```

**The column is the archive's, not the model's.** The `specsolve_` prefix is
reserved for the columns specsolve adds, and a spec that declares a name with
that prefix, in any letter case, is refused. Both readers drop the column from
the answer's frames, so a frame read back is the frame the solve returned.
`load_archive` drops it from the sources too. The sources `scan_archive` gives
back are the archive's own files, and reading one gives the column back.

## Compare cases solved apart

`solved_at` is when the solver returned, so runs solved on different machines
still order:

```python
table = pl.read_parquet('runs/*/answer/record.parquet')
table.sort('solved_at').select('specsolve_run', 'status', 'objective')
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

**Across a whole directory it is a window rather than a join**,
`specsolve_run` being on every row:

```python
inputs = pl.read_parquet('runs/*/sources.parquet')
inputs.sort('specsolve_run').with_columns(before=pl.col('digest').shift().over('source')).filter(
    pl.col('before').is_not_null() & (pl.col('before') != pl.col('digest'))
)
```

**The digest is of the table the solve read, before the run is stamped on**,
so one table archived under two names digests alike. That table is the one
under `sources/`, not the file you passed, and hashing the member does not give
the row back, because the member also carries `specsolve_run`. Two archives of
the same data written by different versions of polars can differ, and reading
an archive does not verify the digests
([the rule](../reference/api.md#specsolve.SolveArchive)).

## See what the runs cost

`answer/metrics.parquet` carries the model's size beside the clocks, so one
query says which cases are growing and where the time goes:

```python
metrics.select('specsolve_run', 'rows', 'nonzeros', 'build_seconds', 'solve_seconds').sort(
    pl.col('build_seconds') + pl.col('solve_seconds'), descending=True
)
```

The columns are [the metrics](../reference/api.md#specsolve.relational.parquet.Metrics). A sweep
writes one row per slice, with the slice's own share of the clocks and
`solves` of `1`
([reading a sweep](../reference/sweeps.md#reading-a-sweep)).

## Query it from a database

A directory archive is parquet where it lies, so a query engine reads it
without polars in the way. In DuckDB, the glob reads solves and sweeps
together:

```sql
select specsolve_run, slice, rows, nonzeros, build_seconds, solve_seconds
from read_parquet('runs/*/answer/metrics.parquet')
order by build_seconds + solve_seconds desc;
```

**A value frame carries `specsolve_run` too**, so a query across runs reads it
as any other column:

```sql
select specsolve_run, snapshot, generator, value
from read_parquet('runs/*/answer/primal/p.parquet')
order by specsolve_run, snapshot;
```

Inside one archive every frame is tidy, so the values join to the sources they
were solved from on the coordinates both carry:

```sql
select p.snapshot, p.generator, p.value, load.value as load
from 'runs/base/answer/primal/p.parquet' p
join 'runs/base/sources/load.parquet' load using (snapshot, specsolve_run);
```

**A zip has to be unpacked first**, because no query engine reads inside one.
Give either reader an `into=` and query what lands there.
