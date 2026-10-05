# The manifest

This page lists every key a manifest takes, how runs inherit them, where data
is read from, and what the `specsolve` command does. Writing one is
[running models from a manifest](../howto/manifest.md).

A manifest is a YAML file, `specsolve.yaml` by default. A run that names its
spec and its data is a whole manifest:

```yaml
manifest: 1
spec: model.yaml
sources: [data/]
runs:
  base: {}
```

## Keys

**A key is the name of an argument of
[`solve`](api.md#specsolve.solve) or [`solve_over`](api.md#specsolve.solve_over)**,
and takes what that argument takes, unless it is one of the six keys the
manifest owns. Every key except `manifest` and `runs` goes at the top level,
as every run's default, or in a run.

| Key | Takes | Calls |
|---|---|---|
| `spec` | the spec file | both |
| `solver_name` | `highs`, `gurobi` or `xpress` | both |
| `solver_options` | a mapping, passed to the solver | both |
| `record_options` | a list of option names | both |
| `axis` | one axis class and its arguments, [below](#axis) | `solve_over` |
| `carry` | `{parameter: variable}` | `solve_over` |
| `key_name` | the name of the slice column | `solve_over` |
| `keep` | `solver`, `progress` or `nothing` | `solve_over` |
| `keep_windows` | `true` or `false` | `solve_over` |

The keys the manifest owns:

| Key | Takes | Means |
|---|---|---|
| `manifest` | `1` | the format version; required, at the top level |
| `runs` | a mapping of run name to its keys | one entry per call, solved in this order |
| `from` | a run name | in a run: start from that run's resolved keys |
| `sources` | a list of [source locations](#source-locations) | the tables read into `sources` |
| `archive` | a directory | each run archives to `<archive>/<run name>` |
| `workers` | a whole number above zero | `solve_over` only: solve the slices in that many local processes |

`executor`, `workers_share_fs` and `spill_to` are not keys. Pass them to
[`Run.solve`](api.md#specsolve.types.Run.solve) from Python.

### axis

**`axis` names one class and its arguments.** The class is
[`EachCoordinate`](api.md#specsolve.EachCoordinate) or
[`EachWindow`](api.md#specsolve.EachWindow), and the arguments are the
names its constructor takes:

<!-- doctest: skip -->
```yaml
axis: {EachCoordinate: {dim: scenario}}
axis: {EachWindow: {dim: snapshot, steps: 24, lookahead: 24, into: t}}
```

A run with an `axis` calls `solve_over`, and a run without one calls `solve`.
A list of slices written by hand is not a manifest form; several unrelated
solves are several runs.

## Defaults and `from`

- **A run's own key replaces the default.** A run that sets `solver_options`
  replaces the whole mapping at the top level, not one option of it. A run
  that sets `sources` reads those locations and no others.
- **`from` starts a run from another run's resolved keys**, its defaults
  included. The run's own keys then replace those.

## Source locations

**A location is a file or a directory, and each table in it is one source.**
The table's name is the source key it fills:

| Location | Tables | Name |
|---|---|---|
| `data/base.xlsx` | one per sheet | the sheet name |
| `data/timeseries/` | one per `.parquet` or `.csv` file in it | the file name without its suffix |
| `data/load.parquet`, `data/load.csv` | one | the file name without its suffix |

- **A later location in one list replaces an earlier table of the same
  name.** This is how a run changes one table: it lists the defaults' locations,
  then its own.
- **Paths are relative to the manifest**, not to the working directory.
- **A directory reads only its `.parquet` and `.csv` files.** It holds one
  file per table: `load.csv` beside `load.parquet` is refused.
- **Reading is the only step.** Each table goes to the
  [data contract](data.md) as it is, which checks it as it checks any table.
- **A CSV file carries no types.** polars infers each column. Use parquet for
  datetimes, and for labels that look like numbers.

## Excel workbooks

**Each sheet is one table, named by the sheet.** The first row holds the
column names, one column per dimension and a `value` column for a parameter,
as the [data contract](data.md) asks of any table. A sheet the spec does not
declare is refused, so keep notes in another workbook.

Excel stores what a cell holds, which is not always what it shows:

- **A whole number reads as an integer.** `80.0` in a cell reads as `80`. A
  `float` parameter accepts it.
- **A label typed as `001` is stored as the number `1`.** Format the column as
  text before you type the labels.
- **A date typed into a number cell is a number.** A cell formatted as a date
  reads as a date; one formatted as a number reads as the day count Excel
  keeps.

Reading a workbook needs `fastexcel`, which the `cli` extra installs. Without
it, reading an `.xlsx` location is refused with `pip install 'specsolve[cli]'`.

## When a manifest is checked

**[`load_manifest`](api.md#specsolve.load_manifest), and every command, refuse
at load what the file alone decides**, before any data is read. The error
names the file and the run:

- an unknown key, with the keys that are valid there;
- a key written twice;
- no `manifest: 1`;
- a run with no `spec`;
- a `from` that names no run, or that makes a cycle;
- an axis class or argument the class does not take, or a value it refuses;
- a key only `solve_over` takes, `workers` included, on a run with no `axis`;
- a run name with a path separator in it.

A missing file, a missing table and a table of the wrong shape are found when
the data is read: by `specsolve check`, or by the solve.

## The command

```
specsolve check [MANIFEST] [RUN]...
specsolve solve [MANIFEST] [RUN]...
specsolve list  [MANIFEST]
```

`MANIFEST` is the first argument when it ends in `.yaml` or `.yml`, and is
`specsolve.yaml` in the working directory otherwise. Each `RUN` is a run name;
none means every run, in manifest order. A name the manifest does not have
exits with 2, naming the runs it has.

| Command | Does | Exits with 1 when |
|---|---|---|
| `check` | reads each run's spec and data and attaches it without solving; a sweep attaches every slice | a run has a problem |
| `solve` | solves each run in turn, and prints how it ended and how long it took | a run failed, or a solve ended other than optimal |
| `list` | prints each run, whether it solves or sweeps, and whether its archive exists | never |

Every command exits with 1 when the manifest is refused.

**`solve` never overwrites an archive.** A run whose `<archive>/<run name>`
exists is skipped and reported. Delete the directory to solve the run again.
