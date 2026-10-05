# Running models from a manifest

This page runs several solves of one model from a YAML file and one command,
with no Python written. The file is a *manifest*: the arguments of
[`solve`](../reference/api.md#specsolve.solve) and
[`solve_over`](../reference/api.md#specsolve.solve_over), one entry per
run. Every key is in [the manifest reference](../reference/manifest.md).

Every command on this page runs when the site is built, on the files in
[`examples/manifest`](https://github.com/fluxopt/specsolve/blob/main/examples/manifest),
and what you see under it is what it printed on this commit.

```python exec="true" session="manifest"
import shutil
import subprocess
import tempfile
from pathlib import Path

study = Path(tempfile.mkdtemp()) / 'manifest'
shutil.copytree('examples/manifest', study)
shutil.copy('examples/dispatch.yaml', study.parent / 'dispatch.yaml')


def specsolve(*arguments: str) -> None:
    done = subprocess.run(['specsolve', *arguments], cwd=study, capture_output=True, text=True, check=False)
    print('$ specsolve', *arguments)
    print(done.stdout + done.stderr, end='')
```

## Install the command

The `cli` extra installs the `specsolve` command and the reader for Excel
workbooks:

```bash
pip install 'specsolve[cli]'
```

## Write the manifest

Keep the data in files beside the manifest. This study holds the
[dispatch model](../examples/dispatch.md) and four places its tables come from:

```
examples/
├── dispatch.yaml                 the spec
└── manifest/
    ├── specsolve.yaml            the manifest
    └── data/
        ├── base.xlsx             sheets generator, p_max, cost
        ├── high_gas.xlsx         sheet cost, with dearer gas
        ├── timeseries/           snapshot.csv, load.csv
        └── scenarios/load.parquet  the load, with a scenario column
```

`specsolve.yaml` names the spec and the data once, at the top level, and
lists three runs under `runs:`:

```yaml
--8<-- "examples/manifest/specsolve.yaml"
```

- **A top-level key is the default of every run.** All three runs solve
  `../dispatch.yaml` with HiGHS and a ten-minute limit.
- **A run's own `sources` are read after the defaults.** `high_gas` reads
  `cost` from its own workbook, and that table replaces the `cost` sheet of
  `base.xlsx`.
- **`from` starts a run from another run.** `scenarios` takes everything
  `high_gas` resolved to, and adds a `load` table with a `scenario` column.
- **A run with an `axis` is a sweep.** `scenarios` solves once per
  scenario, two at a time in separate processes.
- **`archive` is a directory of runs.** Each run archives to
  `runs/<run name>`.

Paths are relative to the manifest, wherever the command runs from.

## Check the runs

`specsolve check` reads every run's spec and data, and attaches the data
without solving. It reports each run on one line:

```python exec="true" result="console" session="manifest"
specsolve('check')
```

A run with a problem names it, and the command exits with 1.

## Solve the runs

`specsolve solve` solves every run in manifest order, one after another, and
reports how each one ended:

```python exec="true" result="console" session="manifest"
specsolve('solve')
```

Name runs to solve only those. A run whose archive exists is skipped, never
overwritten, so the same command resumes a study that stopped part-way:

```python exec="true" result="console" session="manifest"
specsolve('solve', 'base')
```

Delete `runs/base` to solve `base` again. `specsolve list` says which runs
are archived:

```python exec="true" result="console" session="manifest"
specsolve('list')
```

Each command reads `specsolve.yaml` in the working directory. To read another
file, name it first: `specsolve solve studies/winter.yaml base`.

## Read the answers

Each archive holds the spec, the data it was solved with and the answer.
[`load_archive`](../reference/api.md#specsolve.load_archive) reads one back:

```python exec="true" source="material-block" result="text" session="manifest"
import specsolve as sps

archive = sps.load_archive(study / 'runs' / 'scenarios')
print(archive.sweep.record.select('scenario', 'termination_condition', 'objective'))
```

## Run a manifest from Python

[`load_manifest`](../reference/api.md#specsolve.load_manifest) reads the same
file. Each run's `arguments` are the keyword arguments of its call, and
`solve` makes the call, with any keyword replacing the manifest's:

```python exec="true" source="material-block" result="text" session="manifest"
manifest = sps.load_manifest(study / 'specsolve.yaml')
run = manifest.runs['high_gas']

result = run.solve(archive=None, solver_options={'time_limit': 60})
print(result.termination_condition, result.objective)
```

Use Python for what the manifest does not hold: data computed before the
solve, logic between solves, or a remote executor such as a dask `Client`:

```python
sweep = manifest.runs['scenarios'].solve(executor=client)
```
