# Checking a model repository in CI

How to gate every commit of a repository of model files: each file against
this package, and each example that ships data against a solver. The first
step needs no data and no solver. The second is the only step that can say
whether a solver takes the model.

## Check every file

`check` loads a file through the language and refuses what this package
cannot build. That is a `piecewise:` block that is not yet expanded, and two
declarations whose names differ only by case. It needs no data and no solver,
and raises on the first fault
([`check`](../reference/api.md#specsolve.check)):

```python
from pathlib import Path

import specsolve as sps

for path in sorted(Path('models').glob('*.yaml')):
    sps.check(path)
```

The language's advice, such as a dimension nothing uses as an axis, is issued
as a `SpecsolveWarning`. To fail the job on it, turn the warning into an
error before the loop:

```python
import warnings

import specsolve as sps

warnings.simplefilter('error', sps.SpecsolveWarning)
```

To check a file against the language alone, with no Python, use
[mathspec's own check](https://mathspec.readthedocs.io/en/latest/howto/check/).

## Solve or write every example that has data

Whether a solver takes a model is a fact about the built model, not about the
file. A file can declare a quadratic cost that one dataset prices at zero and
another does not. So the solver question is answered by building. `solve` on
the solver the repository claims, or `write` to `.lp` where no solver is
installed. Either refuses a built model the sink cannot take, before the
load, and names the sinks that do
([what each sink takes](../reference/api.md#what-each-sink-takes)).

```python
import specsolve as sps

result = sps.solve('models/dispatch.yaml', sources)
assert result.is_ok, result.termination_condition
assert abs(result.objective - 9800.0) < 1e-6
```

```python
import specsolve as sps

sps.write('models/dispatch.yaml', sources, 'dispatch.lp')
```

Hold the objective to a known number where one exists. `is_ok` alone passes a
model whose rows changed and still solve.
