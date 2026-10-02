# Checking a model repository in CI

How to gate every commit of a repository of model files: each file against
this package, and each example that ships data against a solver. The first
step needs no data and no solver. The second builds and solves nothing, and
is the only step that can say whether a solver takes the model.

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

## Build every example that has data, and check its solver

Whether a solver takes a model is a fact about the built model, not about the
file. A file can declare a quadratic cost that one dataset prices at zero and
another does not. So build the example, and ask the built model. `Model.check`
refuses a model the sink cannot take and names the sinks that do, and it
solves nothing
([`Model.check`](../reference/api.md#specsolve.Model.check)):

```python
import specsolve as sps

with sps.build('models/dispatch.yaml', sources) as model:
    model.check('highs')
```

The sink is a solver name or an output suffix, so `model.check('.lp')` asks
whether the file can be written
([what each sink takes](../reference/api.md#what-each-sink-takes)).

## Solve where the answer is known

A build that checks can still be a model whose rows changed and still solve.
Where the repository holds a known objective, solve and hold the number to it:

```python
import specsolve as sps

result = sps.solve('models/dispatch.yaml', sources)
assert result.is_ok, result.termination_condition
assert abs(result.objective - 9800.0) < 1e-6
```
