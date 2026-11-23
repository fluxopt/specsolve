# Running a sweep in parallel

One slice of a [sweep](../reference/sweeps.md) per worker, on a pool you
start. The reference is
[running slices in parallel](../reference/sweeps.md#running-slices-in-parallel).

## Start the pool with `spawn`

A forked worker hangs, so pass a `spawn` or `forkserver` context and guard
the entry point:

```python
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import specsolve as sps


def main():
    ctx = multiprocessing.get_context('spawn')  # or 'forkserver'
    with ProcessPoolExecutor(4, mp_context=ctx) as pool:
        sweep = sps.solve_over('spec.yaml', sources, sps.EachLabel('scenario'), executor=pool)
    print(sweep.records)


if __name__ == '__main__':  # spawn re-imports your module; without this it recurses
    main()
```

Four workers need four times the memory of one slice.

## Run on a cluster

A cluster's client is an executor too. A path source is read here and sent to
the workers as bytes, as a table is:

```python
sweep = sps.solve_over('spec.yaml', sources, sps.EachLabel('scenario'), executor=client)
```

## Read the sweep back

`sweep` reads as it does for a serial sweep:

```python
sweep.records  # one Record per slice, the scenario column first
sweep.primal('p')  # (scenario, snapshot, generator, value)
sweep.metrics  # one row per slice; every slice loaded its own solver
```

A sweep with a `carry` cannot run in parallel.
