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
        sweep = sps.solve_over('spec.yaml', sources, sps.EachCoordinate('scenario'), executor=pool)
    print(sweep.record)


if __name__ == '__main__':  # spawn re-imports your module; without this it recurses
    main()
```

Four workers need four times the memory of one slice.

## Hand the workers paths

On a cluster, write the sources to a filesystem the workers mount and pass the
paths. Without `workers_share_fs=True`, the paths are read here and shipped:

```python
sources = {'load': '/shared/load.parquet', 'cost': '/shared/cost.parquet', 'p_max': '/shared/p_max.parquet'}
sweep = sps.solve_over('spec.yaml', sources, sps.EachCoordinate('scenario'), executor=client, workers_share_fs=True)
```

## Read the sweep back

`sweep` reads as it does for a serial sweep:

```python
sweep.record  # one Record per slice, the scenario column first
sweep.primal('p')  # (scenario, snapshot, generator, value)
sweep.metrics  # one row per slice; every slice loaded its own solver
```

A sweep of one chain with a `carry` cannot run in parallel, because each
slice reads the one before it. A sweep over
[several axes](../reference/sweeps.md#several-axes) runs its chains in
parallel instead.
