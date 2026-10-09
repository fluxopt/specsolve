# Building many small models faster

Set polars' engine to in-memory when one process builds many small models,
such as a long [sweep](../reference/sweeps.md) or a rolling horizon.

## Set the engine before the first build

```python
import polars as pl

import specsolve as sps

pl.Config.set_engine_affinity('in-memory')
sweep = sps.solve_over('spec.yaml', sources, sps.EachCoordinate('scenario'))
```

`POLARS_ENGINE_AFFINITY=in-memory` in the environment does the same. polars
keeps the setting in the environment, so workers that a
[parallel sweep](parallel.md) starts after it inherit it.

The setting applies to the whole process. Your own polars queries in that
process use the in-memory engine too.

## When it helps

specsolve asks polars for its `auto` engine, and polars runs that on the
streaming engine. The streaming engine pays a fixed cost on every query, and a
small model does little work for each query it pays for.

- **Below one million variables, in-memory builds faster.** A model of 10,000
  to 170,000 variables built in up to 32% less time, at the same peak memory
  ([#1920](https://github.com/fluxopt/specsolve/pull/1920)).
- **Above one million variables, keep the default.** The streaming engine
  builds faster there, and it can hold less memory. At 10 million
  variables, in-memory took up to 35% more time and 36% more memory
  ([#1920](https://github.com/fluxopt/specsolve/pull/1920)).
- **Around one million variables, the two are level.** Neither engine was
  faster in every repeat.
