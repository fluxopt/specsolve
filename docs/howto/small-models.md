# Building many small models faster

specsolve builds a small model on polars' in-memory engine by itself. Set
polars' engine to in-memory yourself only to move the rest of a process onto it
too.

## What specsolve does by itself

specsolve reads and attaches every source on the in-memory engine. It then
builds a model of at most 250,000 variable columns on the in-memory engine, and
a larger model on polars' default engine. It counts the columns before it builds
them, from the sizes of the dimensions each variable is declared over.

The default engine is polars' streaming engine. It pays a fixed cost on every
query, and a small model does little work for each query it pays for.

- **Below one million variables, in-memory builds faster.** A model of 10,000
  to 170,000 variables built in up to 32% less time, at the same peak memory
  ([#1920](https://github.com/fluxopt/specsolve/pull/1920)).
- **Above one million variables, the default engine is better.** It builds
  faster there, and it can hold less memory. At 10 million variables,
  in-memory took up to 35% more time and 36% more memory
  ([#1920](https://github.com/fluxopt/specsolve/pull/1920)).
- **Around one million variables, the two are level.** Neither engine was
  faster in every repeat.

## Set the engine for the whole process

```python
import polars as pl

import specsolve as sps

pl.Config.set_engine_affinity('in-memory')
sweep = sps.solve_over('spec.yaml', sources, sps.EachCoordinate('scenario'))
```

`POLARS_ENGINE_AFFINITY=in-memory` in the environment does the same. polars
keeps the setting in the environment, so workers that a
[parallel sweep](parallel.md) starts after it inherit it.

The setting moves every query in the process onto the in-memory engine: a model
above 250,000 columns, what specsolve reads after a build, such as an answer,
and your own polars queries.
