# Glossary

The one definition of each name this project uses.

> A **spec** is what you write: the math, with no data. A **model** is a spec
> with your data attached, a `Model` in Python. A **result** is one answer read
> back.

```
spec ──▶ build ──▶ Model ──▶ solve ──▶ Result
 │       (+data)                     (one answer)
 └─────▶ check ──▶ Program
                   (the plan, for reading)
```

## The chain

**Spec**
: The math before any data: a YAML file, a mapping, or a `Spec`. What it may
  contain is
  [the language](https://mathspec.readthedocs.io/en/latest/reference/language/).

**Program**
: What [`check`](api.md#specsolve.check) returns: the spec as the plan a build
  reads. No verb takes one back, so keep the `Spec`.

**Model**
: A spec with data attached: what [`build`](api.md) returns.

**Result**
: One answer read back from a solve:
  [`Result`](api.md#specsolve.types.Result). It owns its tables, so it outlives
  its model.

**Answer**
: What came back: a `Result` for one solve, a [`Sweep`](#sweeps) for a sweep.

**Archive**
: A spec, its data and its answer, written as one zip or directory by
  `archive=` ([archiving](../howto/archiving.md)).

## The verbs

**check** · **build** · **solve** · **write**
: `check` validates a spec with no data, and `build` returns a
  [Model](#the-chain). `solve` and `write` build and then solve or write a file
  ([the verbs](api.md#run-a-spec)).

**evaluate**
: Values an expression with no solver: of a spec with no variables, or of a
  `Result` at its solution.

**update**
: `model.update(sources)` puts new numbers on a built model in place. A change
  that moves a mask builds again.

**load** · **scan**
: `load_…` reads a saved answer whole. `scan_…` reads each frame when asked,
  so the files must outlive the value
  ([reading one too big to hold](../howto/archiving.md#read-one-too-big-to-hold)).

## The data

**Sources**
: The data you attach: a `dict` from each declared name to a table, and from
  each dimension to its labels ([the data contract](data.md)).

**Index**
: A dimension's labels in order, each once
  ([where coordinates come from](data.md#where-coordinates-come-from)).

**Coordinate**
: One point of a declaration's dimensions: one snapshot for one generator.

**Table**
: A polars `DataFrame` with one column per dimension and a `value` column, one
  row per coordinate.

**Relation**
: A named map between dimensions, such as the bus each generator sits on.

**Mask**
: The `where:` on a declaration. A coordinate it excludes has no row
  ([absence](https://mathspec.readthedocs.io/en/latest/reference/language/absence/)).

## How it runs

**attach**
: Fitting sources onto a spec to make a [Model](#the-chain), in `build` and
  `update`.

**Sink**
: Where a built model goes: a solver (`highs`, `gurobi`, `xpress`) or a file
  writer (`.lp`, `.mps`). What a sink can take is its **capability**
  ([what each sink takes](api.md#what-each-sink-takes)).

**start**
: What a solve begins from: `'previous'`, the default, or `None`. It never
  changes the optimum ([making a loop of solves faster](../howto/warm-start.md)).

## Sweeps

**Sweep**
: What [`solve_over`](sweeps.md) returns when it solves one spec once per
  slice. It has `Result`'s readers.

**Axis** · **slice** · **key**
: An axis, such as `EachWindow`, says how the sources split. A slice is one
  set of sources, solved as one model, and its key names it in every table.

**carry**
: `carry={parameter: variable}` hands one slice's solution to the next as
  data.

**held** · **spilled**
: A held sweep keeps its answers in memory. A spilled one writes them to
  `spill_to=` ([spilling](sweeps.md#spilling-a-sweep-to-disk)).
