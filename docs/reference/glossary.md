# Glossary

The one definition of each name this project uses. A word of the language,
such as entry, condition or operator, means what the
[mathspec glossary](https://mathspec.readthedocs.io/en/latest/reference/glossary/)
says.

> A **spec** is what you write: the math, with no data. A **model** is a spec
> with your data attached, a `Model` in Python. A **result** is one answer read
> back.

```
spec ──▶ build ──▶ Model ──▶ solve ──▶ Result
 │       (+data)                     (one answer)
 └─────▶ check ──▶ Program
                   (the compiled spec, for reading)
```

## The chain

**Spec**
: The math before any data: a YAML file, a mapping, or a `Spec`
  ([mathspec](https://mathspec.readthedocs.io/en/latest/reference/glossary/#spec)). What it may
  contain is
  [the language](https://mathspec.readthedocs.io/en/latest/reference/language/).

**Program**
: What [`check`](api.md#specsolve.check) returns: what the spec compiles to
  ([mathspec](https://mathspec.readthedocs.io/en/latest/reference/glossary/#program)), and what a build reads. No function takes one back, so keep the `Spec`.

**Model**
: A spec with data attached ([mathspec](https://mathspec.readthedocs.io/en/latest/reference/glossary/#model)): what
  [`build`](api.md) returns.

**Result**
: One answer read back from a solve:
  [`Result`](api.md#specsolve.types.Result). It owns its tables, so it outlives
  its model.

**Answer**
: What came back: a `Result` for one solve, a [`Sweep`](#sweeps) for a sweep.

**Archive**
: A spec, its data and its answer, written as one zip or directory by
  `archive=` ([archiving](../howto/archiving.md)).

## The functions

**check** · **build** · **solve** · **write**
: `check` validates a spec with no data, and `build` returns a
  [Model](#the-chain). `solve` and `write` build and then solve or write a file
  ([the API](api.md#run-a-spec)).

**evaluate**
: Values an expression with no solver: of a spec with no variables, or of a
  `Result` at its solution.

**update**
: `model.update(sources)` puts new numbers on a built model in place. A change
  that alters which coordinates a `where:` keeps builds again.

**`load_…`** · **`scan_…`**
: `load_…` reads a saved answer whole. `scan_…` reads each table when asked,
  so the files must outlive the value
  ([reading one too big to hold](../howto/archiving.md#read-one-too-big-to-hold)).

## The data

**Sources**
: The data you attach: a `dict` from each declared name to a table, and from
  each dimension to its labels ([the data contract](data.md)).

**Label**
: One member of a dimension, such as `wind` in `generator`
  ([mathspec](https://mathspec.readthedocs.io/en/latest/reference/glossary/#label)). A dimension's
  labels arrive in `sources` under its name, in order, each once
  ([where coordinates come from](data.md#where-coordinates-come-from)).

**Coordinate**
: One point of an entry's dimensions: one snapshot for one generator
  ([mathspec](https://mathspec.readthedocs.io/en/latest/reference/glossary/#coordinate)).

**Table**
: A polars `DataFrame` with one column per dimension and a `value` column, one
  row per coordinate.

**Relation**
: A named map between dimensions, such as the bus each generator sits on.

**`where:`**
: Narrows an entry to the coordinates where its condition is true, and
  leaves the entry absent elsewhere ([mathspec](https://mathspec.readthedocs.io/en/latest/reference/glossary/#where),
  [absence](https://mathspec.readthedocs.io/en/latest/reference/language/absence/)).

## How it runs

**Engine**
: A tool that attaches data to a spec, builds the model and hands it to a
  solver, as mathspec
  [defines it](https://mathspec.readthedocs.io/en/latest/reference/glossary/#tool).
  specsolve is one.

**attach**
: Fitting sources onto a spec to make a [Model](#the-chain), in `build` and
  `update` ([mathspec](https://mathspec.readthedocs.io/en/latest/reference/glossary/#attach)).

**Sink**
: Where a built model goes: a solver (`highs`, `gurobi`, `xpress`) or a file
  writer (`.lp`, `.mps`).

**Capability**
: One kind of model a sink takes, such as integer variables or a quadratic
  constraint. A model that needs a capability its sink lacks is refused before
  the solve ([what each sink takes](api.md#what-each-sink-takes)).

**start**
: What a solve begins from: `'previous'`, the default, or `None`. It never
  changes the optimum ([starting from an earlier answer](../howto/warm-start.md)).

## Sweeps

**Sweep**
: What [`solve_over`](sweeps.md) returns when it solves one spec once per
  slice. It has `Result`'s readers.

**Axis** · **slice** · **key**
: An axis, such as `EachWindow`, says how the sources split. A slice is one
  set of sources, solved as one model, and its key names it in every table.

**carry**
: An axis's `carry={parameter: variable}` hands one slice's solution to the
  next as data.

**held** · **spilled**
: A held sweep keeps its answers in memory. A spilled one writes them to
  `spill_to=` ([spilling](sweeps.md#spilling-a-sweep-to-disk)).
