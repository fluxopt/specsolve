# Glossary

The one definition of each name this project uses. The rest hang off one
distinction:

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
: Short for specification. The math before any data: a YAML file, a
  mapping, or a `Spec` from `mathspec.to_spec`. It carries no numbers, and
  every verb takes it first. What it may contain is
  [the language](https://mathspec.readthedocs.io/en/latest/reference/language/).

**Program**
: What [`check`](api.md#specsolve.check) returns: the spec as the plan a build
  reads, still with no data. No verb takes one back, so keep the `Spec`.

**Model**
: A spec with data attached: what [`build`](api.md) returns. One `Model`
  feeds any sink through `solve()` or `write(path)`, and `update(...)` puts
  new numbers on it in place.

**Result**
: One answer read back from a solve: `objective`, `primal(name)`,
  `dual(name)`, `evaluate(expression)` and the rest of
  [`Result`](api.md#specsolve.types.Result). It owns its tables, so it
  outlives its model.

**Answer**
: What came back, whichever verb asked: a `Result` for one solve, a
  [`Sweep`](#sweeps) for a sweep.

**Archive**
: A spec, the data it was solved with and what came back, written together
  as one zip or one directory by `archive=`
  ([archiving](../howto/archiving.md)).

## The verbs

**check** · **build** · **solve** · **write**
: `check(spec)` validates the spec with no data. `build(spec, sources)`
  returns a [Model](#the-chain). `solve` and `write` build and then solve or
  write a file in one call ([the verbs](api.md#run-a-spec)).

**evaluate**
: Values an expression with no solver: of a spec that declares no variables,
  or of a `Result` at its solution.

**update**
: `model.update(sources)` puts new numbers on a built model in place, naming
  only what changed. A change that moves a mask builds again.

**load** · **scan**
: The two ways a saved answer is read back. `load_…` reads it whole;
  `scan_…` reads each frame when asked, so the files have to outlive the value
  ([reading one too big to hold](../howto/archiving.md#read-one-too-big-to-hold)).

## The data

**Sources**
: The data you attach: a `dict` from each name the spec declares to a table,
  and from each dimension to its labels ([the data contract](data.md)).

**Index**
: A dimension's labels in order, each once, under the dimension's own key in
  `sources`. `shift` reads that order
  ([where coordinates come from](data.md#where-coordinates-come-from)).

**Coordinate**
: One point of a declaration's dimensions: one snapshot for one generator. A
  parameter has a value at each coordinate it covers, or no row there.

**Table**
: A polars `DataFrame` with one column per dimension and a `value` column,
  one row per coordinate: what a parameter arrives as, and what `primal` hands
  back. A relation's table has one column per column it declares.

**Relation**
: A named map between dimensions, such as the bus each generator sits on,
  supplied as a table of the rows it has.

**Mask**
: The `where:` on a declaration. A coordinate it excludes has no row
  ([absence](https://mathspec.readthedocs.io/en/latest/reference/language/absence/)).

## How it runs

**attach**
: Fitting sources onto a spec to make a [Model](#the-chain): what `build`
  does, and `update` does again.

**Sink**
: Where a built model goes: a solver (`highs`, `gurobi`, `xpress`) or a file
  writer (`.lp`, `.mps`). What a sink can take is its **capability**, such as
  special-ordered sets ([what each sink takes](api.md#what-each-sink-takes)).

**start**
: What a solve begins from: `'previous'`, the default, carries on from the
  solve before, and `None` begins from nothing. A start changes how soon the
  solver reaches the optimum, never the optimum
  ([making a loop of solves faster](../howto/warm-start.md)).

## Sweeps

**Sweep**
: What [`solve_over`](sweeps.md) returns when it solves one spec once per
  slice. It has `Result`'s readers. A windowed sweep answers over the labels of
  the dimension it cut; any other sweep is keyed by slice.

**Axis** · **slice** · **key**
: An axis says how the sources split: `EachCoordinate(dim)`, one slice per
  label, or `EachWindow(dim, ...)`, one per window of consecutive labels. A
  slice is one set of sources, solved as one model, and its key names it in
  every table the sweep hands back.

**carry**
: An axis's `carry={parameter: variable}` hands one slice's solution to the
  next as data.

**held** · **spilled**
: A held sweep keeps its answers in memory. A spilled one writes them to the
  directory `spill_to=` names, and `sweep.scan(name)` reads them lazily
  ([spilling](sweeps.md#spilling-a-sweep-to-disk)).
