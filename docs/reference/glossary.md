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
  every verb takes it first.
  A `Spec` carries its own program, so one handed back to a verb is not read
  again. What it may contain is
  [the language](https://mathspec.readthedocs.io/en/latest/reference/language/).

**Program**
: The spec lowered to the plan a build reads its rows off: what [`check`](api.md)
  returns, still with no data, for reading the plan. No verb takes one back:
  lowering has no inverse, so keep the `Spec`
  ([`check`](api.md#specsolve.check)). The two states are the
  language's
  ([`Spec` and `Program`](https://mathspec.readthedocs.io/en/latest/reference/reading/#spec-and-program)).

**Formulation**
: A block that states rows nothing lowers, `piecewise:` today. Every verb
  refuses a spec still carrying one; `to_spec(spec).expand('piecewise')` or
  `.expand()` writes it out first
  ([`check`](api.md#specsolve.check)).

**Model**
: A spec with data attached, the language's own meaning of the word
  ([glossary](https://mathspec.readthedocs.io/en/latest/reference/glossary/)).
  These docs use it in no other sense. `specsolve.Model`, what
  [`build`](api.md) returns, is one. One `Model` feeds any sink through
  `solve()` or `write(path)`; `row(...)` and `diagnostics()` read it without
  solving.
  `update(...)` puts new numbers on it in place.

**Result**
: One answer read back from a solve: `objective`, `primal(name)`,
  `dual(name)`, `evaluate(expression)` and the rest of
  [`Result`](api.md#specsolve.Result). It owns its tables, so it
  outlives its model.

**Answer**
: What came back, whichever verb asked: a `Result` for one solve, a
  [`Sweep`](#sweeps) for a sweep. `Result.save` writes one as a directory —
  `record.parquet` for how it terminated, then `primal/`, `dual/`,
  `activity/` and `expression/` — and an archive holds that directory as
  `answer/`. The archive of a sweep holds its answer at the same paths, one
  file per name.

**Archive**
: A spec, the data it was solved with and what came back, written together as
  one zip or one directory by `archive=` ([archiving](../howto/archiving.md)).
  It reads back as a `SolveArchive`, or a `SweepArchive` where the sources were
  cut. Every table it holds carries `specsolve_run`, the archive's own name,
  stamped when it is written. Never "artifact".

**Digest**
: A hash that says whether two things are the same input. `spec_digest` names
  the document an answer came from, and an archive whose answer names another
  is refused; `archive.source_digests` names each data member, so two archives
  of one spec say which input moved.

## The verbs

**check** · **build** · **solve** · **write**
: `check(spec)` validates and lowers. `build(spec, sources)` returns a
  [Model](#the-chain), whose `check(sink)` asks whether a sink takes it.
  `solve` and `write` build and then solve or stream in one call. There is no
  Python API for constructing a spec. Each has
  [its entry](api.md#run-a-spec).

**evaluate**
: `evaluate(spec, sources, expression)` values one expression of a spec that
  declares no variables: arithmetic on the attached data, no solver.
  `result.evaluate(expression)` is the same read at a solution.

**update**
: `model.update(sources)` puts new numbers on a built model in place, naming
  only what changed. A change that moves a mask rebuilds and solves cold.

**load** · **scan**
: The two ways a saved answer is read back. `load_result`, `load_sweep` and
  `load_archive` read **whole**, so the directory is free afterwards.
  `scan_result`, `scan_sweep` and `scan_archive` read each frame at the call
  that asks for it, so the files have to outlive the value
  ([reading one too big to hold](../howto/archiving.md#read-one-too-big-to-hold)). Never "open".

**Buildable**
: The type alias for a spec argument: `str | Path | Mapping | Spec`. The
  lowered `Program` that `check` returns is not one.

**Source**
: The type alias for one value of `sources`. The shapes it covers are
  [the data contract](data.md#what-a-parameter-accepts).

**Label**
: One member of a dimension, `wind` say, and its type alias:
  `int | float | str | datetime`. A sweep's slice key is a label too, and
  `EachCoordinate(dim)` slices on one label of `dim` at a time.

## The data

**Index**
: A dimension's labels in order, supplied under the dimension's own key in
  `sources`. `shift` reads that order positionally
  ([the data contract](data.md#where-coordinates-come-from)).

**Coordinate**
: One point of a declaration's dimensions: one snapshot for one generator. A
  parameter has a value at each coordinate it covers, or no row there. The
  language calls the dimensions themselves the declaration's *frame*
  ([named expressions](https://mathspec.readthedocs.io/en/latest/reference/language/named/#expressions)).

**Table**
: A polars `DataFrame` with one column per dimension, a `value` column and one
  row per coordinate: what a parameter arrives as, and what `primal` hands
  back. A [relation](#the-data)'s table is the exception, one column per
  column it declares. The code calls one a **frame** and means the same thing.

**Relation**
: A named map between dimensions, supplied under its own key as a table of the
  rows it has. A keyed relation declares key columns and value columns and
  holds one row per key; a bare one holds each row at most once
  ([the data contract](data.md#where-coordinates-come-from)).

**Assumption**
: An `assumptions:` entry: a predicate on the data that only the numbers can
  answer, checked when data attaches and refused as a `DataError` where it
  fails. The conditions a `piecewise:` method puts on its breakpoints arrive
  the same way.

**Mask**
: The `where:` on a declaration. What an excluded coordinate means is
  [absence](https://mathspec.readthedocs.io/en/latest/reference/language/absence/).

## How it runs

**Lane**
: A way a spec is executed. specsolve's is the **relational lane**: it
  validates at load time, lowers to the plan and streams on polars. The test
  suite's **linopy lane** builds the same spec as a `linopy.Model`, as the
  oracle the relational lane is checked against
  ([relationship to linopy](../about/linopy.md#2-it-is-the-oracle)).

**Engine**
: The relational lane's builder: it fills the model's tables from the attached
  data and hands them to a sink.

**Sink**
: Where the [handoff](#the-built-form) lands: a solver (`highs`, `gurobi`,
  `xpress`) or a file writer (`.lp`, `.mps`). `linopy` is a lane, not a sink.
  What a sink can ingest is its **capability**: a special-ordered set is one,
  and a sink without it refuses a built model carrying a set rather than
  rewriting it ([what each sink takes](api.md#what-each-sink-takes)).

**Sources**
: The data you attach: parameter, dimension and relation names to tables, and
  dimension names to their labels.

**attach**
: Fitting sources onto a spec to make a [Model](#the-chain); what `build` does
  and `update` does again. Never "bind", so that `bound` means one thing.

## The built form

**Handoff**
: The built model as a sink sees it, and all a sink sees: `cols` (bounds,
  type), `obj`, `rows`, `matrix` (CSR), `quad` (the objective's quadratic
  part), `qmatrix` (the constraints') and `sos`. Handing it over is the phase
  the metrics clock as `handoff_seconds`.

**keep**
: How much of a session `model.solve` carries to the next solve: `solver`
  (default), `progress` (its work too) or `nothing`
  ([`Model.solve`](api.md#specsolve.Model.solve)).

## Sweeps

**solve_over** (a sweep)
: Solve one spec once per slice of an axis and fold the answers into a
  `Sweep`, releasing each slice's model as it goes. A sweep, never a "study"
  ([sweeps](sweeps.md)).

**Axis** · **slice** · **key**
: An axis says how the sources split: `EachCoordinate(dim)`, one slice per
  label; `EachWindow(dim, ...)`, one per window of consecutive labels; or a
  hand-built list of `(key, sources)` pairs. A slice is one set of sources,
  solved as one model, and its key is the [label](#the-verbs) its rows are
  prefixed with in every table the sweep hands back.

**Sweep** · **per window**
: What `solve_over` returns: `Result`'s readers, each returning the answer.
  A windowed sweep answers over the labels of the dimension it sliced. Any
  other sweep answers keyed by slice, the key column first. `per_window=True`
  reads a windowed sweep one window at a time, keyed by where each window
  started, lookahead rows included.

**carry**
: `carry={parameter: variable}` hands one slice's solution to the next as
  data, in slice order.

**held** · **spilled**
: Where a sweep's frames are. A **held** sweep carries them in memory, and
  `sweep.primal(name)` and the exports — the **frame readers**, the ones that
  hand back a table — answer off them. A **spilled** sweep left them in a
  directory, which is what `spill_to=` writes and what `scan_sweep` reads: there
  `sweep.scan(name)` is the reader and the frame readers refuse
  ([spilling](sweeps.md#spilling-a-sweep-to-disk)).

## Row types

**Record** · **Metrics**
: The two saved rows, each a `NamedTuple` that names its own columns:
  [`Record`](api.md#specsolve.relational.parquet.Record), how a solve
  terminated, and [`Metrics`](api.md#specsolve.relational.parquet.Metrics),
  what it took. A sweep writes one of each per slice, with the same columns
  as a single solve; `slice_axis` and `slice` say which slice.

  A **row** is a value and gets a type; a **table** stays a
  [Table](#the-data). So a result hands back its one `Record`, while
  `Record` and `Metrics` are the rows behind `sweep.record` and
  `sweep.metrics` rather than what those hand back, and a reader that wants
  one row of a table asks the frame for it.

## `bound` means one thing

**bound**
: A lower or upper limit on a variable or a constraint row: the `bounds:` of
  a declaration, the `BOUNDS` section of an `.mps` file, an absent bound the
  solver reads as infinity. Nothing else; data is
  [attached](#how-it-runs), never bound.
