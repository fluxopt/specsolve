# Glossary

The one definition of each word specsolve uses, in its code, its messages and
these pages. The language itself is
[mathspec's](https://mathspec.readthedocs.io/en/latest/reference/language/);
the words for it are defined here, and
[the words this project does not use](#words-this-project-does-not-use) end
the page.

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
: The math before any data: a YAML file, a mapping, or a `Spec`. What it may
  contain is
  [the language](https://mathspec.readthedocs.io/en/latest/reference/language/).

**Program**
: What [`check`](api.md#specsolve.check) returns: what the spec compiles to,
  and what a build reads. No function takes one back, so keep the `Spec`.

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

## The language

**Entry**
: One named item of a spec, under one of its sections: a dimension, a
  parameter, a variable, a constraint, a named expression.

**Dimension**
: A set of labels that entries are indexed over, such as `generator`.

**Label**
: One member of a dimension, such as `wind` in `generator`. A dimension's
  labels arrive in `sources` under its name, in order, each once
  ([where coordinates come from](data.md#where-coordinates-come-from)).

**Coordinate**
: One point of an entry's dimensions: one snapshot for one generator.

**Parameter**
: An entry the data fills: one number per coordinate of its dimensions.

**Variable**
: An entry the solver decides: one column of the model per coordinate.

**Constraint**
: An entry the solution must hold: one row of the model per coordinate.

**Relation**
: A named map between dimensions, such as the bus each generator sits on. Its
  rows arrive with the data.

**Condition**
: A true-or-false expression, such as `p_max > 0`.

**`where:`**
: Narrows an entry to the coordinates where its condition is true, and
  leaves the entry absent elsewhere
  ([absence](https://mathspec.readthedocs.io/en/latest/reference/language/absence/)).

**Absent**
: Holding no value at a coordinate. An absent value is not zero: a `where:`
  leaves an entry absent, and so does a missing row under `missing: absent`.

**Operator**
: A construct built into the language, such as `sum`, `at` or `shift`. A file
  cannot add one ([operators](https://mathspec.readthedocs.io/en/latest/reference/language/operators/)).

**Compile**
: To turn a spec into its program. Every function compiles the spec first, so
  a spec the language refuses is refused before any data is read.

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

**Table**
: A polars `DataFrame` with one column per dimension and a `value` column, one
  row per coordinate.

## How it runs

**Engine**
: Software that attaches data to a spec, builds the model and hands it to a
  solver. specsolve is one.

**attach**
: Fitting sources onto a spec to make a [Model](#the-chain), in `build` and
  `update`.

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

**handover**
: The last label of the windowed dimension that a window owns. A carry reads
  the window's solution there and hands it to the next window as data.

**held** · **spilled**
: A held sweep keeps its answers in memory. A spilled one writes them to
  `spill_to=` ([spilling](sweeps.md#spilling-a-sweep-to-disk)).

## Words this project does not use

Each word on the left has a word above that means the same, or is vague.
`tests/test_words.py` refuses it in code, messages and pages, and names the
word to write. A plural is refused with its word.

| Not | Write |
| --- | --- |
| `declaration` | entry |
| `mask`, `predicate` | condition, or `where:` |
| `masked`, `masked out`, `unbuilt` | absent |
| `unmasked` | with no condition |
| `primitive` | operator |
| `lowered`, `lowering`, `lowers` | compiled, compiling, compiles |
| `summand` | a term of a sum |
| `dim`, `dims` | dimension, dimensions; `dims:` is the key |
| `frame` | table, for a polars table; its dimensions, for what an entry is indexed over |
| `consumer` | a tool, or the code that reads it |
| `bind data` | attach data |
| `reject`, `rejected`, `error out` | refuse, refused |
| `normal form` | the canonical form |
| `streaming lane` | relational lane |
| `sayable` | in the language |
| `seam`, `surface`, `tax`, `closure` | the plain thing |

A dimension's labels are its labels, never its index: an index is a solver's
position of a column or a row, or a pandas index.
