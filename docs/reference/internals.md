# Internal glossary

The names the code uses for its own parts, for anyone reading `src/` or
[the architecture](../about/architecture.md). The names a caller meets are in
[the glossary](glossary.md); this page does not repeat them.

Each entry says what the thing is, then the module it lives in.

## The two builds

**Lane**
: One way to build a spec. The **relational lane** is the package: it builds a
  spec into frames and streams them to a sink (`relational/`). The
  **oracle** is the test suite's second lane: it builds the same spec as a
  `linopy.Model` (`tests/linopy_lane/`). The tests compare the two.

**The door**
: The one function both lanes read data through, `sources.tidy_sources`. It
  turns every shape a caller may pass into one tidy table per name, and makes
  every check on the data there (`sources.py`).

**Lowering**
: Turning a spec into its `Program`. `inputs.lowered` is the one gate every
  verb and the oracle call, so both refuse the same specs. `inputs.lower`
  lowers one expression the file never named, by adding it to the spec as a
  named expression first (`inputs.py`).

## The engine

**Engine**
: What turns a `Program` and the tidy tables into a `Handoff`, and reads a
  solve back. There is one, `Engine` (`relational/engine/engine.py`).

**Attached sources**
: The tidy tables encoded the way every query reads them: a dimension as
  `(val, ord)`, a label column as an `Enum` in declared order
  (`relational/engine/attaching.py`).

**Scope**
: What a name stands for while the compiler works: the program, the attached
  sources and the variable frames built so far. It also holds the one
  row-major rule every index reads (`relational/engine/scope.py`).

**Compiler**
: What turns a plan node into lazy polars queries. It reads no data
  (`relational/engine/compiler.py`).

**Piece**
: One additive part of a compiled expression: linear terms
  (`var_label`, `coeff`), quadratic terms (`var_label`, `var_label_2`,
  `coeff`), or a constant (`cval`), each over its dims. An expression compiles
  to a list of pieces, a `CompiledExpression`. "Fragment" is not used here,
  because the language uses it for a partial spec
  (`relational/engine/pieces.py`).

**Presence**
: Where the variable under a piece exists. A piece's frame cannot say this,
  because a missing row is a zero coefficient when a parameter is sparse and
  an absence when a variable is masked (`relational/engine/pieces.py`).

**Region**
: The part of a `cases:` block a piece covers. The coverage check asks its
  question only inside the region (`relational/engine/pieces.py`).

**Fan-in**
: How the output rows of a plan node relate to its input slots: one to one,
  many to one, or one to many. A node that is not one to one sums several
  slots into a row, so absence is propagated before it
  (`relational/engine/pieces.py`).

**Shift**
: Moving a piece's rows along one dimension's own order. `shift` is one
  shift, and `sum_back` is a sum of shifts (`relational/engine/shifts.py`).

**Label**
: The solver's index of one column or one row. Each declaration owns a
  contiguous run of labels, so its share of a solver vector is a slice
  (`relational/engine/labels.py`).

**Assembly**
: One build in progress: every declaration turned into rows of the model
  tables. It is discarded once the build is frozen into a `BuiltModel`
  (`relational/engine/assembly.py`).

**Coverage**
: Whether data is there where a declaration reads it. A divisor and a
  constant piece are refused at the last moment the gap can still be seen
  (`relational/engine/coverage.py`).

**Collect engine**
: Which polars engine materialises a frame: polars' own `auto` choice where
  this polars has the streaming engine, the in-memory one otherwise. A collect
  can ask for the in-memory engine by name (`in_memory=True`). `collected()` is
  the one way a frame is collected, on that engine and with polars' join
  reordering off. `collect_engine()` is not the `Engine`
  (`relational/collect.py`).

## Sinks

**Handoff**
: The built model as every sink reads it: the column, objective, row and
  matrix tables, the quadratic tables and the SOS sets
  (`relational/sinks/handoff.py`).

**Sink**
: Where a handoff goes. A **solver** runs it and reads an answer back, chosen
  by name (`relational/sinks/solvers/`). A **writer** renders it to a file,
  chosen by suffix (`relational/sinks/writers/`).

**Capability**
: Something a model may need a sink to take: integrality, SOS sets, a
  quadratic objective, a nonconvex one, a quadratic constraint. Each sink
  lists the ones it takes, and a refusal names the sinks that take what this
  one does not (`relational/sinks/capabilities.py`).

**Structure digest**
: A digest of everything a re-solve may not change: the counts, the matrix,
  each row's comparison, each column's type and the sets. A loaded solver
  keeps its model only while this digest stays the same. The **contents
  digest** adds the numbers, and says whether a saved answer belongs to a
  rebuilt model (`relational/sinks/handoff.py`).

**Warm start**
: What a solve starts from instead of from scratch, laid onto the build by
  coordinate, so a model that gained or lost rows or columns takes it too: a
  basis or values for an LP, values for a mixed-integer model. `solve(start=)` is its
  caller (`relational/engine/readback.py`,
  `relational/sinks/solvers/base.py`).

## Answers on disk

**Kind**
: One of the frames an answer holds, named after the reader it comes back
  through: `primal`, `dual` and `expression` always, and each output the solve
  asked for with `outputs=`, such as `activity`. Each kind is also the
  directory its frames are saved under (`relational/answer_layout.py`).

**Output**
: What an answer holds only on request, named in `outputs=`. Each carries
  one kind, of its own name, except `basis`, which carries `variable_basis`
  and `constraint_basis`. `OUTPUT_KINDS` is the one table: each kind, the
  output that asks for it, and whether it holds a frame per variable or per
  constraint. The readers, `kind=`, the fold, the spill,
  the archive and its catalog all read it. `format.json` names the outputs an answer holds, so a reader tells "not
  asked for" from "not defined", and a new output adds no `ANSWER_LAYOUT` bump
  (`relational/answer_layout.py`).

**Answer layout**
: What a result and a sweep write: `<kind>/<name>.parquet`, the `Record` and
  `Metrics` rows, `reasons.parquet` and the `format.json` stamp.
  `ANSWER_LAYOUT` is its version, and a change to it raises the number
  (`relational/answer_layout.py`).

**Archive layout**
: What an archive holds: `spec.yaml`, `sources/`, `catalog.parquet`, the
  answer layout under `answer/`, `axis.json` for a sweep, and a `format.json`
  stamp of its own (`archive_layout.py`). `INPUTS_LAYOUT` is the version of
  `spec.yaml`, `sources/`, `sources.parquet` and `axis.json`, and a change to
  any of them raises it. Any other change, `catalog.parquet` included, raises
  `ANSWER_LAYOUT`. Reading an archive back is `archive.py`.

**Run stamp**
: The `specsolve_run` column an archive adds to every table it holds. Names
  that start with `specsolve_` are reserved for columns like it
  (`relational/answer_layout.py`).

## Sweeps

**Axis**
: What cuts the sources into slices: `EachCoordinate` or `EachWindow`. A
  hand-built list of `(key, sources)` pairs is the other form `axis=` takes
  (`axes.py`).

**Slice**
: One model of a sweep: its key, its sources, and how many coordinates of
  the windowed dimension it owns (`axes.py`).

**Stitch**
: How a windowed sweep's frames are put back over the dimension the windows
  cut, keeping each coordinate from the window that owns it and dropping the
  lookahead rows (`axes.py`).

**Slice answer**
: One slice, solved and read out as plain frames, so it can cross a process
  (`sweep.py`).

**Spill**
: A sweep's answers on disk instead of in memory, one file per slice and
  name. A spill is also how an interrupted sweep resumes (`sweep.py`).

**Carry**
: One slice's answer copied into the next slice's data, through a
  `{parameter: variable}` mapping. The seam is the last coordinate a slice
  owns (`strategy.py`).

**The wire**
: How a slice's sources cross to an executor's worker: a path stays a path
  where the worker can read it, and a table travels as parquet bytes
  (`strategy.py`).
