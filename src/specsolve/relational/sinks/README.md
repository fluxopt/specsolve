# Sinks

How a built model leaves the engine — the boxes downstream of the engine in
[docs/about/architecture.md](../../../../docs/about/architecture.md)'s pipeline, which
carries the argument for the split. This page is the membership list.

**Two families.** A **solver** takes the handoff and runs it; a **writer**
takes the handoff and renders it to a file. Everything else follows.

| | solvers/ | writers/ |
|---|---|---|
| answers | a `Solver` subclass holding one model | `(handoff, path) -> None` |
| chosen by | **name**, at the call — `solver_name='gurobi'` | **suffix**, from the output — `tables.lp` |
| registry | `SOLVERS`, closed, holding the classes | `WRITERS`, closed |
| members | `highs.py` (`highspy`, ships), `gurobi.py` (`[gurobi]`: `gurobipy`, `scipy`), `xpress.py` (`[xpress]`), over `base.py` | `lp_file.py`, `mps_file.py` (nothing beyond polars), over `base.py` |

## Staying loaded

`base.py` is what a solver **is**: a loaded model with a lifecycle, which is
linopy's shape and its word — their `Solver` is the persistent object too. It
holds no solver of its own, which is the whole reason it is allowed to exist
beside the leaves: it cannot carry an optional dependency across the fence
below, and sharing through it is what stops one leaf importing the other.

The split is by who can answer. `solvers.loaded(held, name, …)` is the whole of
**reuse or load again**: it keeps a held solver exactly when it is the named
class whose recorded digest and options match the new handoff — and then pushes
the new numbers onto it — closing and replacing it otherwise. The base records
that evidence at the load; a subclass owns **the hand-off**:

| | |
|---|---|
| `solvers.loaded(held, name, …)` | reuse or load again — the whole of that decision |
| `Solver.run(tables)` | `_run`, plus the refusal of a vector that does not span the model |
| `Solver.warm(ws)` | `_warm`, plus the refusal of a `WarmStart` from another solver or another shape |
| `_load(tables, batch_rows)` | hand the model over and hold what reads it back |
| `push(tables)` | only after `loaded` matched the digest — new bounds, costs and right-hand sides |
| `_run(tables)` | solve what is loaded, and read it back |
| `warm_start()` | the basis the last solve left — the incumbent, after a MIP — or `None` |
| `_warm(ws)` | set it on the loaded model, spans already checked |
| `forget()` | discard the work the last solve did, keeping the model loaded |
| `close()` | drop the handle, and any licence with it |

The first three are the family's and identical for everyone; the last seven are a
member's, and are its own library's shape. Nothing above the family decides
which solver to keep or checks what one returned — an engine hands over a `Handoff`
and is given an answer.

So a model rebuilt with new numbers (`model.update`) has them pushed onto what
the solver already holds. Whether it also solves from the basis the last one
ended on is the caller's `keep=`: `'progress'` keeps it, and `'solver'` — the
default — calls `forget()` so the run begins as if the model were new. Every
member implements both; a solver with nothing to discard implements `forget()`
as a no-op.

`forget()` rather than a reload because the two costs are different ones. A
caller keeping the *solver* skips the hand-off, which nothing pays for; one
keeping its *progress* trades against whatever the member prepares for a run
that starts from nothing, and that trade goes either way by model. Splitting
them is what lets a caller take the first without the second.

A **genuine rebuild** gets no carry at all: the new session holds a fresh model
and starts cold, and `PolarsEngine.solve(keep='nothing')` is how a caller asks
for that on purpose — the held solver is discarded, so cold is structural
rather than scrubbed.

`warm_start()` / `warm(ws)` are the machinery for carrying one anyway:
`warm_start()` reads the basis — or, after a mixed-integer solve, the
incumbent, no solver leaving a valid basis behind one — out of a session as an
opaque `WarmStart`, and `warm(ws)` sets it on the next, refusing a start from
another solver or one whose spans do not match the ingested model. **Nothing
above the family calls either**, and `WarmStart` is deliberately not
re-exported: the case that wants a carry most, a cutting-plane master
re-solved after gaining a cut, gains a *row*, so the span check refuses it by
construction. [#382](https://github.com/fluxopt/specsolve/issues/382) holds what
has to be answered before this reaches a caller.

The guard is `Handoff.structure` — a digest of everything a re-solve may
not change, recorded by the solver at its load and cached on the handoff. **Values are re-pushed, not diffed**: linopy's persistent layer
(`persistent/diff.py`) computes a delta against a snapshot of the previous
model, where here the previous model is released before the new one exists, that
release being what keeps an updated build at one model's peak. What a diff would
need is exactly what is not kept.

`handoff.py` is what both read. Neither family imports the other and no member
imports a sibling — `tests/test_architecture.py` reads all of that off the
path, which is what keeps `gurobipy` off the import path of a caller who
solves with HiGHS.

## The contract

A sink takes a `Handoff` and nothing else: the frames `cols`
(col, lb, ub, vtype), `obj` (col, coeff), `rows` (row, sense, rhs), `matrix`
(row, col, coeff) and `sos` (set, type, col, weight), plus the counts it
chunks by and the objective's sense and constant — those last two live outside
the frames because a constant has no column to attach to.

A sink never learns how the handoff was filled, and the engine never learns
how it is drained. That is the point: `mps_file.py` is a module beside
`lp_file.py`, not another method on `PolarsEngine`.

The one thing sinks may share is a *projection* of those frames, never a step
of the work — `Handoff.dense_columns`, which every solver reads — or a
family `base`, which holds no member's own answer: `solvers/base.py` is the
lifecycle without a solver in it, `writers/base.py` the three renderings
without a format in them.

## Row-major, and the one format that is not

`matrix` is CSR, so every sink but one walks it by row and slices rather than
sorts. MPS is column-major — it hands a reader each column with its whole
column of the matrix — so `mps_file` sorts the matrix into `(col, row)` order
once and builds its own offsets from the result. That sort is the writer's
peak, and it is the only place in `sinks/` where the format, rather than the
engine, decides the order. The engine holds no column index because this is
its one consumer, and building one on every build to serve it would be paid by
every caller who never writes a file.

## The one uneven stream

Four of the five are the same question to every sink. `sos` is not: Gurobi
branches on a set, `lp_file` writes it as text, and HiGHS has no such concept
at all. A sink **declares** whether it takes one, in the descriptor *what a
sink can ingest* below gives it —

```python
'sos': 'native'  # gurobi: addSOS, no binaries and no bound to have
                 # highs: absent, and the refusal names the way past it
```

— and `sinks.refusal(program, name)` answers before the load. Nothing is
rewritten at the hand-off: the language writes a set out itself,
`to_spec(...).expand()` stating each as binaries and linking rows, and that is
the model a sink without the concept is given. `_load`, `push`, `_run` and the
span check see one model, the one that was built, so a member is written as
though the fifth stream were never there.

## How the three take the matrix

`matrix` is CSR, and two of the three want it that way: `highs.py` hands over
the three arrays and `xpress.py` hands `addRows` the same triple a block
already is. `gurobi.py` is the exception — its matrix API takes a matrix
*object*, which is what the `[gurobi]` extra's scipy is for.

The rest of what separates them is each library's own spelling, and the two
places it bites are worth naming because neither is a choice:

- **The objective's constant.** HiGHS and Gurobi have an attribute for it;
  Xpress spells it as the objective coefficient of column `-1`, *negated*.
- **Discarding a solve.** `Model.reset()` on Gurobi keeps the model and drops
  the solution, which is exactly `forget()`. `problem.reset()` on Xpress
  clears the problem itself, so there `forget()` is the `keepbasis` control
  instead.

## Adding one

**A solver:** `solvers/<name>.py` named for the solver, defining a `Solver`
subclass named for it — `_load`, `push`, `_run`, `close` — plus the
`build_<name>` seam `bench/` measures, and one line in `SOLVERS` holding the
class.
Import the solver **inside the function** and declare an extra for it — the
module boundary is the fence, the lazy import is what keeps this package free
to import for callers who will never use it. Copy linopy's status map for it
and pin the copy in `tests/test_solve_status.py`, including anywhere you
deliberately diverge.

**A writer:** `writers/<format>.py`, one line in `WRITERS` keyed by suffix,
holding a `Writer(write, capabilities)` — a function has nowhere to carry a
fact about itself, so the pair travels together. Render through `base.py`
rather than casting in the module — that is what makes two files describe one
model to a reader holding both.

Either way it declares what it can ingest, and stream — nothing here may
materialise the model a second time — and nothing above changes. No method on the engine, no branch in `api.py`,
no name on the Python surface.

## What a sink can ingest

`capabilities.py` is the second axis of [the
ceiling](https://mathspec.readthedocs.io/en/latest/about/what-counts-as-language/#what-each-tool-decides-for-itself) —
what a sink takes, as against what the language may say — and its docstring is
where the entries are argued. One descriptor per sink, declared in the sink's
own module: a `ClassVar` on a `Solver`, a field on a `Writer`.

Three callers read it, and between them a construct a sink has no spelling for
cannot reach that sink by any door: `check(spec, sink=...)` before any data is
attached, and the engine's `solve` and `write` — which ask without being asked,
a file written without the section being a different model that parses and
solves.

## Stable output

Two runs of one model produce the same bytes
([#109](https://github.com/fluxopt/specsolve/issues/109)). It is not free and it
is easy to lose: a parallel join hands back a group in whatever order it
finished it, so a sink that gathers a row's terms and *then* orders the rows
has already lost the order within one. `lp_file` emits one frame of lines
carrying its own sort key instead, and sorts once. The solvers are ordered for
a different reason — `searchsorted` requires it, and the CSR `indptr` it
produces is only a row's extent if the rows it indexes are sorted.
