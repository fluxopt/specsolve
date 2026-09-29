# How this suite is built

Four tiers. A new test lands in the cheapest tier that can catch its failure,
using the harness that serves it — not a hand-rolled copy.

| tier | the claim | the harness |
|---|---|---|
| **corpus sweeps** | every model in the repo loads, round-trips, stays inside the language, and its gallery page is current | `conftest.SPEC_PATHS` (= `tools.constructs.models()`) — one list, so a model added anywhere is covered the day it lands |
| **differential** | the same YAML means the same thing on the linopy lane and the relational one — the same objective, and the same shape in columns and rows — and in the written LP file, with `lp=True` | `tests.differential.differential()`; importing it *is* the `[linopy]` guard. `at_a_point()` for a read: no solver, every coordinate compared |
| **probes** | one mechanism each, pinned on the smallest model whose data can reach it | `conftest.DISPATCH_SPEC` + `override`, or a purpose-built module constant |
| **goldens** | an example prints what the docs show | `conftest.run_example` + `assert_golden`; regenerate with `--update-golden` |

`examples/ports/` cuts across the tiers: the one corpus checked against optima
that did not come from us (`conftest.port`, `references.json`) — data and
reference committed *because* the provenance is external.

**Across the lanes, compare the model — not the answer.** The corpus sweep
compares the two lanes' **coefficient matrices**, constraint by constraint,
canonically: each constraint as the sorted multiset of its rows, each row as
the sorted multiset of its coefficients, so the claim survives the two lanes
numbering rows and columns in their own declaration order. Structure exactly,
values to a tolerance — the same coefficient reached by a different order of
operations differs in its last bit, and that is not a difference in the model.

That is the strongest cross-lane claim available, and the one an *answer*
cannot make. An LP with alternative optima has many optimal primal and dual
solutions, so comparing answers compares which vertex a solver happened to
reach: `genx_piecewise_fuel` agrees on the objective to nine decimals and
disagrees on a quarter of one dual vector, while its two matrices are
identical to the entry ([#992]). Duals and primals are therefore **never**
compared lane to lane. A read at a point is the exception that proves
it: `at_a_point()` holds every variable at a seeded value rather than asking a
solver for one, so the point is the test's choice, and each lane's value of an
expression at it is a claim about the lane alone. That comparison sees a term in
a row that never binds, which an objective cannot.

A *recorded* dual is a different claim: that this instance has a **unique**
one, which is a property of the instance and something a port designs for
(#938). Both lanes owe it the same
answer, and both are asked — `test_ports` of the relational lane,
`test_corpus_parity` of the linopy one, which linopy-free `test_ports` cannot
reach.

[#992]: https://github.com/fluxopt/specsolve/pull/992

## Rules

- **A probe model is purpose-built.** A shared model cannot be assumed to
  express a failure: no data change moves a coefficient of
  `examples/dispatch.yaml`, because they are all 1 (#658). Build the smallest
  model whose data reaches the guard, with data small enough to read in a
  failure.
- **A probe moves to `conftest.py` on its second importer.** Not before.
- **A claim decided at `to_spec` lives in mathspec's own suite** (#1150),
  and nothing there imports a consumer — including `specsolve` itself, whose
  top-level namespace is the runner. The door the claim is *decided at* is the test, not
  the subject it is about: "a stray dim is refused" is the language's, while
  "`check` reaches that refusal with no sources bound" is `check`'s and belongs
  beside it. The package docstring carries the rule and the two tests that
  deliberately span both halves.
- **Data is generated, seeded and feasible by construction**
  (`conftest.transport_data` is the model). Committed data needs one of two
  reasons: external provenance (the ports) or a golden whose diff is the
  review artifact.
- **linopy, xarray and pandas arrive through `tests.oracle`.** Its docstring
  says why any other spelling is a guard bypass; a fixture that hands out
  pandas imports it in its own body (`conftest.py`'s docstring says why).
- **A correctness guard lands with its mutation table** (AGENTS.md, Part 2):
  delete the guard, run the suite, show the result in the PR — a deletion the
  suite survives gets a probe first.
