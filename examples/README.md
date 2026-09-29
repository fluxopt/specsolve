# Examples

**These are fixtures, not samples.** The test suite loads them by path and
`docs/examples/` embeds them, so the file you read here is the file CI runs.
Renaming one breaks tests; changing one changes what the docs claim, and a test
will say so.

**A new model says which side it is on.** The language is being extracted to
[mathspec](https://github.com/energy-models/mathspec), and almost everything
here stays: a model with a gallery page is read by a person and solved by an
engine, and both of those are specsolve's. What moves is
`operators/` — one construct per file, no page — because the operator
reference is *generated from* it. Name a new model in
[`extraction.paths`](../extraction.paths) or in `EXAMPLES_THAT_STAY`
(`tests/test_architecture.py`); a model in neither fails the suite rather than
defaulting to a side.

**Read them explained** in [docs/examples/](../docs/examples/index.md) — the maths,
what each construct exercises, and for a port a side-by-side against the
reference implementation. This directory is the source; that is the guided tour.

| | |
|---|---|
| `dispatch.yaml` | least-cost generation against a load profile — the smallest complete spec |
| `storage.yaml` | dispatch plus a cyclic battery (`shift(edge='wrap')`) |
| `transport.yaml` | a network: coordinates on a dimension *are* the topology (`sum(by=)`) |
| `piecewise.yaml` | per-generator convex cost curves (`piecewise:`) |
| `sos.yaml` | the same curve stated as a set the solver branches on, rather than built from binaries (`method: sos2`) |
| `piecewise_lp.yaml` | the same curve stated as the lines its segments lie on, which declares no weights at all (`method: lp`) |
| `piecewise_conversion.yaml` | converters whose flows are tied to one curve each, where how many flows a converter ties is data rather than a `links:` list (`sos:`, `at()`) |
| `monthly_budget.yaml` | a cap per calendar month: time grouped through a coordinate, exactly as a generator sits on a bus (`sum(by=)`) |
| `multi_period.yaml` | capacity decided once per investment period and binding at every snapshot in it (`at()`) |
| `reserves.yaml` | energy and reserves co-optimized on a two-bus grid: every many-to-many shape at once — a pair set reified as a dimension whose legs are relations, and weighted membership left as data (`relations:`, `at()`) |
| `walkthrough.yaml` | the spec `walkthrough.py` prints every pipeline stage for |
| `rolling/` | a storage schedule solved a window at a time, and what the lookahead buys (`solve_over`, `EachWindow`) |
| `myopic/` | an investment pathway over periods of typical days, each inheriting the last one's fleet (`solve_over`, `carry`) |
| `benders/` | the problem split in two and reassembled, checked against the monolith it decomposes |
| `operators/` | one minimal spec per operator in [the operator reference](https://mathspec.readthedocs.io/en/latest/reference/language/operators/), which is where the math on that page comes from |
| `ports/` | 33 models somebody else already solved, checked against an optimum that did not come from us |

`walkthrough.py` runs one model through YAML → schema → AST → plan → frames →
LP text → solution, printing what each stage produces, then two specs the
language refuses and why. Its output is committed as `walkthrough.out` and
asserted, so it cannot drift from what the code does:

```bash
python examples/walkthrough.py
```

The other direction — not how a model is built but how one is *changed* — is a
page rather than a script, and lives with the docs it belongs to:
[docs/change.md](../docs/change.md) runs `dispatch.yaml` through the
three loops a session has, and
[docs/howto/fix-relax-remove.md](../docs/howto/fix-relax-remove.md) aims them
at linopy's `fix`, `relax` and `remove_constraints`. Both hold their code
in blocks the site runs during the build, so what a reader sees rendered is
what that build produced.

`ports/` carries three or four files per model — the YAML, the instance, the
recorded objective with its provenance, and a reference implementation
importing no specsolve. That last one is absent where the optimum is *published*
and needs nothing of ours to reproduce it: `facility_location` (OR-Library) and
`tsp_mtz` (TSPLIB) cite the literature instead, which is the strongest tier the
corpus has.

Reference scripts are **never run by CI** and carry their dependencies inline;
adding a port is described in
[CONTRIBUTING.md](../CONTRIBUTING.md#adding-a-ported-model).
