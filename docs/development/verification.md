# Verification

How we know an answer is right. Four checks, from the cheapest to the most
convincing. Each one catches what the one before cannot.

| Check | It catches | Where |
|---|---|---|
| [The oracle](#the-oracle) | a build that differs from linopy's on the same file | `tests/linopy_lane/`, the differential tests |
| [Published optima](#published-optima) | a misreading the oracle shares | `examples/ports/`, [the gallery](../examples/index.md#does-it-get-the-right-answer) |
| [The PyPSA ladder](#the-pypsa-ladder) | a model that solves but is not PyPSA's | `differential/pypsa/`, [the ladder](../examples/pypsa_ladder.md) |
| [Decomposition](#decomposition) | an algorithm outside the language that cannot be written over it | [`examples/benders/`](https://github.com/fluxopt/specsolve/blob/main/examples/benders/README.md) |

## The oracle

The test suite builds every file twice: through the relational engine, and as
a `linopy.Model` in `tests/linopy_lane/`, from the same `sources`. The
differential tests compare the two. linopy is a test dependency, and nothing
under `src/` imports it.

The comparison checks one language only because both builds pass the same
`inputs.lowered` gate. A construct one build refuses, the other refuses in the
same sentence. Both read data through `sources.tidy_sources`, so a malformed
source gets the same refusal from both
([#60](https://github.com/fluxopt/specsolve/issues/60)).

**Where a concept is already linopy's, specsolve copies its name**: solve
statuses, `status` and `termination_condition` with `is_ok` as the rollup, the
shape of a result. The engine may not import linopy, so the tables are copied,
and `tests/test_solve_status.py` imports linopy to check the copy still
matches.

### What a construct becomes in the oracle

What the oracle calls for each thing a file can say. Each row lives in
`tests/linopy_lane/builder.py`.

| Declaration | linopy |
|---|---|
| `variables:` | `Model.add_variables(lower, upper, coords, name, mask, binary, integer)` |
| `sos:` | `Model.add_sos_constraints(variable, sos_type, sos_dim)`, the block handed over rather than a formulation rebuilt |
| `constraints:` | `Model.add_constraints(lhs, sign, rhs, name, mask)`, one rule per declaration |
| `objective:` | `Model.add_objective(expr, sense)`, each additive term summed over the dims it carries |
| `expressions:` | evaluated at the solution as xarray arithmetic, every variable its `.solution` and every `dual(c)` the constraint's `.dual`; an entry the math never reads is read at whatever degree it was written |

| In an expression | linopy or xarray |
|---|---|
| `x` — a variable | `Model.variables['x']`, `.fillna(0)` under `absence: zero` |
| `p` — a parameter | its `xr.DataArray`, `.fillna(0.0)` where it stands as a coefficient |
| `+` `-` `*` `/` | the Python operators linopy overloads |
| `sum(x, over=t)` | `.sum('t')` |
| `sum(x, by=r, over=c, into=d)` | the column walked to attached as a coordinate, then `.groupby()`, reindexed onto its dimension's declared labels; a walk to two columns groups onto both at once, and a key of several columns groups by the value and the rest of the key |
| `at(p, by=r, over=c, into=d)` | `.sel({into: relation})`, xarray's vectorised selection; a read of two columns reads a tuple of labels at once |
| `shift(x, along=t, offset=n)` | `.shift({t: n})`; `.roll({t: n})` under `edge: wrap`; a `.sel()` gather where the offset differs per entity or `by=` groups it |
| `sum_back(x, along=t, window=w)` | a sum of `w` scalar gathers, each unreachable position contributing zero; under `by=` each gather reads inside the group, so the window stops at its edge |
| `dual(c)` | `Model.constraints['c'].dual`, at a read only; the language keeps a dual out of the math, and a solve that stored none refuses the read |

| A `where:` | linopy |
|---|---|
| on a declaration | the `mask=` argument; a mask that excludes nothing is passed as `None` |
| `defined(x)` | `Model.variables['x'].labels != -1`, linopy's own marker for an absent slot |
| a comparison | the Python comparison operators element-wise, absence reading as false |

A missing parameter row is zero in a coefficient, an error in `bounds:`, and
false in a `where` operand. `tests/linopy_lane/absence.py` holds each spelling.

### The same language, but not every build

Three files pass `check` and build on one side only. Each refusal names the
rewrite.

- **An objective with a constant** builds in specsolve and not in the oracle,
  because `linopy.Objective` has no slot for one. The oracle refuses it rather
  than drop the constant (`osemosys_utopia`,
  [#894](https://github.com/fluxopt/specsolve/issues/894)).
- **A relation that is not a single-valued map** builds in specsolve and not in
  the oracle, which holds each relation as one dense array
  (`tests/linopy_lane/loader.py`).
- **An operator along a dimension a constant part does not carry**, such as
  `sum(x * k + d, over=t)` with a scalar `d`, builds in the oracle and not in
  specsolve. The refusal says to declare the parameter over the dimension
  ([#1137](https://github.com/fluxopt/specsolve/issues/1137)).

## Published optima

The oracle cannot catch a misreading both builds share. A published optimum
from outside can. Each model in `examples/ports/` carries the optimum somebody
else solved, with its source, in `references.json`, and `tests/test_ports.py`
solves to it. The table is
[Does it get the right answer?](../examples/index.md#does-it-get-the-right-answer)
on the gallery page.

## The PyPSA ladder

[The PyPSA ladder](../examples/pypsa_ladder.md) grows mathspec's one-file PyPSA
model a feature at a time. Each rung solves the same network in PyPSA and in
specsolve, and compares the objective, and the duals where PyPSA reports them.
`tools/ladder.py` writes the pages from `differential/pypsa/`. The ladder is
the proof that the language holds a real model.

## Decomposition

[`examples/benders/`](https://github.com/fluxopt/specsolve/blob/main/examples/benders/README.md)
writes Benders decomposition as a loop over three specs, and
`tests/test_benders_example.py` checks that it reaches the optimum of the
monolith. It shows that an algorithm the language does not contain can be
written over the specs it does.

## What a port could not say

Each row is a verdict for the roadmap: macro, primitive, formulation, or
refused.

| Port | What could not be said | Worked around by | Verdict |
|---|---|---|---|
| PyPSA transport model | a bound of `-rating` — PyPSA's `p_min_pu = -1` | shipping `neg_rating` as data | **primitive**: bounds as expressions, [#31](https://github.com/fluxopt/specsolve/issues/31). A second model asking for it |
| Travelling salesman | subtour cuts **generated lazily** inside branch-and-cut, which is how every serious TSP code works | [MTZ](../examples/tsp_mtz.md), O(n²) and static | **refused, and correctly**: a solve loop is an algorithm, not a model |

### Shapes still without a witness

The language can say each of these, and no outside model in the corpus has yet
needed one. A model that does is worth more to the corpus than one that uses a
shape already covered.

| Shape | Where it was looked for |
|---|---|
| one axis grouped several ways, all of them load-bearing | OSeMOSYS UTOPIA declares three maps out of its timeslice, but they feed only storage constraints and the instance builds none |
| a chain whose coarse end carries a constraint | PyPSA's `ac-dc-meshed` has a country per bus and nothing constrains a country; GAMSLIB `alum` composes three such chains and its shipped scenario switches two of them off |
| a group that carries its own constraint and appears nowhere else | GAMSLIB `mexls` is exactly this, and its optimum is published only in a book with no reachable text |
| a partial map whose null membership moves an optimum | `reserves` exercises it, but that model is ours and was built to |
| opposite-sign legs onto one dimension, plus a second hop a constraint reads | an airline fleet-assignment text has it; the data and both optima are not obtainable |
