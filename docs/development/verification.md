# Verification

How we know an answer is right: four checks, from the cheapest to the most
convincing. Each one catches what the one before cannot.

| Check | It catches | Where |
|---|---|---|
| [The oracle](#the-oracle) | a build that differs from linopy's on the same file | `tests/linopy_lane/`, the differential tests |
| [Published optima](#published-optima) | a misreading the oracle shares | `examples/ports/`, [the gallery](../examples/index.md#does-it-get-the-right-answer) |
| [The PyPSA ladder](#the-pypsa-ladder) | a model that solves but is not PyPSA's | `differential/pypsa/`, [the ladder](../examples/pypsa_ladder.md) |
| [Decomposition](#decomposition) | an algorithm outside the language that cannot be written over it | [`examples/benders/`](https://github.com/fluxopt/specsolve/blob/main/examples/benders/README.md) |

## The oracle

The test suite builds every file twice: through the relational engine, and as
a `linopy.Model` in `tests/linopy_lane/`. The differential tests compare the
two. Both builds pass the same `inputs.lowered` gate and read data through
`sources.tidy_sources`, so both refuse the same constructs and the same
malformed sources ([#60](https://github.com/fluxopt/specsolve/issues/60)).

**Where a concept is already linopy's, specsolve copies its name**: solve
statuses, `is_ok`, the shape of a result. `tests/test_solve_status.py` checks
that the copy still matches.

### What a construct becomes in the oracle

Each row lives in `tests/linopy_lane/builder.py`.

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
false in a `where` operand (`tests/linopy_lane/absence.py`).

### The same language, but not every build

Three files pass `check` and build on one side only.

- **An objective with a constant** builds only in specsolve, because
  `linopy.Objective` has no slot for one (`osemosys_utopia`,
  [#894](https://github.com/fluxopt/specsolve/issues/894)).
- **A relation that is not a single-valued map** builds only in specsolve,
  because the oracle holds each relation as one dense array.
- **An operator along a dimension a constant part does not carry**, such as
  `sum(x * k + d, over=t)` with a scalar `d`, builds only in the oracle. The
  refusal says to declare the parameter over the dimension
  ([#1137](https://github.com/fluxopt/specsolve/issues/1137)).

## Published optima

A published optimum catches a misreading both builds share. Each model in
`examples/ports/` carries one, with its source, in `references.json`, and
`tests/test_ports.py` solves to it. The table is
[Does it get the right answer?](../examples/index.md#does-it-get-the-right-answer)

## The PyPSA ladder

[The PyPSA ladder](../examples/pypsa_ladder.md) grows a one-file PyPSA model a
feature at a time. Each rung solves the same network in PyPSA and in
specsolve, and compares the objective and the duals. `tools/ladder.py` writes
the pages from `differential/pypsa/`.

## Decomposition

[`examples/benders/`](https://github.com/fluxopt/specsolve/blob/main/examples/benders/README.md)
writes Benders decomposition as a loop over three specs.
`tests/test_benders_example.py` checks that it reaches the optimum of the
monolith.

## What a port could not say

Each row is a verdict: macro, primitive, formulation, or refused.

| Port | What could not be said | Worked around by | Verdict |
|---|---|---|---|
| PyPSA transport model | a bound of `-rating` — PyPSA's `p_min_pu = -1` | shipping `neg_rating` as data | **primitive**: bounds as expressions, [#31](https://github.com/fluxopt/specsolve/issues/31). A second model asking for it |
| Travelling salesman | subtour cuts **generated lazily** inside branch-and-cut | [MTZ](../examples/tsp_mtz.md), O(n²) and static | **refused**: a solve loop is an algorithm, not a model |
