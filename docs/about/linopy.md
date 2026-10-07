# Relationship to linopy

Everything about [linopy](https://github.com/PyPSA/linopy) in one place, for a
reader who arrives from linopy or PyPSA. There are two separate
relationships:

| | What | Where it matters |
|---|---|---|
| **Not a dependency** | nothing in the package imports it | packaging |
| **The oracle** | how we know the answers are right | testing |

## 1. It is not a dependency

`sps.solve`, `sps.build`, `sps.write` and `sps.check` go YAML → polars → HiGHS
or file, and import nothing from linopy, xarray or pandas. The bare-install job
runs the whole suite with none of the three present. No install of specsolve
brings linopy: it is a test dependency, in the `dev` group.

The `to_pandas` / `to_dataarray` bridges out of a
[result](../reference/glossary.md#the-chain) need pandas and xarray, which the
caller installs.

**Nothing a bare install can reach names linopy, including a traceback.** The
public exception tree is rooted at `SpecsolveError`, with no alias
([#389](https://github.com/fluxopt/specsolve/issues/389)).

## 2. It is the oracle

Correctness here is **the same YAML, built both ways, produces the same
model**. The differential suite builds a model through the relational engine
and through linopy, and compares the two. The linopy build is
`tests/linopy_lane`, a test module that builds a file as a `linopy.Model` from
the same `sources` `sps.build` takes.

The comparison means something only because both paths consume the *same
resolved AST*, the narrow waist in
[the architecture notes](architecture.md#one-contract-many-consumers). If each
path resolved names on its own, the suite would compare two dialects rather than
check one language.

The oracle has one blind spot: a **shared misreading** passes the differential
suite green. Only a published optimum from outside catches it, and
[docs/examples/index.md](../examples/index.md) is where those live.

**Where a concept is already linopy's, specsolve copies its name.** Solve statuses;
`status` and `termination_condition` as two axes with `is_ok` as the rollup; the
shape of a result. A second vocabulary for one fact taxes a reader arriving from
linopy or PyPSA. But **copy it, do not import it.** The engine may not import
linopy, so the tables live here. A test imports linopy to assert the copy still
matches. A copy nobody checks is a copy that rots.

### What a construct becomes in the oracle

What the oracle calls for each thing a file can say. Each row lives in
`tests/linopy_lane/builder.py`, one section per group below.

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

Absence has no single row. It is positional: a missing parameter row is zero in a coefficient, an error in `bounds:`, and false in a `where` operand.
`tests/linopy_lane/absence.py` holds all four spellings, and the builder calls them
qualified, as `absence.coefficient(...)`, so a reader meets the name at the
call.

### The same language, and the same data

**The oracle accepts exactly the same language**, which is what makes it an
oracle. The equality is structural: both builds run the same `inputs.lowered`
gate. A construct one refuses, the other refuses in the same sentence.

**Accepting is not building, and three constructs part the two builds, two on
the oracle's side and one on the package's.** None is a language limit: every
such file passes `check`. The refusal names the wall *and* the rewrite, and
that is what parts it from a language error.

**The first is the oracle's wall: an objective carrying a constant.**
`linopy.Objective` rejects any expression whose `const` is nonzero:
*"Constant values in objective function not supported."* There is no slot to
put one in, which is why PyPSA carries `n.objective_constant` out of band. So
`examples/ports/osemosys_utopia.yaml`, whose objective carries a fixed cost on
capacity that already stood in 1990, builds in specsolve and not in the
oracle. **Dropping the constant is the one repair that must not happen.** A
quietly shortened objective would recalibrate every differential test on such a
model to the wrong number. So the oracle checks for a constant before linopy is
asked, and refuses the model. `tests/test_corpus_parity.py` carries
the strict xfail ([#894](https://github.com/fluxopt/specsolve/issues/894)).

**The second is the oracle's too: a relation that is not the single-valued
map.** The oracle holds each value column as one dense array over the
dimensions the relation's key names, so a walk is an `assign_coords` and a
`groupby`, or a vectorised `sel`. A bare relation, and a partition grouped by a
map keyed on more than the dimension it walks or by more than one column, have
no such array, and `tests/linopy_lane/loader.py` refuses each. specsolve builds
every shape the language admits.

**The third is specsolve's wall, and it is the mirror: an operator acting along
a dimension that a constant part does not carry**, beside a term that does.
Take `sum(x * k + d, over=t)` where `d` is a scalar. specsolve compiles a
constant part as its own [table](../reference/glossary.md#the-data), and a
piece with no rows for `t` has no slots for the operator to act on. The
oracle has no such split: the operand is one masked expression, so the constant
is dropped wherever the term is, and the oracle builds the file as written. All four operators that act along
a dimension (`sum(over=)`, `sum(by=)`, `shift`, `sum_back`) reach the one wall
and share one refusal. It names the rewrite: declare the parameter over the
dimension and supply it there
([#1137](https://github.com/fluxopt/specsolve/issues/1137)).

**The oracle takes the same data too**
([#60](https://github.com/fluxopt/specsolve/issues/60)). It reads every shape
[the data contract](../reference/data.md) accepts and follows every index rule
in [where coordinates come from](../reference/data.md#where-coordinates-come-from).
A malformed source gets the same refusal from both builds, in the same sentence.

## Parts of linopy not taken

specsolve does not take array operations (`merge`, `reindex`, `stack`), the Python
modeling API, or the solver layer. The first is data prep
([the limits](https://mathspec.readthedocs.io/en/latest/reference/language/errors/#what-the-language-will-not-express)).
The second is [hard rule 5](architecture.md#hard-rules): the spec is the file
you review and diff. The third is
[#106](https://github.com/fluxopt/specsolve/issues/106), where specsolve adopts
linopy's *design* for declared solver capabilities without adopting its code.

The modeling API is what a reader arriving from linopy misses first. Two
pages replace it. The tutorial [Change a model](../change.md) covers the
loops: `update` for new numbers, a longer table for more rows, a patched `dict`
for new math. The how-to [Fixing, relaxing and removing](../howto/fix-relax-remove.md)
aims the same loops at `fix`, `relax` and `remove_constraints`. A built row is
read with [`row`](../reference/api.md#specsolve.types.Model.row), in linopy's
own form, and an IIS with [`infeasible_subsystem`](../reference/api.md#specsolve.types.Model.infeasible_subsystem),
by declaration and coordinate.

Where linopy is ahead, and why none of it is a ceiling question, is
[the roadmap](roadmap.md#honest-snapshot). What is *owed* to linopy rather than
merely true of it is [prior art and credit](prior-art.md). The same page credits
Calliope, whose math language this surface is derived from.
