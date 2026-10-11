# sos

A piecewise-linear cost curve stated as a **special-ordered set** —
[piecewise](piecewise.md) with one line changed, handed to the solver as a set
it branches on itself.

## The problem

A convex-combination curve needs one thing said about its weights: at most two
may be nonzero, and they must be **neighbours**. Otherwise the weights mix
distant breakpoints and the model prices the chord under the curve instead of
the curve:

$$p = \sum_k \lambda_k x_k, \quad
\mathrm{cost} = \sum_k \lambda_k y_k, \quad
\sum_k \lambda_k = 1, \quad
\lambda \in \mathrm{SOS2}$$

`method:` names two ways to say the last line. `adjacency`, the default,
*builds* it: a binary per segment, an adjacency row per breakpoint, and one
more row picking a segment. `sos2` *declares* it: the expansion emits an
[`sos:` block](https://mathspec.readthedocs.io/en/latest/reference/language/piecewise/#sos) over
the same weights and leaves the formulation to the sink. A solver that knows
what SOS2 means branches on the set directly rather than searching binaries
written for it. The raw `sos:` block stays in the language for a set that is
not a curve, such as picking at most one of several build sizes, where there
is no `piecewise:` declaration to emit it.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

A piecewise-linear cost curve stated as a special-ordered set, so the solver is handed the adjacency restriction rather than binaries that encode it.

#### Sets

| Symbol | Meaning |
|---|---|
| $`\mathcal{T}`$ | index $`t`$ — `snapshot` — dispatch periods |
| $`\mathcal{G}`$ | index $`g`$ — `generator` — dispatchable units |
| $`\mathcal{B}`$ | index $`b`$ — `bp` — breakpoints of the cost curve |

#### Parameters

| Symbol | Meaning |
|---|---|
| $`\mathrm{p}^{\mathrm{max}}`$ | `p_max` over $`\mathcal{G}`$ — maximum dispatch |
| $`\mathrm{load}`$ | `load` over $`\mathcal{T}`$ — demand to be met |
| $`\mathrm{bp\_x}`$ | `bp_x` over $`\mathcal{G} \times \mathcal{B}`$ — breakpoint dispatch levels, one curve per generator |
| $`\mathrm{bp\_y}`$ | `bp_y` over $`\mathcal{G} \times \mathcal{B}`$ — cost at each breakpoint, one curve per generator |

#### Variables

| Symbol | Meaning |
|---|---|
| $`p`$ | `p` over $`\mathcal{T} \times \mathcal{G}`$ — dispatched power |
| $`\mathit{op\_cost}`$ | `op_cost` over $`\mathcal{T} \times \mathcal{G}`$ — operating cost, piecewise-linear in dispatch |

Upright is what the data supplies — a parameter such as $`\mathrm{p}^{\mathrm{max}}`$, a coordinate map, a label — and italic is what the solver chooses, such as $`p`$. An index is italic too, being what a quantifier chooses, and a set is script.

#### Objective

```math
\min \sum_{t \in \mathcal{T},\ g \in \mathcal{G}} \mathit{op\_cost}_{t,g}
```

#### Subject to

**`balance`**

```math
\sum_{g \in \mathcal{G}} p_{t,g} = \mathrm{load}_{t} \qquad \forall\, t \in \mathcal{T}
```

**`cost_curve`**

```math
\left( p_{t,g},\ \mathit{op\_cost}_{t,g} \right) \in \mathrm{pwl}_{b \in \mathcal{B}}(\mathrm{bp\_x}_{g,b},\ \mathrm{bp\_y}_{g,b}) \qquad \forall\, t \in \mathcal{T},\ g \in \mathcal{G}
```

#### Variable domains

**`p`**

```math
0 \le p_{t,g} \le \mathrm{p}^{\mathrm{max}}_{g} \qquad \forall\, t \in \mathcal{T},\ g \in \mathcal{G}
```

**`op_cost`**

```math
\mathit{op\_cost}_{t,g} \ge 0 \qquad \forall\, t \in \mathcal{T},\ g \in \mathcal{G}
```

#### Assumptions

**`cost_curve_complete`**

```math
\mathrm{bp\_x}_{g,b} \text{ is defined} \wedge \mathrm{bp\_y}_{g,b} \text{ is defined} \qquad \forall\, g \in \mathcal{G},\ b \in \mathcal{B}
```

</details>
<!-- math:end -->

```yaml
description: >-
  A piecewise-linear cost curve stated as a special-ordered set, so the solver
  is handed the adjacency restriction rather than binaries that encode it.

dimensions:
  snapshot:
    description: dispatch periods
    dtype: int
  generator:
    description: dispatchable units
    dtype: str
  bp:
    description: breakpoints of the cost curve
    dtype: int

parameters:
  p_max:
    description: maximum dispatch
    dims: [generator]
  load:
    description: demand to be met
    dims: [snapshot]
  bp_x:
    description: breakpoint dispatch levels, one curve per generator
    dims: [generator, bp]
  bp_y:
    description: cost at each breakpoint, one curve per generator
    dims: [generator, bp]

variables:
  p:
    description: dispatched power
    dims: [snapshot, generator]
    bounds:
      lower: 0
      upper: p_max
  op_cost:
    description: operating cost, piecewise-linear in dispatch
    dims: [snapshot, generator]
    bounds:
      lower: 0

piecewise:
  cost_curve:
    description: >-
      cost read off the generator's curve, with at most two adjacent weights
      non-zero — the restriction the default method builds out of binaries,
      declared as a set instead
    over: bp
    links:
      - [p, bp_x]
      - [op_cost, bp_y]
    method: sos2

constraints:
  balance:
    dims: [snapshot]
    expression: sum(p, over=generator) == load

objective:
  sense: minimize
  description: total operating cost, taken off the curves rather than from a marginal rate
  expression: sum(op_cost)
```

## What it exercises

`method: sos2` expands into the same weights, convexity row and link rows as
the default, plus a set instead of the segment binaries. That set adds neither
a column nor a row: it names columns the expansion already made and says which
of them may be nonzero together. It leaves the engine as a **fifth stream**
beside `cols`, `obj`, `rows` and the matrix.

Not every sink can take that stream:

| | what it does with this model |
|---|---|
| `gurobi`, `xpress` | `addSOS` — branches on the set, no binaries in the model at all |
| `.lp`, `.mps` | an `sos` section, read by any solver whose parser has one |
| `highs` | **no SOS concept** — the model is refused before the solver loads, and the error names the way past it |

Nothing is rewritten at the hand-off. To solve this file on HiGHS, write the
sets out first: `mathspec.to_spec(...).expand()` states each set as binaries
and linking rows, which every sink takes. For a set a `piecewise:` block
emitted, `method: adjacency` gives the same result in the file itself. Either
way the model is mixed-integer, so an otherwise continuous model gives up its
duals.

Compare [piecewise](piecewise.md), the same file except for `method: convex`.
Both expand before the plan exists, and nothing called *piecewise* survives
into it. What differs is what the expansion leaves behind: a pure LP there,
and here a set that stays a set up to the sink that takes it.

---

[`examples/sos.yaml`](https://github.com/fluxopt/specsolve/blob/main/examples/sos.yaml) · back to [all models](index.md)
