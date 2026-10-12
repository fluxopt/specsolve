# piecewise_lp

The [piecewise](piecewise.md) model, with each cost curve stated as the lines
its segments lie on.

## The problem

A convex curve is the upper envelope of its own segment lines:

$$f(x) = \max_k \left( f(x_k) + \frac{f(x_{k+1}) - f(x_k)}{x_{k+1} - x_k} \cdot (x - x_k) \right)$$

So `cost` $\ge$ every one of those lines says `cost` $\ge f(p)$, and a
minimising objective pushes it down onto the curve. That takes one row for
each segment, and no variable beyond the model's own two.

The saving is a **trade**. Against `method: convex` on 20 generators × 48
snapshots × 6 breakpoints:

| | columns | rows | nonzeros |
|---|---:|---:|---:|
| `method: convex` | 7680 | 2928 | 18240 |
| `method: lp` | **1920** | 6768 | **12480** |

`method: lp` drops three quarters of the columns and a third of the nonzeros,
and carries more than twice the rows.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

The same least-cost dispatch as `piecewise.yaml`, with each generator's cost curve stated as the lines its segments lie on rather than interpolated between its breakpoints. The curve is convex and the objective pushes the cost down, so a cost above every segment line settles on the curve — which needs no interpolation weights, and so declares no auxiliary variable at all.

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
| $`\mathit{op\_cost}`$ | `op_cost` over $`\mathcal{T} \times \mathcal{G}`$ — operating cost, held above every segment of the generator's curve |

Upright is what the data supplies — a parameter such as $`\mathrm{p}^{\mathrm{max}}`$, a coordinate map, a label — and italic is what the solver chooses, such as $`p`$. An index is italic too, being what a quantifier chooses, and a set is script.

$`t \boxminus_{v} k`$ denotes translation with $`v`$ standing where index $`t-k`$ leaves the dimension (`shift(edge=v)`), so the row at that boundary is built and carries $`v`$ rather than being dropped.

$`\mathrm{pos}(t)`$ denotes where index $`t`$ sits along its dimension's own order — the order `shift` steps along, not the order labels sort in — counted from $`0`$. The index itself stays the coordinate, so $`t`$ compares against labels and $`\mathrm{pos}(t)`$ against positions.

$`\lvert \mathcal{T} \rvert`$ denotes the size of the set being counted along, and a position counted from the end prints against it — $`\lvert \mathcal{T} \rvert - 1`$ is the last position, one less than the size because the first is $`0`$.

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
\mathit{op\_cost}_{t,g} \ge \mathrm{pwl}_{b \in \mathcal{B}}(\mathrm{bp\_x}_{g,b},\ \mathrm{bp\_y}_{g,b})(p_{t,g}) \qquad \forall\, t \in \mathcal{T},\ g \in \mathcal{G}
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

**`cost_curve_increasing`**

```math
\mathrm{bp\_x}_{g,b \boxminus_{0} 1} < \mathrm{bp\_x}_{g,b} \qquad \forall\, g \in \mathcal{G},\ b \in \mathcal{B} \,:\, \mathrm{pos}(b) > 0
```

**`cost_curve_curvature`**

```math
\left( \mathrm{bp\_y}_{g,b} - \mathrm{bp\_y}_{g,b \boxminus_{0} 1} \right) \cdot \left( \mathrm{bp\_x}_{g,b \boxplus_{0} 1} - \mathrm{bp\_x}_{g,b} \right) \le \left( \mathrm{bp\_y}_{g,b \boxplus_{0} 1} - \mathrm{bp\_y}_{g,b} \right) \cdot \left( \mathrm{bp\_x}_{g,b} - \mathrm{bp\_x}_{g,b \boxminus_{0} 1} \right) \qquad \forall\, g \in \mathcal{G},\ b \in \mathcal{B} \,:\, \mathrm{pos}(b) > 0 \wedge \mathrm{pos}(b) \neq \lvert \mathcal{B} \rvert - 1
```

**`cost_curve_breakpoints`**

```math
\lvert \{ b \in \mathcal{B} \,:\, \mathrm{bp\_x}_{g,b} \text{ is defined} \} \rvert \ge 2 \qquad \forall\, g \in \mathcal{G}
```

</details>
<!-- math:end -->

```yaml
description: >-
  The same least-cost dispatch as `piecewise.yaml`, with each generator's cost
  curve stated as the lines its segments lie on rather than interpolated
  between its breakpoints. The curve is convex and the objective pushes the
  cost down, so a cost above every segment line settles on the curve — which
  needs no interpolation weights, and so declares no auxiliary variable at all.

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
    description: operating cost, held above every segment of the generator's curve
    dims: [snapshot, generator]
    bounds:
      lower: 0

piecewise:
  cost_curve:
    description: >-
      cost bounded below by the curve — the `>=` is what says which side of the
      lines the cost sits on, and the curvature has to match it: lines that
      envelope a convex curve would cut a concave one, and the solve comes back
      optimal either way
    over: bp
    links:
      - [p, bp_x]
      - [op_cost, bp_y, ">="]
    method: lp

constraints:
  balance:
    dims: [snapshot]
    expression: sum(p, over=generator) == load

objective:
  sense: minimize
  description: total operating cost, taken off the curves rather than from a marginal rate
  expression: sum(op_cost)
```

---

[`examples/piecewise_lp.yaml`](https://github.com/fluxopt/specsolve/blob/main/examples/piecewise_lp.yaml) · back to [all models](index.md)
