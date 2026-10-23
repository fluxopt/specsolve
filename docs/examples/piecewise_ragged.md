# piecewise_ragged

Dispatch where each generator's cost curve has as many breakpoints as its own
data.

> **✔ Agrees with hand-written linopy 0.10.0** — objective **426**, matched to `rtol=1e-09`.

## The problem

The hydro unit here has two breakpoints, the coal plant three, the gas turbine
four. Nothing in the model has to know which is longest.

$$p_{t,g} = \sum_{k \in \mathcal{K}_g} \lambda_{t,g,k}\, x_{g,k}, \quad
\mathrm{cost}_{t,g} \ge \sum_{k \in \mathcal{K}_g} \lambda_{t,g,k}\, y_{g,k}, \quad
\sum_{k \in \mathcal{K}_g} \lambda_{t,g,k} = 1$$

The weights exist only where a curve has a breakpoint, so the hydro unit
carries two weights rather than four, and no curve is padded.

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

Least-cost dispatch where each generator's cost curve has as many breakpoints as its data gives it — two for the hydro unit, four for the gas turbine — rather than as many as the longest curve in the system.

#### Sets

| Symbol | Meaning |
|---|---|
| $`\mathcal{T}`$ | index $`t`$ — `snapshot` — dispatch periods |
| $`\mathcal{G}`$ | index $`g`$ — `generator` — dispatchable units |
| $`\mathcal{B}`$ | index $`b`$ — `bp` — breakpoints, as many as the longest curve needs |

#### Parameters

| Symbol | Meaning |
|---|---|
| $`\mathrm{p}^{\mathrm{max}}`$ | `p_max` over $`\mathcal{G}`$ — maximum dispatch |
| $`\mathrm{load}`$ | `load` over $`\mathcal{T}`$ — demand to be met |
| $`\mathrm{bp\_x}`$ | `bp_x` over $`\mathcal{G} \times \mathcal{B}`$, `absent` where the data has no row — breakpoint dispatch levels, one curve per generator and no two the same length |
| $`\mathrm{bp\_y}`$ | `bp_y` over $`\mathcal{G} \times \mathcal{B}`$, `absent` where the data has no row — cost at each breakpoint |

#### Variables

| Symbol | Meaning |
|---|---|
| $`p`$ | `p` over $`\mathcal{T} \times \mathcal{G}`$ — dispatched power |
| $`\mathit{op\_cost}`$ | `op_cost` over $`\mathcal{T} \times \mathcal{G}`$ — operating cost, piecewise-linear in dispatch |

#### Notation

| Symbol | Meaning |
|---|---|
| $`t \boxminus_{v} k`$ | `edge=v` — $`t-k`$, and $`v`$ past the edge |

#### Variable domains

**`p`**

```math
0 \le p_{t,g} \le \mathrm{p}^{\mathrm{max}}_{g} \qquad \forall\, t \in \mathcal{T},\ g \in \mathcal{G}
```

**`op_cost`**

```math
\mathit{op\_cost}_{t,g} \ge 0 \qquad \forall\, t \in \mathcal{T},\ g \in \mathcal{G}
```

#### Objective

```math
\min \sum_{t \in \mathcal{T}} \sum_{g \in \mathcal{G}} \mathit{op\_cost}_{t,g}
```

#### Subject to

**`balance`**

```math
\sum_{g \in \mathcal{G}} p_{t,g} = \mathrm{load}_{t} \qquad \forall\, t \in \mathcal{T}
```

**`cost_curve`**

```math
\mathit{op\_cost}_{t,g} \ge \mathrm{conv}_{b \in \mathcal{B} \,:\, \mathrm{bp\_x}_{g,b} \text{ is defined}}(\mathrm{bp\_x}_{g,b},\ \mathrm{bp\_y}_{g,b})(p_{t,g}) \qquad \forall\, g \in \mathcal{G},\ t \in \mathcal{T}
```

#### Assumptions

**`cost_curve_complete`**

```math
\mathrm{bp\_x}_{g,b} \text{ is defined} \wedge \mathrm{bp\_y}_{g,b} \text{ is defined} \qquad \forall\, g \in \mathcal{G},\ b \in \mathcal{B} \,:\, \mathrm{bp\_x}_{g,b} \text{ is defined}
```

**`cost_curve_increasing`**

```math
\mathrm{bp\_x}_{g,b \boxminus_{0} 1} < \mathrm{bp\_x}_{g,b} \qquad \forall\, g \in \mathcal{G},\ b \in \mathcal{B} \,:\, \mathrm{bp\_x}_{g,b} \text{ is defined} \wedge \mathrm{bp\_x}_{g,b - 1} \text{ is defined}
```

**`cost_curve_curvature`**

```math
\left( \mathrm{bp\_y}_{g,b} - \mathrm{bp\_y}_{g,b \boxminus_{0} 1} \right) \cdot \left( \mathrm{bp\_x}_{g,b \boxplus_{0} 1} - \mathrm{bp\_x}_{g,b} \right) \le \left( \mathrm{bp\_y}_{g,b \boxplus_{0} 1} - \mathrm{bp\_y}_{g,b} \right) \cdot \left( \mathrm{bp\_x}_{g,b} - \mathrm{bp\_x}_{g,b \boxminus_{0} 1} \right) \qquad \forall\, g \in \mathcal{G},\ b \in \mathcal{B} \,:\, \mathrm{bp\_x}_{g,b} \text{ is defined} \wedge \mathrm{bp\_x}_{g,b - 1} \text{ is defined} \wedge \mathrm{bp\_x}_{g,b + 1} \text{ is defined}
```

**`cost_curve_contiguous`**

```math
\lvert \{ b \in \mathcal{B} \,:\, \mathrm{bp\_x}_{g,b} \text{ is defined} \wedge \neg \left( \mathrm{bp\_x}_{g,b - 1} \text{ is defined} \right) \} \rvert = 1 \qquad \forall\, g \in \mathcal{G}
```

</details>
<!-- math:end -->

The tabs start from [the instance's tables](../howto/data.md) — one frame per parameter.

=== "specsolve"

    ```yaml
    description: >-
      Least-cost dispatch where each generator's cost curve has as many breakpoints
      as its data gives it — two for the hydro unit, four for the gas turbine —
      rather than as many as the longest curve in the system.

    dimensions:
      snapshot:
        description: dispatch periods
        dtype: int
        ordered: true
      generator:
        description: dispatchable units
        dtype: str
      bp:
        description: breakpoints, as many as the longest curve needs
        dtype: int
        ordered: true

    parameters:
      p_max:
        description: maximum dispatch
        dims: [generator]
      load:
        description: demand to be met
        dims: [snapshot]
      bp_x:
        description: breakpoint dispatch levels, one curve per generator and no two the same length
        dims: [generator, bp]
        missing: absent
      bp_y:
        description: cost at each breakpoint
        dims: [generator, bp]
        missing: absent

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
          each unit's own curve — `points: bp_x` says a curve runs as far as its own
          breakpoints do, so the hydro unit pays for two weights and the gas turbine
          for four
        over: bp
        points: bp_x
        links:
          - [p, bp_x]
          - [op_cost, bp_y, ">="]
        method: convex

    constraints:
      balance:
        description: every period's demand is met
        dims: [snapshot]
        expression: sum(p, over=generator) == load

    objective:
      sense: minimize
      description: total operating cost, taken off each unit's own curve
      expression: sum(sum(op_cost, over=generator), over=snapshot)
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with sps.solve(to_spec('examples/piecewise_ragged.yaml').expand(), sources) as solution:
        solution.objective  # 426.0
        solution.dual('balance')
    ```

=== "linopy"

    The model-building half of `examples/ports/references/linopy/piecewise_ragged.py`:

    ```python
    def build(tables: dict[str, pd.DataFrame]) -> linopy.Model:
        """The instance's tables as a linopy model, row for row.

        ``tables`` is the same mapping the specsolve call attaches as ``sources``.
        """
        p_max: pd.Series = tables['p_max'].set_index('generator')['value']
        load: pd.Series = tables['load'].set_index('snapshot')['value']
        reach = tables['bp_x'].groupby('generator')['value'].max().reindex(p_max.index)

        m = linopy.Model()
        p = m.add_variables(lower=0, upper=reach, coords=[load.index, p_max.index], name='p')
        op_cost = m.add_variables(lower=0, coords=[load.index, p_max.index], name='op_cost')
        for g, lines in segments(tables).items():
            for k, (slope, intercept) in enumerate(lines):
                m.add_constraints(
                    op_cost.sel(generator=g) - slope * p.sel(generator=g) >= intercept,
                    name=f'chord_{g}_{k}',
                )
        m.add_constraints(p.sum('generator') == load, name='balance')
        m.add_objective(op_cost.sum())
        return m
    ```

---

[`examples/piecewise_ragged.yaml`](https://github.com/fluxopt/specsolve/blob/main/examples/piecewise_ragged.yaml) · back to [all models](index.md)
