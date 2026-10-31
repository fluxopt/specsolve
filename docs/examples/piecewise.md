# piecewise

Dispatch where each generator's cost follows its own convex curve.

> **✔ Agrees with hand-written linopy 0.10.0** — objective **3850**, matched to `rtol=1e-09`.

## The problem

Each generator has its own breakpoints, and its cost is read off its own
curve:

$$p_g = \sum_k \lambda_{g,k}\, x_{g,k}, \quad
\mathrm{cost}_g = \sum_k \lambda_{g,k}\, y_{g,k}, \quad
\sum_k \lambda_{g,k} = 1, \quad \lambda \ge 0$$

## Choosing a method

`method:` decides what the solver receives. Two more files in `examples/` hold
this model with only that line changed.

| `method:` | the solver receives | which solvers |
|---|---|---|
| `convex`, on this page | a weight per breakpoint, as an LP | all |
| `lp`, in [`piecewise_lp.yaml`](https://github.com/fluxopt/specsolve/blob/main/examples/piecewise_lp.yaml) | one row per segment, and no extra variable, as an LP | all |
| `sos2`, in [`sos.yaml`](https://github.com/fluxopt/specsolve/blob/main/examples/sos.yaml) | a weight per breakpoint, and a set it branches on | gurobi and xpress, and the `.lp` and `.mps` files; HiGHS refuses the model, and the error names the way past it |
| `adjacency`, the default | a weight per breakpoint and a binary per segment, as a MILP | all |

`convex` and `lp` are exact here because the curve is convex and the cost is
minimised. Both check the curve when the data attaches. On this instance `convex` and `lp`
reach the same optimum, 3850, and the same duals on `balance`.

The two LP methods trade columns for rows. On 20 generators × 48 snapshots × 6
breakpoints:

| | columns | rows | nonzeros |
|---|---:|---:|---:|
| `method: convex` | 7680 | 2928 | 18240 |
| `method: lp` | **1920** | 6768 | **12480** |

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

Least-cost dispatch where each generator's cost curve is piecewise-linear in its output, expanded into a lambda formulation.

#### Sets

| Symbol | Meaning |
|---|---|
| $`\mathcal{T}`$ | index $`t`$ — `snapshot` — dispatch periods |
| $`\mathcal{G}`$ | index $`g`$ — `generator` — dispatchable units |
| $`\mathcal{K}`$ | index $`k`$ — `bp` — breakpoints of the cost curve |

#### Parameters

| Symbol | Meaning |
|---|---|
| $`\mathrm{p}^{\mathrm{max}}`$ | `p_max` over $`\mathcal{G}`$ — maximum dispatch |
| $`\mathrm{load}`$ | `load` over $`\mathcal{T}`$ — demand to be met |
| $`x`$ | `bp_x` over $`\mathcal{G} \times \mathcal{K}`$ — breakpoint dispatch levels, one curve per generator |
| $`y`$ | `bp_y` over $`\mathcal{G} \times \mathcal{K}`$ — cost at each breakpoint, one curve per generator |

#### Variables

| Symbol | Meaning |
|---|---|
| $`p`$ | `p` over $`\mathcal{T} \times \mathcal{G}`$ — dispatched power |
| $`\mathrm{cost}`$ | `op_cost` over $`\mathcal{T} \times \mathcal{G}`$ — operating cost, piecewise-linear in dispatch |

#### Notation

| Symbol | Meaning |
|---|---|
| $`t \boxminus_{v} k`$ | `edge=v` — $`t-k`$, and $`v`$ past the edge |
| $`\mathrm{pos}(t)`$ | `position()` — the position of $`t`$, counted from $`0`$; $`t`$ is the label |
| $`\lvert \mathcal{T} \rvert`$ | the size of $`\mathcal{T}`$; $`\lvert \mathcal{T} \rvert - 1`$ is the last position |

#### Variable domains

**`p`**

```math
0 \le p_{t,g} \le \mathrm{p}^{\mathrm{max}}_{g} \qquad \forall\, t \in \mathcal{T},\ g \in \mathcal{G}
```

**`op_cost`**

```math
\mathrm{cost}_{t,g} \ge 0 \qquad \forall\, t \in \mathcal{T},\ g \in \mathcal{G}
```

#### Objective

```math
\min \sum_{t \in \mathcal{T},\ g \in \mathcal{G}} \mathrm{cost}_{t,g}
```

#### Subject to

**`balance`**

```math
\sum_{g \in \mathcal{G}} p_{t,g} = \mathrm{load}_{t} \qquad \forall\, t \in \mathcal{T}
```

**`cost_curve`**

```math
\left( p_{t,g},\ \mathrm{cost}_{t,g} \right) \in \mathrm{conv}_{k \in \mathcal{K}}(x_{g,k},\ y_{g,k}) \qquad \forall\, g \in \mathcal{G},\ t \in \mathcal{T}
```

#### Assumptions

**`cost_curve_complete`**

```math
x_{g,k} \text{ is defined} \wedge y_{g,k} \text{ is defined} \qquad \forall\, g \in \mathcal{G},\ k \in \mathcal{K}
```

**`cost_curve_increasing`**

```math
x_{g,k \boxminus_{0} 1} < x_{g,k} \qquad \forall\, g \in \mathcal{G},\ k \in \mathcal{K} \,:\, \mathrm{pos}(k) > 0
```

**`cost_curve_curvature`**

```math
\lvert \{ k \in \mathcal{K} \,:\, \left( y_{g,k} - y_{g,k \boxminus_{0} 1} \right) \cdot \left( x_{g,k \boxplus_{0} 1} - x_{g,k} \right) > \left( y_{g,k \boxplus_{0} 1} - y_{g,k} \right) \cdot \left( x_{g,k} - x_{g,k \boxminus_{0} 1} \right) \wedge \mathrm{pos}(k) > 0 \wedge \mathrm{pos}(k) \neq \lvert \mathcal{K} \rvert - 1 \} \rvert = 0 \vee \lvert \{ k \in \mathcal{K} \,:\, \left( y_{g,k} - y_{g,k \boxminus_{0} 1} \right) \cdot \left( x_{g,k \boxplus_{0} 1} - x_{g,k} \right) < \left( y_{g,k \boxplus_{0} 1} - y_{g,k} \right) \cdot \left( x_{g,k} - x_{g,k \boxminus_{0} 1} \right) \wedge \mathrm{pos}(k) > 0 \wedge \mathrm{pos}(k) \neq \lvert \mathcal{K} \rvert - 1 \} \rvert = 0 \qquad \forall\, g \in \mathcal{G}
```

</details>
<!-- math:end -->

The tabs start from [the instance's tables](../howto/data.md) — one frame per parameter.

=== "specsolve"

    ```yaml
    description: >-
      Least-cost dispatch where each generator's cost curve is piecewise-linear in
      its output, expanded into a lambda formulation.

    dimensions:
      snapshot:
        description: dispatch periods
        dtype: int
        ordered: true
      generator:
        description: dispatchable units
        dtype: str
      bp:
        description: breakpoints of the cost curve
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
          cost read off the generator's curve — convex, so the weights need no
          binaries to keep them on one segment
        over: bp
        links:
          - [p, bp_x]
          - [op_cost, bp_y]
        method: convex

    constraints:
      balance:
        dims: [snapshot]
        expression: sum(p, over=generator) == load

    objective:
      sense: minimize
      description: total operating cost, taken off the curves rather than from a marginal rate
      expression: sum(op_cost)
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with sps.solve(to_spec('examples/piecewise.yaml').expand(), sources) as solution:
        solution.objective  # 3850.0
        solution.dual('balance')
    ```

=== "linopy"

    The model-building half of `examples/ports/references/linopy/piecewise.py`:

    ```python
    def build(tables: dict[str, pd.DataFrame]) -> linopy.Model:
        """The instance's tables as a linopy model, row for row.

        ``tables`` is the same mapping the specsolve call attaches as ``sources``.
        """
        p_max: pd.Series = tables['p_max'].set_index('generator')['value']
        load: pd.Series = tables['load'].set_index('snapshot')['value']
        curve_x: pd.DataFrame = tables['bp_x'].pivot(index='generator', columns='bp', values='value').reindex(p_max.index)
        curve_y: pd.DataFrame = tables['bp_y'].pivot(index='generator', columns='bp', values='value').reindex(p_max.index)
        bp_x = linopy.breakpoints(curve_x, dim='generator')
        bp_y = linopy.breakpoints(curve_y, dim='generator')

        m = linopy.Model()
        p = m.add_variables(lower=0, upper=p_max, coords=[load.index, p_max.index], name='p')
        op_cost = m.add_variables(lower=0, coords=[load.index, p_max.index], name='op_cost')
        m.add_piecewise_formulation((p, bp_x), (op_cost, bp_y, '>='))
        m.add_constraints(p.sum('generator') == load, name='balance')
        m.add_objective(op_cost.sum())
        return m
    ```

---

[`examples/piecewise.yaml`](https://github.com/fluxopt/specsolve/blob/main/examples/piecewise.yaml) · back to [all models](index.md)
