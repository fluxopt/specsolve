# dispatch

Least-cost generation against a load profile — the smallest model that is still a model.

> **✔ Agrees with hand-written linopy 0.10.0** — objective **10500**, matched to `rtol=1e-09`.

## The problem

Pick an output $p_{s,g}$ for every generator in every snapshot, so that the
fleet meets the load exactly and costs as little as possible:

$$\min \sum_{s,g} c_g \thinspace p_{s,g}
\quad\text{s.t.}\quad \sum_g p_{s,g} = \ell_s ,\quad 0 \le p_{s,g} \le \bar p_g
\quad\text{where}\quad \bar p_g > 0$$

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

Least-cost dispatch of a generator fleet against an hourly load.

#### Sets

| Symbol | Meaning |
|---|---|
| $`\mathcal{S}`$ | index $`s`$ — `snapshot` — dispatch periods |
| $`\mathcal{G}`$ | index $`g`$ — `generator` — generating units |

#### Parameters

| Symbol | Meaning |
|---|---|
| $`\bar p`$ | `p_max` over $`\mathcal{G}`$ — installed capacity |
| $`\ell`$ | `load` over $`\mathcal{S}`$ — demand to be met |
| $`c`$ | `cost` over $`\mathcal{G}`$ — marginal cost |

#### Variables

| Symbol | Meaning |
|---|---|
| $`p`$ | `p` over $`\mathcal{S} \times \mathcal{G}`$ — output of a generator in a snapshot |

#### Variable domains

**`p`**

```math
0 \le p_{s,g} \le \bar p_{g} \qquad \forall\, s \in \mathcal{S},\ g \in \mathcal{G} \,:\, \bar p_{g} > 0
```

#### Objective

```math
\min \sum_{s \in \mathcal{S},\ g \in \mathcal{G}} p_{s,g} \cdot c_{g}
```

#### Subject to

**`power_balance`**

```math
\sum_{g \in \mathcal{G}} p_{s,g} = \ell_{s} \qquad \forall\, s \in \mathcal{S}
```

</details>
<!-- math:end -->

The tabs start from [the instance's tables](../howto/data.md) — one frame per parameter.

=== "specsolve"

    ```yaml
    description: Least-cost dispatch of a generator fleet against an hourly load.

    dimensions:
      snapshot:
        description: dispatch periods
        dtype: int
        ordered: true
      generator:
        description: generating units

    parameters:
      p_max:
        description: installed capacity
        dims: [generator]
      load:
        description: demand to be met
        dims: [snapshot]
      cost:
        description: marginal cost
        dims: [generator]

    variables:
      p:
        description: output of a generator in a snapshot
        dims: [snapshot, generator]
        where: "p_max > 0"
        bounds:
          lower: 0
          upper: p_max

    constraints:
      power_balance:
        dims: [snapshot]
        expression: sum(p, over=generator) == load

    objective:
      sense: minimize
      description: total cost of generation over the horizon
      expression: sum(p * cost)
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with sps.solve('examples/dispatch.yaml', sources) as solution:
        solution.objective  # 10500.0
        solution.dual('power_balance')
    ```

=== "linopy"

    The model-building half of `examples/ports/references/linopy/dispatch.py`:

    ```python
    def build(tables: dict[str, pd.DataFrame]) -> linopy.Model:
        """The instance's tables as a linopy model, row for row.

        ``tables`` is the same mapping the specsolve call attaches as ``sources``.
        """
        p_max: pd.Series = tables['p_max'].set_index('generator')['value']
        cost: pd.Series = tables['cost'].set_index('generator')['value']
        load: pd.Series = tables['load'].set_index('snapshot')['value']

        m = linopy.Model()
        p = m.add_variables(lower=0, upper=p_max, coords=[load.index, p_max.index], name='p')
        m.add_constraints(p.sum('generator') == load, name='power_balance')
        m.add_objective((p * cost).sum())
        return m
    ```

---

[`examples/dispatch.yaml`](https://github.com/fluxopt/specsolve/blob/main/examples/dispatch.yaml) · back to [all models](index.md)
