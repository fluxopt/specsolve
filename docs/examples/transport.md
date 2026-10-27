# transport

A network: generators sit on buses, lines connect buses, and power balances at every bus.

> **✔ Agrees with hand-written linopy 0.10.0** — objective **4400**, matched to `rtol=1e-09`.

## The problem

$$\sum_{g \thinspace:\thinspace \mathrm{bus}(g) = b} p_{s,g} \quad+\quad \sum_{\ell \thinspace:\thinspace \mathrm{to}(\ell) = b} f_{s,\ell} \quad-\quad \sum_{\ell \thinspace:\thinspace \mathrm{from}(\ell) = b} f_{s,\ell} \quad=\quad d_{s,b}$$

Each sum runs over the generators or lines a relation sends to bus $b$.
$\mathrm{bus}$, $\mathrm{to}$ and $\mathrm{from}$ are relations the dimensions
declare, not sets in their own right. Load is $d$ here, because $\ell$ is
already the line index.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

Least-cost dispatch over a network, where a generator sits on a bus, a line joins two of them, and what is generated has to reach the load over the lines.

#### Sets

| Symbol | Meaning |
|---|---|
| $`\mathcal{S}`$ | index $`s`$ — `snapshot` — dispatch periods |
| $`\mathcal{G}`$ | index $`g`$ — `generator` with $`\mathrm{gen\_bus}: \mathcal{G} \to \mathcal{B}`$ — generating units, each sitting on one bus |
| $`\mathcal{B}`$ | index $`b`$ — `bus` with $`\mathrm{gen\_bus}: \mathcal{G} \to \mathcal{B},\ \mathrm{line\_from}: \mathcal{L} \to \mathcal{B},\ \mathrm{line\_to}: \mathcal{L} \to \mathcal{B}`$ — network nodes |
| $`\mathcal{L}`$ | index $`\ell`$ — `line` with $`\mathrm{line\_from}: \mathcal{L} \to \mathcal{B},\ \mathrm{line\_to}: \mathcal{L} \to \mathcal{B}`$ — transmission lines, each joining two buses |

#### Parameters

| Symbol | Meaning |
|---|---|
| $`\bar p`$ | `p_max` over $`\mathcal{G}`$ — installed capacity |
| $`c`$ | `cost` over $`\mathcal{G}`$ — marginal cost |
| $`\bar f`$ | `cap` over $`\mathcal{L}`$ — forward transmission limit |
| $`\underline{f}`$ | `neg_cap` over $`\mathcal{L}`$ — reverse transmission limit |
| $`d`$ | `load` over $`\mathcal{S} \times \mathcal{B}`$ — demand at each bus |

#### Variables

| Symbol | Meaning |
|---|---|
| $`p`$ | `p` over $`\mathcal{S} \times \mathcal{G}`$ — output of a generator in a snapshot |
| $`f`$ | `f` over $`\mathcal{S} \times \mathcal{L}`$ — flow on a line, signed towards its `line_to` bus |

#### Definitions

| Symbol | Meaning |
|---|---|
| $`\mathit{gen\_at\_bus}`$ | `gen_at_bus` over $`\mathcal{B} \times \mathcal{S}`$ — what the generators sitting on a bus produce there |
| $`\mathit{net\_inflow}`$ | `net_inflow` over $`\mathcal{B} \times \mathcal{S}`$ — flow arriving at a bus minus flow leaving it, so a negative value is a net export |

#### Variable domains

**`p`**

```math
0 \le p_{s,g} \le \bar p_{g} \qquad \forall\, s \in \mathcal{S},\ g \in \mathcal{G}
```

**`f`**

```math
\underline{f}_{\ell} \le f_{s,\ell} \le \bar f_{\ell} \qquad \forall\, s \in \mathcal{S},\ \ell \in \mathcal{L}
```

#### Definitions

**`gen_at_bus`**

```math
\mathit{gen\_at\_bus}_{b,s} = \sum_{g \in \mathcal{G} \,:\, \mathrm{gen\_bus}(g) = b} p_{s,g} \qquad \forall\, b \in \mathcal{B},\ s \in \mathcal{S}
```

**`net_inflow`**

```math
\mathit{net\_inflow}_{b,s} = \sum_{\ell \in \mathcal{L} \,:\, \mathrm{line\_to}(\ell) = b} f_{s,\ell} - \left( \sum_{\ell \in \mathcal{L} \,:\, \mathrm{line\_from}(\ell) = b} f_{s,\ell} \right) \qquad \forall\, b \in \mathcal{B},\ s \in \mathcal{S}
```

#### Objective

```math
\min \sum_{s \in \mathcal{S},\ g \in \mathcal{G}} p_{s,g} \cdot c_{g}
```

#### Subject to

**`balance`**

```math
\mathit{gen\_at\_bus}_{b,s} + \mathit{net\_inflow}_{b,s} = d_{s,b} \qquad \forall\, s \in \mathcal{S},\ b \in \mathcal{B}
```

</details>
<!-- math:end -->

The tabs start from [the instance's tables](../howto/data.md) — one frame per parameter.

=== "specsolve"

    ```yaml
    description: >-
      Least-cost dispatch over a network, where a generator sits on a bus, a line
      joins two of them, and what is generated has to reach the load over the
      lines.

    dimensions:
      snapshot:
        description: dispatch periods
        dtype: int
        ordered: true
      generator:
        description: generating units, each sitting on one bus
        dtype: str
      bus:
        description: network nodes
        dtype: str
      line:
        description: transmission lines, each joining two buses
        dtype: str

    relations:
      gen_bus:
        description: the bus a generator sits on
        key: generator
        values: bus
      line_from:
        description: the bus a line leaves
        key: line
        values: bus
      line_to:
        description: the bus a line arrives at
        key: line
        values: bus

    parameters:
      p_max:
        description: installed capacity
        dims: [generator]
      cost:
        description: marginal cost
        dims: [generator]
      cap:
        description: forward transmission limit
        dims: [line]
      neg_cap:
        description: reverse transmission limit
        dims: [line]
      load:
        description: demand at each bus
        dims: [snapshot, bus]

    variables:
      p:
        description: output of a generator in a snapshot
        dims: [snapshot, generator]
        bounds:
          lower: 0
          upper: p_max
      f:
        description: flow on a line, signed towards its `line_to` bus
        dims: [snapshot, line]
        bounds:
          lower: neg_cap
          upper: cap

    expressions:
      gen_at_bus:
        expression: sum(p, over=generator, by=gen_bus[bus])
        description: what the generators sitting on a bus produce there
      net_inflow:
        expression: sum(f, over=line, by=line_to[bus]) - sum(f, over=line, by=line_from[bus])
        description: flow arriving at a bus minus flow leaving it, so a negative value is a net export

    constraints:
      balance:
        description: what is generated at a bus plus what arrives over the lines meets the load there
        dims: [snapshot, bus]
        expression: gen_at_bus + net_inflow == load

    objective:
      sense: minimize
      description: total cost of generation; moving power over a line is free here
      expression: sum(p * cost)
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with sps.solve('examples/transport.yaml', sources) as solution:
        solution.objective  # 4400.0
        solution.dual('balance')
    ```

=== "linopy"

    The model-building half of `examples/ports/references/linopy/transport.py`:

    ```python
    def build(tables: dict[str, pd.DataFrame]) -> linopy.Model:
        """The instance's tables as a linopy model, row for row.

        ``tables`` is the same mapping the specsolve call attaches as ``sources``.
        """
        p_max: pd.Series = tables['p_max'].set_index('generator')['value']
        cost: pd.Series = tables['cost'].set_index('generator')['value']
        cap: pd.Series = tables['cap'].set_index('line')['value']
        neg_cap: pd.Series = tables['neg_cap'].set_index('line')['value']
        load = xr.DataArray(tables['load'].pivot(index='snapshot', columns='bus', values='value'))
        snapshots, buses = load.indexes['snapshot'], load.indexes['bus']

        gen_at = pd.DataFrame(0.0, index=buses, columns=p_max.index)
        for gen, bus in zip(tables['gen_bus']['generator'], tables['gen_bus']['bus'], strict=True):
            gen_at.loc[bus, gen] = 1.0
        flow_in = pd.DataFrame(0.0, index=buses, columns=cap.index)
        for line, src, dst in zip(
            tables['line_from']['line'], tables['line_from']['bus'], tables['line_to']['bus'], strict=True
        ):
            flow_in.loc[dst, line] += 1.0
            flow_in.loc[src, line] -= 1.0

        m = linopy.Model()
        p = m.add_variables(lower=0, upper=p_max, coords=[snapshots, p_max.index], name='p')
        f = m.add_variables(lower=neg_cap, upper=cap, coords=[snapshots, cap.index], name='f')
        m.add_constraints(
            (p * xr.DataArray(gen_at)).sum('generator') + (f * xr.DataArray(flow_in)).sum('line') == load,
            name='balance',
        )
        m.add_objective((p * cost).sum())
        return m
    ```

---

[`examples/transport.yaml`](https://github.com/fluxopt/specsolve/blob/main/examples/transport.yaml) · back to [all models](index.md)
