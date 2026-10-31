# piecewise_conversion

Heat and power from converters whose flows share one curve each, where how many
flows a converter has is data.

> **✔ Agrees with hand-written linopy 0.10.0** — objective **5990**, matched to `rtol=1e-06`.

## The problem

A converter ties its flows together through one operating point: a boiler
burns fuel to make heat, a CHP (combined heat and power) unit burns fuel to
make heat *and* power. The curve is one convex combination per converter. How
many flows it ties is a property of the system, not of the file: two here,
three there, and a unit with a fourth is a row in a table.

$$\mathit{rate}_{f,t} = \sum_{k \in \mathcal{K}_{c(f)}} \lambda_{c(f),t,k}\, v_{f,k} \qquad \forall\thinspace f, t$$

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

Least-cost heat and power from two converters whose flows are tied to one piecewise curve each — a boiler tying two flows, a CHP unit tying three, and neither number written anywhere in the file.

#### Sets

| Symbol | Meaning |
|---|---|
| $`\mathcal{T}`$ | index $`t`$ — `time` — dispatch periods |
| $`\mathcal{C}`$ | index $`c`$ — `converter` with $`\mathrm{converter\_of}: \mathcal{F} \to \mathcal{C}`$ — units converting one carrier into others |
| $`\mathcal{F}`$ | index $`f`$ — `flow` with $`\mathrm{converter\_of}: \mathcal{F} \to \mathcal{C}`$ — a converter's inputs and outputs, one row each |
| $`\mathcal{B}`$ | index $`b`$ — `bp` — breakpoints, as many as the longest curve needs |

#### Parameters

| Symbol | Meaning |
|---|---|
| $`\mathrm{bp\_rate}`$ | `bp_rate` over $`\mathcal{F} \times \mathcal{B}`$, `absent` where the data has no row — what each flow runs at, at each breakpoint of its converter's curve |
| $`\mathrm{bp\_present}`$ | `bp_present` over $`\mathcal{C} \times \mathcal{B}`$, `neutral` where the data has no row — how far each converter's curve runs |
| $`\mathrm{rate}^{\mathrm{max}}`$ | `rate_max` over $`\mathcal{F}`$ — what each flow runs at when its converter is at its last breakpoint |
| $`\mathrm{is\_heat}`$ | `is_heat` over $`\mathcal{F}`$, `neutral` where the data has no row — which flows deliver heat |
| $`\mathrm{is\_power}`$ | `is_power` over $`\mathcal{F}`$, `neutral` where the data has no row — which flows deliver power |
| $`\mathrm{fuel\_price}`$ | `fuel_price` over $`\mathcal{F}`$, `neutral` where the data has no row — what a unit of each input flow costs |
| $`\mathrm{heat\_demand}`$ | `heat_demand` over $`\mathcal{T}`$ — heat to be delivered |
| $`\mathrm{power\_demand}`$ | `power_demand` over $`\mathcal{T}`$ — power to be delivered |

#### Variables

| Symbol | Meaning |
|---|---|
| $`\mathit{rate}`$ | `rate` over $`\mathcal{F} \times \mathcal{T}`$ — what each flow runs at |
| $`\mathit{weight}`$ | `weight` over $`\mathcal{C} \times \mathcal{T} \times \mathcal{B}`$ — how much of each breakpoint the converter's operating point is made of — one convex combination per converter and period, over the breakpoints its own curve runs to |

#### Variable domains

**`rate`**

```math
0 \le \mathit{rate}_{f,t} \le \mathrm{rate}^{\mathrm{max}}_{f} \qquad \forall\, f \in \mathcal{F},\ t \in \mathcal{T}
```

**`weight`**

```math
0 \le \mathit{weight}_{c,t,b} \le 1 \qquad \forall\, c \in \mathcal{C},\ t \in \mathcal{T},\ b \in \mathcal{B} \,:\, \mathrm{bp\_present}_{c,b}
```

#### Objective

```math
\min \sum_{t \in \mathcal{T}} \sum_{f \in \mathcal{F}} \mathit{rate}_{f,t} \cdot \mathrm{fuel\_price}_{f}
```

#### Subject to

**`one_operating_point`**

```math
\sum_{b \in \mathcal{B}} \mathit{weight}_{c,t,b} = 1 \qquad \forall\, c \in \mathcal{C},\ t \in \mathcal{T}
```

**`on_the_curve`**

```math
\mathit{rate}_{f,t} = \sum_{b \in \mathcal{B}} \mathit{weight}_{\mathrm{converter\_of}(f),t,b} \cdot \mathrm{bp\_rate}_{f,b} \qquad \forall\, f \in \mathcal{F},\ t \in \mathcal{T}
```

**`heat_balance`**

```math
\sum_{f \in \mathcal{F}} \mathit{rate}_{f,t} \cdot \mathrm{is\_heat}_{f} = \mathrm{heat\_demand}_{t} \qquad \forall\, t \in \mathcal{T}
```

**`power_balance`**

```math
\sum_{f \in \mathcal{F}} \mathit{rate}_{f,t} \cdot \mathrm{is\_power}_{f} = \mathrm{power\_demand}_{t} \qquad \forall\, t \in \mathcal{T}
```

**`on_one_segment`**

```math
\left( \mathit{weight}_{c,t,b} \right)_{b \in \mathcal{B}} \in \mathrm{SOS}2 \qquad \forall\, c \in \mathcal{C},\ t \in \mathcal{T}
```

</details>
<!-- math:end -->

The tabs start from [the instance's tables](../howto/data.md) — one frame per parameter.

=== "specsolve"

    ```yaml
    description: >-
      Least-cost heat and power from two converters whose flows are tied to one
      piecewise curve each — a boiler tying two flows, a CHP unit tying three, and
      neither number written anywhere in the file.

    dimensions:
      time:
        description: dispatch periods
        dtype: int
        ordered: true
      converter:
        description: units converting one carrier into others
        dtype: str
      flow:
        description: a converter's inputs and outputs, one row each
        dtype: str
      bp:
        description: breakpoints, as many as the longest curve needs
        dtype: int
        ordered: true

    relations:
      converter_of:
        description: which converter a flow belongs to
        key: flow
        values: converter

    parameters:
      bp_rate:
        description: what each flow runs at, at each breakpoint of its converter's curve
        dims: [flow, bp]
        missing: absent
      bp_present:
        description: how far each converter's curve runs
        dims: [converter, bp]
        dtype: bool
        missing: neutral
      rate_max:
        description: what each flow runs at when its converter is at its last breakpoint
        dims: [flow]
      is_heat:
        description: which flows deliver heat
        dims: [flow]
        missing: neutral
      is_power:
        description: which flows deliver power
        dims: [flow]
        missing: neutral
      fuel_price:
        description: what a unit of each input flow costs
        dims: [flow]
        missing: neutral
      heat_demand:
        description: heat to be delivered
        dims: [time]
      power_demand:
        description: power to be delivered
        dims: [time]

    variables:
      rate:
        description: what each flow runs at
        dims: [flow, time]
        bounds:
          lower: 0
          upper: rate_max
      weight:
        description: >-
          how much of each breakpoint the converter's operating point is made of —
          one convex combination per converter and period, over the breakpoints its
          own curve runs to
        dims: [converter, time, bp]
        where: bp_present
        bounds:
          lower: 0
          upper: 1

    sos:
      on_one_segment:
        description: >-
          at most two weights, and those two neighbours — which is what puts the
          operating point on a segment of the curve rather than anywhere in its hull
        variable: weight
        along: bp
        type: 2

    constraints:
      one_operating_point:
        description: each converter sits somewhere on its curve, in every period
        dims: [converter, time]
        expression: sum(weight, over=bp) == 1
      on_the_curve:
        description: every flow reads its own value off its converter's weights
        dims: [flow, time]
        expression: rate == sum(at(weight, by=converter_of[converter]) * bp_rate, over=bp)
      heat_balance:
        description: heat delivered meets the demand
        dims: [time]
        expression: sum(rate * is_heat, over=flow) == heat_demand
      power_balance:
        description: power delivered meets the demand
        dims: [time]
        expression: sum(rate * is_power, over=flow) == power_demand

    objective:
      sense: minimize
      description: what the input flows cost
      expression: sum(sum(rate * fuel_price, over=flow), over=time)
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with sps.solve(to_spec('examples/piecewise_conversion.yaml').expand(), sources) as solution:
        solution.objective  # 5990.0
    ```

=== "linopy"

    The model-building half of `examples/ports/references/linopy/piecewise_conversion.py`:

    ```python
    def build(tables: dict[str, pd.DataFrame]) -> linopy.Model:
        """The instance's tables as a linopy model, row for row.

        ``tables`` is the same mapping the specsolve call attaches as ``sources``.
        """
        flows: pd.DataFrame = tables['flow'].set_index('flow').join(tables['converter_of'].set_index('flow'))
        times = pd.Index(tables['time']['time'], name='time')
        present = tables['bp_present'].set_index(['converter', 'bp'])['value']
        runs_to = {c: int(present[c].sum()) for c in present.index.get_level_values('converter').unique()}

        m = linopy.Model()
        rate = m.add_variables(lower=0, coords=[pd.Index(flows.index, name='flow'), times], name='rate')

        for converter, members in flows.groupby('converter'):
            pairs = [
                (rate.sel(flow=flow, drop=True), linopy.breakpoints(curve_of(tables, flow, runs_to[converter])))
                for flow in members.index
            ]
            m.add_piecewise_formulation(*pairs, name=f'curve_{converter}')

        for carrier, demand in (('is_heat', 'heat_demand'), ('is_power', 'power_demand')):
            weights = tables[carrier].set_index('flow')['value'].reindex(flows.index)
            wanted = tables[demand].set_index('time')['value'].reindex(times)
            m.add_constraints((rate * weights).sum('flow') == wanted, name=demand)

        price = tables['fuel_price'].set_index('flow')['value'].reindex(flows.index)
        m.add_objective((rate * price).sum())
        return m
    ```

---

[`examples/piecewise_conversion.yaml`](https://github.com/fluxopt/specsolve/blob/main/examples/piecewise_conversion.yaml) · back to [all models](index.md)
