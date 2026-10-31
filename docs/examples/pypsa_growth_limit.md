# PyPSA growth limit — what may be built depends on what was built last time

A cap on new capacity per investment period, which grows with the period before it. The first `shift` in the corpus along an axis that is not time-of-day.

> **✔ Verified against pypsa 1.2.4 (its own linopy 0.10.0)** — objective **47110.0**, matched to `rtol=1e-09`.

`Carrier.max_growth` caps how much of a technology may be *newly built* in one
period. `max_relative_growth` adds a share of the previous period's new build to
that allowance, which turns a flat cap into a growth rate
(`global_constraints.py:184`):

```
new[period] - max_relative_growth * new[period - 1] <= max_growth
```

Two things the source settles. The quantity on both sides is **newly built**
capacity, not standing capacity: `vars.where(first_active)` counts an asset in
the period it first exists and never again. And the first period has no
predecessor, so its row is the bare allowance.

The three wind units are one per period, which is how a build year becomes a
column. Each is extendable and each first stands in its own period, so
`new[period]` is that unit's capacity.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

PyPSA's carrier growth limit: how much of a technology may be built in one investment period, given how much was built in the one before. Three periods, one capped carrier, and a row that couples each period to its predecessor — the first `shift` in the corpus over an axis that is not time-of-day. Optimum 47110.0, from PyPSA itself.

#### Sets

| Symbol | Meaning |
|---|---|
| $`\mathcal{T}`$ | index $`t`$ — `snapshot` with $`\mathrm{period\_of}: \mathcal{T} \to \mathcal{E}`$ — dispatch periods, each falling in one investment period |
| $`\mathcal{E}`$ | index $`e`$ — `period` with $`\mathrm{build\_period}: \mathcal{G} \to \mathcal{E},\ \mathrm{period\_of}: \mathcal{T} \to \mathcal{E}`$ — investment periods, the axis capacity is built along |
| $`\mathcal{C}`$ | index $`c`$ — `carrier` with $`\mathrm{gen\_carrier}: \mathcal{G} \to \mathcal{C}`$ — what a generator burns, and what a growth limit is a property of |
| $`\mathcal{G}`$ | index $`g`$ — `generator` with $`\mathrm{gen\_carrier}: \mathcal{G} \to \mathcal{C},\ \mathrm{build\_period}: \mathcal{G} \to \mathcal{E}`$ — generating units, each built in one period and standing from then on |

#### Parameters

| Symbol | Meaning |
|---|---|
| $`\mathrm{load}`$ | `load` over $`\mathcal{T}`$ — demand to be met |
| $`\mathrm{period\_weight}`$ | `period_weight` over $`\mathcal{E}`$ — what one period's costs are worth at the horizon's start |
| $`\mathrm{opex}`$ | `opex` over $`\mathcal{G}`$ — cost of one unit of output |
| $`\mathrm{capex}`$ | `capex` over $`\mathcal{G}`$ — cost of holding one unit of capacity through one period |
| $`\mathrm{p}^{\mathrm{nom,max}}`$ | `p_nom_max` over $`\mathcal{G}`$ — most capacity a generator may build |
| $`\mathrm{activity}`$ | `activity` over $`\mathcal{E} \times \mathcal{G}`$ — 1 where a generator stands in a period and 0 where it does not |
| $`\mathrm{capped\_carrier}`$ | `capped_carrier` over $`\mathcal{C}`$ — 1 for the carrier whose growth is capped, 0 for the rest — the selection PyPSA makes by reading its carrier table |
| $`\mathrm{max\_growth}`$ | `max_growth` (scalar) — most capacity of that carrier that may be newly built in one period |
| $`\mathrm{max\_relative\_growth}`$ | `max_relative_growth` (scalar) — how much of the previous period's new capacity is added to that allowance — what makes the limit a growth rate rather than a flat cap |

#### Variables

| Symbol | Meaning |
|---|---|
| $`p`$ | `p` over $`\mathcal{T} \times \mathcal{G}`$ — output of a generator in a snapshot, zero where it does not yet stand |
| $`p^{\mathrm{nom}}`$ | `p_nom` over $`\mathcal{G}`$ — capacity built at a generator |

#### Definitions

| Symbol | Meaning |
|---|---|
| $`\mathit{new\_capacity}`$ | `new_capacity` over $`\mathcal{E}`$ — capacity of the capped carrier first standing in a period: each generator's capacity counted once, in the period it is built, and never again |

#### Notation

| Symbol | Meaning |
|---|---|
| $`t \boxminus_{v} k`$ | `edge=v` — $`t-k`$, and $`v`$ past the edge |

#### Variable domains

**`p`**

```math
p_{t,g} \ge 0 \qquad \forall\, t \in \mathcal{T},\ g \in \mathcal{G}
```

**`p_nom`**

```math
0 \le p^{\mathrm{nom}}_{g} \le \mathrm{p}^{\mathrm{nom,max}}_{g} \qquad \forall\, g \in \mathcal{G}
```

#### Definitions

**`new_capacity`**

```math
\mathit{new\_capacity}_{e} = \sum_{g \in \mathcal{G} \,:\, \mathrm{build\_period}(g) = e} p^{\mathrm{nom}}_{g} \cdot \mathrm{capped\_carrier}_{\mathrm{gen\_carrier}(g)} \qquad \forall\, e \in \mathcal{E}
```

#### Objective

```math
\min \sum_{t \in \mathcal{T},\ g \in \mathcal{G}} p_{t,g} \cdot \mathrm{opex}_{g} \cdot \mathrm{period\_weight}_{\mathrm{period\_of}(t)} + \sum_{e \in \mathcal{E},\ g \in \mathcal{G}} p^{\mathrm{nom}}_{g} \cdot \mathrm{capex}_{g} \cdot \mathrm{activity}_{e,g} \cdot \mathrm{period\_weight}_{e}
```

#### Subject to

**`within_capacity`**

```math
p_{t,g} \le p^{\mathrm{nom}}_{g} \cdot \mathrm{activity}_{\mathrm{period\_of}(t),g} \qquad \forall\, t \in \mathcal{T},\ g \in \mathcal{G}
```

**`power_balance`**

```math
\sum_{g \in \mathcal{G}} p_{t,g} = \mathrm{load}_{t} \qquad \forall\, t \in \mathcal{T}
```

**`growth_limit`**

```math
\mathit{new\_capacity}_{e} - \mathit{new\_capacity}_{e \boxminus_{0} 1} \cdot \mathrm{max\_relative\_growth} \le \mathrm{max\_growth} \qquad \forall\, e \in \mathcal{E}
```

</details>
<!-- math:end -->

The tabs start from [the instance's tables](../howto/data.md) — one frame per parameter.

=== "specsolve"

    ```yaml
    description: >-
      PyPSA's carrier growth limit: how much of a technology may be built in one
      investment period, given how much was built in the one before. Three periods,
      one capped carrier, and a row that couples each period to its predecessor —
      the first `shift` in the corpus over an axis that is not time-of-day.
      Optimum 47110.0, from PyPSA itself.

    dimensions:
      snapshot:
        description: dispatch periods, each falling in one investment period
        dtype: int
        ordered: true
      period:
        description: investment periods, the axis capacity is built along
        dtype: int
        ordered: true
      carrier:
        description: what a generator burns, and what a growth limit is a property of
        dtype: str
      generator:
        description: generating units, each built in one period and standing from then on
        dtype: str

    relations:
      gen_carrier:
        description: the carrier a generator burns
        key: generator
        values: carrier
      build_period:
        description: the period a generator is first built in, and so counted as new in
        key: generator
        values: period
      period_of:
        description: the investment period a snapshot falls in
        key: snapshot
        values: period

    parameters:
      load:
        description: demand to be met
        dims: [snapshot]
      period_weight:
        description: what one period's costs are worth at the horizon's start
        dims: [period]
      opex:
        description: cost of one unit of output
        dims: [generator]
      capex:
        description: cost of holding one unit of capacity through one period
        dims: [generator]
      p_nom_max:
        description: most capacity a generator may build
        dims: [generator]
      activity:
        description: 1 where a generator stands in a period and 0 where it does not
        dims: [period, generator]
      capped_carrier:
        description: >-
          1 for the carrier whose growth is capped, 0 for the rest — the selection
          PyPSA makes by reading its carrier table
        dims: [carrier]
      max_growth:
        description: most capacity of that carrier that may be newly built in one period
        dims: []
      max_relative_growth:
        description: >-
          how much of the previous period's new capacity is added to that allowance —
          what makes the limit a growth rate rather than a flat cap
        dims: []

    expressions:
      new_capacity:
        description: >-
          capacity of the capped carrier first standing in a period: each generator's
          capacity counted once, in the period it is built, and never again
        expression: sum(p_nom * at(capped_carrier, by=gen_carrier[carrier]), over=generator, by=build_period[period])

    variables:
      p:
        description: output of a generator in a snapshot, zero where it does not yet stand
        dims: [snapshot, generator]
        bounds:
          lower: 0
      p_nom:
        description: capacity built at a generator
        dims: [generator]
        bounds:
          lower: 0
          upper: p_nom_max

    constraints:
      within_capacity:
        description: >-
          a generator produces no more than the capacity built for it, and nothing in
          a period it does not stand in
        dims: [snapshot, generator]
        expression: p <= p_nom * at(activity, by=period_of[period])

      power_balance:
        dims: [snapshot]
        expression: sum(p, over=generator) == load

      growth_limit:
        description: >-
          what may be built of the capped carrier in a period is its allowance plus a
          share of what was built in the period before. `edge=0` keeps the first
          period's row, where there is no predecessor to grow from, as the bare
          allowance — which is the row PyPSA emits there.
        dims: [period]
        expression: >-
          new_capacity - shift(new_capacity, along=period, offset=1, edge=0) * max_relative_growth
          <= max_growth

    objective:
      sense: minimize
      description: >-
        operating and capacity cost, each discounted by the weight of the period it
        falls in — capacity once per period the generator stands in
      expression: >-
        sum(p * opex * at(period_weight, by=period_of[period]))
        + sum(p_nom * capex * activity * period_weight)
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with sps.solve('examples/ports/pypsa_growth_limit.yaml', sources) as solution:
        solution.objective  # 47110.0
        solution.dual('power_balance')
    ```

=== "PyPSA"

    The model-building half of `examples/ports/references/pypsa/pypsa_growth_limit.py`:

    ```python
    def build(tables: dict[str, pd.DataFrame], growth_limit: bool = True) -> pypsa.Network:
        """The port's tables as a PyPSA network, column for column.

        ``tables`` is the same mapping the specsolve call attaches as ``sources``.

        ``growth_limit=False`` drops the two carrier attributes, which is how
        ``main`` measures what the limit is worth. The port's ``build_period`` relation
        is PyPSA's ``build_year``; its ``activity`` table is what ``build_year`` and
        ``lifetime`` derive.
        """
        n = pypsa.Network()
        snapshots: pd.DataFrame = tables['snapshot']
        n.set_snapshots(
            pd.MultiIndex.from_arrays(
                [tables['period_of'].set_index('snapshot')['period'][snapshots['snapshot']], snapshots['snapshot']]
            )
        )
        n.investment_periods = list(tables['period']['period'])
        n.investment_period_weightings['years'] = 10
        n.investment_period_weightings['objective'] = tables['period_weight'].set_index('period')['value']

        n.add('Bus', 'hub')
        for carrier in tables['carrier']['carrier']:
            limited = growth_limit and carrier == 'wind'
            n.add(
                'Carrier',
                carrier,
                max_growth=float(tables['max_growth']) if limited else float('inf'),
                max_relative_growth=float(tables['max_relative_growth']) if limited else 0.0,
            )

        generators: pd.DataFrame = tables['generator'].set_index('generator')
        n.add(
            'Generator',
            generators.index,
            bus='hub',
            carrier=tables['gen_carrier'].set_index('generator')['carrier'],
            p_nom_extendable=True,
            build_year=tables['build_period'].set_index('generator')['period'],
            lifetime=LIFETIME,
            p_nom_max=tables['p_nom_max'].set_index('generator')['value'],
            marginal_cost=tables['opex'].set_index('generator')['value'],
            capital_cost=tables['capex'].set_index('generator')['value'],
        )

        load: pd.Series = tables['load'].set_index('snapshot')['value']
        n.add('Load', 'l', bus='hub', p_set=load.to_numpy())
        return n
    ```

**The limit binds in every period, and it is worth 4630.** Wind is capped to 15,
then 22.5, then 26.25: each period's allowance is `15 + 0.5 ×` the last build.
`gas` grows to 86.25 to cover what wind may not. Drop the two carrier attributes
and the same instance builds 30, 40 and 50 of wind, 30 of gas, and costs
**42480.0** against **47110.0**. The duals on the coupled rows are −148 and
−120.

**`edge=0` keeps the first period's row.** Without it the shifted term is absent
in the first period and the *row* goes with it. A masked variable term deletes
the row rather than zeroing it. PyPSA emits the bare allowance there:

```
[wind, 2030]: +1 Generator-p_nom[wind_2030]                                  ≤ 15.0
[wind, 2040]: +1 Generator-p_nom[wind_2040] - 0.5 Generator-p_nom[wind_2030] ≤ 15.0
[wind, 2050]: +1 Generator-p_nom[wind_2050] - 0.5 Generator-p_nom[wind_2040] ≤ 15.0
```

**A named expression is used twice in one row.** `new_capacity` appears once as
itself and once shifted. Writing the grouped sum twice would be two chances to
write it differently.

## What it exercises

`shift` over an investment-period axis rather than a snapshot one, a named
expression used twice in one row, and a capacity grouped onto the period it is
built in through a `build_period` relation.
