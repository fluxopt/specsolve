# PyPSA multi-period investment — an asset exists for the years it exists

A build year and a lifetime decide which rows an asset appears in, and each period's costs carry its own discount.

> **✔ Verified against pypsa 1.2.4 (its own linopy 0.10.0)** — objective **85300.0**, matched to `rtol=1e-09`.

The [multi-period](multi_period.md) model has no build years, no lifetimes, no
discounting, and no asset that exists in one period and not the next. This
model is those four, as PyPSA writes them.

Three generators, one of each case. All three are built and all three run, so
none is decoration.

| | build year | lifetime | exists in | capacity built |
|---|---|---|---|---|
| `coal` | 2030 | 5 | 2030 | 40 |
| `gas` | 2030 | 40 | 2030, 2040 | 60 |
| `wind` | 2040 | 30 | 2040 | 80 |

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

PyPSA multi-period investment: a build year and a lifetime decide which periods an asset exists in, so a generator that has retired has no dispatch to choose, and capacity is paid for once per period it stands in — discounted by that period's weight. One unit retires between the two periods, one is built for the second, one lives through both. Optimum 85300.0, from PyPSA itself.

#### Sets

| Symbol | Meaning |
|---|---|
| $`\mathcal{T}`$ | index $`t`$ — `snapshot` with $`\mathrm{period\_of}: \mathcal{T} \to \mathcal{E}`$ — dispatch periods, each falling in one investment period |
| $`\mathcal{E}`$ | index $`e`$ — `period` with $`\mathrm{period\_of}: \mathcal{T} \to \mathcal{E}`$ — investment periods, the grouping capacity is decided and paid over |
| $`\mathcal{G}`$ | index $`g`$ — `generator` — generating units, each existing in some periods and not others |

#### Parameters

| Symbol | Meaning |
|---|---|
| $`\mathrm{load}`$ | `load` over $`\mathcal{T}`$ — demand to be met |
| $`\mathrm{period\_weight}`$ | `period_weight` over $`\mathcal{E}`$ — what one period's costs are worth at the horizon's start — the discount that makes a 2040 decision comparable with a 2030 one |
| $`\mathrm{opex}`$ | `opex` over $`\mathcal{G}`$ — cost of one unit of output |
| $`\mathrm{capex}`$ | `capex` over $`\mathcal{G}`$ — cost of holding one unit of capacity through one period |
| $`\mathrm{p}^{\mathrm{nom,max}}`$ | `p_nom_max` over $`\mathcal{G}`$ — most capacity a generator may build |
| $`\mathrm{activity}`$ | `activity` over $`\mathcal{E} \times \mathcal{G}`$ — 1 where a generator exists in a period and 0 where it does not — PyPSA derives this from a build year and a lifetime, and it decides both what may run and what is paid for |

#### Variables

| Symbol | Meaning |
|---|---|
| $`p`$ | `p` over $`\mathcal{T} \times \mathcal{G}`$ — output of a generator in a snapshot, held at zero in the snapshots whose period the generator does not exist in |
| $`p^{\mathrm{nom}}`$ | `p_nom` over $`\mathcal{G}`$ — capacity built at a generator |

#### Variable domains

**`p`**

```math
p_{t,g} \ge 0 \qquad \forall\, t \in \mathcal{T},\ g \in \mathcal{G}
```

**`p_nom`**

```math
0 \le p^{\mathrm{nom}}_{g} \le \mathrm{p}^{\mathrm{nom,max}}_{g} \qquad \forall\, g \in \mathcal{G}
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

</details>
<!-- math:end -->

The tabs start from [the instance's tables](../howto/data.md) — one frame per parameter.

=== "specsolve"

    ```yaml
    description: >-
      PyPSA multi-period investment: a build year and a lifetime decide which
      periods an asset exists in, so a generator that has retired has no dispatch to
      choose, and capacity is paid for once per period it stands in — discounted by
      that period's weight. One unit retires between the two periods, one is built
      for the second, one lives through both.
      Optimum 85300.0, from PyPSA itself.

    dimensions:
      snapshot:
        description: dispatch periods, each falling in one investment period
        dtype: int
        ordered: true
      period:
        description: investment periods, the grouping capacity is decided and paid over
        dtype: int
        ordered: true
      generator:
        description: generating units, each existing in some periods and not others
        dtype: str

    relations:
      period_of:
        description: the investment period a snapshot falls in
        key: snapshot
        values: period

    parameters:
      load:
        description: demand to be met
        dims: [snapshot]
      period_weight:
        description: >-
          what one period's costs are worth at the horizon's start — the discount
          that makes a 2040 decision comparable with a 2030 one
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
        description: >-
          1 where a generator exists in a period and 0 where it does not — PyPSA
          derives this from a build year and a lifetime, and it decides both what may
          run and what is paid for
        dims: [period, generator]

    variables:
      p:
        description: >-
          output of a generator in a snapshot, held at zero in the snapshots whose
          period the generator does not exist in
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
          a generator produces no more than the capacity built for it, and nothing at
          all in a period it does not exist in — the activity is read down onto the
          snapshot through the period relation
        dims: [snapshot, generator]
        expression: p <= p_nom * at(activity, by=period_of[period])

      power_balance:
        description: what exists in this snapshot meets the load
        dims: [snapshot]
        expression: sum(p, over=generator) == load

    objective:
      sense: minimize
      description: >-
        operating cost and capacity cost, each discounted by the weight of the
        period it falls in — capacity once per period the asset stands in, which is
        why an asset alive in both is paid for twice
      expression: >-
        sum(p * opex * at(period_weight, by=period_of[period]))
        + sum(p_nom * capex * activity * period_weight)
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with sps.solve('examples/ports/pypsa_multi_period.yaml', sources) as solution:
        solution.objective  # 85300.0
        solution.dual('power_balance')
    ```

=== "PyPSA"

    The model-building half of `examples/ports/references/pypsa/pypsa_multi_period.py`:

    ```python
    def build(tables: dict[str, pd.DataFrame]) -> pypsa.Network:
        """The port's tables as a PyPSA network, column for column.

        ``tables`` is the same mapping the specsolve call attaches as ``sources``.

        The port's flat ``snapshot`` axis carries a relation into ``period``; PyPSA
        wants the same fact as a ``(period, timestep)`` MultiIndex, so the snapshots
        are paired with the period each falls in. ``investment_period_weightings``
        takes the port's ``period_weight`` as its ``objective`` column, and ``years``
        is 10 for both periods — set explicitly, because a default of 1 would make
        the decade an accident rather than a statement.
        """
        n = pypsa.Network()
        snapshots: pd.DataFrame = tables['snapshot']
        periods = list(tables['period']['period'])
        n.set_snapshots(
            pd.MultiIndex.from_arrays(
                [tables['period_of'].set_index('snapshot')['period'][snapshots['snapshot']], snapshots['snapshot']]
            )
        )
        n.investment_periods = periods
        n.investment_period_weightings['years'] = 10
        n.investment_period_weightings['objective'] = tables['period_weight'].set_index('period')['value']

        n.add('Bus', 'hub')
        generators: pd.DataFrame = tables['generator'].set_index('generator')
        n.add(
            'Generator',
            generators.index,
            bus='hub',
            p_nom_extendable=True,
            build_year=[LIFETIMES[g][0] for g in generators.index],
            lifetime=[LIFETIMES[g][1] for g in generators.index],
            p_nom_max=tables['p_nom_max'].set_index('generator')['value'],
            marginal_cost=tables['opex'].set_index('generator')['value'],
            capital_cost=tables['capex'].set_index('generator')['value'],
        )

        load: pd.Series = tables['load'].set_index('snapshot')['value']
        n.add('Load', 'l', bus='hub', p_set=load.to_numpy())
        return n
    ```

**Capacity is paid for once per period the asset stands in.** `gas` is active in
both, so its capital cost enters twice, at 1.0 for 2030 and 0.7 for 2040.
`coal` pays once and `wind` pays once, discounted. That is what
`p_nom * capex * activity * period_weight` says. The term is summed over the
`(period, generator)` pairs the activity table holds, which is PyPSA's
`weighted_cost` loop over `investment_period_weightings.objective`
(`optimize.py:322`) written as one product.

**The other weighting does not reach the objective.**
`investment_period_weightings.years` is 10 for both periods here and changes
nothing, because `capital_cost` is given directly rather than annuitised from an
`overnight_cost` (`costs.py:119`). The port carries no `years` parameter.

**A retired asset is pinned, not absent**, the one place the port and PyPSA
differ in shape. PyPSA gives `coal` no dispatch variable in 2040 at all
(`Generator-p` is masked to the active pairs). The port keeps the variable and
multiplies its capacity bound by the activity read down through the period
relation, so `p <= p_nom * 0` holds it at zero. Same optimum and same duals, since
a variable pinned to `[0, 0]` contributes nothing, but not the same model on
paper.

Writing the absence exactly would need `where: at(activity, by=period_of[period])` on
the variable, and a `where:` cannot call `at()`: its grammar compares a name
against a literal. A second table keyed by `(snapshot, generator)` would state
one fact twice. [#982](https://github.com/fluxopt/specsolve/issues/982) asks
whether a mask may read a parameter one declared relation away.

## What it exercises

A second axis over time (`period`) with a relation down onto the snapshots, a
parameter read through that relation in both a constraint and the objective, and a
capacity variable whose cost is summed over the periods it exists in rather than
paid once. No construct here is new.
