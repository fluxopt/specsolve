# PyPSA energy totals — a bound across the whole horizon

A generator's dispatch reduced over every snapshot and bounded: a contracted delivery, a reservoir's season.

> **✔ Verified against pypsa 1.2.4 (its own linopy 0.10.0)** — objective **21400.0**, matched to `rtol=1e-09`.

Every other bound in the corpus holds *within* one snapshot. `e_sum_min` and
`e_sum_max` hold across all of them at once, which is how a fuel allowance, a
take-or-pay contract and a hydro reservoir's season are all written.

Three generators, two bounds. `hydro` is cheapest but capped at 200 MWh;
`contract` is dearest yet owes 150 MWh whatever the merit order says; `gas` is
free to balance. Both bounds bind, and they bind in opposite directions.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

PyPSA energy-total bounds: a generator's dispatch reduced over the whole horizon and bounded — a contracted delivery on one, a reservoir's season on another, and a third free to balance. Optimum 21400.0, from PyPSA itself.

#### Sets

| Symbol | Meaning |
|---|---|
| $`\mathcal{T}`$ | index $`t`$ — `snapshot` — dispatch periods |
| $`\mathcal{B}`$ | index $`b`$ — `bus` with $`\mathrm{gen\_bus}: \mathcal{G} \to \mathcal{B}`$ — network nodes |
| $`\mathcal{G}`$ | index $`g`$ — `generator` with $`\mathrm{gen\_bus}: \mathcal{G} \to \mathcal{B}`$ — generating units, each sitting on one bus |

#### Parameters

| Symbol | Meaning |
|---|---|
| $`\mathrm{weighting}`$ | `weighting` over $`\mathcal{T}`$ — hours a snapshot stands for — what turns a power into an energy |
| $`\mathrm{p}^{\mathrm{nom}}`$ | `p_nom` over $`\mathcal{G}`$ — installed capacity of a generator |
| $`\mathrm{marginal\_cost}`$ | `marginal_cost` over $`\mathcal{G}`$ — cost of one unit of output |
| $`\mathrm{e}^{\mathrm{sum,max}}`$ | `e_sum_max` over $`\mathcal{G}`$, `absent` where the data has no row — most energy a generator may deliver over the whole horizon, for the generators that have such a limit |
| $`\mathrm{e}^{\mathrm{sum,min}}`$ | `e_sum_min` over $`\mathcal{G}`$, `absent` where the data has no row — least energy a generator must deliver over the whole horizon, for the generators that owe one |
| $`\mathrm{load}`$ | `load` over $`\mathcal{T} \times \mathcal{B}`$ — demand at each bus in each snapshot |

#### Variables

| Symbol | Meaning |
|---|---|
| $`p`$ | `p` over $`\mathcal{T} \times \mathcal{G}`$ — output of a generator in a snapshot |

#### Variable domains

**`p`**

```math
0 \le p_{t,g} \le \mathrm{p}^{\mathrm{nom}}_{g} \qquad \forall\, t \in \mathcal{T},\ g \in \mathcal{G}
```

#### Objective

```math
\min \sum_{t \in \mathcal{T},\ g \in \mathcal{G}} p_{t,g} \cdot \mathrm{marginal\_cost}_{g} \cdot \mathrm{weighting}_{t}
```

#### Subject to

**`nodal_balance`**

```math
\sum_{g \in \mathcal{G} \,:\, \mathrm{gen\_bus}(g) = b} p_{t,g} = \mathrm{load}_{t,b} \qquad \forall\, t \in \mathcal{T},\ b \in \mathcal{B}
```

**`energy_cap`**

```math
\sum_{t \in \mathcal{T}} p_{t,g} \cdot \mathrm{weighting}_{t} \le \mathrm{e}^{\mathrm{sum,max}}_{g} \qquad \forall\, g \in \mathcal{G} \,:\, \mathrm{e}^{\mathrm{sum,max}}_{g} \text{ is defined}
```

**`energy_floor`**

```math
\sum_{t \in \mathcal{T}} p_{t,g} \cdot \mathrm{weighting}_{t} \ge \mathrm{e}^{\mathrm{sum,min}}_{g} \qquad \forall\, g \in \mathcal{G} \,:\, \mathrm{e}^{\mathrm{sum,min}}_{g} \text{ is defined}
```

</details>
<!-- math:end -->

The tabs start from [the instance's tables](../howto/data.md) — one frame per parameter.

=== "specsolve"

    ```yaml
    description: >-
      PyPSA energy-total bounds: a generator's dispatch reduced over the whole
      horizon and bounded — a contracted delivery on one, a reservoir's season on
      another, and a third free to balance. Optimum 21400.0, from PyPSA itself.

    dimensions:
      snapshot:
        description: dispatch periods
        dtype: int
        ordered: true
      bus:
        description: network nodes
        dtype: str
      generator:
        description: generating units, each sitting on one bus
        dtype: str

    relations:
      gen_bus:
        description: the bus a generator sits on
        key: generator
        values: bus

    parameters:
      weighting:
        description: hours a snapshot stands for — what turns a power into an energy
        dims: [snapshot]
      p_nom:
        description: installed capacity of a generator
        dims: [generator]
      marginal_cost:
        description: cost of one unit of output
        dims: [generator]
      e_sum_max:
        description: >-
          most energy a generator may deliver over the whole horizon, for the
          generators that have such a limit
        dims: [generator]
        missing: absent
      e_sum_min:
        description: >-
          least energy a generator must deliver over the whole horizon, for the
          generators that owe one
        dims: [generator]
        missing: absent
      load:
        description: demand at each bus in each snapshot
        dims: [snapshot, bus]

    variables:
      p:
        description: output of a generator in a snapshot
        dims: [snapshot, generator]
        bounds:
          lower: 0
          upper: p_nom

    constraints:
      nodal_balance:
        description: what is generated at a bus meets the load there
        dims: [snapshot, bus]
        expression: sum(p, over=generator, by=gen_bus[bus]) == load

      energy_cap:
        description: >-
          a generator with a ceiling on total energy delivers no more than it over
          the horizon — the weighting is what makes the sum an energy rather than a
          count of snapshots
        dims: [generator]
        where: e_sum_max
        expression: sum(p * weighting, over=snapshot) <= e_sum_max

      energy_floor:
        description: a generator that owes a total delivers at least it over the horizon
        dims: [generator]
        where: e_sum_min
        expression: sum(p * weighting, over=snapshot) >= e_sum_min

    objective:
      sense: minimize
      description: what the fleet costs to run, each snapshot weighted by the hours it stands for
      expression: sum(p * marginal_cost * weighting)
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with sps.solve('examples/ports/pypsa_energy_sum.yaml', sources) as solution:
        solution.objective  # 21400.0
        solution.dual('nodal_balance')
    ```

=== "PyPSA"

    The model-building half of `examples/ports/references/pypsa/pypsa_energy_sum.py`:

    ```python
    def build(tables: dict[str, pd.DataFrame]) -> pypsa.Network:
        """The port's tables as a PyPSA network, column for column.

        ``tables`` is the same mapping the specsolve call attaches as ``sources``.

        The energy bounds arrive as short frames — one row per generator that has
        one — and are reindexed onto the full generator index, which is where the
        infinities PyPSA tests for come from. All three weighting columns are set
        together: ``generators`` scales the energy the bounds see, ``objective``
        scales the cost, and leaving them to disagree would make the number an
        accident of a default.
        """
        n = pypsa.Network()
        n.set_snapshots(tables['snapshot']['snapshot'])
        n.snapshot_weightings.loc[:, :] = tables['weighting'].set_index('snapshot')['value'].to_numpy()[:, None]
        n.add('Bus', tables['bus']['bus'])

        generators: pd.DataFrame = tables['generator'].set_index('generator')
        e_sum_max = tables['e_sum_max'].set_index('generator')['value'].reindex(generators.index, fill_value=float('inf'))
        e_sum_min = tables['e_sum_min'].set_index('generator')['value'].reindex(generators.index, fill_value=float('-inf'))
        n.add(
            'Generator',
            generators.index,
            bus=tables['gen_bus'].set_index('generator')['bus'],
            p_nom=tables['p_nom'].set_index('generator')['value'],
            marginal_cost=tables['marginal_cost'].set_index('generator')['value'],
            e_sum_max=e_sum_max,
            e_sum_min=e_sum_min,
        )

        load: pd.DataFrame = tables['load'].pivot(index='snapshot', columns='bus', values='value')
        for bus in tables['bus']['bus']:
            n.add('Load', f'load_{bus}', bus=bus, p_set=load[bus])
        return n
    ```

**A bound only some rows have is written, not inferred.** PyPSA defaults the two
attributes to ±∞ and emits a row only where the value is finite. Here the
tables are short, one row in `e_sum_max` and one in `e_sum_min`, and the
constraint carries `where: e_sum_max`. Leaving the `where:` off is refused at
load time. A missing row on a comparison's constant side would be the bound,
so `x <= 0` would bind where the model said nothing.

**The snapshot weightings are not 1, and that changes what a dual means.** They
are the hours each snapshot stands for, and they enter twice: once in the
energy the bounds see, once in the cost. PyPSA divides the nodal-balance dual
by the objective weighting before publishing it as `marginal_price`, so its
figure reads per unit energy. That is a flat **60** against a dual of
**60, 120, 180, 120**. Every model above weights its snapshots 1 and hides the
division. The recorded reference is the dual, the object both models hold.

## What it exercises

A reduction over a dimension the constraint does not span: `sum(p * weighting,
over=snapshot)` with `dims: [generator]`. The `where:` masking a row to where
its bound exists, on both a `<=` and a `>=`. And a parameter that multiplies
inside a reduction and again in the objective.
