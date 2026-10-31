# PyPSA Store — one signed power, and no rating at all

The component every sector-coupled PyPSA model uses for hydrogen, heat and gas.

> **✔ Verified against pypsa 1.2.4 (its own linopy 0.10.0)** — objective **7005.5025000000005**, matched to `rtol=1e-09`.

[Storage units](pypsa_storage.md) ports the `StorageUnit`: a dispatch/store pair
of non-negative variables, one per efficiency, and a power rating of its own. A
`Store` is a different component:

| | `StorageUnit` | `Store` |
|---|---|---|
| power | two non-negative variables | **one signed** variable |
| efficiencies | store and dispatch, separately | **none** |
| power rating | `p_nom` | **none** — the level is the only limit |
| capacity built | `p_nom` (power) | `e_nom` (energy) |

The tank starts 20 MWh full, fills over the three quiet snapshots and drains
over the two busy ones, losing 5% of what it holds each step. Its capacity is a
decision: 75.1 MWh, the exact peak of its own trajectory.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

PyPSA's Store component: one signed power at the bus, no efficiencies and no power rating — only the energy level limits how fast it moves. The tank fills early and drains late, losing a share of what it holds every snapshot, and its energy capacity is built rather than given. Optimum 7005.5025000000005, from PyPSA itself.

#### Sets

| Symbol | Meaning |
|---|---|
| $`\mathcal{T}`$ | index $`t`$ — `snapshot` — dispatch periods |
| $`\mathcal{B}`$ | index $`b`$ — `bus` with $`\mathrm{gen\_bus}: \mathcal{G} \to \mathcal{B},\ \mathrm{store\_bus}: \mathcal{S} \to \mathcal{B}`$ — network nodes |
| $`\mathcal{G}`$ | index $`g`$ — `generator` with $`\mathrm{gen\_bus}: \mathcal{G} \to \mathcal{B}`$ — generating units, each sitting on one bus |
| $`\mathcal{S}`$ | index $`s`$ — `store` with $`\mathrm{store\_bus}: \mathcal{S} \to \mathcal{B}`$ — energy stores, each sitting on one bus |

#### Parameters

| Symbol | Meaning |
|---|---|
| $`\mathrm{p}^{\mathrm{nom}}`$ | `p_nom` over $`\mathcal{G}`$ — installed capacity of a generator |
| $`\mathrm{marginal\_cost}`$ | `marginal_cost` over $`\mathcal{G}`$ — cost of one unit of output |
| $`\mathrm{e}^{\mathrm{nom,max}}`$ | `e_nom_max` over $`\mathcal{S}`$ — most energy capacity that may be built at a store |
| $`\mathrm{e}^{\mathrm{capital,cost}}`$ | `e_capital_cost` over $`\mathcal{S}`$ — cost of holding one unit of energy capacity over the horizon |
| $`\mathrm{e}^{\mathrm{initial}}`$ | `e_initial` over $`\mathcal{S}`$ — energy in the store before the first snapshot |
| $`\mathrm{standing\_loss}`$ | `standing_loss` over $`\mathcal{S}`$ — share of the carried-over level lost between snapshots |
| $`\mathrm{load}`$ | `load` over $`\mathcal{T} \times \mathcal{B}`$ — demand at each bus in each snapshot |

#### Variables

| Symbol | Meaning |
|---|---|
| $`p`$ | `p` over $`\mathcal{T} \times \mathcal{G}`$ — output of a generator in a snapshot |
| $`\mathit{store\_p}`$ | `store_p` over $`\mathcal{T} \times \mathcal{S}`$ — power a store puts onto its bus, negative when it takes power off — unbounded, because a Store has no rating of its own |
| $`e`$ | `e` over $`\mathcal{T} \times \mathcal{S}`$ — energy in the store at the end of a snapshot |
| $`e^{\mathrm{nom}}`$ | `e_nom` over $`\mathcal{S}`$ — energy capacity built at a store |

#### Notation

| Symbol | Meaning |
|---|---|
| $`\mathrm{pos}(t)`$ | `position()` — the position of $`t`$, counted from $`0`$; $`t`$ is the label |

#### Variable domains

**`p`**

```math
0 \le p_{t,g} \le \mathrm{p}^{\mathrm{nom}}_{g} \qquad \forall\, t \in \mathcal{T},\ g \in \mathcal{G}
```

**`store_p`**

```math
\mathit{store\_p}_{t,s} \in \mathbb{R} \qquad \forall\, t \in \mathcal{T},\ s \in \mathcal{S}
```

**`e`**

```math
e_{t,s} \ge 0 \qquad \forall\, t \in \mathcal{T},\ s \in \mathcal{S}
```

**`e_nom`**

```math
0 \le e^{\mathrm{nom}}_{s} \le \mathrm{e}^{\mathrm{nom,max}}_{s} \qquad \forall\, s \in \mathcal{S}
```

#### Objective

```math
\min \sum_{t \in \mathcal{T},\ g \in \mathcal{G}} p_{t,g} \cdot \mathrm{marginal\_cost}_{g} + \sum_{s \in \mathcal{S}} e^{\mathrm{nom}}_{s} \cdot \mathrm{e}^{\mathrm{capital,cost}}_{s}
```

#### Subject to

**`nodal_balance`**

```math
\sum_{g \in \mathcal{G} \,:\, \mathrm{gen\_bus}(g) = b} p_{t,g} + \sum_{s \in \mathcal{S} \,:\, \mathrm{store\_bus}(s) = b} \mathit{store\_p}_{t,s} = \mathrm{load}_{t,b} \qquad \forall\, t \in \mathcal{T},\ b \in \mathcal{B}
```

**`within_capacity`**

```math
e_{t,s} \le e^{\mathrm{nom}}_{s} \qquad \forall\, t \in \mathcal{T},\ s \in \mathcal{S}
```

**`energy_balance_initial`**

```math
e_{t,s} = \mathrm{e}^{\mathrm{initial}}_{s} - \mathit{store\_p}_{t,s} \qquad \forall\, t \in \mathcal{T},\ s \in \mathcal{S} \,:\, \mathrm{pos}(t) = 0
```

**`energy_balance`**

```math
e_{t,s} = e_{t - 1,s} \cdot \left( 1 - \mathrm{standing\_loss}_{s} \right) - \mathit{store\_p}_{t,s} \qquad \forall\, t \in \mathcal{T},\ s \in \mathcal{S}
```

</details>
<!-- math:end -->

The tabs start from [the instance's tables](../howto/data.md) — one frame per parameter.

=== "specsolve"

    ```yaml
    description: >-
      PyPSA's Store component: one signed power at the bus, no efficiencies and no
      power rating — only the energy level limits how fast it moves. The tank fills
      early and drains late, losing a share of what it holds every snapshot, and its
      energy capacity is built rather than given. Optimum 7005.5025000000005, from
      PyPSA itself.

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
      store:
        description: energy stores, each sitting on one bus
        dtype: str

    relations:
      gen_bus:
        description: the bus a generator sits on
        key: generator
        values: bus
      store_bus:
        description: the bus a store sits on
        key: store
        values: bus

    parameters:
      p_nom:
        description: installed capacity of a generator
        dims: [generator]
      marginal_cost:
        description: cost of one unit of output
        dims: [generator]
      e_nom_max:
        description: most energy capacity that may be built at a store
        dims: [store]
      e_capital_cost:
        description: cost of holding one unit of energy capacity over the horizon
        dims: [store]
      e_initial:
        description: energy in the store before the first snapshot
        dims: [store]
      standing_loss:
        description: share of the carried-over level lost between snapshots
        dims: [store]
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
      store_p:
        description: >-
          power a store puts onto its bus, negative when it takes power off —
          unbounded, because a Store has no rating of its own
        dims: [snapshot, store]
      e:
        description: energy in the store at the end of a snapshot
        dims: [snapshot, store]
        bounds:
          lower: 0
      e_nom:
        description: energy capacity built at a store
        dims: [store]
        bounds:
          lower: 0
          upper: e_nom_max

    constraints:
      nodal_balance:
        description: what is generated at a bus, plus what the stores supply, meets the load there
        dims: [snapshot, bus]
        expression: sum(p, over=generator, by=gen_bus[bus]) + sum(store_p, over=store, by=store_bus[bus]) == load

      within_capacity:
        description: a store holds no more energy than the capacity built for it
        dims: [snapshot, store]
        expression: e <= e_nom

      energy_balance_initial:
        description: >-
          the first snapshot's level starts from the initial energy, which the
          standing loss does not decay because nothing was carried into it
        dims: [snapshot, store]
        where: "position(snapshot) == 0"
        expression: e == e_initial - store_p

      energy_balance:
        description: >-
          the level carried into a snapshot, decayed by the standing loss, less what
          the store supplied to its bus
        dims: [snapshot, store]
        expression: >-
          e == shift(e, along=snapshot, offset=1) * (1 - standing_loss) - store_p

    objective:
      sense: minimize
      description: what the fleet costs to run, plus what the store's energy capacity costs to build
      expression: sum(p * marginal_cost) + sum(e_nom * e_capital_cost)
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with sps.solve('examples/ports/pypsa_store.yaml', sources) as solution:
        solution.objective  # 7005.5025000000005
        solution.dual('nodal_balance')
    ```

=== "PyPSA"

    The model-building half of `examples/ports/references/pypsa/pypsa_store.py`:

    ```python
    def build(tables: dict[str, pd.DataFrame]) -> pypsa.Network:
        """The port's tables as a PyPSA network, column for column.

        ``tables`` is the same mapping the specsolve call attaches as ``sources``.

        ``Store`` takes no power rating: ``e_nom`` bounds the level, and the power
        that moves it is limited only by what the level allows within one snapshot.
        That is why the port declares its store power with no bounds at all.
        """
        n = pypsa.Network()
        n.set_snapshots(tables['snapshot']['snapshot'])
        n.add('Bus', tables['bus']['bus'])

        generators: pd.DataFrame = tables['generator'].set_index('generator')
        n.add(
            'Generator',
            generators.index,
            bus=tables['gen_bus'].set_index('generator')['bus'],
            p_nom=tables['p_nom'].set_index('generator')['value'],
            marginal_cost=tables['marginal_cost'].set_index('generator')['value'],
        )

        stores: pd.DataFrame = tables['store'].set_index('store')
        n.add(
            'Store',
            stores.index,
            bus=tables['store_bus'].set_index('store')['bus'],
            e_nom_extendable=True,
            e_nom_max=tables['e_nom_max'].set_index('store')['value'],
            capital_cost=tables['e_capital_cost'].set_index('store')['value'],
            e_initial=tables['e_initial'].set_index('store')['value'],
            standing_loss=tables['standing_loss'].set_index('store')['value'],
            e_cyclic=False,
        )

        load: pd.DataFrame = tables['load'].pivot(index='snapshot', columns='bus', values='value')
        for bus in tables['bus']['bus']:
            n.add('Load', f'load_{bus}', bus=bus, p_set=load[bus])
        return n
    ```

**The standing loss is visible in the price vector, so the reference reads it.**
The nodal prices run `68.79, 72.41, 76.23, 85.50, 90.00, 10.00`. Each earlier
snapshot's price is the next one's divided by 0.95, because a unit stored now is
worth 0.95 of a unit later. A port that dropped the decay would still solve and
still look sensible. It would hold more energy than it should, buy less gas, and
report a lower cost. The dual vector catches that where a single objective
figure might not.

**The initial level is not decayed, and the instance can tell.** PyPSA's first
row is `e = e_initial - p`, so the 20 MWh in the tank before the horizon arrives
whole. Decay it and the same instance costs **7074.30** against **7005.50**. A
version with an empty tank reports 3116.36 either way, which is why the tank
starts full.

**A store with no rating still cannot move arbitrary power**, because the level
it draws from is bounded and so is the level it charges into. The port declares
`store_p` with no bounds at all. An upper bound here would be a limit PyPSA does
not have.

## What it exercises

An energy recurrence over one signed variable, a capacity variable bounding a
time-indexed variable (`e <= e_nom`, two decisions rather than a decision and a
bound), and a `shift` across the horizon's opening boundary.
