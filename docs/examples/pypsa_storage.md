# PyPSA LOPF — storage units

[Ramp limits](pypsa_ramp.md) plus a `StorageUnit` carrying energy between snapshots.

> **✔ Verified against pypsa 1.2.4 (its own linopy 0.10.0)** — objective **15253.178322993519**, matched to `rtol=1e-09`.

Non-cyclic: the horizon starts at `soc_initial` and its end is free.
[Cyclic storage](pypsa_cyclic_storage.md) closes that loop, kept separate so
it can fail on its own.

The battery sits at `south`, beside the expensive oil generator. It displaces
oil entirely: every snapshot runs oil at zero. It drains to empty by the third
snapshot, which is what a free end of horizon buys.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

PyPSA linear optimal power flow with a storage unit carrying energy between snapshots. Non-cyclic — the horizon starts at the initial state of charge and ends free. Optimum 15253.178322993519, from PyPSA itself.

#### Sets

| Symbol | Meaning |
|---|---|
| $`\mathcal{T}`$ | index $`t`$ — `snapshot` — dispatch periods |
| $`\mathcal{B}`$ | index $`b`$ — `bus` with $`\mathrm{gen\_bus}: \mathcal{G} \to \mathcal{B},\ \mathrm{link\_from}: \mathcal{L} \to \mathcal{B},\ \mathrm{link\_to}: \mathcal{L} \to \mathcal{B},\ \mathrm{storage\_bus}: \mathcal{S} \to \mathcal{B}`$ — network nodes |
| $`\mathcal{G}`$ | index $`g`$ — `generator` with $`\mathrm{gen\_bus}: \mathcal{G} \to \mathcal{B}`$ — generating units, each sitting on one bus |
| $`\mathcal{L}`$ | index $`l`$ — `link` with $`\mathrm{link\_from}: \mathcal{L} \to \mathcal{B},\ \mathrm{link\_to}: \mathcal{L} \to \mathcal{B}`$ — controllable connections, each joining two buses |
| $`\mathcal{S}`$ | index $`s`$ — `storage` with $`\mathrm{storage\_bus}: \mathcal{S} \to \mathcal{B}`$ — storage units, each sitting on one bus |

#### Parameters

| Symbol | Meaning |
|---|---|
| $`\mathrm{p}^{\mathrm{nom}}`$ | `p_nom` over $`\mathcal{G}`$ — installed capacity of a generator |
| $`\mathrm{marginal\_cost}`$ | `marginal_cost` over $`\mathcal{G}`$ — cost of one unit of output |
| $`\mathrm{ramp\_limit\_up}`$ | `ramp_limit_up` over $`\mathcal{G}`$ — share of capacity output may rise by from one snapshot to the next |
| $`\mathrm{ramp\_limit\_down}`$ | `ramp_limit_down` over $`\mathcal{G}`$ — share of capacity output may fall by from one snapshot to the next |
| $`\mathrm{rating}`$ | `rating` over $`\mathcal{L}`$ — most a link may carry towards its `link_to` bus |
| $`\mathrm{neg\_rating}`$ | `neg_rating` over $`\mathcal{L}`$ — most a link may carry the other way, negative by convention |
| $`\mathrm{storage\_p\_nom}`$ | `storage_p_nom` over $`\mathcal{S}`$ — most a storage unit may charge or discharge in one snapshot |
| $`\mathrm{soc}^{\mathrm{max}}`$ | `soc_max` over $`\mathcal{S}`$ — how much energy a storage unit holds when full — PyPSA's capacity times its hours of storage, carried as a column because a bound takes a name or a number, never arithmetic (issue 31) |
| $`\mathrm{soc}^{\mathrm{initial}}`$ | `soc_initial` over $`\mathcal{S}`$ — energy in the store before the first snapshot |
| $`\mathrm{efficiency\_store}`$ | `efficiency_store` over $`\mathcal{S}`$ — share of charging energy that reaches the store |
| $`\mathrm{efficiency\_dispatch}`$ | `efficiency_dispatch` over $`\mathcal{S}`$ — share of stored energy that reaches the bus on the way out |
| $`\mathrm{standing\_loss}`$ | `standing_loss` over $`\mathcal{S}`$ — share of the carried-over level lost between snapshots |
| $`\mathrm{load}`$ | `load` over $`\mathcal{T} \times \mathcal{B}`$ — demand at each bus in each snapshot |

#### Variables

| Symbol | Meaning |
|---|---|
| $`p`$ | `p` over $`\mathcal{T} \times \mathcal{G}`$ — output of a generator in a snapshot |
| $`f`$ | `f` over $`\mathcal{T} \times \mathcal{L}`$ — flow on a link, signed towards its `link_to` bus |
| $`p^{\mathrm{dispatch}}`$ | `p_dispatch` over $`\mathcal{T} \times \mathcal{S}`$ — power a storage unit puts onto its bus — PyPSA splits a unit's power into two non-negative variables rather than one signed one, so the two efficiencies can differ |
| $`p^{\mathrm{store}}`$ | `p_store` over $`\mathcal{T} \times \mathcal{S}`$ — power a storage unit takes off its bus |
| $`\mathit{soc}`$ | `soc` over $`\mathcal{T} \times \mathcal{S}`$ — energy in the store at the end of a snapshot |

#### Notation

| Symbol | Meaning |
|---|---|
| $`\mathrm{pos}(t)`$ | `position()` — the position of $`t`$, counted from $`0`$; $`t`$ is the label |

#### Variable domains

**`p`**

```math
0 \le p_{t,g} \le \mathrm{p}^{\mathrm{nom}}_{g} \qquad \forall\, t \in \mathcal{T},\ g \in \mathcal{G}
```

**`f`**

```math
\mathrm{neg\_rating}_{l} \le f_{t,l} \le \mathrm{rating}_{l} \qquad \forall\, t \in \mathcal{T},\ l \in \mathcal{L}
```

**`p_dispatch`**

```math
0 \le p^{\mathrm{dispatch}}_{t,s} \le \mathrm{storage\_p\_nom}_{s} \qquad \forall\, t \in \mathcal{T},\ s \in \mathcal{S}
```

**`p_store`**

```math
0 \le p^{\mathrm{store}}_{t,s} \le \mathrm{storage\_p\_nom}_{s} \qquad \forall\, t \in \mathcal{T},\ s \in \mathcal{S}
```

**`soc`**

```math
0 \le \mathit{soc}_{t,s} \le \mathrm{soc}^{\mathrm{max}}_{s} \qquad \forall\, t \in \mathcal{T},\ s \in \mathcal{S}
```

#### Objective

```math
\min \sum_{t \in \mathcal{T},\ g \in \mathcal{G}} p_{t,g} \cdot \mathrm{marginal\_cost}_{g}
```

#### Subject to

**`nodal_balance`**

```math
\sum_{g \in \mathcal{G} \,:\, \mathrm{gen\_bus}(g) = b} p_{t,g} + \sum_{l \in \mathcal{L} \,:\, \mathrm{link\_to}(l) = b} f_{t,l} - \left( \sum_{l \in \mathcal{L} \,:\, \mathrm{link\_from}(l) = b} f_{t,l} \right) + \sum_{s \in \mathcal{S} \,:\, \mathrm{storage\_bus}(s) = b} p^{\mathrm{dispatch}}_{t,s} - \left( \sum_{s \in \mathcal{S} \,:\, \mathrm{storage\_bus}(s) = b} p^{\mathrm{store}}_{t,s} \right) = \mathrm{load}_{t,b} \qquad \forall\, t \in \mathcal{T},\ b \in \mathcal{B}
```

**`ramp_up`**

```math
p_{t,g} - p_{t - 1,g} \le \mathrm{ramp\_limit\_up}_{g} \cdot \mathrm{p}^{\mathrm{nom}}_{g} \qquad \forall\, t \in \mathcal{T},\ g \in \mathcal{G}
```

**`ramp_down`**

```math
p_{t - 1,g} - p_{t,g} \le \mathrm{ramp\_limit\_down}_{g} \cdot \mathrm{p}^{\mathrm{nom}}_{g} \qquad \forall\, t \in \mathcal{T},\ g \in \mathcal{G}
```

**`energy_balance_initial`**

```math
\mathit{soc}_{t,s} = \mathrm{soc}^{\mathrm{initial}}_{s} + p^{\mathrm{store}}_{t,s} \cdot \mathrm{efficiency\_store}_{s} - \frac{p^{\mathrm{dispatch}}_{t,s}}{\mathrm{efficiency\_dispatch}_{s}} \qquad \forall\, t \in \mathcal{T},\ s \in \mathcal{S} \,:\, \mathrm{pos}(t) = 0
```

**`energy_balance`**

```math
\mathit{soc}_{t,s} = \mathit{soc}_{t - 1,s} \cdot \left( 1 - \mathrm{standing\_loss}_{s} \right) + p^{\mathrm{store}}_{t,s} \cdot \mathrm{efficiency\_store}_{s} - \frac{p^{\mathrm{dispatch}}_{t,s}}{\mathrm{efficiency\_dispatch}_{s}} \qquad \forall\, t \in \mathcal{T},\ s \in \mathcal{S}
```

</details>
<!-- math:end -->

The tabs start from [the instance's tables](../howto/data.md) — one frame per parameter.

=== "specsolve"

    ```yaml
    description: >-
      PyPSA linear optimal power flow with a storage unit carrying energy between
      snapshots. Non-cyclic — the horizon starts at the initial
      state of charge and ends free. Optimum 15253.178322993519, from PyPSA itself.

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
      link:
        description: controllable connections, each joining two buses
        dtype: str
      storage:
        description: storage units, each sitting on one bus
        dtype: str

    relations:
      gen_bus:
        description: the bus a generator sits on
        key: generator
        values: bus
      link_from:
        description: the bus a link leaves
        key: link
        values: bus
      link_to:
        description: the bus a link arrives at
        key: link
        values: bus
      storage_bus:
        description: the bus a storage unit sits on
        key: storage
        values: bus

    parameters:
      p_nom:
        description: installed capacity of a generator
        dims: [generator]
      marginal_cost:
        description: cost of one unit of output
        dims: [generator]
      ramp_limit_up:
        description: share of capacity output may rise by from one snapshot to the next
        dims: [generator]
      ramp_limit_down:
        description: share of capacity output may fall by from one snapshot to the next
        dims: [generator]
      rating:
        description: most a link may carry towards its `link_to` bus
        dims: [link]
      neg_rating:
        description: most a link may carry the other way, negative by convention
        dims: [link]
      storage_p_nom:
        description: most a storage unit may charge or discharge in one snapshot
        dims: [storage]
      soc_max:
        description: >-
          how much energy a storage unit holds when full — PyPSA's capacity times its
          hours of storage, carried as a column because a bound takes a name or a
          number, never arithmetic (issue 31)
        dims: [storage]
      soc_initial:
        description: energy in the store before the first snapshot
        dims: [storage]
      efficiency_store:
        description: share of charging energy that reaches the store
        dims: [storage]
      efficiency_dispatch:
        description: share of stored energy that reaches the bus on the way out
        dims: [storage]
      standing_loss:
        description: share of the carried-over level lost between snapshots
        dims: [storage]
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
      f:
        description: flow on a link, signed towards its `link_to` bus
        dims: [snapshot, link]
        bounds:
          lower: neg_rating
          upper: rating
      p_dispatch:
        description: >-
          power a storage unit puts onto its bus — PyPSA splits a unit's power into
          two non-negative variables rather than one signed one, so the two
          efficiencies can differ
        dims: [snapshot, storage]
        bounds:
          lower: 0
          upper: storage_p_nom
      p_store:
        description: power a storage unit takes off its bus
        dims: [snapshot, storage]
        bounds:
          lower: 0
          upper: storage_p_nom
      soc:
        description: energy in the store at the end of a snapshot
        dims: [snapshot, storage]
        bounds:
          lower: 0
          upper: soc_max

    constraints:
      nodal_balance:
        description: >-
          what is generated at a bus, plus what arrives over the links and out of
          the stores, meets the load there
        dims: [snapshot, bus]
        expression: >-
          sum(p, over=generator, by=gen_bus[bus])
          + sum(f, over=link, by=link_to[bus])
          - sum(f, over=link, by=link_from[bus])
          + sum(p_dispatch, over=storage, by=storage_bus[bus])
          - sum(p_store, over=storage, by=storage_bus[bus])
          == load

      ramp_up:
        dims: [snapshot, generator]
        expression: p - shift(p, along=snapshot, offset=1) <= ramp_limit_up * p_nom

      ramp_down:
        dims: [snapshot, generator]
        expression: shift(p, along=snapshot, offset=1) - p <= ramp_limit_down * p_nom

      energy_balance_initial:
        description: >-
          the first snapshot's level is its own equation, because standing loss
          decays only what was carried over and PyPSA does not apply it to the
          initial state of charge
        dims: [snapshot, storage]
        where: "position(snapshot) == 0"
        expression: >-
          soc == soc_initial
          + p_store * efficiency_store
          - p_dispatch / efficiency_dispatch

      energy_balance:
        description: >-
          the level carried into a snapshot, decayed, plus what was stored and less
          what was taken — charging is derated on the way in and discharging on the
          way out, so the two efficiencies enter on opposite sides of the division
        dims: [snapshot, storage]
        expression: >-
          soc == shift(soc, along=snapshot, offset=1) * (1 - standing_loss)
          + p_store * efficiency_store
          - p_dispatch / efficiency_dispatch

    objective:
      sense: minimize
      description: total cost of generation; storage and transmission are free here
      expression: sum(p * marginal_cost)
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with sps.solve('examples/ports/pypsa_storage.yaml', sources) as solution:
        solution.objective  # 15253.178322993519
        solution.dual('nodal_balance')
    ```

=== "PyPSA"

    The model-building half of `examples/ports/references/pypsa/pypsa_storage.py`:

    ```python
    def build(tables: dict[str, pd.DataFrame]) -> pypsa.Network:
        """The port's tables as a PyPSA network, column for column.

        ``tables`` is the same mapping the specsolve call attaches as ``sources``.

        ``max_hours`` is the ratio PyPSA stores; the port carries the product it
        implies (``soc_max``), because a bound there takes a name, not arithmetic.
        """
        n = pypsa.Network()
        n.set_snapshots(tables['snapshot']['snapshot'])
        n.add('Bus', tables['bus']['bus'])

        generators: pd.DataFrame = tables['generator'].set_index('generator')
        links: pd.DataFrame = tables['link'].set_index('link')
        storages: pd.DataFrame = tables['storage'].set_index('storage')

        n.add(
            'Generator',
            generators.index,
            bus=tables['gen_bus'].set_index('generator')['bus'],
            p_nom=tables['p_nom'].set_index('generator')['value'],
            marginal_cost=tables['marginal_cost'].set_index('generator')['value'],
            ramp_limit_up=tables['ramp_limit_up'].set_index('generator')['value'],
            ramp_limit_down=tables['ramp_limit_down'].set_index('generator')['value'],
        )
        n.add(
            'Link',
            links.index,
            bus0=tables['link_from'].set_index('link')['bus'],
            bus1=tables['link_to'].set_index('link')['bus'],
            p_nom=tables['rating'].set_index('link')['value'],
            p_min_pu=-1.0,
            efficiency=1.0,
        )
        p_nom: pd.Series = tables['storage_p_nom'].set_index('storage')['value']
        n.add(
            'StorageUnit',
            storages.index,
            bus=tables['storage_bus'].set_index('storage')['bus'],
            p_nom=p_nom,
            max_hours=tables['soc_max'].set_index('storage')['value'] / p_nom,
            state_of_charge_initial=tables['soc_initial'].set_index('storage')['value'],
            efficiency_store=tables['efficiency_store'].set_index('storage')['value'],
            efficiency_dispatch=tables['efficiency_dispatch'].set_index('storage')['value'],
            standing_loss=tables['standing_loss'].set_index('storage')['value'],
            cyclic_state_of_charge=False,
        )

        load: pd.DataFrame = tables['load'].pivot(index='snapshot', columns='bus', values='value')
        for bus in tables['bus']['bus']:
            n.add('Load', f'load_{bus}', bus=bus, p_set=load[bus])
        return n
    ```

**Two efficiencies, on opposite sides of the division.** PyPSA splits a storage
unit's power into two non-negative variables rather than one signed one.
Charging and discharging then derate differently: `p_store * efficiency_store`
on the way in, `p_dispatch / efficiency_dispatch` on the way out.

**`standing_loss` decays only what was carried over.** PyPSA does not apply it
to `soc_initial`, so the first snapshot is its own equation rather than a
carry-over with a seeded value. Applying the loss to the seed as well, a
one-token change, moves the objective to **15272.957445031367**. That is wrong
by 0.13%: too small to see on a plot, too large to be rounding.

## What it exercises

`shift` across a boundary condition, division of a variable by a parameter, and
a five-term `sum(by=)` nodal balance — generators, both ends of every link,
and both directions of storage, all projected onto `bus`.

It also asks for [#31](https://github.com/fluxopt/specsolve/issues/31) a third
time. `soc_max` is `p_nom × max_hours` in PyPSA, and a bound here takes a name
or a number, so the product ships as a column.
