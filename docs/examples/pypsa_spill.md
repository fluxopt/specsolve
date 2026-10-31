# PyPSA spillage — water a reservoir cannot hold

A hydro unit takes inflow it did not choose, and spills what neither turbine nor reservoir can absorb.

> **✔ Verified against pypsa 1.2.4 (its own linopy 0.10.0)** — objective **3200.0**, matched to `rtol=1e-09`.

`inflow` is energy that arrives whether or not the model wanted it. When the
reservoir is full and the turbine is at its limit, the energy balance closes
only if something lets the surplus go. That is `spill`.

Two storage units share the bus: `res` receives inflow, `bat` receives none. The
battery absorbs water that would otherwise be spilled, so neither unit is
decoration.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

PyPSA storage spillage: water a reservoir cannot hold leaves through a second sink. A hydro unit takes inflow it did not choose and spills what neither its turbine nor its reservoir can absorb; a battery beside it has no inflow, and so no spill variable at all. Optimum 3200.0, from PyPSA itself.

#### Sets

| Symbol | Meaning |
|---|---|
| $`\mathcal{T}`$ | index $`t`$ — `snapshot` — dispatch periods |
| $`\mathcal{B}`$ | index $`b`$ — `bus` with $`\mathrm{gen\_bus}: \mathcal{G} \to \mathcal{B},\ \mathrm{storage\_bus}: \mathcal{S} \to \mathcal{B}`$ — network nodes |
| $`\mathcal{G}`$ | index $`g`$ — `generator` with $`\mathrm{gen\_bus}: \mathcal{G} \to \mathcal{B}`$ — generating units, each sitting on one bus |
| $`\mathcal{S}`$ | index $`s`$ — `storage` with $`\mathrm{storage\_bus}: \mathcal{S} \to \mathcal{B}`$ — storage units, each sitting on one bus |

#### Parameters

| Symbol | Meaning |
|---|---|
| $`\mathrm{p}^{\mathrm{nom}}`$ | `p_nom` over $`\mathcal{G}`$ — installed capacity of a generator |
| $`\mathrm{marginal\_cost}`$ | `marginal_cost` over $`\mathcal{G}`$ — cost of one unit of output |
| $`\mathrm{storage\_p\_nom}`$ | `storage_p_nom` over $`\mathcal{S}`$ — most a storage unit may charge or discharge in one snapshot |
| $`\mathrm{soc}^{\mathrm{max}}`$ | `soc_max` over $`\mathcal{S}`$ — how much energy a storage unit holds when full |
| $`\mathrm{soc}^{\mathrm{initial}}`$ | `soc_initial` over $`\mathcal{S}`$ — energy in the store before the first snapshot |
| $`\mathrm{inflow}`$ | `inflow` over $`\mathcal{T} \times \mathcal{S}`$, `neutral` where the data has no row — energy arriving at a storage unit whether or not it was wanted — zero for a unit that receives none |
| $`\mathrm{load}`$ | `load` over $`\mathcal{T} \times \mathcal{B}`$ — demand at each bus in each snapshot |

#### Variables

| Symbol | Meaning |
|---|---|
| $`p`$ | `p` over $`\mathcal{T} \times \mathcal{G}`$ — output of a generator in a snapshot |
| $`p^{\mathrm{dispatch}}`$ | `p_dispatch` over $`\mathcal{T} \times \mathcal{S}`$ — power a storage unit puts onto its bus |
| $`p^{\mathrm{store}}`$ | `p_store` over $`\mathcal{T} \times \mathcal{S}`$ — power a storage unit takes off its bus |
| $`\mathit{soc}`$ | `soc` over $`\mathcal{T} \times \mathcal{S}`$ — energy in the store at the end of a snapshot |
| $`\mathit{spill}`$ | `spill` over $`\mathcal{T} \times \mathcal{S}`$, `neutral` where the mask leaves it out — inflow let go rather than kept, and never more than that snapshot's arrival. A unit that receives no inflow has none to let go, which is a spill of zero rather than a quantity with no value — so the energy balance keeps its row there. |

#### Notation

| Symbol | Meaning |
|---|---|
| $`\mathrm{pos}(t)`$ | `position()` — the position of $`t`$, counted from $`0`$; $`t`$ is the label |

#### Variable domains

**`p`**

```math
0 \le p_{t,g} \le \mathrm{p}^{\mathrm{nom}}_{g} \qquad \forall\, t \in \mathcal{T},\ g \in \mathcal{G}
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

**`spill`**

```math
0 \le \mathit{spill}_{t,s} \le \mathrm{inflow}_{t,s} \qquad \forall\, t \in \mathcal{T},\ s \in \mathcal{S} \,:\, \mathrm{inflow}_{t,s} \neq 0
```

#### Objective

```math
\min \sum_{t \in \mathcal{T},\ g \in \mathcal{G}} p_{t,g} \cdot \mathrm{marginal\_cost}_{g}
```

#### Subject to

**`nodal_balance`**

```math
\sum_{g \in \mathcal{G} \,:\, \mathrm{gen\_bus}(g) = b} p_{t,g} + \sum_{s \in \mathcal{S} \,:\, \mathrm{storage\_bus}(s) = b} p^{\mathrm{dispatch}}_{t,s} - \left( \sum_{s \in \mathcal{S} \,:\, \mathrm{storage\_bus}(s) = b} p^{\mathrm{store}}_{t,s} \right) = \mathrm{load}_{t,b} \qquad \forall\, t \in \mathcal{T},\ b \in \mathcal{B}
```

**`energy_balance_initial`**

```math
\mathit{soc}_{t,s} = \mathrm{soc}^{\mathrm{initial}}_{s} + p^{\mathrm{store}}_{t,s} - p^{\mathrm{dispatch}}_{t,s} + \mathrm{inflow}_{t,s} - \mathit{spill}_{t,s} \qquad \forall\, t \in \mathcal{T},\ s \in \mathcal{S} \,:\, \mathrm{pos}(t) = 0
```

**`energy_balance`**

```math
\mathit{soc}_{t,s} = \mathit{soc}_{t - 1,s} + p^{\mathrm{store}}_{t,s} - p^{\mathrm{dispatch}}_{t,s} + \mathrm{inflow}_{t,s} - \mathit{spill}_{t,s} \qquad \forall\, t \in \mathcal{T},\ s \in \mathcal{S}
```

</details>
<!-- math:end -->

The tabs start from [the instance's tables](../howto/data.md) — one frame per parameter.

=== "specsolve"

    ```yaml
    description: >-
      PyPSA storage spillage: water a reservoir cannot hold leaves through a second
      sink. A hydro unit takes inflow it did not choose and spills what neither its
      turbine nor its reservoir can absorb; a battery beside it has no inflow, and
      so no spill variable at all. Optimum 3200.0, from PyPSA itself.

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
      storage:
        description: storage units, each sitting on one bus
        dtype: str

    relations:
      gen_bus:
        description: the bus a generator sits on
        key: generator
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
      storage_p_nom:
        description: most a storage unit may charge or discharge in one snapshot
        dims: [storage]
      soc_max:
        description: how much energy a storage unit holds when full
        dims: [storage]
      soc_initial:
        description: energy in the store before the first snapshot
        dims: [storage]
      inflow:
        description: >-
          energy arriving at a storage unit whether or not it was wanted — zero for
          a unit that receives none
        dims: [snapshot, storage]
        missing: neutral
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
      p_dispatch:
        description: power a storage unit puts onto its bus
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
      spill:
        description: >-
          inflow let go rather than kept, and never more than that snapshot's
          arrival. A unit that receives no inflow has none to let go, which is a
          spill of zero rather than a quantity with no value — so the energy
          balance keeps its row there.
        dims: [snapshot, storage]
        where: "inflow != 0"
        missing: neutral
        bounds:
          lower: 0
          upper: inflow

    constraints:
      nodal_balance:
        description: >-
          what is generated at a bus, plus what comes out of the stores less what
          goes into them, meets the load there
        dims: [snapshot, bus]
        expression: >-
          sum(p, over=generator, by=gen_bus[bus])
          + sum(p_dispatch, over=storage, by=storage_bus[bus])
          - sum(p_store, over=storage, by=storage_bus[bus])
          == load

      energy_balance_initial:
        description: the first snapshot's level is carried from the initial state of charge
        dims: [snapshot, storage]
        where: "position(snapshot) == 0"
        expression: soc == soc_initial + p_store - p_dispatch + inflow - spill

      energy_balance:
        description: >-
          the level carried into a snapshot, plus what was stored and what arrived,
          less what was taken and what was let go
        dims: [snapshot, storage]
        expression: >-
          soc == shift(soc, along=snapshot, offset=1)
          + p_store - p_dispatch + inflow - spill

    objective:
      sense: minimize
      description: total cost of generation; storage and spillage are free here
      expression: sum(p * marginal_cost)
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with sps.solve('examples/ports/pypsa_spill.yaml', sources) as solution:
        solution.objective  # 3200.0
        solution.dual('nodal_balance')
    ```

=== "PyPSA"

    The model-building half of `examples/ports/references/pypsa/pypsa_spill.py`:

    ```python
    def build(tables: dict[str, pd.DataFrame]) -> pypsa.Network:
        """The port's tables as a PyPSA network, column for column.

        ``tables`` is the same mapping the specsolve call attaches as ``sources``.

        ``inflow`` is a time-varying attribute, so it arrives pivoted to snapshots
        by names. PyPSA declares the spill variable only for units whose inflow is
        positive somewhere; the port declares it for every unit and bounds it above
        by the inflow, which pins it to zero for the battery. Same model, and the
        port's spelling is the one that keeps the energy balance a single block.
        """
        n = pypsa.Network()
        n.set_snapshots(tables['snapshot']['snapshot'])
        n.add('Bus', tables['bus']['bus'])

        generators: pd.DataFrame = tables['generator'].set_index('generator')
        storages: pd.DataFrame = tables['storage'].set_index('storage')
        n.add(
            'Generator',
            generators.index,
            bus=tables['gen_bus'].set_index('generator')['bus'],
            p_nom=tables['p_nom'].set_index('generator')['value'],
            marginal_cost=tables['marginal_cost'].set_index('generator')['value'],
        )

        p_nom: pd.Series = tables['storage_p_nom'].set_index('storage')['value']
        n.add(
            'StorageUnit',
            storages.index,
            bus=tables['storage_bus'].set_index('storage')['bus'],
            p_nom=p_nom,
            max_hours=tables['soc_max'].set_index('storage')['value'] / p_nom,
            state_of_charge_initial=tables['soc_initial'].set_index('storage')['value'],
            inflow=tables['inflow']
            .pivot(index='snapshot', columns='storage', values='value')
            .reindex(columns=storages.index)
            .fillna(0.0),
            cyclic_state_of_charge=False,
        )

        load: pd.DataFrame = tables['load'].pivot(index='snapshot', columns='bus', values='value')
        for bus in tables['bus']['bus']:
            n.add('Load', f'load_{bus}', bus=bus, p_set=load[bus])
        return n
    ```

**Spilling is forced, not chosen.** Snapshot 1 opens with a full 60 MWh
reservoir and 50 MWh more arriving against a 30 MW turbine. At least 20 MWh has
to go. Total gas burn is then pinned at `20 + spill` = 40 MWh, which is the
entire objective. A port that dropped the spill variable would be **infeasible**
rather than wrong.

**The battery has no spill decision at all.** PyPSA declares the spill variable
only for units whose inflow is positive somewhere. The port matches that with
`where: "inflow != 0"`, so `spill` has six coordinates rather than twelve.

The mask is safe only because of the line beside it. A constraint mentioning a
masked variable loses its **row**, not only the term. On its own the mask would
delete the battery's whole energy balance. Its stored energy would come from
nowhere, and the model would report **0.0** instead of 3200. `missing: neutral`
says the missing coordinates hold a spill of zero rather than a quantity with no
value, so the row stands without the term.

**The mask compares a value rather than naming the parameter.** `where: inflow`
would be a **no-op** here: a `where:` on a bare parameter reads *defined and
finite*, and the padded `0.0` is both. Nor can the zeros be dropped from the
table. `inflow` is a term on the energy balance's constant side, and a sparse
parameter there is refused at load. A missing row read as zero would be a bound
rather than an absence. The sparsity that matters is in the *value*, and
`!= 0` is how the model asks for it.

## What it exercises

`missing: neutral` on a masked variable, the declaration that keeps a row whose
term has gone, against an energy balance carrying two independent sinks. Also
the asymmetry underneath it: a masked **variable** takes its row, while a sparse
**parameter** on a constant side is refused. The two halves of this model's
sparsity are spelled in two different ways.
