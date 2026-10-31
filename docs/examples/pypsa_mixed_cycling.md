# pypsa_mixed_cycling

PyPSA's `cyclic_state_of_charge` is a per-unit flag, so one network runs both
regimes at once: a unit that must end each horizon where it began, beside one
handed a level it may simply spend.

## The problem

[Storage units](pypsa_storage.md) seeds every storage from
`state_of_charge_initial`; [cyclic storage](pypsa_cyclic_storage.md) closes
every storage on itself. A real PyPSA model has both in the same
`StorageUnit` frame, because the flag is a column:

```python
n.add('StorageUnit', ..., cyclic_state_of_charge=[True, False])
```

The two regimes are **one rule and a different predecessor**. A cyclic unit's
first snapshot carries from its last. A seeded unit's first snapshot carries
from a level in the data. Here that is three blocks under complementary masks,
and the masks are the flag itself:

| Block | Where | What carries into the first snapshot |
|---|---|---|
| `energy_balance_cyclic` | `cyclic` | the unit's own last snapshot (`edge='wrap'`) |
| `energy_balance_carry` | `NOT cyclic` | nothing — the vacated position is absent, so no row is built there |
| `energy_balance_seed` | `NOT cyclic AND position(snapshot) == 0` | `soc_initial` |

`NOT` is a real complement over a boolean column, so every unit falls in exactly
one regime. A unit whose flag row is missing reads as not cyclic.
`energy_balance_carry` builds no row at a seeded unit's first snapshot,
and that is reported: `diagnostics().omissions` gives 1. `energy_balance_seed`
writes the row instead.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

PyPSA's `cyclic_state_of_charge` is a column of the StorageUnit frame, so one network runs both regimes at once: a unit that must end each horizon where it began, beside one handed a level it may simply spend. The two are one rule under complementary masks, and the seeded unit's boundary row is what a cyclic one does not have. Optimum 4800.0, from PyPSA itself.

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
| $`\mathrm{cyclic}`$ | `cyclic` over $`\mathcal{S}`$ — whether a unit closes its own horizon. PyPSA's flag, and the one column that decides which of the two balance rules a unit obeys |
| $`\mathrm{p}^{\mathrm{nom}}`$ | `p_nom` over $`\mathcal{G}`$ — installed capacity of a generator |
| $`\mathrm{marginal\_cost}`$ | `marginal_cost` over $`\mathcal{G}`$ — cost of one unit of generation |
| $`\mathrm{storage\_p\_nom}`$ | `storage_p_nom` over $`\mathcal{S}`$ — how fast a storage unit may charge or discharge |
| $`\mathrm{soc}^{\mathrm{max}}`$ | `soc_max` over $`\mathcal{S}`$ — most energy a storage unit may hold |
| $`\mathrm{soc}^{\mathrm{initial}}`$ | `soc_initial` over $`\mathcal{S}`$ — the level a seeded unit begins with. A cyclic unit has one in the data and never reads it — its own last snapshot is its opening level |
| $`\mathrm{load}`$ | `load` over $`\mathcal{T} \times \mathcal{B}`$ — demand to be met at a bus |

#### Variables

| Symbol | Meaning |
|---|---|
| $`p`$ | `p` over $`\mathcal{T} \times \mathcal{G}`$ — output of a generator in a snapshot |
| $`p^{\mathrm{dispatch}}`$ | `p_dispatch` over $`\mathcal{T} \times \mathcal{S}`$ — power a storage unit puts onto its bus |
| $`p^{\mathrm{store}}`$ | `p_store` over $`\mathcal{T} \times \mathcal{S}`$ — power a storage unit takes off its bus |
| $`\mathit{soc}`$ | `soc` over $`\mathcal{T} \times \mathcal{S}`$ — energy in the store at the end of a snapshot |

#### Notation

| Symbol | Meaning |
|---|---|
| $`t \ominus k`$ | `edge='wrap'` — $`t-k`$, cyclic |
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

#### Objective

```math
\min \sum_{t \in \mathcal{T},\ g \in \mathcal{G}} p_{t,g} \cdot \mathrm{marginal\_cost}_{g}
```

#### Subject to

**`nodal_balance`**

```math
\sum_{g \in \mathcal{G} \,:\, \mathrm{gen\_bus}(g) = b} p_{t,g} + \sum_{s \in \mathcal{S} \,:\, \mathrm{storage\_bus}(s) = b} p^{\mathrm{dispatch}}_{t,s} - \left( \sum_{s \in \mathcal{S} \,:\, \mathrm{storage\_bus}(s) = b} p^{\mathrm{store}}_{t,s} \right) = \mathrm{load}_{t,b} \qquad \forall\, t \in \mathcal{T},\ b \in \mathcal{B}
```

**`energy_balance_cyclic`**

```math
\mathit{soc}_{t,s} = \mathit{soc}_{t \ominus 1,s} + p^{\mathrm{store}}_{t,s} - p^{\mathrm{dispatch}}_{t,s} \qquad \forall\, t \in \mathcal{T},\ s \in \mathcal{S} \,:\, \mathrm{cyclic}_{s}
```

**`energy_balance_carry`**

```math
\mathit{soc}_{t,s} = \mathit{soc}_{t - 1,s} + p^{\mathrm{store}}_{t,s} - p^{\mathrm{dispatch}}_{t,s} \qquad \forall\, t \in \mathcal{T},\ s \in \mathcal{S} \,:\, \neg \mathrm{cyclic}_{s}
```

**`energy_balance_seed`**

```math
\mathit{soc}_{t,s} = \mathrm{soc}^{\mathrm{initial}}_{s} + p^{\mathrm{store}}_{t,s} - p^{\mathrm{dispatch}}_{t,s} \qquad \forall\, t \in \mathcal{T},\ s \in \mathcal{S} \,:\, \neg \mathrm{cyclic}_{s} \wedge \mathrm{pos}(t) = 0
```

</details>
<!-- math:end -->

The tabs start from [the instance's tables](../howto/data.md) — one frame per parameter.

=== "specsolve"

    ```yaml
    description: >-
      PyPSA's `cyclic_state_of_charge` is a column of the StorageUnit frame, so one
      network runs both regimes at once: a unit that must end each horizon where it
      began, beside one handed a level it may simply spend. The two are one rule
      under complementary masks, and the seeded unit's boundary row is what a cyclic
      one does not have. Optimum 4800.0, from PyPSA itself.

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
      cyclic:
        description: >-
          whether a unit closes its own horizon. PyPSA's flag, and the one column
          that decides which of the two balance rules a unit obeys
        dims: [storage]
        dtype: bool
      p_nom:
        description: installed capacity of a generator
        dims: [generator]
      marginal_cost:
        description: cost of one unit of generation
        dims: [generator]
      storage_p_nom:
        description: how fast a storage unit may charge or discharge
        dims: [storage]
      soc_max:
        description: most energy a storage unit may hold
        dims: [storage]
      soc_initial:
        description: >-
          the level a seeded unit begins with. A cyclic unit has one in the data and
          never reads it — its own last snapshot is its opening level
        dims: [storage]
      load:
        description: demand to be met at a bus
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

    constraints:
      nodal_balance:
        description: what is generated at a bus, plus what the stores give back, meets the load there
        dims: [snapshot, bus]
        expression: >-
          sum(p, over=generator, by=gen_bus[bus])
          + sum(p_dispatch, over=storage, by=storage_bus[bus])
          - sum(p_store, over=storage, by=storage_bus[bus])
          == load

      energy_balance_cyclic:
        description: >-
          a cyclic unit's level wraps at the horizon, so its first snapshot inherits
          from its last and it ends every horizon where it began
        dims: [snapshot, storage]
        where: "cyclic"
        expression: soc == shift(soc, along=snapshot, offset=1, edge='wrap') + p_store - p_dispatch

      energy_balance_carry:
        description: >-
          a seeded unit carries from the snapshot before, and has no predecessor at
          the first — the vacated position is absent, so that row is not built here
        dims: [snapshot, storage]
        where: "NOT cyclic"
        expression: soc == shift(soc, along=snapshot, offset=1) + p_store - p_dispatch

      energy_balance_seed:
        description: >-
          and the row the vacated position left is written here instead, from the
          level the unit was handed
        dims: [snapshot, storage]
        where: "NOT cyclic AND position(snapshot) == 0"
        expression: soc == soc_initial + p_store - p_dispatch

    objective:
      sense: minimize
      description: total cost of generation; storage is free to operate here
      expression: sum(p * marginal_cost)
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with sps.solve('examples/ports/pypsa_mixed_cycling.yaml', sources) as solution:
        solution.objective  # 4800.0
        solution.dual('nodal_balance')
    ```

=== "PyPSA"

    The model-building half of `examples/ports/references/pypsa/pypsa_mixed_cycling.py`:

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
            cyclic_state_of_charge=tables['cyclic'].set_index('storage')['value'],
            state_of_charge_initial=tables['soc_initial'].set_index('storage')['value'],
        )

        load: pd.DataFrame = tables['load'].pivot(index='snapshot', columns='bus', values='value')
        for bus in tables['bus']['bus']:
            n.add('Load', f'load_{bus}', bus=bus, p_set=load[bus])
        return n
    ```

## Both flags bind

Neither regime is decoration on this instance. Flipping either flag changes the
answer, in opposite directions:

| Instance | Objective |
|---|---|
| as shipped — `ring` cyclic, `seasonal` seeded | **4800.0** |
| `ring` flipped to seeded | 4400.0 |
| `seasonal` flipped to cyclic | 7200.0 |

`ring` carries a `soc_initial` of 20 that it never reads. A cyclic unit ignores
the level in the data, and flipping its flag hands it 400 of free energy.
`seasonal`'s 30 is worth 2400, because a cyclic unit must give back everything
it spends.

## What the answer looks like

```text
snapshot  price   ring   seasonal
0         20      15     30      ← seasonal opens at its given level; ring opens where it will close
1         80      0      15      ← both discharge into the peak
2         20      10     15
3         20      0      0       ← ring is back where it started; seasonal need not be
```

The price is the dual of `nodal_balance`: 20 where the base plant is marginal
and 80 at the snapshot where the peaker runs. That spread is what makes moving
energy worth anything.

## What this model is for

*A data column choosing between two boundary regimes.* Neither storage model
has that shape: the two differ by one deleted `where`, and each is uniform. It
needs no language feature: three blocks, complementary masks, and the dropped
row visible in `omissions`.
