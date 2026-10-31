# PyPSA fixed by data — a schedule, and a capacity already decided

A row of data that is present pins its variable; a row that is absent leaves it free.

> **✔ Verified against pypsa 1.2.4 (its own linopy 0.10.0)** — objective **49900.0**, matched to `rtol=1e-09`.

`p_set` pins a dispatch, `p_nom_set` pins a capacity, and each equality exists
only where the table has a row: the must-run unit, the pre-committed schedule,
the capacity somebody already signed for.

`chp` is the dearest generator in the fleet and still runs in the two snapshots
it is scheduled in. A fixing that only agreed with the merit order would never
show up in the objective.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

PyPSA dispatch and capacity fixed by data: a row that is present pins its variable, a row that is absent leaves it free. A must-run unit runs in the two snapshots it is pinned in even though it is the dearest in the fleet. Optimum 49900.0, from PyPSA itself.

#### Sets

| Symbol | Meaning |
|---|---|
| $`\mathcal{T}`$ | index $`t`$ — `snapshot` — dispatch periods |
| $`\mathcal{B}`$ | index $`b`$ — `bus` with $`\mathrm{gen\_bus}: \mathcal{G} \to \mathcal{B}`$ — network nodes |
| $`\mathcal{G}`$ | index $`g`$ — `generator` with $`\mathrm{gen\_bus}: \mathcal{G} \to \mathcal{B}`$ — generating units, each sitting on one bus |

#### Parameters

| Symbol | Meaning |
|---|---|
| $`\mathrm{p}^{\mathrm{nom,max}}`$ | `p_nom_max` over $`\mathcal{G}`$ — most capacity that may stand at a generator once built |
| $`\mathrm{capital\_cost}`$ | `capital_cost` over $`\mathcal{G}`$ — cost of holding one unit of capacity over the horizon |
| $`\mathrm{marginal\_cost}`$ | `marginal_cost` over $`\mathcal{G}`$ — cost of one unit of output |
| $`\mathrm{p}^{\mathrm{nom,set}}`$ | `p_nom_set` over $`\mathcal{G}`$, `absent` where the data has no row — the capacity a generator is to hold, for the generators whose capacity is already decided |
| $`\mathrm{p}^{\mathrm{set}}`$ | `p_set` over $`\mathcal{T} \times \mathcal{G}`$, `absent` where the data has no row — the output a generator is to deliver, for the snapshots in which it is scheduled rather than chosen |
| $`\mathrm{load}`$ | `load` over $`\mathcal{T} \times \mathcal{B}`$ — demand at each bus in each snapshot |

#### Variables

| Symbol | Meaning |
|---|---|
| $`p`$ | `p` over $`\mathcal{T} \times \mathcal{G}`$ — output of a generator in a snapshot |
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
\min \sum_{t \in \mathcal{T},\ g \in \mathcal{G}} p_{t,g} \cdot \mathrm{marginal\_cost}_{g} + \sum_{g \in \mathcal{G}} p^{\mathrm{nom}}_{g} \cdot \mathrm{capital\_cost}_{g}
```

#### Subject to

**`within_capacity`**

```math
p_{t,g} \le p^{\mathrm{nom}}_{g} \qquad \forall\, t \in \mathcal{T},\ g \in \mathcal{G}
```

**`capacity_fixed`**

```math
p^{\mathrm{nom}}_{g} = \mathrm{p}^{\mathrm{nom,set}}_{g} \qquad \forall\, g \in \mathcal{G} \,:\, \mathrm{p}^{\mathrm{nom,set}}_{g} \text{ is defined}
```

**`dispatch_fixed`**

```math
p_{t,g} = \mathrm{p}^{\mathrm{set}}_{t,g} \qquad \forall\, t \in \mathcal{T},\ g \in \mathcal{G} \,:\, \mathrm{p}^{\mathrm{set}}_{t,g} \text{ is defined}
```

**`nodal_balance`**

```math
\sum_{g \in \mathcal{G} \,:\, \mathrm{gen\_bus}(g) = b} p_{t,g} = \mathrm{load}_{t,b} \qquad \forall\, t \in \mathcal{T},\ b \in \mathcal{B}
```

</details>
<!-- math:end -->

The tabs start from [the instance's tables](../howto/data.md) — one frame per parameter.

=== "specsolve"

    ```yaml
    description: >-
      PyPSA dispatch and capacity fixed by data: a row that is present pins its
      variable, a row that is absent leaves it free. A must-run unit runs in the two
      snapshots it is pinned in even though it is the dearest in the fleet.
      Optimum 49900.0, from PyPSA itself.

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
      p_nom_max:
        description: most capacity that may stand at a generator once built
        dims: [generator]
      capital_cost:
        description: cost of holding one unit of capacity over the horizon
        dims: [generator]
      marginal_cost:
        description: cost of one unit of output
        dims: [generator]
      p_nom_set:
        description: >-
          the capacity a generator is to hold, for the generators whose capacity is
          already decided
        dims: [generator]
        missing: absent
      p_set:
        description: >-
          the output a generator is to deliver, for the snapshots in which it is
          scheduled rather than chosen
        dims: [snapshot, generator]
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
      p_nom:
        description: capacity built at a generator
        dims: [generator]
        bounds:
          lower: 0
          upper: p_nom_max

    constraints:
      within_capacity:
        description: a generator produces no more than the capacity built for it
        dims: [snapshot, generator]
        expression: p <= p_nom

      capacity_fixed:
        description: >-
          a generator whose capacity is already decided holds exactly that — the
          row exists only where the table has one
        dims: [generator]
        where: p_nom_set
        expression: p_nom == p_nom_set

      dispatch_fixed:
        description: >-
          a generator scheduled for a snapshot delivers exactly its schedule there,
          and is free everywhere else
        dims: [snapshot, generator]
        where: p_set
        expression: p == p_set

      nodal_balance:
        description: what is generated at a bus meets the load there
        dims: [snapshot, bus]
        expression: sum(p, over=generator, by=gen_bus[bus]) == load

    objective:
      sense: minimize
      description: what the fleet costs to run, plus what its capacity costs to build
      expression: sum(p * marginal_cost) + sum(p_nom * capital_cost)
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with sps.solve('examples/ports/pypsa_fixed.yaml', sources) as solution:
        solution.objective  # 49900.0
        solution.dual('nodal_balance')
    ```

=== "PyPSA"

    The model-building half of `examples/ports/references/pypsa/pypsa_fixed.py`:

    ```python
    def build(tables: dict[str, pd.DataFrame]) -> pypsa.Network:
        """The port's tables as a PyPSA network, column for column.

        ``tables`` is the same mapping the specsolve call attaches as ``sources``.

        Both fixings arrive as short frames and are widened to the shape PyPSA
        reads, with ``NaN`` where the port simply has no row: ``p_nom_set``
        reindexed onto the generator index, ``p_set`` pivoted to snapshots by names.
        NaN is PyPSA's own spelling for *not fixed here* — it is what its mask
        tests — so the widening is the translation, not a defaulting choice.

        Every generator is extendable, because ``p_nom_set`` fixes the capacity
        *variable*: a non-extendable component has none for the equality to bind.
        """
        n = pypsa.Network()
        n.set_snapshots(tables['snapshot']['snapshot'])
        n.add('Bus', tables['bus']['bus'])

        generators: pd.DataFrame = tables['generator'].set_index('generator')
        p_set = (
            tables['p_set']
            .pivot(index='snapshot', columns='generator', values='value')
            .reindex(index=n.snapshots, columns=generators.index)
        )
        n.add(
            'Generator',
            generators.index,
            bus=tables['gen_bus'].set_index('generator')['bus'],
            p_nom_extendable=True,
            p_nom_max=tables['p_nom_max'].set_index('generator')['value'],
            p_nom_set=tables['p_nom_set'].set_index('generator')['value'].reindex(generators.index, fill_value=np.nan),
            capital_cost=tables['capital_cost'].set_index('generator')['value'],
            marginal_cost=tables['marginal_cost'].set_index('generator')['value'],
            p_set=p_set,
        )

        load: pd.DataFrame = tables['load'].pivot(index='snapshot', columns='bus', values='value')
        for bus in tables['bus']['bus']:
            n.add('Load', f'load_{bus}', bus=bus, p_set=load[bus])
        return n
    ```

**Two partial tables, at different ranks.** `p_set` is sparse over *(snapshot,
generator)*, two rows and both `chp`, and `p_nom_set` over *(generator)*
alone. The mask is the entire feature, so a model that pinned everything would
prove nothing. Both constraints carry `where:` naming the parameter, the
language's spelling for *the rows this table has*. PyPSA spells the same thing
as `NaN` in a widened frame and tests `~isnull()`.

**The capacity fixing needs a capacity variable.** Every generator here is
extendable, including the two whose capacity is pinned. A non-extendable
component has no variable for the equality to bind, so `p_nom_set` would be
silently ignored. The port declares `p_nom` over every generator and lets the
data decide which are still a decision.

## What it exercises

`where:` on an equality, at two ranks, against a partial table on the *constant*
side. The language refuses to guess there: a missing row read as zero would
pin the variable to zero rather than leave it free.

## A note on the instance

`gas` carries a `p_nom_max` of 200 it never approaches. A cap of 60, exactly
the capacity it builds, makes the problem dual-degenerate. With the build
limit and the capacity limit both active at one snapshot, `80` and `90` are
both optimal nodal prices. The two implementations pick differently while the
objective agrees. A limit off the optimum keeps the dual unique, which a
recorded dual vector needs.
