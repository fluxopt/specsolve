# PyPSA modular capacity — a technology bought in whole units

Capacity that comes in whole modules: an integer count decides it, not a continuous bound.

> **✔ Verified against pypsa 1.2.4 (its own linopy 0.10.0)** — objective **56700.0**, matched to `rtol=1e-09`.

The capacity variable survives, but `p_nom = n_mod × p_nom_mod` ties it to a
whole number of modules. A technology sold in 30 MW turbines cannot be built
23 MW at a time.

One bus and no network: a model that fails to match should implicate one
feature, and here that feature is the module count.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

PyPSA modular capacity expansion: a technology bought in whole units. The capacity variable survives, but an integer module count decides it, so the optimum may only land on a multiple of the module size. Optimum 56700.0, from PyPSA itself.

#### Sets

| Symbol | Meaning |
|---|---|
| $`\mathcal{T}`$ | index $`t`$ — `snapshot` — dispatch periods |
| $`\mathcal{B}`$ | index $`b`$ — `bus` with $`\mathrm{gen\_bus}: \mathcal{G} \to \mathcal{B}`$ — network nodes |
| $`\mathcal{G}`$ | index $`g`$ — `generator` with $`\mathrm{gen\_bus}: \mathcal{G} \to \mathcal{B}`$ — generating units, each sitting on one bus |

#### Parameters

| Symbol | Meaning |
|---|---|
| $`\mathrm{p}^{\mathrm{nom,mod}}`$ | `p_nom_mod` over $`\mathcal{G}`$ — capacity of one module — what a single unit of this technology adds |
| $`\mathrm{p}^{\mathrm{nom,max}}`$ | `p_nom_max` over $`\mathcal{G}`$ — most capacity that may stand at a generator once built |
| $`\mathrm{capital\_cost}`$ | `capital_cost` over $`\mathcal{G}`$ — cost of holding one unit of capacity over the horizon |
| $`\mathrm{marginal\_cost}`$ | `marginal_cost` over $`\mathcal{G}`$ — cost of one unit of output |
| $`\mathrm{load}`$ | `load` over $`\mathcal{T} \times \mathcal{B}`$ — demand at each bus in each snapshot |

#### Variables

| Symbol | Meaning |
|---|---|
| $`p`$ | `p` over $`\mathcal{T} \times \mathcal{G}`$ — output of a generator in a snapshot |
| $`p^{\mathrm{nom}}`$ | `p_nom` over $`\mathcal{G}`$ — capacity built at a generator |
| $`n^{\mathrm{mod}}`$ | `n_mod` over $`\mathcal{G}`$ — how many whole modules are built |

#### Variable domains

**`p`**

```math
p_{t,g} \ge 0 \qquad \forall\, t \in \mathcal{T},\ g \in \mathcal{G}
```

**`p_nom`**

```math
0 \le p^{\mathrm{nom}}_{g} \le \mathrm{p}^{\mathrm{nom,max}}_{g} \qquad \forall\, g \in \mathcal{G}
```

**`n_mod`**

```math
n^{\mathrm{mod}}_{g} \ge 0, n^{\mathrm{mod}}_{g} \in \mathbb{Z} \qquad \forall\, g \in \mathcal{G}
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

**`modularity`**

```math
p^{\mathrm{nom}}_{g} = n^{\mathrm{mod}}_{g} \cdot \mathrm{p}^{\mathrm{nom,mod}}_{g} \qquad \forall\, g \in \mathcal{G}
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
      PyPSA modular capacity expansion: a technology bought in whole units. The
      capacity variable survives, but an integer module count decides it, so the
      optimum may only land on a multiple of the module size. Optimum 56700.0, from
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

    relations:
      gen_bus:
        description: the bus a generator sits on
        key: generator
        values: bus

    parameters:
      p_nom_mod:
        description: capacity of one module — what a single unit of this technology adds
        dims: [generator]
      p_nom_max:
        description: most capacity that may stand at a generator once built
        dims: [generator]
      capital_cost:
        description: cost of holding one unit of capacity over the horizon
        dims: [generator]
      marginal_cost:
        description: cost of one unit of output
        dims: [generator]
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
      n_mod:
        description: how many whole modules are built
        dims: [generator]
        domain: integer
        bounds:
          lower: 0

    constraints:
      within_capacity:
        description: a generator produces no more than the capacity built for it
        dims: [snapshot, generator]
        expression: p <= p_nom

      modularity:
        description: >-
          capacity is the module count times the module size, which is what makes
          the count rather than the capacity the decision
        dims: [generator]
        expression: p_nom == n_mod * p_nom_mod

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
    with sps.solve('examples/ports/pypsa_modular.yaml', sources) as solution:
        solution.objective  # 56700.0
    ```

=== "PyPSA"

    The model-building half of `examples/ports/references/pypsa/pypsa_modular.py`:

    ```python
    def build(tables: dict[str, pd.DataFrame]) -> pypsa.Network:
        """The port's tables as a PyPSA network, column for column.

        ``tables`` is the same mapping the specsolve call attaches as ``sources``.

        ``p_nom_extendable`` and a positive ``p_nom_mod`` together are what make the
        capacity modular: PyPSA takes the module count only where a component is in
        both index sets.
        """
        n = pypsa.Network()
        n.set_snapshots(tables['snapshot']['snapshot'])
        n.add('Bus', tables['bus']['bus'])

        generators: pd.DataFrame = tables['generator'].set_index('generator')
        n.add(
            'Generator',
            generators.index,
            bus=tables['gen_bus'].set_index('generator')['bus'],
            p_nom_extendable=True,
            p_nom_mod=tables['p_nom_mod'].set_index('generator')['value'],
            p_nom_max=tables['p_nom_max'].set_index('generator')['value'],
            capital_cost=tables['capital_cost'].set_index('generator')['value'],
            marginal_cost=tables['marginal_cost'].set_index('generator')['value'],
        )

        load: pd.DataFrame = tables['load'].pivot(index='snapshot', columns='bus', values='value')
        for bus in tables['bus']['bus']:
            n.add('Load', f'load_{bus}', bus=bus, p_set=load[bus])
        return n
    ```

**The module count binds.** The three module sizes are 30, 25 and 20; peak
load is 143. Wind fills 120, four whole modules and its own ceiling, leaving
23. No single gas module covers 23, and one 25 MW module overshoots it. Drop
`p_nom_mod` and the same instance builds 108 of wind and 35 of oil, neither a
multiple of anything, for **54040.0** against the modular **56700.0**. A port
that ignored the integer constraint would report the cheaper number.

## What it exercises

`domain: integer` on a variable that is not a status. The module count has no
upper bound of its own; only the capacity ceiling above it holds it down.
Every other integrality in the corpus is a 0/1 decision. And `modularity`, an
equality between two decisions: a capacity variable decided by another
variable rather than by a bound.
