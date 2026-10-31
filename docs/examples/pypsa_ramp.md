# PyPSA LOPF — ramp limits

[The transport model](pypsa_transport.md) plus a limit on how fast each generator may change output between snapshots.

> **✔ Verified against pypsa 1.2.4 (its own linopy 0.10.0)** — objective **18200**, matched to `rtol=1e-09`.

PyPSA states a ramp limit as a fraction of `p_nom` bounding the change between
consecutive snapshots. The row exists from the *second* snapshot on: there is
no dispatch before the first to ramp from.

**The limit binds.** The links of [the transport model](pypsa_transport.md)
run saturated, which fixes every generator's output exactly. A ramp limit on
that instance could only make it infeasible, never change the answer. This
model widens the ratings to 200 and lets merit order pick the dispatch. Gas
then moves 70 → 100 → 80 → 50, hits its ±30 limit twice, and calls oil on at
the two middle snapshots. Without the limits the same instance costs
**17000**; with them, 18200.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

PyPSA linear optimal power flow with a limit on how fast a generator may change output between snapshots. Optimum 18200.0, from PyPSA itself.

#### Sets

| Symbol | Meaning |
|---|---|
| $`\mathcal{T}`$ | index $`t`$ — `snapshot` — dispatch periods |
| $`\mathcal{B}`$ | index $`b`$ — `bus` with $`\mathrm{gen\_bus}: \mathcal{G} \to \mathcal{B},\ \mathrm{link\_from}: \mathcal{L} \to \mathcal{B},\ \mathrm{link\_to}: \mathcal{L} \to \mathcal{B}`$ — network nodes |
| $`\mathcal{G}`$ | index $`g`$ — `generator` with $`\mathrm{gen\_bus}: \mathcal{G} \to \mathcal{B}`$ — generating units, each sitting on one bus |
| $`\mathcal{L}`$ | index $`l`$ — `link` with $`\mathrm{link\_from}: \mathcal{L} \to \mathcal{B},\ \mathrm{link\_to}: \mathcal{L} \to \mathcal{B}`$ — controllable connections, each joining two buses |

#### Parameters

| Symbol | Meaning |
|---|---|
| $`\mathrm{p}^{\mathrm{nom}}`$ | `p_nom` over $`\mathcal{G}`$ — installed capacity of a generator |
| $`\mathrm{marginal\_cost}`$ | `marginal_cost` over $`\mathcal{G}`$ — cost of one unit of output |
| $`\mathrm{ramp\_limit\_up}`$ | `ramp_limit_up` over $`\mathcal{G}`$ — share of capacity output may rise by from one snapshot to the next |
| $`\mathrm{ramp\_limit\_down}`$ | `ramp_limit_down` over $`\mathcal{G}`$ — share of capacity output may fall by from one snapshot to the next |
| $`\mathrm{rating}`$ | `rating` over $`\mathcal{L}`$ — most a link may carry towards its `link_to` bus |
| $`\mathrm{neg\_rating}`$ | `neg_rating` over $`\mathcal{L}`$ — most a link may carry the other way, negative by convention |
| $`\mathrm{load}`$ | `load` over $`\mathcal{T} \times \mathcal{B}`$ — demand at each bus in each snapshot |

#### Variables

| Symbol | Meaning |
|---|---|
| $`p`$ | `p` over $`\mathcal{T} \times \mathcal{G}`$ — output of a generator in a snapshot |
| $`f`$ | `f` over $`\mathcal{T} \times \mathcal{L}`$ — flow on a link, signed towards its `link_to` bus |

#### Variable domains

**`p`**

```math
0 \le p_{t,g} \le \mathrm{p}^{\mathrm{nom}}_{g} \qquad \forall\, t \in \mathcal{T},\ g \in \mathcal{G}
```

**`f`**

```math
\mathrm{neg\_rating}_{l} \le f_{t,l} \le \mathrm{rating}_{l} \qquad \forall\, t \in \mathcal{T},\ l \in \mathcal{L}
```

#### Objective

```math
\min \sum_{t \in \mathcal{T},\ g \in \mathcal{G}} p_{t,g} \cdot \mathrm{marginal\_cost}_{g}
```

#### Subject to

**`nodal_balance`**

```math
\sum_{g \in \mathcal{G} \,:\, \mathrm{gen\_bus}(g) = b} p_{t,g} + \sum_{l \in \mathcal{L} \,:\, \mathrm{link\_to}(l) = b} f_{t,l} - \left( \sum_{l \in \mathcal{L} \,:\, \mathrm{link\_from}(l) = b} f_{t,l} \right) = \mathrm{load}_{t,b} \qquad \forall\, t \in \mathcal{T},\ b \in \mathcal{B}
```

**`ramp_up`**

```math
p_{t,g} - p_{t - 1,g} \le \mathrm{ramp\_limit\_up}_{g} \cdot \mathrm{p}^{\mathrm{nom}}_{g} \qquad \forall\, t \in \mathcal{T},\ g \in \mathcal{G}
```

**`ramp_down`**

```math
p_{t - 1,g} - p_{t,g} \le \mathrm{ramp\_limit\_down}_{g} \cdot \mathrm{p}^{\mathrm{nom}}_{g} \qquad \forall\, t \in \mathcal{T},\ g \in \mathcal{G}
```

</details>
<!-- math:end -->

The tabs start from [the instance's tables](../howto/data.md) — one frame per parameter.

=== "specsolve"

    ```yaml
    description: >-
      PyPSA linear optimal power flow with a limit on how fast a generator may
      change output between snapshots. Optimum 18200.0, from PyPSA
      itself.

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

    constraints:
      nodal_balance:
        description: what is generated at a bus plus what arrives over the links meets the load there
        dims: [snapshot, bus]
        expression: >-
          sum(p, over=generator, by=gen_bus[bus])
          + sum(f, over=link, by=link_to[bus])
          - sum(f, over=link, by=link_from[bus])
          == load

      ramp_up:
        dims: [snapshot, generator]
        expression: p - shift(p, along=snapshot, offset=1) <= ramp_limit_up * p_nom

      ramp_down:
        dims: [snapshot, generator]
        expression: shift(p, along=snapshot, offset=1) - p <= ramp_limit_down * p_nom

    objective:
      sense: minimize
      description: total cost of generation; moving power over a link is free here
      expression: sum(p * marginal_cost)
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with sps.solve('examples/ports/pypsa_ramp.yaml', sources) as solution:
        solution.objective  # 18200.0
        solution.dual('nodal_balance')
    ```

=== "PyPSA"

    The reference builds the same network with PyPSA's own objects. The delta from
    the transport model is two keyword arguments:

    The model-building half of `examples/ports/references/pypsa/pypsa_ramp.py`:

    ```python
    def build(tables: dict[str, pd.DataFrame]) -> pypsa.Network:
        """The port's tables as a PyPSA network, column for column.

        ``tables`` is the same mapping the specsolve call attaches as ``sources``.
        """
        n = pypsa.Network()
        n.set_snapshots(tables['snapshot']['snapshot'])
        n.add('Bus', tables['bus']['bus'])

        generators: pd.DataFrame = tables['generator'].set_index('generator')
        links: pd.DataFrame = tables['link'].set_index('link')

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

        load: pd.DataFrame = tables['load'].pivot(index='snapshot', columns='bus', values='value')
        for bus in tables['bus']['bus']:
            n.add('Load', f'load_{bus}', bus=bus, p_set=load[bus])
        return n
    ```

`shift` vacates the first snapshot, and a vacated position is *absent*, so the
row there drops on its own. That is the boundary PyPSA wants, and no `where`
states it. `edge='wrap'` would put the last snapshot onto the first and build
a different model, and a gate such as `snapshot > 0` would hardcode the index
origin. [Cyclic storage](pypsa_cyclic_storage.md) wants the wrap and asks for
it by name.

## What it exercises

`shift` along a dimension, and the acyclic boundary it carries. Also parameter
arithmetic on a constraint's right-hand side (`ramp_limit_up * p_nom`), kept
as arithmetic so the file states what PyPSA states.
