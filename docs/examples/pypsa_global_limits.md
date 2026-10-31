# PyPSA global limits — four bounds, one shape

Limits that hold over a whole set at once: an energy total, a capacity at one bus, and the built network measured twice.

> **✔ Verified against pypsa 1.2.4 (its own linopy 0.10.0)** — objective **127211.66666666666**, matched to `rtol=1e-09`.

Every other bound in this corpus belongs to a component. A *global* limit
belongs to a set PyPSA selects by an attribute: all generators burning gas, all
wind capacity at one bus, every extendable link. [AC-DC](pypsa_ac_dc.md) ports
one of them, the CO₂ cap, which groups nothing: a single bound over everything.

Four limits here, and each is the same sentence in the language: **a sum over a
group, against a bound only some rows carry**.

| PyPSA | what it selects | the port's row |
|---|---|---|
| `operational_limit` | generators burning a carrier | `sum(sum(p, over=generator, by=gen_placement[carrier]), over=snapshot) <= energy_cap` |
| per-`(bus, carrier)` capacity cap | that carrier's capacity at one bus | `sum(p_nom, over=generator, by=gen_placement[bus, carrier]) <= bus_capacity_cap` |
| `transmission_volume_expansion_limit` | extendable links, weighted by length | `sum(link_p_nom * link_length, over=link) <= volume_cap` |
| `transmission_expansion_cost_limit` | the same links, weighted by money | `sum(link_p_nom * link_capital_cost, over=link) <= expansion_cost_cap` |

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

PyPSA's global constraints: four limits over four different selected sets — the energy a carrier may deliver, the capacity a carrier may hold at one bus, and the built transmission measured twice, once as length and once as money. Each is one grouped sum against a bound only some rows carry. Optimum 127211.66666666666, from PyPSA itself.

#### Sets

| Symbol | Meaning |
|---|---|
| $`\mathcal{T}`$ | index $`t`$ — `snapshot` — dispatch periods |
| $`\mathcal{B}`$ | index $`b`$ — `bus` with $`\mathrm{gen\_placement}: \mathcal{E} \to \mathcal{B} \times \mathcal{C},\ \mathrm{link\_from}: \mathcal{L} \to \mathcal{B},\ \mathrm{link\_to}: \mathcal{L} \to \mathcal{B}`$ — network nodes |
| $`\mathcal{C}`$ | index $`c`$ — `carrier` with $`\mathrm{gen\_placement}: \mathcal{E} \to \mathcal{B} \times \mathcal{C}`$ — what a generator burns, and what a global limit selects on |
| $`\mathcal{E}`$ | index $`e`$ — `generator` with $`\mathrm{gen\_placement}: \mathcal{E} \to \mathcal{B} \times \mathcal{C}`$ — generating units, each sitting on a bus and burning a carrier |
| $`\mathcal{L}`$ | index $`l`$ — `link` with $`\mathrm{link\_from}: \mathcal{L} \to \mathcal{B},\ \mathrm{link\_to}: \mathcal{L} \to \mathcal{B}`$ — controllable connections, each joining two buses |

#### Parameters

| Symbol | Meaning |
|---|---|
| $`\mathrm{load}`$ | `load` over $`\mathcal{T} \times \mathcal{B}`$ — demand at each bus in each snapshot |
| $`\mathrm{p}^{\mathrm{max,pu}}`$ | `p_max_pu` over $`\mathcal{T} \times \mathcal{E}`$ — share of built capacity a generator can produce in a snapshot |
| $`\mathrm{marginal\_cost}`$ | `marginal_cost` over $`\mathcal{E}`$ — cost of one unit of output |
| $`\mathrm{gen\_capital\_cost}`$ | `gen_capital_cost` over $`\mathcal{E}`$ — annualised cost of a unit of generator capacity |
| $`\mathrm{link\_capital\_cost}`$ | `link_capital_cost` over $`\mathcal{L}`$ — annualised cost of a unit of link capacity |
| $`\mathrm{link\_length}`$ | `link_length` over $`\mathcal{L}`$ — how far a link reaches — what turns built capacity into a volume |
| $`\mathrm{energy\_cap}`$ | `energy_cap` over $`\mathcal{C}`$, `absent` where the data has no row — energy a carrier may deliver over the whole horizon, for the carriers that have such a limit |
| $`\mathrm{bus\_capacity\_cap}`$ | `bus_capacity_cap` over $`\mathcal{B} \times \mathcal{C}`$, `absent` where the data has no row — capacity of one carrier a bus may hold, for the pairs that cap it — PyPSA writes the carrier into a column name (`nom_max_wind`), so the pair is the limit's own key |
| $`\mathrm{volume\_cap}`$ | `volume_cap` (scalar) — capacity times length the whole network may build |
| $`\mathrm{expansion\_cost\_cap}`$ | `expansion_cost_cap` (scalar) — money the whole network may spend building links |

#### Variables

| Symbol | Meaning |
|---|---|
| $`p`$ | `p` over $`\mathcal{T} \times \mathcal{E}`$ — output of a generator in a snapshot |
| $`p^{\mathrm{nom}}`$ | `p_nom` over $`\mathcal{E}`$ — capacity built at a generator |
| $`g`$ | `g` over $`\mathcal{T} \times \mathcal{L}`$ — flow on a link, towards the bus it delivers at |
| $`\mathit{link\_p\_nom}`$ | `link_p_nom` over $`\mathcal{L}`$ — capacity built on a link |

#### Variable domains

**`p`**

```math
p_{t,e} \ge 0 \qquad \forall\, t \in \mathcal{T},\ e \in \mathcal{E}
```

**`p_nom`**

```math
p^{\mathrm{nom}}_{e} \ge 0 \qquad \forall\, e \in \mathcal{E}
```

**`g`**

```math
g_{t,l} \ge 0 \qquad \forall\, t \in \mathcal{T},\ l \in \mathcal{L}
```

**`link_p_nom`**

```math
\mathit{link\_p\_nom}_{l} \ge 0 \qquad \forall\, l \in \mathcal{L}
```

#### Objective

```math
\min \sum_{t \in \mathcal{T},\ e \in \mathcal{E}} p_{t,e} \cdot \mathrm{marginal\_cost}_{e} + \sum_{e \in \mathcal{E}} p^{\mathrm{nom}}_{e} \cdot \mathrm{gen\_capital\_cost}_{e} + \sum_{l \in \mathcal{L}} \mathit{link\_p\_nom}_{l} \cdot \mathrm{link\_capital\_cost}_{l}
```

#### Subject to

**`within_capacity`**

```math
p_{t,e} \le p^{\mathrm{nom}}_{e} \cdot \mathrm{p}^{\mathrm{max,pu}}_{t,e} \qquad \forall\, t \in \mathcal{T},\ e \in \mathcal{E}
```

**`within_link_capacity`**

```math
g_{t,l} \le \mathit{link\_p\_nom}_{l} \qquad \forall\, t \in \mathcal{T},\ l \in \mathcal{L}
```

**`nodal_balance`**

```math
\sum_{e \in \mathcal{E} \,:\, \mathrm{gen\_placement.bus}(e) = b} p_{t,e} + \sum_{l \in \mathcal{L} \,:\, \mathrm{link\_to}(l) = b} g_{t,l} - \left( \sum_{l \in \mathcal{L} \,:\, \mathrm{link\_from}(l) = b} g_{t,l} \right) = \mathrm{load}_{t,b} \qquad \forall\, t \in \mathcal{T},\ b \in \mathcal{B}
```

**`carrier_energy`**

```math
\sum_{t \in \mathcal{T}} \sum_{e \in \mathcal{E} \,:\, \mathrm{gen\_placement.carrier}(e) = c} p_{t,e} \le \mathrm{energy\_cap}_{c} \qquad \forall\, c \in \mathcal{C} \,:\, \mathrm{energy\_cap}_{c} \text{ is defined}
```

**`carrier_capacity_at_bus`**

```math
\sum_{e \in \mathcal{E} \,:\, \mathrm{gen\_placement.bus}(e) = b \wedge \mathrm{gen\_placement.carrier}(e) = c} p^{\mathrm{nom}}_{e} \le \mathrm{bus\_capacity\_cap}_{b,c} \qquad \forall\, b \in \mathcal{B},\ c \in \mathcal{C} \,:\, \mathrm{bus\_capacity\_cap}_{b,c} \text{ is defined}
```

**`transmission_volume`**

```math
\sum_{l \in \mathcal{L}} \mathit{link\_p\_nom}_{l} \cdot \mathrm{link\_length}_{l} \le \mathrm{volume\_cap}
```

**`transmission_cost`**

```math
\sum_{l \in \mathcal{L}} \mathit{link\_p\_nom}_{l} \cdot \mathrm{link\_capital\_cost}_{l} \le \mathrm{expansion\_cost\_cap}
```

</details>
<!-- math:end -->

The tabs start from [the instance's tables](../howto/data.md) — one frame per parameter.

=== "specsolve"

    ```yaml
    description: >-
      PyPSA's global constraints: four limits over four different selected sets —
      the energy a carrier may deliver, the capacity a carrier may hold at one bus,
      and the built transmission measured twice, once as length and once as money.
      Each is one grouped sum against a bound only some rows carry.
      Optimum 127211.66666666666, from PyPSA itself.

    dimensions:
      snapshot:
        description: dispatch periods
        dtype: int
        ordered: true
      bus:
        description: network nodes
        dtype: str
      carrier:
        description: what a generator burns, and what a global limit selects on
        dtype: str
      generator:
        description: generating units, each sitting on a bus and burning a carrier
        dtype: str
      link:
        description: controllable connections, each joining two buses
        dtype: str

    relations:
      gen_placement:
        description: the bus a generator sits on and the carrier it burns
        key: generator
        values: [bus, carrier]
      link_from:
        description: the bus a link leaves
        key: link
        values: bus
      link_to:
        description: the bus a link arrives at
        key: link
        values: bus

    parameters:
      load:
        description: demand at each bus in each snapshot
        dims: [snapshot, bus]
      p_max_pu:
        description: share of built capacity a generator can produce in a snapshot
        dims: [snapshot, generator]
      marginal_cost:
        description: cost of one unit of output
        dims: [generator]
      gen_capital_cost:
        description: annualised cost of a unit of generator capacity
        dims: [generator]
      link_capital_cost:
        description: annualised cost of a unit of link capacity
        dims: [link]
      link_length:
        description: how far a link reaches — what turns built capacity into a volume
        dims: [link]
      energy_cap:
        description: >-
          energy a carrier may deliver over the whole horizon, for the carriers that
          have such a limit
        dims: [carrier]
        missing: absent
      bus_capacity_cap:
        description: >-
          capacity of one carrier a bus may hold, for the pairs that cap it — PyPSA
          writes the carrier into a column name (`nom_max_wind`), so the pair is the
          limit's own key
        dims: [bus, carrier]
        missing: absent
      volume_cap:
        description: capacity times length the whole network may build
        dims: []
      expansion_cost_cap:
        description: money the whole network may spend building links
        dims: []

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
      g:
        description: flow on a link, towards the bus it delivers at
        dims: [snapshot, link]
        bounds:
          lower: 0
      link_p_nom:
        description: capacity built on a link
        dims: [link]
        bounds:
          lower: 0

    constraints:
      within_capacity:
        description: a generator produces no more than the built capacity available to it
        dims: [snapshot, generator]
        expression: p <= p_nom * p_max_pu

      within_link_capacity:
        dims: [snapshot, link]
        expression: g <= link_p_nom

      nodal_balance:
        description: what is generated at a bus plus what arrives over the links meets the load there
        dims: [snapshot, bus]
        expression: >-
          sum(p, over=generator, by=gen_placement[bus])
          + sum(g, over=link, by=link_to[bus]) - sum(g, over=link, by=link_from[bus])
          == load

      carrier_energy:
        description: >-
          a carrier with an energy limit delivers no more than it over the horizon —
          the generators are grouped onto the carrier they burn, which is the
          selection PyPSA makes by querying its own table
        dims: [carrier]
        where: energy_cap
        expression: sum(sum(p, over=generator, by=gen_placement[carrier]), over=snapshot) <= energy_cap

      carrier_capacity_at_bus:
        description: >-
          a bus that caps a carrier holds no more of it than that — one grouping
          lands the built capacity on the (bus, carrier) pair the limit is keyed by,
          so no selector column and no mask stand between the two
        dims: [bus, carrier]
        where: bus_capacity_cap
        expression: sum(p_nom, over=generator, by=gen_placement[bus, carrier]) <= bus_capacity_cap

      transmission_volume:
        description: >-
          capacity times length, summed over the links — a limit on how much network
          is built, in the unit a planner is granted
        dims: []
        expression: sum(link_p_nom * link_length, over=link) <= volume_cap

      transmission_cost:
        description: >-
          the same set weighted by money instead of distance, which is why the two
          limits are not each other: the long link is the cheap one
        dims: []
        expression: sum(link_p_nom * link_capital_cost, over=link) <= expansion_cost_cap

    objective:
      sense: minimize
      description: what the fleet costs to run, plus what the generation and link capacity cost to build
      expression: >-
        sum(p * marginal_cost)
        + sum(p_nom * gen_capital_cost)
        + sum(link_p_nom * link_capital_cost)
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with sps.solve('examples/ports/pypsa_global_limits.yaml', sources) as solution:
        solution.objective  # 127211.66666666666
        solution.dual('nodal_balance')
    ```

=== "PyPSA"

    The model-building half of `examples/ports/references/pypsa/pypsa_global_limits.py`:

    ```python
    def build(
        tables: dict[str, pd.DataFrame],
        limits: dict[str, dict[str, object]] | None = None,
        bus_capacity_cap: bool = True,
    ) -> pypsa.Network:
        """The port's tables as a PyPSA network, column for column.

        ``tables`` is the same mapping the specsolve call attaches as ``sources``.

        ``limits`` defaults to all three global-constraint rows and
        ``bus_capacity_cap`` to on; dropping one is how ``main`` measures what it is
        worth. Every generator and every link is extendable, because a limit on
        capacity has nothing to bind on a component whose capacity is data.
        """
        n = pypsa.Network()
        n.set_snapshots(tables['snapshot']['snapshot'])
        n.add('Bus', tables['bus']['bus'])
        n.add('Carrier', tables['carrier']['carrier'])
        n.add('Carrier', 'DC')

        generators: pd.DataFrame = (
            tables['generator'].set_index('generator').join(tables['gen_placement'].set_index('generator'))
        )
        p_max_pu: pd.DataFrame = tables['p_max_pu'].pivot(index='snapshot', columns='generator', values='value')
        n.add(
            'Generator',
            generators.index,
            bus=generators['bus'],
            carrier=generators['carrier'],
            p_nom_extendable=True,
            p_max_pu=p_max_pu[generators.index],
            marginal_cost=tables['marginal_cost'].set_index('generator')['value'],
            capital_cost=tables['gen_capital_cost'].set_index('generator')['value'],
        )

        links: pd.DataFrame = tables['link'].set_index('link')
        n.add(
            'Link',
            links.index,
            bus0=tables['link_from'].set_index('link')['bus'],
            bus1=tables['link_to'].set_index('link')['bus'],
            carrier='DC',
            p_nom_extendable=True,
            length=tables['link_length'].set_index('link')['value'],
            capital_cost=tables['link_capital_cost'].set_index('link')['value'],
        )

        load: pd.DataFrame = tables['load'].pivot(index='snapshot', columns='bus', values='value')
        for bus in tables['bus']['bus']:
            n.add('Load', f'load_{bus}', bus=bus, p_set=load[bus])

        for name, attributes in (LIMITS if limits is None else limits).items():
            n.add('GlobalConstraint', name, **attributes)

        if bus_capacity_cap:
            bus, column, cap = BUS_CAPACITY_CAP
            n.buses.loc[bus, column] = cap
        return n
    ```

**All four bind, and each is worth something.** Dropping one at a time, on the
same instance: the gas energy cap is worth **7411.67**, the volume limit
**541.67**, the cost limit **176.67**, and the cap on wind at `east` **283.89**.
A global limit that does not bind proves nothing about the language, so the
reference drops each in turn and prints the four numbers.

**The two link limits are not each other.** `north_south` is 100 km at 200/MW,
`east_south` 50 km at 400/MW, so length and money rank the two links in
*opposite* orders. Both limits are tight at the optimum, which fixes the build
exactly: 27.83 MW and 12.33 MW solve `100a + 50b = 3400` and
`200a + 400b = 10500` together.

**The selection is data, not a construct.** PyPSA selects by querying its own
tables, `carrier == "gas"`, and writes the per-bus cap into a column name,
`nom_max_wind`. That column name is a `(bus, carrier)` pair, and the port says
so. One grouping through both maps lands the built capacity on that pair, and
`bus_capacity_cap` is a table keyed by it.

## The fifth limit, which PyPSA does not build

`tech_capacity_expansion_limit`, a carrier's capacity across the whole network,
is missing from the table above. The language can say it, but in pypsa 1.2.4 a
single-period network cannot get one built at all:

```
no investment_period   emits no constraint at all
investment_period=0    raises ValueError: Investment period not in `n.investment_periods`
```

`global_constraints.py:48` groups the rows by
`["carrier_attribute", "sense", "investment_period"]`. Where no period is
given that key is `NaN`, pandas drops NaN keys, and the row leaves no
constraint behind. The next line reads
`period = None if isnan(period) else int(period)`, so NaN is expected to
arrive; [#966](https://github.com/fluxopt/specsolve/issues/966) tracks reporting
it upstream. [Multi-period investment](pypsa_multi_period.md) proves the limit
itself, where a period exists to name.

## What it exercises

A reduction over two dimensions at once (`sum(sum(p, over=generator, by=gen_placement[carrier]), over=snapshot)`),
one grouping landing on a pair of dimensions (`sum(p_nom, over=generator, by=gen_placement[bus, carrier])`),
and two scalar-bounded sums over one set with different weights. No construct
here is new, for five constraints PyPSA implements in five functions.
