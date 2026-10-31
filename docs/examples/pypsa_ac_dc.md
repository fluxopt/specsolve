# PyPSA LOPF — two coordinates on one dimension

A meshed AC–DC network under a CO₂ budget. **PyPSA's own `ac-dc-meshed` example.**

> **✔ Verified against pypsa 1.2.4 (its own linopy 0.10.0)** — objective **18441021.477729216**, matched to `rtol=1e-09`, nodal prices included.

Every model above puts a generator on a bus and stops there. Here a generator
also burns a **carrier**, and both maps do work. The nodal balance groups
generation through `bus`; the CO₂ budget reads an emission rate back down
through `carrier`. Two coordinates on one dimension, landing on two different
targets.

Nine buses, six generators, seven passive lines and four controllable links
across three sub-networks: large enough that the two kinds of branch matter.
Capacity is a decision everywhere, so the model prices what it builds as well
as what it runs.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

PyPSA linear optimal power flow on a meshed AC-DC network whose generators sit on a bus and burn a carrier, with capacity to build and a CO2 budget priced through the second map: the nodal balance groups through the bus coordinate, and the budget reads an emissions rate back down through the carrier. PyPSA's own ac-dc-meshed example. Optimum 18441021.477729216, from PyPSA itself.

#### Sets

| Symbol | Meaning |
|---|---|
| $`\mathcal{T}`$ | index $`t`$ — `snapshot` — dispatch periods |
| $`\mathcal{B}`$ | index $`b`$ — `bus` with $`\mathrm{gen\_bus}: \mathcal{E} \to \mathcal{B},\ \mathrm{line\_from}: \mathcal{L} \to \mathcal{B},\ \mathrm{line\_to}: \mathcal{L} \to \mathcal{B},\ \mathrm{link\_from}: \mathcal{I} \to \mathcal{B},\ \mathrm{link\_to}: \mathcal{I} \to \mathcal{B}`$ — network nodes |
| $`\mathcal{C}`$ | index $`c`$ — `carrier` with $`\mathrm{gen\_carrier}: \mathcal{E} \to \mathcal{C}`$ — what a generator burns, and what its emissions are a property of |
| $`\mathcal{E}`$ | index $`e`$ — `generator` with $`\mathrm{gen\_bus}: \mathcal{E} \to \mathcal{B},\ \mathrm{gen\_carrier}: \mathcal{E} \to \mathcal{C}`$ — generating units, each sitting on a bus and burning a carrier — two coordinates on one dimension, landing on two different axes |
| $`\mathcal{L}`$ | index $`l`$ — `line` with $`\mathrm{line\_from}: \mathcal{L} \to \mathcal{B},\ \mathrm{line\_to}: \mathcal{L} \to \mathcal{B}`$ — passive AC lines, each joining two buses |
| $`\mathcal{I}`$ | index $`i`$ — `link` with $`\mathrm{link\_from}: \mathcal{I} \to \mathcal{B},\ \mathrm{link\_to}: \mathcal{I} \to \mathcal{B}`$ — controllable connections, each joining two buses |
| $`\mathcal{Y}`$ | index $`y`$ — `cycle` — one independent loop per meshed sub-network |

#### Parameters

| Symbol | Meaning |
|---|---|
| $`\mathrm{load}`$ | `load` over $`\mathcal{T} \times \mathcal{B}`$ — demand at each bus in each snapshot |
| $`\mathrm{p}^{\mathrm{max,pu}}`$ | `p_max_pu` over $`\mathcal{T} \times \mathcal{E}`$ — share of built capacity a generator can produce in a snapshot |
| $`\mathrm{p}^{\mathrm{nom,min}}`$ | `p_nom_min` over $`\mathcal{E}`$ — capacity a generator already has, and cannot fall below |
| $`\mathrm{marginal\_cost}`$ | `marginal_cost` over $`\mathcal{E}`$ — cost of one unit of output |
| $`\mathrm{gen\_capital\_cost}`$ | `gen_capital_cost` over $`\mathcal{E}`$ — annualised cost of a unit of generator capacity |
| $`\mathrm{efficiency}`$ | `efficiency` over $`\mathcal{E}`$ — share of the carrier's energy a generator turns into output |
| $`\mathrm{co2\_per\_mwh}`$ | `co2_per_mwh` over $`\mathcal{C}`$ — emissions per unit of carrier burned, a property of the carrier |
| $`\mathrm{line\_capital\_cost}`$ | `line_capital_cost` over $`\mathcal{L}`$ — annualised cost of a unit of line capacity |
| $`\mathrm{link\_capital\_cost}`$ | `link_capital_cost` over $`\mathcal{I}`$ — annualised cost of a unit of link capacity |
| $`\mathrm{link\_p\_max\_pu}`$ | `link_p_max_pu` over $`\mathcal{I}`$ — share of its capacity a link may carry forwards |
| $`\mathrm{link\_p\_min\_pu}`$ | `link_p_min_pu` over $`\mathcal{I}`$ — share of its capacity a link may carry backwards, negative by convention |
| $`\mathrm{cycle\_incidence}`$ | `cycle_incidence` over $`\mathcal{Y} \times \mathcal{L}`$, `neutral` where the data has no row — the cycle basis, as a sparse table of impedance times direction. A line may belong to several cycles, so this cannot be a coordinate. |
| $`\mathrm{co2\_limit}`$ | `co2_limit` (scalar) — emissions the whole horizon is allowed |

#### Variables

| Symbol | Meaning |
|---|---|
| $`p`$ | `p` over $`\mathcal{T} \times \mathcal{E}`$ — output of a generator in a snapshot |
| $`p^{\mathrm{nom}}`$ | `p_nom` over $`\mathcal{E}`$ — generator capacity to hold, built on top of what already stands |
| $`f`$ | `f` over $`\mathcal{T} \times \mathcal{L}`$ — flow on a line, signed towards its `line_to` bus — not chosen, but whatever the voltage law leaves |
| $`s^{\mathrm{nom}}`$ | `s_nom` over $`\mathcal{L}`$ — line capacity to build |
| $`g`$ | `g` over $`\mathcal{T} \times \mathcal{I}`$ — flow on a link, signed towards the bus it delivers at — chosen, which is what makes it a link and not a line |
| $`\mathit{link\_p\_nom}`$ | `link_p_nom` over $`\mathcal{I}`$ — link capacity to build |

#### Variable domains

**`p`**

```math
p_{t,e} \ge 0 \qquad \forall\, t \in \mathcal{T},\ e \in \mathcal{E}
```

**`p_nom`**

```math
p^{\mathrm{nom}}_{e} \ge \mathrm{p}^{\mathrm{nom,min}}_{e} \qquad \forall\, e \in \mathcal{E}
```

**`f`**

```math
f_{t,l} \in \mathbb{R} \qquad \forall\, t \in \mathcal{T},\ l \in \mathcal{L}
```

**`s_nom`**

```math
s^{\mathrm{nom}}_{l} \ge 0 \qquad \forall\, l \in \mathcal{L}
```

**`g`**

```math
g_{t,i} \in \mathbb{R} \qquad \forall\, t \in \mathcal{T},\ i \in \mathcal{I}
```

**`link_p_nom`**

```math
\mathit{link\_p\_nom}_{i} \ge 0 \qquad \forall\, i \in \mathcal{I}
```

#### Objective

```math
\min \sum_{t \in \mathcal{T},\ e \in \mathcal{E}} p_{t,e} \cdot \mathrm{marginal\_cost}_{e} + \sum_{e \in \mathcal{E}} p^{\mathrm{nom}}_{e} \cdot \mathrm{gen\_capital\_cost}_{e} + \sum_{l \in \mathcal{L}} s^{\mathrm{nom}}_{l} \cdot \mathrm{line\_capital\_cost}_{l} + \sum_{i \in \mathcal{I}} \mathit{link\_p\_nom}_{i} \cdot \mathrm{link\_capital\_cost}_{i}
```

#### Subject to

**`within_capacity`**

```math
p_{t,e} \le p^{\mathrm{nom}}_{e} \cdot \mathrm{p}^{\mathrm{max,pu}}_{t,e} \qquad \forall\, t \in \mathcal{T},\ e \in \mathcal{E}
```

**`line_upper`**

```math
f_{t,l} \le s^{\mathrm{nom}}_{l} \qquad \forall\, t \in \mathcal{T},\ l \in \mathcal{L}
```

**`line_lower`**

```math
f_{t,l} \ge -s^{\mathrm{nom}}_{l} \qquad \forall\, t \in \mathcal{T},\ l \in \mathcal{L}
```

**`link_upper`**

```math
g_{t,i} \le \mathit{link\_p\_nom}_{i} \cdot \mathrm{link\_p\_max\_pu}_{i} \qquad \forall\, t \in \mathcal{T},\ i \in \mathcal{I}
```

**`link_lower`**

```math
g_{t,i} \ge \mathit{link\_p\_nom}_{i} \cdot \mathrm{link\_p\_min\_pu}_{i} \qquad \forall\, t \in \mathcal{T},\ i \in \mathcal{I}
```

**`nodal_balance`**

```math
\sum_{e \in \mathcal{E} \,:\, \mathrm{gen\_bus}(e) = b} p_{t,e} + \sum_{l \in \mathcal{L} \,:\, \mathrm{line\_to}(l) = b} f_{t,l} - \left( \sum_{l \in \mathcal{L} \,:\, \mathrm{line\_from}(l) = b} f_{t,l} \right) + \sum_{i \in \mathcal{I} \,:\, \mathrm{link\_to}(i) = b} g_{t,i} - \left( \sum_{i \in \mathcal{I} \,:\, \mathrm{link\_from}(i) = b} g_{t,i} \right) = \mathrm{load}_{t,b} \qquad \forall\, t \in \mathcal{T},\ b \in \mathcal{B}
```

**`kirchhoff_voltage_law`**

```math
\sum_{l \in \mathcal{L}} f_{t,l} \cdot \mathrm{cycle\_incidence}_{y,l} = 0 \qquad \forall\, t \in \mathcal{T},\ y \in \mathcal{Y}
```

**`co2_budget`**

```math
\sum_{t \in \mathcal{T}} \sum_{e \in \mathcal{E}} \frac{p_{t,e} \cdot \mathrm{co2\_per\_mwh}_{\mathrm{gen\_carrier}(e)}}{\mathrm{efficiency}_{e}} \le \mathrm{co2\_limit}
```

</details>
<!-- math:end -->

The tabs start from [the instance's tables](../howto/data.md) — one frame per parameter.

=== "specsolve"

    ```yaml
    description: >-
      PyPSA linear optimal power flow on a meshed AC-DC network whose
      generators sit on a bus and burn a carrier, with capacity to build and a CO2
      budget priced through the second map: the nodal balance groups through the
      bus coordinate, and the budget reads an emissions rate back down through the
      carrier. PyPSA's own ac-dc-meshed example.
      Optimum 18441021.477729216, from PyPSA itself.

    dimensions:
      snapshot:
        description: dispatch periods
        dtype: int
        ordered: true
      bus:
        description: network nodes
        dtype: str
      carrier:
        description: what a generator burns, and what its emissions are a property of
        dtype: str
      generator:
        description: >-
          generating units, each sitting on a bus and burning a carrier — two
          coordinates on one dimension, landing on two different axes
        dtype: str
      line:
        description: passive AC lines, each joining two buses
        dtype: str
      link:
        description: controllable connections, each joining two buses
        dtype: str
      cycle:
        description: one independent loop per meshed sub-network
        dtype: str

    relations:
      gen_bus:
        description: the bus a generator sits on
        key: generator
        values: bus
      gen_carrier:
        description: the carrier a generator burns
        key: generator
        values: carrier
      line_from:
        description: the bus a line leaves
        key: line
        values: bus
      line_to:
        description: the bus a line arrives at
        key: line
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
      load:
        description: demand at each bus in each snapshot
        dims: [snapshot, bus]
      p_max_pu:
        description: share of built capacity a generator can produce in a snapshot
        dims: [snapshot, generator]
      p_nom_min:
        description: capacity a generator already has, and cannot fall below
        dims: [generator]
      marginal_cost:
        description: cost of one unit of output
        dims: [generator]
      gen_capital_cost:
        description: annualised cost of a unit of generator capacity
        dims: [generator]
      efficiency:
        description: share of the carrier's energy a generator turns into output
        dims: [generator]
      co2_per_mwh:
        description: emissions per unit of carrier burned, a property of the carrier
        dims: [carrier]
      line_capital_cost:
        description: annualised cost of a unit of line capacity
        dims: [line]
      link_capital_cost:
        description: annualised cost of a unit of link capacity
        dims: [link]
      link_p_max_pu:
        description: share of its capacity a link may carry forwards
        dims: [link]
      link_p_min_pu:
        description: share of its capacity a link may carry backwards, negative by convention
        dims: [link]
      cycle_incidence:
        description: >-
          the cycle basis, as a sparse table of impedance times direction. A line
          may belong to several cycles, so this cannot be a coordinate.
        dims: [cycle, line]
        missing: neutral
      co2_limit:
        description: emissions the whole horizon is allowed
        dims: []

    variables:
      p:
        description: output of a generator in a snapshot
        dims: [snapshot, generator]
        bounds:
          lower: 0
      p_nom:
        description: generator capacity to hold, built on top of what already stands
        dims: [generator]
        bounds:
          lower: p_nom_min
      f:
        description: >-
          flow on a line, signed towards its `line_to` bus — not chosen, but whatever
          the voltage law leaves
        dims: [snapshot, line]
      s_nom:
        description: line capacity to build
        dims: [line]
        bounds:
          lower: 0
      g:
        description: >-
          flow on a link, signed towards the bus it delivers at — chosen, which is
          what makes it a link and not a line
        dims: [snapshot, link]
      link_p_nom:
        description: link capacity to build
        dims: [link]
        bounds:
          lower: 0

    constraints:
      within_capacity:
        description: a generator produces no more than the built capacity available to it
        dims: [snapshot, generator]
        expression: p <= p_nom * p_max_pu

      line_upper:
        dims: [snapshot, line]
        expression: f <= s_nom
      line_lower:
        dims: [snapshot, line]
        expression: f >= -s_nom
      link_upper:
        dims: [snapshot, link]
        expression: g <= link_p_nom * link_p_max_pu
      link_lower:
        dims: [snapshot, link]
        expression: g >= link_p_nom * link_p_min_pu

      nodal_balance:
        description: >-
          what is generated at a bus plus what arrives over the lines and links
          meets the load there
        dims: [snapshot, bus]
        expression: >-
          sum(p, over=generator, by=gen_bus[bus])
          + sum(f, over=line, by=line_to[bus]) - sum(f, over=line, by=line_from[bus])
          + sum(g, over=link, by=link_to[bus]) - sum(g, over=link, by=link_from[bus])
          == load

      kirchhoff_voltage_law:
        description: around each independent cycle the impedance-weighted flows sum to zero
        dims: [snapshot, cycle]
        expression: sum(f * cycle_incidence, over=line) == 0

      co2_budget:
        description: >-
          PyPSA's primary-energy constraint — a generator's emissions are its
          output divided by its efficiency, priced at its carrier's rate, and the
          horizon's total stays inside the budget
        dims: []
        expression: >-
          sum(sum(p * at(co2_per_mwh, by=gen_carrier[carrier]) / efficiency, over=generator), over=snapshot)
          <= co2_limit

    objective:
      sense: minimize
      description: what the fleet costs to run, plus what the generation and network capacity cost to build
      expression: >-
        sum(p * marginal_cost)
        + sum(p_nom * gen_capital_cost)
        + sum(s_nom * line_capital_cost)
        + sum(link_p_nom * link_capital_cost)
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with sps.solve('examples/ports/pypsa_ac_dc.yaml', sources) as solution:
        solution.objective  # 18441021.477729216
        solution.dual('nodal_balance')
    ```

=== "PyPSA"

    The model-building half of `examples/ports/references/pypsa/pypsa_ac_dc.py`:

    ```python
    def build(tables: dict[str, pd.DataFrame]) -> pypsa.Network:
        """The port's tables as a PyPSA network, column for column.

        ``tables`` is the same mapping the specsolve call attaches as ``sources``.

        The port carries its cycle basis as ``cycle_incidence`` because computing
        one is a graph algorithm and so data preparation; PyPSA derives its own
        from ``line_x`` / ``line_r``, which is why both are in the instance. The
        two must describe the same cycle space, and the objectives agreeing is
        what says they do.
        """
        n = pypsa.Network()
        n.set_snapshots(tables['snapshot']['snapshot'].tolist())

        carrier = _series(tables['bus_carrier'], 'bus')
        for bus, ct in carrier.items():
            n.add('Bus', bus, carrier=ct)

        co2 = _series(tables['co2_per_mwh'], 'carrier')
        for name, rate in co2.items():
            n.add('Carrier', name, co2_emissions=rate)

        generators = (
            tables['generator']
            .set_index('generator')
            .join([tables['gen_bus'].set_index('generator'), tables['gen_carrier'].set_index('generator')])
        )
        p_max_pu = _wide(tables['p_max_pu'], 'generator')
        for g, row in generators.iterrows():
            n.add(
                'Generator',
                g,
                bus=row['bus'],
                carrier=row['carrier'],
                p_nom_extendable=True,
                p_nom=_series(tables['gen_p_nom_existing'], 'generator')[g],
                p_nom_min=_series(tables['p_nom_min'], 'generator')[g],
                marginal_cost=_series(tables['marginal_cost'], 'generator')[g],
                capital_cost=_series(tables['gen_capital_cost'], 'generator')[g],
                efficiency=_series(tables['efficiency'], 'generator')[g],
                p_max_pu=p_max_pu[g],
            )

        bus0_of = tables['line_from'].set_index('line')['bus']
        bus1_of = tables['line_to'].set_index('line')['bus']
        for line in tables['line']['line']:
            bus0, bus1 = bus0_of[line], bus1_of[line]
            n.add(
                'Line',
                line,
                bus0=bus0,
                bus1=bus1,
                x=_series(tables['line_x'], 'line')[line],
                r=_series(tables['line_r'], 'line')[line],
                s_nom=_series(tables['line_s_nom_existing'], 'line')[line],
                s_nom_extendable=True,
                capital_cost=_series(tables['line_capital_cost'], 'line')[line],
            )

        bus0_of = tables['link_from'].set_index('link')['bus']
        bus1_of = tables['link_to'].set_index('link')['bus']
        for link in tables['link']['link']:
            bus0, bus1 = bus0_of[link], bus1_of[link]
            n.add(
                'Link',
                link,
                bus0=bus0,
                bus1=bus1,
                p_nom_extendable=True,
                p_nom=_series(tables['link_p_nom_existing'], 'link')[link],
                p_min_pu=_series(tables['link_p_min_pu'], 'link')[link],
                p_max_pu=_series(tables['link_p_max_pu'], 'link')[link],
                capital_cost=_series(tables['link_capital_cost'], 'link')[link],
            )

        load = _wide(tables['load'], 'bus')
        for bus in load.columns:
            if load[bus].any():
                n.add('Load', bus, bus=bus, p_set=load[bus])

        n.add(
            'GlobalConstraint',
            'co2_limit',
            type='primary_energy',
            carrier_attribute='co2_emissions',
            sense='<=',
            constant=float(tables['co2_limit']),
        )
        return n
    ```

**An emission rate is a property of the carrier, and `at()` is how a generator
reads it.** `co2_per_mwh` is dimensioned over `carrier` alone: six generators,
two rates. `at(co2_per_mwh, by=gen_carrier[carrier])` walks the map backwards to put
the right rate beside each generator's output. PyPSA does the same join
through `n.carriers`.

**The recorded optimum is the system cost, not `n.objective`.** Every component
here is extendable, so PyPSA credits the capital already standing in `p_nom`.
It reports the change against that starting point, a negative number on this
network. The port has no starting point to credit and states the cost
outright, so the recorded figure is `n.objective + n.objective_constant`.

## What it exercises

Two relations keyed over one dimension onto different dimensions, and `at()` reading a
parameter that lives only on the coarse end. Beside them, the shapes the
models above already established: `sum(by=)` on both ends of two different branch
dimensions, and a cycle basis as a sparse `(cycle, line)` parameter.

The cycle basis carries impedance rather than reactance alone. PyPSA applies
the voltage law with `x` inside an AC sub-network and `r` inside a DC one, and
this network has one meshed loop of each. Which value belongs in the row is
decided in data preparation, where
[the limits](https://mathspec.readthedocs.io/en/latest/about/limits/#what-counts-as-data-preparation)
put graph work; the language sees one incidence table either way.

No new construct was needed.
