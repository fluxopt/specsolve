# PyPSA LOPF — Kirchhoff's voltage law

Passive AC lines: flow is decided by physics, not chosen.

> **✔ Verified against pypsa 1.2.4 (its own linopy 0.10.0)** — objective **17000**, matched to `rtol=1e-09`, nodal prices and line flows included.

Every model above moves power over `Link` objects, whose flow is a decision
variable. A `Line` is passive: around every independent cycle of the network,
the reactance-weighted flows sum to zero.

It builds on [the transport model](pypsa_transport.md) rather than on
[cyclic storage](pypsa_cyclic_storage.md). Ramps, storage and a closed horizon
couple snapshots in time; a voltage law couples buses in space. Stacking the
two would leave a mismatch ambiguous about which caused it.

Three buses in a triangle, so the cycle space has exactly one dimension. The
flows come out fractional, 46.67 / 26.67 / −33.33 at the first snapshot,
because reactance forces the split rather than cost choosing it.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

PyPSA linear optimal power flow over passive AC lines under Kirchhoff's voltage law, rather than links whose flow is chosen. Optimum 17000.0, from PyPSA itself.

#### Sets

| Symbol | Meaning |
|---|---|
| $`\mathcal{T}`$ | index $`t`$ — `snapshot` — dispatch periods |
| $`\mathcal{B}`$ | index $`b`$ — `bus` with $`\mathrm{gen\_bus}: \mathcal{G} \to \mathcal{B},\ \mathrm{line\_from}: \mathcal{L} \to \mathcal{B},\ \mathrm{line\_to}: \mathcal{L} \to \mathcal{B}`$ — network nodes |
| $`\mathcal{G}`$ | index $`g`$ — `generator` with $`\mathrm{gen\_bus}: \mathcal{G} \to \mathcal{B}`$ — generating units, each sitting on one bus |
| $`\mathcal{L}`$ | index $`l`$ — `line` with $`\mathrm{line\_from}: \mathcal{L} \to \mathcal{B},\ \mathrm{line\_to}: \mathcal{L} \to \mathcal{B}`$ — passive AC lines, each joining two buses |
| $`\mathcal{C}`$ | index $`c`$ — `cycle` — one independent loop of the network |

#### Parameters

| Symbol | Meaning |
|---|---|
| $`\mathrm{p}^{\mathrm{nom}}`$ | `p_nom` over $`\mathcal{G}`$ — installed capacity of a generator |
| $`\mathrm{marginal\_cost}`$ | `marginal_cost` over $`\mathcal{G}`$ — cost of one unit of output |
| $`\mathrm{s}^{\mathrm{nom}}`$ | `s_nom` over $`\mathcal{L}`$ — most a line may carry towards its `line_to` bus |
| $`\mathrm{neg\_s\_nom}`$ | `neg_s_nom` over $`\mathcal{L}`$ — most a line may carry the other way, negative by convention |
| $`\mathrm{cycle\_incidence}`$ | `cycle_incidence` over $`\mathcal{C} \times \mathcal{L}`$, `neutral` where the data has no row — the cycle basis, as a sparse table of reactance times direction — a line may belong to several cycles, so this is a parameter over both dimensions rather than a coordinate, and rows are absent where a line is in no cycle |
| $`\mathrm{load}`$ | `load` over $`\mathcal{T} \times \mathcal{B}`$ — demand at each bus in each snapshot |

#### Variables

| Symbol | Meaning |
|---|---|
| $`p`$ | `p` over $`\mathcal{T} \times \mathcal{G}`$ — output of a generator in a snapshot |
| $`f`$ | `f` over $`\mathcal{T} \times \mathcal{L}`$ — flow on a line, signed towards its `line_to` bus — not chosen, but whatever the voltage law leaves |

#### Variable domains

**`p`**

```math
0 \le p_{t,g} \le \mathrm{p}^{\mathrm{nom}}_{g} \qquad \forall\, t \in \mathcal{T},\ g \in \mathcal{G}
```

**`f`**

```math
\mathrm{neg\_s\_nom}_{l} \le f_{t,l} \le \mathrm{s}^{\mathrm{nom}}_{l} \qquad \forall\, t \in \mathcal{T},\ l \in \mathcal{L}
```

#### Objective

```math
\min \sum_{t \in \mathcal{T},\ g \in \mathcal{G}} p_{t,g} \cdot \mathrm{marginal\_cost}_{g}
```

#### Subject to

**`nodal_balance`**

```math
\sum_{g \in \mathcal{G} \,:\, \mathrm{gen\_bus}(g) = b} p_{t,g} + \sum_{l \in \mathcal{L} \,:\, \mathrm{line\_to}(l) = b} f_{t,l} - \left( \sum_{l \in \mathcal{L} \,:\, \mathrm{line\_from}(l) = b} f_{t,l} \right) = \mathrm{load}_{t,b} \qquad \forall\, t \in \mathcal{T},\ b \in \mathcal{B}
```

**`kirchhoff_voltage_law`**

```math
\sum_{l \in \mathcal{L}} f_{t,l} \cdot \mathrm{cycle\_incidence}_{c,l} = 0 \qquad \forall\, t \in \mathcal{T},\ c \in \mathcal{C}
```

</details>
<!-- math:end -->

The tabs start from [the instance's tables](../howto/data.md) — one frame per parameter.

=== "specsolve"

    ```yaml
    description: >-
      PyPSA linear optimal power flow over passive AC lines under Kirchhoff's
      voltage law, rather than links whose flow is chosen. Optimum 17000.0, from
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
      line:
        description: passive AC lines, each joining two buses
        dtype: str
      cycle:
        description: one independent loop of the network
        dtype: str

    relations:
      gen_bus:
        description: the bus a generator sits on
        key: generator
        values: bus
      line_from:
        description: the bus a line leaves
        key: line
        values: bus
      line_to:
        description: the bus a line arrives at
        key: line
        values: bus

    parameters:
      p_nom:
        description: installed capacity of a generator
        dims: [generator]
      marginal_cost:
        description: cost of one unit of output
        dims: [generator]
      s_nom:
        description: most a line may carry towards its `line_to` bus
        dims: [line]
      neg_s_nom:
        description: most a line may carry the other way, negative by convention
        dims: [line]
      cycle_incidence:
        description: >-
          the cycle basis, as a sparse table of reactance times direction — a line
          may belong to several cycles, so this is a parameter over both dimensions
          rather than a coordinate, and rows are absent where a line is in no cycle
        dims: [cycle, line]
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
      f:
        description: >-
          flow on a line, signed towards its `line_to` bus — not chosen, but whatever
          the voltage law leaves
        dims: [snapshot, line]
        bounds:
          lower: neg_s_nom
          upper: s_nom

    constraints:
      nodal_balance:
        description: what is generated at a bus plus what arrives over the lines meets the load there
        dims: [snapshot, bus]
        expression: >-
          sum(p, over=generator, by=gen_bus[bus])
          + sum(f, over=line, by=line_to[bus])
          - sum(f, over=line, by=line_from[bus])
          == load

      kirchhoff_voltage_law:
        description: >-
          around each independent cycle the reactance-weighted flows sum to zero.
          The incidence table carries both which lines are in the cycle and which
          way round they run, so this is one equation rather than a case analysis
          over the topology — and a coordinate could not hold it, being
          single-valued per label.
        dims: [snapshot, cycle]
        expression: sum(f * cycle_incidence, over=line) == 0

    objective:
      sense: minimize
      description: total cost of generation; the lines carry power for nothing
      expression: sum(p * marginal_cost)
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with sps.solve('examples/ports/pypsa_kvl.yaml', sources) as solution:
        solution.objective  # 17000.0
        solution.dual('nodal_balance')
    ```

=== "PyPSA"

    The model-building half of `examples/ports/references/pypsa/pypsa_kvl.py`:

    ```python
    def build(tables: dict[str, pd.DataFrame]) -> pypsa.Network:
        """The port's tables as a PyPSA network, column for column.

        ``tables`` is the same mapping the specsolve call attaches as ``sources``.

        ``r=0`` keeps a line purely reactive: the linearised power flow is a
        function of ``x`` alone, and a resistance would only add losses the DC
        approximation does not model anyway.
        """
        n = pypsa.Network()
        n.set_snapshots(tables['snapshot']['snapshot'])
        n.add('Bus', tables['bus']['bus'])

        generators: pd.DataFrame = tables['generator'].set_index('generator')
        lines: pd.DataFrame = tables['line'].set_index('line')

        n.add(
            'Generator',
            generators.index,
            bus=tables['gen_bus'].set_index('generator')['bus'],
            p_nom=tables['p_nom'].set_index('generator')['value'],
            marginal_cost=tables['marginal_cost'].set_index('generator')['value'],
        )
        n.add(
            'Line',
            lines.index,
            bus0=tables['line_from'].set_index('line')['bus'],
            bus1=tables['line_to'].set_index('line')['bus'],
            x=tables['reactance'].set_index('line')['value'],
            r=0.0,
            s_nom=tables['s_nom'].set_index('line')['value'],
        )

        load: pd.DataFrame = tables['load'].pivot(index='snapshot', columns='bus', values='value')
        for bus in tables['bus']['bus']:
            n.add('Load', f'load_{bus}', bus=bus, p_set=load[bus])
        return n
    ```

**The cycle basis is a parameter, not a coordinate.** A line may belong to
several cycles, and a declared coordinate holds one value per label. So
`cycle_incidence` is a parameter over `(cycle, line)` carrying reactance ×
direction, with rows absent where a line is not in a cycle. The constraint is
then one equation, `sum(f * cycle_incidence, over=line) == 0`, and
[topology stays data](pypsa_transport.md): a fourth bus is more rows, not a
different file.

**Computing the basis is data preparation, outside the language.** Finding a
cycle basis is a graph algorithm, iteration over a structure discovered from
data, which the
[limits](https://mathspec.readthedocs.io/en/latest/about/limits/#what-counts-as-data-preparation)
refuse. The reference prints the rows PyPSA derived so the two can be
compared. They need only agree on the cycle space: PyPSA scales its
coefficients for conditioning, and a row that is `= 0` says the same thing
under any nonzero multiple.

## What it exercises

A parameter over two dimensions multiplying a variable over one, reduced along
the shared dimension: the shape of an incidence matrix. Plus `sum(by=)` on
both line endpoints for the nodal balance, as in the transport model.
Kirchhoff's voltage law needed no new construct.
