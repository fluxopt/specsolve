# PyPSA multi-link

One `Link`, one input bus, several output buses, each output derated by its
own efficiency — PyPSA's spelling for a CHP plant, an electrolyser with waste
heat, any conversion with more than one product.

> **✔ Verified against pypsa 1.2.4 (its own linopy 0.10.0)** — objective **1100**, matched to `rtol=1e-09`.

**The schema varies, not the network.** PyPSA holds the relation wide: `bus0`,
`bus1`, `bus2`, `efficiency`, `efficiency2`, and an empty `bus2` where a link
has only two ends. Every arity the data reaches adds a column pair to the
component. Here the relation is one incidence parameter over `(link, bus)`:
`-1` at the input, `+efficiency` at each output, rows absent elsewhere. Arity
is the number of rows that name the link. The three-ended CHP and the
two-ended boiler sit in the same three columns, and a four-ended link would
change only the data.

One decision per link: `p`, what the link draws at its input, PyPSA's `p0`.
The balance is the contraction `sum(incidence * p, over=link)`, which lands
the draw on every bus the link's rows name. A linopy user writes the same sum
as `(incidence * p).sum('link')` against a dense array.

The instance is a toy gas-to-energy system: a CHP (gas → 0.4 elec + 0.4 heat,
capacity 50), a boiler (gas → 0.8 heat), an OCGT (gas → 0.5 elec), gas at 10.
The boiler sets the heat price and the OCGT the electricity price:
`10/0.8 = 12.5` and `10/0.5 = 20`. One unit of gas through the CHP earns
`0.4·20 + 0.4·12.5 = 13` against 10, so the CHP runs at its cap of 50 and the
others top up. Flows `(50, 20, 40)`, gas 110, objective **1100**.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

PyPSA multi-link: one Link, one input bus, several output buses, each output derated by its own efficiency. PyPSA spells the relation wide — bus0, bus1, bus2, efficiency, efficiency2, an empty bus2 where a link has no third terminal — and grows a column pair per arity. Here the relation is one incidence parameter over link and bus, so arity is the number of rows that name the link. Optimum 1100.0, from PyPSA itself.

#### Sets

| Symbol | Meaning |
|---|---|
| $`\mathcal{B}`$ | index $`b`$ — `bus` with $`\mathrm{gen\_bus}: \mathcal{G} \to \mathcal{B}`$ — network nodes |
| $`\mathcal{G}`$ | index $`g`$ — `generator` with $`\mathrm{gen\_bus}: \mathcal{G} \to \mathcal{B}`$ — generating units, each sitting on one bus |
| $`\mathcal{L}`$ | index $`l`$ — `link` — conversions, each drawing at one bus and delivering at several |

#### Parameters

| Symbol | Meaning |
|---|---|
| $`\mathrm{gen}^{\mathrm{p,nom}}`$ | `gen_p_nom` over $`\mathcal{G}`$ — installed capacity of a generator |
| $`\mathrm{marginal\_cost}`$ | `marginal_cost` over $`\mathcal{G}`$ — cost of one unit of output |
| $`\mathrm{p}^{\mathrm{nom}}`$ | `p_nom` over $`\mathcal{L}`$ — the link's own capacity — a cap on what it draws at its input, p0 in PyPSA |
| $`\mathrm{incidence}`$ | `incidence` over $`\mathcal{L} \times \mathcal{B}`$, `neutral` where the data has no row — each bus's share of the link's draw — minus one at the input and plus the efficiency at each output, with rows absent elsewhere; PyPSA's efficiency columns and the input's fixed minus one, tidied into rows |
| $`\mathrm{load}`$ | `load` over $`\mathcal{B}`$ — demand at each bus |

#### Variables

| Symbol | Meaning |
|---|---|
| $`\mathit{gen}`$ | `gen` over $`\mathcal{G}`$ — output of a generator |
| $`p`$ | `p` over $`\mathcal{L}`$ — the one decision per link, PyPSA's p — what it draws at its input. Every other end's flow is that draw scaled by its incidence entry, so it needs no variable of its own. |

#### Variable domains

**`gen`**

```math
0 \le \mathit{gen}_{g} \le \mathrm{gen}^{\mathrm{p,nom}}_{g} \qquad \forall\, g \in \mathcal{G}
```

**`p`**

```math
0 \le p_{l} \le \mathrm{p}^{\mathrm{nom}}_{l} \qquad \forall\, l \in \mathcal{L}
```

#### Objective

```math
\min \sum_{g \in \mathcal{G}} \mathit{gen}_{g} \cdot \mathrm{marginal\_cost}_{g}
```

#### Subject to

**`nodal_balance`**

```math
\sum_{g \in \mathcal{G} \,:\, \mathrm{gen\_bus}(g) = b} \mathit{gen}_{g} + \sum_{l \in \mathcal{L}} \mathrm{incidence}_{l,b} \cdot p_{l} = \mathrm{load}_{b} \qquad \forall\, b \in \mathcal{B}
```

</details>
<!-- math:end -->

The tabs start from [the instance's tables](../howto/data.md) — one frame per parameter.

=== "specsolve"

    ```yaml
    description: >-
      PyPSA multi-link: one Link, one input bus, several output buses, each output
      derated by its own efficiency. PyPSA spells the relation wide — bus0, bus1,
      bus2, efficiency, efficiency2, an empty bus2 where a link has no third
      terminal — and grows a column pair per arity. Here the relation is one
      incidence parameter over link and bus, so arity is the number of rows that
      name the link. Optimum 1100.0, from PyPSA itself.

    dimensions:
      bus:
        description: network nodes
        dtype: str
      generator:
        description: generating units, each sitting on one bus
        dtype: str
      link:
        description: conversions, each drawing at one bus and delivering at several
        dtype: str

    relations:
      gen_bus: {key: generator, values: bus, description: "the bus a generator sits on"}

    parameters:
      gen_p_nom:
        description: installed capacity of a generator
        dims: [generator]
      marginal_cost:
        description: cost of one unit of output
        dims: [generator]
      p_nom:
        description: the link's own capacity — a cap on what it draws at its input, p0 in PyPSA
        dims: [link]
      incidence:
        description: >-
          each bus's share of the link's draw — minus one at the input and plus the
          efficiency at each output, with rows absent elsewhere; PyPSA's efficiency
          columns and the input's fixed minus one, tidied into rows
        dims: [link, bus]
        missing: neutral
      load:
        description: demand at each bus
        dims: [bus]

    variables:
      gen:
        description: output of a generator
        dims: [generator]
        bounds:
          lower: 0
          upper: gen_p_nom
      p:
        description: >-
          the one decision per link, PyPSA's p — what it draws at its input. Every
          other end's flow is that draw scaled by its incidence entry, so it needs
          no variable of its own.
        dims: [link]
        bounds:
          lower: 0
          upper: p_nom

    constraints:
      nodal_balance:
        description: >-
          what is generated at a bus plus what the links deliver there meets the
          load. The contraction lands the draw on every bus its link's incidence
          rows name — three ends or two, the expression never says.
        dims: [bus]
        expression: >-
          sum(gen, over=generator, by=gen_bus[bus])
          + sum(incidence * p, over=link)
          == load

    objective:
      sense: minimize
      description: total cost of generation; the conversions themselves are free here
      expression: sum(gen * marginal_cost)
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with sps.solve('examples/ports/pypsa_multilink.yaml', sources) as solution:
        solution.objective  # 1100.0
        solution.dual('nodal_balance')
    ```

=== "PyPSA"

    The model-building half of `examples/ports/references/pypsa/pypsa_multilink.py`:

    ```python
    def build(tables: dict[str, pd.DataFrame]) -> pypsa.Network:
        """The port's tables as a PyPSA network.

        ``tables`` is the same mapping the specsolve call attaches as ``sources``; only
        the incidence table changes shape on the way in, pivoted from one row per
        link end into PyPSA's one row per link. The input end is the one with the
        negative value — PyPSA fixes its share at -1, so the pivot asserts it: a
        different input share is sayable in rows and not in these columns. Each
        output end becomes the link's next port, its value the port's efficiency,
        in the incidence table's row order. A link narrower than the instance's
        widest is padded with ``''`` — PyPSA's spelling for a port a link does not
        have — and a filler efficiency of 1.0 that no equation reads.
        """
        incidence = tables['incidence']
        inputs = incidence[incidence['value'] < 0].set_index('link')
        assert (inputs['value'] == -1.0).all(), 'PyPSA fixes the input share of a Link at -1; this instance must too'

        outputs = incidence[incidence['value'] > 0].copy()
        outputs['port'] = outputs.groupby('link', sort=False).cumcount() + 1
        buses = outputs.pivot(index='link', columns='port', values='bus')
        efficiencies = outputs.pivot(index='link', columns='port', values='value')

        links = pd.DataFrame(index=pd.Index(tables['link']['link'], name='link'))
        links['bus0'] = inputs['bus']
        for port in buses.columns:
            links[f'bus{port}'] = buses[port].reindex(links.index).fillna('')
            links['efficiency' if port == 1 else f'efficiency{port}'] = efficiencies[port].reindex(links.index).fillna(1.0)

        n = pypsa.Network()
        n.add('Bus', tables['bus']['bus'])

        generators: pd.DataFrame = tables['generator'].set_index('generator')
        n.add(
            'Generator',
            generators.index,
            bus=tables['gen_bus'].set_index('generator')['bus'],
            p_nom=tables['gen_p_nom'].set_index('generator')['value'],
            marginal_cost=tables['marginal_cost'].set_index('generator')['value'],
        )
        n.add('Link', links.index, p_nom=tables['p_nom'].set_index('link')['value'], **dict(links.items()))

        load: pd.Series = tables['load'].set_index('bus')['value']
        for bus in tables['bus']['bus']:
            n.add('Load', f'load_{bus}', bus=bus, p_set=load[bus])
        return n
    ```

**The PyPSA tab pivots first.** Its first half turns rows into columns:
finding the input, numbering the outputs, padding the narrow links with `''`
and a filler efficiency no equation reads. The wide schema demands that
reshape of any tidy source; the specsolve tab attaches the incidence table as it
stands. PyPSA fixes the input's share at `-1`, so the pivot asserts it. In
rows that constant is data: an input entry of `-1.05` models a link burning 5%
of its draw in station load, with no new column and no new construct.

**When a link end needs a name of its own, the incidence entry stops being
enough.** A per-end variable or bound (a heat-offtake cap on the CHP's heat
port alone), a value pulled through an end's bus with `at(..., by=…)`, or a
link touching the same bus twice all need the ends as a dimension with leg
relations. A parameter holds one value per `(link, bus)` pair and gives the pair
no identity. [Reserves](reserves.md) proves that other idiom with its
three-legged offers.
