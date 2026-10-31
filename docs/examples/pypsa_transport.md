# PyPSA LOPF — a transport model

PyPSA linear optimal power flow at its smallest: transport model, linear marginal cost, no KVL.

> **✔ Verified against pypsa 1.2.4 (its own linopy 0.10.0)** — objective **22000**, matched to `rtol=1e-09`.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

PyPSA linear optimal power flow at its smallest: a transport model — linear marginal cost, controllable links, no voltage law. Optimum 22000.0, from PyPSA itself.

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
| $`\mathrm{rating}`$ | `rating` over $`\mathcal{L}`$ — most a link may carry towards its `link_to` bus |
| $`\mathrm{neg\_rating}`$ | `neg_rating` over $`\mathcal{L}`$ — most a link may carry the other way, negative by convention |
| $`\mathrm{load}`$ | `load` over $`\mathcal{T} \times \mathcal{B}`$ — demand at each bus in each snapshot |

#### Variables

| Symbol | Meaning |
|---|---|
| $`p`$ | `p` over $`\mathcal{T} \times \mathcal{G}`$ — output of a generator in a snapshot |
| $`f`$ | `f` over $`\mathcal{T} \times \mathcal{L}`$ — PyPSA's p0 — flow measured at the link's `link_from` end, so a positive value withdraws there and injects at `link_to` |

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

</details>
<!-- math:end -->

The tabs start from [the instance's tables](../howto/data.md) — one frame per parameter.

=== "specsolve"

    ```yaml
    description: >-
      PyPSA linear optimal power flow at its smallest: a transport model — linear
      marginal cost, controllable links, no voltage law. Optimum 22000.0, from PyPSA itself.

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
        description: >-
          PyPSA's p0 — flow measured at the link's `link_from` end, so a positive value
          withdraws there and injects at `link_to`
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

    objective:
      sense: minimize
      description: total cost of generation; moving power over a link is free here
      expression: sum(p * marginal_cost)
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with sps.solve('examples/ports/pypsa_transport.yaml', sources) as solution:
        solution.objective  # 22000.0
        solution.dual('nodal_balance')
    ```

=== "PyPSA"

    The model-building half of `examples/ports/references/pypsa/pypsa_transport.py`:

    ```python
    def build(tables: dict[str, pd.DataFrame]) -> pypsa.Network:
        """The port's tables as a PyPSA network, column for column.

        ``tables`` is the same mapping the specsolve call attaches as ``sources``.

        ``p_min_pu = -1`` makes a link bidirectional. The port cannot say that in
        a bound — bounds take a name or a number, never arithmetic (the declaration rules) — so
        it ships ``neg_rating`` as data instead. That is the ledger row.
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

**PyPSA is a domain package, so its tab is short.** `n.add('Generator', ...)`
and `n.add('Link', ...)` carry a power-systems model inside them. The YAML
states the nodal balance PyPSA implies, so it is more explicit rather than
shorter. [The Dantzig page](transport_dantzig.md) compares against a
general-purpose alternative, where both sides write the maths out.

## What it exercises

The smallest whole PyPSA model. A full PyPSA objective mixes marginal and
capital cost, ramp limits, storage cycling and KVL, so a mismatch would
implicate five features at once. Each feature is therefore switched off in
PyPSA and reproduced in its own model: **a transport model** (this one) ·
[ramp limits](pypsa_ramp.md) · [storage](pypsa_storage.md) ·
[a cyclic horizon](pypsa_cyclic_storage.md) · [KVL](pypsa_kvl.md).

**This model hit the ceiling once.** PyPSA's `p_min_pu = -1` is a bound of
`-rating`, an expression `bounds:` cannot take. It ships as a `neg_rating`
column instead. The gap is
[issue #31](https://github.com/fluxopt/specsolve/issues/31), verdict *primitive*,
and [the ledger](../development/verification.md#what-a-port-could-not-say) records it.

---

[`examples/ports/pypsa_transport.yaml`](https://github.com/fluxopt/specsolve/blob/main/examples/ports/pypsa_transport.yaml) · back to [all models](index.md)
