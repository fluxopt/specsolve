# reserves

Energy and reserve co-optimization on a two-bus grid: offers are (generator, market, tranche) triples, reserve zones overlap, and one line dangles.

> **✔ Agrees with hand-written linopy 0.10.0** — objective **915**, matched to `rtol=1e-09`.

## The problem

A relation either **is an axis**, a pair set reified as a dimension whose legs
are relations, or **is data** weighting one aggregation. Both appear here. The
offer set is the first kind, three-legged:

$$r_o \;\le\; \phi_{\mathrm{tranche\_of}(o)} \cdot \bar p_{\mathrm{gen\_of}(o)} \qquad \forall\, o$$

a per-offer cap assembled by pulling two other dimensions' parameters back
through two legs of one edge set. The zones are the second kind: membership
with a weight is a sparse parameter, and the zone requirement is the
contraction

$$\sum_{g} \sigma_{g,z} \cdot \Big( \sum_{o \,:\, \mathrm{gen\_of}(o) = g} r_o \Big) \;\ge\; \underline{R}_z \qquad \forall\, z$$

Multiply by the incidence table and sum the dimension away. A generator may
back several zones at different weights, which no relation can say.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

Energy and reserve co-optimization on a two-bus grid: an offer is a generator, market and tranche together, reserve zones overlap, and one line dangles. The model exists to prove a claim — every many-to-many shape the language covers, in one instance, each one load-bearing.

#### Sets

| Symbol | Meaning |
|---|---|
| $`\mathcal{B}`$ | index $`b`$ — `bus` with $`\mathrm{gen\_bus}: \mathcal{G} \to \mathcal{B},\ \mathrm{line\_from}: \mathcal{L} \to \mathcal{B},\ \mathrm{line\_to}: \mathcal{L} \to \mathcal{B}`$, `line_from` is `absent` where the data has no row, `line_to` is `absent` where the data has no row — network nodes |
| $`\mathcal{G}`$ | index $`g`$ — `generator` with $`\mathrm{gen\_bus}: \mathcal{G} \to \mathcal{B},\ \mathrm{gen\_of}: \mathcal{O} \to \mathcal{G}`$ — generating units, each sitting on one bus |
| $`\mathcal{M}`$ | index $`m`$ — `market` with $`\mathrm{market\_of}: \mathcal{O} \to \mathcal{M}`$ — reserve markets, each with a requirement to fill |
| $`\mathcal{T}`$ | index $`t`$ — `tranche` with $`\mathrm{tranche\_of}: \mathcal{O} \to \mathcal{T}`$ — how fast a reserve has to be deliverable |
| $`\mathcal{Z}`$ | index $`z`$ — `zone` — reserve zones, which overlap |
| $`\mathcal{L}`$ | index $`l`$ — `line` with $`\mathrm{line\_from}: \mathcal{L} \to \mathcal{B},\ \mathrm{line\_to}: \mathcal{L} \to \mathcal{B}`$, `line_from` is `absent` where the data has no row, `line_to` is `absent` where the data has no row — transmission lines, which may have an open end |
| $`\mathcal{O}`$ | index $`o`$ — `offer` with $`\mathrm{gen\_of}: \mathcal{O} \to \mathcal{G},\ \mathrm{market\_of}: \mathcal{O} \to \mathcal{M},\ \mathrm{tranche\_of}: \mathcal{O} \to \mathcal{T}`$ — one generator's bid into one market at one tranche |

#### Parameters

| Symbol | Meaning |
|---|---|
| $`\mathrm{p}^{\mathrm{max}}`$ | `p_max` over $`\mathcal{G}`$ — installed capacity |
| $`\mathrm{energy\_cost}`$ | `energy_cost` over $`\mathcal{G}`$ — cost of one unit of output |
| $`\mathrm{load}`$ | `load` over $`\mathcal{B}`$ — demand at each bus |
| $`\mathrm{cap}`$ | `cap` over $`\mathcal{L}`$ — forward transmission limit |
| $`\mathrm{neg\_cap}`$ | `neg_cap` over $`\mathcal{L}`$ — reverse transmission limit |
| $`\mathrm{bus\_cap}`$ | `bus_cap` over $`\mathcal{B}`$ — most a bus may export over any one line |
| $`\mathrm{offer\_cost}`$ | `offer_cost` over $`\mathcal{O}`$ — cost of holding one unit of reserve on an offer |
| $`\mathrm{req}`$ | `req` over $`\mathcal{M}`$ — reserve a market has to be filled with |
| $`\mathrm{tranche\_frac}`$ | `tranche_frac` over $`\mathcal{T}`$ — share of capacity a generator may offer at a tranche |
| $`\mathrm{zone\_share}`$ | `zone_share` over $`\mathcal{G} \times \mathcal{Z}`$, `neutral` where the data has no row — how much of a generator's reserve counts towards a zone — a generator may back several zones at a per-zone weight, so this cannot be a relation over the generator, which is single-valued per label; rows are absent where a generator backs no part of a zone |
| $`\mathrm{zone\_req}`$ | `zone_req` over $`\mathcal{Z}`$ — reserve a zone has to be covered by |

#### Variables

| Symbol | Meaning |
|---|---|
| $`p`$ | `p` over $`\mathcal{G}`$ — output of a generator |
| $`f`$ | `f` over $`\mathcal{L}`$ — flow on a line, signed towards its `line_to` bus |
| $`r`$ | `r` over $`\mathcal{O}`$ — reserve held against an offer |

#### Definitions

| Symbol | Meaning |
|---|---|
| $`\mathit{reserve\_of}`$ | `reserve_of` over $`\mathcal{G}`$ — all the reserve a generator holds, across every offer it made |

#### Variable domains

**`p`**

```math
p_{g} \ge 0 \qquad \forall\, g \in \mathcal{G}
```

**`f`**

```math
\mathrm{neg\_cap}_{l} \le f_{l} \le \mathrm{cap}_{l} \qquad \forall\, l \in \mathcal{L}
```

**`r`**

```math
r_{o} \ge 0 \qquad \forall\, o \in \mathcal{O}
```

#### Definitions

**`reserve_of`**

```math
\mathit{reserve\_of}_{g} = \sum_{o \in \mathcal{O} \,:\, \mathrm{gen\_of}(o) = g} r_{o} \qquad \forall\, g \in \mathcal{G}
```

#### Objective

```math
\min \sum_{g \in \mathcal{G}} p_{g} \cdot \mathrm{energy\_cost}_{g} + \sum_{o \in \mathcal{O}} r_{o} \cdot \mathrm{offer\_cost}_{o}
```

#### Subject to

**`balance`**

```math
\sum_{g \in \mathcal{G} \,:\, \mathrm{gen\_bus}(g) = b} p_{g} + \sum_{l \in \mathcal{L} \,:\, \mathrm{line\_to}(l) = b} f_{l} - \left( \sum_{l \in \mathcal{L} \,:\, \mathrm{line\_from}(l) = b} f_{l} \right) = \mathrm{load}_{b} \qquad \forall\, b \in \mathcal{B}
```

**`export_cap`**

```math
f_{l} \le \mathrm{bus\_cap}_{\mathrm{line\_from}(l)} \qquad \forall\, l \in \mathcal{L}
```

**`requirement`**

```math
\sum_{o \in \mathcal{O} \,:\, \mathrm{market\_of}(o) = m} r_{o} \ge \mathrm{req}_{m} \qquad \forall\, m \in \mathcal{M}
```

**`headroom`**

```math
p_{g} + \mathit{reserve\_of}_{g} \le \mathrm{p}^{\mathrm{max}}_{g} \qquad \forall\, g \in \mathcal{G}
```

**`offer_cap`**

```math
r_{o} \le \mathrm{tranche\_frac}_{\mathrm{tranche\_of}(o)} \cdot \mathrm{p}^{\mathrm{max}}_{\mathrm{gen\_of}(o)} \qquad \forall\, o \in \mathcal{O}
```

**`zone_cover`**

```math
\sum_{g \in \mathcal{G}} \mathrm{zone\_share}_{g,z} \cdot \mathit{reserve\_of}_{g} \ge \mathrm{zone\_req}_{z} \qquad \forall\, z \in \mathcal{Z}
```

</details>
<!-- math:end -->

The tabs start from [the instance's tables](../howto/data.md) — one frame per parameter.

=== "specsolve"

    ```yaml
    description: >-
      Energy and reserve co-optimization on a two-bus grid: an offer is a
      generator, market and tranche together, reserve zones overlap, and one line
      dangles. The model exists to prove a claim — every many-to-many shape the
      language covers, in one instance, each one load-bearing.

    dimensions:
      bus:
        description: network nodes
        dtype: str
      generator:
        description: generating units, each sitting on one bus
        dtype: str
      market:
        description: reserve markets, each with a requirement to fill
        dtype: str
      tranche:
        description: how fast a reserve has to be deliverable
        dtype: str
      zone:
        description: reserve zones, which overlap
        dtype: str
      line:
        description: transmission lines, which may have an open end
        dtype: str
      offer:
        description: one generator's bid into one market at one tranche
        dtype: str

    relations:
      gen_bus: {key: generator, values: bus, description: "the bus a generator sits on"}
      line_from: {key: line, values: bus, description: "the bus a line leaves, null where the end is open", missing: absent}
      line_to: {key: line, values: bus, description: "the bus a line arrives at, null where the end is open", missing: absent}
      gen_of: {key: offer, values: generator, description: "the generator behind an offer"}
      market_of: {key: offer, values: market, description: "the market an offer is made into"}
      tranche_of: {key: offer, values: tranche, description: "the tranche an offer is made at"}

    parameters:
      p_max:
        description: installed capacity
        dims: [generator]
      energy_cost:
        description: cost of one unit of output
        dims: [generator]
      load:
        description: demand at each bus
        dims: [bus]
      cap:
        description: forward transmission limit
        dims: [line]
      neg_cap:
        description: reverse transmission limit
        dims: [line]
      bus_cap:
        description: most a bus may export over any one line
        dims: [bus]
      offer_cost:
        description: cost of holding one unit of reserve on an offer
        dims: [offer]
      req:
        description: reserve a market has to be filled with
        dims: [market]
      tranche_frac:
        description: share of capacity a generator may offer at a tranche
        dims: [tranche]
      zone_share:
        description: >-
          how much of a generator's reserve counts towards a zone — a generator may
          back several zones at a per-zone weight, so this cannot be a relation over
          the generator, which is single-valued per label; rows are absent where a
          generator backs no part of a zone
        dims: [generator, zone]
        missing: neutral
      zone_req:
        description: reserve a zone has to be covered by
        dims: [zone]

    variables:
      p:
        description: output of a generator
        dims: [generator]
        bounds:
          lower: 0
      f:
        description: flow on a line, signed towards its `line_to` bus
        dims: [line]
        bounds:
          lower: neg_cap
          upper: cap
      r:
        description: reserve held against an offer
        dims: [offer]
        bounds:
          lower: 0

    expressions:
      reserve_of:
        expression: sum(r, over=offer, by=gen_of[generator])
        description: all the reserve a generator holds, across every offer it made

    constraints:
      balance:
        description: what is generated at a bus plus what arrives over the lines meets the load there
        dims: [bus]
        expression: >-
          sum(p, over=generator, by=gen_bus[bus])
          + sum(f, over=line, by=line_to[bus])
          - sum(f, over=line, by=line_from[bus])
          == load
      export_cap:
        description: a line carries no more than the bus it leaves is allowed to export
        dims: [line]
        expression: f <= at(bus_cap, by=line_from[bus])
      requirement:
        description: the offers made into a market fill its requirement
        dims: [market]
        expression: sum(r, over=offer, by=market_of[market]) >= req
      headroom:
        description: a generator's output plus the reserve it holds stays inside its capacity
        dims: [generator]
        expression: p + reserve_of <= p_max
      offer_cap:
        description: >-
          an offer is capped by its tranche's share of its generator's capacity —
          two other dimensions' parameters pulled back through two legs of one edge
          set
        dims: [offer]
        expression: r <= at(tranche_frac, by=tranche_of[tranche]) * at(p_max, by=gen_of[generator])
      zone_cover:
        description: the weighted reserve of the generators backing a zone covers its requirement
        dims: [zone]
        expression: sum(zone_share * reserve_of, over=generator) >= zone_req

    objective:
      sense: minimize
      description: what the energy costs to generate, plus what the reserve costs to hold
      expression: sum(p * energy_cost) + sum(r * offer_cost)
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with sps.solve('examples/reserves.yaml', sources) as solution:
        solution.objective  # 915.0
        solution.dual('balance')
    ```

=== "linopy"

    The model-building half of `examples/ports/references/linopy/reserves.py`,
    which states every mapping as a dense incidence matrix (its `indicator`
    helper) and multiplies through by hand — the identical algebra with no
    specsolve construct anywhere near it:

    ```python
    def build(tables: dict[str, pd.DataFrame]) -> linopy.Model:
        """The instance's tables as a linopy model, row for row.

        ``tables`` is the same mapping the specsolve call attaches as ``sources``.
        """
        series = {
            k: tables[k].set_index(tables[k].columns[0])['value']
            for k in (
                'p_max',
                'energy_cost',
                'load',
                'cap',
                'neg_cap',
                'bus_cap',
                'offer_cost',
                'req',
                'tranche_frac',
                'zone_req',
            )
        }
        buses = pd.Index(series['load'].index, name='bus')
        offers = tables['offer'].set_index('offer')
        zones = pd.Index(series['zone_req'].index, name='zone')

        gen_at = indicator(buses, tables['gen_bus'], 'generator', 'bus')
        line_in = indicator(buses, tables['line_to'], 'line', 'bus')
        line_out = indicator(buses, tables['line_from'], 'line', 'bus')
        offer_gen = indicator(pd.Index(series['p_max'].index, name='generator'), tables['gen_of'], 'offer', 'generator')
        offer_market = indicator(pd.Index(series['req'].index, name='market'), tables['market_of'], 'offer', 'market')

        zone_at = pd.DataFrame(0.0, index=zones, columns=series['p_max'].index)
        for gen, zone, share in zip(
            tables['zone_share']['generator'], tables['zone_share']['zone'], tables['zone_share']['value'], strict=True
        ):
            zone_at.loc[zone, gen] = share
        zone_at.columns.name = 'generator'

        tranche_of = tables['tranche_of'].set_index('offer')['tranche']
        gen_of = tables['gen_of'].set_index('offer')['generator']
        r_cap = tranche_of.map(series['tranche_frac']) * gen_of.map(series['p_max'])
        f_cap = tables['line_from'].set_index('line')['bus'].map(series['bus_cap'])

        m = linopy.Model()
        p = m.add_variables(lower=0, coords=[series['p_max'].index], name='p')
        f = m.add_variables(lower=series['neg_cap'], upper=series['cap'], coords=[series['cap'].index], name='f')
        r = m.add_variables(lower=0, upper=r_cap, coords=[offers.index], name='r')

        m.add_constraints(
            (p * gen_at).sum('generator') + (f * line_in).sum('line') - (f * line_out).sum('line') == series['load'],
            name='balance',
        )
        m.add_constraints(f <= f_cap, name='export_cap')
        m.add_constraints((r * offer_market).sum('offer') >= series['req'], name='requirement')
        reserve_of = (r * offer_gen).sum('offer')
        m.add_constraints(p + reserve_of <= series['p_max'], name='headroom')
        m.add_constraints((reserve_of * xr.DataArray(zone_at)).sum('generator') >= series['zone_req'], name='zone_cover')
        m.add_objective((p * series['energy_cost']).sum() + (r * series['offer_cost']).sum())
        return m
    ```

## The optimum, by hand

Energy: b2's cheap surplus exports over `l1`, pinned at 15 by `bus_cap` rather
than its own 20, and over `l2` at its own 8. So `g3` runs 40 local plus 23
export, 63 in all, and `g1` covers the remaining 47 at b1. Energy costs 785.
Reserves: `m1`'s 55 takes both parallel `g1` offers at their 25 caps, `o2`
first at cost 1 and then `o1` at 2, plus 5 of `o3`. That seat on `g2` is also
what closes zone `z2` at exactly 25. `m2`'s 20 is `o4` at its tranche cap.
Reserve costs 130. Total **915**, nodal prices 10 at b1 and 5 at b2.
