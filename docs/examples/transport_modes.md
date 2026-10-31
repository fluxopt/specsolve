# Choosing the mode of transport

Moving 180 tonnes of chemicals out of four depots, where a depot may reach a centre by rail *or* by road at different cost.

> **✔ Verified against the published optimum** — **1715**, from Guéret, Prins, Sevaux & Heipcke, *Applications of Optimization with Xpress-MP* §10.2.3.

**The connection has a name, so two of them may join the same depot and centre**
and keep their own cost, minimum and capacity. That is the whole model. On
p. 142 the book observes that its data *"cannot be coded as a (two-dimensional)
matrix: for instance the element COST\(_{ij}\) of a cost matrix can only define a
single cost"*. It works around that with **a fictitious node per mode per
connection**, six of them, turning each parallel pair into two paths through
distinct intermediates.

Here the connection is the axis, so the six nodes are not needed. `d2_c2_rail`
and `d2_c2_road` are two rows that disagree on cost (12 against 14) and on band
(10–50 against unbounded). The book's graph carries 15 nodes. This model carries
four.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

Moving 180 tonnes of chemicals out of four depots to three recycling centres, where a depot may reach a centre by rail or by road at different cost. The connection is a thing with a name, so two of them may join the same depot and centre and keep their own cost and their own band; a cost matrix indexed by depot and centre has one cell for the pair and cannot. The rail and road links between the same depot and centre disagree on all three numbers. Guéret, Prins, Sevaux and Heipcke, Applications of Optimization with Xpress-MP, section 10.2. Optimum 1715, published in section 10.2.3.

#### Sets

| Symbol | Meaning |
|---|---|
| $`\mathcal{D}`$ | index $`d`$ — `depot` with $`\mathrm{origin}: \mathcal{C} \to \mathcal{D}`$ — depots the chemicals leave from |
| $`\mathcal{C}`$ | index $`c`$ — `connection` with $`\mathrm{origin}: \mathcal{C} \to \mathcal{D}`$ — one way of reaching one centre from one depot, by rail or by road |

#### Parameters

| Symbol | Meaning |
|---|---|
| $`\mathrm{stock}`$ | `stock` over $`\mathcal{D}`$ — tonnes standing at a depot |
| $`\mathrm{cost}`$ | `cost` over $`\mathcal{C}`$ — cost per tonne moved over a connection |
| $`\mathrm{min\_load}`$ | `min_load` over $`\mathcal{C}`$, `absent` where the data has no row — smallest delivery a connection accepts — rail carries between 10 and 50 tonnes per delivery, and road is unconstrained |
| $`\mathrm{max\_load}`$ | `max_load` over $`\mathcal{C}`$, `absent` where the data has no row — largest delivery a connection accepts |
| $`\mathrm{total\_to\_move}`$ | `total_to_move` (scalar) — tonnes that have to leave the depots altogether |

#### Variables

| Symbol | Meaning |
|---|---|
| $`\mathit{moved}`$ | `moved` over $`\mathcal{C}`$ — tonnes sent over a connection |

#### Variable domains

**`moved`**

```math
\mathrm{min\_load}_{c} \le \mathit{moved}_{c} \le \mathrm{max\_load}_{c} \qquad \forall\, c \in \mathcal{C}
```

#### Objective

```math
\min \sum_{c \in \mathcal{C}} \mathit{moved}_{c} \cdot \mathrm{cost}_{c}
```

#### Subject to

**`within_stock`**

```math
\sum_{c \in \mathcal{C} \,:\, \mathrm{origin}(c) = d} \mathit{moved}_{c} \le \mathrm{stock}_{d} \qquad \forall\, d \in \mathcal{D}
```

**`move_the_lot`**

```math
\sum_{c \in \mathcal{C}} \mathit{moved}_{c} = \mathrm{total\_to\_move}
```

</details>
<!-- math:end -->

The tabs start from [the instance's tables](../howto/data.md) — one frame per parameter.

=== "specsolve"

    ```yaml
    description: >-
      Moving 180 tonnes of chemicals out of four depots to three recycling centres,
      where a depot may reach a centre by rail or by road at different cost. The
      connection is a thing with a name, so two of them may join the same depot and
      centre and keep their own cost and their own band; a cost matrix indexed by
      depot and centre has one cell for the pair and cannot. The rail and road
      links between the same depot and centre disagree on all three numbers.
      Guéret, Prins, Sevaux and Heipcke, Applications of Optimization with
      Xpress-MP, section 10.2. Optimum 1715, published in section 10.2.3.

    dimensions:
      depot:
        description: depots the chemicals leave from
        dtype: str
      connection:
        description: one way of reaching one centre from one depot, by rail or by road
        dtype: str

    relations:
      origin:
        description: the depot a connection leaves
        key: connection
        values: depot

    parameters:
      stock:
        description: tonnes standing at a depot
        dims: [depot]
      cost:
        description: cost per tonne moved over a connection
        dims: [connection]
      min_load:
        description: >-
          smallest delivery a connection accepts — rail carries between 10 and 50
          tonnes per delivery, and road is unconstrained
        dims: [connection]
        missing: absent
      max_load:
        description: largest delivery a connection accepts
        dims: [connection]
        missing: absent
      total_to_move:
        description: tonnes that have to leave the depots altogether
        dims: []

    variables:
      moved:
        description: tonnes sent over a connection
        dims: [connection]
        bounds:
          lower: min_load
          upper: max_load

    constraints:
      within_stock:
        dims: [depot]
        expression: sum(moved, over=connection, by=origin[depot]) <= stock

      move_the_lot:
        dims: []
        expression: sum(moved, over=connection) == total_to_move

    objective:
      sense: minimize
      description: total cost of the deliveries
      expression: sum(moved * cost)
    ```

**The centre is in the connection's name and in nothing else.** This problem
sets no per-centre demand, only a total of 180 t, so no constraint reads where a
connection ends. A `destination` coordinate would be a declaration nothing
consumes, and the language warns about that, so it is not written. The pairing
is still visible where it matters: two connections out of `d2` differing only in
mode, which is what a `(depot, centre)` table cannot hold.

The second leg was load-bearing in the *book's* formulation, because its
fictitious nodes need conservation, which reads both ends of every arc.
Reifying the connection removed that job. A problem with per-destination demand
needs both legs. §12.3 of the same book is one, and is a port of its own.

**The rail band is a hard bound here, as in the source.** The prose says rail
carries *"at least 10 tonnes and at most 50 tonnes for any single delivery"*,
which reads as semi-continuous: either nothing or at least ten. The book's own
Mosel model writes `flow(a) >= MINCAP(a)` unconditionally, so every rail
connection carries at least 10 t whether it is wanted or not. The published
1715 is that reading. The port matches the model that produced the number. The
semi-continuous variant is
[#383](https://github.com/fluxopt/specsolve/issues/383).

## What it exercises

`sum(by=)` through a relation whose target carries a constraint of its own, and
per-label bounds read from data, where the *label* is a connection rather than
a pair.
