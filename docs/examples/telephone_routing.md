# Routing telephone calls

How many of 425 requested circuits a five-city network can carry at once — and by which routes.

> **✔ Verified against the published optimum** — **380 circuits**, from Guéret, Prins, Sevaux & Heipcke, *Applications of Optimization with Xpress-MP* §12.3.3.

**Both kinds of relation in one model.** A path serves
exactly one city pair, so `call_of` is a coordinate and the demand limit groups
through it. A path traverses several arcs, so `uses` is a `(path, arc)` table
and the capacity limit contracts against it. Neither could stand in for the
other: a coordinate is single-valued, and an incidence table cannot be grouped
by.

Five cities, six undirected links, five city pairs asking for circuits, and 17
elementary paths between them. A circuit reserves both directions of every link
it crosses, which is why the network is undirected and the flow carries no sign.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

Routing telephone calls over a five-city network: how many of the 425 requested circuits can be carried at once. A path serves exactly one city pair — a coordinate — and traverses several arcs — an incidence parameter; both relations, each said the way it is. Guéret, Prins, Sevaux and Heipcke, Applications of Optimization with Xpress-MP, section 12.3. Optimum 380 circuits, published in section 12.3.3.

#### Sets

| Symbol | Meaning |
|---|---|
| $`\mathcal{A}`$ | index $`a`$ — `arc` — an undirected link between two cities, with capacity in circuits |
| $`\mathcal{C}`$ | index $`c`$ — `call` with $`\mathrm{call\_of}: \mathcal{P} \to \mathcal{C}`$ — a city pair with circuits to place |
| $`\mathcal{P}`$ | index $`p`$ — `path` with $`\mathrm{call\_of}: \mathcal{P} \to \mathcal{C}`$ — a route end to end, serving one city pair |

#### Parameters

| Symbol | Meaning |
|---|---|
| $`\mathrm{capacity}`$ | `capacity` over $`\mathcal{A}`$ — circuits an arc can carry |
| $`\mathrm{demand}`$ | `demand` over $`\mathcal{C}`$ — circuits a city pair asked for |
| $`\mathrm{uses}`$ | `uses` over $`\mathcal{P} \times \mathcal{A}`$, `neutral` where the data has no row — which arcs a path traverses — a path uses an arc or it does not, so the value is 1 and absence is 0 |

#### Variables

| Symbol | Meaning |
|---|---|
| $`\mathit{flow}`$ | `flow` over $`\mathcal{P}`$ — circuits carried on a path — integral because a multi-commodity flow is not integral by nature, even though this instance's relaxation happens to be |

#### Variable domains

**`flow`**

```math
\mathit{flow}_{p} \ge 0, \mathit{flow}_{p} \in \mathbb{Z} \qquad \forall\, p \in \mathcal{P}
```

#### Objective

```math
\max \sum_{p \in \mathcal{P}} \mathit{flow}_{p}
```

#### Subject to

**`within_demand`**

```math
\sum_{p \in \mathcal{P} \,:\, \mathrm{call\_of}(p) = c} \mathit{flow}_{p} \le \mathrm{demand}_{c} \qquad \forall\, c \in \mathcal{C}
```

**`within_capacity`**

```math
\sum_{p \in \mathcal{P}} \mathit{flow}_{p} \cdot \mathrm{uses}_{p,a} \le \mathrm{capacity}_{a} \qquad \forall\, a \in \mathcal{A}
```

</details>
<!-- math:end -->

The tabs start from [the instance's tables](../howto/data.md) — one frame per parameter.

=== "specsolve"

    ```yaml
    description: >-
      Routing telephone calls over a five-city network: how many of the 425
      requested circuits can be carried at once. A path serves exactly one city
      pair — a coordinate — and traverses several arcs — an incidence parameter;
      both relations, each said the way it is. Guéret, Prins, Sevaux and Heipcke,
      Applications of Optimization with Xpress-MP, section 12.3. Optimum 380
      circuits, published in section 12.3.3.

    dimensions:
      arc:
        description: an undirected link between two cities, with capacity in circuits
        dtype: str
      call:
        description: a city pair with circuits to place
        dtype: str
      path:
        description: a route end to end, serving one city pair
        dtype: str

    relations:
      call_of:
        description: the city pair a path serves, end to end
        key: path
        values: call

    parameters:
      capacity:
        description: circuits an arc can carry
        dims: [arc]
      demand:
        description: circuits a city pair asked for
        dims: [call]
      uses:
        description: >-
          which arcs a path traverses — a path uses an arc or it does not, so the
          value is 1 and absence is 0
        dims: [path, arc]
        missing: neutral

    variables:
      flow:
        description: >-
          circuits carried on a path — integral because a multi-commodity flow is
          not integral by nature, even though this instance's relaxation happens to
          be
        dims: [path]
        domain: integer
        bounds:
          lower: 0

    constraints:
      within_demand:
        description: a pair cannot be carried more than it asked for, however many paths serve it
        dims: [call]
        expression: sum(flow, over=path, by=call_of[call]) <= demand

      within_capacity:
        description: >-
          an arc carries every path that traverses it, and no more than its
          capacity. A circuit reserves both directions of every arc it crosses,
          which is why the network is undirected and the flow is not signed.
        dims: [arc]
        expression: sum(flow * uses, over=path) <= capacity

    objective:
      sense: maximize
      description: circuits carried, summed over every path
      expression: sum(flow, over=path)
    ```

**The answer is 380 and not 425 because of one cut.** Troyes is reachable only
over `troyes_nice` (80) and `troyes_valenciennes` (70), so at most 150 circuits
can terminate there. Troyes must absorb the 80 Nantes–Troyes and the 70
Paris–Troyes, which is 150 exactly. Valenciennes is reached only over
`paris_valenciennes` (200) and `troyes_valenciennes` (70), capping everything
ending there at 270 against a demand of 175 plus whatever transits. The binding
cut leaves 45 of the Nantes–Troyes circuits unplaced, and the published routing
says the same: 35 of 80 carried. The optimum is arithmetic anybody can check
before a solver runs.

## What it exercises

A coordinate and an incidence parameter side by side, each carrying the relation
it fits, in a model from outside the project. `reserves` proves both idioms on a
model built to prove them. This is the outside witness.

Integrality is the source's. A multi-commodity flow is not integral by nature,
and the book says so before observing that this instance's relaxation happens to
be. A MILP has no dual solution, so the entry records none.
