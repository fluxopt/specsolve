# Uncapacitated facility location

Where do you put the warehouses? Open a set of them, assign every customer to one, and trade the fixed cost of opening against the cost of serving from further away.

> **✔ Verified against OR-Library's published optimum** — **932615.750**, matched to `rtol=1e-09`.

OR-Library instance `cap71`: 16 candidate warehouses, 50 customers. The optimum
is [published by Beasley](http://people.brunel.ac.uk/~mastjjb/jeb/orlib/uncapinfo.html)
in the file `uncapopt`, alongside the instance itself.

**No reference script.** The number comes from the literature, and nothing of
ours is in the loop that produced it.
[Dantzig transport](transport_dantzig.md) is verified the same way.

It also brings **fixed charge**. Every other verified model prices what flows.
This one prices a *decision*: opening a warehouse costs money whether or not it
ends up busy.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

Uncapacitated facility location, OR-Library instance cap71: 16 possible warehouses, 50 customers. Open a set of warehouses and assign every customer to one, trading fixed opening costs against the cost of serving from further away. Optimum 932615.750, published by OR-Library.

#### Sets

| Symbol | Meaning |
|---|---|
| $`\mathcal{W}`$ | index $`w`$ — `warehouse` — sites a warehouse may be opened on |
| $`\mathcal{C}`$ | index $`c`$ — `customer` — customers, each served in full from one warehouse |

#### Parameters

| Symbol | Meaning |
|---|---|
| $`\mathrm{fixed\_cost}`$ | `fixed_cost` over $`\mathcal{W}`$ — what opening a warehouse costs, whoever it ends up serving |
| $`\mathrm{serve}^{\mathrm{cost}}`$ | `serve_cost` over $`\mathcal{W} \times \mathcal{C}`$ — what it costs to serve all of this customer's demand from this warehouse |

#### Variables

| Symbol | Meaning |
|---|---|
| $`\mathit{is\_open}`$ | `is_open` over $`\mathcal{W}`$ — is this warehouse open? The only integrality in the model |
| $`\mathit{serve}`$ | `serve` over $`\mathcal{W} \times \mathcal{C}`$ — the share of a customer's demand served from a warehouse |

#### Variable domains

**`is_open`**

```math
\mathit{is\_open}_{w} \in \{0, 1\} \qquad \forall\, w \in \mathcal{W}
```

**`serve`**

```math
0 \le \mathit{serve}_{w,c} \le 1 \qquad \forall\, w \in \mathcal{W},\ c \in \mathcal{C}
```

#### Objective

```math
\min \sum_{w \in \mathcal{W}} \mathit{is\_open}_{w} \cdot \mathrm{fixed\_cost}_{w} + \sum_{w \in \mathcal{W},\ c \in \mathcal{C}} \mathit{serve}_{w,c} \cdot \mathrm{serve}^{\mathrm{cost}}_{w,c}
```

#### Subject to

**`every_customer_served`**

```math
\sum_{w \in \mathcal{W}} \mathit{serve}_{w,c} = 1 \qquad \forall\, c \in \mathcal{C}
```

**`only_from_open_warehouses`**

```math
\mathit{serve}_{w,c} - \mathit{is\_open}_{w} \le 0 \qquad \forall\, w \in \mathcal{W},\ c \in \mathcal{C}
```

</details>
<!-- math:end -->

```yaml
description: >-
  Uncapacitated facility location, OR-Library instance cap71: 16 possible
  warehouses, 50 customers. Open a set of warehouses and assign every customer
  to one, trading fixed opening costs against the cost of serving from further
  away. Optimum 932615.750, published by OR-Library.

dimensions:
  warehouse:
    description: sites a warehouse may be opened on
    dtype: str
  customer:
    description: customers, each served in full from one warehouse
    dtype: str

parameters:
  fixed_cost:
    description: what opening a warehouse costs, whoever it ends up serving
    dims: [warehouse]
  serve_cost:
    description: what it costs to serve all of this customer's demand from this warehouse
    dims: [warehouse, customer]

variables:
  is_open:
    description: is this warehouse open? The only integrality in the model
    dims: [warehouse]
    domain: binary
  serve:
    description: the share of a customer's demand served from a warehouse
    dims: [warehouse, customer]
    bounds:
      lower: 0
      upper: 1

constraints:
  every_customer_served:
    description: a customer's demand is met in full, from one warehouse or several
    dims: [customer]
    expression: sum(serve, over=warehouse) == 1

  only_from_open_warehouses:
    description: >-
      a closed warehouse serves nobody, written per pair — the strong
      formulation, because summing it over customers instead would give a valid
      but much weaker relaxation, and the LP bound is what makes this instance
      solve at all. It is also why serve comes out integral on its own, with no
      integrality declared on it.
    dims: [warehouse, customer]
    expression: serve - is_open <= 0

objective:
  sense: minimize
  description: what the open warehouses cost, plus what serving from them costs
  expression: sum(is_open * fixed_cost) + sum(serve * serve_cost)
```

**`serve` is not declared binary.** Only `is_open` carries integrality. `serve`
is free to take fractional values and comes out integral anyway, because the
linking constraint is written **per (warehouse, customer) pair**.

The aggregated form, `sum(serve, over=customer) <= 50 * is_open`, is one row per
warehouse instead of 800. It is equally *valid* and much *weaker*. Its LP
relaxation lets a warehouse open a fiftieth of the way and serve one customer
for a fiftieth of its fixed cost. The per-pair form makes the LP bound tight
enough for the instance to solve at once. The language does not make that choice
for you, and the objective does not show it. With the weak form the answer is
still 932615.750, reached much more slowly.

## What it finds

Eleven of the sixteen warehouses open (`w01`–`w04`, `w06`–`w09`, `w11`–`w13`)
for a total of **932615.75**.

Ten of them cost 7500 to open. **`w11` costs nothing**: `cap71` gives it a fixed
cost of 0, so opening it is never a trade-off. A reader checking the arithmetic
against "7500 apiece" would come up 7500 short.

## What it exercises

`binary` on its own dimension while a second, larger variable stays continuous;
a two-dimensional parameter against a two-dimensional variable; and a
two-term objective mixing a fixed charge with a flow cost.
