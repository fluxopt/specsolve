# Travelling salesman — MTZ

Visit every city once and come home, as cheaply as possible. The most famous problem in combinatorial optimisation, and the one most often assumed to be out of reach here.

> **✔ Verified against TSPLIB's published optimum** — **2085**, matched to `rtol=1e-09`. Instance `gr17` (Groetschel, 17 cities, explicit distance matrix).

**It is not out of reach.** Miller–Tucker–Zemlin (MTZ) is inside the language.
Only lazy subtour generation is outside it.

## What genuinely is refused, and why

TSP's textbook formulation, Dantzig–Fulkerson–Johnson (DFJ), forbids subtours
with one constraint per subset of cities:

$$\sum_{i \in S}\sum_{j \in S} x_{ij} \le |S| - 1 \qquad \text{for every subset } S$$

Every row is linear and there are finitely many, so DFJ is an ordinary MILP.
Which parts of it the language can say:

| | Status |
|---|---|
| DFJ, subsets **written out** | **sayable** — the subsets go in as data, exactly as [KVL's cycle basis](pypsa_kvl.md) does. Verified on an 8-city instance: 246 subsets, one correct tour |
| DFJ, subsets **generated lazily** | **outside** — solve, find violations, add rows, re-solve is an *algorithm*, not a model. Nothing declarative describes it |
| MTZ | sayable, and what this port uses |

The refusal is only the second row. The first is not a language limit: it is 2ⁿ
rows, which stops being practical somewhere around twenty cities. A
data-dependent *row count* is not itself a refusal, since the
[cycle basis](pypsa_kvl.md) has one and is ordinary. What rules DFJ out at scale
is the size of the data, not the shape of the language.

**Lazy generation is what every serious TSP code does.** specsolve can express TSP.
It is not a good way to solve a large one.

## What that leaves

MTZ, the polynomial alternative: give each city a position in the tour and
require that an arc `i → j` puts `j` later than `i`. O(n²) rows, every one known
before the data is read: static, relational, degree 1. Inside the language.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

The travelling salesman problem in the Miller-Tucker-Zemlin formulation: visit every city once and come home as cheaply as possible. TSPLIB instance gr17 — 17 cities, explicit distance matrix, published optimum 2085.

#### Sets

| Symbol | Meaning |
|---|---|
| $`\mathcal{C}`$ | index $`c`$ — `city` with $`\mathrm{as\_from}: \mathcal{C} \to \mathcal{F},\ \mathrm{as\_to}: \mathcal{C} \to \mathcal{T}`$ — the cities of the tour, each also read as an arc endpoint |
| $`\mathcal{F}`$ | index $`f`$ — `from_city` with $`\mathrm{as\_from}: \mathcal{C} \to \mathcal{F}`$ — the city an arc leaves |
| $`\mathcal{T}`$ | index $`t`$ — `to_city` with $`\mathrm{as\_to}: \mathcal{C} \to \mathcal{T}`$ — the city an arc arrives at |

#### Parameters

| Symbol | Meaning |
|---|---|
| $`\mathrm{distance}`$ | `distance` over $`\mathcal{F} \times \mathcal{T}`$, `absent` where the data has no row — distance along an arc, with no row on the diagonal — a city has no distance to itself, so no arc variable exists there |
| $`\mathrm{n}`$ | `n` (scalar) — the number of cities, which is the big-M the ordering rows need |

#### Variables

| Symbol | Meaning |
|---|---|
| $`\mathit{travel}`$ | `travel` over $`\mathcal{F} \times \mathcal{T}`$ — is this arc on the tour? |
| $`u`$ | `u` over $`\mathcal{C}`$ — position of a city in the tour — continuous, because the formulation needs only that the positions be orderable |

#### Variable domains

**`travel`**

```math
\mathit{travel}_{f,t} \in \{0, 1\} \qquad \forall\, f \in \mathcal{F},\ t \in \mathcal{T} \,:\, \mathrm{distance}_{f,t} \text{ is defined}
```

**`u`**

```math
1 \le u_{c} \le 17 \qquad \forall\, c \in \mathcal{C}
```

#### Objective

```math
\min \sum_{f \in \mathcal{F},\ t \in \mathcal{T}} \mathit{travel}_{f,t} \cdot \mathrm{distance}_{f,t}
```

#### Subject to

**`leave_each_city_once`**

```math
\sum_{t \in \mathcal{T}} \mathit{travel}_{f,t} = 1 \qquad \forall\, f \in \mathcal{F}
```

**`enter_each_city_once`**

```math
\sum_{f \in \mathcal{F}} \mathit{travel}_{f,t} = 1 \qquad \forall\, t \in \mathcal{T}
```

**`ordering`**

```math
\sum_{c \in \mathcal{C} \,:\, \mathrm{as\_from}(c) = f} u_{c} - \left( \sum_{c \in \mathcal{C} \,:\, \mathrm{as\_to}(c) = t} u_{c} \right) + \mathrm{n} \cdot \mathit{travel}_{f,t} \le \mathrm{n} - 1 \qquad \forall\, f \in \mathcal{F},\ t \in \mathcal{T} \,:\, f \neq \text{'}\mathrm{c01}\text{'} \wedge t \neq \text{'}\mathrm{c01}\text{'}
```

</details>
<!-- math:end -->

```yaml
description: >-
  The travelling salesman problem in the Miller-Tucker-Zemlin formulation:
  visit every city once and come home as cheaply as possible. TSPLIB instance
  gr17 — 17 cities, explicit distance matrix, published optimum 2085.

dimensions:
  city:
    description: the cities of the tour, each also read as an arc endpoint
    dtype: str
  from_city:
    description: the city an arc leaves
    dtype: str
  to_city:
    description: the city an arc arrives at
    dtype: str

relations:
  as_from: {key: city, values: from_city}
  as_to: {key: city, values: to_city}

parameters:
  distance:
    description: >-
      distance along an arc, with no row on the diagonal — a city has no
      distance to itself, so no arc variable exists there
    dims: [from_city, to_city]
    missing: absent
  n:
    description: the number of cities, which is the big-M the ordering rows need
    dims: []

variables:
  travel:
    description: is this arc on the tour?
    dims: [from_city, to_city]
    where: distance
    domain: binary
  u:
    description: >-
      position of a city in the tour — continuous, because the formulation
      needs only that the positions be orderable
    dims: [city]
    bounds:
      lower: 1
      upper: 17

constraints:
  leave_each_city_once:
    dims: [from_city]
    expression: sum(travel, over=to_city) == 1

  enter_each_city_once:
    dims: [to_city]
    expression: sum(travel, over=from_city) == 1

  ordering:
    description: >-
      no subtours — if the tour goes from one city to another then the second
      is later in the numbering, and the big-M leaves the row saying nothing
      when it does not. Written for every ordered pair except those touching
      the depot, which anchors the numbering.
    dims: [from_city, to_city]
    where: "from_city != c01 AND to_city != c01"
    expression: >-
      sum(u, over=city, by=as_from[from_city])
      - sum(u, over=city, by=as_to[to_city])
      + n * travel
      <= n - 1

objective:
  sense: minimize
  description: the length of the tour
  expression: sum(travel * distance)
```

**The tour position sits at both ends of one row.** MTZ needs `u`, the tour
position, at *both ends of the same row*: `u_i − u_j`. A variable indexed by one
dimension appears twice under two different roles, a self-join that looks like
it should need a primitive.

It does not. Declare the identity map from `city` onto each end of the pair:

```yaml
relations:
  as_from: {key: city, values: from_city}
  as_to: {key: city, values: to_city}
```

and `sum(u, over=city, by=as_from[from_city])` becomes a **relabel** rather than a reduction. Each
city keys one row and each `from_city` is named once, so nothing is added up:
`u` moves from the `city` axis onto the `from_city` axis. Doing it twice with
different relations puts the same variable at both ends of one row. A relation
is a join, and a join does not care how many rows a group holds
([topology is data](pypsa_transport.md)).

**The diagonal takes care of itself.** `distance` has no row where a city meets
itself, `travel`'s `where` is that parameter, and absence spreads, so no row
mentioning a self-arc is built. No `i ≠ j` guard is written anywhere.
[Dimension-to-dimension comparison is not in the
language](https://mathspec.readthedocs.io/en/latest/reference/language/expressions/#where-strings),
and here it is not needed.

## What it finds

A single tour of all 17 cities, closing at the start:

```
c01 → c16 → c12 → c09 → c05 → c02 → c10 → c11 → c03
    → c15 → c14 → c17 → c06 → c08 → c07 → c13 → c04 → c01
```

Length **2085**, TSPLIB's published optimum. One tour, not several, which is
what MTZ guarantees. Check that on the primal rather than trusting the
objective, because a solution with subtours would be *cheaper*.

It solves in about two and a half seconds. MTZ's LP relaxation is weak, which is
the price of a small formulation and why nobody solves large instances this way.

## What it exercises

`sum(by=)` as a relabel through a relation that names each label once, a `where`
comparing a dimension against a string label, sparsity standing in for an
`i ≠ j` guard, and `binary` over a two-dimensional index.

No new construct. The ceiling refuses an *algorithm*, not a *problem*.
