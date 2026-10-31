# seasons

A store that cycles inside each season rather than across the horizon, with seasons of different lengths.

## The problem

A store that must come back to where it started needs its first position to read
its last. `edge='wrap'` says that about **the axis**:

```yaml
soc == shift(soc, along=snapshot, offset=1, edge='wrap') + inflow - release
```

On this instance that links snapshot 7 to snapshot 1 and makes the whole
horizon one cycle: winter opens holding what summer left, and sells it at
winter's best price. That is a different model. It solves, and its objective
is *higher*.

The cycle a multi-period model means is per period, and `by=` says which:

```yaml
soc == shift(soc, along=snapshot, offset=1, edge='wrap', within=season_of[season]) + inflow - release
```

$$\mathit{soc}_{t} = \mathit{soc}_{t \ominus_{\mathrm{season\_of}(t)} 1} + \mathit{inflow}_{t} - \mathit{release}_{t}$$

The translation walks inside the group the relation makes, so a season's first
snapshot reads that season's last, whatever the season's length. Winter has
four snapshots and summer three here, and nothing in the file says so.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

A store that cycles inside each season rather than across the horizon, with seasons of different lengths — one balance row says it, because the wrap is the season's own and no level is carried from one season into the next.

#### Sets

| Symbol | Meaning |
|---|---|
| $`\mathcal{T}`$ | index $`t`$ — `snapshot` with $`\mathrm{season\_of}: \mathcal{T} \to \mathcal{S}`$ — dispatch periods in order |
| $`\mathcal{S}`$ | index $`s`$ — `season` with $`\mathrm{season\_of}: \mathcal{T} \to \mathcal{S}`$ — the blocks the store cycles over |

#### Parameters

| Symbol | Meaning |
|---|---|
| $`\mathrm{inflow}`$ | `inflow` over $`\mathcal{T}`$ — energy arriving in a snapshot, whether or not it is wanted then |
| $`\mathrm{price}`$ | `price` over $`\mathcal{T}`$ — what one unit of release earns in a snapshot |

#### Variables

| Symbol | Meaning |
|---|---|
| $`\mathit{soc}`$ | `soc` over $`\mathcal{T}`$ — energy held at the end of a snapshot |
| $`\mathit{release}`$ | `release` over $`\mathcal{T}`$ — energy released in a snapshot |

#### Notation

| Symbol | Meaning |
|---|---|
| $`t \ominus k`$ | `edge='wrap'` — $`t-k`$, cyclic |
| $`t \ominus^{\mathrm{relation}(t)} k`$ | `within=relation[...]` — $`t-k`$ inside the group of $`t`$ |

#### Variable domains

**`soc`**

```math
0 \le \mathit{soc}_{t} \le 60 \qquad \forall\, t \in \mathcal{T}
```

**`release`**

```math
\mathit{release}_{t} \ge 0 \qquad \forall\, t \in \mathcal{T}
```

#### Objective

```math
\max \sum_{t \in \mathcal{T}} \mathit{release}_{t} \cdot \mathrm{price}_{t}
```

#### Subject to

**`season_balance`**

```math
\mathit{soc}_{t} = \mathit{soc}_{t \ominus^{\mathrm{season\_of}(t)} 1} + \mathrm{inflow}_{t} - \mathit{release}_{t} \qquad \forall\, t \in \mathcal{T}
```

</details>
<!-- math:end -->

The tabs start from [the instance's tables](../howto/data.md) — one frame per parameter.

=== "specsolve"

    ```yaml
    description: >-
      A store that cycles inside each season rather than across the horizon, with
      seasons of different lengths — one balance row says it, because the wrap is
      the season's own and no level is carried from one season into the next.

    dimensions:
      snapshot:
        description: dispatch periods in order
        dtype: int
        ordered: true
      season:
        description: the blocks the store cycles over
        dtype: str
        ordered: true

    relations:
      season_of:
        description: the season a snapshot falls in
        key: snapshot
        values: season

    parameters:
      inflow:
        description: energy arriving in a snapshot, whether or not it is wanted then
        dims: [snapshot]
      price:
        description: what one unit of release earns in a snapshot
        dims: [snapshot]

    variables:
      soc:
        description: energy held at the end of a snapshot
        dims: [snapshot]
        bounds:
          lower: 0
          upper: 60
      release:
        description: energy released in a snapshot
        dims: [snapshot]
        bounds:
          lower: 0

    constraints:
      season_balance:
        description: >-
          the level carried into a snapshot is the previous snapshot's, and a
          season's first snapshot carries from that season's own last — so each
          season ends where it began and hands the next one nothing
        dims: [snapshot]
        expression: soc == shift(soc, along=snapshot, offset=1, edge='wrap', within=season_of[season]) + inflow - release

    objective:
      sense: maximize
      description: revenue from what the store releases
      expression: sum(release * price)
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with sps.solve('examples/seasons.yaml', sources) as solution:
        solution.objective  # 74.0
        solution.primal('soc')
    ```

## What the answer looks like

```text
snapshot  season  price  inflow  release  soc
1         winter  1      0       0        0
2         winter  2      10      0        10
3         winter  5      0       10       0    ← winter's inflow, at winter's best price
4         winter  3      0       0        0    ← closes where it opened
5         summer  4      0       6        0    ← sells *before* its inflow arrives
6         summer  1      6       0        6
7         summer  2      0       0        6    ← closes where it opened, three snapshots later
```

Objective **74.0**. Summer opens holding 6, sells it at the price-4 snapshot
before its own inflow arrives, and the inflow at snapshot 6 puts the 6 back, so
the season closes where it opened. Nothing constrains the level a season
starts at, except that it must return to it.

The same instance written against the axis gives **80.0**: the extra 6 comes
out of winter, which had it only because summer's closing level leaked across
the boundary.

## The grouping is data

`season_of` is a table passed beside the snapshot index, so the same file
expresses maintenance campaigns, representative days, market quarters or
contract windows. Move two snapshots between seasons by editing that table and
no clause changes:

```text
shape: (7, 2)
┌──────────┬────────┐
│ snapshot ┆ season │
│ ---      ┆ ---    │
│ i64      ┆ str    │
╞══════════╪════════╡
│ 1        ┆ winter │
│ 2        ┆ winter │
│ 3        ┆ winter │
│ 4        ┆ winter │
│ 5        ┆ summer │
│ 6        ┆ summer │
│ 7        ┆ summer │
└──────────┴────────┘
```

A snapshot the table has no row for belongs to no season, so it reaches nothing
and its row is not built. Absence reads the same way in
[`sum(by=)`](https://mathspec.readthedocs.io/en/latest/reference/language/operators/).

Compare [monthly budget](monthly_budget.md), where such a column groups a *sum*,
and [multi-period](multi_period.md), where it carries a capacity decision down
onto the snapshots it covers. One relation, three jobs.
