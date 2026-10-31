# OSeMOSYS — UTOPIA

What to build and how hard to run it, 1990–2010, to meet three end-use demands at least discounted cost.

> **✔ Verified against OSeMOSYS** — objective **29446.86269**, matched to `rtol=1e-09`. Asserted upstream in `tests/test_gnu_mathprog.py`, and re-run here directly on GLPK.

**The only optimum in this corpus that comes from outside Python.** UTOPIA is
the reference system bundled with MARKAL and the case OSeMOSYS validates itself
against. Its model is GNU MathProg and its solver is GLPK. Neither shares a line
of code, a data model or a language family with anything here.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

OSeMOSYS's UTOPIA: what to build and how hard to run it, 1990-2010, to meet three end-use demands at least discounted cost. The reference system bundled with MARKAL. Discounting, the annuity, salvage value and the operational-life window are arithmetic over years, so they are folded into coefficients before the model is built; what is left is the decision. Optimum 29446.86269, from OSeMOSYS itself under GLPK.

#### Sets

| Symbol | Meaning |
|---|---|
| $`\mathcal{T}`$ | index $`t`$ — `technology` — the plants and processes that may be built and run |
| $`\mathcal{F}`$ | index $`f`$ — `fuel` — energy carriers, produced by one technology and consumed by another |
| $`\mathcal{I}`$ | index $`i`$ — `timeslice` — the slices a year is dispatched over |
| $`\mathcal{M}`$ | index $`m`$ — `mode` — the way a technology is being operated |
| $`\mathcal{Y}`$ | index $`y`$ — `year` — the years the pathway covers |
| $`\mathcal{V}`$ | index $`v`$ — `vintage` — a second axis over the same years — capacity standing in a year was built in some vintage |

#### Parameters

| Symbol | Meaning |
|---|---|
| $`\mathrm{still\_live}`$ | `still_live` over $`\mathcal{T} \times \mathcal{Y} \times \mathcal{V}`$, `neutral` where the data has no row — 1 where a vintage is still inside its technology's operational life in that year |
| $`\mathrm{residual\_capacity}`$ | `residual_capacity` over $`\mathcal{T} \times \mathcal{Y}`$, `neutral` where the data has no row — capacity that already stood in 1990 and has not yet retired |
| $`\mathrm{build}^{\mathrm{cost}}`$ | `build_cost` over $`\mathcal{T} \times \mathcal{V}`$, `neutral` where the data has no row — discounted cost of building a unit of capacity in a vintage |
| $`\mathrm{holding\_cost}`$ | `holding_cost` over $`\mathcal{T} \times \mathcal{Y}`$, `neutral` where the data has no row — discounted fixed cost of holding a unit of capacity through a year |
| $`\mathrm{running\_cost}`$ | `running_cost` over $`\mathcal{I} \times \mathcal{T} \times \mathcal{M} \times \mathcal{Y}`$ — discounted variable cost of a unit of activity |
| $`\mathrm{year\_split}`$ | `year_split` over $`\mathcal{I} \times \mathcal{Y}`$ — share of the year a timeslice stands for |
| $`\mathrm{capacity}^{\mathrm{available}}`$ | `capacity_available` over $`\mathcal{T} \times \mathcal{I} \times \mathcal{Y}`$ — share of its capacity a technology can offer in a timeslice |
| $`\mathrm{input\_ratio}`$ | `input_ratio` over $`\mathcal{T} \times \mathcal{F} \times \mathcal{M} \times \mathcal{Y}`$, `neutral` where the data has no row — fuel a technology consumes per unit of activity |
| $`\mathrm{output\_ratio}`$ | `output_ratio` over $`\mathcal{T} \times \mathcal{F} \times \mathcal{M} \times \mathcal{Y}`$, `neutral` where the data has no row — fuel a technology produces per unit of activity |
| $`\mathrm{sliced\_demand}`$ | `sliced_demand` over $`\mathcal{F} \times \mathcal{I} \times \mathcal{Y}`$, `neutral` where the data has no row — demand for a fuel placed on one timeslice |
| $`\mathrm{annual\_demand}`$ | `annual_demand` over $`\mathcal{F} \times \mathcal{Y}`$, `neutral` where the data has no row — demand for a fuel placed on the year as a whole |
| $`\mathrm{max\_capacity}`$ | `max_capacity` over $`\mathcal{T} \times \mathcal{Y}`$, `absent` where the data has no row — most capacity a technology may stand at |
| $`\mathrm{min\_capacity}`$ | `min_capacity` over $`\mathcal{T} \times \mathcal{Y}`$, `absent` where the data has no row — least capacity a technology must stand at |
| $`\mathrm{reserve\_margin}`$ | `reserve_margin` over $`\mathcal{Y}`$ — how far firm capacity must exceed the demand of the moment |
| $`\mathrm{reserve\_tagged}`$ | `reserve_tagged` over $`\mathcal{T} \times \mathcal{Y}`$, `neutral` where the data has no row — 1 where a technology's capacity counts towards the reserve |
| $`\mathrm{reserve\_demand}`$ | `reserve_demand` over $`\mathcal{T} \times \mathcal{F} \times \mathcal{M} \times \mathcal{Y}`$, `neutral` where the data has no row — the activity the reserve margin is measured against |
| $`\mathrm{residual\_holding}`$ | `residual_holding` (scalar) — fixed operating cost owed on the capacity that already stood in 1990 |

#### Variables

| Symbol | Meaning |
|---|---|
| $`\mathit{activity}`$ | `activity` over $`\mathcal{I} \times \mathcal{T} \times \mathcal{M} \times \mathcal{Y}`$ — how hard a technology runs, per timeslice and mode |
| $`\mathit{build}`$ | `build` over $`\mathcal{T} \times \mathcal{V}`$ — how much capacity is built, and when |

#### Definitions

| Symbol | Meaning |
|---|---|
| $`\mathit{built\_capacity}`$ | `built_capacity` over $`\mathcal{T} \times \mathcal{Y}`$ — capacity standing in a year from every vintage still inside its life. A plant's life is read from data and differs by technology, so the window cannot be a fixed shift — it is an incidence table, the shape the KVL port uses for a cycle basis. |
| $`\mathit{capacity}`$ | `capacity` over $`\mathcal{T} \times \mathcal{Y}`$ — all the capacity standing in a year, including what was already there in 1990 |

#### Variable domains

**`activity`**

```math
\mathit{activity}_{i,t,m,y} \ge 0 \qquad \forall\, i \in \mathcal{I},\ t \in \mathcal{T},\ m \in \mathcal{M},\ y \in \mathcal{Y}
```

**`build`**

```math
\mathit{build}_{t,v} \ge 0 \qquad \forall\, t \in \mathcal{T},\ v \in \mathcal{V}
```

#### Definitions

**`built_capacity`**

```math
\mathit{built\_capacity}_{t,y} = \sum_{v \in \mathcal{V}} \mathit{build}_{t,v} \cdot \mathrm{still\_live}_{t,y,v} \qquad \forall\, t \in \mathcal{T},\ y \in \mathcal{Y}
```

**`capacity`**

```math
\mathit{capacity}_{t,y} = \mathit{built\_capacity}_{t,y} + \mathrm{residual\_capacity}_{t,y} \qquad \forall\, t \in \mathcal{T},\ y \in \mathcal{Y}
```

#### Objective

```math
\min \sum_{t \in \mathcal{T},\ v \in \mathcal{V}} \mathit{build}_{t,v} \cdot \mathrm{build}^{\mathrm{cost}}_{t,v} + \sum_{t \in \mathcal{T},\ y \in \mathcal{Y}} \mathit{built\_capacity}_{t,y} \cdot \mathrm{holding\_cost}_{t,y} + \sum_{t \in \mathcal{T},\ i \in \mathcal{I},\ m \in \mathcal{M},\ y \in \mathcal{Y}} \mathit{activity}_{i,t,m,y} \cdot \mathrm{running\_cost}_{i,t,m,y} + \mathrm{residual\_holding}
```

#### Subject to

**`within_capacity`**

```math
\sum_{m \in \mathcal{M}} \mathit{activity}_{i,t,m,y} \le \mathit{capacity}_{t,y} \cdot \mathrm{capacity}^{\mathrm{available}}_{t,i,y} \qquad \forall\, i \in \mathcal{I},\ t \in \mathcal{T},\ y \in \mathcal{Y}
```

**`fuel_balance`**

```math
\left( \sum_{t \in \mathcal{T}} \sum_{m \in \mathcal{M}} \mathit{activity}_{i,t,m,y} \cdot \mathrm{output\_ratio}_{t,f,m,y} \right) \cdot \mathrm{year\_split}_{i,y} \ge \mathrm{sliced\_demand}_{f,i,y} + \left( \sum_{t \in \mathcal{T}} \sum_{m \in \mathcal{M}} \mathit{activity}_{i,t,m,y} \cdot \mathrm{input\_ratio}_{t,f,m,y} \right) \cdot \mathrm{year\_split}_{i,y} \qquad \forall\, i \in \mathcal{I},\ f \in \mathcal{F},\ y \in \mathcal{Y}
```

**`annual_balance`**

```math
\sum_{i \in \mathcal{I}} \sum_{t \in \mathcal{T}} \sum_{m \in \mathcal{M}} \mathit{activity}_{i,t,m,y} \cdot \mathrm{output\_ratio}_{t,f,m,y} \cdot \mathrm{year\_split}_{i,y} \ge \mathrm{annual\_demand}_{f,y} + \sum_{i \in \mathcal{I}} \sum_{t \in \mathcal{T}} \sum_{m \in \mathcal{M}} \mathit{activity}_{i,t,m,y} \cdot \mathrm{input\_ratio}_{t,f,m,y} \cdot \mathrm{year\_split}_{i,y} \qquad \forall\, f \in \mathcal{F},\ y \in \mathcal{Y}
```

**`capacity_ceiling`**

```math
\mathit{capacity}_{t,y} \le \mathrm{max\_capacity}_{t,y} \qquad \forall\, t \in \mathcal{T},\ y \in \mathcal{Y}
```

**`capacity_floor`**

```math
\mathit{capacity}_{t,y} \ge \mathrm{min\_capacity}_{t,y} \qquad \forall\, t \in \mathcal{T},\ y \in \mathcal{Y}
```

**`reserve`**

```math
\left( \sum_{t \in \mathcal{T}} \sum_{f \in \mathcal{F}} \sum_{m \in \mathcal{M}} \mathit{activity}_{i,t,m,y} \cdot \mathrm{reserve\_demand}_{t,f,m,y} \right) \cdot \mathrm{reserve\_margin}_{y} \le \sum_{t \in \mathcal{T}} \mathit{capacity}_{t,y} \cdot \mathrm{reserve\_tagged}_{t,y} \qquad \forall\, i \in \mathcal{I},\ y \in \mathcal{Y}
```

</details>
<!-- math:end -->

The tabs start from [the instance's tables](../howto/data.md) — one frame per parameter.

=== "specsolve"

    ```yaml
    description: >-
      OSeMOSYS's UTOPIA: what to build and how hard to run it, 1990-2010, to meet
      three end-use demands at least discounted cost. The reference system bundled
      with MARKAL. Discounting, the annuity, salvage value and the operational-life
      window are arithmetic over years, so they are folded into coefficients before
      the model is built; what is left is the decision. Optimum 29446.86269, from
      OSeMOSYS itself under GLPK.

    dimensions:
      technology:
        description: the plants and processes that may be built and run
        dtype: str
      fuel:
        description: energy carriers, produced by one technology and consumed by another
        dtype: str
      timeslice:
        description: the slices a year is dispatched over
        dtype: str
      mode:
        description: the way a technology is being operated
        dtype: int
      year:
        description: the years the pathway covers
        dtype: int
        ordered: true
      vintage:
        description: >-
          a second axis over the same years — capacity standing in a year was built
          in some vintage
        dtype: int
        ordered: true

    parameters:
      still_live:
        description: 1 where a vintage is still inside its technology's operational life in that year
        dims: [technology, year, vintage]
        missing: neutral
      residual_capacity:
        description: capacity that already stood in 1990 and has not yet retired
        dims: [technology, year]
        missing: neutral

      build_cost:
        description: discounted cost of building a unit of capacity in a vintage
        dims: [technology, vintage]
        missing: neutral
      holding_cost:
        description: discounted fixed cost of holding a unit of capacity through a year
        dims: [technology, year]
        missing: neutral
      running_cost:
        description: discounted variable cost of a unit of activity
        dims: [timeslice, technology, mode, year]

      year_split:
        description: share of the year a timeslice stands for
        dims: [timeslice, year]
      capacity_available:
        description: share of its capacity a technology can offer in a timeslice
        dims: [technology, timeslice, year]
      input_ratio:
        description: fuel a technology consumes per unit of activity
        dims: [technology, fuel, mode, year]
        missing: neutral
      output_ratio:
        description: fuel a technology produces per unit of activity
        dims: [technology, fuel, mode, year]
        missing: neutral

      sliced_demand:
        description: demand for a fuel placed on one timeslice
        dims: [fuel, timeslice, year]
        missing: neutral
      annual_demand:
        description: demand for a fuel placed on the year as a whole
        dims: [fuel, year]
        missing: neutral

      max_capacity:
        description: most capacity a technology may stand at
        dims: [technology, year]
        missing: absent
      min_capacity:
        description: least capacity a technology must stand at
        dims: [technology, year]
        missing: absent

      reserve_margin:
        description: how far firm capacity must exceed the demand of the moment
        dims: [year]
      reserve_tagged:
        description: 1 where a technology's capacity counts towards the reserve
        dims: [technology, year]
        missing: neutral
      reserve_demand:
        description: the activity the reserve margin is measured against
        dims: [technology, fuel, mode, year]
        missing: neutral

      residual_holding:
        description: fixed operating cost owed on the capacity that already stood in 1990
        dims: []

    variables:
      activity:
        description: how hard a technology runs, per timeslice and mode
        dims: [timeslice, technology, mode, year]
        bounds:
          lower: 0
      build:
        description: how much capacity is built, and when
        dims: [technology, vintage]
        bounds:
          lower: 0

    expressions:
      built_capacity:
        expression: sum(build * still_live, over=vintage)
        description: >-
          capacity standing in a year from every vintage still inside its life. A
          plant's life is read from data and differs by technology, so the window
          cannot be a fixed shift — it is an incidence table, the shape the KVL port
          uses for a cycle basis.
      capacity:
        expression: built_capacity + residual_capacity
        description: all the capacity standing in a year, including what was already there in 1990

    constraints:
      within_capacity:
        description: a technology cannot run beyond the capacity standing that year
        dims: [timeslice, technology, year]
        expression: sum(activity, over=mode) <= capacity * capacity_available

      fuel_balance:
        description: >-
          every fuel balances in every timeslice — what is produced covers the
          demand placed on it plus what other technologies consume
        dims: [timeslice, fuel, year]
        expression: >-
          sum(sum(activity * output_ratio, over=mode), over=technology) * year_split
          >= sliced_demand
          + sum(sum(activity * input_ratio, over=mode), over=technology) * year_split

      annual_balance:
        description: and balances again over the year, for demands that are not sliced
        dims: [fuel, year]
        expression: >-
          sum(sum(sum(activity * output_ratio * year_split, over=mode), over=technology), over=timeslice)
          >= annual_demand
          + sum(sum(sum(activity * input_ratio * year_split, over=mode), over=technology), over=timeslice)

      capacity_ceiling:
        dims: [technology, year]
        expression: capacity <= max_capacity

      capacity_floor:
        dims: [technology, year]
        expression: capacity >= min_capacity

      reserve:
        description: firm capacity exceeds the electricity demand of the moment by the reserve margin
        dims: [timeslice, year]
        expression: >-
          sum(sum(sum(activity * reserve_demand, over=mode), over=fuel), over=technology) * reserve_margin
          <= sum(capacity * reserve_tagged, over=technology)

    objective:
      sense: minimize
      description: discounted cost of building, holding and running the system over the pathway
      expression: >-
        sum(build * build_cost)
        + sum(built_capacity * holding_cost)
        + sum(activity * running_cost)
        + residual_holding
    ```

## What the port had to decide

**An operational life is a window read from data, and that is an incidence
table.** Capacity standing in a year is every vintage still inside its
technology's life. The lives differ, so this is not a fixed `shift`. The port
carries `still_live[technology, year, vintage]`, one row per pair that is still
live, and the standing capacity is a contraction against it. That is the shape
[`pypsa_kvl`](pypsa_kvl.md) uses for a cycle basis. Building it is arithmetic
over years, which
[the limits](https://mathspec.readthedocs.io/en/latest/about/limits/#what-counts-as-data-preparation)
count as data preparation.

**Discounting, the annuity and salvage value never reach the model.** OSeMOSYS
spends four parameters and four constraint families on them:
`CapitalRecoveryFactor`, `PvAnnuity`, `DiscountFactor`, and `SV1`–`SV4` with
three depreciation cases. Every one is a function of the year and the
technology, so all of it folds into a single coefficient per `(technology,
vintage)`. What survives into the model is the decision: how much to build, and
when.

The one piece that cannot fold is the fixed cost owed on capacity that already
stood in 1990. It is owed whatever the model chooses, so it enters the objective
as a constant.

## Same answer, a twenty-third of the model

| | rows | columns |
|---|---|---|
| OSeMOSYS, as generated by GLPK | 119,273 | 147,171 |
| this port | 5,124 | 5,733 |

OSeMOSYS's long formulation defines an intermediate *variable* for every
accounting quantity and ties each to its definition with an equality.
`RateOfProductionByTechnologyByMode` alone is one per region × timeslice ×
technology × mode × fuel × year. Those are names for expressions, not decisions,
and writing the model as expressions substitutes them. **Both reach
29446.86269**, so the substitution is exact.

## What this port does not carry

**Storage.** UTOPIA declares a reservoir and OSeMOSYS carries fifteen
constraints for it, but the instance builds none. `NewStorageCapacity` is empty
in the reference solution, as is `Trade`. Their constraints are satisfied at
zero, so the port omits them. The optimum agreeing to ten digits says the
omission was safe.

`Conversionls`, `Conversionld` and `Conversionlh`, the three maps from a
timeslice to its season, day type and daily time bracket, appear **only** in
those storage constraints. With storage inert they read nothing, so this port is
no evidence about grouping one axis several ways.

## What it exercises

A window read from data as an incidence table; a second axis over the same
years joined to the first; and a cost chain that is entirely data preparation.
No construct here is new.
