# multi_period

Capacity decided once per investment period, binding at every snapshot inside
it — and the periods need not be the same size.

> **✔ Agrees with hand-written linopy 0.10.0** — objective **10020**, matched to `rtol=1e-09`.

## The problem

$$p_{t,g} \quad\le\quad \hat p_{\thinspace\mathrm{period}(t),\thinspace g}$$

Two dimensions cannot state this at the resolution a real study wants.
`period × snapshot` is a **rectangle**, so every period gets the same number of
snapshots. A study that models 2030 hourly and 2050 in four-hour blocks wants
the opposite.

So `snapshot` is one flat dimension carrying $\mathrm{period}$ as a
[relation](https://mathspec.readthedocs.io/en/latest/reference/language/relations/), as `generator`
carries $\mathrm{bus}$ in [transport](transport.md). Ragged periods then cost
nothing: a relation is a per-row column, and four snapshots in 2030 beside two in
2050 is a column with four of one value and two of another.

## Both directions of one mapping

Grouping reads the relation one way:
`sum(p, over=snapshot, by=period_of[period])` is a per-period CO₂ budget, and
[monthly_budget](monthly_budget.md) is the same construct on a different
relation.

`within_cap` reads it the other way. Capacity lives on `period` and binds at
each `snapshot`, so a coarse quantity is pulled onto a fine one:

```yaml
within_cap:
  dims: [snapshot, generator]
  expression: p <= at(p_nom, by=period_of[period])
```

`at` and `sum(by=)` take the same argument: the relation names one table, and
the operator says which direction it is walked.

A per-period **parameter** needs neither: join it onto the snapshot index
before the model sees it. `p_nom` is a **variable**, which no join can reach,
so the pullback is a construct in the language.

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

Least-cost investment and dispatch together: capacity is decided once per period and binds at every snapshot inside it, and a snapshot's weight says how much time it stands for, so periods of different size are comparable.

#### Sets

| Symbol | Meaning |
|---|---|
| $`\mathcal{T}`$ | index $`t`$ — `snapshot` with $`\mathrm{period\_of}: \mathcal{T} \to \mathcal{E}`$ — dispatch periods, each falling in one investment period |
| $`\mathcal{E}`$ | index $`e`$ — `period` with $`\mathrm{period\_of}: \mathcal{T} \to \mathcal{E}`$ — investment periods, the grouping capacity is decided over |
| $`\mathcal{G}`$ | index $`g`$ — `generator` — generating units |

#### Parameters

| Symbol | Meaning |
|---|---|
| $`\mathrm{load}`$ | `load` over $`\mathcal{T}`$ — demand to be met |
| $`\mathrm{weight}`$ | `weight` over $`\mathcal{T}`$ — what one snapshot stands for — a 2050 snapshot represents four hours, so the operating cost of a coarse period is not understated against a fine one |
| $`\mathrm{opex}`$ | `opex` over $`\mathcal{G}`$ — cost of running a generator for one snapshot-hour |
| $`\mathrm{capex}`$ | `capex` over $`\mathcal{G} \times \mathcal{E}`$ — cost of holding a unit of capacity through a period |

#### Variables

| Symbol | Meaning |
|---|---|
| $`p`$ | `p` over $`\mathcal{T} \times \mathcal{G}`$ — output of a generator in a snapshot |
| $`p^{\mathrm{nom}}`$ | `p_nom` over $`\mathcal{E} \times \mathcal{G}`$ — capacity a generator holds for the whole of a period |

#### Variable domains

**`p`**

```math
p_{t,g} \ge 0 \qquad \forall\, t \in \mathcal{T},\ g \in \mathcal{G}
```

**`p_nom`**

```math
0 \le p^{\mathrm{nom}}_{e,g} \le 100 \qquad \forall\, e \in \mathcal{E},\ g \in \mathcal{G}
```

#### Objective

```math
\min \sum_{t \in \mathcal{T},\ g \in \mathcal{G}} p_{t,g} \cdot \mathrm{opex}_{g} \cdot \mathrm{weight}_{t} + \sum_{e \in \mathcal{E},\ g \in \mathcal{G}} p^{\mathrm{nom}}_{e,g} \cdot \mathrm{capex}_{g,e}
```

#### Subject to

**`within_cap`**

```math
p_{t,g} \le p^{\mathrm{nom}}_{\mathrm{period\_of}(t),g} \qquad \forall\, t \in \mathcal{T},\ g \in \mathcal{G}
```

**`balance`**

```math
\sum_{g \in \mathcal{G}} p_{t,g} = \mathrm{load}_{t} \qquad \forall\, t \in \mathcal{T}
```

</details>
<!-- math:end -->

The tabs start from [the instance's tables](../howto/data.md) — one frame per parameter.

=== "specsolve"

    ```yaml
    description: >-
      Least-cost investment and dispatch together: capacity is decided once per
      period and binds at every snapshot inside it, and a snapshot's weight says
      how much time it stands for, so periods of different size are comparable.

    dimensions:
      snapshot:
        description: dispatch periods, each falling in one investment period
        dtype: int
        ordered: true
      period:
        description: investment periods, the grouping capacity is decided over
        dtype: int
        ordered: true
      generator:
        description: generating units
        dtype: str

    relations:
      period_of: {key: snapshot, values: period}

    parameters:
      load:
        description: demand to be met
        dims: [snapshot]
      weight:
        description: >-
          what one snapshot stands for — a 2050 snapshot represents four hours, so
          the operating cost of a coarse period is not understated against a fine
          one
        dims: [snapshot]
      opex:
        description: cost of running a generator for one snapshot-hour
        dims: [generator]
      capex:
        description: cost of holding a unit of capacity through a period
        dims: [generator, period]

    variables:
      p:
        description: output of a generator in a snapshot
        dims: [snapshot, generator]
        bounds:
          lower: 0
      p_nom:
        description: capacity a generator holds for the whole of a period
        dims: [period, generator]
        bounds:
          lower: 0
          upper: 100

    constraints:
      within_cap:
        description: output in a snapshot is capped by the capacity of the period it falls in
        dims: [snapshot, generator]
        expression: p <= at(p_nom, by=period_of[period])
      balance:
        dims: [snapshot]
        expression: sum(p, over=generator) == load

    objective:
      sense: minimize
      description: weighted operating cost over the horizon, plus what the capacity costs to hold
      expression: sum(p * opex * weight) + sum(p_nom * capex)
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with sps.solve('examples/multi_period.yaml', sources) as solution:
        solution.objective  # 10020.0
        solution.dual('balance')
    ```

=== "linopy"

    The model-building half of `examples/ports/references/linopy/multi_period.py`:

    ```python
    def build(tables: dict[str, pd.DataFrame]) -> linopy.Model:
        """The instance's tables as a linopy model, row for row.

        ``tables`` is the same mapping the specsolve call attaches as ``sources``.
        """
        load: pd.Series = tables['load'].set_index('snapshot')['value']
        weight: pd.Series = tables['weight'].set_index('snapshot')['value']
        opex: pd.Series = tables['opex'].set_index('generator')['value']
        capex = xr.DataArray(tables['capex'].pivot(index='period', columns='generator', values='value'))
        period = xr.DataArray(tables['period_of'].set_index('snapshot')['period'])

        m = linopy.Model()
        p = m.add_variables(lower=0, coords=[load.index, opex.index], name='p')
        p_nom = m.add_variables(lower=0, upper=100, coords=[capex.indexes['period'], opex.index], name='p_nom')
        m.add_constraints(p <= p_nom.sel(period=period), name='within_cap')
        m.add_constraints(p.sum('generator') == load, name='balance')
        m.add_objective((p * opex * weight).sum() + (p_nom * capex).sum())
        return m
    ```

## Reading the answer

On the smaller instance `tests/test_at.py` holds, costs are chosen so each
period picks a different technology, which makes the per-period capacity
visible:

| period | wind | gas |
|---|---|---|
| 2030 | 20 | 10 |
| 2050 | 60 | 0 |

2030 peaks at 30 and splits the build: wind is dearer to install but free to
run. 2050 peaks at 60 with every snapshot weighted four times, so the operating
term dominates and the whole build goes to wind. Objective **750.0**, agreed
integer for integer by both lanes.

The weights make the two periods comparable: a coarse snapshot standing for
four hours contributes four hours of operating cost, so a period is not made
cheap by being modelled coarsely.
