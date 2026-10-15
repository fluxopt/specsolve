# monthly_budget

A cap on what each technology may generate in each calendar month, with the months read off the timestamps.

> **✔ Agrees with hand-written linopy 0.10.0** — objective **9500**, matched to `rtol=1e-09`.

## The problem

$$\sum_{t \thinspace:\thinspace \mathrm{month}(t) = m} p_{t,g} \quad\le\quad \bar E_{m,g}$$

$\mathrm{month}$ is a **coordinate the snapshot dimension declares**, not a
calendar the language understands. Its values arrive as a column in the
snapshot index.

Compare [transport](transport.md): there, $\mathrm{gen\_bus}$ is a relation over
`generator` and the sum is over generators at a bus. Here $\mathrm{month\_of}$ is a
relation over `snapshot` and the sum is over snapshots in a month. **It is the
same construct** — `sum(by=)` — and time is not a special axis.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

A cap on what each technology may generate per calendar month — an aggregate over a coarser grouping of time than the model is dispatched on.

#### Sets

| Symbol | Meaning |
|---|---|
| $`\mathcal{T}`$ | index $`t`$ — `snapshot` with $`\mathrm{month\_of}: \mathcal{T} \to \mathcal{M}`$ — dispatch periods, each falling in one month |
| $`\mathcal{M}`$ | index $`m`$ — `month` with $`\mathrm{month\_of}: \mathcal{T} \to \mathcal{M}`$ — the grouping the budget is stated over |
| $`\mathcal{G}`$ | index $`g`$ — `generator` — generating units |

#### Parameters

| Symbol | Meaning |
|---|---|
| $`\bar p`$ | `p_max` over $`\mathcal{G}`$ — installed capacity |
| $`c`$ | `cost` over $`\mathcal{G}`$ — marginal cost |
| $`\ell`$ | `load` over $`\mathcal{T}`$ — demand to be met |
| $`\bar E`$ | `monthly_cap` over $`\mathcal{M} \times \mathcal{G}`$ — the budget the group sum is checked against, one per month and technology |

#### Variables

| Symbol | Meaning |
|---|---|
| $`p`$ | `p` over $`\mathcal{T} \times \mathcal{G}`$ — output of a generator in a snapshot |

#### Variable domains

**`p`**

```math
0 \le p_{t,g} \le \bar p_{g} \qquad \forall\, t \in \mathcal{T},\ g \in \mathcal{G}
```

#### Objective

```math
\min \sum_{t \in \mathcal{T},\ g \in \mathcal{G}} p_{t,g} \cdot c_{g}
```

#### Subject to

**`balance`**

```math
\sum_{g \in \mathcal{G}} p_{t,g} = \ell_{t} \qquad \forall\, t \in \mathcal{T}
```

**`monthly_budget`**

```math
\sum_{t \in \mathcal{T} \,:\, \mathrm{month\_of}(t) = m} p_{t,g} \le \bar E_{m,g} \qquad \forall\, m \in \mathcal{M},\ g \in \mathcal{G}
```

</details>
<!-- math:end -->

The tabs start from [the instance's tables](../howto/data.md) — one frame per parameter.

=== "specsolve"

    ```yaml
    description: >-
      A cap on what each technology may generate per calendar month — an aggregate
      over a coarser grouping of time than the model is dispatched on.

    dimensions:
      snapshot:
        description: dispatch periods, each falling in one month
        dtype: datetime
        ordered: true
      month:
        description: the grouping the budget is stated over
        dtype: str
        ordered: true
      generator:
        description: generating units
        dtype: str

    relations:
      month_of:
        description: the month a snapshot falls in
        key: snapshot
        values: month

    parameters:
      p_max:
        description: installed capacity
        dims: [generator]
      cost:
        description: marginal cost
        dims: [generator]
      load:
        description: demand to be met
        dims: [snapshot]
      monthly_cap:
        description: the budget the group sum is checked against, one per month and technology
        dims: [month, generator]

    variables:
      p:
        description: output of a generator in a snapshot
        dims: [snapshot, generator]
        bounds:
          lower: 0
          upper: p_max

    constraints:
      balance:
        dims: [snapshot]
        expression: sum(p, over=generator) == load
      monthly_budget:
        description: what a generator produces across a month stays inside that month's budget
        dims: [month, generator]
        expression: sum(p, over=snapshot, by=month_of[month]) <= monthly_cap

    objective:
      sense: minimize
      description: total cost of generation over the horizon
      expression: sum(p * cost)
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with sps.solve('examples/monthly_budget.yaml', sources) as solution:
        solution.objective  # 9500.0
        solution.dual('monthly_budget')
    ```

=== "linopy"

    The model-building half of `examples/ports/references/linopy/monthly_budget.py`:

    ```python
    def build(tables: dict[str, pd.DataFrame]) -> linopy.Model:
        """The instance's tables as a linopy model, row for row.

        ``tables`` is the same mapping the specsolve call attaches as ``sources``.
        """
        p_max: pd.Series = tables['p_max'].set_index('generator')['value']
        cost: pd.Series = tables['cost'].set_index('generator')['value']
        load: pd.Series = tables['load'].set_index('snapshot')['value']
        cap = xr.DataArray(tables['monthly_cap'].pivot(index='month', columns='generator', values='value'))
        month = xr.DataArray(tables['month_of'].set_index('snapshot')['month'])

        m = linopy.Model()
        p = m.add_variables(lower=0, upper=p_max, coords=[load.index, p_max.index], name='p')
        m.add_constraints(p.sum('generator') == load, name='balance')
        m.add_constraints(p.groupby(month).sum() <= cap, name='monthly_budget')
        m.add_objective((p * cost).sum())
        return m
    ```

## The grouping is data

You produce the `month_of` relation before the model, by whatever rule you want:

```python
month_of = pl.DataFrame({'snapshot': hours}).with_columns(pl.col('snapshot').dt.strftime('%Y-%m').alias('month'))
```

That produces the table the model attaches under `month_of`: every snapshot
beside the month it falls in, and nothing else:

```text
shape: (6, 2)
┌─────────────────────┬─────────┐
│ snapshot            ┆ month   │
│ ---                 ┆ ---     │
│ datetime[μs]        ┆ str     │
╞═════════════════════╪═════════╡
│ 2030-01-01 00:00:00 ┆ 2030-01 │
│ 2030-01-16 00:00:00 ┆ 2030-01 │
│ 2030-01-31 00:00:00 ┆ 2030-01 │
│ 2030-02-15 00:00:00 ┆ 2030-02 │
│ 2030-03-02 00:00:00 ┆ 2030-03 │
│ 2030-03-17 00:00:00 ┆ 2030-03 │
└─────────────────────┴─────────┘
```

Three snapshots in January, one in February, two in March: `sum(by=)` needs a
partition, not equal groups.

That one expression is the only place a calendar appears. Swap it for
`dt.quarter()`, a fiscal-year relation, a hand-built table of representative
periods or peak/off-peak blocks, and the model is unchanged. The language has
no `resample:` or `reduce_to_monthly()`: a relation covers all of those cases.

## Reading it back

The budget's dual is a shadow price per month and technology, so a binding cap
says what relaxing it would be worth:

```text
month     generator   dual
2030-01   wind       -49.0   ← binding: displacing gas (50) with wind (1)
2030-02   wind        -0.0
2030-03   wind        -0.0
```

Per-month *results* need no language support: a primal is a tidy frame, so a
join and a `group_by` do it:

```python
sol.primal('p').join(index, on='snapshot').group_by('month').agg(pl.col('value').sum())
```

