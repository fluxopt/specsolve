# storage

Dispatch plus a battery that carries its charge from one snapshot to the next, closed into a cycle, so it ends where it started.

> **✔ Agrees with hand-written linopy 0.10.0** — objective **5650**, matched to `rtol=1e-09`.

## The problem

State of charge links each snapshot to the one before it, cyclically:

$$\mathrm{soc}_s = \mathrm{soc}_{s-1} + 0.9\,\mathrm{charge}_s - \mathrm{discharge}_s$$

with $s-1$ wrapping from the first snapshot to the last, so the battery ends
where it started. The model writes the charging efficiency as the literal
`0.9` rather than declaring a parameter, so it appears here as a number and
not as an $\eta$.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

Dispatch plus a battery whose state of charge is closed into a cycle: the level ends where it began.

#### Sets

| Symbol | Meaning |
|---|---|
| $`\mathcal{S}`$ | index $`s`$ — `snapshot` — dispatch periods, the last wrapping to the first |
| $`\mathcal{G}`$ | index $`g`$ — `generator` — generating units |
| $`\mathcal{S}`$ | index $`s`$ — `snapshot` — dispatch periods, cyclic at the horizon |

#### Parameters

| Symbol | Meaning |
|---|---|
| $`c`$ | `cost` over $`\mathcal{G}`$ — marginal cost |
| $`\ell`$ | `load` over $`\mathcal{S}`$ — demand to be met |
| $`\bar p`$ | `p_max` over $`\mathcal{G}`$ — installed capacity |

#### Variables

| Symbol | Meaning |
|---|---|
| $`\mathrm{charge}`$ | `charge` over $`\mathcal{S}`$ — energy into the store |
| $`\mathrm{discharge}`$ | `discharge` over $`\mathcal{S}`$ — energy out of the store |
| $`p`$ | `p` over $`\mathcal{G} \times \mathcal{S}`$ — output of a generator in a snapshot |
| $`\mathrm{soc}`$ | `soc` over $`\mathcal{S}`$ — state of charge carried into the next snapshot |

#### Notation

| Symbol | Meaning |
|---|---|
| $`t \ominus k`$ | `edge='wrap'` — $`t-k`$, cyclic |

#### Variable domains

**`charge`**

```math
0 \le \mathrm{charge}_{s} \le 30 \qquad \forall\, s \in \mathcal{S}
```

**`discharge`**

```math
0 \le \mathrm{discharge}_{s} \le 30 \qquad \forall\, s \in \mathcal{S}
```

**`p`**

```math
0 \le p_{g,s} \le \bar p_{g} \qquad \forall\, g \in \mathcal{G},\ s \in \mathcal{S}
```

**`soc`**

```math
0 \le \mathrm{soc}_{s} \le 100 \qquad \forall\, s \in \mathcal{S}
```

#### Objective

```math
\min \sum_{g \in \mathcal{G},\ s \in \mathcal{S}} p_{g,s} \cdot c_{g}
```

#### Subject to

**`power_balance`**

```math
\sum_{g \in \mathcal{G}} p_{g,s} + \mathrm{discharge}_{s} - \mathrm{charge}_{s} = \ell_{s} \qquad \forall\, s \in \mathcal{S}
```

**`soc_balance`**

```math
\mathrm{soc}_{s} = \mathrm{soc}_{s \ominus 1} + \mathrm{charge}_{s} \cdot 0.9 - \mathrm{discharge}_{s} \qquad \forall\, s \in \mathcal{S}
```

</details>
<!-- math:end -->

The tabs start from [the instance's tables](../howto/data.md) — one table per parameter.

=== "specsolve"

    ```yaml
    description: >-
      Dispatch plus a battery whose state of charge is closed into a cycle: the
      level ends where it began.

    dimensions:
      snapshot:
        description: dispatch periods, the last wrapping to the first
        dtype: int
        ordered: true
      generator:
        description: generating units
        dtype: str

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

    variables:
      p:
        description: output of a generator in a snapshot
        dims: [snapshot, generator]
        bounds:
          lower: 0
          upper: p_max
      charge:
        description: energy into the store
        dims: [snapshot]
        bounds:
          lower: 0
          upper: 30
      discharge:
        description: energy out of the store
        dims: [snapshot]
        bounds:
          lower: 0
          upper: 30
      soc:
        description: state of charge carried into the next snapshot
        dims: [snapshot]
        bounds:
          lower: 0
          upper: 100

    constraints:
      power_balance:
        description: generation plus what the store gives back covers the load, net of charging
        dims: [snapshot]
        expression: sum(p, over=generator) + discharge - charge == load
      soc_balance:
        description: >-
          the level carried out of a snapshot is the one carried in plus what was
          stored, minus what was taken — and it wraps at the last snapshot, so the first
          snapshot inherits from the last
        dims: [snapshot]
        expression: soc == shift(soc, along=snapshot, offset=1, edge='wrap') + charge * 0.9 - discharge

    objective:
      sense: minimize
      description: total cost of generation; storing and releasing energy is free here
      expression: sum(p * cost)
    ```

    ```python
    # sources: parameter name -> table or parquet path
    with sps.solve('examples/storage.yaml', sources) as result:
        result.objective  # 5650.0
        result.dual('power_balance')
    ```

=== "linopy"

    The model-building half of `examples/ports/references/linopy/storage.py`:

    ```python
    def build(tables: dict[str, pd.DataFrame]) -> linopy.Model:
        """The instance's tables as a linopy model, row for row.

        ``tables`` is the same mapping the specsolve call attaches as ``sources``.
        """
        p_max: pd.Series = tables['p_max'].set_index('generator')['value']
        cost: pd.Series = tables['cost'].set_index('generator')['value']
        load: pd.Series = tables['load'].set_index('snapshot')['value']
        snapshots = load.index

        m = linopy.Model()
        p = m.add_variables(lower=0, upper=p_max, coords=[snapshots, p_max.index], name='p')
        charge = m.add_variables(lower=0, upper=30, coords=[snapshots], name='charge')
        discharge = m.add_variables(lower=0, upper=30, coords=[snapshots], name='discharge')
        soc = m.add_variables(lower=0, upper=100, coords=[snapshots], name='soc')

        m.add_constraints(p.sum('generator') + discharge - charge == load, name='power_balance')
        m.add_constraints(soc == soc.roll(snapshot=1) + 0.9 * charge - discharge, name='soc_balance')
        m.add_objective((p * cost).sum())
        return m
    ```

---

[`examples/storage.yaml`](https://github.com/fluxopt/specsolve/blob/main/examples/storage.yaml) · back to [all models](index.md)
