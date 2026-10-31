# PyPSA unit commitment

Which generators are *on*, not just how much they produce — a binary per generator per snapshot, with start-up and shut-down charges.

> **✔ Verified against pypsa 1.2.4 (its own linopy 0.10.0)** — objective **24900**, matched to `rtol=1e-09`.

**Integrality enters here.** Every model above is a continuous LP; this one
carries a binary status. One bus and no network: a model that fails to match
should implicate one feature, and here that feature is commitment.

`min_up_time` and `min_down_time` are left at 0 here;
[minimum up and down times](pypsa_min_up_down.md) is the model that writes them.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

PyPSA unit commitment: which generators are on, not just how much they produce — a binary status per generator per snapshot, with start-up and shut-down charges. Optimum 24900.0, from PyPSA itself.

#### Sets

| Symbol | Meaning |
|---|---|
| $`\mathcal{T}`$ | index $`t`$ — `snapshot` — dispatch periods |
| $`\mathcal{G}`$ | index $`g`$ — `generator` — generating units, each either committed or off |

#### Parameters

| Symbol | Meaning |
|---|---|
| $`\mathrm{p}^{\mathrm{nom}}`$ | `p_nom` over $`\mathcal{G}`$ — installed capacity of a generator |
| $`\mathrm{marginal\_cost}`$ | `marginal_cost` over $`\mathcal{G}`$ — cost of one unit of output |
| $`\mathrm{p}^{\mathrm{min,pu}}`$ | `p_min_pu` over $`\mathcal{G}`$ — share of capacity a committed unit must produce at least |
| $`\mathrm{start\_up\_cost}`$ | `start_up_cost` over $`\mathcal{G}`$, `neutral` where the data has no row — what bringing a unit up costs, once per start |
| $`\mathrm{shut\_down\_cost}`$ | `shut_down_cost` over $`\mathcal{G}`$, `neutral` where the data has no row — what taking a unit down costs, once per stop |
| $`\mathrm{load}`$ | `load` over $`\mathcal{T}`$ — demand to be met |

#### Variables

| Symbol | Meaning |
|---|---|
| $`p`$ | `p` over $`\mathcal{T} \times \mathcal{G}`$ — output of a generator in a snapshot |
| $`\mathit{status}`$ | `status` over $`\mathcal{T} \times \mathcal{G}`$ — is this unit committed in this snapshot? |
| $`\mathit{start\_up}`$ | `start_up` over $`\mathcal{T} \times \mathcal{G}`$ — does this unit come up entering this snapshot? |
| $`\mathit{shut\_down}`$ | `shut_down` over $`\mathcal{T} \times \mathcal{G}`$ — does this unit go down entering this snapshot? |

#### Notation

| Symbol | Meaning |
|---|---|
| $`\mathrm{pos}(t)`$ | `position()` — the position of $`t`$, counted from $`0`$; $`t`$ is the label |

#### Variable domains

**`p`**

```math
p_{t,g} \ge 0 \qquad \forall\, t \in \mathcal{T},\ g \in \mathcal{G}
```

**`status`**

```math
\mathit{status}_{t,g} \in \{0, 1\} \qquad \forall\, t \in \mathcal{T},\ g \in \mathcal{G}
```

**`start_up`**

```math
\mathit{start\_up}_{t,g} \in \{0, 1\} \qquad \forall\, t \in \mathcal{T},\ g \in \mathcal{G}
```

**`shut_down`**

```math
\mathit{shut\_down}_{t,g} \in \{0, 1\} \qquad \forall\, t \in \mathcal{T},\ g \in \mathcal{G}
```

#### Objective

```math
\min \sum_{t \in \mathcal{T},\ g \in \mathcal{G}} p_{t,g} \cdot \mathrm{marginal\_cost}_{g} + \sum_{t \in \mathcal{T},\ g \in \mathcal{G}} \mathit{start\_up}_{t,g} \cdot \mathrm{start\_up\_cost}_{g} + \sum_{t \in \mathcal{T},\ g \in \mathcal{G}} \mathit{shut\_down}_{t,g} \cdot \mathrm{shut\_down\_cost}_{g}
```

#### Subject to

**`power_balance`**

```math
\sum_{g \in \mathcal{G}} p_{t,g} = \mathrm{load}_{t} \qquad \forall\, t \in \mathcal{T}
```

**`commitment_max`**

```math
p_{t,g} - \mathrm{p}^{\mathrm{nom}}_{g} \cdot \mathit{status}_{t,g} \le 0 \qquad \forall\, t \in \mathcal{T},\ g \in \mathcal{G}
```

**`commitment_min`**

```math
p_{t,g} - \mathrm{p}^{\mathrm{min,pu}}_{g} \cdot \mathrm{p}^{\mathrm{nom}}_{g} \cdot \mathit{status}_{t,g} \ge 0 \qquad \forall\, t \in \mathcal{T},\ g \in \mathcal{G}
```

**`start_up_initial`**

```math
\mathit{start\_up}_{t,g} - \mathit{status}_{t,g} \ge -1 \qquad \forall\, t \in \mathcal{T},\ g \in \mathcal{G} \,:\, \mathrm{pos}(t) = 0
```

**`start_up`**

```math
\mathit{start\_up}_{t,g} - \mathit{status}_{t,g} + \mathit{status}_{t - 1,g} \ge 0 \qquad \forall\, t \in \mathcal{T},\ g \in \mathcal{G}
```

**`shut_down_initial`**

```math
\mathit{shut\_down}_{t,g} + \mathit{status}_{t,g} \ge 1 \qquad \forall\, t \in \mathcal{T},\ g \in \mathcal{G} \,:\, \mathrm{pos}(t) = 0
```

**`shut_down`**

```math
\mathit{shut\_down}_{t,g} + \mathit{status}_{t,g} - \mathit{status}_{t - 1,g} \ge 0 \qquad \forall\, t \in \mathcal{T},\ g \in \mathcal{G}
```

</details>
<!-- math:end -->

The tabs start from [the instance's tables](../howto/data.md) — one frame per parameter.

=== "specsolve"

    ```yaml
    description: >-
      PyPSA unit commitment: which generators are on, not just how much they
      produce — a binary status per generator per snapshot, with start-up and
      shut-down charges. Optimum 24900.0, from PyPSA itself.

    dimensions:
      snapshot:
        description: dispatch periods
        dtype: int
        ordered: true
      generator:
        description: generating units, each either committed or off
        dtype: str

    parameters:
      p_nom:
        description: installed capacity of a generator
        dims: [generator]
      marginal_cost:
        description: cost of one unit of output
        dims: [generator]
      p_min_pu:
        description: share of capacity a committed unit must produce at least
        dims: [generator]
      start_up_cost:
        description: what bringing a unit up costs, once per start
        dims: [generator]
        missing: neutral
      shut_down_cost:
        description: what taking a unit down costs, once per stop
        dims: [generator]
        missing: neutral
      load:
        description: demand to be met
        dims: [snapshot]

    variables:
      p:
        description: output of a generator in a snapshot
        dims: [snapshot, generator]
        bounds:
          lower: 0
      status:
        description: is this unit committed in this snapshot?
        dims: [snapshot, generator]
        domain: binary
      start_up:
        description: does this unit come up entering this snapshot?
        dims: [snapshot, generator]
        domain: binary
      shut_down:
        description: does this unit go down entering this snapshot?
        dims: [snapshot, generator]
        domain: binary

    constraints:
      power_balance:
        dims: [snapshot]
        expression: sum(p, over=generator) == load

      commitment_max:
        description: >-
          a committed unit runs at no more than its capacity and an uncommitted one
          is pinned to zero — capacity times status is a parameter against a
          variable, so the product stays degree 1
        dims: [snapshot, generator]
        expression: p - p_nom * status <= 0

      commitment_min:
        description: a committed unit runs at no less than its minimum, an uncommitted one at zero
        dims: [snapshot, generator]
        expression: p - p_min_pu * p_nom * status >= 0

      start_up_initial:
        description: >-
          the first snapshot has no predecessor, and PyPSA's default is that the
          unit was already up before the horizon — so the start-up row is slackened
          here and never binds
        dims: [snapshot, generator]
        where: "position(snapshot) == 0"
        expression: start_up - status >= -1

      start_up:
        description: >-
          a unit whose status rises entering this snapshot pays for a start. The
          start-up and shut-down variables are implied by these transitions, but
          PyPSA declares them binary rather than leaving it to the status, and the
          port matches that.
        dims: [snapshot, generator]
        expression: start_up - status + shift(status, along=snapshot, offset=1) >= 0

      shut_down_initial:
        description: >-
          the mirror of the start-up row, and not slackened: a unit that begins the
          horizon off is charged for the shut-down, which is PyPSA's asymmetry and
          worth 50 on this instance
        dims: [snapshot, generator]
        where: "position(snapshot) == 0"
        expression: shut_down + status >= 1

      shut_down:
        description: a unit whose status falls entering this snapshot pays for a stop
        dims: [snapshot, generator]
        expression: shut_down + status - shift(status, along=snapshot, offset=1) >= 0

    objective:
      sense: minimize
      description: what the fleet costs to run, plus what its starts and stops cost
      expression: sum(p * marginal_cost) + sum(start_up * start_up_cost) + sum(shut_down * shut_down_cost)
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with sps.solve('examples/ports/pypsa_unit_commitment.yaml', sources) as solution:
        solution.objective  # 24900.0
    ```

=== "PyPSA"

    The model-building half of `examples/ports/references/pypsa/pypsa_unit_commitment.py`:

    ```python
    def build(tables: dict[str, pd.DataFrame]) -> pypsa.Network:
        """The port's tables as a PyPSA network, column for column.

        ``tables`` is the same mapping the specsolve call attaches as ``sources``.
        """
        n = pypsa.Network()
        n.set_snapshots(tables['snapshot']['snapshot'])
        n.add('Bus', 'bus')

        generators: pd.DataFrame = tables['generator'].set_index('generator')

        n.add(
            'Generator',
            generators.index,
            bus='bus',
            committable=True,
            p_nom=tables['p_nom'].set_index('generator')['value'],
            marginal_cost=tables['marginal_cost'].set_index('generator')['value'],
            p_min_pu=tables['p_min_pu'].set_index('generator')['value'],
            start_up_cost=tables['start_up_cost'].set_index('generator')['value'],
            shut_down_cost=tables['shut_down_cost'].set_index('generator')['value'],
        )

        load: pd.Series = tables['load'].set_index('snapshot')['value']
        n.add('Load', 'load', bus='bus', p_set=load)
        return n
    ```

**The first snapshot is not like the others.** PyPSA's default is that a unit
was already up before the horizon began. The start-up row is slackened to
`>= -1` there and never binds, while the shut-down row still charges a unit
that begins the horizon off. `peak` begins off, so the instance pays a
shut-down it never visibly performs. That asymmetry is PyPSA's, and it is
worth 50 here.

Two `where` clauses on one constraint block say that the row differs at the
boundary, the shape [storage](storage.md) uses for its initial state of
charge.

## What it costs

| | |
|---|---|
| energy | `30 × 520 + 90 × 100` = 24600 |
| start-ups | `peak` at snapshot 1 = 200 |
| shut-downs | `peak` at snapshot 0 (begins off) and at 3 = 100 |
| **total** | **24900** |

`base` runs throughout; `peak` covers the two peak snapshots. specsolve and PyPSA
agree on the schedule as well as the cost.

## What it exercises

`binary` variables and the integrality path through to HiGHS, `shift` across a
boundary condition, and a three-term objective mixing an energy cost with two
transition charges.
