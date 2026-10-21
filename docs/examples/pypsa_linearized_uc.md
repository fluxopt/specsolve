# PyPSA linearized unit commitment — the status, relaxed

A unit may be committed by a third. PyPSA ships this as a mode, not as a debugging convenience.

> **✔ Verified against pypsa 1.2.4 (its own linopy 0.10.0)** — objective **5540.0**, matched to `rtol=1e-09`.

[Unit commitment](pypsa_unit_commitment.md) is a MILP. Its linear relaxation
keeps every constraint where it is and declares the status and the two
transition variables in [0, 1] instead of {0, 1}. Three `domain:` lines carry
the whole relaxation, and every row the two models share is byte-identical.
They part company in one place, and not over integrality. This instance starts
every unit off, where the integer port takes PyPSA's default of a unit already
running, so the first snapshot's two transition rows differ.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

PyPSA linearized unit commitment: commitment with the status continuous in \[0, 1\] rather than binary, so a unit may be committed by a third. A relaxation and therefore a bound — on this instance worth less than half the integer answer. Optimum 5540.0, from PyPSA itself.

#### Sets

| Symbol | Meaning |
|---|---|
| $`\mathcal{T}`$ | index $`t`$ — `snapshot` — dispatch periods |
| $`\mathcal{G}`$ | index $`g`$ — `generator` — generating units, each committed to some degree or not at all |

#### Parameters

| Symbol | Meaning |
|---|---|
| $`\mathrm{p}^{\mathrm{nom}}`$ | `p_nom` over $`\mathcal{G}`$ — installed capacity of a generator |
| $`\mathrm{p}^{\mathrm{min,pu}}`$ | `p_min_pu` over $`\mathcal{G}`$ — share of capacity a committed unit must produce at least |
| $`\mathrm{marginal\_cost}`$ | `marginal_cost` over $`\mathcal{G}`$ — cost of one unit of output |
| $`\mathrm{start\_up\_cost}`$ | `start_up_cost` over $`\mathcal{G}`$, `neutral` where the data has no row — what bringing a unit up costs, once per start |
| $`\mathrm{shut\_down\_cost}`$ | `shut_down_cost` over $`\mathcal{G}`$, `neutral` where the data has no row — what taking a unit down costs, once per stop |
| $`\mathrm{load}`$ | `load` over $`\mathcal{T}`$ — demand to be met |

#### Variables

| Symbol | Meaning |
|---|---|
| $`p`$ | `p` over $`\mathcal{T} \times \mathcal{G}`$ — output of a generator in a snapshot |
| $`\mathit{status}`$ | `status` over $`\mathcal{T} \times \mathcal{G}`$ — how far this unit is committed in this snapshot — one of the three declarations that separate this model from the integer one |
| $`\mathit{start\_up}`$ | `start_up` over $`\mathcal{T} \times \mathcal{G}`$ — how much of this unit comes up entering this snapshot |
| $`\mathit{shut\_down}`$ | `shut_down` over $`\mathcal{T} \times \mathcal{G}`$ — how much of this unit goes down entering this snapshot |

#### Notation

| Symbol | Meaning |
|---|---|
| $`t \boxminus_{v} k`$ | `edge=v` — $`t-k`$, and $`v`$ past the edge |

#### Variable domains

**`p`**

```math
p_{t,g} \ge 0 \qquad \forall\, t \in \mathcal{T},\ g \in \mathcal{G}
```

**`status`**

```math
0 \le \mathit{status}_{t,g} \le 1 \qquad \forall\, t \in \mathcal{T},\ g \in \mathcal{G}
```

**`start_up`**

```math
0 \le \mathit{start\_up}_{t,g} \le 1 \qquad \forall\, t \in \mathcal{T},\ g \in \mathcal{G}
```

**`shut_down`**

```math
0 \le \mathit{shut\_down}_{t,g} \le 1 \qquad \forall\, t \in \mathcal{T},\ g \in \mathcal{G}
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

**`start_up`**

```math
\mathit{start\_up}_{t,g} - \mathit{status}_{t,g} + \mathit{status}_{t \boxminus_{0} 1,g} \ge 0 \qquad \forall\, t \in \mathcal{T},\ g \in \mathcal{G}
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
      PyPSA linearized unit commitment: commitment with the status continuous in
      [0, 1] rather than binary, so a unit may be committed by a third. A relaxation
      and therefore a bound — on this instance worth less than half the integer
      answer. Optimum 5540.0, from PyPSA itself.

    dimensions:
      snapshot:
        description: dispatch periods
        dtype: int
        ordered: true
      generator:
        description: generating units, each committed to some degree or not at all
        dtype: str

    parameters:
      p_nom:
        description: installed capacity of a generator
        dims: [generator]
      p_min_pu:
        description: share of capacity a committed unit must produce at least
        dims: [generator]
      marginal_cost:
        description: cost of one unit of output
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
        description: >-
          how far this unit is committed in this snapshot — one of the three
          declarations that separate this model from the integer one
        dims: [snapshot, generator]
        bounds:
          lower: 0
          upper: 1
      start_up:
        description: how much of this unit comes up entering this snapshot
        dims: [snapshot, generator]
        bounds:
          lower: 0
          upper: 1
      shut_down:
        description: how much of this unit goes down entering this snapshot
        dims: [snapshot, generator]
        bounds:
          lower: 0
          upper: 1

    constraints:
      power_balance:
        dims: [snapshot]
        expression: sum(p, over=generator) == load

      commitment_max:
        description: a unit runs at no more than the share of capacity it has committed
        dims: [snapshot, generator]
        expression: p - p_nom * status <= 0

      commitment_min:
        description: and at no less than that share of its minimum
        dims: [snapshot, generator]
        expression: p - p_min_pu * p_nom * status >= 0

      start_up:
        description: a unit whose commitment rises entering this snapshot pays for the rise
        dims: [snapshot, generator]
        expression: start_up - status + shift(status, along=snapshot, offset=1, edge=0) >= 0

      shut_down:
        description: a unit whose commitment falls entering this snapshot pays for the fall
        dims: [snapshot, generator]
        expression: shut_down + status - shift(status, along=snapshot, offset=1) >= 0

    objective:
      sense: minimize
      description: what the fleet costs to run, plus what its starts and stops cost
      expression: sum(p * marginal_cost) + sum(start_up * start_up_cost) + sum(shut_down * shut_down_cost)
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with sps.solve('examples/ports/pypsa_linearized_uc.yaml', sources) as solution:
        solution.objective  # 5540.0
        solution.dual('power_balance')
    ```

=== "PyPSA"

    The model-building half of `examples/ports/references/pypsa/pypsa_linearized_uc.py`:

    ```python
    def build(tables: dict[str, pd.DataFrame]) -> pypsa.Network:
        """The port's tables as a PyPSA network, column for column.

        ``tables`` is the same mapping the specsolve call attaches as ``sources``.

        Nothing here says the model is relaxed: ``committable=True`` is the same
        switch the integer model uses, and the mode is chosen at ``optimize`` time.
        Both ``*_time_before`` are 0, so no status is pinned by a prior horizon.
        """
        n = pypsa.Network()
        n.set_snapshots(tables['snapshot']['snapshot'])
        n.add('Bus', 'hub')

        generators: pd.DataFrame = tables['generator'].set_index('generator')
        n.add(
            'Generator',
            generators.index,
            bus='hub',
            committable=True,
            up_time_before=0,
            down_time_before=0,
            p_nom=tables['p_nom'].set_index('generator')['value'],
            p_min_pu=tables['p_min_pu'].set_index('generator')['value'],
            marginal_cost=tables['marginal_cost'].set_index('generator')['value'],
            start_up_cost=tables['start_up_cost'].set_index('generator')['value'],
            shut_down_cost=tables['shut_down_cost'].set_index('generator')['value'],
        )

        n.add('Load', 'l', bus='hub', p_set=tables['load'].set_index('snapshot')['value'])
        return n
    ```

**A bound, not an approximation to trust.** The relaxed statuses come out at
0.3, 0.9, 0.8 and 0.8, a third of a power station. The same instance solved as
a MILP costs **11900.0** against this **5540.0**, less than half. The gap is
what a minimum-output constraint is worth once a fraction of a unit may be
committed.

**This is the only committable model in the corpus with duals.** Integrality
leaves a dual solution undefined, so `pypsa_unit_commitment` and
[minimum up and down times](pypsa_min_up_down.md) record an objective and
nothing else. Relaxing the status turns the model back into an LP, which is
most of why the mode exists, so this port records a price vector too.

**`base` carries unequal start-up and shut-down costs.** PyPSA tightens the
relaxation with an extra dispatch-limit block wherever a generator's two costs
match. That block reaches for the ramp-limit parameters, a second feature. So
`base` is left untightened, and PyPSA logs that it is proceeding without
it. `peak` has two costs of zero, so PyPSA does emit the tightening there:
four blocks of three rows this port does not hold. Each collapses to a row the
port already has, because `p_min_pu` is 0 and there are no ramp limits:
`p ≤ p_nom · status`, or that row differenced against the snapshot before. The
objective and the price vector therefore agree to `rtol=1e-09`; the two are
the same model by redundancy, not row for row.

## What it exercises

Integrality as one declaration on a variable, with nothing above it caring:
`domain: binary` against a `[0, 1]` bound is the whole of the relaxation.
