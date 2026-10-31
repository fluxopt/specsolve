# PyPSA minimum up and down times — the rows that make commitment bite

A unit that has started must stay on; one that has stopped must stay off.

> **✔ Verified against pypsa 1.2.4 (its own linopy 0.10.0)** — objective **32750.0**, matched to `rtol=1e-09`.

[Unit commitment](pypsa_unit_commitment.md) takes the status and the two
transition variables and stops there. A unit may start and stop in consecutive
snapshots, paying the charges. The windows make commitment a scheduling
problem. Over any `min_up_time` consecutive snapshots a unit may have started
at most as often as it is now running, and the mirror holds for stopping.

Each window's length is a property of the generator, not of the model: 3, 2
and 1 here. A single shared length would be satisfied by an operator that
ignored the parameter and used a constant.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

PyPSA minimum up and down times: a unit that has started must stay on, and one that has stopped must stay off. Each window is a backward-looking sum whose length is a property of the generator, and the three units here carry different lengths. Optimum 32750.0, from PyPSA itself.

#### Sets

| Symbol | Meaning |
|---|---|
| $`\mathcal{T}`$ | index $`t`$ — `snapshot` (`int` coordinates) — dispatch periods |
| $`\mathcal{G}`$ | index $`g`$ — `generator` — generating units, each either committed or off |

#### Parameters

| Symbol | Meaning |
|---|---|
| $`\mathrm{p}^{\mathrm{nom}}`$ | `p_nom` over $`\mathcal{G}`$ — installed capacity of a generator |
| $`\mathrm{p}^{\mathrm{min,pu}}`$ | `p_min_pu` over $`\mathcal{G}`$ — share of capacity a committed unit must produce at least |
| $`\mathrm{marginal\_cost}`$ | `marginal_cost` over $`\mathcal{G}`$ — cost of one unit of output |
| $`\mathrm{start\_up\_cost}`$ | `start_up_cost` over $`\mathcal{G}`$, `neutral` where the data has no row — what bringing a unit up costs, once per start |
| $`\mathrm{shut\_down\_cost}`$ | `shut_down_cost` over $`\mathcal{G}`$, `neutral` where the data has no row — what taking a unit down costs, once per stop |
| $`\mathrm{min\_up\_time}`$ | `min_up_time` over $`\mathcal{G}`$ — how many snapshots a unit must stay on once it has started |
| $`\mathrm{min\_down\_time}`$ | `min_down_time` over $`\mathcal{G}`$ — how many snapshots a unit must stay off once it has stopped |
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
| $`t \boxminus_{v} k`$ | `edge=v` — $`t-k`$, and $`v`$ past the edge |

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

**`start_up`**

```math
\mathit{start\_up}_{t,g} - \mathit{status}_{t,g} + \mathit{status}_{t \boxminus_{0} 1,g} \ge 0 \qquad \forall\, t \in \mathcal{T},\ g \in \mathcal{G}
```

**`shut_down`**

```math
\mathit{shut\_down}_{t,g} + \mathit{status}_{t,g} - \mathit{status}_{t - 1,g} \ge 0 \qquad \forall\, t \in \mathcal{T},\ g \in \mathcal{G}
```

**`min_up_time`**

```math
\sum_{t' \in \mathcal{T} \,:\, 0 \le t - t' < \mathrm{min\_up\_time}} \mathit{start\_up}_{t',g} \le \mathit{status}_{t,g} \qquad \forall\, t \in \mathcal{T},\ g \in \mathcal{G} \,:\, t > 0
```

**`min_down_time`**

```math
\mathit{status}_{t,g} + \sum_{t' \in \mathcal{T} \,:\, 0 \le t - t' < \mathrm{min\_down\_time}} \mathit{shut\_down}_{t',g} \le 1 \qquad \forall\, t \in \mathcal{T},\ g \in \mathcal{G} \,:\, t > 0
```

</details>
<!-- math:end -->

The tabs start from [the instance's tables](../howto/data.md) — one frame per parameter.

=== "specsolve"

    ```yaml
    description: >-
      PyPSA minimum up and down times: a unit that has started must stay on, and one
      that has stopped must stay off. Each window is a backward-looking sum whose
      length is a property of the generator, and the three units here carry
      different lengths. Optimum 32750.0, from PyPSA itself.

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
      min_up_time:
        description: how many snapshots a unit must stay on once it has started
        dims: [generator]
        dtype: int
      min_down_time:
        description: how many snapshots a unit must stay off once it has stopped
        dims: [generator]
        dtype: int
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
        description: a committed unit runs at no more than its capacity, an uncommitted one at zero
        dims: [snapshot, generator]
        expression: p - p_nom * status <= 0

      commitment_min:
        description: a committed unit runs at no less than its minimum, an uncommitted one at zero
        dims: [snapshot, generator]
        expression: p - p_min_pu * p_nom * status >= 0

      start_up:
        description: >-
          a unit whose status rises entering this snapshot pays for a start. Every
          unit begins the horizon off, which is the 0 the first snapshot reads where
          it has no predecessor
        dims: [snapshot, generator]
        expression: start_up - status + shift(status, along=snapshot, offset=1, edge=0) >= 0

      shut_down:
        description: >-
          a unit whose status falls entering this snapshot pays for a stop. There is
          no first-snapshot counterpart: a unit that begins off has nothing to stop,
          and the row PyPSA emits there holds for every value of both variables.
        dims: [snapshot, generator]
        expression: shut_down + status - shift(status, along=snapshot, offset=1) >= 0

      min_up_time:
        description: >-
          over the last `min_up_time` snapshots a unit may have started at most as
          often as it is running now — which is what stops it starting and stopping
          again inside its own window
        dims: [snapshot, generator]
        where: "snapshot > 0"
        expression: sum_back(start_up, along=snapshot, window=min_up_time) <= status

      min_down_time:
        description: >-
          the mirror: having stopped inside the window and running now cannot both
          be true
        dims: [snapshot, generator]
        where: "snapshot > 0"
        expression: status + sum_back(shut_down, along=snapshot, window=min_down_time) <= 1

    objective:
      sense: minimize
      description: what the fleet costs to run, plus what its starts and stops cost
      expression: sum(p * marginal_cost) + sum(start_up * start_up_cost) + sum(shut_down * shut_down_cost)
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with sps.solve('examples/ports/pypsa_min_up_down.yaml', sources) as solution:
        solution.objective  # 32750.0
    ```

=== "PyPSA"

    The model-building half of `examples/ports/references/pypsa/pypsa_min_up_down.py`:

    ```python
    def build(tables: dict[str, pd.DataFrame]) -> pypsa.Network:
        """The port's tables as a PyPSA network, column for column.

        ``tables`` is the same mapping the specsolve call attaches as ``sources``.

        One bus and no network: a model that fails to match should implicate one
        feature, and here it is the window length. ``committable`` is what turns the
        status into a variable at all, and the two ``*_time_before`` values are set
        rather than defaulted for the reason in the module docstring.
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
            min_up_time=tables['min_up_time'].set_index('generator')['value'],
            min_down_time=tables['min_down_time'].set_index('generator')['value'],
        )

        n.add('Load', 'l', bus='hub', p_set=tables['load'].set_index('snapshot')['value'])
        return n
    ```

**The windows bind, and the schedule shows where.** `mid` runs, drops out over
snapshots 4 and 5, and comes back at 6. It then has to stay on through 7,
where the load is met more cheaply without it. Set every window to 1 and the
same instance solves at **31800.0**, with `mid` free to cycle in and out of
every trough. The windows are worth 950.

**`sum_back` truncates at the start of the dimension, and so does PyPSA.** At
snapshot 1 with a window of 3, the sum has two terms rather than three: there
is no snapshot −1 to reach. Both shorten the window rather than wrapping or
dropping the row, so no `edge=` argument appears.

**The initial conditions are switched off rather than defaulted.**
`up_time_before` defaults to 1 in PyPSA, meaning the unit was already running.
That default emits a further block pinning the status on for the rest of its
minimum up time. That is a second feature, and this model is about the
windows, so both `*_time_before` values are 0. Every unit therefore begins the
horizon off, and the first snapshot's transition rows mirror the ones
[unit commitment](pypsa_unit_commitment.md) ports. A unit committed in the
first snapshot pays for a start and nothing for a stop. A unit that began the
horizon running pays for no start and is charged if it goes down.

## What it exercises

`sum_back(x, along=dim, window=p)` with `p` an integer parameter: the
per-entity window length, against a variable, inside a MILP. The `dtype: int`
declaration is required and load-time validation says so by name: *a width
counts positions rather than measuring a distance.*
