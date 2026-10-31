# PyPSA stochastic optimisation — capacity once, dispatch per future

Three futures over one network: the fleet is built before anyone knows which arrives, and dispatched after.

> **✔ Verified against pypsa 1.2.4 (its own linopy 0.10.0)** — objective **33940.0**, matched to `rtol=1e-09`.

`n.set_scenarios` makes a network stochastic by adding one dimension to some
variables and not others:

```
Generator-p_nom  ('name',)
Generator-p      ('scenario', 'name', 'snapshot')
```

That is the whole two-stage program. Capacity is a **first-stage** decision: one
number per generator, spanning no scenario, because it is taken while all three
futures are still open. Dispatch is **second stage**, one per future, chosen once
the load is known. `define_objective` then runs every term through `_expected`,
which selects each scenario and multiplies it by that scenario's weight
(`optimize.py:361`).

The three futures differ in one thing, the load:

| | probability | snapshot 0 | 1 | 2 |
|---|---|---|---|---|
| `mild` | 0.6 | 100 | 120 | 90 |
| `cold` | 0.3 | 130 | 160 | 110 |
| `severe` | 0.1 | 170 | 210 | 140 |

`base` costs 150 to build and 10 to run, `peak` 120 and 70, so the expectation
decides the mix as well as the total.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

PyPSA stochastic optimisation: one network and three futures, where capacity is chosen once and dispatch is chosen after the load is known. The two stages differ only in which dimension they span — the capacity variable has no scenario, the dispatch variable does — and the objective is the probability-weighted expectation over them. Optimum 33940.0, from PyPSA itself.

#### Sets

| Symbol | Meaning |
|---|---|
| $`\mathcal{S}`$ | index $`s`$ — `scenario` — the futures the fleet is built against, one of which will happen |
| $`\mathcal{T}`$ | index $`t`$ — `snapshot` — dispatch periods, the same in every future |
| $`\mathcal{G}`$ | index $`g`$ — `generator` — generating units, each built once and run in every future |

#### Parameters

| Symbol | Meaning |
|---|---|
| $`\mathrm{probability}`$ | `probability` over $`\mathcal{S}`$ — how likely a future is — the weights the expectation is taken with |
| $`\mathrm{load}`$ | `load` over $`\mathcal{S} \times \mathcal{T}`$ — demand to be met, and the one thing that differs between futures |
| $`\mathrm{capex}`$ | `capex` over $`\mathcal{G}`$ — cost of holding one unit of capacity over the horizon |
| $`\mathrm{opex}`$ | `opex` over $`\mathcal{G}`$ — cost of one unit of output |

#### Variables

| Symbol | Meaning |
|---|---|
| $`p^{\mathrm{nom}}`$ | `p_nom` over $`\mathcal{G}`$ — capacity built at a generator — the first-stage decision, which spans no scenario because it is taken before anyone knows which future arrived |
| $`p`$ | `p` over $`\mathcal{S} \times \mathcal{T} \times \mathcal{G}`$ — output of a generator in a snapshot of a future — the second-stage decision, one per scenario |

#### Variable domains

**`p_nom`**

```math
p^{\mathrm{nom}}_{g} \ge 0 \qquad \forall\, g \in \mathcal{G}
```

**`p`**

```math
p_{s,t,g} \ge 0 \qquad \forall\, s \in \mathcal{S},\ t \in \mathcal{T},\ g \in \mathcal{G}
```

#### Objective

```math
\min \sum_{s \in \mathcal{S},\ t \in \mathcal{T},\ g \in \mathcal{G}} p_{s,t,g} \cdot \mathrm{opex}_{g} \cdot \mathrm{probability}_{s} + \sum_{g \in \mathcal{G}} p^{\mathrm{nom}}_{g} \cdot \mathrm{capex}_{g}
```

#### Subject to

**`within_capacity`**

```math
p_{s,t,g} \le p^{\mathrm{nom}}_{g} \qquad \forall\, s \in \mathcal{S},\ t \in \mathcal{T},\ g \in \mathcal{G}
```

**`power_balance`**

```math
\sum_{g \in \mathcal{G}} p_{s,t,g} = \mathrm{load}_{s,t} \qquad \forall\, s \in \mathcal{S},\ t \in \mathcal{T}
```

</details>
<!-- math:end -->

The tabs start from [the instance's tables](../howto/data.md) — one frame per parameter.

=== "specsolve"

    ```yaml
    description: >-
      PyPSA stochastic optimisation: one network and three futures, where capacity
      is chosen once and dispatch is chosen after the load is known. The two stages
      differ only in which dimension they span — the capacity variable has no
      scenario, the dispatch variable does — and the objective is the
      probability-weighted expectation over them.
      Optimum 33940.0, from PyPSA itself.

    dimensions:
      scenario:
        description: the futures the fleet is built against, one of which will happen
        dtype: str
      snapshot:
        description: dispatch periods, the same in every future
        dtype: int
        ordered: true
      generator:
        description: generating units, each built once and run in every future
        dtype: str

    parameters:
      probability:
        description: how likely a future is — the weights the expectation is taken with
        dims: [scenario]
      load:
        description: demand to be met, and the one thing that differs between futures
        dims: [scenario, snapshot]
      capex:
        description: cost of holding one unit of capacity over the horizon
        dims: [generator]
      opex:
        description: cost of one unit of output
        dims: [generator]

    variables:
      p_nom:
        description: >-
          capacity built at a generator — the first-stage decision, which spans no
          scenario because it is taken before anyone knows which future arrived
        dims: [generator]
        bounds:
          lower: 0
      p:
        description: >-
          output of a generator in a snapshot of a future — the second-stage
          decision, one per scenario
        dims: [scenario, snapshot, generator]
        bounds:
          lower: 0

    constraints:
      within_capacity:
        description: >-
          a generator produces no more than the capacity built for it, in every
          snapshot of every future — one capacity spanning three scenarios of rows
        dims: [scenario, snapshot, generator]
        expression: p <= p_nom

      power_balance:
        description: what runs in this snapshot of this future meets the load there
        dims: [scenario, snapshot]
        expression: sum(p, over=generator) == load

    objective:
      sense: minimize
      description: >-
        what the fleet costs to build, plus what it is expected to cost to run —
        operating cost weighted by how likely the future it is incurred in is, and
        capital cost paid once whichever future arrives
      expression: sum(p * opex * probability) + sum(p_nom * capex)
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with sps.solve('examples/ports/pypsa_stochastic.yaml', sources) as solution:
        solution.objective  # 33940.0
        solution.dual('power_balance')
    ```

=== "PyPSA"

    The model-building half of `examples/ports/references/pypsa/pypsa_stochastic.py`:

    ```python
    def build(tables: dict[str, pd.DataFrame]) -> pypsa.Network:
        """The port's tables as a PyPSA network, column for column.

        ``tables`` is the same mapping the specsolve call attaches as ``sources``.

        The port's ``probability`` table is PyPSA's ``scenario_weightings``, passed
        to ``set_scenarios``; the port's ``load`` over ``(scenario, snapshot)`` is a
        ``p_set`` frame whose columns carry the scenario level PyPSA gives every
        time-varying input once the network is stochastic. Both generators are
        extendable with no ``p_nom_max``: the fleet is what the model chooses, and a
        ceiling nothing reaches would be a parameter the port carries for nothing.
        """
        n = pypsa.Network()
        n.set_snapshots(list(tables['snapshot']['snapshot']))
        n.set_scenarios(tables['probability'].set_index('scenario')['value'])

        n.add('Bus', 'hub')
        generators: pd.DataFrame = tables['generator'].set_index('generator')
        n.add(
            'Generator',
            generators.index,
            bus='hub',
            p_nom_extendable=True,
            capital_cost=tables['capex'].set_index('generator')['value'],
            marginal_cost=tables['opex'].set_index('generator')['value'],
        )

        n.add('Load', 'l', bus='hub')
        load: pd.DataFrame = tables['load'].pivot(index='snapshot', columns='scenario', values='value')
        n.loads_t.p_set = pd.DataFrame(
            {(s, 'l'): load[s].to_numpy() for s in tables['probability']['scenario']}, index=n.snapshots
        )
        return n
    ```

**The expectation is doing work.** Collapse the three futures into their
probability-weighted mean load, 116, 141, 101, and the same network builds
**141 MW of `base` and no `peak` at all**, for 24730.0. That fleet has no
feasible dispatch in the severe future, which asks for 210.
`what_the_mean_would_build()` in the reference prints it.

**One capacity, three scenarios of rows.** `within_capacity` is `p <= p_nom`
with a `dims:` of `[scenario, snapshot, generator]` against a variable declared
over `[generator]`: nine rows reading one column. That is how a first-stage
decision couples to a second-stage one. The dim algebra broadcasts the capacity
because the constraint's frame says to.

**Prices carry the probability, and they add up to the capital cost.** The nodal
price in `mild` is 6.0 rather than 10.0, because the term it prices is weighted
by 0.6. `marginal_price` off a stochastic PyPSA network is a *weighted* price,
and both implementations agree on that. The scarcity rents are where the two
stages meet:

| | snapshot 0 | 1 | 2 |
|---|---|---|---|
| `mild` | 6 | 6 | 6 |
| `cold` | 3 | **21** | 3 |
| `severe` | 7 | **127** | 1 |

`base` is at its capacity in three cells, and the rents there (18 in `cold`, 6
and 126 in `severe`) sum to 150, its cost to build. `peak` is at capacity in one
cell, and its rent there is 120, its cost to build. A capacity that is chosen
once is paid for by every future that runs it out.

## What it exercises

Two variables that span different dimensions, coupled by a constraint whose
frame is the wider of the two, and an objective that reduces one of them against
a probability. Nothing here declares a stage. The two stages are visible only in
which dimensions each `dims:` lists.
