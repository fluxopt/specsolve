# PyPSA committable and extendable — the one place a big-M is unavoidable

A minimum output that is a share of a capacity still being decided: two variables multiplied, and one constant to take them apart.

> **✔ Verified against pypsa 1.2.4 (its own linopy 0.10.0)** — objective **21700.0**, matched to `rtol=1e-09`.

[Unit commitment](pypsa_unit_commitment.md) holds capacity fixed, so
`p >= p_min_pu * p_nom * status` is a parameter against a variable and stays
degree 1. Make the capacity a decision and the same row is a product of two
variables. That is the one shape in PyPSA's committable machinery that cannot
be written as it reads.

PyPSA's answer is three rows and a constant *M* (`constraints.py:304`):

| row | what it says |
|---|---|
| `p - p_min_pu * p_nom - M * status >= -M` | the floor, and slack by all of *M* while the unit is off |
| `p - M * status <= 0` | off means zero |
| `p - p_nom <= 0` | the capacity limit, committed or not |

*M* is not free. PyPSA takes `p_nom_max * p_max_pu` where a maximum is declared
and otherwise **infers a scale from the network**, which
`create_model(committable_big_m=...)` overrides. This instance declares
`p_nom_max`, so *M* is 100 on both sides, and the port passes that number as
data.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

PyPSA's committable unit whose capacity is also being built: the minimum output is a share of a capacity that is itself a decision, so the product of the status and the capacity is linearised with a big-M the model declares rather than infers. One unit is both committed and built, one is only built. Optimum 21700.0, from PyPSA itself.

#### Sets

| Symbol | Meaning |
|---|---|
| $`\mathcal{T}`$ | index $`t`$ — `snapshot` — dispatch periods |
| $`\mathcal{G}`$ | index $`g`$ — `generator` — generating units, some of which are committed rather than merely dispatched |

#### Parameters

| Symbol | Meaning |
|---|---|
| $`\mathrm{p}^{\mathrm{min,pu}}`$ | `p_min_pu` over $`\mathcal{G}`$ — share of its built capacity a committed unit must produce while on |
| $`\mathrm{p}^{\mathrm{nom,max}}`$ | `p_nom_max` over $`\mathcal{G}`$ — most capacity a generator may build |
| $`\mathrm{big\_m}`$ | `big_m` over $`\mathcal{G}`$, `absent` where the data has no row — a bound on the output of a committed unit, large enough never to bind on its own — the capacity ceiling times the availability, and present only for the units that are committed at all |
| $`\mathrm{marginal\_cost}`$ | `marginal_cost` over $`\mathcal{G}`$ — cost of one unit of output |
| $`\mathrm{capital\_cost}`$ | `capital_cost` over $`\mathcal{G}`$ — cost of holding one unit of capacity over the horizon |
| $`\mathrm{load}`$ | `load` over $`\mathcal{T}`$ — demand to be met |

#### Variables

| Symbol | Meaning |
|---|---|
| $`p`$ | `p` over $`\mathcal{T} \times \mathcal{G}`$ — output of a generator in a snapshot |
| $`p^{\mathrm{nom}}`$ | `p_nom` over $`\mathcal{G}`$ — capacity built at a generator |
| $`\mathit{status}`$ | `status` over $`\mathcal{T} \times \mathcal{G}`$ — is this unit committed in this snapshot? Declared only for the units that carry a big-M, which is what marks a unit as committed rather than merely dispatched |

#### Variable domains

**`p`**

```math
p_{t,g} \ge 0 \qquad \forall\, t \in \mathcal{T},\ g \in \mathcal{G}
```

**`p_nom`**

```math
0 \le p^{\mathrm{nom}}_{g} \le \mathrm{p}^{\mathrm{nom,max}}_{g} \qquad \forall\, g \in \mathcal{G}
```

**`status`**

```math
\mathit{status}_{t,g} \in \{0, 1\} \qquad \forall\, t \in \mathcal{T},\ g \in \mathcal{G} \,:\, \mathrm{big\_m}_{g} \text{ is defined}
```

#### Objective

```math
\min \sum_{t \in \mathcal{T},\ g \in \mathcal{G}} p_{t,g} \cdot \mathrm{marginal\_cost}_{g} + \sum_{g \in \mathcal{G}} p^{\mathrm{nom}}_{g} \cdot \mathrm{capital\_cost}_{g}
```

#### Subject to

**`within_capacity`**

```math
p_{t,g} \le p^{\mathrm{nom}}_{g} \qquad \forall\, t \in \mathcal{T},\ g \in \mathcal{G}
```

**`off_means_zero`**

```math
p_{t,g} - \mathrm{big\_m}_{g} \cdot \mathit{status}_{t,g} \le 0 \qquad \forall\, t \in \mathcal{T},\ g \in \mathcal{G} \,:\, \mathrm{big\_m}_{g} \text{ is defined}
```

**`commitment_floor`**

```math
p_{t,g} - \mathrm{p}^{\mathrm{min,pu}}_{g} \cdot p^{\mathrm{nom}}_{g} - \mathrm{big\_m}_{g} \cdot \mathit{status}_{t,g} \ge -\mathrm{big\_m}_{g} \qquad \forall\, t \in \mathcal{T},\ g \in \mathcal{G} \,:\, \mathrm{big\_m}_{g} \text{ is defined}
```

**`power_balance`**

```math
\sum_{g \in \mathcal{G}} p_{t,g} = \mathrm{load}_{t} \qquad \forall\, t \in \mathcal{T}
```

</details>
<!-- math:end -->

The tabs start from [the instance's tables](../howto/data.md) — one frame per parameter.

=== "specsolve"

    ```yaml
    description: >-
      PyPSA's committable unit whose capacity is also being built: the minimum
      output is a share of a capacity that is itself a decision, so the product of
      the status and the capacity is linearised with a big-M the model declares
      rather than infers. One unit is both committed and built, one is only built.
      Optimum 21700.0, from PyPSA itself.

    dimensions:
      snapshot:
        description: dispatch periods
        dtype: int
        ordered: true
      generator:
        description: generating units, some of which are committed rather than merely dispatched
        dtype: str

    parameters:
      p_min_pu:
        description: share of its built capacity a committed unit must produce while on
        dims: [generator]
      p_nom_max:
        description: most capacity a generator may build
        dims: [generator]
      big_m:
        description: >-
          a bound on the output of a committed unit, large enough never to bind on
          its own — the capacity ceiling times the availability, and present only for
          the units that are committed at all
        dims: [generator]
        missing: absent
      marginal_cost:
        description: cost of one unit of output
        dims: [generator]
      capital_cost:
        description: cost of holding one unit of capacity over the horizon
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
      p_nom:
        description: capacity built at a generator
        dims: [generator]
        bounds:
          lower: 0
          upper: p_nom_max
      status:
        description: >-
          is this unit committed in this snapshot? Declared only for the units that
          carry a big-M, which is what marks a unit as committed rather than merely
          dispatched
        dims: [snapshot, generator]
        where: big_m
        domain: binary

    constraints:
      within_capacity:
        description: a generator produces no more than the capacity built for it, committed or not
        dims: [snapshot, generator]
        expression: p <= p_nom

      off_means_zero:
        description: >-
          an uncommitted unit produces nothing, and a committed one is held only by
          the big-M — which is why the row above is the real capacity limit
        dims: [snapshot, generator]
        where: big_m
        expression: p - big_m * status <= 0

      commitment_floor:
        description: >-
          a committed unit runs at no less than its share of the capacity built for
          it. The share of a *variable* capacity is a product of two decisions, so it
          is linearised: the row is slack by the whole big-M while the unit is off
        dims: [snapshot, generator]
        where: big_m
        expression: p - p_min_pu * p_nom - big_m * status >= -big_m

      power_balance:
        dims: [snapshot]
        expression: sum(p, over=generator) == load

    objective:
      sense: minimize
      description: what the fleet costs to run, plus what its capacity costs to build
      expression: sum(p * marginal_cost) + sum(p_nom * capital_cost)
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with sps.solve('examples/ports/pypsa_committable_extendable.yaml', sources) as solution:
        solution.objective  # 21700.0
    ```

=== "PyPSA"

    The model-building half of `examples/ports/references/pypsa/pypsa_committable_extendable.py`:

    ```python
    def build(tables: dict[str, pd.DataFrame]) -> pypsa.Network:
        """The port's tables as a PyPSA network, column for column.

        ``tables`` is the same mapping the specsolve call attaches as ``sources``.

        ``flex`` is the one asset that is **both** ``committable`` and
        ``p_nom_extendable`` — the intersection this model is about. ``peak`` is
        extendable and not committable, so it is the unit the load falls back on
        when ``flex`` cannot run below its minimum. Both ``*_time_before`` are 0 and
        the start-up and shut-down costs stay at their default 0: the transitions are
        ported by ``pypsa_unit_commitment`` and would give a mismatch here a second
        thing to be about.
        """
        n = pypsa.Network()
        n.set_snapshots(tables['snapshot']['snapshot'])
        n.add('Bus', 'hub')

        generators: pd.DataFrame = tables['generator'].set_index('generator')
        committable = generators.index == 'flex'
        n.add(
            'Generator',
            generators.index,
            bus='hub',
            committable=committable,
            up_time_before=0,
            down_time_before=0,
            p_nom_extendable=True,
            p_min_pu=tables['p_min_pu'].set_index('generator')['value'],
            p_nom_max=tables['p_nom_max'].set_index('generator')['value'],
            marginal_cost=tables['marginal_cost'].set_index('generator')['value'],
            capital_cost=tables['capital_cost'].set_index('generator')['value'],
        )

        n.add('Load', 'l', bus='hub', p_set=tables['load'].set_index('snapshot')['value'])
        return n
    ```

**The commitment binds, and the relaxation says by how much.** `flex` builds 75
and runs 75, 30, 75, 0. At snapshot 1 it sits exactly on its floor
(`0.4 × 75 = 30`). At snapshot 3 the load of 20 is below that floor, so it
shuts off and `peak` covers. Relax the status to `[0, 1]` on the same rows and
the answer drops to **18200.0**, with a third of a power station committed. That
gap is what the integrality is worth here.

**The big-M is written, not inferred.** `big_m` is a parameter like any other,
and its *presence* marks a unit as committed. The `status` variable and the two
commitment rows carry `where: big_m`. Nothing checks that the number is large
enough. Too small an *M* silently cuts the feasible set, as it would in PyPSA.
[#220](https://github.com/fluxopt/specsolve/issues/220) asks for a big-M derived
from the declared `p_nom_max` rather than supplied beside it.

## PyPSA's own relaxation of this model is not a relaxation

Running the same instance with `linearized_unit_commitment=True` reaches
**32100.0**, *above* the integer 21700.0, with `flex` pinned off in every
snapshot:

```
Generator-com-p-current:       +1 Generator-p[2, flex] - 0 Generator-status[2, flex] <= -0.0
Generator-com-partly-start-up: +1 Generator-p[2, flex] - 1 Generator-p[1, flex] - 0 Generator-status[2, flex] <= -0.0
```

The tightening block PyPSA adds where start-up and shut-down costs match is
built from the `p_nom` **column**, which is 0 for a unit not yet built. Every
coefficient collapses and the rows read `p <= 0`. Filed as
[#989](https://github.com/fluxopt/specsolve/issues/989) to report upstream. That
is why the relaxation above is taken on the port's own model instead.

## What it exercises

A variable masked by the presence of a parameter (`where: big_m` on `status`), a
constant on the right of a `>=` row that only exists where that parameter does,
and the one product in PyPSA's committable set that needs a linearisation
constant rather than a rewrite.
