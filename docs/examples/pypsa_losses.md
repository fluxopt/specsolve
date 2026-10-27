# PyPSA transmission losses — a quadratic curve, held up by its own tangents

The loss on a line is `r · s²`. PyPSA approximates it from below with a fan of tangents.

> **✔ Verified against pypsa 1.2.4 (its own linopy 0.10.0)** — objective **24114.237385131008**, matched to `rtol=1e-09`.

For each segment *k* PyPSA takes the point `p_k = k/segments · s_nom`, draws the
tangent to the loss curve there, and adds it once for each sign of the flow.
Each tangent is a half-plane on `(loss, s)`, so the approximation is linear
rows and no auxiliary variable. The objective pushes the loss down, the
tangents hold it up, and it settles on the curve.

Six snapshots, three busy and three quiet, so the flows reach the early
segments of the fan as well as the top. With the busy snapshots alone only two
of five half-planes bind, and the other three coefficients could be anything
without the port noticing.

The network is a path, `b0—b1—b2—b3`. With no independent cycle there is no
voltage law to satisfy, so a mismatch implicates the loss approximation rather
than [Kirchhoff's voltage law](pypsa_kvl.md). The last line has no resistance.
PyPSA gives every passive branch a loss variable and lets `r = 0` pin it to
nothing; the port declares one only where there is a curve to approximate.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

PyPSA transmission losses, tangent form: the quadratic loss on a line, underestimated from below by a fan of tangent half-planes and subtracted half at each end. No auxiliary variable and no piecewise construct — a tangent is one linear row per segment and per sign of the flow. Optimum 3692.705905599654, from PyPSA itself.

#### Sets

| Symbol | Meaning |
|---|---|
| $`\mathcal{T}`$ | index $`t`$ — `snapshot` — dispatch periods |
| $`\mathcal{B}`$ | index $`b`$ — `bus` with $`\mathrm{gen\_bus}: \mathcal{G} \to \mathcal{B},\ \mathrm{from}: \mathcal{L} \to \mathcal{B},\ \mathrm{to}: \mathcal{L} \to \mathcal{B}`$ — network nodes |
| $`\mathcal{G}`$ | index $`g`$ — `generator` with $`\mathrm{gen\_bus}: \mathcal{G} \to \mathcal{B}`$ — generating units, each sitting on one bus |
| $`\mathcal{L}`$ | index $`l`$ — `line` with $`\mathrm{from}: \mathcal{L} \to \mathcal{B},\ \mathrm{to}: \mathcal{L} \to \mathcal{B}`$ — passive branches, each joining two buses |
| $`\mathcal{S}`$ | index $`s`$ — `segment` — the tangent points the loss curve is approximated at |

#### Parameters

| Symbol | Meaning |
|---|---|
| $`\mathrm{p}^{\mathrm{nom}}`$ | `p_nom` over $`\mathcal{G}`$ — installed capacity of a generator |
| $`\mathrm{marginal\_cost}`$ | `marginal_cost` over $`\mathcal{G}`$ — cost of one unit of output |
| $`\mathrm{s}^{\mathrm{nom}}`$ | `s_nom` over $`\mathcal{L}`$ — most a line may carry towards its `to` bus |
| $`\mathrm{neg\_s\_nom}`$ | `neg_s_nom` over $`\mathcal{L}`$ — most a line may carry the other way, negative by convention |
| $`\mathrm{loss}^{\mathrm{max}}`$ | `loss_max` over $`\mathcal{L}`$, `absent` where the data has no row — the loss at a line's rating — the top of the curve being approximated, carried as a column because a bound takes a name or a number, and given only for the lines that dissipate anything |
| $`\mathrm{loss}^{\mathrm{slope}}`$ | `loss_slope` over $`\mathcal{L} \times \mathcal{S}`$, `absent` where the data has no row — the slope of this segment's half-plane — how much loss the flow buys along it. Where the segments come from is the instance's business, not the model's: a tangent to the loss curve and a secant across it both arrive here as a slope and an offset. |
| $`\mathrm{loss}^{\mathrm{offset}}`$ | `loss_offset` over $`\mathcal{L} \times \mathcal{S}`$, `absent` where the data has no row — where this segment's half-plane meets the loss axis, negative for a curve through the origin |
| $`\mathrm{load}`$ | `load` over $`\mathcal{T} \times \mathcal{B}`$ — demand at each bus in each snapshot |

#### Variables

| Symbol | Meaning |
|---|---|
| $`p`$ | `p` over $`\mathcal{T} \times \mathcal{G}`$ — output of a generator in a snapshot |
| $`f`$ | `f` over $`\mathcal{T} \times \mathcal{L}`$ — flow on a line, signed towards its `to` bus — unbounded here, because the rating covers the flow and its loss and so is a row rather than a bound |
| $`\mathit{loss}`$ | `loss` over $`\mathcal{T} \times \mathcal{L}`$, `neutral` where the mask leaves it out — the energy a line dissipates carrying its flow — pushed down by the objective and held up by the tangents, so it settles on the approximated curve rather than needing an equality of its own. A line with no resistance dissipates nothing, which is a loss of zero rather than a quantity with no value, so the balances and ratings that name it keep their rows. |

#### Variable domains

**`p`**

```math
0 \le p_{t,g} \le \mathrm{p}^{\mathrm{nom}}_{g} \qquad \forall\, t \in \mathcal{T},\ g \in \mathcal{G}
```

**`f`**

```math
f_{t,l} \in \mathbb{R} \qquad \forall\, t \in \mathcal{T},\ l \in \mathcal{L}
```

**`loss`**

```math
0 \le \mathit{loss}_{t,l} \le \mathrm{loss}^{\mathrm{max}}_{l} \qquad \forall\, t \in \mathcal{T},\ l \in \mathcal{L} \,:\, \mathrm{loss}^{\mathrm{max}}_{l} \text{ is defined}
```

#### Objective

```math
\min \sum_{t \in \mathcal{T},\ g \in \mathcal{G}} p_{t,g} \cdot \mathrm{marginal\_cost}_{g}
```

#### Subject to

**`nodal_balance`**

```math
\sum_{g \in \mathcal{G} \,:\, \mathrm{gen\_bus}(g) = b} p_{t,g} + \sum_{l \in \mathcal{L} \,:\, \mathrm{to}(l) = b} f_{t,l} - \left( \sum_{l \in \mathcal{L} \,:\, \mathrm{from}(l) = b} f_{t,l} \right) - 0.5 \cdot \left( \sum_{l \in \mathcal{L} \,:\, \mathrm{from}(l) = b} \mathit{loss}_{t,l} \right) - 0.5 \cdot \left( \sum_{l \in \mathcal{L} \,:\, \mathrm{to}(l) = b} \mathit{loss}_{t,l} \right) = \mathrm{load}_{t,b} \qquad \forall\, t \in \mathcal{T},\ b \in \mathcal{B}
```

**`within_rating_forward`**

```math
f_{t,l} + \mathit{loss}_{t,l} \le \mathrm{s}^{\mathrm{nom}}_{l} \qquad \forall\, t \in \mathcal{T},\ l \in \mathcal{L}
```

**`within_rating_reverse`**

```math
f_{t,l} - \mathit{loss}_{t,l} \ge \mathrm{neg\_s\_nom}_{l} \qquad \forall\, t \in \mathcal{T},\ l \in \mathcal{L}
```

**`loss_above_segment_forward`**

```math
\mathit{loss}_{t,l} + \mathrm{loss}^{\mathrm{slope}}_{l,s} \cdot f_{t,l} \ge \mathrm{loss}^{\mathrm{offset}}_{l,s} \qquad \forall\, t \in \mathcal{T},\ l \in \mathcal{L},\ s \in \mathcal{S} \,:\, \mathrm{loss}^{\mathrm{max}}_{l} \text{ is defined}
```

**`loss_above_segment_reverse`**

```math
\mathit{loss}_{t,l} - \mathrm{loss}^{\mathrm{slope}}_{l,s} \cdot f_{t,l} \ge \mathrm{loss}^{\mathrm{offset}}_{l,s} \qquad \forall\, t \in \mathcal{T},\ l \in \mathcal{L},\ s \in \mathcal{S} \,:\, \mathrm{loss}^{\mathrm{max}}_{l} \text{ is defined}
```

</details>
<!-- math:end -->

The tabs start from [the instance's tables](../howto/data.md) — one frame per parameter.

=== "specsolve"

    ```yaml
    description: >-
      PyPSA transmission losses, tangent form: the quadratic loss on a line,
      underestimated from below by a fan of tangent half-planes and subtracted half
      at each end. No auxiliary variable and no piecewise construct — a tangent is
      one linear row per segment and per sign of the flow.
      Optimum 3692.705905599654, from PyPSA itself.

    dimensions:
      snapshot:
        description: dispatch periods
        dtype: int
        ordered: true
      bus:
        description: network nodes
        dtype: str
      generator:
        description: generating units, each sitting on one bus
        dtype: str
      line:
        description: passive branches, each joining two buses
        dtype: str
      segment:
        description: the tangent points the loss curve is approximated at
        dtype: int
        ordered: true

    relations:
      gen_bus:
        description: the bus a generator sits on
        key: generator
        values: bus
      from:
        description: the bus a line leaves
        key: line
        values: bus
      to:
        description: the bus a line arrives at
        key: line
        values: bus

    parameters:
      p_nom:
        description: installed capacity of a generator
        dims: [generator]
      marginal_cost:
        description: cost of one unit of output
        dims: [generator]
      s_nom:
        description: most a line may carry towards its `to` bus
        dims: [line]
      neg_s_nom:
        description: most a line may carry the other way, negative by convention
        dims: [line]
      loss_max:
        description: >-
          the loss at a line's rating — the top of the curve being approximated,
          carried as a column because a bound takes a name or a number, and given
          only for the lines that dissipate anything
        dims: [line]
        missing: absent
      loss_slope:
        description: >-
          the slope of this segment's half-plane — how much loss the flow buys
          along it. Where the segments come from is the instance's business, not
          the model's: a tangent to the loss curve and a secant across it both
          arrive here as a slope and an offset.
        dims: [line, segment]
        missing: absent
      loss_offset:
        description: >-
          where this segment's half-plane meets the loss axis, negative for a curve
          through the origin
        dims: [line, segment]
        missing: absent
      load:
        description: demand at each bus in each snapshot
        dims: [snapshot, bus]

    variables:
      p:
        description: output of a generator in a snapshot
        dims: [snapshot, generator]
        bounds:
          lower: 0
          upper: p_nom
      f:
        description: >-
          flow on a line, signed towards its `to` bus — unbounded here, because the
          rating covers the flow and its loss and so is a row rather than a bound
        dims: [snapshot, line]
      loss:
        description: >-
          the energy a line dissipates carrying its flow — pushed down by the
          objective and held up by the tangents, so it settles on the approximated
          curve rather than needing an equality of its own. A line with no
          resistance dissipates nothing, which is a loss of zero rather than a
          quantity with no value, so the balances and ratings that name it keep
          their rows.
        dims: [snapshot, line]
        where: loss_max
        missing: neutral
        bounds:
          lower: 0
          upper: loss_max

    constraints:
      nodal_balance:
        description: >-
          what is generated at a bus plus what arrives over the lines meets the load
          there, less half of each incident line's loss — PyPSA's convention is that
          a branch dissipates half at either end
        dims: [snapshot, bus]
        expression: >-
          sum(p, over=generator, by=gen_bus[bus])
          + sum(f, over=line, by=to[bus]) - sum(f, over=line, by=from[bus])
          - 0.5 * sum(loss, over=line, by=from[bus]) - 0.5 * sum(loss, over=line, by=to[bus])
          == load

      within_rating_forward:
        description: >-
          a line's rating limits what it carries plus what it dissipates, so the
          loss eats into the capacity rather than riding on top of it
        dims: [snapshot, line]
        expression: f + loss <= s_nom

      within_rating_reverse:
        description: the same limit for flow the other way
        dims: [snapshot, line]
        expression: f - loss >= neg_s_nom

      loss_above_segment_forward:
        description: >-
          the loss sits above every one of its half-planes, which for a convex
          curve is the whole approximation. Only the lines that have a curve get a
          fan.
        dims: [snapshot, line, segment]
        where: loss_max
        expression: loss + loss_slope * f >= loss_offset

      loss_above_segment_reverse:
        description: the same fan mirrored, because the loss depends on the flow's magnitude
        dims: [snapshot, line, segment]
        where: loss_max
        expression: loss - loss_slope * f >= loss_offset

    objective:
      sense: minimize
      description: what the fleet costs to run; the losses are paid for as extra generation
      expression: sum(p * marginal_cost)
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with sps.solve('examples/ports/pypsa_losses.yaml', sources) as solution:
        solution.objective  # 24114.237385131008
        solution.dual('nodal_balance')
    ```

=== "PyPSA"

    The model-building half of `examples/ports/references/pypsa/pypsa_losses.py`:

    ```python
    def build(tables: dict[str, pd.DataFrame]) -> pypsa.Network:
        """The port's tables as a PyPSA network, column for column.

        ``tables`` is the same mapping the specsolve call attaches as ``sources``.

        PyPSA is given ``r``, ``x`` and ``s_nom`` and derives the tangents itself.
        The port is given the tangents, because a slope of ``2 * r * p_k`` is
        arithmetic and the language's coefficients take a name or a number — the
        same reason ``pypsa_storage`` ships ``soc_max`` rather than a ratio. Both
        sides therefore describe one model from the same instance, and
        ``SEGMENTS`` is the one number that has to agree between them.
        """
        n = pypsa.Network()
        n.set_snapshots(tables['snapshot']['snapshot'])
        n.add('Bus', tables['bus']['bus'])

        lines: pd.DataFrame = tables['line'].set_index('line')
        n.add(
            'Line',
            lines.index,
            bus0=tables['from'].set_index('line')['bus'],
            bus1=tables['to'].set_index('line')['bus'],
            r=tables['r'].set_index('line')['value'],
            x=tables['x'].set_index('line')['value'],
            s_nom=tables['s_nom'].set_index('line')['value'],
        )

        generators: pd.DataFrame = tables['generator'].set_index('generator')
        n.add(
            'Generator',
            generators.index,
            bus=tables['gen_bus'].set_index('generator')['bus'],
            p_nom=tables['p_nom'].set_index('generator')['value'],
            marginal_cost=tables['marginal_cost'].set_index('generator')['value'],
        )

        load: pd.DataFrame = tables['load'].pivot(index='snapshot', columns='bus', values='value')
        for bus in load.columns:
            n.add('Load', f'load_{bus}', bus=bus, p_set=load[bus])
        return n
    ```

**The rating covers the flow *and* its loss.** With losses enabled PyPSA
replaces the flow bound `|s| ≤ s_nom` with the rows `s + loss ≤ s_nom` and
`s − loss ≥ −s_nom`. The loss eats into the capacity rather than riding on top
of it. Keeping the bound instead matches PyPSA at snapshots 0 and 2 and pushes
the flow to 120 at snapshot 1, where PyPSA stops at 115.97 plus 4.03 of loss.

**Losses put a gradient in the prices.** Between `b0` and `b2` the recorded duals
climb `10.00 → 85.78 → 90.00`: power is worth more the further it travels,
which no objective figure shows. Across the lossless line the price does not
move. `b3` differs from `b2` only because the local generator sets it.

**The mask needs `missing: neutral`.** `loss` is declared `where: loss_max`, so
the resistanceless line has no loss variable. `loss` also appears as a bare
term in the rating rows, `f + loss <= s_nom`, and a constraint naming a masked
variable loses its row. Without `missing: neutral` those two rows vanish for that
line, it becomes uncapacitated, and the model reports a cheaper answer:

| spelling | rows | `omissions` | objective |
|---|---|---|---|
| `where:` + `missing: neutral` | 132 | 0 | **24114.24** ✔ |
| `where:` alone | 120 | **2** | 17514.24 ✘ |

Twenty-seven per cent low, with an optimal status. `diagnostics().omissions`
([#944](https://github.com/fluxopt/specsolve/issues/944)) counts the two rating
rows a propagated absence deleted, so the wrong model announces itself.

**`r` is 0.0003, not a per-unit textbook figure.** PyPSA's loss term is
`r_pu_eff · s²` with `s` in MW, so a resistance chosen for a per-unit base
makes the loss exceed the flow. At `r = 0.05` this instance is infeasible: the
generators cannot cover a loss larger than the demand. At 0.0003 losses run
about 3% of throughput.

**PyPSA's other loss mode is this same model.** Its default is secants rather
than tangents. Secants lie above a convex curve where tangents lie below, so
they overestimate the losses these underestimate. PyPSA emits the identical
rows, one half-plane per segment per sign of the flow. Only the coefficients
differ, and how many there are. So the secant mode gets no model
file of its own. `test_the_two_loss_approximations_are_one_model` solves this
model with the secant coefficients and reaches PyPSA's secant optimum.

The parameters are therefore `loss_slope` and `loss_offset`, with nothing
tangent-specific in the name, and the coefficients are dumped from PyPSA
rather than recomputed here. Where the breakpoints fall is the instance's
business; a secant's come out of an error tolerance, and so does their number.
The model asks only for a slope and an offset per segment.

## What it exercises

A third dimension that exists only to index an approximation: `segment` is not
a thing in the network, it is a row multiplier. And a variable pinned between
an objective pushing down and a fan of constraints pushing up, with no
equality defining it.

The tangent slopes and offsets ship as data. `2 · r · p_k` is arithmetic, and a
coefficient here takes a name or a number, the same reason
[storage units](pypsa_storage.md) ships `soc_max` rather than a ratio. This
port needs no construct the language lacks.

[`method: lp`](https://mathspec.readthedocs.io/en/latest/reference/language/piecewise/#method)
emits rows of this shape, one linear row per piece and no auxiliary variable,
but it is not a drop-in here. It states the lines through consecutive
breakpoints, which for a convex curve lie above it; PyPSA's tangents lie
below. The two bracket `r · s²` from opposite sides, so swapping one for the
other moves the optimum. This port keeps the fan PyPSA publishes.
