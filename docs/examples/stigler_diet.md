# Stigler's diet problem

The cheapest way to eat for a year and stay alive. 77 foods, 9 nutrients, 1939 prices.

> **✔ Verified against linopy 0.10.0** — objective **0.10866227820675685** dollars/day, matched to `rtol=1e-09`.
> **Corroborated by Laderman (1947)**, who published **$39.69/year** for this data.

Stigler posed it in 1945 and got $39.93 by trial and error, admitting there
was "no direct method" to do better. In 1947 Jack Laderman at the National
Bureau of Standards took it as the first large test of Dantzig's simplex
method: nine clerks on desk calculators, about **120 man-days**, for $39.69.

Every other verified model is a flow of something through a network. This one
has no network: it is a **covering LP**, `min 1ᵀx` subject to `Ax ≥ b`, a
different shape of problem reaching the same engine.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

Stigler's diet problem (1945): the cheapest set of foods meeting a year's nutritional minimums. Stigler's table is normalised per dollar spent, so a variable is money on a food per day rather than a quantity, and the objective is simply the total. The table is sparse on purpose — a food supplying none of a nutrient has no row, which is how this language spells absence everywhere, and 570 of the 693 cells are non-zero.

#### Sets

| Symbol | Meaning |
|---|---|
| $`\mathcal{F}`$ | index $`f`$ — `food` — the 77 foods Stigler priced, at 1939 prices |
| $`\mathcal{N}`$ | index $`n`$ — `nutrient` — the nine nutrients a year's diet has to supply |

#### Parameters

| Symbol | Meaning |
|---|---|
| $`\mathrm{nutrient\_per\_dollar}`$ | `nutrient_per_dollar` over $`\mathcal{F} \times \mathcal{N}`$, `neutral` where the data has no row — how much of each nutrient a dollar of each food buys |
| $`\mathrm{daily\_minimum}`$ | `daily_minimum` over $`\mathcal{N}`$ — how much of a nutrient a day has to supply |

#### Variables

| Symbol | Meaning |
|---|---|
| $`\mathit{spend}`$ | `spend` over $`\mathcal{F}`$ — dollars per day spent on this food |

#### Variable domains

**`spend`**

```math
\mathit{spend}_{f} \ge 0 \qquad \forall\, f \in \mathcal{F}
```

#### Objective

```math
\min \sum_{f \in \mathcal{F}} \mathit{spend}_{f}
```

#### Subject to

**`meet_requirement`**

```math
\sum_{f \in \mathcal{F}} \mathit{spend}_{f} \cdot \mathrm{nutrient\_per\_dollar}_{f,n} \ge \mathrm{daily\_minimum}_{n} \qquad \forall\, n \in \mathcal{N}
```

</details>
<!-- math:end -->

The tabs start from [the instance's tables](../howto/data.md) — one frame per parameter.

=== "specsolve"

    ```yaml
    description: >-
      Stigler's diet problem (1945): the cheapest set of foods meeting a year's
      nutritional minimums. Stigler's table is normalised per dollar spent, so a
      variable is money on a food per day rather than a quantity, and the objective
      is simply the total. The table is sparse on purpose — a food supplying none
      of a nutrient has no row, which is how this language spells absence
      everywhere, and 570 of the 693 cells are non-zero.

    dimensions:
      food:
        description: the 77 foods Stigler priced, at 1939 prices
        dtype: str
      nutrient:
        description: the nine nutrients a year's diet has to supply
        dtype: str

    parameters:
      nutrient_per_dollar:
        description: how much of each nutrient a dollar of each food buys
        dims: [food, nutrient]
        missing: neutral
      daily_minimum:
        description: how much of a nutrient a day has to supply
        dims: [nutrient]

    variables:
      spend:
        description: dollars per day spent on this food
        dims: [food]
        bounds:
          lower: 0

    constraints:
      meet_requirement:
        description: what the basket buys of a nutrient covers the daily minimum
        dims: [nutrient]
        expression: sum(spend * nutrient_per_dollar, over=food) >= daily_minimum

    objective:
      sense: minimize
      description: dollars a day, which is what the variables already are
      expression: sum(spend)
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with sps.solve('examples/ports/stigler_diet.yaml', sources) as solution:
        solution.objective  # 0.10866227820675685
        solution.dual('meet_requirement')
    ```

=== "linopy"

    The model-building half of `examples/ports/references/linopy/stigler_diet.py`:

    ```python
    def build(tables: dict[str, pd.DataFrame]) -> linopy.Model:
        """The port's tables as a linopy model, column for column.

        ``tables`` is the same mapping the specsolve call attaches as ``sources``.
        ``per_dollar`` is the sparse table filled back out: a missing
        (food, nutrient) pair means that food supplies none of that nutrient.
        """
        foods = pd.Index(tables['food']['food'], name='food')
        minimum: pd.Series = tables['daily_minimum'].set_index('nutrient')['value']
        per_dollar: pd.DataFrame = (
            tables['nutrient_per_dollar']
            .pivot(index='food', columns='nutrient', values='value')
            .reindex(index=foods, columns=minimum.index)
            .fillna(0.0)
        )

        m = linopy.Model()
        spend = m.add_variables(lower=0, coords=[foods], name='spend')
        m.add_constraints((spend * per_dollar).sum('food') >= minimum, name='meet_requirement')
        m.add_objective(spend.sum())
        return m
    ```

Stigler's table is normalised **per dollar spent**, so a variable is *money on
a food per day* rather than a quantity, and the objective is the total. That
framing is his, and it makes the matrix price-independent.

**The nutrient table is sparse and stays that way.** 570 of the 693
(food, nutrient) cells are non-zero; a food supplying none of a nutrient has no
row. Row absence is how this language spells "not present" everywhere, and
here it means what a reader would assume.

## What it finds

| food | $/year |
|---|---|
| navy beans (dried) | 22.28 |
| wheat flour (enriched) | 10.77 |
| cabbage | 4.09 |
| spinach | 1.83 |
| beef liver | 0.69 |
| **total** | **39.66** |

**Those are the five foods in the historical solution.** The 0.08% gap against
Laderman's $39.69 is his rounding, not a different model. Matching the
*composition* is the stronger corroboration: two routes to the same five foods
out of seventy-seven.

## What a nutrient costs

Each dual is what one more unit of that nutrient per day would cost:

| nutrient | shadow price |
|---|---|
| calcium | 0.0317 |
| vitamin B2 | 0.0164 |
| calories | 0.0088 |
| vitamin A | 0.0004 |
| vitamin C | 0.00014 |
| protein · iron · vitamin B1 · niacin | **0** |

Four of the nine requirements cost nothing at the margin: they arrive free
alongside the ones that bind. That lesson is a dual, not a primal, which is
why the corpus checks duals as well as objectives.

## What it exercises

A two-dimensional parameter multiplying a one-dimensional variable, reduced
along the shared dimension: the shape [KVL](pypsa_kvl.md) needs for its cycle
incidence, doing a different job. Also a bare variable as the whole objective,
which is as small as an objective gets.
