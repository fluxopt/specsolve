# Verification

How we know an answer is right: four checks, from the cheapest to the most
convincing. Each one catches what the one before cannot.

| Check | It catches | Where |
|---|---|---|
| [The oracle](#the-oracle) | a build that differs from linopy's on the same file | `tests/linopy_lane/`, the differential tests |
| [Published optima](#published-optima) | a misreading the oracle shares | `examples/ports/`, `references.json` |
| [The PyPSA ladder](#the-pypsa-ladder) | a model that solves but is not PyPSA's | `differential/pypsa/`, [the ladder](../examples/pypsa_ladder.md) |
| [Decomposition](#decomposition) | an algorithm outside the language that cannot be written over it | [`examples/benders/`](https://github.com/fluxopt/specsolve/blob/main/examples/benders/README.md) |

## The oracle

The test suite builds every file twice: through the relational engine, and as
a `linopy.Model` in `tests/linopy_lane/`. The differential tests compare the
two. Both builds pass the same `_inputs.lowered` gate and read data through
`_sources.tidy_sources`, so both refuse the same constructs and the same
malformed sources ([#60](https://github.com/fluxopt/specsolve/issues/60)).

**Where a concept is already linopy's, specsolve copies its name**: solve
statuses, `is_ok`, the shape of a result. `tests/test_solve_status.py` checks
that the copy still matches.

### What a construct becomes in the oracle

Each row lives in `tests/linopy_lane/builder.py`.

| Declaration | linopy |
|---|---|
| `variables:` | `Model.add_variables(lower, upper, coords, name, mask, binary, integer)` |
| `sos:` | `Model.add_sos_constraints(variable, sos_type, sos_dim)`, the block handed over rather than a formulation rebuilt |
| `constraints:` | `Model.add_constraints(lhs, sign, rhs, name, mask)`, one rule per declaration |
| `objective:` | `Model.add_objective(expr, sense)`, each additive term summed over the dims it carries |
| `expressions:` | evaluated at the solution as xarray arithmetic, every variable its `.solution` and every `dual(c)` the constraint's `.dual`; an entry the math never reads is read at whatever degree it was written |

| In an expression | linopy or xarray |
|---|---|
| `x` — a variable | `Model.variables['x']`, `.fillna(0)` under `missing: neutral` |
| `p` — a parameter | its `xr.DataArray`, `.fillna(0.0)` where it stands as a coefficient |
| `+` `-` `*` `/` | the Python operators linopy overloads |
| `sum(x, over=t)` | `.sum('t')` |
| `sum(x, over=c, by=r[d])` | the column walked to attached as a coordinate, then `.groupby()`, reindexed onto its dimension's declared labels; a walk to two columns groups onto both at once, and a key of several columns groups by the value and the rest of the key |
| `at(p, by=r[c])` | `.sel({into: relation})`, xarray's vectorised selection; a read of two columns reads a tuple of labels at once |
| `shift(x, along=t, offset=n)` | `.shift({t: n})`; `.roll({t: n})` under `edge: wrap`; a `.sel()` gather where the offset differs per entity or `by=` groups it |
| `sum_back(x, along=t, window=w)` | a sum of `w` scalar gathers, each unreachable position contributing zero; under `by=` each gather reads inside the group, so the window stops at its edge |
| `dual(c)` | `Model.constraints['c'].dual`, at a read only; the language keeps a dual out of the math, and a solve that stored none refuses the read |

| A `where:` | linopy |
|---|---|
| on a declaration | the `mask=` argument; a mask that excludes nothing is passed as `None` |
| `defined(x)` | `Model.variables['x'].labels != -1`, linopy's own marker for an absent slot |
| a comparison | the Python comparison operators element-wise, absence reading as false |

A missing row of a `missing: neutral` parameter is zero in a coefficient, an
error in `bounds:`, and false in a `where` operand
(`tests/linopy_lane/absence.py`).

### The same language, but not every build

Three files pass `check` and build on one side only.

- **An objective with a constant** builds only in specsolve, because
  `linopy.Objective` has no slot for one (`osemosys_utopia`,
  [#894](https://github.com/fluxopt/specsolve/issues/894)).
- **A relation that is not a single-valued map** builds only in specsolve,
  because the oracle holds each relation as one dense array.
- **An operator along a dimension a constant term does not carry**, such as
  `sum(x * k + d, over=t)` with a scalar `d`, builds only in the oracle. The
  refusal says to declare the parameter over the dimension
  ([#1137](https://github.com/fluxopt/specsolve/issues/1137)).

## Published optima

The oracle cannot catch a misreading both builds share. A published optimum
can. Each model in `examples/ports/` carries an optimum somebody else solved,
in `references.json`, and `tests/test_ports.py` solves to it. A **✔** means the
optimum did not come from specsolve; the reference column says where it did.
A `duals` tick means the shadow prices are checked against the reference's
too. A MILP has no duals; those are the `·` rows.

<!-- references:begin -->
| port | optimum | `rtol` | duals | reference |
|---|---|---|---|---|
| [dispatch](../examples/dispatch.md) | 10500.0 | 1e-09 | **✔** | linopy 0.10.0, via examples/ports/references/linopy/dispatch.py — agreement, not a published figure |
| [facility_location](../examples/facility_location.md) | 932615.75 | 1e-09 | · | published by OR-Library (Beasley) for instance cap71 of the uncapacitated warehouse location set, in the file uncapopt: http://people.brunel.ac.uk/~mastjjb/jeb/orlib/uncapinfo.html |
| [genx_piecewise_fuel](../examples/genx_piecewise_fuel.md) | 2341.8230753008093 | 1e-09 | · | published by GenX: asserted in test/test_piecewisefuel.jl as obj_true = 2341.82308 under genx_setup UCommit=2, CO2Cap=1, ParameterScale=1, and reproduced here by running GenX itself (julia 1.12.6, HiGHS) which reports 2341.8230753008093 |
| [monthly_budget](../examples/monthly_budget.md) | 9500.0 | 1e-09 | **✔** | linopy 0.10.0, via examples/ports/references/linopy/monthly_budget.py — agreement, not a published figure |
| [multi_period](../examples/multi_period.md) | 10020.0 | 1e-09 | **✔** | linopy 0.10.0, via examples/ports/references/linopy/multi_period.py — agreement, not a published figure |
| [osemosys_utopia](../examples/osemosys_utopia.md) | 29446.86269 | 1e-09 | · | published by OSeMOSYS: asserted in OSeMOSYS_GNU_MathProg tests/test_gnu_mathprog.py as obj = 2.944686269e+04 for tests/utopia.txt, and reproduced here by running GLPK directly (glpsol 5.0, src/osemosys.txt) — an oracle outside Python entirely |
| [piecewise](../examples/piecewise.md) | 3850.0 | 1e-09 | **✔** | linopy 0.10.0, via examples/ports/references/linopy/piecewise.py — agreement, not a published figure |
| [piecewise_conversion](../examples/piecewise_conversion.md) | 5990.0 | 1e-06 | · | linopy 0.10.0's own add_piecewise_formulation, via examples/ports/references/linopy/piecewise_conversion.py — agreement, not a published figure. There the arity is an argument list built per converter; here it is a constraint over flow. The looser rtol is this model and not the lane: its weights are continuous inside a segment, so a solver's feasibility tolerance on them reaches the objective — gurobi leaves 1e-7 of weight on a neighbouring breakpoint and lands 8e-9 away |
| `piecewise_lp` | 3850.0 | 1e-09 | **✔** | the same instance as piecewise, whose optimum and duals linopy 0.10.0 fixes via examples/ports/references/linopy/piecewise.py — the two formulations agreeing is the claim |
| [piecewise_ragged](../examples/piecewise_ragged.md) | 426.0 | 1e-09 | **✔** | linopy 0.10.0, via examples/ports/references/linopy/piecewise_ragged.py — agreement between two formulations of one convex curve (weights here, segment lines there), not a published figure |
| [pypsa_ac_dc](../examples/pypsa_ac_dc.md) | 18441021.477729216 | 1e-09 | **✔** | pypsa 1.2.4 (its own linopy 0.10.0), via examples/ports/references/pypsa/pypsa_ac_dc.py — n.objective + n.objective_constant, the system cost |
| [pypsa_committable_extendable](../examples/pypsa_committable_extendable.md) | 21700.0 | 1e-09 | · | pypsa 1.2.4 (its own linopy 0.10.0), via examples/ports/references/pypsa/pypsa_committable_extendable.py |
| [pypsa_cvar](../examples/pypsa_cvar.md) | 35410.0 | 1e-09 | **✔** | pypsa 1.2.4 (its own linopy 0.10.0), via examples/ports/references/pypsa/pypsa_cvar.py |
| [pypsa_cyclic_storage](../examples/pypsa_cyclic_storage.md) | 17228.77962151063 | 1e-09 | **✔** | pypsa 1.2.4 (its own linopy 0.10.0), via examples/ports/references/pypsa/pypsa_cyclic_storage.py |
| [pypsa_energy_sum](../examples/pypsa_energy_sum.md) | 21400.0 | 1e-09 | **✔** | pypsa 1.2.4 (its own linopy 0.10.0), via examples/ports/references/pypsa/pypsa_energy_sum.py |
| [pypsa_fixed](../examples/pypsa_fixed.md) | 49900.0 | 1e-09 | **✔** | pypsa 1.2.4 (its own linopy 0.10.0), via examples/ports/references/pypsa/pypsa_fixed.py |
| [pypsa_global_limits](../examples/pypsa_global_limits.md) | 127211.66666666666 | 1e-09 | **✔** | pypsa 1.2.4 (its own linopy 0.10.0), via examples/ports/references/pypsa/pypsa_global_limits.py |
| [pypsa_growth_limit](../examples/pypsa_growth_limit.md) | 47110.0 | 1e-09 | **✔** | pypsa 1.2.4 (its own linopy 0.10.0), via examples/ports/references/pypsa/pypsa_growth_limit.py |
| [pypsa_kvl](../examples/pypsa_kvl.md) | 17000.0 | 1e-09 | **✔** | pypsa 1.2.4 (its own linopy 0.10.0), via examples/ports/references/pypsa/pypsa_kvl.py |
| [pypsa_linearized_uc](../examples/pypsa_linearized_uc.md) | 5540.0 | 1e-09 | **✔** | pypsa 1.2.4 (its own linopy 0.10.0), via examples/ports/references/pypsa/pypsa_linearized_uc.py |
| [pypsa_link_delay](../examples/pypsa_link_delay.md) | 4311.111111111111 | 1e-09 | **✔** | pypsa 1.2.4 (its own linopy 0.10.0), via examples/ports/references/pypsa/pypsa_link_delay.py |
| [pypsa_losses](../examples/pypsa_losses.md) | 24114.237385131008 | 1e-09 | **✔** | pypsa 1.2.4 (its own linopy 0.10.0), via examples/ports/references/pypsa/pypsa_losses.py |
| [pypsa_min_up_down](../examples/pypsa_min_up_down.md) | 32750.0 | 1e-09 | · | pypsa 1.2.4 (its own linopy 0.10.0), via examples/ports/references/pypsa/pypsa_min_up_down.py |
| [pypsa_mixed_cycling](../examples/pypsa_mixed_cycling.md) | 4800.0 | 1e-09 | **✔** | pypsa 1.2.4 (its own linopy 0.10.0), via examples/ports/references/pypsa/pypsa_mixed_cycling.py |
| [pypsa_modular](../examples/pypsa_modular.md) | 56700.0 | 1e-09 | · | pypsa 1.2.4 (its own linopy 0.10.0), via examples/ports/references/pypsa/pypsa_modular.py |
| [pypsa_multi_period](../examples/pypsa_multi_period.md) | 85300.0 | 1e-09 | **✔** | pypsa 1.2.4 (its own linopy 0.10.0), via examples/ports/references/pypsa/pypsa_multi_period.py |
| [pypsa_multilink](../examples/pypsa_multilink.md) | 1100.0 | 1e-09 | **✔** | pypsa 1.2.4 (its own linopy 0.10.0), via examples/ports/references/pypsa/pypsa_multilink.py |
| [pypsa_ramp](../examples/pypsa_ramp.md) | 18200.0 | 1e-09 | **✔** | pypsa 1.2.4 (its own linopy 0.10.0), via examples/ports/references/pypsa/pypsa_ramp.py |
| [pypsa_spill](../examples/pypsa_spill.md) | 3200.0 | 1e-09 | **✔** | pypsa 1.2.4 (its own linopy 0.10.0), via examples/ports/references/pypsa/pypsa_spill.py |
| [pypsa_stochastic](../examples/pypsa_stochastic.md) | 33940.0 | 1e-09 | **✔** | pypsa 1.2.4 (its own linopy 0.10.0), via examples/ports/references/pypsa/pypsa_stochastic.py |
| [pypsa_storage](../examples/pypsa_storage.md) | 15253.178322993519 | 1e-09 | **✔** | pypsa 1.2.4 (its own linopy 0.10.0), via examples/ports/references/pypsa/pypsa_storage.py |
| [pypsa_store](../examples/pypsa_store.md) | 7005.5025000000005 | 1e-09 | **✔** | pypsa 1.2.4 (its own linopy 0.10.0), via examples/ports/references/pypsa/pypsa_store.py |
| [pypsa_transport](../examples/pypsa_transport.md) | 22000.0 | 1e-09 | **✔** | pypsa 1.2.4 (its own linopy 0.10.0), via examples/ports/references/pypsa/pypsa_transport.py |
| [pypsa_unit_commitment](../examples/pypsa_unit_commitment.md) | 24900.0 | 1e-09 | · | pypsa 1.2.4 (its own linopy 0.10.0), via examples/ports/references/pypsa/pypsa_unit_commitment.py |
| [reserves](../examples/reserves.md) | 915.0 | 1e-09 | **✔** | linopy 0.10.0, via examples/ports/references/linopy/reserves.py — agreement, not a published figure |
| [stigler_diet](../examples/stigler_diet.md) | 0.10866227820675685 | 1e-09 | **✔** | linopy 0.10.0, via examples/ports/references/linopy/stigler_diet.py — dollars per day; x365 = $39.6617/year[^stigler_diet] |
| [storage](../examples/storage.md) | 5650.0 | 1e-09 | **✔** | linopy 0.10.0, via examples/ports/references/linopy/storage.py — agreement, not a published figure |
| [telephone_routing](../examples/telephone_routing.md) | 380.0 | 1e-09 | · | published by Gueret, Prins, Sevaux & Heipcke, Applications of Optimization with Xpress-MP (Dash Optimization, 2002) SS12.3.3 p. 182 — "380 out of the required 425 calls are routed"; problem and data in SS12.3, pp. 180-182 |
| [transport](../examples/transport.md) | 4400.0 | 1e-09 | **✔** | linopy 0.10.0, via examples/ports/references/linopy/transport.py — agreement, not a published figure |
| [transport_dantzig](../examples/transport_dantzig.md) | 153.675 | 1e-09 | **✔** | published with GAMS model library #1 (trnsport), after Dantzig, Linear Programming and Extensions (1963) ch. 3.3[^transport_dantzig] |
| [transport_modes](../examples/transport_modes.md) | 1715.0 | 1e-09 | · | published by Gueret, Prins, Sevaux & Heipcke, Applications of Optimization with Xpress-MP (Dash Optimization, 2002) SS10.2.3 p. 143 — "The minimum cost is EUR 1,715k"; problem and data in SS10.2, p. 142 |
| [transport_pwl](../examples/transport_pwl.md) | 8.786852757777865 | 1e-09 | · | linopy 0.10.0's own add_piecewise_formulation, via examples/ports/references/linopy/transport_pwl.py; the model is GAMS model library trnspwl (Dantzig transport with economies of scale), which publishes the formulation and its discretisation but no optimal objective |
| [tsp_mtz](../examples/tsp_mtz.md) | 2085.0 | 1e-09 | · | published by TSPLIB for instance gr17 (Groetschel, 17 cities, EXPLICIT lower-diagonal distance matrix); optimum 2085 as listed in the TSPLIB solutions file |

[^stigler_diet]: Laderman (1947) at the National Bureau of Standards published $39.69/year for this data, the first serious test of the simplex method. This LP's exact optimum is 0.08% under it — his rounding, not a different model — and both select the same five foods: wheat flour, liver, cabbage, spinach, navy beans.

[^transport_dantzig]: examples/ports/references/linopy/transport_dantzig.py — the same LP hand-written in linopy 0.10.0, which reaches 153.675 independently. Secondary: the published figure is what verifies the port.
<!-- references:end -->

Adding a port is four files and five rules:
[CONTRIBUTING.md](https://github.com/fluxopt/specsolve/blob/main/CONTRIBUTING.md#adding-a-ported-model).

## Every construct has a verified model

Each column is a construct, and a tick marks a model that uses it. A tick is a
floor, not a ceiling: one verified model does not cover every shape of a
construct, or every combination.

<!-- constructs:begin -->
| model | verified | `sum` | `sum(by=)` | `at()` | `shift` | `shift(edge='wrap')` | `where` | `bounds` | `piecewise` | `sos` | MILP |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [dispatch](../examples/dispatch.md) | **✔** 10500 | **✓** | · | · | · | · | **✓** | **✓** | · | · | · |
| [monthly_budget](../examples/monthly_budget.md) | **✔** 9500 | **✓** | **✓** | · | · | · | · | **✓** | · | · | · |
| [multi_period](../examples/multi_period.md) | **✔** 10020 | **✓** | · | **✓** | · | · | · | **✓** | · | · | · |
| [piecewise](../examples/piecewise.md) | **✔** 3850 | **✓** | · | · | **✓** | · | **✓** | **✓** | **✓** | · | · |
| [piecewise_conversion](../examples/piecewise_conversion.md) | **✔** 5990 | **✓** | · | **✓** | · | · | **✓** | **✓** | · | **✓** | · |
| `piecewise_lp` | **✔** 3850 | **✓** | · | · | **✓** | · | **✓** | **✓** | **✓** | · | · |
| [piecewise_ragged](../examples/piecewise_ragged.md) | **✔** 426 | **✓** | · | · | **✓** | · | **✓** | **✓** | **✓** | · | · |
| [reserves](../examples/reserves.md) | **✔** 915 | **✓** | **✓** | **✓** | · | · | · | **✓** | · | · | · |
| [seasons](../examples/seasons.md) | · | **✓** | · | · | · | **✓** | · | **✓** | · | · | · |
| `sos` | · | **✓** | · | · | · | · | **✓** | **✓** | **✓** | **✓** | · |
| [storage](../examples/storage.md) | **✔** 5650 | **✓** | · | · | · | **✓** | · | **✓** | · | · | · |
| [transport](../examples/transport.md) | **✔** 4400 | **✓** | **✓** | · | · | · | · | **✓** | · | · | · |
| [facility_location](../examples/facility_location.md) | **✔** 932616 | **✓** | · | · | · | · | · | **✓** | · | · | **✓** |
| [genx_piecewise_fuel](../examples/genx_piecewise_fuel.md) | **✔** 2341.82 | **✓** | · | · | · | **✓** | **✓** | **✓** | · | · | · |
| [osemosys_utopia](../examples/osemosys_utopia.md) | **✔** 29446.9 | **✓** | · | · | · | · | · | **✓** | · | · | · |
| [pypsa_ac_dc](../examples/pypsa_ac_dc.md) | **✔** 1.8441e+07 | **✓** | **✓** | **✓** | · | · | · | **✓** | · | · | · |
| [pypsa_committable_extendable](../examples/pypsa_committable_extendable.md) | **✔** 21700 | **✓** | · | · | · | · | **✓** | **✓** | · | · | **✓** |
| [pypsa_cvar](../examples/pypsa_cvar.md) | **✔** 35410 | **✓** | · | · | · | · | · | **✓** | · | · | · |
| [pypsa_cyclic_storage](../examples/pypsa_cyclic_storage.md) | **✔** 17228.8 | **✓** | **✓** | · | **✓** | **✓** | · | **✓** | · | · | · |
| [pypsa_energy_sum](../examples/pypsa_energy_sum.md) | **✔** 21400 | **✓** | **✓** | · | · | · | **✓** | **✓** | · | · | · |
| [pypsa_fixed](../examples/pypsa_fixed.md) | **✔** 49900 | **✓** | **✓** | · | · | · | **✓** | **✓** | · | · | · |
| [pypsa_global_limits](../examples/pypsa_global_limits.md) | **✔** 127212 | **✓** | **✓** | · | · | · | **✓** | **✓** | · | · | · |
| [pypsa_growth_limit](../examples/pypsa_growth_limit.md) | **✔** 47110 | **✓** | **✓** | **✓** | **✓** | · | · | **✓** | · | · | · |
| [pypsa_kvl](../examples/pypsa_kvl.md) | **✔** 17000 | **✓** | **✓** | · | · | · | · | **✓** | · | · | · |
| [pypsa_linearized_uc](../examples/pypsa_linearized_uc.md) | **✔** 5540 | **✓** | · | · | **✓** | · | · | **✓** | · | · | · |
| [pypsa_link_delay](../examples/pypsa_link_delay.md) | **✔** 4311.11 | **✓** | **✓** | · | **✓** | · | · | **✓** | · | · | · |
| [pypsa_losses](../examples/pypsa_losses.md) | **✔** 24114.2 | **✓** | **✓** | · | · | · | **✓** | **✓** | · | · | · |
| [pypsa_min_up_down](../examples/pypsa_min_up_down.md) | **✔** 32750 | **✓** | · | · | **✓** | · | **✓** | **✓** | · | · | **✓** |
| [pypsa_mixed_cycling](../examples/pypsa_mixed_cycling.md) | **✔** 4800 | **✓** | **✓** | · | **✓** | **✓** | **✓** | **✓** | · | · | · |
| [pypsa_modular](../examples/pypsa_modular.md) | **✔** 56700 | **✓** | **✓** | · | · | · | · | **✓** | · | · | **✓** |
| [pypsa_multi_period](../examples/pypsa_multi_period.md) | **✔** 85300 | **✓** | · | **✓** | · | · | · | **✓** | · | · | · |
| [pypsa_multilink](../examples/pypsa_multilink.md) | **✔** 1100 | **✓** | **✓** | · | · | · | · | **✓** | · | · | · |
| [pypsa_ramp](../examples/pypsa_ramp.md) | **✔** 18200 | **✓** | **✓** | · | **✓** | · | · | **✓** | · | · | · |
| [pypsa_spill](../examples/pypsa_spill.md) | **✔** 3200 | **✓** | **✓** | · | **✓** | · | **✓** | **✓** | · | · | · |
| [pypsa_stochastic](../examples/pypsa_stochastic.md) | **✔** 33940 | **✓** | · | · | · | · | · | **✓** | · | · | · |
| [pypsa_storage](../examples/pypsa_storage.md) | **✔** 15253.2 | **✓** | **✓** | · | **✓** | · | **✓** | **✓** | · | · | · |
| [pypsa_store](../examples/pypsa_store.md) | **✔** 7005.5 | **✓** | **✓** | · | **✓** | · | **✓** | **✓** | · | · | · |
| [pypsa_transport](../examples/pypsa_transport.md) | **✔** 22000 | **✓** | **✓** | · | · | · | · | **✓** | · | · | · |
| [pypsa_unit_commitment](../examples/pypsa_unit_commitment.md) | **✔** 24900 | **✓** | · | · | **✓** | · | **✓** | **✓** | · | · | **✓** |
| [stigler_diet](../examples/stigler_diet.md) | **✔** 0.108662 | **✓** | · | · | · | · | · | **✓** | · | · | · |
| [telephone_routing](../examples/telephone_routing.md) | **✔** 380 | **✓** | **✓** | · | · | · | · | **✓** | · | · | **✓** |
| [transport_dantzig](../examples/transport_dantzig.md) | **✔** 153.675 | **✓** | · | · | · | · | · | **✓** | · | · | · |
| [transport_modes](../examples/transport_modes.md) | **✔** 1715 | **✓** | **✓** | · | · | · | · | **✓** | · | · | · |
| [transport_pwl](../examples/transport_pwl.md) | **✔** 8.78685 | **✓** | · | · | **✓** | · | **✓** | **✓** | **✓** | · | **✓** |
| [tsp_mtz](../examples/tsp_mtz.md) | **✔** 2085 | **✓** | **✓** | · | · | · | **✓** | **✓** | · | · | **✓** |
<!-- constructs:end -->

The witness for `roll / shift` is
[ramp limits](../examples/pypsa_ramp.md), for integrality
[unit commitment](../examples/pypsa_unit_commitment.md), and for `piecewise`
[economies of scale](../examples/transport_pwl.md).

## The PyPSA ladder

[The PyPSA ladder](../examples/pypsa_ladder.md) grows a one-file PyPSA model a
feature at a time. Each rung solves the same network in PyPSA and in
specsolve, and compares the objective and the duals. `tools/ladder.py` writes
the pages from `differential/pypsa/`.

## Decomposition

[`examples/benders/`](https://github.com/fluxopt/specsolve/blob/main/examples/benders/README.md)
writes Benders decomposition as a loop over three specs.
`tests/test_benders_example.py` checks that it reaches the optimum of the
monolith.

## What a port could not say

Each row is a verdict: macro, primitive, formulation, or refused.

| Port | What could not be said | Worked around by | Verdict |
|---|---|---|---|
| PyPSA transport model | a bound of `-rating` — PyPSA's `p_min_pu = -1` | shipping `neg_rating` as data | **primitive**: bounds as expressions, [#31](https://github.com/fluxopt/specsolve/issues/31). A second model asking for it |
| Travelling salesman | subtour cuts **generated lazily** inside branch-and-cut | [MTZ](../examples/tsp_mtz.md), O(n²) and static | **refused**: a solve loop is an algorithm, not a model |
