# Examples

This page lists every model in the repository, which constructs each
exercises and whether an outside optimum checks its answer, so you can decide
whether the language can say yours.

Three questions:
**[can it say my model?](#can-it-say-my-model)** ·
**[is it readable?](#every-model)** ·
**[does it get the right answer?](#does-it-get-the-right-answer)** The first
and the third are the two tables on this page. The second is each model page,
where the file sits beside the same model written on another stack.

Every page starts from data in the shape
[preparing the data](../howto/data.md) produces.

## Every model

<!-- catalogue:begin -->
### Teaching examples

| | |
|---|---|
| [dispatch](dispatch.md) | Least-cost generation against a load profile — the smallest model that is still a model. |
| [storage](storage.md) | Dispatch plus a battery, and the only construct in the language whose cost is not obviously linear. |
| [transport](transport.md) | A network: generators sit on buses, lines connect buses, and power balances at every bus. |
| [piecewise](piecewise.md) | Per-generator convex cost curves, expanded into a λ-formulation. |
| [piecewise segment lines](piecewise_lp.md) | A piecewise-linear cost curve stated as **the lines its segments lie on** — [piecewise](piecewise.md) with one line changed, and the only method that declares no auxiliary variable at all. |
| [piecewise curves of differing length](piecewise_ragged.md) | Per-generator cost curves of **different lengths**, each as long as its own data. |
| [piecewise conversion](piecewise_conversion.md) | Converters whose flows share one curve, where **how many flows** is data. |
| [special-ordered sets](sos.md) | A piecewise-linear cost curve stated as a **special-ordered set** — [piecewise](piecewise.md) with one line changed, handed to the solver as a set it branches on itself. |
| [monthly budget](monthly_budget.md) | A cap on what each technology may generate per calendar month — an aggregate over a *coarser grouping of time*, written with the same operator that places a generator on a bus. |
| [multi-period](multi_period.md) | Capacity decided once per investment period, binding at every snapshot inside it — and the periods need not be the same size. |
| [seasons](seasons.md) | A store that cycles inside each season rather than across the horizon, with seasons of different lengths — one balance row says it, because the wrap is the season's own and no level is carried from one season into the next. |
| [reserves](reserves.md) | Energy and reserve co-optimization on a two-bus grid: offers are (generator, market, tranche) triples, reserve zones overlap, and one line dangles. The model exists to prove a claim — every many-to-many shape the language covers, in one instance, each one load-bearing. |
| [walkthrough](walkthrough.md) | The dispatch model plus a macro and a named expression — the one used to print every pipeline stage. |

### PyPSA examples

| | |
|---|---|
| [transport model](pypsa_transport.md) | PyPSA linear optimal power flow at its smallest: transport model, linear marginal cost, no KVL. |
| [ramp limits](pypsa_ramp.md) | [The transport model](pypsa_transport.md) plus a limit on how fast each generator may change output between snapshots. |
| [storage units](pypsa_storage.md) | [Ramp limits](pypsa_ramp.md) plus a `StorageUnit` carrying energy between snapshots. |
| [cyclic storage](pypsa_cyclic_storage.md) | [Storage units](pypsa_storage.md) with the horizon closed on itself: the first snapshot's state of charge carries over from the *last*. |
| [Kirchhoff's voltage law](pypsa_kvl.md) | Passive AC lines: flow is decided by physics, not chosen. |
| [transmission losses](pypsa_losses.md) | The loss on a line is `r · s²`. PyPSA approximates it from below with a fan of tangents. |
| [AC-DC, two coordinates](pypsa_ac_dc.md) | A meshed AC–DC network under a CO₂ budget. **PyPSA's own `ac-dc-meshed` example.** |
| [unit commitment](pypsa_unit_commitment.md) | Which generators are *on*, not just how much they produce — a binary per generator per snapshot, with start-up and shut-down charges. |
| [minimum up and down times](pypsa_min_up_down.md) | A unit that has started must stay on; one that has stopped must stay off. |
| [linearized unit commitment](pypsa_linearized_uc.md) | A unit may be committed by a third. PyPSA ships this as a mode, not as a debugging convenience. |
| [global limits](pypsa_global_limits.md) | Limits that hold over a whole set at once: an energy total, a capacity at one bus, and the built network measured twice. |
| [multi-link](pypsa_multilink.md) | One `Link`, one input bus, several output buses, each output derated by its own efficiency — PyPSA's spelling for a CHP plant, an electrolyser with waste heat, any conversion with more than one product. |
| [modular capacity](pypsa_modular.md) | Capacity that comes in whole modules: an integer count decides it, not a continuous bound. |
| [energy totals](pypsa_energy_sum.md) | A generator's dispatch reduced over every snapshot and bounded: a contracted delivery, a reservoir's season. |
| [fixed by data](pypsa_fixed.md) | A row of data that is present pins its variable; a row that is absent leaves it free. |
| [spillage](pypsa_spill.md) | A hydro unit takes inflow it did not choose, and spills what neither turbine nor reservoir can absorb. |
| [the Store component](pypsa_store.md) | The component every sector-coupled PyPSA model uses for hydrogen, heat and gas. |
| [mixed cycling](pypsa_mixed_cycling.md) | PyPSA's `cyclic_state_of_charge` is a per-unit flag, so one network runs both regimes at once: a unit that must end each horizon where it began, beside one handed a level it may simply spend. |
| [link delay](pypsa_link_delay.md) | A shipment is the input shifted along time: withdrawn at one snapshot, delivered at another, derated on the way. |
| [multi-period investment](pypsa_multi_period.md) | A build year and a lifetime decide which rows an asset appears in, and each period's costs carry its own discount. |
| [committable and extendable](pypsa_committable_extendable.md) | A minimum output that is a share of a capacity still being decided: two variables multiplied, and one constant to take them apart. |
| [growth limit](pypsa_growth_limit.md) | A cap on new capacity per investment period, which grows with the period before it. The first `shift` in the corpus along an axis that is not time-of-day. |
| [stochastic scenarios](pypsa_stochastic.md) | Three futures over one network: the fleet is built before anyone knows which arrives, and dispatched after. |
| [CVaR risk preference](pypsa_cvar.md) | The same three futures as [the stochastic model](pypsa_stochastic.md), planned against the tail as well as the expectation. |

### Published optima

| | |
|---|---|
| [Dantzig transport](transport_dantzig.md) | Dantzig's transportation problem — GAMS model library #1, and the oldest LP in the corpus. |
| [Dantzig, economies of scale](transport_pwl.md) | GAMS model library `trnspwl`: the same shipping problem, but a big consignment is cheaper per unit — cost grows as `sqrt(x)`, not linearly. |
| [Stigler's diet](stigler_diet.md) | The cheapest way to eat for a year and stay alive. 77 foods, 9 nutrients, 1939 prices. |
| [Facility location](facility_location.md) | Where do you put the warehouses? Open a set of them, assign every customer to one, and trade the fixed cost of opening against the cost of serving from further away. |
| [GenX piecewise fuel](genx_piecewise_fuel.md) | A day of dispatch for two carbon-capture plants and a wind farm under a net-zero carbon cap, where the gas plant's fuel use bends with its output. |
| [Routing telephone calls](telephone_routing.md) | How many of 425 requested circuits a five-city network can carry at once — and by which routes. |
| [Choosing the mode of transport](transport_modes.md) | Moving 180 tonnes of chemicals out of four depots, where a depot may reach a centre by rail *or* by road at different cost. |
| [OSeMOSYS UTOPIA](osemosys_utopia.md) | What to build and how hard to run it, 1990–2010, to meet three end-use demands at least discounted cost. |
| [Travelling salesman](tsp_mtz.md) | Visit every city once and come home, as cheaply as possible. The most famous problem in combinatorial optimisation, and the one most often assumed to be out of reach here. |
<!-- catalogue:end -->

The three blocks on this page are generated. The catalogue comes from the site
nav and each page's opening line. The constructs matrix comes from each
model's resolved plan rather than its YAML text. The reference table comes
from `examples/ports/references.json`, the file the tests assert against.
`pixi run python -m tools.constructs` regenerates all three, and a test fails
if any is stale.

Every page also carries the model **as math**, typeset from the file the
engine builds by `pixi run python -m tools.gallery_math` and gated the same
way. A model with a symbol table in `examples/symbols/` is typeset in the
notation of its own prose.

## Can it say my model?

A model here is one [spec](../reference/glossary.md) with its data attached.

<!-- constructs:begin -->
| model | verified | `sum` | `sum(by=)` | `at()` | `shift` | `shift(edge='wrap')` | `where` | `bounds` | `piecewise` | `sos` | MILP |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [dispatch](dispatch.md) | **✔** 10500 | **✓** | · | · | · | · | **✓** | **✓** | · | · | · |
| [monthly_budget](monthly_budget.md) | **✔** 9500 | **✓** | **✓** | · | · | · | · | **✓** | · | · | · |
| [multi_period](multi_period.md) | **✔** 10020 | **✓** | · | **✓** | · | · | · | **✓** | · | · | · |
| [piecewise](piecewise.md) | **✔** 3850 | **✓** | · | · | **✓** | · | **✓** | **✓** | **✓** | · | · |
| [piecewise_conversion](piecewise_conversion.md) | **✔** 5990 | **✓** | · | **✓** | · | · | **✓** | **✓** | · | **✓** | · |
| [piecewise_lp](piecewise_lp.md) | **✔** 3850 | **✓** | · | · | **✓** | · | **✓** | **✓** | **✓** | · | · |
| [piecewise_ragged](piecewise_ragged.md) | **✔** 426 | **✓** | · | · | **✓** | · | **✓** | **✓** | **✓** | · | · |
| [reserves](reserves.md) | **✔** 915 | **✓** | **✓** | **✓** | · | · | · | **✓** | · | · | · |
| [seasons](seasons.md) | · | **✓** | · | · | · | **✓** | · | **✓** | · | · | · |
| [sos](sos.md) | · | **✓** | · | · | · | · | **✓** | **✓** | **✓** | **✓** | · |
| [storage](storage.md) | **✔** 5650 | **✓** | · | · | · | **✓** | · | **✓** | · | · | · |
| [transport](transport.md) | **✔** 4400 | **✓** | **✓** | · | · | · | · | **✓** | · | · | · |
| [walkthrough](walkthrough.md) | · | **✓** | · | · | · | · | **✓** | **✓** | · | · | · |
| [facility_location](facility_location.md) | **✔** 932616 | **✓** | · | · | · | · | · | **✓** | · | · | **✓** |
| [genx_piecewise_fuel](genx_piecewise_fuel.md) | **✔** 2341.82 | **✓** | · | · | · | **✓** | **✓** | **✓** | · | · | · |
| [osemosys_utopia](osemosys_utopia.md) | **✔** 29446.9 | **✓** | · | · | · | · | · | **✓** | · | · | · |
| [pypsa_ac_dc](pypsa_ac_dc.md) | **✔** 1.8441e+07 | **✓** | **✓** | **✓** | · | · | · | **✓** | · | · | · |
| [pypsa_committable_extendable](pypsa_committable_extendable.md) | **✔** 21700 | **✓** | · | · | · | · | **✓** | **✓** | · | · | **✓** |
| [pypsa_cvar](pypsa_cvar.md) | **✔** 35410 | **✓** | · | · | · | · | · | **✓** | · | · | · |
| [pypsa_cyclic_storage](pypsa_cyclic_storage.md) | **✔** 17228.8 | **✓** | **✓** | · | **✓** | **✓** | · | **✓** | · | · | · |
| [pypsa_energy_sum](pypsa_energy_sum.md) | **✔** 21400 | **✓** | **✓** | · | · | · | **✓** | **✓** | · | · | · |
| [pypsa_fixed](pypsa_fixed.md) | **✔** 49900 | **✓** | **✓** | · | · | · | **✓** | **✓** | · | · | · |
| [pypsa_global_limits](pypsa_global_limits.md) | **✔** 127212 | **✓** | **✓** | · | · | · | **✓** | **✓** | · | · | · |
| [pypsa_growth_limit](pypsa_growth_limit.md) | **✔** 47110 | **✓** | **✓** | **✓** | **✓** | · | · | **✓** | · | · | · |
| [pypsa_kvl](pypsa_kvl.md) | **✔** 17000 | **✓** | **✓** | · | · | · | · | **✓** | · | · | · |
| [pypsa_linearized_uc](pypsa_linearized_uc.md) | **✔** 5540 | **✓** | · | · | **✓** | · | · | **✓** | · | · | · |
| [pypsa_link_delay](pypsa_link_delay.md) | **✔** 4311.11 | **✓** | **✓** | · | **✓** | · | · | **✓** | · | · | · |
| [pypsa_losses](pypsa_losses.md) | **✔** 24114.2 | **✓** | **✓** | · | · | · | **✓** | **✓** | · | · | · |
| [pypsa_min_up_down](pypsa_min_up_down.md) | **✔** 32750 | **✓** | · | · | **✓** | · | **✓** | **✓** | · | · | **✓** |
| [pypsa_mixed_cycling](pypsa_mixed_cycling.md) | **✔** 4800 | **✓** | **✓** | · | **✓** | **✓** | **✓** | **✓** | · | · | · |
| [pypsa_modular](pypsa_modular.md) | **✔** 56700 | **✓** | **✓** | · | · | · | · | **✓** | · | · | **✓** |
| [pypsa_multi_period](pypsa_multi_period.md) | **✔** 85300 | **✓** | · | **✓** | · | · | · | **✓** | · | · | · |
| [pypsa_multilink](pypsa_multilink.md) | **✔** 1100 | **✓** | **✓** | · | · | · | · | **✓** | · | · | · |
| [pypsa_ramp](pypsa_ramp.md) | **✔** 18200 | **✓** | **✓** | · | **✓** | · | · | **✓** | · | · | · |
| [pypsa_spill](pypsa_spill.md) | **✔** 3200 | **✓** | **✓** | · | **✓** | · | **✓** | **✓** | · | · | · |
| [pypsa_stochastic](pypsa_stochastic.md) | **✔** 33940 | **✓** | · | · | · | · | · | **✓** | · | · | · |
| [pypsa_storage](pypsa_storage.md) | **✔** 15253.2 | **✓** | **✓** | · | **✓** | · | **✓** | **✓** | · | · | · |
| [pypsa_store](pypsa_store.md) | **✔** 7005.5 | **✓** | **✓** | · | **✓** | · | **✓** | **✓** | · | · | · |
| [pypsa_transport](pypsa_transport.md) | **✔** 22000 | **✓** | **✓** | · | · | · | · | **✓** | · | · | · |
| [pypsa_unit_commitment](pypsa_unit_commitment.md) | **✔** 24900 | **✓** | · | · | **✓** | · | **✓** | **✓** | · | · | **✓** |
| [stigler_diet](stigler_diet.md) | **✔** 0.108662 | **✓** | · | · | · | · | · | **✓** | · | · | · |
| [telephone_routing](telephone_routing.md) | **✔** 380 | **✓** | **✓** | · | · | · | · | **✓** | · | · | **✓** |
| [transport_dantzig](transport_dantzig.md) | **✔** 153.675 | **✓** | · | · | · | · | · | **✓** | · | · | · |
| [transport_modes](transport_modes.md) | **✔** 1715 | **✓** | **✓** | · | · | · | · | **✓** | · | · | · |
| [transport_pwl](transport_pwl.md) | **✔** 8.78685 | **✓** | · | · | **✓** | · | **✓** | **✓** | **✓** | · | **✓** |
| [tsp_mtz](tsp_mtz.md) | **✔** 2085 | **✓** | **✓** | · | · | · | **✓** | **✓** | · | · | **✓** |
<!-- constructs:end -->

**Every construct has a verified model behind it**, one whose optimum came
from somebody else. The witness for `roll / shift` is
[ramp limits](pypsa_ramp.md), for integrality
[unit commitment](pypsa_unit_commitment.md), and for `piecewise`
[economies of scale](transport_pwl.md).

**A tick is a floor, not a ceiling.** One verified model exercises the
construct. It does not cover every shape of the construct, or the construct in
combination with others.

## Does it get the right answer?

**✔ means the optimum did not come from specsolve.** It came from a figure
published with the model or from a reference implementation hand-written on
another stack. The provenance column says which. Every model on this page runs
in the test suite, so a test alone distinguishes nothing. The badge marks the
one check that catches a shared misreading, the differential suite's
[blind spot](../about/linopy.md#3-it-is-the-oracle): both
[lanes](../reference/glossary.md#how-it-runs) agreeing on a meaning the
modeller did not intend. This table is the evidence behind
[the limits of the language](https://mathspec.readthedocs.io/en/latest/about/limits/#how-a-new-construct-enters).

<!-- references:begin -->
| port | optimum | `rtol` | duals | reference |
|---|---|---|---|---|
| [dispatch](dispatch.md) | 10500.0 | 1e-09 | **✔** | linopy 0.9.0, via examples/ports/references/linopy/dispatch.py — agreement, not a published figure |
| [facility_location](facility_location.md) | 932615.75 | 1e-09 | · | published by OR-Library (Beasley) for instance cap71 of the uncapacitated warehouse location set, in the file uncapopt: http://people.brunel.ac.uk/~mastjjb/jeb/orlib/uncapinfo.html |
| [genx_piecewise_fuel](genx_piecewise_fuel.md) | 2341.8230753008093 | 1e-09 | · | published by GenX: asserted in test/test_piecewisefuel.jl as obj_true = 2341.82308 under genx_setup UCommit=2, CO2Cap=1, ParameterScale=1, and reproduced here by running GenX itself (julia 1.12.6, HiGHS) which reports 2341.8230753008093 |
| [monthly_budget](monthly_budget.md) | 9500.0 | 1e-09 | **✔** | linopy 0.9.0, via examples/ports/references/linopy/monthly_budget.py — agreement, not a published figure |
| [multi_period](multi_period.md) | 10020.0 | 1e-09 | **✔** | linopy 0.9.0, via examples/ports/references/linopy/multi_period.py — agreement, not a published figure |
| [osemosys_utopia](osemosys_utopia.md) | 29446.86269 | 1e-09 | · | published by OSeMOSYS: asserted in OSeMOSYS_GNU_MathProg tests/test_gnu_mathprog.py as obj = 2.944686269e+04 for tests/utopia.txt, and reproduced here by running GLPK directly (glpsol 5.0, src/osemosys.txt) — an oracle outside Python entirely |
| [piecewise](piecewise.md) | 3850.0 | 1e-09 | **✔** | linopy 0.9.0, via examples/ports/references/linopy/piecewise.py — agreement, not a published figure |
| [piecewise_conversion](piecewise_conversion.md) | 5990.0 | 1e-06 | · | linopy 0.9.0's own add_piecewise_formulation, via examples/ports/references/linopy/piecewise_conversion.py — agreement, not a published figure. There the arity is an argument list built per converter; here it is a constraint over flow. The looser rtol is this model and not the lane: its weights are continuous inside a segment, so a solver's feasibility tolerance on them reaches the objective — gurobi leaves 1e-7 of weight on a neighbouring breakpoint and lands 8e-9 away |
| [piecewise_lp](piecewise_lp.md) | 3850.0 | 1e-09 | **✔** | the same instance as piecewise, whose optimum and duals linopy 0.9.0 fixes via examples/ports/references/linopy/piecewise.py — the two formulations agreeing is the claim |
| [piecewise_ragged](piecewise_ragged.md) | 426.0 | 1e-09 | **✔** | linopy 0.9.0, via examples/ports/references/linopy/piecewise_ragged.py — agreement between two formulations of one convex curve (weights here, segment lines there), not a published figure |
| [pypsa_ac_dc](pypsa_ac_dc.md) | 18441021.477729216 | 1e-09 | **✔** | pypsa 1.2.4 (its own linopy 0.9.0), via examples/ports/references/pypsa/pypsa_ac_dc.py — n.objective + n.objective_constant, the system cost |
| [pypsa_committable_extendable](pypsa_committable_extendable.md) | 21700.0 | 1e-09 | · | pypsa 1.2.4 (its own linopy 0.9.0), via examples/ports/references/pypsa/pypsa_committable_extendable.py |
| [pypsa_cvar](pypsa_cvar.md) | 35410.0 | 1e-09 | **✔** | pypsa 1.2.4 (its own linopy 0.9.0), via examples/ports/references/pypsa/pypsa_cvar.py |
| [pypsa_cyclic_storage](pypsa_cyclic_storage.md) | 17228.77962151063 | 1e-09 | **✔** | pypsa 1.2.4 (its own linopy 0.9.0), via examples/ports/references/pypsa/pypsa_cyclic_storage.py |
| [pypsa_energy_sum](pypsa_energy_sum.md) | 21400.0 | 1e-09 | **✔** | pypsa 1.2.4 (its own linopy 0.9.0), via examples/ports/references/pypsa/pypsa_energy_sum.py |
| [pypsa_fixed](pypsa_fixed.md) | 49900.0 | 1e-09 | **✔** | pypsa 1.2.4 (its own linopy 0.9.0), via examples/ports/references/pypsa/pypsa_fixed.py |
| [pypsa_global_limits](pypsa_global_limits.md) | 127211.66666666666 | 1e-09 | **✔** | pypsa 1.2.4 (its own linopy 0.9.0), via examples/ports/references/pypsa/pypsa_global_limits.py |
| [pypsa_growth_limit](pypsa_growth_limit.md) | 47110.0 | 1e-09 | **✔** | pypsa 1.2.4 (its own linopy 0.9.0), via examples/ports/references/pypsa/pypsa_growth_limit.py |
| [pypsa_kvl](pypsa_kvl.md) | 17000.0 | 1e-09 | **✔** | pypsa 1.2.4 (its own linopy 0.9.0), via examples/ports/references/pypsa/pypsa_kvl.py |
| [pypsa_linearized_uc](pypsa_linearized_uc.md) | 5540.0 | 1e-09 | **✔** | pypsa 1.2.4 (its own linopy 0.9.0), via examples/ports/references/pypsa/pypsa_linearized_uc.py |
| [pypsa_link_delay](pypsa_link_delay.md) | 4311.111111111111 | 1e-09 | **✔** | pypsa 1.2.4 (its own linopy 0.9.0), via examples/ports/references/pypsa/pypsa_link_delay.py |
| [pypsa_losses](pypsa_losses.md) | 24114.237385131008 | 1e-09 | **✔** | pypsa 1.2.4 (its own linopy 0.9.0), via examples/ports/references/pypsa/pypsa_losses.py |
| [pypsa_min_up_down](pypsa_min_up_down.md) | 32750.0 | 1e-09 | · | pypsa 1.2.4 (its own linopy 0.9.0), via examples/ports/references/pypsa/pypsa_min_up_down.py |
| [pypsa_mixed_cycling](pypsa_mixed_cycling.md) | 4800.0 | 1e-09 | **✔** | pypsa 1.2.4 (its own linopy 0.9.0), via examples/ports/references/pypsa/pypsa_mixed_cycling.py |
| [pypsa_modular](pypsa_modular.md) | 56700.0 | 1e-09 | · | pypsa 1.2.4 (its own linopy 0.9.0), via examples/ports/references/pypsa/pypsa_modular.py |
| [pypsa_multi_period](pypsa_multi_period.md) | 85300.0 | 1e-09 | **✔** | pypsa 1.2.4 (its own linopy 0.9.0), via examples/ports/references/pypsa/pypsa_multi_period.py |
| [pypsa_multilink](pypsa_multilink.md) | 1100.0 | 1e-09 | **✔** | pypsa 1.2.4 (its own linopy 0.9.0), via examples/ports/references/pypsa/pypsa_multilink.py |
| [pypsa_ramp](pypsa_ramp.md) | 18200.0 | 1e-09 | **✔** | pypsa 1.2.4 (its own linopy 0.9.0), via examples/ports/references/pypsa/pypsa_ramp.py |
| [pypsa_spill](pypsa_spill.md) | 3200.0 | 1e-09 | **✔** | pypsa 1.2.4 (its own linopy 0.9.0), via examples/ports/references/pypsa/pypsa_spill.py |
| [pypsa_stochastic](pypsa_stochastic.md) | 33940.0 | 1e-09 | **✔** | pypsa 1.2.4 (its own linopy 0.9.0), via examples/ports/references/pypsa/pypsa_stochastic.py |
| [pypsa_storage](pypsa_storage.md) | 15253.178322993519 | 1e-09 | **✔** | pypsa 1.2.4 (its own linopy 0.9.0), via examples/ports/references/pypsa/pypsa_storage.py |
| [pypsa_store](pypsa_store.md) | 7005.5025000000005 | 1e-09 | **✔** | pypsa 1.2.4 (its own linopy 0.9.0), via examples/ports/references/pypsa/pypsa_store.py |
| [pypsa_transport](pypsa_transport.md) | 22000.0 | 1e-09 | **✔** | pypsa 1.2.4 (its own linopy 0.9.0), via examples/ports/references/pypsa/pypsa_transport.py |
| [pypsa_unit_commitment](pypsa_unit_commitment.md) | 24900.0 | 1e-09 | · | pypsa 1.2.4 (its own linopy 0.9.0), via examples/ports/references/pypsa/pypsa_unit_commitment.py |
| [reserves](reserves.md) | 915.0 | 1e-09 | **✔** | linopy 0.9.0, via examples/ports/references/linopy/reserves.py — agreement, not a published figure |
| [stigler_diet](stigler_diet.md) | 0.10866227820675685 | 1e-09 | **✔** | linopy 0.9.0, via examples/ports/references/linopy/stigler_diet.py — dollars per day; x365 = $39.6617/year[^stigler_diet] |
| [storage](storage.md) | 5650.0 | 1e-09 | **✔** | linopy 0.9.0, via examples/ports/references/linopy/storage.py — agreement, not a published figure |
| [telephone_routing](telephone_routing.md) | 380.0 | 1e-09 | · | published by Gueret, Prins, Sevaux & Heipcke, Applications of Optimization with Xpress-MP (Dash Optimization, 2002) SS12.3.3 p. 182 — "380 out of the required 425 calls are routed"; problem and data in SS12.3, pp. 180-182 |
| [transport](transport.md) | 4400.0 | 1e-09 | **✔** | linopy 0.9.0, via examples/ports/references/linopy/transport.py — agreement, not a published figure |
| [transport_dantzig](transport_dantzig.md) | 153.675 | 1e-09 | **✔** | published with GAMS model library #1 (trnsport), after Dantzig, Linear Programming and Extensions (1963) ch. 3.3[^transport_dantzig] |
| [transport_modes](transport_modes.md) | 1715.0 | 1e-09 | · | published by Gueret, Prins, Sevaux & Heipcke, Applications of Optimization with Xpress-MP (Dash Optimization, 2002) SS10.2.3 p. 143 — "The minimum cost is EUR 1,715k"; problem and data in SS10.2, p. 142 |
| [transport_pwl](transport_pwl.md) | 8.786852757777865 | 1e-09 | · | linopy 0.9.0's own add_piecewise_formulation, via examples/ports/references/linopy/transport_pwl.py; the model is GAMS model library trnspwl (Dantzig transport with economies of scale), which publishes the formulation and its discretisation but no optimal objective |
| [tsp_mtz](tsp_mtz.md) | 2085.0 | 1e-09 | · | published by TSPLIB for instance gr17 (Groetschel, 17 cities, EXPLICIT lower-diagonal distance matrix); optimum 2085 as listed in the TSPLIB solutions file |

[^stigler_diet]: Laderman (1947) at the National Bureau of Standards published $39.69/year for this data, the first serious test of the simplex method. This LP's exact optimum is 0.08% under it — his rounding, not a different model — and both select the same five foods: wheat flour, liver, cabbage, spinach, navy beans.

[^transport_dantzig]: examples/ports/references/linopy/transport_dantzig.py — the same LP hand-written in linopy 0.9.0, which reaches 153.675 independently. Secondary: the published figure is what verifies the port.
<!-- references:end -->

**A `duals` tick means the shadow prices are checked too**, against the
reference's own. For the PyPSA models that is `buses_t.marginal_price`, the
nodal price. A dual vector is where two implementations most often disagree
quietly: which side of a constraint the price belongs to, and what sign an
inequality carries. [Dantzig transport](transport_dantzig.md) is checked
because both of its constraints are inequalities pointing opposite ways. A
MILP has no dual solution, and specsolve does not invent one; those are the `·`
rows.

Adding a port is four files and five rules:
[CONTRIBUTING.md](https://github.com/fluxopt/specsolve/blob/main/CONTRIBUTING.md#adding-a-ported-model).

## PyPSA, one feature at a time

**[The PyPSA ladder](pypsa_ladder.md) is the conformance instrument.** It
holds sixteen networks, each carrying what the one below it did not. Each
is solved through PyPSA and through specsolve and compared four ways: the
objective, the constraint and variable names, the size of the model handed to
the solver, and every constraint's dual. A difference needs a recorded reason
or the run is red. The numbers are there.

**The PyPSA pages above are the gallery.** Each shows one idiom, held by a
test to an optimum that did not come from specsolve. A page shows how something
is *said*; the ladder shows that it is said *exactly*. What could not be said
is a row in [the ledger](#ledger--what-a-port-could-not-say).

The first pages grow one network a feature at a time: a transport model, then
[ramp limits](pypsa_ramp.md) on a widened instance, storage, a
[cyclic horizon](pypsa_cyclic_storage.md) that deletes a clause rather than
adding one, [KVL](pypsa_kvl.md) (Kirchhoff's voltage law) with the cycle
basis as a parameter, and [AC-DC](pypsa_ac_dc.md) with two load-bearing
coordinates on one dimension. Each page says what it shows.

## Ledger — what a port could not say

Each row feeds [the roadmap](../about/roadmap.md) with the verdict
[AGENTS.md](https://github.com/fluxopt/specsolve/blob/main/AGENTS.md) asks for:
macro, primitive, formulation, or refused.

| Port | What could not be said | Worked around by | Verdict |
|---|---|---|---|
| PyPSA transport model | a bound of `-rating` — PyPSA's `p_min_pu = -1` | shipping `neg_rating` as data | **primitive**: bounds as expressions, [#31](https://github.com/fluxopt/specsolve/issues/31). A second model asking for it |
| Travelling salesman | subtour cuts **generated lazily** inside branch-and-cut, which is how every serious TSP code works | [MTZ](tsp_mtz.md), O(n²) and static | **refused, and correctly**: a solve loop is an algorithm, not a model |

[Minimum up and down times](pypsa_min_up_down.md) says `min_up_time` as
`sum_back(start_up, along=snapshot, window=min_up_time)`, each generator's own
width read off the column.

Two rows from 33 ports is the current rate.

### Shapes still without a witness

The language can say each of these; no *outside* model in the corpus has yet
been found to need it. Each row is a standing request, not a gap:

| Shape | Where it was looked for |
|---|---|
| one axis grouped several ways, all of them load-bearing | OSeMOSYS UTOPIA declares three maps out of its timeslice, but they feed only storage constraints and the instance builds none |
| a chain whose coarse end carries a constraint | PyPSA's `ac-dc-meshed` has a country per bus and nothing constrains a country; GAMSLIB `alum` composes three such chains and its shipped scenario switches two of them off |
| a group that carries its own constraint and appears nowhere else | GAMSLIB `mexls` is exactly this, and its optimum is published only in a book with no reachable text |
| a partial map whose null membership moves an optimum | `reserves` exercises it, but that model is ours and was built to |
| opposite-sign legs onto one dimension, plus a second hop a constraint reads | an airline fleet-assignment text has it; the data and both optima are not obtainable |

A model that needs one of these is worth more to the corpus than one that
exercises a shape already covered.

**The TSP row is narrower than it reads.** The DFJ
(Dantzig–Fulkerson–Johnson) subtour rows written out in full *are* sayable,
and [the TSP page](tsp_mtz.md) draws the line between that and lazy
generation.
