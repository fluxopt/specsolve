# Examples

The published optima are models somebody else solved first, solved again
with specsolve to the same optimum. The other models are this repository's
own, and the test suite solves each one. Each page shows the model and the
same model as math.

## Every model

<!-- catalogue:begin -->
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

### More models

| | |
|---|---|
| [dispatch](dispatch.md) | Least-cost generation against a load profile — the smallest model that is still a model. |
| [storage](storage.md) | Dispatch plus a battery that carries its charge from one snapshot to the next, closed into a cycle over the horizon. |
| [transport](transport.md) | A network: generators sit on buses, lines connect buses, and power balances at every bus. |
| [piecewise](piecewise.md) | Dispatch where each generator's cost follows its own convex curve. |
| [piecewise segment lines](piecewise_lp.md) | The [piecewise](piecewise.md) model, with each cost curve stated as the lines its segments lie on. |
| [piecewise curves of differing length](piecewise_ragged.md) | Dispatch where each generator's cost curve has as many breakpoints as its own data. |
| [piecewise conversion](piecewise_conversion.md) | Heat and power from converters whose flows share one curve each, where how many flows a converter has is data. |
| [special-ordered sets](sos.md) | The [piecewise](piecewise.md) model, with each cost curve handed to the solver as a special-ordered set. |
| [monthly budget](monthly_budget.md) | A cap on what each technology may generate in each calendar month, with the months read off the timestamps. |
| [multi-period](multi_period.md) | Capacity decided once per investment period, binding at every snapshot inside it — and the periods need not be the same size. |
| [seasons](seasons.md) | A store that cycles inside each season rather than across the horizon, with seasons of different lengths. |
| [reserves](reserves.md) | Energy and reserve co-optimization on a two-bus grid: offers are (generator, market, tranche) triples, reserve zones overlap, and one line dangles. |

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
<!-- catalogue:end -->
