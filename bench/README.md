# The performance harness

Not shipped in the wheel and not imported by `specsolve`. It exists so that
[docs/about/benchmarks.md](../docs/about/benchmarks.md) has a *provenance* — the last set of
published numbers came from a `scratch/` script that was deleted, and a claim
nobody can re-run is a claim with a shelf life.

**No measurement is taken in CI, but this directory is checked there.**
`pixi run test-bench` runs `test_harness.py` in the default environment on
every PR, and `codspeed.yml` runs `test_ladder.py` under its own instrument.
That gate is new because it was missing: four entry points here were broken by
`src/` refactors inside one week — a renamed parquet column (#1185) took
`bench/floor.py` down, a class that moved (#1245) took both profilers — and
every one of them was already covered by a test that nothing ran.

It is **one pytest suite** (`bench/test_ladder.py`), and every question below is
a selection out of it: `--cases / --sizes / --arms / --sinks`, plus `-k`.

```bash
# every rung docs/about/benchmarks.md publishes, then both writers. The size
# ladder and each sweep go to separate files: a run REPLACES its results file
# rather than adding to it, and the report takes as many files as you give it
pixi run refresh

# or a rung at a time — the same five tasks `refresh` depends on, in order
pixi run ladder        # --sizes xs s m l          -> bench/results/latest.json
pixi run density       # --sizes d100 d50 d25 d08  -> bench/results/density.json
pixi run declarations  # --sizes n002 … n128, n008m n128m -> bench/results/declarations.json
pixi run report        # every results file       -> written into the page
pixi run plot          #                           -> the chart page

# anything narrower than the published ladder: send it somewhere else
pixi run -e bench pytest bench --cases dispatch --sizes m l --benchmark-json=/tmp/two.json
pixi run -e bench pytest bench --sinks highs --benchmark-json=/tmp/highs.json
```

The selections behind those five are in `pyproject.toml`, under
`[tool.pixi.feature.bench.tasks]`, and that is the only place they are written
down: a published number that came from a ladder somebody retyped is a number
whose fingerprint no longer describes it.

The committed `results/latest-<sink>-<case>.json` are the provenance of the
tables `docs/about/benchmarks.md` publishes *today*: Published benchmark run
34818523523, on `8d27e88b`, which finished one case of eight. Every failure
whose log survives is one of two: a `test_window` cell whose verb was still a
closure `isolate=True` could not ship to its child (fixed since, #1617), or
`storage`/`gurobi` killed by the watchdog with `gurobipy-matrix` at `w1000`
reaching 26 GB. What it left committed is `highs` for `dispatch` and `fleet`,
and `gurobi` for `dispatch`, `fleet` and `transport`; the page has no table for
the rest until a run completes. `results.records` still reads the pre-pytest
`.jsonl` shape, though no such file is committed any more.

**The published ladder measures what the page publishes**: `pixi run ladder`
selects `test_emit` and `test_rebuild`. The windows, the read-back and the
sweep are measured where their history is kept — CodSpeed, below — and by
anyone who runs `pytest bench` without the `-k`; spending the box's hours on
cells no table renders would lengthen a run that took 2 h 23 min the last
time (34818523523).

**The readers take the directory, not a list of names.** `bench.report` and
`bench.tidy` default to `bench/results` and read every file in it, `.jsonl`
first. They used to name three files, two of which no run had ever written, so
`pixi run report` failed on a clean checkout — and no written-out list can be
right both before a refresh and after one.

A bare `pytest bench` is **not** the committed ladder: `--sizes` defaults to
`xs s m`, so it stops below the rung every interesting claim lives at.
Narrowing the run and then committing the file leaves the published tables with
no provenance, and nothing about the file looks wrong afterwards.

**`bench.plot` rewrites one line of `docs/about/benchmarks-scaling.html`** — the
`const DATA = {...};` literal — and nothing else. The page is a tracked source
file, so its markup and prose are reviewed in the diff like any other code and
only the measurements inside it are mechanical.

**A short run pointed at the committed results is refused**, not merely
discouraged — `refuse_to_overwrite_the_provenance` in `conftest.py` compares the
rungs asked for against the ones `pixi run ladder` defines and stops the session
before anything is measured. Narrower sinks or libraries are fine, and the
scheduled run uses both; leaving out *rungs* is what makes a run a smoke test.

Worth knowing while poking at the task: **`pixi run ladder --help` does not
print help.** Pixi forwards unknown arguments to the task, so that starts a
three-hour measurement. `pixi task list -e bench` is the safe way to look.

**Point `--benchmark-json` somewhere else for every run that is not the full
ladder.** Aim it at the committed `results/latest.json` and the run *replaces*
it, so a one-rung smoke test overwrites the provenance of every published table
with four measurements — silently, and in a file whose diff nobody reads
closely. `git checkout` gets it back; noticing is the hard part.

## Taking the published run on a machine of your own

A hosted runner cannot take this ladder. `transport-w100-linopy-highs` peaks at
**14.26 GB** and `ubuntu-24.04` has 16, so the VM is reclaimed part-way and the
job reports `exit 143` with nothing about memory in it (#1399).

**32 GB is not enough either**, and two dead runs paid to find that out. A
measurement holds the model twice — the timed rounds in the pytest process, and
`benchmem(isolate=True)` again in a child, with glibc returning neither to the
OS in between — so that 14.26 GB cell wants about 28 GB of machine on top of
whatever the finished cases are still holding. The published numbers need
**64 GB and a dedicated CPU rather than a burst one**, with nothing else on the
box. `bench/memory-watchdog.sh` is the backstop for a cell that outgrows even
that: it kills the case, where the harness' own budget can only decline the
*next* rung and so cannot see the one it is inside.

**A platform change re-baselines the page.** The committed results were taken on
the dedicated Linux `x86_64` box, eight cores of an AMD EPYC, so a run anywhere
else does not continue that series — every absolute wall time and peak moves.
The cross-library *ratios* survive, because they compare arms measured against
each other on one machine, which is the page's actual claim. What it costs is
that the whole ladder has to be re-taken in one run rather than a rung at a
time, or the page mixes two machines.

The workflow reads the repository variable `BENCH_RUNNER` for its label, so
pointing the run at a box is a settings change rather than a commit. Unset, it
falls back to a hosted runner, which is a fallback and not a place to take
numbers.

**Set the box up once, snapshot it, and let the workflow do the rest.** After
that a run is one dispatch: `provision` creates the server from the snapshot,
`benchmark` lands on the runner that snapshot already carries, and `teardown`
deletes it — `if: always()`, on a hosted runner, because deleting is the only
thing that stops the bill and a teardown running on the machine it is deleting
dies half way. `reap-benchmark-box.yml` sweeps daily for a box that outlived
any believable run.

The runner is installed as a *service* so a restored snapshot brings it up at
boot with nothing to log into, and it is not `--ephemeral`, which would
unregister it after its first job.

```bash
# on the box, as a non-root user
mkdir actions-runner && cd actions-runner
curl -o r.tar.gz -L https://github.com/actions/runner/releases/download/v2.330.0/actions-runner-linux-x64-2.330.0.tar.gz
tar xzf r.tar.gz

# TOKEN from Settings -> Actions -> Runners -> New self-hosted runner
./config.sh --url https://github.com/fluxopt/specsolve --token TOKEN \
    --labels bench-box --name bench-box --unattended
sudo ./svc.sh install runner && sudo ./svc.sh start   # comes up with the box
```

**Turn the distribution's housekeeping off before anything else.** A stock
Ubuntu image runs `unattended-upgrades` on timers keyed to boot, and an upgrade
that trips `needrestart` bounces services — the runner among them. A job whose
runner is restarted under it is reported as `The operation was canceled`, with
the log streaming until the moment it stops and the runner healthy again
afterwards, which is indistinguishable from a cancellation somebody made. Two
runs died that way with 29 GB of memory free, 23 and 54 minutes after their box
booted.

```bash
sudo systemctl disable --now unattended-upgrades apt-daily.timer apt-daily-upgrade.timer
sudo systemctl mask unattended-upgrades
```

Then warm what a run would otherwise download, power off, snapshot, and delete
the server. What is worth warming is `~/.cache/rattler`, the gigabytes of
packages; the pixi binary must not survive, because `setup-pixi` downloads to
that same path each run and refuses to overwrite it — the workflow deletes it
first, and so does the last line here:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
git clone --depth 1 https://github.com/fluxopt/specsolve.git /tmp/warm
cd /tmp/warm && ~/.pixi/bin/pixi install -e bench && rm -rf /tmp/warm
rm -f ~/.pixi/bin/pixi
sudo shutdown -h now
```

Four settings make the workflow find it, all under Settings -> Secrets and
variables -> Actions:

| | |
|---|---|
| `BENCH_RUNNER` (variable) | the runner label and the server name, e.g. `bench-box` |
| `BENCH_SNAPSHOT` (variable) | the snapshot id, from `hcloud image list --type snapshot` |
| `BENCH_SERVER_TYPE` (variable, optional) | defaults to `ccx33`; not every type is offered in every location |
| `BENCH_LOCATION` (variable, optional) | defaults to `nbg1`; use the one the snapshot was built in |
| `BENCH_SSH_KEY` (variable, optional) | a key name from Hetzner's Security tab; without one the box's root password is set and expired, so logging in to debug is a password change first |
| `HCLOUD_TOKEN` (secret) | a Hetzner API token, Read & Write, scoped to that project |

Unset `BENCH_RUNNER` and the whole thing falls back to a hosted runner and
skips the provisioning, which is what a fork sees.

**Test the plumbing before paying for a ladder.** Dispatch with `mode: smoke`
and the run takes one rung, one sink, two arms — a couple of minutes of
measuring instead of hours — while exercising everything around it: the box
comes up, the runner claims the job, the environment solves, `report` and
`plot` render, the artifact lands and the teardown deletes the server. If the
label in `BENCH_RUNNER` does not match the runner's, this is where you find
out, for the price of a few minutes rather than a full run.

**On the bill.** Hetzner charges for what *exists*, not for what runs — a
powered-off server bills exactly as a running one does, so `teardown` deletes
rather than stops. What should be left between runs is the snapshot and nothing
else: `hcloud server list` and `hcloud primary-ip list` both empty. Do not
enable Backups on the box; it is deleted after every run.

Then set `BENCH_RUNNER` to `bench-box` once, under Settings -> Secrets and
variables -> Actions -> Variables.

**The repository is public, so this is the part to get right.** A self-hosted
runner reachable from a `pull_request` runs a contributor's code on your
machine — GitHub says not to do it, and the published benchmark is
`workflow_dispatch` only so that it cannot happen here.
`tests/test_architecture.py::test_no_fork_can_reach_a_runner_we_own` fails the
moment a fork-reachable trigger is added to a workflow that names an owned
runner. On top of that, keep *Fork pull request workflows* set to require
approval for all outside collaborators, and prefer a throwaway cloud box to a
machine that holds anything.

Nothing else should run on it while the ladder does: the harness refuses to
start on a machine already under load, and a shared box makes the numbers
wrong in a way that still looks fine.

## Where a run's numbers come back

Nothing is committed by the run. The box is deleted seconds after it finishes,
so the artifacts are all that survive it — and there is **one per case**, not
one at the end:

```
results-<sink>-<case>-<run id>/     uploaded as each case finishes
  bench/results/…                   every case measured so far
published-benchmark-<run id>/       only if the run reaches the end
  bench/results/…
  docs/about/benchmarks.md           tables already rewritten
  docs/about/benchmarks-scaling.html chart data already rewritten
```

**The per-case artifacts exist because a dying runner skips every step it has
left**, `if: always()` included — so a single upload at the end hands back
nothing at all when the box goes, however many cases were measured first. Each
carries the whole results directory rather than its own case's file, so the
newest one is the complete set and a reader needs only that.

One file per sink, because the job measures each in turn; `bench.report` and
`bench.plot` read the *directory*, so the pair needs no merging.

```bash
gh run download <run id> -R fluxopt/specsolve -D ./bench-out
```

Then commit it as a change somebody reviews. A scheduled job that pushes to a
docs page is a number nobody read.

**The first publish from a runner replaces `results/latest.json`** with the two
per-sink files. The readers do not care, but it is a rename in the diff rather
than an edit, and `test_the_report_renders_from_the_committed_results` renders
whatever is committed — so it is one deliberate PR, not a detail to meet in
review.

**A run on a runner is not a rung-at-a-time top-up.** The platform differs from
the machine the committed tables were taken on, so absolute wall times and peaks
move together; what carries across machines is the ratio between arms measured
against each other. Publish a whole ladder or none of it.

## What it measures

**Peak RSS and wall time**, per phase, for one model into five destinations:

| | `lp` / `mps` | `highs` | `gurobi` | `xpress` |
|---|---|---|---|---|
| `specsolve` | `sps.build(...)` then `model.write(...)` | `sps.build(...)` then `Highs(handoff)` | `sps.build(...)` then `Gurobi(handoff)` | `sps.build(...)` then `Xpress(handoff)` |
| `linopy` | `Model.to_file(...)`, `io_api='lp-polars'` for LP | `Model.to_highspy(set_names=False)` | `Model.to_gurobipy(set_names=False)` | — |
| `pyomo` | `ConcreteModel.write(...)` | appsi `Highs().set_instance(...)` | appsi `Gurobi().set_instance(...)` | — |
| `gurobipy-loop` | — | — | `addVar` per entity, `addConstrs(quicksum(...))`, then `update()` | — |
| `gurobipy-matrix` | — | — | `addMVar` + `addMConstr` over a scipy CSR, then `update()` | — |
| `highspy-matrix` | — | one `addCols` + one `addRows` over the same CSR | — | — |

**The two matrix arms build one matrix, not two.** Each case writes it once in
`bench/models/<case>/matrix.py`, which hands back an `Lp` — bounds, objective,
CSR, a sense per row, right-hand side — and the arm pushes that into its own
solver's bulk API. A second copy per arm is the copy that drifts, and it would
take a published floor down with it.

`gurobi` is opt-in (`--sinks gurobi`): it needs the `[gurobi]` extra, where the
other two need nothing a contributor does not already have. So are `mps`, the
other file format every arm with a writer reaches, and `xpress`, which only
specsolve reaches — its wheel's Community licence refuses a solve over 5,000 rows
plus columns but not a load, so the hand-off measures at every rung. A cell
whose solver package is not installed is skipped with the package's name. It is also the only
sink two of the three arms can reach, and a cell an arm cannot reach is skipped
with the reason rather than left to look like a measurement that failed.

**One framework, two dialects, on purpose.** `gurobipy-loop` and
`gurobipy-matrix` are the same library written the two ways people write it.
Publishing both is the answer to *"you wrote their arm badly"*: the gap between
them is how much of any result is the library and how much is the style, and
that is a question about our arm too — `gurobipy-matrix` reaches the same
`addMVar`/`addMConstr` seam our own `Gurobi` constructor does, so what separates it
from `specsolve` is only where the matrix came from.

**`pyomo` is here because leaving it out would look chosen.** Slow is the
expected answer and not the point: it is the baseline most readers already
have, and appsi's `set_instance` reaches the same seam our sinks do — the
solver's own model, populated, with nothing solved. Nothing of pyomo's is
switched off that pyomo does not switch off itself; `symbolic_solver_labels`
is already false by default, which is the cheap side of the same choice
`set_names=False` makes on the linopy arm.

**The `linopy` arm is hand-written, and its formulations are the gallery's.**
`bench/models/<case>/linopy.py` is `examples/ports/references/linopy/<case>.py`
against the ladder's parquet — scripts #681 reviewed for idiom and the docs
execute, which is what keeps this arm from being a strawman somebody wrote in
an afternoon. The linopy oracle in `tests/linopy_lane` is not this arm and is
not measured.

**A hand-written arm is a model somebody typed twice**, and nothing structural
stops it being a *different* model that benchmarks beautifully. The linopy arm
never had that risk — it read the same YAML. So each dialect's smallest rung is
solved against `specsolve`'s and the objectives compared, in
`test_the_hand_written_arm...` under `bench/test_harness.py`, which CI runs on
every pull request.

**The solver sinks stop at the handoff — `run()` / `optimize()` is never
called.** That is the whole discipline of it. HiGHS's simplex is the same work
whoever filled the model, so including it would swamp the phase this harness
exists to measure and publish a number about HiGHS under our name.

`highs` is the sink most callers actually reach for, and it is **not the lp sink
minus a file** — HiGHS's own dense model is resident in the process and narrows
every ratio drawn against it. Measuring only the LP path reports the wrong
number for the common case, which is why both run by default.

**What an arm's own defaults cost is the next arm's problem, and it is
load-bearing.** The retired linopy arm passed `set_names=False` because linopy
names every variable and constraint while our sinks name nothing, and naming is
**82% of linopy's HiGHS hand-off** (0.11s against 0.02s at 200k variables) and
35% of its Gurobi one. Any arm added here answers the same question in its own
docstring: which defaults were switched off, and what each one cost.

Not measured, deliberately: solve time (that is the solver, identical either
way, and it would swamp the build), and anything about expressiveness.

**An arm stops climbing a ladder it cannot afford.** A library that builds per
entity costs about what the rung is wide, and the rungs grow tenfold — so one
measurement settles the next one. After every cell the harness projects the rung
above it, and if that projection is over `--budget` (120 s by default) the arm
skips the rest of that ladder with a sentence saying what it measured and what
it projected. *That a library is far slower is the finding*; an hour spent
measuring it again to two decimal places is a machine kept busy for nothing.

The projection is never recorded as a measurement — it is arithmetic on one
rung, not a second rung — and a skipped cell leaves no row in the results file,
because a measurement nobody took is an absent row rather than a null. The
reasons print together under `over budget` when the run ends, which is the one
place a skipped cell can still say something. `--budget 0` measures everything,
which is what to pass when the slow number is the point.

**A number the run cannot stand behind is marked, not dropped.** Every
measurement's distribution — `iqr`, `median`, `rounds` — is carried into the
result file beside the minimum the tables publish, and `bench.report` appends
`~` to any wall cell, and to the ratio beside it, whose IQR exceeds
`SPREAD_BUDGET` of its own median. That is the case `min` cannot survive: not
one wild round, which the minimum ignores by construction, but *every* round
slow, which leaves no clean one to fall back on — #797 is the cell that was
publication-ready at 2.33x wrong. A marked cell is one to re-take on an idle
machine, never one to quote.

**Nine rounds is the floor, and the harness sets it.** pytest-benchmark's own
default is 5, and its calibration gives the fewest rounds to the slowest cells —
exactly where sustained interference is most likely and a clean round hardest
to come by. `--benchmark-min-rounds` still wins where a run wants more.

**Every round starts with a full garbage collection.** A round otherwise
inherits the last one's garbage, and on the gurobi sink that is a million
`Var` objects the collector's generation-1 pass walks on every second round —
0.45 s against 0.67 s in one distribution, and nine published comparisons
decided by which one the median fell on (#1288). What a round measures is one
build.

## Why it is built this way

**One process per measurement.** Peak RSS is a property of a process: a second
arm in the same interpreter inherits the first's high-water mark and its warm
allocator. `@pytest.mark.benchmem(isolate=True)` is what gives each pass its own
interpreter, and it is the same declaration that makes whole-process `rss`
available at all.

**`rss`, not the memray peak, for anything published.** pytest-benchmem records
both. `rss` is the whole-process high-water mark — the number `/usr/bin/time -l`
agrees with — and it is the only one honest across two libraries; the memray
peak is deterministic and attributable to a call stack, which is what makes it
right for comparing specsolve to itself. Both are in every result file, and which
one a table reads is a decision, not an accident. The measured reason is below.

**The harness is pytest, and deliberately nothing more.** Selection, the
ragged parametrization, per-pass isolation, the JSON, the repeats and the
minimum are all things pytest and its plugins already do and have tested. What
is left in this directory is what is specific to specsolve: the cases and the
verbs.

**There is no parity gate while there is one arm.** It solved each case's
smallest rung on every arm and compared the objectives to 1e-9 relative — a
check with one counterparty, which with one arm it has none of. It comes back
with the second arm, where it means something again. Until then the same-model
guards are `test_ladder._record`, which checks every measurement's counts
against the rung it claims, and `bench/floor.py --check`, which solves
`transport` two ways and compares.

## Where the clock starts and stops

The easiest way to publish a wrong number is to time something in one arm that
another never does. The boundaries are therefore explicit, and a new arm is
written against this table:

| | specsolve |
|---|---|
| **before the clock** | `prepare` — splitting parquet paths into parameters vs dimensions (harness bookkeeping: it re-parses the YAML only because the *runner* decides which file is which) |
| `import` | `import specsolve` |
| `build` | `sps.build(...)` — the engine scans the parquet itself |
| `emit` | `model.write(path)` / `Highs(_handoff(model))` |
| `teardown` | `model.close()` — releases the built model |
| **after the clock** | row, column and nonzero counts off the built frames |

Two of those are deliberate calls rather than defaults:

- **Import is excluded from `wall_seconds`** but recorded. It is fixed, paid
  once per process, and a modelling library's import can exceed specsolve's entire
  build at the `xs` rung — including it would make the small end meaningless.
- **Teardown is included, and it is now near-free.** It was there to charge the
  arm holding a scratch database for releasing it. There is no scratch database
  any more — `close()` drops frames this process owns — so the phase is kept as
  a tripwire rather than a cost: if it ever stops reading ~0, something
  acquired a lifetime again.
Every arm starts from the same parquet files and stops at the same seam, so
each pays for its own data ingestion. That is the honest unit. The *phases* are
not comparable one-for-one across arms — a library that defers coefficient
materialisation to its writer spends nothing in `build` that another spends
there. Compare totals, and read the phases as attribution within an arm.

**Peak RSS is the whole cost, because nothing spills to disk.** An engine that
traded RAM for a workdir could show a peak-RSS win while holding a
multi-gigabyte temp file, and the harness once recorded `workdir_bytes` to stop
that. Nothing writes anything but the LP file now, so that field is gone rather
than left reading zero — a column that is always 0 reads as "measured and
fine", which is the same failure in the other direction. Restore it in
`bench/arms/` if a sink ever spills again.

**Failures are results.** A run that dies is written to the JSONL with the
exception line that killed it, and the report renders it as a cell. An OOM is
the single most informative thing this harness can find — and this is where a
cost claim is settled, because cost is not one of the architecture's rules.

**Repeats collapse by minimum.** Noise only ever adds.

**Comparing two versions of the same arm? Alternate them.** Repeats inside one
invocation collapse noise *within* a few seconds; they do nothing about drift
across a session, and this machine has drifted 2x on wall time between the
start of a session and the end of one. Check out A, measure, check out B,
measure, and go back — not A once and B once an hour later. A second arm used
to be the tell — if it moved too, the machine moved, because nothing in
`src/specsolve/relational/` can reach it — and `bench/floor.py` is that tell now. Peak RSS is far steadier than
wall time and is usually the honest half of a before/after claim.

## The cases

Chosen so each stresses a *different* SQL shape (docs/about/architecture.md, "read the
verdict off the SQL"), not to cover the language:

| case | shape | why |
|---|---|---|
| `dispatch` | pointwise bounds + one `sum` per row | raw throughput, and the case a dense eager broadcast is best at — so our worst ratio |
| `nodal` | `(snapshot, node, tech)`, `where: installed > 0` | sparsity as it actually occurs — see below |
| `transport` | three `sum(by=)` joins per row | the mapping-table path, where the linopy lane must materialise a bus x generator product |
| `sector` | dense snapshots x dense carriers x sparse portfolio | mixed density in one model — the shape a sector-coupled model actually has, and where the sparsity claim is visible |
| `storage` | a cyclic `shift` recurrence | the self-join, and the only locality class with no linopy cost analogue: xarray shifts an array, we join a term stream against itself on `snapshot.ord - 1` |
| `commitment` | dispatch gated by a binary `u`, `p <= p_max * u` | the MILP — the only case whose `vtype` stream is not all-continuous, so integrality reaches every sink at scale |

**`nodal` is the case worth explaining.** It is dispatch over nodes and
technologies, and a technology only generates at a node where it is installed:
no offshore wind inland, no hydro without a river. PyPSA spells that by
attaching generators to buses, Calliope by declaring techs at nodes; in YAML it
is a `where` over the capacity table. 50 nodes x 12 technologies is 600
coordinates per snapshot, of which 3 per node — a quarter — exist. That gap is
the comparison: relationally an absent pair is an absent row, eagerly it is a
NaN that still costs eight bytes and a broadcast.

The sparsity is *structural and time-invariant*, which is not incidental —
`installed` carries node and tech but not snapshot. A random Bernoulli mask
would sweep the same densities while misrepresenting the shape, and the shape is
what an engine can exploit.

**Measured, this sweep alone does not show it** — at a 1.2M coordinate product
a dense array over it is ~10 MB and the fixed cost of the process dominates.
`sector` runs the same sparsity at a 12M product and the effect is plain. See
[docs/about/benchmarks.md](../docs/about/benchmarks.md#the-density-sweep-and-a-claim-it-refuses).

`Shape.density` (technologies per node: 12 / 6 / 3 / 1) is swept at one model
size, because sweeping size and density together leaves no way to tell one
effect from the other. Run the full ladder with `--sizes all`.

**The declaration count is the third swept axis.** Every size rung grows
`snapshot` and holds its case's declaration count fixed, so a cost paid *per
declaration* — a labelled frame each, a stack at the end — is sampled at
whatever counts the cases happen to have. The `declarations` case splits a
fixed pool of 512 units per snapshot into 2 / 8 / 32 / 128 variable
declarations (rungs `n002`…`n128`), each with its own capacity constraint and
objective term and one balance over all of them, at one model size for the
density sweep's reason. The balance and the objective bracket their terms in
halves rather than writing a flat chain, because a chain of N terms parses to a
tree N deep and the language admits 100 levels; bracketing makes it log2(N) and
leaves the terms and the rows they cover alone. Its model YAML varies per rung,
so it is generated — `_declarations_spec` in `bench/cases.py` — and cached
beside the rung's data.

**Two of its rungs are paired rather than swept.** `n008m` and `n128m` are
`n008` and `n128` with a `where:` on every declaration, and nothing else
changed: same units, same snapshots, same parquet, one keyword apart. They are
there because *how many declarations carry a mask* is an axis of its own — the
engine compiles a predicate and decides a semi-join per masked declaration,
before it pays anything per row — and no other case reaches past two. The mask
is `dispatch`'s vacuous one (`p_max > 0`, drawn strictly positive), so the pair
builds the identical model and the difference is the price of the `where:`
alone, with no row-count change mixed into it. The density sweep answers the
other half of the question, how much a mask *keeps*, and cannot answer this
one: the mask that isolates the per-declaration cost is the one that keeps
everything.

**The report measures what survived rather than trusting the declaration.**
`dispatch` declares `where: p_max > 0` against a p_max that is always positive,
so its mask removes nothing and the engine pays for it anyway; the `live` column
says `100%` and makes that visible instead of leaving it as a trap. Keeping that
vacuous mask is itself a measurement, which is why `nodal` is a separate case
rather than a fix to `dispatch`'s data.

Data is generated deterministically (a blake2b digest of the shape seeds the
RNG — `hash()` is salted per process and would give the two arms different
numbers), cached under `bench/.cache/`, and feasible by construction.

**`commitment` is a MILP, and the gate still costs one cheap solve.** The gate
solves only the *smallest* rung of a case, once per arm, and the measured pass
never solves at all — so the `l`/`xl` rungs of a MILP ladder cost the gate
nothing. What the case has to guarantee is that its bottom rung solves to
proven optimality: `GATE_RTOL` is 1e-9 and HiGHS's default `mip_rel_gap` is
1e-4, so a rung where branch and bound stops at a gap could hand the two arms
different incumbents. The bottom rung is therefore deliberately tiny, with
every cost a distinct float — there is no MIP-aware tolerance, and that is a
decision rather than an omission.

## The speed-of-light floor

**This is the ladder's only denominator now.** A wall time on its own says
nothing about how much headroom is left in it. `bench/floor.py` is what a
number is read against: it hand-writes **one** model — `transport` — from the
case's cached parquet straight into numpy arrays and a CSR matrix, no specsolve
and no expression engine anywhere in the path, and ends at the same seam as
the `highs` sink: a populated `highspy.Highs` with `run()` never called. What
it costs is the irreducible price of emitting the coefficients, and with it
the sentence becomes *"we are at Nx the floor"* — a claim about engineering
rather than a ranking.

```bash
pixi run -e bench python -m bench.floor l            # phase minima + peak RSS
pixi run -e bench python -m bench.floor xs --check   # one solve each way, objectives compared
```

It is **not a fourth arm**: it hardcodes one model, so it has no place in the
`case x size x sink x arm` product, and its numbers are quoted beside the
ladder's rather than inside it. `--check` solves the smallest rung through the
floor and through specsolve and compares objectives at the gate's tolerance;
`bench/test_harness.py` pins the cheaper fingerprint — the floor's column, row
and nonzero counts against specsolve's — on every bare `pytest bench`.

## The second window

`test_window` prices what `model.update(x).solve()` costs before the solver
runs — the rebuild, the digest a loaded solver is checked against, and then
either a push of bounds, costs and right-hand sides or a load from scratch.
Which of those two an update takes is decided by the data, so the test is
parametrized by what the window moves:

| change | the window | path |
|---|---|---|
| `values` | re-attaches the same sources | push |
| `shape` | attaches the rung one snapshot shorter (`cases.shortened`) | load |
| `one` | updates the first declared parameter alone, as `update` is usually called | push |
| `cold` | re-attaches the same sources under `keep='nothing'` | load |

Values are not varied, because a push sends whole vectors whatever they hold.
`one` costs a whole rebuild all the same — `update` re-reads every source — so
it is the baseline an incremental build would have to beat. `keep='progress'`
differs from the default only in what the solver remembers, which moves the
solve and not anything timed here; `bench/warm_payoff.py` is where a carried
basis is weighed.

The clock stops at `Engine._hand_off`, the half of `Engine.solve` before the run, so
nothing about the solver's own work lands in the wall time or the peak. The
arm reports whether its window reloaded and the test holds that to the change,
so a rung cannot quietly measure the other path; `test_harness.py` checks the
same on every case's smallest rung on every pull request. The phases are
`window`, `window-reshaped`, `window-one` and `window-cold`, and only
`bench.tidy` renders them.

`test_sweep` is `solve_over` across four hand-built slices of one rung, folded
in order on one model: the first slice loads and the rest push, which the test
asserts. A sweep cannot be timed short of its solves, so the solver runs on
every slice, and the sweep's own per-slice clocks are summed into the result's
phases — `solve_seconds` is what the wall time owes the solver, and the rest is
the sweep's. Its phase is `sweep`. It is taken at `xs` and `s` only
(`SWEPT_SIZES`), where the solver is not yet the whole cost, and never under
CodSpeed (`NOT_UNDER_CODSPEED` in `conftest.py`): its memory instrument tracks
the solver's own allocations, and `commitment` swept at `s` ran that job's 20
minutes out on its own. So a sweep keeps no history: it is measured where
someone runs `pytest bench -k test_sweep`.

## Reading an answer back

`test_read` prices what a solve costs after the solver returns: the answer laid
out against the build as a `Result`, then every value read back — `frames`
collects each variable's primal and each constraint's dual and activity,
`parquet` is `Result.save`. The answer is built before the clock rather than
solved for (`read_setup`): every column and row zero, reported optimal, and no
duals for a model with an integer variable, which is what a real solve leaves.
The cost is set by how many values are laid out, not what they are, so no
solver runs and every rung is affordable. It is sink-free and specsolve-only —
the other libraries read their answer back inside their solve — and its phases
are `read-frames` and `read-parquet`, rendered by `bench.tidy` alone.

## The warm-start payoff

*Does carrying a basis across a genuine rebuild pay?* is the question #382 has
to answer before the engine work is worth writing, and until this module there
was nothing in the tree to answer it on: `examples/benders/run.py` is the only
driver that rebuilds a model every iteration, and its master is 3 columns and
25 rows, where a cold solve costs one simplex iteration.

`bench/warm_payoff.py` is that missing case — a capacity-expansion Benders
whose master is sized from data (`bench/expansion/*.yaml`), with the master
solved three ways at every rebuild: cold, from the previous iteration's basis
spliced per declaration, and from that basis merely truncated to the new
height. The subproblem is a real dispatch LP and is deliberately *not* what is
measured: `cap_hat * avail` reaches the rows as a right-hand side, so a new
capacity pushes values onto the loaded solver and never rebuilds.

```bash
pixi run -e bench python -m bench.warm_payoff s m l --steps 400
pixi run -e bench python -m bench.warm_payoff m --wall   # only on an idle box
```

**Simplex iterations are the measurement.** They are deterministic, so this
ladder needs no idle machine, and they are the quantity a basis actually moves.
`--wall` prints seconds and the load averages beside them, and carries none of
the argument.

It is **not an arm** — like `floor.py` it hardcodes one model, prints its own
table and never touches the ladder's results files — and it is **not a
feature**: no `src/` code carries a basis across a rebuild, and the splice
lives here so that the evidence could be taken before the engine work was
written. Its models sit under `bench/expansion/` rather than `bench/models/`,
which `tests/test_bench_models.py` reserves for files backing a ladder case.

The splice exists because **rows do not append**: a master with two cut
families numbers rows per declaration, so a row gained by `optimality_cut`
shifts every row of `feasibility_cut`. A wrong carry cannot produce a wrong
answer — a basis moves the route, not the optimum — so the third arm is how the
splice is shown to be worth its complication at all.

## The other question: regressions

*Did this change make it worse?* is a different question from *how do we
compare to another library*, and it wants a different metric — but it does not
want a different harness. It is the same suite, run twice:

```bash
pixi run -e bench pytest bench --sizes s m --benchmark-memory --benchmark-autosave   # base
pixi run -e bench pytest bench --sizes s m --benchmark-memory --benchmark-autosave   # head
pixi run -e bench benchmem compare .benchmarks/*/0001_*.json .benchmarks/*/0002_*.json \
    --columns peak --fail-on peak:20%
```

That is what `.github/workflows/bench.yml` runs on a pull request labelled
`trigger:bench` — the base's `src/` and then the head's, under one harness —
and the memray peak at 20% is what it gates on. Wall time and `rss` are
reported beside it and never gated.

**Why the metric changes with the question.** Measured on `dispatch/m`:

| arm | `ru_maxrss` | memray peak |
|---|---|---|
| specsolve | 309 MB | 211 MB |
| the retired linopy arm | 604 MB | **2967 MB** |

memray counts polars' reserved arenas as allocated and does not count the
interpreter or mapped libraries at all, so the bias points in *opposite*
directions in the two lanes: the peak ratio is 0.51x by RSS and 0.07x by memray.
A published cross-library claim built on that would be false the moment a reader
ran `/usr/bin/time`. Within one lane the same bias sits on both sides of a diff
and cancels, leaving a metric that is deterministic and attributable to a call
stack — which RSS, sensitive to machine load, is not.

So: `rss` for the comparison we publish, the memray peak for the regressions we
chase. Both come out of the same run — the choice is which column a table reads,
and `--benchmark-memory-compare-fail` is what turns the second into a gate.

## The same suite, a third instrument: CodSpeed

`bench/` is a plain `pytest-benchmark` suite, so the fixture its tests ask for
is whichever plugin is loaded. That is not a detail — it is why there is no
second set of benchmarks in this repository:

```bash
pixi run -e bench pytest bench --benchmark-memory   # memray peak + rss + timing
pixi run -e codspeed pytest bench --codspeed           # what CI measures
```

`--benchmark-memory` patches the stock fixture and reads the `benchmem` marker;
`--codspeed` replaces the fixture outright and the marker goes inert. Same
tests, same workloads, same rungs — a different instrument. The workloads
cannot drift between them, because there is one of them.

[CodSpeed](https://codspeed.io) (`.github/workflows/codspeed.yml`) is where
the history is kept: it stores a number per benchmark for every commit on
`main`, which is what a regression found months later is bisected against.
What it adds over `bench.yml` is not the metric but **the baseline** —
`bench.yml` can only compare against a base it checks out and measures itself,
which costs two passes and is why it waits for a label. Two jobs:

| job | instrument | runner | rungs | runs on |
|---|---|---|---|---|
| Memory (heap) | `memory` | free, one polars thread | `s` | every pull request, every push to `main` |
| Wall time | `walltime` | CodSpeed's metered bare metal (`codspeed-macro`) | `m` `l` | every push to `main`, and a pull request labelled `trigger:bench` |

`walltime` runs only on the metered runner because a shared runner's clock says
nothing, and `simulation` — CodSpeed's default — runs the workload under an
emulator, which suits neither multi-threaded native code nor these rungs. The
`l` rung is there because #520 measured the instrument's overhead at -5..-10%
at `m` and ±4% at `l`. Every benchmark in `test_ladder.py` runs under both
jobs except the two `NOT_UNDER_CODSPEED` in `conftest.py` names: `test_rebuild`,
which pedantic rounds make meaningless there, and `test_sweep`, which solves.

**It gates nothing.** Both jobs are `continue-on-error` and no ruleset names
them; `bench.yml` remains the check that fails a pull request.

## Two ladders

Every case grows `snapshot` and holds its entity counts fixed. `transport` and
`storage` also carry a **width ladder** — entity counts x N with the snapshots
frozen — because one axis is not scaling and the omission was not neutral:
`transport`'s bus x generator incidence is 20 x 100 at *every* rung of its size
ladder, so the join the case exists to expose never grew.

The multipliers are chosen so each width rung matches a size rung variable for
variable: `w1` is `xs`, `w10` is `s`, `w100` is `m`, `w1000` is `l`. Same
model, same size, different shape — which is what makes the two tables readable
against each other, and what `test_a_width_rung_matches_its_size_twin...` holds
them to. They render as their own table: sorting `w10` and `s` into one column
would read as a single curve that is really two shapes.

## Adding a case

Add `bench/models/<case>/spec.yaml`, and a data generator and a ladder to
`CASES` in `bench/cases.py`. Nothing else: the parametrization reads `CASES`,
and the report is case-agnostic.

## Reproducing a published number

`bench/reproduce.py` is a PEP 723 script and `bench/reproduce.py.lock` beside it
freezes every version it runs on, git commits included:

```bash
uv run --locked bench/reproduce.py
```

**Why it exists.** `pixi.lock` is not committed, and two of the libraries here
install from git — specsolve itself, and linopy from `master`, a branch that moves.
Before the lock, "the versions that produced this number" existed only inside a
results file, after the fact, in a form nobody could install. `--locked` refuses
to run if the resolution has drifted, `test_the_lock_pins_every_library...`
refuses a merge where an arm was added and the lock forgot it, and
`test_the_lock_installs_what_the_published_numbers_were_taken_on` refuses one
that installs a version no published file was measured on.

It **drives** the harness rather than repeating it: the models and rungs live
here, and a standalone script that rebuilt them would be a second definition of
every model, free to disagree with the one being measured. Re-lock with
`uv lock --script bench/reproduce.py` whenever an arm or a pin changes, and
whenever a run is published: what it pins is the commit that produced the
numbers, which is not in general the release nearest to it.

## Adding an arm

Two files, because an arm is two different kinds of knowledge:

- **`bench/arms/<arm>.py`** — everything true of the library whatever the model
  is: what its `prepare` needs before the clock, how it builds and emits, which
  `SINKS` it can reach, how a solution comes back, and which of its defaults are
  switched off with what each one costs. Register it in `ARMS`.
- **`bench/models/<case>/<arm>.py`** — the model itself, one per case, listed in
  that case's `FORMULATIONS`. Modelling only: no timing, no sink, no counts, no
  specsolve import. That is what lets the dialects of one case be read side by
  side, which is how a reader judges whether the comparison is fair.

A case an arm cannot express ships no module and prints its reason. Write the
model the way that library's own community writes it, and expect the objective
check above to be the first thing that fails.

A case whose YAML has to vary per rung sets `generate_spec` instead of
`spec` — `declarations` is the template — and `Case.spec_path(shape)` hands
every consumer whichever of the two the case has.

## The map

| file | |
|---|---|
| `cases.py` | the models, the data generators, the ladders |
| `models/<case>/` | one directory per case: `spec.yaml`, and the same model in each hand-written dialect |
| `arms/` | one module per arm — `prepare` before the clock, then build-and-emit, build-only, objective. Picklable, and the library imported inside the verb |
| `conftest.py` | selection flags, the ragged parametrization, the data fixture, the machine interlock |
| `test_ladder.py` | the benchmarks: build-and-emit, the second window up to its solve, reading an answer back, and rebuild-in-one-process |
| `results.py` | pytest-benchmark JSON -> the flat records the report and the plot read |
| `tidy.py` | the same records as one long CSV — a row per number, dims in columns, no nulls. What a plot nobody planned for is built from |
| `floor.py` | the speed-of-light floor — `transport` hand-written into a populated `Highs`, no engine involved |
| `warm_payoff.py` / `expansion/` | does a basis carried across a rebuild pay? A scaled Benders, its master solved cold and warm at every rebuild |
| `report.py` / `plot.py` | the published tables, and the chart page's data literal |
| `profile_build.py` | which *query* inside one build spends the time — a profiler, not a benchmark. Wraps every collect, so read its shares and not its seconds |
| `profile_phases.py` | which *phase*, in seconds comparable to a real run. Hoists the parse, the lowering and the parquet read out of the loop and reuses one attachment, which takes the spread from 12-55% down to a few percent — the difference between a 10% change being visible and not |
