# Contributing

Procedure lives here. **Why** the project is shaped the way it is lives in
[docs/about/architecture.md](docs/about/architecture.md), and that split is deliberate: this file
should be readable in one sitting and go stale only when a command changes.

## Setup

```bash
pixi install                   # the default environment: tools + the linopy oracle
pixi run pre-commit install    # once per clone
```

That is the whole of it — pixi brings its own interpreter, so there is no
Python to install first and no virtualenv to activate. [Install
pixi](https://pixi.sh/latest/) if you have not; `pixi run <task>` installs or
updates whatever environment the task names before running it, so the two lines
above are a convenience rather than a step you can forget.

`pixi task list` prints every task with what it does. The ones you want most:
`test`, `lint`, `format`, `typecheck`, `check` (all four), `docs`.

The default environment carries linopy, from the `dev` group, because the
differential test suite compares every build against the linopy oracle in
`tests/linopy_lane`. It carries `[gurobi]` and `[xpress]` too, because a solver
sink is checked against another solver. Both of
those wheels carry a size-limited licence of their own — gurobipy's needs
nothing, and xpress's Community licence is active on import — so those tests run
on a plain checkout with no licence of your own. They skip where the package is
absent. The engine itself never imports linopy, xarray, pandas, gurobipy or
xpress — see *the floors gate* below.

## The loop

```bash
pixi run test                      # the suite across cores, ~20 s
pixi run lint --fix && pixi run format
pixi run typecheck
pixi run check                     # all four, which is what `ci` requires
```

Narrower runs while working:

```bash
pixi run pytest tests/test_absence.py -q
pixi run pytest -k piecewise -q
pixi run pytest --lf  # last failures only
```

### Which file a test goes in

**Named for the construct, not for the module it happens to exercise.**
`test_shift.py`, `test_absence.py`, `test_grouped_sum.py`, `test_piecewise.py`:
the same word the [language reference](https://mathspec.readthedocs.io/en/latest/reference/language/) uses,
so "where is `shift` specified" and "where is `shift` tested" have one answer.
It is also the axis that survives the two lanes — a behaviour spans both, while
a module is on one side of the fence or the other.

A module-shaped file is the exception and earns it by being about the module as
such: `test_compiler.py`, `test_schema.py`, and `test_relational.py`, which is
matrix assembly and the solver hand-off rather than a language construct.

The failure mode is a file that grows into everything that had no other home.
`test_relational.py` reached 2,338 lines that way, larger than any module in
`src/`. If a test does not obviously belong to the file you are about to add it
to, that is the signal to name its construct.

## What each CI gate means

`main` requires two checks: **`ci`** and **`Conventional commit subject`**.
Everything below is the first one, in the order it runs.

| gate | what a failure means |
|---|---|
| `pixi run lint` | a lint rule fired. `--fix` handles most; if the finding is wrong, silence the one line with a `# noqa: RULE` and say why. |
| `pixi run format-check` | formatting drifted. Run `pixi run format`. |
| `pixi run typecheck` | a type is wrong. **Fix the type, don't widen it** — if the finding is genuinely wrong, `# pyrefly: ignore[rule-name]` on the one line with a reason, never the rule off globally. |
| `pixi run test` | the suite. Includes the differential lanes and the ported models. |
| `pixi run docs-build` | the site. A dead cross-link or an anchor that no longer resolves — see *the docs* below. |
| `pixi run docs-test` | a page's math reached the reader as literal backticks. Needs the docs environment, which is why it is not in `test`. |
| `pixi run test-floors` | the engine reached for something it does not declare. |

Every one of them runs the same way locally as it does in CI, because the
command is the task and the task is defined once, in `pyproject.toml`.

**The floors gate is the one worth understanding.** It is an environment of its
own: every runtime dependency pinned to the exact version
`[project.dependencies]` names as its lower bound, the project installed from a
built wheel, and no dev group, no extras and no linopy anywhere. It proves two
things at once: that the relational lane builds, solves and reads results back
with no pandas, pyarrow, linopy or xarray; and that the declared lower bounds
are real rather than decorative. Tests that need a second lane route through
`tests/oracle.py`, which skips them when it is not installed — a bare
`import pandas` in a test file breaks this gate, and only this gate.

Because it is an environment rather than a re-install, running it costs your
working environment nothing: `pixi run test-floors` builds `.pixi/envs/floors`
beside the default one and leaves it there.

Raise a floor when the code relies on that version's behaviour. Do not raise
one to chase a newer interpreter.

## The docs

`docs/` is both the site and what you read on GitHub. Write for the repo —
relative links, no site-only syntax — and the build handles the difference.

```bash
pixi run docs        # http://127.0.0.1:8000, live-reloading
pixi run docs-build  # what CI runs, and what Read the Docs runs
pixi run docs-test   # the math the build cannot check
```

The site is built by [Zensical](https://zensical.org/) from `mkdocs.yml`. The
two tutorial pages hold their code in `python exec="true"` blocks, which
markdown-exec runs during the build: a block that raises fails it, and what you
read under a block is what it printed on that commit.

**What a page is for decides where it goes, in the nav and in the tree.** A
tutorial (`docs/`), a how-to guide (`docs/howto/`), reference
(`docs/reference/`, and the model pages in `docs/examples/`) or explanation
(`docs/about/`) — the four kinds of [Diátaxis](https://diataxis.fr) — and one
page is one kind. The rules each kind has to meet, and the sentence-level bar,
are in [the docs-writing skill](.claude/skills/docs-writing/SKILL.md). Design
notes, measured cost, project direction and the changelog are explanation,
reachable and out of the way.

Three rules on top of that, each enforced, so none has to be remembered:

- **Every page under `docs/` needs a `nav:` entry** in `mkdocs.yml`. Adding a
  model page without one ships an unreachable page, so `pixi run test` refuses
  it. The build does not: zensical validates links and leaves navigation alone.
  `docs/README.md` is the one file with no entry — it is the folder view GitHub
  renders, the site builds no page from it, and `docs/index.md` is the home
  page.
- **Inside `docs/`, link relatively.** `../reference/api.md`,
  `examples/index.md`. The build resolves and validates these; a dead one fails
  it.
- **Outside `docs/`, write the full GitHub URL** —
  `https://github.com/fluxopt/specsolve/blob/main/bench/README.md`, not
  `../bench/README.md`. The site has no file above `docs/` to resolve to, and
  the build does *not* flag the relative form: it ships as a silent 404. This
  is the same convention the model pages already use to link at their `.yaml`.

`tests/test_docs_site.py` enforces all three — the nav in both directions, no
relative link escaping `docs/`, and every blob URL naming a file that exists.
None is checkable by the build, which is why they are tests.

Headings are slugged the way GitHub slugs them, so `#track-4--sink-capabilities`
means the same thing in both places.

Read the Docs builds and publishes from `main` (`.readthedocs.yaml`); nothing
needs deploying by hand.

## Branches, commits, PRs

**Never commit on `main`.** It takes squash merges through a PR only, and the
ruleset enforces it.

Merges are squashed, so the **PR title** becomes the commit on `main`. Write it
so it also works as the line you add to `CHANGELOG.md` by hand. It has to be a
conventional-commit subject:

```
feat: streaming engine for indexed constraints
fix(parser): where clauses with a trailing comma
refactor(api): closed helper set, no monkey-patch
```

A `feat`, `fix`, `perf`, `refactor`, `docs` or `revert` PR adds its title, with
a link to the PR, under `## Upcoming version` in `CHANGELOG.md`. A `chore`,
`test`, `ci`, `build` or `style` PR adds no line. The
`Conventional commit subject` check enforces the format on every pull request,
and the `Changelog line` check enforces the line. The label `no changelog` opts
a PR out of the second. Fixing either is an edit to the PR, not a branch
rewrite.

`main` is protected: no force-push, no deletion, squash-only through a PR, and
the required checks: `ci` and the two above. Approvals are not required, but the
PR is.

A release is a PR that renames `## Upcoming version` to the version and the day,
such as `## 0.1.0 (2026-10-01)`, and edits the section into the release notes.
Merging it tags the release, opens the GitHub release and publishes the package
to PyPI. The steps, what to do when one fails, and the one-time setup:
[RELEASING.md](RELEASING.md).

## Filing issues

**Cite behaviour and a file, not a private symbol and a line range.** An issue
outlives several refactors; one written against internals dies with the next and
takes its argument with it — four have had to be closed and re-filed for exactly
that. So write `bounds accept a parameter name or a number, not an expression
(mathspec/model.py)`, not a line number inside the loop that enforces it.

**`now` is the only order label**, capped at five; everything else is backlog.
Grouping is sub-issue parentage — a track is a parent issue — because parentage
is structural where a label mirroring a list is a copy that drifts.
`blocked:upstream` and `blocked:decision` say what an issue waits on. A
`decision` closes by *resolution*, not by work: on yes it becomes `roadmap`, on
no it becomes a row in the deliberate non-primitives table.

## Breaking changes are free

**The project holds no compatibility promise before 1.0.** So a construct that
is named wrong, a default that is wrong, or a permissive input that hides a
silent wrong answer gets **fixed in place**: rename, move and delete outright —
no alias for the old spelling, no `DeprecationWarning` cycle, no `legacy_` path
beside the new one.

Spend nothing on the retirement either. The closed schema already fails at load
naming the valid keys and the near miss, and the operator table already names
what it accepts — that is the whole migration story a release before 1.0 owes
anyone. A hand-written message per retired spelling is a second place the old
surface lives, it needs a test of its own, and it outlives every file it was
written for: `shift(by=)` had one for a day before `by=` became a legal keyword
again and the message started refusing the new spelling.

This binds **agents working in this repo** too, and it is the habit most often
imported from elsewhere: asked to change something, change it — do not add
backwards compatibility nobody requested.

It is the *surface* that is unfrozen, not the behaviour. What exists is tested
and differentially verified against linopy; a break is a deliberate rewrite, not
licence for churn.

## Changing the language

**Triage first: macro, primitive, or escape?** Most requests are compositions
and cost nothing. A genuinely new shape earns a primitive only if it clears the
expressive ceiling — relational ∩ local, degree 2 in the math and 1 beside it. Unsayable math goes to a
declared `escape:` island rather than into the language.

Read, in order:

1. [the deliberate non-primitives](https://mathspec.readthedocs.io/en/latest/about/limits/#deliberate-non-primitives) — parity with
   another tool is not by itself a reason to add anything, and several
   plausible-sounding features are refused there on purpose;
2. [the limits in mathspec](https://mathspec.readthedocs.io/en/latest/about/limits/#what-a-new-primitive-has-to-satisfy) —
   the admissibility test;
3. [the extension checklists](docs/about/architecture.md#extension-checklists), which sit directly under that
   test. They stay there rather than moving here: *may I?* and *how?* are one
   question, and splitting them invites answering the second without the first.

A PR that adds, renames or retires a construct updates the [language
reference](https://mathspec.readthedocs.io/en/latest/reference/language/).
Rationale belongs in the PR description or a code comment; "this used to work
differently" belongs in git.

## Adding a ported model

A port is a model somebody else already solved, said again in this language and
checked against **an optimum that did not come from us**. It is the only test
class that can catch a *shared misreading* — both lanes agreeing on a meaning
the modeller did not intend — because every other test compares specsolve against
specsolve. The corpus and the ledger of what a port could *not* say are in
[docs/examples/index.md](docs/examples/index.md), where the reference table is
generated from `examples/ports/references.json` — the same file the tests assert
against. The PyPSA ladder is a different instrument and lives in
[docs/examples/pypsa_ladder.md](docs/examples/pypsa_ladder.md): one page per
rung of mathspec's PyPSA corpus, generated from the parity runs, not ports. Each port's page there shows the model and a side-by-side
against its reference.

**Check the claim against the shipped instance before writing a file.** A model
is picked because its structure exercises something; twelve candidates chosen
that way produced six whose structure was not in the data they ship. Three
checks, in the order they catch things:

1. **Count the rows.** A map into a one-member dimension cannot change an
   answer, and a table with no rows is not structure. An empty `zone` table and
   a one-member `DAYTYPE` both read as topology until counted.
2. **Quote the source's own constraint that reads the map** — not its schema,
   not its prose. A source that models the same problem over three separate
   sets needs no map at all, whatever its data dictionary suggests.
3. **Run the reference and read the *solution*.** The first two are about the
   model; this one is about the instance. A constraint that holds at zero
   reads nothing, and a scenario parameter can switch one off — OSeMOSYS's
   UTOPIA passes 1 and 2 and still fails, because its season and day-type maps
   feed only storage constraints and the instance builds no storage.

Running the reference costs minutes and settles what a port otherwise
discovers after days of transcription. It is also what tells you the published
figure is still the model's answer: one candidate's asserted optimum had drifted
from what its own code computes, hidden by a loose tolerance.

Four files per port:

```
examples/ports/<name>.yaml                  the model
examples/ports/data/<name>.json             the instance
examples/ports/references/<arm>/<name>.py   a reference implementation, importing no specsolve
examples/ports/references.json              the recorded objective and where it came from
docs/examples/<name>.md                     the gallery page — maths, model, side-by-side
```

`<arm>` is the library the reference is written in — `linopy` or `pypsa` today,
`pyomo` when one arrives. The gallery page shows each arm as a tab beside the
YAML, and `tests/test_models_gallery.py` holds the two in lockstep: a script
with no tab fails, and so does a tab with no script behind it.

A *teaching* model may carry the same three reference files — the instance, the
script, the `references.json` entry — while its model file stays in
`examples/`, where the guide points. Its check is weaker and the provenance
says so: agreement with an independent hand-written formulation, not a
published figure.

- **A published optimum needs no script.** `transport_dantzig`'s number comes
  from the literature, and the citation *is* its provenance. It also ships a
  reference implementation, but as `corroborated_by` rather than as what
  verifies it — a second opinion, where the published figure is the first.
- **Reference scripts are never run by CI.** Pinning PyPSA into this project
  would hand their release cadence a veto over the suite. They carry their
  dependencies inline ([PEP 723](https://peps.python.org/pep-0723/)), pinned to
  whatever produced the recorded number, and are run out of band:
  ```bash
  pixi exec -s uv uv run --script examples/ports/references/pypsa/pypsa_transport.py
  ```
  pixi does not read inline script metadata, so `pixi exec` fetches the one
  tool that does into a throwaway environment. Nothing in this project's
  manifest names it: the pins belong beside the script that earned them.
- **Both sides read the same instance.** A reference optimum against a
  different instance means nothing. What must stay independent is the
  formulation, not the data.
- **`rtol` is per port.** A published optimum is rounded; a solved one is not.
- **Record the duals too, if the model has any.** An objective is one number;
  a dual vector is where two implementations disagree quietly — which side of a
  constraint the price belongs to, and what sign an inequality carries. The
  reference script prints a `duals {...}` line keyed by constraint name; paste
  it into the port's entry as a `duals` block. A MILP has no dual solution, so
  it records none and the test skips rather than passing vacuously.
- **Regenerate the gallery's tables**, don't write them: both the construct
  matrix and the reference table in `docs/examples/index.md` come from
  `pixi run python -m tools.constructs`, and a test fails if either is stale. The
  reference table is rendered straight from `references.json`, so the published
  optimum and the asserted one cannot disagree.
- **A rung that cannot be said is also a result.** It goes in the ledger with a
  verdict — macro, primitive or escape — and feeds docs/about/roadmap.md. Do not work
  around a gap silently.

## Refreshing the benchmarks

Full method, and why each measurement is taken the way it is, in
[bench/README.md](bench/README.md). The short version:

```bash
pixi run refresh
```

That is the five rungs and both writers in order — `ladder`, `density`,
`declarations`, then `report` and `plot`, each runnable on its own. The flags
live in `pyproject.toml` so the published numbers cannot come from a selection
somebody retyped.

Three things that have each cost us a wrong published number:

- **Measure on an idle machine.** A ladder taken while the laptop was busy
  inflated one case by 55% — enough to turn "level" into "the one case we lose".
- **A run replaces its output file.** Anything narrower than the published
  ladder goes to `--benchmark-json=/tmp/something.json`, or the tables keep
  their old numbers with a fingerprint that no longer describes them.
- **Never retype a number.** `bench.report` prints the markdown and
  `bench.plot` writes the chart page's rows, both from the results file. A
  figure typed by hand outlives the run that produced it.
