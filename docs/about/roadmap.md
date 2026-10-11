# Roadmap

This page says why specsolve exists, where it is going and what it will not
become, for anyone about to propose a feature. **No work items live here.** The
work is issues, grouped under three parents:

- [Track 1 — primitives](https://github.com/fluxopt/specsolve/issues/470)
- [Track 2 — the operational surface](https://github.com/fluxopt/specsolve/issues/471)
- [Track 3 — capabilities, and the degree line](https://github.com/fluxopt/specsolve/issues/472)

A hand-maintained index beside an issue tracker is a second copy that drifts.
The issues are the list, and this page is the argument for what the list is
*for*.

## Why

An optimisation model is math worth reading, and it usually arrives as Python
that *builds* math. The equations are entangled with the loops, the
[tables](../reference/glossary.md#the-data) and the library that assembled
them. A diff then shows scaffolding rather than constraints, and nothing can
read the model except the program that wrote it. Reviewing such a model means
reviewing a program.

specsolve makes the math the artifact: a YAML file says what the variables,
constraints and objective *are*. The file is validated at load time and built
at runtime. Someone who understands the math can review it without
understanding the builder. Every rule below follows from that.

## Where it is going

**A build that streams, with a ceiling you can declare.** The model is tables
and the build is relational, so nothing dense is ever materialised. Peak memory
tracks the model rather than a number someone guessed. What is missing is the
*declaration*: there is no way to say "build this within N gigabytes or fail".
The honest version is partition-wise execution. Every operator in the language
today reads a bounded number of rows per output row, so the partitions are
already safe. An operator that reads a whole table would put one full pass
before the partitions rather than rule them out.

**A solve that explains itself.** A solved model should tell you why it is
infeasible, what a row costs and what changed since the last solve. It should
do so without opening a file no editor can hold. Most of that is a query over
tables that already exist.

**Component libraries, composed rather than generated.** A fixed set of
parametrised templates agree on a port/flow convention and merge into one
[program](../reference/glossary.md#the-chain) before a single build pass.
**Topology is data.** Wiring a system is rows in a connectivity table, never
generated YAML. So structure stays bounded by the number of component *types*,
while cardinality lives in data.

## What it will not become

**Two durable losses, and they are the price of the closed AST (abstract
syntax tree).** One is structure that needs the solver's *answer* to decide
the next row, inside one plan. The other is imperative modelling of any kind.
That price buys load-time validation, two lanes on one language, and a build
that streams.
Everything else is scheduling.

The specific refusals, each with its reason and its rewrite, are in
[the ceiling](https://mathspec.readthedocs.io/en/latest/about/limits/#deliberate-non-primitives):
data prep, arbitrary array ops, domain helpers, normalisation, in-plan
conditionals, a Python modelling API. Parity with another tool is not by
itself a reason to add anything.

## Honest snapshot

**Cheaper here, because the model is tables:** serialisation to parquet;
elastic relaxation; dualisation, since transposing a COO matrix is swapping two
column names. Model statistics and coefficient ranges were the first of these
and already ship ([diagnostics](../reference/api.md#specsolve.types.Diagnostics)).

**Ahead of comparable declarative layers:** a sparse-by-construction build with
no dense intermediate, and a hand-off straight to the solver rather than
through a file; parametrised `macros:` ([Calliope](prior-art.md)'s
sub-expressions take no arguments); binary and integer variables; piecewise as
N links with per-link signs, convex mode and `active` gating; load-time
validation of every expression, `where` string and *uncalled* macro template.

**Behind linopy**, and none of it a ceiling question: the post-solve object
(labelled DataArrays vs tidy tables; `to_dataarray` bridges), lifecycle
(`relax` and `fix` as verbs, where here they are
[loops](../howto/fix-relax-remove.md)), solver breadth (ten backends and four handoffs vs three direct
[sinks](../reference/glossary.md#how-it-runs) plus files), and the variable
types and constraint kinds the capability model still gates.

**The ranking this implies:** indexed access blocks whole model classes today;
the operational verbs block running the engine unattended; solver breadth blocks
arrival from linopy at all; semi-continuous and `cumsum`-over-data are cheap,
unblocked and unscheduled.
