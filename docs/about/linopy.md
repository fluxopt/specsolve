# Relationship to linopy

Everything about [linopy](https://github.com/PyPSA/linopy) in one place, for a
reader who arrives from linopy or PyPSA. There are two separate
relationships:

| | What | Where it matters |
|---|---|---|
| **Not a dependency** | nothing in the package imports it | packaging |
| **The oracle** | how we know the answers are right | testing |

## 1. It is not a dependency

`sps.solve`, `sps.build`, `sps.write` and `sps.check` go YAML → polars → HiGHS
or file, and import nothing from linopy, xarray or pandas. The bare-install job
runs the whole suite with none of the three present. No install of specsolve
brings linopy: it is a test dependency, in the `dev` group.

The `to_pandas` / `to_dataarray` bridges out of a
[result](../reference/glossary.md#the-chain) need pandas and xarray, which the
caller installs.

**Nothing a bare install can reach names linopy, including a traceback.** The
public exception tree is rooted at `SpecsolveError`, with no alias
([#389](https://github.com/fluxopt/specsolve/issues/389)).

## 2. It is the oracle

The test suite builds every file a second time as a `linopy.Model`, and
compares the two builds. [Verification](../development/verification.md#the-oracle)
says how, and what the oracle cannot catch.

## Parts of linopy not taken

specsolve does not take array operations (`merge`, `reindex`, `stack`), the Python
modeling API, or the solver layer. The first is data prep
([the limits](https://mathspec.readthedocs.io/en/latest/reference/language/errors/#what-the-language-will-not-express)).
The second is [hard rule 5](../development/architecture.md#hard-rules): the spec is the file
you review and diff. The third is
[#106](https://github.com/fluxopt/specsolve/issues/106), where specsolve adopts
linopy's *design* for declared solver capabilities without adopting its code.

The modeling API is what a reader arriving from linopy misses first. Two
pages replace it. The tutorial [Change it and solve again](../change.md)
covers `update` for new numbers and a longer table for more rows. The how-to
[Fixing, relaxing and removing](../howto/fix-relax-remove.md) aims the same
loops at `fix`, `relax` and `remove_constraints`. A built row is read with [`row`](../reference/api.md#specsolve.types.Model.row), in linopy's
own form, and an IIS with [`infeasible_subsystem`](../reference/api.md#specsolve.types.Model.infeasible_subsystem).

Where linopy is ahead, and why none of it is a ceiling question, is
[the roadmap](roadmap.md#honest-snapshot). What is *owed* to linopy rather than
merely true of it is [prior art and credit](prior-art.md). The same page credits
Calliope, whose math language this surface is derived from.
