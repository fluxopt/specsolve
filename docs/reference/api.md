# Python API

Every public name, rendered from its docstring. A *spec* is the YAML file;
what it may contain is
[the language](https://mathspec.readthedocs.io/en/latest/reference/language/).

## Reference

`specsolve` holds what you call, `specsolve.types` what a call hands back,
and `specsolve.errors` what a call raises or warns. Every other module starts
with an underscore, and any other name under `specsolve.` is internal. The
[glossary](glossary.md) defines the house terms.

### Run a spec

::: specsolve.check
    options:
      heading_level: 4

::: specsolve.build
    options:
      heading_level: 4

::: specsolve.solve
    options:
      heading_level: 4

::: specsolve.write
    options:
      heading_level: 4

::: specsolve.evaluate
    options:
      heading_level: 4

::: specsolve.tidy
    options:
      heading_level: 4

### Run it many times

[Sweeps](sweeps.md) says how a sweep is cut and read.

::: specsolve.solve_over
    options:
      heading_level: 4

::: specsolve.EachCoordinate
    options:
      heading_level: 4

::: specsolve.EachWindow
    options:
      heading_level: 4

### What comes back

::: specsolve.types.Model
    options:
      heading_level: 4

::: specsolve.types.ConstraintRow
    options:
      heading_level: 4

::: specsolve.types.Result
    options:
      heading_level: 4

::: specsolve.types.InfeasibleSubsystem
    options:
      heading_level: 4

::: specsolve.types.Sweep
    options:
      heading_level: 4

::: specsolve.types.ResultArchive
    options:
      heading_level: 4

::: specsolve.types.SweepArchive
    options:
      heading_level: 4

::: specsolve.types.ArchivedInputs
    options:
      heading_level: 4

The rows and frames those hand back:

::: specsolve.types.Diagnostics
    options:
      heading_level: 4

::: specsolve.types.Record
    options:
      heading_level: 4

::: specsolve.types.Provenance
    options:
      heading_level: 4

::: specsolve.types.Metrics
    options:
      heading_level: 4

::: specsolve.types.Output
    options:
      heading_level: 4

::: specsolve.types.Start
    options:
      heading_level: 4

### Read an answer back

::: specsolve.load_archive
    options:
      heading_level: 4

::: specsolve.load_inputs
    options:
      heading_level: 4

::: specsolve.load_result
    options:
      heading_level: 4

::: specsolve.load_sweep
    options:
      heading_level: 4

::: specsolve.scan_archive
    options:
      heading_level: 4

::: specsolve.scan_result
    options:
      heading_level: 4

::: specsolve.scan_sweep
    options:
      heading_level: 4

### Errors and warnings

Every error is rooted at `SpecsolveError`. A spec the language accepts and
specsolve cannot build raises `SpecsolveError` itself. `LanguageError`,
`SchemaError` and `DimensionError` are faults in the spec, and the language's
own:
[which error you get](https://mathspec.readthedocs.io/en/latest/reference/language/errors/#which-error-you-get).

::: specsolve.errors.SpecsolveError
    options:
      heading_level: 4

::: specsolve.errors.LanguageError
    options:
      heading_level: 4

::: specsolve.errors.SchemaError
    options:
      heading_level: 4

::: specsolve.errors.DimensionError
    options:
      heading_level: 4

The rest are specsolve's:

::: specsolve.errors.DataError
    options:
      heading_level: 4

::: specsolve.errors.LayoutError
    options:
      heading_level: 4

::: specsolve.errors.NoSolutionError
    options:
      heading_level: 4

::: specsolve.errors.SpecsolveWarning
    options:
      heading_level: 4

## Rules across the functions

### Names that differ only by case

**Two declarations of one namespace whose names differ only by case are
refused** by every function. A case-insensitive filesystem folds `p` and `P` into
one file on disk.

```
variable 'P' and variable 'p' differ only by case, and one answer on disk
cannot hold both: ... Tell them apart by a suffix rather than a capital:
'p_rated' beside 'p'.
```

Constraints are a namespace of their own, so a constraint `P` beside a
variable `p` is accepted.

### Names that start with `specsolve_`

**A declared name that starts with `specsolve_`, in any letter case, is
refused** by every function. The prefix is reserved for the columns specsolve adds,
such as `specsolve_run`. The rule covers dimensions, sets and their
members, properties, parameters, variables, constraints, named expressions, `sos:` sets and
assumptions. A `key_name=` with the prefix is refused too, and so is a
dimension called `value`.

```
variable 'Specsolve_p' starts with 'specsolve_', which is reserved in any
letter case for the columns specsolve adds, so it could collide with one.
Rename it.
```

### What each sink takes

`Model.check(sink)` refuses a built model the sink cannot ingest, naming the
sinks that do. `solve` and `write` refuse the same. The check reads the built
model, so a square the data prices at zero, an integer variable with no column
or an empty set asks for nothing. Where a model can land is
[a separate question](https://mathspec.readthedocs.io/en/latest/about/what-counts-as-language/#what-each-tool-decides-for-itself)
from whether it is sayable.

| | `lp_file` | `mps_file` | HiGHS direct | Gurobi direct | Xpress direct |
|---|---|---|---|---|---|
| affine rows, COO, integrality | text | text, `MARKER` | native | native | native |
| semi-continuous | text | **not written** — no `SC` bound | `kSemiContinuous` | native | native |
| SOS1 / SOS2 | text section | `SOS` section | **no concept** — refused, naming `Spec.expand()` | `addSOS` | native |
| indicator | text section | **not written** | **no concept** | `addGenConstrIndicator` | native |
| convex quadratic objective | text section | **not written** | `passHessian` | `setMObjective` | **no path here** |
| nonconvex quadratic objective | text section | **not written** | **refused** | native, at default parameters | **no path here** |
| quadratic objective **and** integrality | text section | **not written** | **refused** | native (MIQP) | **no path here** |
| quadratic constraint | text section, unreadable | **not written** | **no concept** | `addQConstr` | **no path here** |

- **HiGHS refuses a nonconvex quadratic, and a quadratic with integrality.**
- **The `lp_file` column says what is written, not what reads back.** HiGHS
  reads the quadratic-objective section back and refuses the `sos` and
  quadratic-constraint sections.
- **"No path here" describes this package, not Xpress.** The Xpress sink never
  hands the Optimizer a Hessian.
