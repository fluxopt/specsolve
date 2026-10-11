# Python API

This page is the reference for running a spec from Python: every public name,
rendered from its docstring. A *spec* is the YAML file; what it may contain is
[the language](https://mathspec.readthedocs.io/en/latest/reference/language/).

```python
import specsolve as sps

sps.check('spec.yaml')  # compiles? no data needed

result = sps.solve('spec.yaml', sources)
result.objective
result.primal('p')  # a polars.DataFrame
result.dual('power_balance')
```

## Reference

Every public name, rendered from its docstring. Three modules hold them:

- `specsolve` holds what you call;
- `specsolve.types` holds what a call hands back;
- `specsolve.errors` holds what a call raises or warns.

Any other name under `specsolve.` is internal, and any release can change it.
The [glossary](glossary.md) defines *model*, *result*, *sink* and the other
house terms the entries use.

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

The fold and its two axes; [sweeps](sweeps.md) says how a sweep is cut and read.

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

The rows and frames those hand back: how a solve terminated, what produced
it, what the build and its solves took, and what a slice of a sweep took.

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

Every error is one tree, rooted at `SpecsolveError`. A spec the language
accepts and specsolve cannot build raises `SpecsolveError` itself, and its
message names the rewrite. `LanguageError`, with `SchemaError` and
`DimensionError`, is a fault in the spec, and is the language's own:
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

## Rules across the verbs

What no single entry above holds, because every verb keeps it.

### Names that differ only by case

**Two declarations of one namespace whose names differ only by case are
refused**, whichever verb lowers the spec. Every declaration is written to
disk as a file named after it, and a case-insensitive filesystem, which a
stock macOS or Windows volume is, folds `p` and `P` into one file.

```
variable 'P' and variable 'p' differ only by case, and one answer on disk
cannot hold both: ... Tell them apart by a suffix rather than a capital:
'p_rated' beside 'p'.
```

The namespaces are the language's own: one flat namespace holding dimensions,
relations, parameters, variables and named expressions, and constraints beside
it. A constraint may carry a variable's name already, so a constraint `P`
beside a variable `p` is accepted. The two are written under `dual/` and
`primal/`, which nothing folds together.

### Names that start with `specsolve_`

**A declared name that starts with `specsolve_` is refused**, in any letter
case, whichever verb lowers the spec. The prefix is reserved for the columns
specsolve adds, such as `specsolve_run` on every table an archive holds, so a
declared name cannot collide with one. Case does not tell two columns apart:
SQL, DuckDB and Power BI read `Specsolve_run` as `specsolve_run`. The rule
covers dimensions, relations and their columns, parameters, variables,
constraints, named expressions, `sos:` sets and assumptions. A `key_name=`
with the prefix is refused too, and so is a dimension called `value`, which
would collide with the column that holds a parameter's numbers.

```
variable 'Specsolve_p' starts with 'specsolve_', which is reserved in any
letter case for the columns specsolve adds, so it could collide with one.
Rename it.
```

### What each sink takes

`Model.check(sink)` refuses a built model the sink cannot ingest, naming the
sinks that do, and `solve` and `write` refuse the same. The answer is read off
the model the build produced, not the file: a square the data prices at zero,
an integer variable no column is built for or a set with no members asks for
nothing. Where a model can land is
[a separate question](https://mathspec.readthedocs.io/en/latest/about/what-counts-as-language/#what-each-tool-decides-for-itself)
from whether it is sayable. The four quadratic rows, and the two sections
HiGHS writes but will not read back, are probed against the shipped solvers by
`tests/test_sink_capability_probes.py` and
`tests/test_gurobi_capability_probes.py`. The rest are read off the APIs.

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

- **HiGHS excludes quadratic twice**: by convexity, and by conjunction with
  integrality.
- **The `lp_file` column says what can be written, not what reads back.** The
  same HiGHS parser takes the quadratic-objective section and refuses the
  `sos` and quadratic-constraint sections.
- **"No path here" describes this package, not Xpress.** The Optimizer takes
  a Hessian; the sink in `solvers/xpress.py` never hands it one.
