# Code names against the glossary

This page is a worksheet and is not for merge. The glossary
([docs/reference/glossary.md](docs/reference/glossary.md)) now fixes the words
of prose, messages and docs. This page lists the identifiers that still carry
another word, one name per line, so that a comment or an edit applies to one
row. Counts are identifier uses in `src/` and `tests/` at the head of
fluxopt/specsolve#2166.

| Status | Meaning |
| --- | --- |
| 🔴 | rename: the identifier says a word the glossary does not use |
| 🔵 | decision needed: a rename touches a public name, a file on disk, or another PR |
| 🟢 | keep: the identifier says the glossary's word, or a library's own |

## One name per line

<!-- prettier-ignore-start -->
| Status | Name now | Uses | What it holds | Proposed | Public | Note | Agreed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 🔴 | `frame`, `frames` (locals, parameters) | src 400, tests 250 | a polars table | `table`, `tables` | no | polars' `DataFrame` and `LazyFrame` stay | yes, in review |
| 🔴 | `Term.frame`, `Presence.frame`, `Carrier.frame`, `Indexed.frame` | src 120 | a polars table | `.table` | no | | yes, in review |
| 🔴 | `_frames.py`, `as_frame`, `FrameWriter`, `parameter_frame`, `indices.frame()` | src 40, tests 5 | reading a caller's table; one entry's indexed table | `_tables.py`, `as_table`, `TableWriter`, `parameter_table`, `indices.table()` | no | `relations.table` exists; check for a clash | yes, in review |
| 🔴 | `param`, `join_param` | src 18 | a parameter's name | `parameter`, `join_parameter` | no | `pytest.param` stays | yes, in review |
| 🟢 | `mask` (locals holding a `program.Mask`) | src 5, tests about 30 | the condition of a `where:`, applied to an entry | keep `mask`, and the glossary defines it: the condition a `where:` narrows an entry by | no | a `when:` or `holds:` condition narrows no entry, so it is a condition and not a `mask`; mathspec's type is `Mask`, and linopy's `mask=` is the same idea; the glossary's table stops refusing `mask` | yes, in review |
| 🔴 | `document` | src 20 | the spec as written | `spec` | no | `_api.py`, `_inputs.py` | yes, in review |
| 🔴 | `by_coord` (tests/conftest.py), `coord_dims`, `COORDS` | tests 60 | a variable's primal keyed by coordinate | `primal_by_coordinate`, `coordinate_dims`, `COORDINATES` | no | coordinate is the word all three packages agree on (mathspec#1037); `tests/differential.py` already has a `by_coordinate`, so the helper takes the longer name; xarray's `coords` and linopy's `master_coords` stay | |
| 🔴 | `solution` naming a `Result`; fixture `dispatch_solution` | docs 103, tests 133 | a `Result` | `result`, `dispatch_result` | docs snippets | "the solution" for the values at the optimum stays, and so does `NoSolutionError` | yes, in review |
| 🔴 | `col_index` beside `row` | src 80 | a variable's and a constraint's solver index | `col` and `row`, as the matrix spells them | no | see the prefix row | yes, in review |
| 🔴 | `cval` | src 3 literals, many uses | a constant term's value | `constant` | no | see the prefix row | yes, in review |
| 🔴 | `val`, `ord` | src 30 | a dimension table's label, and its position from 0 in label order | `label`, `position` | no | `ord` and `specsolve_position` (`_names.POSITION`) hold the same number; one name, the one mathspec's `position()` counts; see the prefix row | |
| 🔴 | the engine's own columns: `row`, `col`, `coeff`, `constant`, `label`, `position`, `col_2` | src | columns a build writes beside a dimension's own | name each once in `_names.py` under `specsolve_`, as `RUN` and `POSITION` are: `pl.col(ROW)` reads plain, and `specsolve_row` is a name a dimension cannot take | no | plain names collide with a dimension named `row`, `label` or `constant`; under the prefix fluxopt/specsolve#2167 refuses nothing new | yes, in review |
| 🔴 | `Compiler`, `CompiledExpression`, `_engine/compiler.py` | src 96, tests 148 | an expression turned into polars queries | `QueryBuilder`, `ExpressionTerms`, `_engine/queries.py` | no | the glossary's compile is spec → program; `CompiledExpression` holds an expression's linear, quadratic and constant terms | rename yes, in review; names open |
| 🟢 | `dim` (field of `EachWindow`, `EachLabel`, `Stitch`; key in `axis.json`) | src 230 | a dimension's name | keep | yes, and on disk | the singular of the language's `dims:` | yes, in review |
| 🔴 | `key` (slice), `keys` / `keyed_by` (term), `key:` (relation) | src 375 | three things | keep the slice's key: a `Sweep` is a mapping from key to answer, which is Python's own word; a term's `keys` / `keyed_by` → `join_columns` / `joined_on` | `key_name=` is | a relation's `key:` is mathspec's YAML | yes, in review |
| 🟢 | `dims` | src 370, tests 390 | an entry's dimensions | keep | mathspec's attribute | the language's key and `Program`'s attribute | |
| 🟢 | `Label`, `labels`, `EachLabel` | src 190, tests 450 | a dimension's members | keep | yes | | |
| 🟢 | `Presence`, `presences` | src 100 | where the variable under a term exists | keep | no | defined in architecture.md | |
| 🟢 | `region`, `Region` | src 44 | one region of a `cases:` block | keep | no | mathspec's `Region` | |
| 🟢 | `Stitch`, `stitch` | src 43 | windowed tables put back over the dimension | keep | no | defined in architecture.md | |
| 🟢 | `Carrier` | src 53 | a table a walk joins onto | keep | no | | |
| 🟢 | `built`, `declared` | src 70, tests 270 | a built model; a declared name | keep | no | the glossary's verbs | |
| 🟢 | `tests/linopy_lane/`, `lane` | tests | the oracle | keep | no | defined in architecture.md | |
<!-- prettier-ignore-end -->

## How to work on it

- **Object to a row:** comment on its line under "Files changed".
- **Settle a row:** fill `Agreed`, and set the status to 🟢 or 🔴.
- **Done** when no row is 🔵. Then close without merging; each settled 🔴
  row lands in a PR of its own, the identifier and its uses together.
