# Code names against the glossary

This page is a worksheet and is not for merge. The glossary
([docs/reference/glossary.md](docs/reference/glossary.md)) now fixes the words
of prose, messages and docs. This page lists the identifiers that still carry
another word, one name per line, so that a comment or an edit applies to one
row. Counts are identifier uses in `src/` and `tests/` at the head of
fluxopt/specsolve#2166.

| Status | Meaning |
| --- | --- |
| 🔴 | rename proposed: the identifier says a word the glossary does not use |
| 🔵 | decision needed: a rename touches a public name, a file on disk, or another PR |
| 🟢 | keep: the identifier says the glossary's word, or a library's own |

## One name per line

<!-- prettier-ignore-start -->
| Status | Name now | Uses | What it holds | Proposed | Public | Note | Agreed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 🔴 | `frame`, `frames` (locals, parameters) | src 400, tests 250 | a polars table | `table`, `tables` | no | polars' `DataFrame` and `LazyFrame` stay | |
| 🔴 | `Term.frame`, `Presence.frame`, `Carrier.frame`, `Indexed.frame` | src 120 | a polars table | `.table` | no | | |
| 🔴 | `_frames.py`, `as_frame`, `FrameWriter`, `parameter_frame`, `indices.frame()` | src 40, tests 5 | reading a caller's table; one entry's indexed table | `_tables.py`, `as_table`, `TableWriter`, `parameter_table`, `indices.table()` | no | `relations.table` exists; check for a clash | |
| 🔴 | `param`, `join_param` | src 18 | a parameter's name | `parameter`, `join_parameter` | no | `pytest.param` stays | |
| 🔴 | `mask` (locals holding a `program.Mask`) | src 5, tests about 30 | the value of a `where:` | `where`, as `indices.table(scope, dims, where, …)` already says | no | `program.Mask` is mathspec's type; linopy's `mask=` in `tests/linopy_lane/` stays | |
| 🔴 | `document` | src 20 | the spec as written | `spec` | no | `_api.py`, `_inputs.py` | |
| 🔴 | `by_coord`, `coord_dims`, `COORDS` | tests 60 | a coordinate | `by_coordinate`, `coordinate_dims`, `COORDINATES` | no | xarray's `coords`, `assign_coords`, linopy's `master_coords` stay | |
| 🔴 | `solution` naming a `Result`; fixture `dispatch_solution` | docs 103, tests 133 | a `Result` | `result`, `dispatch_result` | docs snippets | "the solution" for the values at the optimum stays, and so does `NoSolutionError` | |
| 🔵 | `col_index` beside `row` | src 80 | a variable's and a constraint's solver index | one pair, `col` and `row`, as the matrix spells them | no | a dimension called `col` then collides; fluxopt/specsolve#2167 adds it to the refused names | |
| 🔵 | `cval` | src 3 literals, many uses | a constant term's value | `constant` | no | in fluxopt/specsolve#2167's refused names; the refusal moves with it | |
| 🔵 | `val`, `ord` | src | a dimension table's label and ordinal | `label`, `ordinal` | no | `ord` is in fluxopt/specsolve#2167's refused names | |
| 🔵 | `Compiler`, `CompiledExpression`, `_engine/compiler.py` | src 96, tests 148 | an expression turned into polars queries | `QueryBuilder`, `Query`, `_engine/queries.py`; or keep and define both senses of compile in architecture.md | no | the glossary's compile is spec → program | |
| 🔵 | `dim` (field of `EachWindow`, `EachLabel`, `Stitch`; key in `axis.json`) | src 230 | a dimension's name | keep, or `dimension` | yes, and on disk | `dims:` is the language's key, so `dim` is its singular; a rename raises `INPUTS_LAYOUT` | |
| 🔵 | `key` (slice), `keys` / `keyed_by` (term), `key:` (relation) | src 375 | three things | keep the slice's key, as the glossary has it; a term's `keys` → `join_columns` | `key_name=` is | a relation's `key:` is mathspec's YAML | |
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
