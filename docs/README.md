# Docs

This folder is both the published site and what you read on GitHub.
[index.md](index.md) is the site's front door and this page is the folder
view. Start at [running a model](run.md).

**The nav and the tree are arranged by what a page is for**, the four kinds
of [Diátaxis](https://diataxis.fr): tutorials (`run.md`, `tables.md`,
`change.md` and `sweep.md`, at the root), how-to guides (`howto/`), reference
(`reference/`, and the model pages in `examples/`) and explanation
(`about/`). The rules each kind
has to meet are
[the docs-writing skill](https://github.com/fluxopt/specsolve/blob/main/.claude/skills/docs-writing/SKILL.md).
The language is a dependency documented with itself, so the nav links out to
mathspec.

**Two link rules**, enforced by `tests/test_docs_site.py`: inside `docs/`,
link relatively; outside it, write the full GitHub URL, because the relative
form 404s on the site. This page follows them too. The rest is
[CONTRIBUTING.md](https://github.com/fluxopt/specsolve/blob/main/CONTRIBUTING.md#the-docs).

**Generated, so do not hand-edit:** the catalogue, construct matrix and
reference table in [examples/index.md](examples/index.md)
(`tools/constructs.py`), the *"the same model, as math"* block on each model
page (`tools/gallery_math.py`), and the tables in
[benchmarks.md](about/benchmarks.md) (`bench.report`, `bench.plot`). The
catalogue is read off the nav, so a model joins the gallery by joining the
sidebar. The YAML and Python on the model pages, and the model `README.md`
lends [index.md](index.md), are asserted against the files that run.

**What stays hand-written, and what checks it.** A model page opens with a
summary sentence, which the catalogue quotes, and on six pages the math as
that problem is usually written. `tests/test_typeset_gallery.py` requires each
of those six to use only symbols the generator can reach, or to name why it
deviates: `tsp_mtz` states DFJ, which the language refuses; `storage` writes
`soc_{s-1}` where the generator writes the cyclic `⊖`. A page in neither list
fails.
