# Docs

This folder is the published site. [index.md](index.md) is its front door;
this page is the folder view on GitHub.

**The nav and the tree are arranged by the four kinds of
[Diátaxis](https://diataxis.fr)**: tutorials at the root, how-to guides
(`howto/`), reference (`reference/`, and the model pages in `examples/`) and
explanation (`about/`). Development (`development/`) holds what a user does
not need; its PyPSA pages stay in `examples/`. Each kind meets
[the docs-writing skill](https://github.com/fluxopt/specsolve/blob/main/.claude/skills/docs-writing/SKILL.md).
The language is documented in mathspec, and the nav links there.

**Two link rules**, enforced by `tests/test_docs_site.py`: inside `docs/`,
link relatively; outside it, write the full GitHub URL, because the relative
form 404s on the site. The rest is
[CONTRIBUTING.md](https://github.com/fluxopt/specsolve/blob/main/CONTRIBUTING.md#the-docs).

**Generated, so do not hand-edit:** the catalogue in
[examples/index.md](examples/index.md) and the construct matrix and reference
table in [development/verification.md](development/verification.md)
(`tools/constructs.py`), the *"the same model, as math"* block on each model
page (`tools/gallery_math.py`), and the tables in
[benchmarks.md](about/benchmarks.md) (`bench.report`) and the rows in
`about/benchmarks.json` (`bench.plot`). The
catalogue is read off the nav. The YAML and Python on the model pages, and the
model `README.md` lends [index.md](index.md), are asserted against the files
that run.

**What stays hand-written, and what checks it.** A model page opens with a
summary sentence, which the catalogue quotes. On six pages it also gives the
math as that field writes it, and `tests/test_typeset_gallery.py` requires
each to use only symbols the generator can reach, or to name why it deviates.
A page in neither list fails.
