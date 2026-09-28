# Changelog

Each pull request adds its line under `## Upcoming version`. A release pull
request renames that heading to the version and the day, such as
`## 0.1.0 (2026-10-01)`, and edits the section into the release notes. Merging
it releases that version ([RELEASING.md](https://github.com/fluxopt/specsolve/blob/main/RELEASING.md)).

## Upcoming version

- fix(api): a reported quotient is absent where its divisor is absent or zero, and a divisor that is a sum divides as its total ([#1776](https://github.com/fluxopt/specsolve/pull/1776))

## 0.2.0 (2026-09-25)

A saved answer now records which layout it was written in and which specsolve
wrote it, so a later release can name an answer it no longer reads. The Python
API reference is now rendered from the docstrings.

This release breaks one thing:

- An answer or an archive that 0.1.0 or earlier wrote is refused with a
  `LayoutError` that says to solve the model again and save it. Its
  `format.json` reads `{"answer": 0}`; 0.2.0 writes
  `{"layout": 1, "specsolve": "0.2.0"}`. An archive still holds the spec and
  the data it was solved with, under `spec.yaml` and `sources/`.

The pull requests since 0.1.0:

- feat(api)!: a saved answer is stamped with layout 1 and the specsolve version that wrote it, and one written by 0.1.0 or earlier is refused by name ([#1765](https://github.com/fluxopt/specsolve/pull/1765))
- docs: each fact about the Python API is stated once, in the docstring of the name it belongs to ([#1769](https://github.com/fluxopt/specsolve/pull/1769))
- docs: the Python API page renders every public name from its docstring ([#1766](https://github.com/fluxopt/specsolve/pull/1766))
- docs(sweeps): the sweep reference says how many sessions a sweep opens on a remote Gurobi ([#1771](https://github.com/fluxopt/specsolve/pull/1771))

## 0.1.0 (2026-09-25)

🎉 specsolve is officially on PyPI!

```bash
pip install specsolve
```

specsolve 0.1.0 requires mathspec 0.2.0 and uses its words. A **spec** is the
problem a file states, with no data. A **model** is a spec with data attached,
which `build` returns as a `Model`
([glossary](https://specsolve.readthedocs.io/en/latest/reference/glossary/)).
The messages and the docs use the two words in these senses.

Since 0.1.0rc1, this release breaks three things:

- An archive holds its spec as `spec.yaml`, not `model.yaml`. An archive that
  0.1.0rc1 wrote is refused with a `LayoutError`. To read one, rename its
  `model.yaml` member to `spec.yaml`.
- specsolve requires mathspec 0.2.0. There, the first parameter of `to_spec`,
  `advice` and the typesetting functions is `spec`, not `model`, and the module
  `mathspec.model` is `mathspec.spec`. A call that passes `model=` by keyword
  fails. A positional call works as before.
- Messages say "spec" where they said "model" for the file. A test that
  matches the old text fails.

The pull requests since 0.1.0rc1:

- docs: the site follows the reader's light or dark setting in its own violet colour, with code, diagrams and footnotes readable in both ([#1767](https://github.com/fluxopt/specsolve/pull/1767))
- feat!: specsolve requires mathspec 0.2.0 and uses its words spec and model, so an archive holds its spec as spec.yaml ([#1768](https://github.com/fluxopt/specsolve/pull/1768))
- docs: every link to the language's documentation points at mathspec.readthedocs.io ([#1764](https://github.com/fluxopt/specsolve/pull/1764))
- fix: the PyPI page links the docs, issues and changelog, and a stale saved answer says the layout moves before 1.0 ([#1763](https://github.com/fluxopt/specsolve/pull/1763))

## 0.1.0rc1 (2026-09-25)

The release candidate for 0.1.0, the first release on PyPI:
`pip install specsolve==0.1.0rc1`. The `0.0.1-alpha.N` releases below were tags
and GitHub releases only.

- refactor(api): specsolve no longer builds a linopy model, and its linopy extra and LaneError are gone ([#1755](https://github.com/fluxopt/specsolve/pull/1755))
- docs: the solve example archives its run and reads it back, with annotations that name the other options ([#1748](https://github.com/fluxopt/specsolve/pull/1748))

## [0.0.1-alpha.359](https://github.com/fluxopt/specsolve/compare/v0.0.1-alpha.358...v0.0.1-alpha.359) (2026-09-25)


### Documentation

* the configuration files keep a one-line reason only where a setting would look like a mistake ([#1756](https://github.com/fluxopt/specsolve/issues/1756)) ([7491f91](https://github.com/fluxopt/specsolve/commit/7491f91f538332efdcf1cc9c53aea701d904adf4))

## [0.0.1-alpha.358](https://github.com/fluxopt/specsolve/compare/v0.0.1-alpha.357...v0.0.1-alpha.358) (2026-09-25)


### Documentation

* a tutorial feeds the dispatch model from parquet, reads the answer as tables and queries its archive ([#1747](https://github.com/fluxopt/specsolve/issues/1747)) ([5ecc2ec](https://github.com/fluxopt/specsolve/commit/5ecc2ecd4195094ec21806a55619558496c9de11))
* a tutorial solves one model once per scenario and then window by window ([#1746](https://github.com/fluxopt/specsolve/issues/1746)) ([7c15f18](https://github.com/fluxopt/specsolve/commit/7c15f18fe81577875494360450ef3e56eb41a855))
* the readme and home page lead with four benefits for energy system modellers, in cards with icons, and the readme is under half its length ([#1743](https://github.com/fluxopt/specsolve/issues/1743)) ([e14f3b8](https://github.com/fluxopt/specsolve/commit/e14f3b822129c37738375ec51375e267baa3d013))

## [0.0.1-alpha.357](https://github.com/fluxopt/specsolve/compare/v0.0.1-alpha.356...v0.0.1-alpha.357) (2026-09-25)


### Features

* **language:** an open bound is null and an infinite bound is refused, the way mathspec 0.0.0-alpha.127 does ([#1744](https://github.com/fluxopt/specsolve/issues/1744)) ([1b1dbef](https://github.com/fluxopt/specsolve/commit/1b1dbefe7687b052dc75d55cde8f4c8cfab1cd26))

## [0.0.1-alpha.356](https://github.com/fluxopt/specsolve/compare/v0.0.1-alpha.355...v0.0.1-alpha.356) (2026-09-25)


### Documentation

* the glossary says a model holds no data, as math-spec uses the word, and every link into math-spec's docs resolves ([#1740](https://github.com/fluxopt/specsolve/issues/1740)) ([da3f211](https://github.com/fluxopt/specsolve/commit/da3f211f0a073880d3b9bb37c62afe27fcc52a0d))

## [0.0.1-alpha.355](https://github.com/fluxopt/specsolve/compare/v0.0.1-alpha.354...v0.0.1-alpha.355) (2026-09-25)


### Features

* **api:** a result hands back how its solve terminated as one Record ([#1738](https://github.com/fluxopt/specsolve/issues/1738)) ([c6ffe74](https://github.com/fluxopt/specsolve/commit/c6ffe7483d3f94c736d3853557e52b06b2d67f8b))

## [0.0.1-alpha.354](https://github.com/fluxopt/specsolve/compare/v0.0.1-alpha.353...v0.0.1-alpha.354) (2026-09-25)


### Refactoring

* **api:** a sweep's answer is a Sweep, read back with load_sweep and scan_sweep ([#1732](https://github.com/fluxopt/specsolve/issues/1732)) ([24b1238](https://github.com/fluxopt/specsolve/commit/24b123840eb8eba1d71a0ade5029b0ba63d3d428))
* **api:** how each slice of a sweep terminated is runs.record, saved as record.parquet, so objective always means the number ([#1731](https://github.com/fluxopt/specsolve/issues/1731)) ([7c14833](https://github.com/fluxopt/specsolve/commit/7c1483395cc2752e54b0f1c6867121ffc81139b1))


### Documentation

* the glossary defines answers, sweeps, relations and assumptions, and no longer shows build taking a Program ([#1730](https://github.com/fluxopt/specsolve/issues/1730)) ([25d594b](https://github.com/fluxopt/specsolve/commit/25d594b9929a584acea48961d8f8fe3ab68db606))

## [0.0.1-alpha.353](https://github.com/fluxopt/specsolve/compare/v0.0.1-alpha.352...v0.0.1-alpha.353) (2026-09-24)


### Documentation

* data is attached to a spec everywhere, so bound only ever means a limit ([#1728](https://github.com/fluxopt/specsolve/issues/1728)) ([ecf1019](https://github.com/fluxopt/specsolve/commit/ecf1019a6ec1ea68180a2b78abf39671755bd0d1))

## [0.0.1-alpha.352](https://github.com/fluxopt/specsolve/compare/v0.0.1-alpha.351...v0.0.1-alpha.352) (2026-09-24)


### Refactoring

* **api:** the package is named specsolve, installed and imported under that name ([#1726](https://github.com/fluxopt/specsolve/issues/1726)) ([a9aea61](https://github.com/fluxopt/specsolve/commit/a9aea61df5e5a60db76912e2fe5ab196c11ebe97))

## [0.0.1-alpha.351](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.350...v0.0.1-alpha.351) (2026-09-23)


### Features

* **language:** a special-ordered set names its dimension with along, the way math-spec 0.0.0-alpha.120 does ([#1720](https://github.com/fluxopt/lpspec/issues/1720)) ([fdc47c5](https://github.com/fluxopt/lpspec/commit/fdc47c51aa21b9f5ad6b9cd81b1a3381797bce1c))

## [0.0.1-alpha.350](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.349...v0.0.1-alpha.350) (2026-09-23)


### Features

* **language:** a where may read a predicate through a relation with at() ([#1718](https://github.com/fluxopt/lpspec/issues/1718)) ([cfaa01b](https://github.com/fluxopt/lpspec/commit/cfaa01bfd880a8550c335e3ff4054992a6a56332))

## [0.0.1-alpha.349](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.348...v0.0.1-alpha.349) (2026-09-22)


### Features

* **api:** a curve is written out before a verb reads it, and a sink with no SOS concept refuses a set rather than rewriting it ([#1713](https://github.com/fluxopt/lpspec/issues/1713)) ([7516597](https://github.com/fluxopt/lpspec/commit/7516597014817d061f754e98ced921d1bbd68930))


### Bug Fixes

* **api:** the language refuses a curve left as written, so the door no longer has to ([#1714](https://github.com/fluxopt/lpspec/issues/1714)) ([8021377](https://github.com/fluxopt/lpspec/commit/8021377eba0fdb9e92dfb2ee87da84c34f3c3adb))

## [0.0.1-alpha.348](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.347...v0.0.1-alpha.348) (2026-09-22)


### Refactoring

* **data:** the caller binds every parameter a piecewise curve reads, and no name the expansion invented ([#1711](https://github.com/fluxopt/lpspec/issues/1711)) ([7496de9](https://github.com/fluxopt/lpspec/commit/7496de928fe2ba336dbd576b1b2e3ae282d95eb7))

## [0.0.1-alpha.347](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.346...v0.0.1-alpha.347) (2026-09-22)


### Features

* **language:** a model states what it assumes of its data, and a where may compare arithmetic, count coordinates and shift a predicate ([#1707](https://github.com/fluxopt/lpspec/issues/1707)) ([e107514](https://github.com/fluxopt/lpspec/commit/e10751445e3f20ba942781f63050304da771aecd))

## [0.0.1-alpha.346](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.345...v0.0.1-alpha.346) (2026-09-21)


### Bug Fixes

* **language:** a pullback that reads a column over the dimension it joins on is refused at load ([#1703](https://github.com/fluxopt/lpspec/issues/1703)) ([e92827f](https://github.com/fluxopt/lpspec/commit/e92827fd4b50a9e6c06893363e72dad34d6c2962))

## [0.0.1-alpha.345](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.344...v0.0.1-alpha.345) (2026-09-19)


### Refactoring

* only the linopy lane and the solver sinks say Any ([#1699](https://github.com/fluxopt/lpspec/issues/1699)) ([067f14d](https://github.com/fluxopt/lpspec/commit/067f14dd8814a6e3e16f9b5a4e41cfa8337d1e71))

## [0.0.1-alpha.344](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.343...v0.0.1-alpha.344) (2026-09-19)


### Documentation

* the site is built by zensical, and the two tutorials run as pages rather than notebooks ([#1693](https://github.com/fluxopt/lpspec/issues/1693)) ([48633b2](https://github.com/fluxopt/lpspec/commit/48633b2b1cca7ce986c15edbe3eff41a7bc6135b))

## [0.0.1-alpha.343](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.342...v0.0.1-alpha.343) (2026-09-18)


### Features

* **language:** a call through a relation names the columns it walks, so a relation may gain a value column without changing what the call means ([#1691](https://github.com/fluxopt/lpspec/issues/1691)) ([451b7c7](https://github.com/fluxopt/lpspec/commit/451b7c73982b474b57a0c45abe990329bc42082f))

## [0.0.1-alpha.342](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.341...v0.0.1-alpha.342) (2026-09-18)


### Bug Fixes

* **api:** an archive in a run=&lt;name&gt; directory stamps the name the path gives ([#1689](https://github.com/fluxopt/lpspec/issues/1689)) ([7fc0251](https://github.com/fluxopt/lpspec/commit/7fc02510f62ba6eafdf5d3d4a4318f0b36d524d5))

## [0.0.1-alpha.341](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.340...v0.0.1-alpha.341) (2026-09-18)


### Features

* **api:** an answer says which data it came back from, not just which spec ([#1672](https://github.com/fluxopt/lpspec/issues/1672)) ([f66fffc](https://github.com/fluxopt/lpspec/commit/f66fffcfe30a118bed891ef9fe899695774657d7))

## [0.0.1-alpha.340](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.339...v0.0.1-alpha.340) (2026-09-18)


### Features

* **api:** an infeasible solve says why there is no solution, so a feasibility cut needs no second model ([#1682](https://github.com/fluxopt/lpspec/issues/1682)) ([7cc52eb](https://github.com/fluxopt/lpspec/commit/7cc52ebf61539d2ada415b3752808e8dfe1d3c25))

## [0.0.1-alpha.339](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.338...v0.0.1-alpha.339) (2026-09-18)


### Features

* **language:** a relation declares key columns and value columns ([#1681](https://github.com/fluxopt/lpspec/issues/1681)) ([df67ebc](https://github.com/fluxopt/lpspec/commit/df67ebc1b15e463e6fab147967412ba0c39a6708))

## [0.0.1-alpha.338](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.337...v0.0.1-alpha.338) (2026-09-18)


### Documentation

* the decomposition page shows the loop the example runs, and says lpspec owns no driver ([#1678](https://github.com/fluxopt/lpspec/issues/1678)) ([f1d33b3](https://github.com/fluxopt/lpspec/commit/f1d33b345d838cb59bce50ab66ec9f6e2a859981))

## [0.0.1-alpha.337](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.336...v0.0.1-alpha.337) (2026-09-16)


### Bug Fixes

* **engine:** a cased quantity summed onto a constant side builds, and the coverage guards have one home ([#1665](https://github.com/fluxopt/lpspec/issues/1665)) ([96a437d](https://github.com/fluxopt/lpspec/commit/96a437d8e38aa856881aa4652a1e5ac2d5cd08ad))

## [0.0.1-alpha.336](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.335...v0.0.1-alpha.336) (2026-09-16)


### Features

* **engine:** the relational lane builds every shape of relation the language admits ([#1659](https://github.com/fluxopt/lpspec/issues/1659)) ([88914c4](https://github.com/fluxopt/lpspec/commit/88914c41b7795acbc40c6739218481e0a80e0012))

## [0.0.1-alpha.335](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.334...v0.0.1-alpha.335) (2026-09-16)


### Bug Fixes

* **data:** a relation whose two columns are over one dimension builds ([#1657](https://github.com/fluxopt/lpspec/issues/1657)) ([2f96905](https://github.com/fluxopt/lpspec/commit/2f969057c0e5528539abf439e69eed5dc55521b0))

## [0.0.1-alpha.334](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.333...v0.0.1-alpha.334) (2026-09-15)


### Features

* **language:** a model states its topology as relations, the way math-spec 0.0.0-alpha.89 does ([#1646](https://github.com/fluxopt/lpspec/issues/1646)) ([1fd8926](https://github.com/fluxopt/lpspec/commit/1fd892677e4929da7158e8032f64190a1684fe31))

## [0.0.1-alpha.333](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.332...v0.0.1-alpha.333) (2026-09-15)


### Bug Fixes

* a polars without the streaming engine still solves and reads back ([#1644](https://github.com/fluxopt/lpspec/issues/1644)) ([c47ae6c](https://github.com/fluxopt/lpspec/commit/c47ae6c5741a2bfa863bf3642662ee474f301f23))

## [0.0.1-alpha.332](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.331...v0.0.1-alpha.332) (2026-09-15)


### Documentation

* the about pages and the model pages say each fact once, in sentences of one idea ([#1640](https://github.com/fluxopt/lpspec/issues/1640)) ([5d4efa3](https://github.com/fluxopt/lpspec/commit/5d4efa3d16d3029fdaf3d3543a14414240fda637))
* the howtos, tutorials and reference say each fact once and drop the argument for it ([#1639](https://github.com/fluxopt/lpspec/issues/1639)) ([697f29d](https://github.com/fluxopt/lpspec/commit/697f29d58a4f450d292b48d978d788a7bc8ae492))

## [0.0.1-alpha.331](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.330...v0.0.1-alpha.331) (2026-09-15)


### Features

* **language:** a variable, a constraint and a cased expression declare their shape as dims, as a parameter does ([#1634](https://github.com/fluxopt/lpspec/issues/1634)) ([7bdcdeb](https://github.com/fluxopt/lpspec/commit/7bdcdeb85b0702d91cb42eae1b18e9541a322b08))

## [0.0.1-alpha.330](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.329...v0.0.1-alpha.330) (2026-09-15)


### Documentation

* the language closes its own gaps instead of offering an escape hatch ([#1631](https://github.com/fluxopt/lpspec/issues/1631)) ([72928bd](https://github.com/fluxopt/lpspec/commit/72928bd8f8367320b6e3ca9367ea5e9ea76bb899))

## [0.0.1-alpha.329](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.328...v0.0.1-alpha.329) (2026-09-14)


### Bug Fixes

* **compat:** a constant beside a variable term is refused where its data is short ([#1621](https://github.com/fluxopt/lpspec/issues/1621)) ([545ed6d](https://github.com/fluxopt/lpspec/commit/545ed6d1bbc2e3215ca2e3ac58f497a036fe86e6))

## [0.0.1-alpha.328](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.327...v0.0.1-alpha.328) (2026-09-14)


### Documentation

* **bench:** the benchmark page publishes the newest run, and says where that run left a gap ([#1618](https://github.com/fluxopt/lpspec/issues/1618)) ([089f816](https://github.com/fluxopt/lpspec/commit/089f816a926eeedb3e00fe894ac1823ef32704da))

## [0.0.1-alpha.327](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.326...v0.0.1-alpha.327) (2026-09-14)


### Performance

* **engine:** a model is hashed for a re-solve only once a re-solve is reached ([#1608](https://github.com/fluxopt/lpspec/issues/1608)) ([8d27e88](https://github.com/fluxopt/lpspec/commit/8d27e88b791335c20412faab45a7727a1b6a7a6c))

## [0.0.1-alpha.326](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.325...v0.0.1-alpha.326) (2026-09-14)


### Documentation

* **howto:** reading a directory of runs is its own page ([#1611](https://github.com/fluxopt/lpspec/issues/1611)) ([88105f5](https://github.com/fluxopt/lpspec/commit/88105f5fc16f8bf7540f186ffdf3e58437242de3))

## [0.0.1-alpha.325](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.324...v0.0.1-alpha.325) (2026-09-13)


### Features

* **api:** an archive says which data it was solved over and what the solve took ([#1601](https://github.com/fluxopt/lpspec/issues/1601)) ([758e71f](https://github.com/fluxopt/lpspec/commit/758e71f240c41e85114a082d2cf60204a377b104))

## [0.0.1-alpha.324](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.323...v0.0.1-alpha.324) (2026-09-13)


### Features

* **api:** load reads an archive whole, and scan leaves it on disk ([#1603](https://github.com/fluxopt/lpspec/issues/1603)) ([001de65](https://github.com/fluxopt/lpspec/commit/001de65eed85ef40c9d838162cf32cd87dd887cc))


### Performance

* **engine:** a build stops hashing labels it can count ([#1589](https://github.com/fluxopt/lpspec/issues/1589)) ([bc49db5](https://github.com/fluxopt/lpspec/commit/bc49db5418ead02b755d0da733ef68af1666d553))
* **engine:** a magnitude is read off each side of a column instead of being built ([#1592](https://github.com/fluxopt/lpspec/issues/1592)) ([019e22c](https://github.com/fluxopt/lpspec/commit/019e22c3fd9ec68a60702bc14caa3c8abcceb969))
* **engine:** a model reaches HiGHS in one call instead of a batch at a time ([#1591](https://github.com/fluxopt/lpspec/issues/1591)) ([77ccb7d](https://github.com/fluxopt/lpspec/commit/77ccb7ddf159ef78270b9a1e78667c579c66e8a9))


### Documentation

* **api:** the glossary defines load and scan, and the tree spells each name one way ([#1604](https://github.com/fluxopt/lpspec/issues/1604)) ([3d43f6d](https://github.com/fluxopt/lpspec/commit/3d43f6d9e1e1fdb1306c570f0173b837689c2e51))
* the published speed claims match the measurements behind them ([#1586](https://github.com/fluxopt/lpspec/issues/1586)) ([cb98d58](https://github.com/fluxopt/lpspec/commit/cb98d58e2aa147c480100bfceb17ec68d410a418))

## [0.0.1-alpha.323](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.322...v0.0.1-alpha.323) (2026-09-12)


### Features

* **api:** an archived answer says which run it came from and when, in one record file ([#1585](https://github.com/fluxopt/lpspec/issues/1585)) ([650a3ad](https://github.com/fluxopt/lpspec/commit/650a3ad41a8bd8704a9bd2772f0fcf04994390e1))

## [0.0.1-alpha.322](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.321...v0.0.1-alpha.322) (2026-09-12)


### Features

* **api:** an archive records how big the model was and how long it took ([#1598](https://github.com/fluxopt/lpspec/issues/1598)) ([80f6a6e](https://github.com/fluxopt/lpspec/commit/80f6a6efd11fffc50242626bd20db722a401a772))

## [0.0.1-alpha.321](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.320...v0.0.1-alpha.321) (2026-09-12)


### Documentation

* **bench:** the benchmark page is re-taken from the newest run, its claims about the numbers included ([#1587](https://github.com/fluxopt/lpspec/issues/1587)) ([0210774](https://github.com/fluxopt/lpspec/commit/02107748aba0fd82b1b22c298a9bbe187e8ec91d))

## [0.0.1-alpha.320](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.319...v0.0.1-alpha.320) (2026-09-11)


### Features

* **api:** a model, its data and its answer are one file that reads back into polars or a database ([#1579](https://github.com/fluxopt/lpspec/issues/1579)) ([8dc36f9](https://github.com/fluxopt/lpspec/commit/8dc36f94447aa77c38e3984a9e080138266a2017))
* **api:** two names that differ only by case are refused, whichever verb reads the spec ([#1582](https://github.com/fluxopt/lpspec/issues/1582)) ([fc19d17](https://github.com/fluxopt/lpspec/commit/fc19d171d25b380c4fccb75e66078a24bcb8a7f8))

## [0.0.1-alpha.319](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.318...v0.0.1-alpha.319) (2026-09-10)


### Features

* **data:** a lookup maps into a declared dimension rather than owning its values ([#1576](https://github.com/fluxopt/lpspec/issues/1576)) ([005b106](https://github.com/fluxopt/lpspec/commit/005b106d314b7763661b3e09bf3b7ec18966522a))

## [0.0.1-alpha.318](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.317...v0.0.1-alpha.318) (2026-09-10)


### Features

* **strategy:** a rolling horizon may keep a different number of coordinates in each window ([#1572](https://github.com/fluxopt/lpspec/issues/1572)) ([e52bc2d](https://github.com/fluxopt/lpspec/commit/e52bc2d587d145b28581773cf36479a1ee5df890))


### Bug Fixes

* **strategy:** a carry hands on the last row a slice owns rather than a coordinate the caller names ([#1571](https://github.com/fluxopt/lpspec/issues/1571)) ([561d05c](https://github.com/fluxopt/lpspec/commit/561d05cfa289c0c5cacd75fd557d2c8e83c33776))
* **strategy:** a hand-built axis refuses the stitched answer rather than returning the keyed one ([#1568](https://github.com/fluxopt/lpspec/issues/1568)) ([7332207](https://github.com/fluxopt/lpspec/commit/733220790377f8f1ccd7130ae3e7184892e76238))

## [0.0.1-alpha.317](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.316...v0.0.1-alpha.317) (2026-09-10)


### Documentation

* descriptions set as prose under math-spec's escaping, pinned to the head of energy-models/math-spec[#421](https://github.com/fluxopt/lpspec/issues/421) ([#1565](https://github.com/fluxopt/lpspec/issues/1565)) ([5c6b606](https://github.com/fluxopt/lpspec/commit/5c6b60666195e6eb65e6ed4d0181d76d410c7fad))

## [0.0.1-alpha.316](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.315...v0.0.1-alpha.316) (2026-09-10)


### Documentation

* the About pages keep each rule and drop the story behind it ([#1564](https://github.com/fluxopt/lpspec/issues/1564)) ([ec09ace](https://github.com/fluxopt/lpspec/commit/ec09ace8e5e00ec1d09b4a1bd854c2453e20b50f))

## [0.0.1-alpha.315](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.314...v0.0.1-alpha.315) (2026-09-09)


### Documentation

* each shared claim has one page, and two how-tos cover a parallel sweep and a wrong answer ([#1561](https://github.com/fluxopt/lpspec/issues/1561)) ([7dc8553](https://github.com/fluxopt/lpspec/commit/7dc85536ee4ebc35a099a7864427d82a5e10c17e))
* every hand-written page meets the docs-writing skill, and the tree follows the nav ([#1560](https://github.com/fluxopt/lpspec/issues/1560)) ([645a346](https://github.com/fluxopt/lpspec/commit/645a34616d39b8047fb5498937862ed2ad87df31))

## [0.0.1-alpha.314](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.313...v0.0.1-alpha.314) (2026-09-09)


### Documentation

* the site nav is arranged by what a page is for rather than by subject ([#1558](https://github.com/fluxopt/lpspec/issues/1558)) ([6d346c5](https://github.com/fluxopt/lpspec/commit/6d346c59102248f308fad262e087b4c209c5ae98))

## [0.0.1-alpha.313](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.312...v0.0.1-alpha.313) (2026-09-09)


### Refactoring

* **api:** every verb says in its signature what a source and a slice key may be ([#1554](https://github.com/fluxopt/lpspec/issues/1554)) ([3767f7a](https://github.com/fluxopt/lpspec/commit/3767f7afa859d6066c45870af5f2673080d089b4))

## [0.0.1-alpha.312](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.311...v0.0.1-alpha.312) (2026-09-09)


### Features

* **strategy:** a lowered program crosses a process, so a pooled sweep lowers nothing twice ([#1549](https://github.com/fluxopt/lpspec/issues/1549)) ([09cd887](https://github.com/fluxopt/lpspec/commit/09cd88752eab82577f6413a4ce028d382d6146a8))

## [0.0.1-alpha.311](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.310...v0.0.1-alpha.311) (2026-09-09)


### Features

* **strategy:** a sweep spills each slice to disk and resumes where it stopped, and every export and bridge reads every kind ([#1547](https://github.com/fluxopt/lpspec/issues/1547)) ([c5b4828](https://github.com/fluxopt/lpspec/commit/c5b48288db32b921794f78938d0489285fdf7386))

## [0.0.1-alpha.310](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.309...v0.0.1-alpha.310) (2026-09-08)


### Bug Fixes

* **strategy:** a sweep takes every source shape solve takes, and a carry that cannot start is refused before a slice is cut ([#1543](https://github.com/fluxopt/lpspec/issues/1543)) ([b56f8f2](https://github.com/fluxopt/lpspec/commit/b56f8f210df0f381c6a188c21a50db01b37543ea))

## [0.0.1-alpha.309](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.308...v0.0.1-alpha.309) (2026-09-08)


### Refactoring

* **api:** a sweep's slice statuses have one name rather than two ([#1540](https://github.com/fluxopt/lpspec/issues/1540)) ([9dfe10b](https://github.com/fluxopt/lpspec/commit/9dfe10ba86f633d998f2062932e97bdd572f7e53))

## [0.0.1-alpha.308](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.307...v0.0.1-alpha.308) (2026-09-08)


### Features

* **strategy:** a sweep asks the model what a window can bear before it cuts one ([#1514](https://github.com/fluxopt/lpspec/issues/1514)) ([d77cdfb](https://github.com/fluxopt/lpspec/commit/d77cdfb40de132d8d9483d7e547c2fd225a77b2f))

## [0.0.1-alpha.307](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.306...v0.0.1-alpha.307) (2026-09-08)


### Bug Fixes

* **linopy:** a comparison with its terms on the right builds, and so does a window whose width no lookup reaches ([#1537](https://github.com/fluxopt/lpspec/issues/1537)) ([46e29cc](https://github.com/fluxopt/lpspec/commit/46e29cccbf9e0fa44f41bfa5d1ddbd1f6f2e5085))

## [0.0.1-alpha.306](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.305...v0.0.1-alpha.306) (2026-09-08)


### Bug Fixes

* **engine:** a variable declared absence: zero reads as zero under a nonlinear operator ([#1532](https://github.com/fluxopt/lpspec/issues/1532)) ([038af9e](https://github.com/fluxopt/lpspec/commit/038af9e601092a1b5a04602f933fd8208042b34e))

## [0.0.1-alpha.305](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.304...v0.0.1-alpha.305) (2026-09-08)


### Features

* **api:** a named expression is read back at any degree and may read a constraint's dual ([#1529](https://github.com/fluxopt/lpspec/issues/1529)) ([b946506](https://github.com/fluxopt/lpspec/commit/b9465061be5d7003c32607f184ad5906a57c91eb))

## [0.0.1-alpha.304](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.303...v0.0.1-alpha.304) (2026-09-03)


### Features

* **api:** a badly scaled bound names the declaration that holds it ([#1525](https://github.com/fluxopt/lpspec/issues/1525)) ([b8546fa](https://github.com/fluxopt/lpspec/commit/b8546fa8a3eaec3390d22671fcdedd63e17294b4))

## [0.0.1-alpha.303](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.302...v0.0.1-alpha.303) (2026-09-02)


### Bug Fixes

* **data:** a hole on a constant side is refused even under a sum ([#1523](https://github.com/fluxopt/lpspec/issues/1523)) ([caa3627](https://github.com/fluxopt/lpspec/commit/caa36279856b980758b53261c48d6235c4256593))

## [0.0.1-alpha.302](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.301...v0.0.1-alpha.302) (2026-09-02)


### Documentation

* **architecture:** the plan-node census covers expression nodes, not only predicates ([#1505](https://github.com/fluxopt/lpspec/issues/1505)) ([c4a5b94](https://github.com/fluxopt/lpspec/commit/c4a5b9436d35f5bdb27cf5048ec0b0b25f0cf2cf))

## [0.0.1-alpha.301](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.300...v0.0.1-alpha.301) (2026-09-02)


### Bug Fixes

* **data:** a convex piecewise curve is checked without xarray, so a bare install attaches it ([#1507](https://github.com/fluxopt/lpspec/issues/1507)) ([8b50241](https://github.com/fluxopt/lpspec/commit/8b50241bba809fadca33704253f99bd4bc33b83d))


### Refactoring

* **data:** a caller's data is read and checked once, at one door, for both lanes ([#1506](https://github.com/fluxopt/lpspec/issues/1506)) ([787f9c7](https://github.com/fluxopt/lpspec/commit/787f9c70696bda25f1bbc7eec5f80ae46fec4a2d))

## [0.0.1-alpha.300](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.299...v0.0.1-alpha.300) (2026-09-02)


### Performance

* **deps:** every build spends less time reading and lowering the model file ([#1503](https://github.com/fluxopt/lpspec/issues/1503)) ([48d1a7a](https://github.com/fluxopt/lpspec/commit/48d1a7afde4eca4733d1fcfa2ebbfaeba0ee9a4a))

## [0.0.1-alpha.299](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.298...v0.0.1-alpha.299) (2026-09-01)


### Performance

* **engine:** a model builds faster for each declaration it carries ([#1500](https://github.com/fluxopt/lpspec/issues/1500)) ([9126391](https://github.com/fluxopt/lpspec/commit/912639191dd01a6bd6a8d99f85a6a26ff8ed8cf0))

## [0.0.1-alpha.298](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.297...v0.0.1-alpha.298) (2026-09-01)


### Documentation

* **bench:** the published numbers come from a run of main, and the reproduction installs that commit ([#1497](https://github.com/fluxopt/lpspec/issues/1497)) ([99a7633](https://github.com/fluxopt/lpspec/commit/99a7633388161d1e176d5017f7c7552b30457377))

## [0.0.1-alpha.297](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.296...v0.0.1-alpha.297) (2026-09-01)


### Features

* **bench:** a cell that exhausts the machine is recorded, not lost ([#1480](https://github.com/fluxopt/lpspec/issues/1480)) ([8a332e0](https://github.com/fluxopt/lpspec/commit/8a332e056ae01090750fcaaf8d9e77a1a5ebb557))

## [0.0.1-alpha.296](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.295...v0.0.1-alpha.296) (2026-09-01)


### Documentation

* **bench:** the published numbers come from one run of a box the project can rebuild ([#1488](https://github.com/fluxopt/lpspec/issues/1488)) ([bb51e4d](https://github.com/fluxopt/lpspec/commit/bb51e4de8895129f3cc473fd02e34c07accd8af1))

## [0.0.1-alpha.295](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.294...v0.0.1-alpha.295) (2026-09-01)


### Refactoring

* **api:** data attaches to a spec, so bound means only a variable's limit ([#1479](https://github.com/fluxopt/lpspec/issues/1479)) ([9583701](https://github.com/fluxopt/lpspec/commit/9583701a589d236f6033de787461ec88047fe9c2))

## [0.0.1-alpha.294](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.293...v0.0.1-alpha.294) (2026-09-01)


### Refactoring

* **api:** the input is a spec, a Model is that spec with your data ([#1474](https://github.com/fluxopt/lpspec/issues/1474)) ([0fdf674](https://github.com/fluxopt/lpspec/commit/0fdf674530ec651f972866b04329a884153d592b))

## [0.0.1-alpha.293](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.292...v0.0.1-alpha.293) (2026-08-31)


### Bug Fixes

* **differential:** a rung's projection names its quantities in the file's order, so a tree nobody changed stops going red ([#1469](https://github.com/fluxopt/lpspec/issues/1469)) ([1603efa](https://github.com/fluxopt/lpspec/commit/1603efa5502c6b3af0165d4d395c809fea79d7bb))

## [0.0.1-alpha.292](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.291...v0.0.1-alpha.292) (2026-08-31)


### Features

* **language:** a quantity defined by cases may mix a constant, a parameter and a variable across its regions ([#1464](https://github.com/fluxopt/lpspec/issues/1464)) ([482e9ec](https://github.com/fluxopt/lpspec/commit/482e9ec1cf92987e2195d0d7ee2e693a8a97df2c))

## [0.0.1-alpha.291](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.290...v0.0.1-alpha.291) (2026-08-31)


### Documentation

* **bench:** the box's own housekeeping is turned off before it is snapshotted ([#1461](https://github.com/fluxopt/lpspec/issues/1461)) ([6e011f8](https://github.com/fluxopt/lpspec/commit/6e011f8f3f43e4d790aa38cbf649865d53ac7bc4))

## [0.0.1-alpha.290](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.289...v0.0.1-alpha.290) (2026-08-31)


### Bug Fixes

* **bench:** a case hands back what it measured before the next one can take the box ([#1455](https://github.com/fluxopt/lpspec/issues/1455)) ([365ced0](https://github.com/fluxopt/lpspec/commit/365ced0a34ff27200fe70bca276b58ed1d39ae27))

## [0.0.1-alpha.289](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.288...v0.0.1-alpha.289) (2026-08-29)


### Bug Fixes

* **bench:** the kill reaches the spawned child that is holding the model ([#1453](https://github.com/fluxopt/lpspec/issues/1453)) ([e91c844](https://github.com/fluxopt/lpspec/commit/e91c8445a84d21b0a34d6cec51d1f4e20d0d8a70))

## [0.0.1-alpha.288](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.287...v0.0.1-alpha.288) (2026-08-28)


### Bug Fixes

* **bench:** a cell too big for the box takes the case down, not the runner ([#1451](https://github.com/fluxopt/lpspec/issues/1451)) ([58e073f](https://github.com/fluxopt/lpspec/commit/58e073fa44b13376ddf35191b4f652e3c8391171))

## [0.0.1-alpha.287](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.286...v0.0.1-alpha.287) (2026-08-28)


### Bug Fixes

* **bench:** a cell the memory budget stopped says so, instead of naming the seconds ([#1447](https://github.com/fluxopt/lpspec/issues/1447)) ([e16fc0b](https://github.com/fluxopt/lpspec/commit/e16fc0b31e243c4cce197f3c0800ae861b04f065))
* **bench:** the scaling chart reads only the ceilings of the ladder it draws ([#1446](https://github.com/fluxopt/lpspec/issues/1446)) ([5991e91](https://github.com/fluxopt/lpspec/commit/5991e91cffbf08aa30d0049db82e876af2cd4935))

## [0.0.1-alpha.286](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.285...v0.0.1-alpha.286) (2026-08-28)


### Features

* **api:** check hands back the plan the other verbs take, so a model is lowered once per run ([#1443](https://github.com/fluxopt/lpspec/issues/1443)) ([371bade](https://github.com/fluxopt/lpspec/commit/371bade60d8d3fe89484f8e700c2e02a4cb5c4c3))


### Bug Fixes

* **differential:** a dual the file writes negated is held to the negative of PyPSA's rather than excused ([#1444](https://github.com/fluxopt/lpspec/issues/1444)) ([8f68665](https://github.com/fluxopt/lpspec/commit/8f686657f6fdfeba101bf530cf5a5aa2f0b7304e))

## [0.0.1-alpha.285](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.284...v0.0.1-alpha.285) (2026-08-28)


### Documentation

* an infeasible model whose file still reads right is diagnosed by the row it built ([#1440](https://github.com/fluxopt/lpspec/issues/1440)) ([82f4485](https://github.com/fluxopt/lpspec/commit/82f4485de62ea0927c0c9dff3e58e6e30e4c35ec))
* **examples:** the PyPSA ladder is the fifteen rungs, and the gallery is examples rather than a second ladder ([#1441](https://github.com/fluxopt/lpspec/issues/1441)) ([ff525e4](https://github.com/fluxopt/lpspec/commit/ff525e49448bc8fcb9d2c4f8b4189d53e6e8e42c))

## [0.0.1-alpha.284](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.283...v0.0.1-alpha.284) (2026-08-28)


### Bug Fixes

* **data:** a negative window width is refused at load, and a spelled-out `AND True` builds ([#1432](https://github.com/fluxopt/lpspec/issues/1432)) ([aeab021](https://github.com/fluxopt/lpspec/commit/aeab0213c423b2372c4faab27f358ed753c66955))

## [0.0.1-alpha.283](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.282...v0.0.1-alpha.283) (2026-08-28)


### Documentation

* every type and module the documentation names is one that exists ([#1427](https://github.com/fluxopt/lpspec/issues/1427)) ([91c9b7c](https://github.com/fluxopt/lpspec/commit/91c9b7cac205c7a4259affe25e7586438f3d3699))

## [0.0.1-alpha.282](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.281...v0.0.1-alpha.282) (2026-08-28)


### Performance

* **bench:** the rebuild pass is measured once, not once per sink ([#1425](https://github.com/fluxopt/lpspec/issues/1425)) ([2b876f7](https://github.com/fluxopt/lpspec/commit/2b876f78fb6a8c65e2b80f1b22667482158ffba8))

## [0.0.1-alpha.281](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.280...v0.0.1-alpha.281) (2026-08-28)


### Performance

* **bench:** the memory pass builds the model once instead of four times ([#1420](https://github.com/fluxopt/lpspec/issues/1420)) ([3e9c67c](https://github.com/fluxopt/lpspec/commit/3e9c67c7ec64a3ddbb7a03865909a688b360dd94))

## [0.0.1-alpha.280](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.279...v0.0.1-alpha.280) (2026-08-28)


### Bug Fixes

* **bench:** a case does not carry the last one's memory into the next ([#1417](https://github.com/fluxopt/lpspec/issues/1417)) ([6b5c103](https://github.com/fluxopt/lpspec/commit/6b5c1037f91a1439a0abb117b35c63e4461bcf3c))

## [0.0.1-alpha.279](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.278...v0.0.1-alpha.279) (2026-08-28)


### Documentation

* **engine:** the program's own rules read without knowing what builds it ([#1413](https://github.com/fluxopt/lpspec/issues/1413)) ([6cd559b](https://github.com/fluxopt/lpspec/commit/6cd559bf421906def811339352bde9398c8a2f16))

## [0.0.1-alpha.278](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.277...v0.0.1-alpha.278) (2026-08-28)


### Refactoring

* **engine:** a program states what it means, so the direction and the edge cannot be inferred wrongly ([#1411](https://github.com/fluxopt/lpspec/issues/1411)) ([bf67cc4](https://github.com/fluxopt/lpspec/commit/bf67cc4ecc8cf536003a24e024f9040c828d5583))


### Documentation

* **agents:** the cheap gates run here, and CI is reported rather than watched ([#1410](https://github.com/fluxopt/lpspec/issues/1410)) ([6c1d91c](https://github.com/fluxopt/lpspec/commit/6c1d91ca7b28b0923b233fd8cf4b7765ed324c45))

## [0.0.1-alpha.277](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.276...v0.0.1-alpha.277) (2026-08-28)


### Performance

* **linopy:** a trailing window merges its lags once instead of one at a time ([#1341](https://github.com/fluxopt/lpspec/issues/1341)) ([f225422](https://github.com/fluxopt/lpspec/commit/f225422510b9be355f7a1f22dbd2133b339d6dcf))

## [0.0.1-alpha.276](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.275...v0.0.1-alpha.276) (2026-08-28)


### Features

* **engine:** a malformed plan is refused at the boundary, in plan vocabulary, before any query compiles ([#1394](https://github.com/fluxopt/lpspec/issues/1394)) ([e435d63](https://github.com/fluxopt/lpspec/commit/e435d638cd50fb0e67768ce831ba17ddbb19ef7b))
* **engine:** the plan says what type a dimension's labels are, as it already says so for a parameter's values ([#1391](https://github.com/fluxopt/lpspec/issues/1391)) ([05c07f5](https://github.com/fluxopt/lpspec/commit/05c07f5f3d1704d953900d25d6db82fac3ed9a96))


### Refactoring

* **compat:** the linopy lane builds from the logical plan, so one lowering answers for both lanes ([#1385](https://github.com/fluxopt/lpspec/issues/1385)) ([995fa34](https://github.com/fluxopt/lpspec/commit/995fa343f4f5a99d12ca6331e93a9e14fea6e04e))
* **compat:** the linopy loader reads its declarations off the plan, so the plan and the data are the whole seam ([#1388](https://github.com/fluxopt/lpspec/issues/1388)) ([a20a042](https://github.com/fluxopt/lpspec/commit/a20a042d4e57ad84945f9baa6075509adc93be51))
* **engine:** a shape operator declares its own fan-in, and the absence pass reads it ([#1395](https://github.com/fluxopt/lpspec/issues/1395)) ([beaace8](https://github.com/fluxopt/lpspec/commit/beaace8cbafe56bb387ac5b664b49b21659c4946))
* **engine:** one walk over a plan's expressions, and every question about one is a filter of it ([#1398](https://github.com/fluxopt/lpspec/issues/1398)) ([5c7f2a0](https://github.com/fluxopt/lpspec/commit/5c7f2a0a52c5497dcab5c152a41f6134ee06c862))
* **engine:** the expression nodes are one closed type, so a walk that forgets one no longer compiles ([#1397](https://github.com/fluxopt/lpspec/issues/1397)) ([e8be601](https://github.com/fluxopt/lpspec/commit/e8be6011b3d6bb73ed4af5eb919d61ce48a34ac4))
* **engine:** the plan answers which dimension a lookup labels, instead of five callers walking for it ([#1387](https://github.com/fluxopt/lpspec/issues/1387)) ([8d4cd28](https://github.com/fluxopt/lpspec/commit/8d4cd28deb660112c7ac7ed0edd2e4967ac8d0c1))
* **engine:** the plan decides what a file may say, and the engine only builds it ([#1400](https://github.com/fluxopt/lpspec/issues/1400)) ([a5582fe](https://github.com/fluxopt/lpspec/commit/a5582fe3e9e63f108f226aae436d8dbb6b750bee))
* **engine:** two plan fields say what rule holds them, rather than what they saved a lookup on ([#1402](https://github.com/fluxopt/lpspec/issues/1402)) ([79c8961](https://github.com/fluxopt/lpspec/commit/79c8961581c3fe306955f1d7a30d3081ff4cefd6))


### Documentation

* the plan's thirteen nodes are one table, each with the query and the linopy call it becomes ([#1396](https://github.com/fluxopt/lpspec/issues/1396)) ([0e0ae80](https://github.com/fluxopt/lpspec/commit/0e0ae80036d3a44ad31a2ebed36a5eca517993c0))

## [0.0.1-alpha.275](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.274...v0.0.1-alpha.275) (2026-08-27)


### Features

* **differential:** rung 14 binds — a network with scenarios flattens to the file's dimensions, and a name PyPSA spells per scenario is held per label ([#1376](https://github.com/fluxopt/lpspec/issues/1376)) ([ffd53ce](https://github.com/fluxopt/lpspec/commit/ffd53cea01257778fc9434c3597cefc8e7fe887c))
* **differential:** rung 15 binds — a multi-period network flattens to one snapshot axis, and PyPSA's network is solved under the semantics PyPSA speaks ([#1379](https://github.com/fluxopt/lpspec/issues/1379)) ([4fb41cc](https://github.com/fluxopt/lpspec/commit/4fb41cc0a5c73d91eb5f5aa8942c7836d7c2b8c1))
* **differential:** rungs 12 and 13 bind — the tightening mask and the tangent loss fan in prep, and a block over a segment dimension held to PyPSA's row per segment ([#1373](https://github.com/fluxopt/lpspec/issues/1373)) ([90d7df2](https://github.com/fluxopt/lpspec/commit/90d7df28963a99a693b2adfcb2ab80b0a834e8a3))

## [0.0.1-alpha.274](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.273...v0.0.1-alpha.274) (2026-08-27)


### Refactoring

* **linopy:** a grouped sum's empty group is filled by the caller rather than by a hand-written storage row ([#1375](https://github.com/fluxopt/lpspec/issues/1375)) ([bb50cb0](https://github.com/fluxopt/lpspec/commit/bb50cb021463a35f4b4dc12b492f3b49b101f486))

## [0.0.1-alpha.273](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.272...v0.0.1-alpha.273) (2026-08-27)


### Refactoring

* **differential:** snapshots join PyPSA's timestamps natively, the position map is gone ([#1369](https://github.com/fluxopt/lpspec/issues/1369)) ([00c1da1](https://github.com/fluxopt/lpspec/commit/00c1da1d593fc3e6bd603028c7a9cadc2c7b857b))

## [0.0.1-alpha.272](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.271...v0.0.1-alpha.272) (2026-08-27)


### Refactoring

* **linopy:** the empty sum is built through linopy's public constructor rather than its storage layout ([#1365](https://github.com/fluxopt/lpspec/issues/1365)) ([88de413](https://github.com/fluxopt/lpspec/commit/88de4133385dfd4e16ab2bc94c11423e5f5bdb14))

## [0.0.1-alpha.271](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.270...v0.0.1-alpha.271) (2026-08-27)


### Features

* **differential:** the commitment rung compares model for model, with its deliberate differences recorded ([#1360](https://github.com/fluxopt/lpspec/issues/1360)) ([eeeb149](https://github.com/fluxopt/lpspec/commit/eeeb149d01399490eeda23b2395a85b21a6a883a))
* **differential:** the structure column holds the solver model's rows, columns and nonzeros equal on both sides ([#1362](https://github.com/fluxopt/lpspec/issues/1362)) ([24a73b0](https://github.com/fluxopt/lpspec/commit/24a73b048f2e5ff89f9c4b3a86ec997467bd916d))


### Bug Fixes

* **linopy:** a sum whose term carries an empty dimension is the empty sum rather than an error ([#1359](https://github.com/fluxopt/lpspec/issues/1359)) ([a7a3040](https://github.com/fluxopt/lpspec/commit/a7a3040f502896d833df5c1c256189573708a220))

## [0.0.1-alpha.270](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.269...v0.0.1-alpha.270) (2026-08-27)


### Features

* **differential:** each rung has its own model, the file projected onto what the rung builds and held to its objective ([#1338](https://github.com/fluxopt/lpspec/issues/1338)) ([0400235](https://github.com/fluxopt/lpspec/commit/04002351073c98d9a93f34131b7a1d39f9a31f40))
* **differential:** the ladder holds each PyPSA name to one block with an equal count rather than a sum over its splits ([#1351](https://github.com/fluxopt/lpspec/issues/1351)) ([46050c3](https://github.com/fluxopt/lpspec/commit/46050c3fdb063044e33d2dbf4edfa963ebcd99de))
* **differential:** the ladder holds every constraint's dual to PyPSA's, and a difference needs a recorded reason ([#1352](https://github.com/fluxopt/lpspec/issues/1352)) ([dc37007](https://github.com/fluxopt/lpspec/commit/dc3700738014eec47cbe1feb0bbb5eeba4701207))


### Documentation

* **differential:** the PyPSA ladder page shows the tables each rung binds, committed and held to the corpus ([#1336](https://github.com/fluxopt/lpspec/issues/1336)) ([cbd00a3](https://github.com/fluxopt/lpspec/commit/cbd00a3cca40b3bdf9d12c997f8a5d404151087a))
* **differential:** the PyPSA ladder, one page per rung, held to PyPSA by objective, structure, duals and the linopy lane ([#1348](https://github.com/fluxopt/lpspec/issues/1348)) ([a72cc75](https://github.com/fluxopt/lpspec/commit/a72cc755d0b8a8504ca38706cb88ac9011232d85))

## [0.0.1-alpha.269](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.268...v0.0.1-alpha.269) (2026-08-26)


### Bug Fixes

* **linopy:** a parameter with no dimensions keeps the type it declares ([#1340](https://github.com/fluxopt/lpspec/issues/1340)) ([89f57f1](https://github.com/fluxopt/lpspec/commit/89f57f1c9c5d822f79a7a62b5d62e02d5b6f54b7))

## [0.0.1-alpha.268](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.267...v0.0.1-alpha.268) (2026-08-26)


### Bug Fixes

* **differential:** a stamp moves when its claim changes, not when the solver returns different bits ([#1333](https://github.com/fluxopt/lpspec/issues/1333)) ([a890c11](https://github.com/fluxopt/lpspec/commit/a890c11ad94bdab595c79556b1d04a951719aff7))

## [0.0.1-alpha.267](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.266...v0.0.1-alpha.267) (2026-08-26)


### Refactoring

* **differential:** the pypsa corpus lives beside its runner, and the certificate is a committed file this tree must reproduce ([#1328](https://github.com/fluxopt/lpspec/issues/1328)) ([1e0aa53](https://github.com/fluxopt/lpspec/commit/1e0aa53477c153492539f6d77c45777e39ec73a6))

## [0.0.1-alpha.266](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.265...v0.0.1-alpha.266) (2026-08-26)


### Bug Fixes

* **data:** a piecewise method's exact curvature is the language's answer, not this package's copy ([#1323](https://github.com/fluxopt/lpspec/issues/1323)) ([99994a5](https://github.com/fluxopt/lpspec/commit/99994a5ca0d3c40c4203c9d1a3ae7311ecf6c001))
* **differential:** the KVL basis is PyPSA's own cycle matrix, and the objective is compared net of its constant ([#1322](https://github.com/fluxopt/lpspec/issues/1322)) ([9b2aa70](https://github.com/fluxopt/lpspec/commit/9b2aa70066ab71745afd1f3cc1ba80df106efea8))

## [0.0.1-alpha.265](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.264...v0.0.1-alpha.265) (2026-08-26)


### Features

* **differential:** the pypsa runner holds the nodal prices to PyPSA's ([#1318](https://github.com/fluxopt/lpspec/issues/1318)) ([b45bf15](https://github.com/fluxopt/lpspec/commit/b45bf15b33077837ae3aca459af91c2eb2179c4b))
* the pypsa differential suite lives here, and the corpus keeps only documents ([#1310](https://github.com/fluxopt/lpspec/issues/1310)) ([c1db712](https://github.com/fluxopt/lpspec/commit/c1db712ed6a64da7d07cc80ae802bbe74bb9fb71))


### Bug Fixes

* **bench:** the scheduled ladder measures the selection it was given, not pytest's defaults ([#1312](https://github.com/fluxopt/lpspec/issues/1312)) ([3150844](https://github.com/fluxopt/lpspec/commit/315084487d1fd305f62bc59a93764a9d64e6eb80))
* **differential:** the pypsa runner certifies the rung folders, and refuses an empty corpus ([#1313](https://github.com/fluxopt/lpspec/issues/1313)) ([2092629](https://github.com/fluxopt/lpspec/commit/209262931b85b52553a07bf20a2897d6badd2137))

## [0.0.1-alpha.264](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.263...v0.0.1-alpha.264) (2026-08-26)


### Bug Fixes

* **linopy:** a sum_back window whose every width is zero builds no row instead of crashing ([#1307](https://github.com/fluxopt/lpspec/issues/1307)) ([5525e7c](https://github.com/fluxopt/lpspec/commit/5525e7c91156c73bc50b951237f23a94ecff9b88))

## [0.0.1-alpha.263](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.262...v0.0.1-alpha.263) (2026-08-26)


### Refactoring

* **api:** check tells a lane from a sink in one place ([#1295](https://github.com/fluxopt/lpspec/issues/1295)) ([6392a09](https://github.com/fluxopt/lpspec/commit/6392a09a53220e7a317968d7c33fae9bfcf17677))


### Documentation

* **agents:** a break is described in the PR body, and a verified claim names its gate ([#1301](https://github.com/fluxopt/lpspec/issues/1301)) ([e385dcc](https://github.com/fluxopt/lpspec/commit/e385dccc7ea454c43f2637e1b9d9f3134c0b49b1))

## [0.0.1-alpha.262](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.261...v0.0.1-alpha.262) (2026-08-26)


### Bug Fixes

* **solver:** a gurobi solver releases its model and licence however it is dropped ([#1292](https://github.com/fluxopt/lpspec/issues/1292)) ([f33c3e9](https://github.com/fluxopt/lpspec/commit/f33c3e9fd4f1b1c63a303f6242648331805882af))


### Refactoring

* **solver:** every solver states what it holds and releases it the same way ([#1297](https://github.com/fluxopt/lpspec/issues/1297)) ([ab848b7](https://github.com/fluxopt/lpspec/commit/ab848b7db471e5b66ed1af0f135be0deb3e12b4f))

## [0.0.1-alpha.261](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.260...v0.0.1-alpha.261) (2026-08-26)


### Bug Fixes

* **bench:** a time budget decides one ladder without deciding the other ([#1285](https://github.com/fluxopt/lpspec/issues/1285)) ([84850d9](https://github.com/fluxopt/lpspec/commit/84850d905239e475fb4fceb116ab88eac0b83be9))


### Documentation

* **api:** a docstring citing a hard rule says where the rules are ([#1296](https://github.com/fluxopt/lpspec/issues/1296)) ([2560343](https://github.com/fluxopt/lpspec/commit/2560343c201ac20f59d61761a9729b553d7f7aa2))

## [0.0.1-alpha.260](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.259...v0.0.1-alpha.260) (2026-08-26)


### Refactoring

* **api:** check lowers the expressions of the model it lowers against ([#1290](https://github.com/fluxopt/lpspec/issues/1290)) ([efc8e01](https://github.com/fluxopt/lpspec/commit/efc8e01da5b4c744961129579387292a838bf432))

## [0.0.1-alpha.259](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.258...v0.0.1-alpha.259) (2026-08-26)


### Refactoring

* the package and its test suite state each shared fact once ([#1284](https://github.com/fluxopt/lpspec/issues/1284)) ([2065436](https://github.com/fluxopt/lpspec/commit/20654367b4e4c6dfa770466bb5beaafc59cdb9f8))

## [0.0.1-alpha.258](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.257...v0.0.1-alpha.258) (2026-08-25)


### Features

* **bench:** a table shows the arms the run measured, not the two it was written for ([#1271](https://github.com/fluxopt/lpspec/issues/1271)) ([29b84a3](https://github.com/fluxopt/lpspec/commit/29b84a30f9e5c426b1b49a21c4aac4ab32a27432))

## [0.0.1-alpha.257](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.256...v0.0.1-alpha.257) (2026-08-25)


### Documentation

* **agents:** a PR title is a changelog line ([#1274](https://github.com/fluxopt/lpspec/issues/1274)) ([1890651](https://github.com/fluxopt/lpspec/commit/1890651e8a82058bf9acc8c22b9811d4a560e3ab))

## [0.0.1-alpha.256](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.255...v0.0.1-alpha.256) (2026-08-25)


### Features

* **bench:** every measurement is a row, so a plot is a filter not a patch ([#1265](https://github.com/fluxopt/lpspec/issues/1265)) ([99de9ed](https://github.com/fluxopt/lpspec/commit/99de9ed708721e4214885a4aa5def5d67ac4c6a9))


### Bug Fixes

* **bench:** a rename in src cannot retire a bench entry point unnoticed ([#1267](https://github.com/fluxopt/lpspec/issues/1267)) ([7a72612](https://github.com/fluxopt/lpspec/commit/7a72612a879ede2992936fc7f4967f343f264c4f))


### Refactoring

* **bench:** an arm is a module, not a branch in three verbs ([#1264](https://github.com/fluxopt/lpspec/issues/1264)) ([e62cc33](https://github.com/fluxopt/lpspec/commit/e62cc332831f933931c3a5b864286b60699af420))
* **bench:** the eager lane is not one of the libraries we compare against ([#1268](https://github.com/fluxopt/lpspec/issues/1268)) ([37ca9c3](https://github.com/fluxopt/lpspec/commit/37ca9c3436483c234507b1be989a00f794b6b5b9))

## [0.0.1-alpha.255](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.254...v0.0.1-alpha.255) (2026-08-25)


### Features

* **api:** the types the verbs return can be named ([#1262](https://github.com/fluxopt/lpspec/issues/1262)) ([1b39969](https://github.com/fluxopt/lpspec/commit/1b399695eb54098b3c17db49b636acafdec3862a))

## [0.0.1-alpha.254](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.253...v0.0.1-alpha.254) (2026-08-25)


### Features

* **operators:** a window may stop at each group's edge ([#1260](https://github.com/fluxopt/lpspec/issues/1260)) ([027345c](https://github.com/fluxopt/lpspec/commit/027345cd8a5ea41248f370e1415c907953b3224b))

## [0.0.1-alpha.253](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.252...v0.0.1-alpha.253) (2026-08-25)


### Refactoring

* **architecture:** each layer holds what its docstring says it holds ([#1257](https://github.com/fluxopt/lpspec/issues/1257)) ([d3fa90a](https://github.com/fluxopt/lpspec/commit/d3fa90a09c63c7749103608a497e04b3a54d6d90))

## [0.0.1-alpha.252](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.251...v0.0.1-alpha.252) (2026-08-24)


### Bug Fixes

* **diagnostics:** a build that raises says which half of the report survived ([#1252](https://github.com/fluxopt/lpspec/issues/1252)) ([216cd2d](https://github.com/fluxopt/lpspec/commit/216cd2df206cf67c9031cbde3e53bca6b7e649f6))


### Documentation

* **api:** diagnostics() says what it answers with when a build raised ([#1253](https://github.com/fluxopt/lpspec/issues/1253)) ([b4d91b9](https://github.com/fluxopt/lpspec/commit/b4d91b9a561e3d49d1bbf065ff40d0d768243abf))

## [0.0.1-alpha.251](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.250...v0.0.1-alpha.251) (2026-08-24)


### Refactoring

* **data:** a curve's numbers are checked somewhere of their own ([#1236](https://github.com/fluxopt/lpspec/issues/1236)) ([d69dad1](https://github.com/fluxopt/lpspec/commit/d69dad13fce8758b5bdf97826ec28995602f1eba))
* **engine:** a build is a value, so the engine holds one field not twenty-six ([#1245](https://github.com/fluxopt/lpspec/issues/1245)) ([3daf67d](https://github.com/fluxopt/lpspec/commit/3daf67df4ce385ba233724025365dcad1cde3cc8))
* **engine:** a where-mask is its own vocabulary, not a method ([#1240](https://github.com/fluxopt/lpspec/issues/1240)) ([fde62ee](https://github.com/fluxopt/lpspec/commit/fde62ee52f46ede3b0c987bf7702598227e5cc9c))
* **engine:** walking a dimension's own order is one subject, not nine methods ([#1244](https://github.com/fluxopt/lpspec/issues/1244)) ([cf229e6](https://github.com/fluxopt/lpspec/commit/cf229e636b78fed43c50339b1246a366c222323f))
* **engine:** what a term is stops being the compiler's business ([#1234](https://github.com/fluxopt/lpspec/issues/1234)) ([49b655f](https://github.com/fluxopt/lpspec/commit/49b655f7aec461f5b6c71afccbcbf9a98ecf552d))
* **linopy:** the builder builds; it stops also being the operator table ([#1235](https://github.com/fluxopt/lpspec/issues/1235)) ([dfb6283](https://github.com/fluxopt/lpspec/commit/dfb6283cec5a4a66ae571613d43ff324bdb1ee47))
* **lowering:** a walk stops handing itself the same three arguments ([#1249](https://github.com/fluxopt/lpspec/issues/1249)) ([5db1df9](https://github.com/fluxopt/lpspec/commit/5db1df9bbbc2c0281fcd5e1c9d21b9f7986494d9))
* **lowering:** each built-in lowers in a function with its own name ([#1247](https://github.com/fluxopt/lpspec/issues/1247)) ([7c009d1](https://github.com/fluxopt/lpspec/commit/7c009d1189148dbd8d6251efd060031687af9102))


### Documentation

* **refactor:** the modules that moved stop being described where they were ([#1250](https://github.com/fluxopt/lpspec/issues/1250)) ([e3cc1e0](https://github.com/fluxopt/lpspec/commit/e3cc1e0a0ba7a8dfa7539c31cfe0c37ca0eed8e8))

## [0.0.1-alpha.250](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.249...v0.0.1-alpha.250) (2026-08-24)


### Refactoring

* **tests:** the whole repository imports the language by the surface it pins ([#1237](https://github.com/fluxopt/lpspec/issues/1237)) ([e6e1d72](https://github.com/fluxopt/lpspec/commit/e6e1d72d37cc33932c18a4a5c5f1f18237458825))

## [0.0.1-alpha.249](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.248...v0.0.1-alpha.249) (2026-08-23)


### Features

* **language:** a boundary clause names a position, not the label at one ([#1230](https://github.com/fluxopt/lpspec/issues/1230)) ([3e018da](https://github.com/fluxopt/lpspec/commit/3e018da9c85586dd9061ee91a794b7d74f213f43))

## [0.0.1-alpha.248](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.247...v0.0.1-alpha.248) (2026-08-21)


### Documentation

* **architecture:** the map describes the tree the cut left behind ([#1227](https://github.com/fluxopt/lpspec/issues/1227)) ([a987491](https://github.com/fluxopt/lpspec/commit/a9874915b10edddc35a52a00ef6b1fb0746ac934))

## [0.0.1-alpha.247](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.246...v0.0.1-alpha.247) (2026-08-21)


### Documentation

* the site's sections are the three subjects this package still has ([#1226](https://github.com/fluxopt/lpspec/issues/1226)) ([2b04cb3](https://github.com/fluxopt/lpspec/commit/2b04cb321426f3f753632bd37cf780d2fcaf96b7))

## [0.0.1-alpha.246](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.245...v0.0.1-alpha.246) (2026-08-21)


### Refactoring

* **language:** the language is a dependency, not a directory ([#1223](https://github.com/fluxopt/lpspec/issues/1223)) ([ffbbee6](https://github.com/fluxopt/lpspec/commit/ffbbee60802538e89359875885d5d61d331d7461))

## [0.0.1-alpha.245](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.244...v0.0.1-alpha.245) (2026-08-21)


### Bug Fixes

* **tests:** the renderer's tests read only what travels with it ([#1217](https://github.com/fluxopt/lpspec/issues/1217)) ([c486fff](https://github.com/fluxopt/lpspec/commit/c486fffb5067aab4cdb5751a8d63a216f8a4f712))

## [0.0.1-alpha.244](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.243...v0.0.1-alpha.244) (2026-08-21)


### Refactoring

* **tests:** the sweep's rules are readable one at a time ([#1213](https://github.com/fluxopt/lpspec/issues/1213)) ([fd3df65](https://github.com/fluxopt/lpspec/commit/fd3df65cecce4911641b92929f52c7673ca40ac0))

## [0.0.1-alpha.243](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.242...v0.0.1-alpha.243) (2026-08-21)


### Features

* **tools:** a mutation table is produced rather than claimed ([#1199](https://github.com/fluxopt/lpspec/issues/1199)) ([8a50254](https://github.com/fluxopt/lpspec/commit/8a502541e118a4d84640e621446c66378861afd7))

## [0.0.1-alpha.242](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.241...v0.0.1-alpha.242) (2026-08-21)


### Refactoring

* **tests:** a test is named for its construct, and the residue file is split ([#1209](https://github.com/fluxopt/lpspec/issues/1209)) ([935bfd7](https://github.com/fluxopt/lpspec/commit/935bfd7ca5ebe3f3e911125ceb8a682866835113))
* **tests:** the relational tests say what each group shares ([#1211](https://github.com/fluxopt/lpspec/issues/1211)) ([c389f6c](https://github.com/fluxopt/lpspec/commit/c389f6c7b4a60e6711673b720dac9bebb43b7ce6))


### Documentation

* **agents:** coverage is a local instrument, and the tree says so ([#1207](https://github.com/fluxopt/lpspec/issues/1207)) ([a1dc544](https://github.com/fluxopt/lpspec/commit/a1dc544ba0573763835ab11ec927ed362f359ce7))

## [0.0.1-alpha.241](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.240...v0.0.1-alpha.241) (2026-08-21)


### Documentation

* **language:** a consumer of the AST has a page to read the contract in ([#1205](https://github.com/fluxopt/lpspec/issues/1205)) ([4e05dc6](https://github.com/fluxopt/lpspec/commit/4e05dc60fd8ca89ff96a836e119994799a098752))

## [0.0.1-alpha.240](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.239...v0.0.1-alpha.240) (2026-08-21)


### Bug Fixes

* **linopy:** reading a named expression over a masked curve derives its mask ([#1202](https://github.com/fluxopt/lpspec/issues/1202)) ([a83883b](https://github.com/fluxopt/lpspec/commit/a83883bbb7bf15574a1d366473763db7a88d8abc))

## [0.0.1-alpha.239](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.238...v0.0.1-alpha.239) (2026-08-21)


### Features

* **language:** a model still owing declarations cannot reach a builder ([#1200](https://github.com/fluxopt/lpspec/issues/1200)) ([5f53d6d](https://github.com/fluxopt/lpspec/commit/5f53d6dcb1332b05af6f80ae78e4ae22adab4eae))


### Documentation

* **ceiling:** the spec's ceiling names no solver ([#1196](https://github.com/fluxopt/lpspec/issues/1196)) ([992f1aa](https://github.com/fluxopt/lpspec/commit/992f1aa260c1035f5eeb98b1d01b897a8b2f8960))

## [0.0.1-alpha.238](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.237...v0.0.1-alpha.238) (2026-08-20)


### Documentation

* a docstring names the number it measured and the symbol that exists ([#1195](https://github.com/fluxopt/lpspec/issues/1195)) ([343ea93](https://github.com/fluxopt/lpspec/commit/343ea93a2c18580274d0be44af0d3e3e7f12ff55))

## [0.0.1-alpha.237](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.236...v0.0.1-alpha.237) (2026-08-20)


### Refactoring

* **errors:** a refusal with one raiser is worded where it is raised ([#1189](https://github.com/fluxopt/lpspec/issues/1189)) ([477ae47](https://github.com/fluxopt/lpspec/commit/477ae4747de313fcbadb5f2485c038b1bac0f27c))

## [0.0.1-alpha.236](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.235...v0.0.1-alpha.236) (2026-08-20)


### Documentation

* a shown model, a key and a count each say what the tree says ([#1192](https://github.com/fluxopt/lpspec/issues/1192)) ([2b89ebf](https://github.com/fluxopt/lpspec/commit/2b89ebfb031e66e558ec631fdd4503db0f76f1ec))

## [0.0.1-alpha.235](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.234...v0.0.1-alpha.235) (2026-08-20)


### Features

* **data:** a lookup's map has one data transport, not two ([#1185](https://github.com/fluxopt/lpspec/issues/1185)) ([65079b6](https://github.com/fluxopt/lpspec/commit/65079b63d38a4179770f97ce1f8f5e55e0d13029))


### Refactoring

* **data:** a lookup's values are checked once, not once per lane ([#1190](https://github.com/fluxopt/lpspec/issues/1190)) ([73c8bf6](https://github.com/fluxopt/lpspec/commit/73c8bf6b8adcb5d49b58f825c5bbda025c6a3891))
* **engine:** an unmapped label is a row that is not there, inside as well ([#1187](https://github.com/fluxopt/lpspec/issues/1187)) ([903ec33](https://github.com/fluxopt/lpspec/commit/903ec338d27284299e3d22b9e1717025dca05d3d))

## [0.0.1-alpha.234](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.233...v0.0.1-alpha.234) (2026-08-20)


### Bug Fixes

* **data:** a map joined onto an index leaves its label order alone ([#1184](https://github.com/fluxopt/lpspec/issues/1184)) ([3116666](https://github.com/fluxopt/lpspec/commit/31166669fb94dc7e3c40e109b3bc6daecdd266e7))

## [0.0.1-alpha.233](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.232...v0.0.1-alpha.233) (2026-08-20)


### Features

* **api:** check answers whether a named sink will take the model ([#928](https://github.com/fluxopt/lpspec/issues/928)) ([4a9d80e](https://github.com/fluxopt/lpspec/commit/4a9d80e5f00cc171175983b31a2159f676503726))
* **language:** a constraint may be quadratic where something can build one ([#942](https://github.com/fluxopt/lpspec/issues/942)) ([7ef15c7](https://github.com/fluxopt/lpspec/commit/7ef15c771ce3c351f92d807ce1caecb37ec0879f))
* **language:** a discount factor is arithmetic, not a column to prepare ([#1177](https://github.com/fluxopt/lpspec/issues/1177)) ([6c5c806](https://github.com/fluxopt/lpspec/commit/6c5c8068484c1cef0c9acf8ae5dc80c4574eeb1d))
* **language:** the objective takes a quadratic term ([#929](https://github.com/fluxopt/lpspec/issues/929)) ([e8c88cb](https://github.com/fluxopt/lpspec/commit/e8c88cbe6e5959a2d8b3b671007660c43319eae0))
* **sinks:** a sink declares what it can ingest, and what it refuses in pairs ([#927](https://github.com/fluxopt/lpspec/issues/927)) ([3af0c43](https://github.com/fluxopt/lpspec/commit/3af0c43e687ab6117d7accccf2d42ee6ba67601a))
* **solver:** gurobi takes the quadratic objectives highs refuses ([#930](https://github.com/fluxopt/lpspec/issues/930)) ([344804b](https://github.com/fluxopt/lpspec/commit/344804be780a838787836a7dc8ddc2aa3b197dd8))


### Documentation

* **sinks:** the capability table says what each sink actually refuses ([#925](https://github.com/fluxopt/lpspec/issues/925)) ([8e0b7c8](https://github.com/fluxopt/lpspec/commit/8e0b7c8804a2a35be27211b162227040595375a2))

## [0.0.1-alpha.232](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.231...v0.0.1-alpha.232) (2026-08-20)


### Features

* **data:** a lookup is supplied without rewriting the index it rides on ([#1179](https://github.com/fluxopt/lpspec/issues/1179)) ([385c048](https://github.com/fluxopt/lpspec/commit/385c0486fa39b425e7f6898e234a89efa162cfd8))

## [0.0.1-alpha.231](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.230...v0.0.1-alpha.231) (2026-08-20)


### Documentation

* **examples:** a curve whose arity is data is four declarations ([#1131](https://github.com/fluxopt/lpspec/issues/1131)) ([9cae48d](https://github.com/fluxopt/lpspec/commit/9cae48d2953c443507dca0606bab64312d798871))

## [0.0.1-alpha.230](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.229...v0.0.1-alpha.230) (2026-08-20)


### Bug Fixes

* **language:** a constraint that decides nothing is refused where the file is read ([#1173](https://github.com/fluxopt/lpspec/issues/1173)) ([c94d330](https://github.com/fluxopt/lpspec/commit/c94d33083ee0032f0ccd5958f21de4ca665460ec)), closes [#1171](https://github.com/fluxopt/lpspec/issues/1171)
* **typeset:** a fill and a group share the operator's one subscript ([#1169](https://github.com/fluxopt/lpspec/issues/1169)) ([8455a73](https://github.com/fluxopt/lpspec/commit/8455a734f5b8b3f4b3fe99e02365657217052a2b)), closes [#1165](https://github.com/fluxopt/lpspec/issues/1165)

## [0.0.1-alpha.229](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.228...v0.0.1-alpha.229) (2026-08-20)


### Bug Fixes

* **compat:** a reduction over an empty dimension builds no row on either lane ([#1170](https://github.com/fluxopt/lpspec/issues/1170)) ([7e348a1](https://github.com/fluxopt/lpspec/commit/7e348a15cd95865694e77883b219abcb16d8d7c3)), closes [#1108](https://github.com/fluxopt/lpspec/issues/1108)

## [0.0.1-alpha.228](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.227...v0.0.1-alpha.228) (2026-08-20)


### Bug Fixes

* **language:** a per-group offset translates each group by its own lag ([#1166](https://github.com/fluxopt/lpspec/issues/1166)) ([2b80850](https://github.com/fluxopt/lpspec/commit/2b80850e0c9eff4cee2a798643fd976b253c5c69)), closes [#1161](https://github.com/fluxopt/lpspec/issues/1161)

## [0.0.1-alpha.227](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.226...v0.0.1-alpha.227) (2026-08-20)


### ⚠ BREAKING CHANGES

* **language:** a piecewise gate is named for the number it is ([#1136](https://github.com/fluxopt/lpspec/issues/1136))

### Bug Fixes

* **language:** a piecewise curve whose gate does not exist there is ungated, not relaxed ([#1159](https://github.com/fluxopt/lpspec/issues/1159)) ([1b927cc](https://github.com/fluxopt/lpspec/commit/1b927ccb38ede4cbfbadfd4f7522740fe495a46e))
* **release:** the alpha stream holds against a breaking commit title ([#1162](https://github.com/fluxopt/lpspec/issues/1162)) ([27f03c3](https://github.com/fluxopt/lpspec/commit/27f03c3673ea56f887284ddc64abff5110c2823f))


### Refactoring

* **language:** a piecewise gate is named for the number it is ([#1136](https://github.com/fluxopt/lpspec/issues/1136)) ([ef119f2](https://github.com/fluxopt/lpspec/commit/ef119f2aa3bf92f6ae988a9251369dadd3124ab6))

## [0.0.1-alpha.226](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.225...v0.0.1-alpha.226) (2026-08-20)


### Documentation

* **language:** the rule that decides what belongs has its own page ([#1153](https://github.com/fluxopt/lpspec/issues/1153)) ([f5c5336](https://github.com/fluxopt/lpspec/commit/f5c533643b80dff6d90892a83a8b231603bd92b7))

## [0.0.1-alpha.225](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.224...v0.0.1-alpha.225) (2026-08-20)


### Refactoring

* **extraction:** the corpus that moves is the operator reference's source ([#1149](https://github.com/fluxopt/lpspec/issues/1149)) ([a71eb62](https://github.com/fluxopt/lpspec/commit/a71eb626cfaa6568f38e31a14cfb5a43ad23a61a))

## [0.0.1-alpha.224](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.223...v0.0.1-alpha.224) (2026-08-20)


### Refactoring

* **tools:** a generator that reads only the language lives with it ([#1147](https://github.com/fluxopt/lpspec/issues/1147)) ([810f1bd](https://github.com/fluxopt/lpspec/commit/810f1bd6a25420c16639bcbf7d008e72cd900cdf))

## [0.0.1-alpha.223](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.222...v0.0.1-alpha.223) (2026-08-20)


### Bug Fixes

* **engine:** four operators refuse a constant beside a term as a lane gap ([#1140](https://github.com/fluxopt/lpspec/issues/1140)) ([b59aa4c](https://github.com/fluxopt/lpspec/commit/b59aa4c6828ef35393493a254ca6ee44a994f757))
* **engine:** sum_back drops a constant at a slot its term is absent from ([#1143](https://github.com/fluxopt/lpspec/issues/1143)) ([2845133](https://github.com/fluxopt/lpspec/commit/28451338c9440dbdbf580740d558c5ca5e2dac06))


### Refactoring

* **errors:** the language raises its own errors ([#1135](https://github.com/fluxopt/lpspec/issues/1135)) ([38c00af](https://github.com/fluxopt/lpspec/commit/38c00af3c84a09c07bf62568dd1d9984d211c663))
* **language:** the AST is read through one declared surface ([#1132](https://github.com/fluxopt/lpspec/issues/1132)) ([fd8bfb1](https://github.com/fluxopt/lpspec/commit/fd8bfb1b0d318b0331eb96eae131ab8e85def026))

## [0.0.1-alpha.222](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.221...v0.0.1-alpha.222) (2026-08-20)


### Features

* **api:** a single constraint row has a spelling ([#1114](https://github.com/fluxopt/lpspec/issues/1114)) ([54bd9e4](https://github.com/fluxopt/lpspec/commit/54bd9e4f402556f5ba99d296d7a8803b9d72cf78))

## [0.0.1-alpha.221](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.220...v0.0.1-alpha.221) (2026-08-20)


### Features

* **language:** a curve shorter than its breakpoint dimension says so ([#1115](https://github.com/fluxopt/lpspec/issues/1115)) ([45dc672](https://github.com/fluxopt/lpspec/commit/45dc672976aff62964ddfb1cf742e00f8addc7bc))
* **language:** a curve's length is a fact of the curve ([#1117](https://github.com/fluxopt/lpspec/issues/1117)) ([3996273](https://github.com/fluxopt/lpspec/commit/3996273f6861fb23c15dd117328167cb34202933))


### Bug Fixes

* **data:** a breakpoint dimension with no index keeps its own message ([#1125](https://github.com/fluxopt/lpspec/issues/1125)) ([0ffef63](https://github.com/fluxopt/lpspec/commit/0ffef63df4e38ab84b35394073fe2ff85f7c9338))
* **data:** a piecewise curve is judged as the model builds it ([#1126](https://github.com/fluxopt/lpspec/issues/1126)) ([c23102b](https://github.com/fluxopt/lpspec/commit/c23102b60342a37875e8096665740249923b82d7))
* **language:** a curve varying along a dim no link carries is named on the link ([#1127](https://github.com/fluxopt/lpspec/issues/1127)) ([a51c151](https://github.com/fluxopt/lpspec/commit/a51c1518137eac091783fc7eab24533b31832d4d))


### Documentation

* **examples:** a gallery model whose curves are not all the same length ([#1119](https://github.com/fluxopt/lpspec/issues/1119)) ([1e9e7d1](https://github.com/fluxopt/lpspec/commit/1e9e7d1edec0e1591c01c1df9bafb8ab8e1df7cc))

## [0.0.1-alpha.220](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.219...v0.0.1-alpha.220) (2026-08-19)


### Features

* **data:** a parameter's dims arrive in columns, never in a pandas index ([#1088](https://github.com/fluxopt/lpspec/issues/1088)) ([d06a601](https://github.com/fluxopt/lpspec/commit/d06a601c49468e842e980215a8f60168c785b161))


### Refactoring

* **data:** both lanes read their data through one front door ([#1076](https://github.com/fluxopt/lpspec/issues/1076)) ([8923df9](https://github.com/fluxopt/lpspec/commit/8923df987fc91f77bc4442cd6da85a480b505046))
* **linopy:** the builder groups by what it translates ([#1092](https://github.com/fluxopt/lpspec/issues/1092)) ([380cdbe](https://github.com/fluxopt/lpspec/commit/380cdbeefb1b26ab49bddcae47654ffc5ba8ee67))

## [0.0.1-alpha.219](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.218...v0.0.1-alpha.219) (2026-08-19)


### Documentation

* **language:** the short-curve remedy names the two methods that refuse it ([#1111](https://github.com/fluxopt/lpspec/issues/1111)) ([a22f099](https://github.com/fluxopt/lpspec/commit/a22f0991b4796041503d6b39596cf85dd9e41c99))

## [0.0.1-alpha.218](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.217...v0.0.1-alpha.218) (2026-08-19)


### Features

* **solver:** a model arrives on the solver its author licences ([#1107](https://github.com/fluxopt/lpspec/issues/1107)) ([262872a](https://github.com/fluxopt/lpspec/commit/262872a9eb6eb836254908ff63e79ae9c4d44010))

## [0.0.1-alpha.217](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.216...v0.0.1-alpha.217) (2026-08-19)


### Bug Fixes

* **data:** a piecewise curve short of a breakpoint is refused, not read as the origin ([#1105](https://github.com/fluxopt/lpspec/issues/1105)) ([af93d7e](https://github.com/fluxopt/lpspec/commit/af93d7eb7f860d7520dfe106a5c0621cf841147d))

## [0.0.1-alpha.216](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.215...v0.0.1-alpha.216) (2026-08-19)


### Features

* **file-out:** a model leaves as MPS, not only as LP ([#1097](https://github.com/fluxopt/lpspec/issues/1097)) ([76dc58b](https://github.com/fluxopt/lpspec/commit/76dc58ba0bd822787fb8bd1f3a7d4bfc2595fac4))
* **language:** a convex cost curve stops paying a variable per breakpoint ([#926](https://github.com/fluxopt/lpspec/issues/926)) ([50d1984](https://github.com/fluxopt/lpspec/commit/50d198412ad5f76ef43375d70a41162ef1cd831b))

## [0.0.1-alpha.215](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.214...v0.0.1-alpha.215) (2026-08-19)


### Features

* **language:** a capacity limit may be grouped by location and technology at once ([#899](https://github.com/fluxopt/lpspec/issues/899)) ([2cabc65](https://github.com/fluxopt/lpspec/commit/2cabc650154a765fb6fc09b6d35e76e0772d1f5c))

## [0.0.1-alpha.214](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.213...v0.0.1-alpha.214) (2026-08-19)


### Documentation

* **errors:** the error table names every class a caller can catch ([#1095](https://github.com/fluxopt/lpspec/issues/1095)) ([497aa56](https://github.com/fluxopt/lpspec/commit/497aa56a2ab96b4183f0e895211e24b6b6ba9b05))

## [0.0.1-alpha.213](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.212...v0.0.1-alpha.213) (2026-08-19)


### Features

* **language:** check names the variable that makes a model unbounded ([#909](https://github.com/fluxopt/lpspec/issues/909)) ([7781e22](https://github.com/fluxopt/lpspec/commit/7781e2253da5e027a4a803b2ff1f00cd40ce09f1))

## [0.0.1-alpha.212](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.211...v0.0.1-alpha.212) (2026-08-19)


### Bug Fixes

* **compat:** the lane says in its own words what it cannot build ([#1087](https://github.com/fluxopt/lpspec/issues/1087)) ([b6a90df](https://github.com/fluxopt/lpspec/commit/b6a90df53f1051c6a1bc93666a129cc818c3eaa9))

## [0.0.1-alpha.211](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.210...v0.0.1-alpha.211) (2026-08-19)


### Refactoring

* **language:** a closed vocabulary is a type, not a string plus a check ([#1083](https://github.com/fluxopt/lpspec/issues/1083)) ([92f615e](https://github.com/fluxopt/lpspec/commit/92f615eeb1200ec3321dc1b446544c97b332b793))

## [0.0.1-alpha.210](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.209...v0.0.1-alpha.210) (2026-08-19)


### Features

* **language:** a sum can take every dim its operand carries ([#1071](https://github.com/fluxopt/lpspec/issues/1071)) ([9874d64](https://github.com/fluxopt/lpspec/commit/9874d649a947814909231eb8120f1e1cefab8c4b))
* **language:** an objective spells the sums that make it one number ([#1077](https://github.com/fluxopt/lpspec/issues/1077)) ([d40ca88](https://github.com/fluxopt/lpspec/commit/d40ca88f84686e63d6aacc6f5802db5149bf9c24))

## [0.0.1-alpha.209](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.208...v0.0.1-alpha.209) (2026-08-19)


### Features

* **language:** the YAML surface has a machine-readable schema ([#718](https://github.com/fluxopt/lpspec/issues/718)) ([d3c2611](https://github.com/fluxopt/lpspec/commit/d3c26113bcfdf6c2584fb02d56c3afea31b7e372))

## [0.0.1-alpha.208](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.207...v0.0.1-alpha.208) (2026-08-19)


### Bug Fixes

* **language:** a coordinate in no group is absent, not an edge the fill speaks for ([#1070](https://github.com/fluxopt/lpspec/issues/1070)) ([60e3574](https://github.com/fluxopt/lpspec/commit/60e3574aeaddebccb011da769c45c6799a1e5ebb))

## [0.0.1-alpha.207](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.206...v0.0.1-alpha.207) (2026-08-19)


### Performance

* **ci:** the LaTeX gate stops costing a minute a run ([#1075](https://github.com/fluxopt/lpspec/issues/1075)) ([661e7e0](https://github.com/fluxopt/lpspec/commit/661e7e0e218a3e8e41d5bab04d50a6d65dec747a))

## [0.0.1-alpha.206](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.205...v0.0.1-alpha.206) (2026-08-19)


### Bug Fixes

* **packaging:** the linopy extra resolves again after upstream merged v1 ([#1073](https://github.com/fluxopt/lpspec/issues/1073)) ([aafac01](https://github.com/fluxopt/lpspec/commit/aafac01d025edc8ebdcabd6b0c2af6d00a8e1d19))

## [0.0.1-alpha.205](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.204...v0.0.1-alpha.205) (2026-08-18)


### Features

* **api:** diagnostics says which parameters arrived short of their dims ([#1067](https://github.com/fluxopt/lpspec/issues/1067)) ([c904400](https://github.com/fluxopt/lpspec/commit/c904400a9e72d6cf731168c1e182076435f70291))

## [0.0.1-alpha.204](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.203...v0.0.1-alpha.204) (2026-08-18)


### Bug Fixes

* **compat:** the eager lane's edge fills the shift's own vacated slot ([#1050](https://github.com/fluxopt/lpspec/issues/1050)) ([3f965f7](https://github.com/fluxopt/lpspec/commit/3f965f7c2cff126b21e61c77e523b42270ddc309))
* **engine:** a per-entity offset's edge is per entity ([#1053](https://github.com/fluxopt/lpspec/issues/1053)) ([53fc3f7](https://github.com/fluxopt/lpspec/commit/53fc3f71228ce0c313557b26aac9004cd8af4d7c))

## [0.0.1-alpha.203](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.202...v0.0.1-alpha.203) (2026-08-18)


### Features

* **ports:** a data column decides which boundary regime a storage obeys ([#1038](https://github.com/fluxopt/lpspec/issues/1038)) ([9879cde](https://github.com/fluxopt/lpspec/commit/9879cde42fa4218fd144a9538289a6dbac6dd163))

## [0.0.1-alpha.202](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.201...v0.0.1-alpha.202) (2026-08-18)


### Bug Fixes

* **tests:** the golden objective reaches the grouping the walk does ([#1056](https://github.com/fluxopt/lpspec/issues/1056)) ([db7b3fb](https://github.com/fluxopt/lpspec/commit/db7b3fbace6d89640820bbaa40371bd09cb9e3f0))


### Refactoring

* **ports:** a pre-horizon state of zero is an edge, not a block ([#1057](https://github.com/fluxopt/lpspec/issues/1057)) ([dd12a37](https://github.com/fluxopt/lpspec/commit/dd12a37bcfb0a3d515f4f535efd7798cb3d008c8))


### Documentation

* **typeset:** every construct is on one page, as the math it prints ([#1039](https://github.com/fluxopt/lpspec/issues/1039)) ([8c6bcca](https://github.com/fluxopt/lpspec/commit/8c6bcca1ba5fecef16afca055bebe548a86144ab))

## [0.0.1-alpha.201](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.200...v0.0.1-alpha.201) (2026-08-18)


### Bug Fixes

* **language:** a divisor that adds is refused at load, not at build ([#1048](https://github.com/fluxopt/lpspec/issues/1048)) ([4153653](https://github.com/fluxopt/lpspec/commit/41536535aada5712e1942c1ab8cf37756fa5d28e))
* **typeset:** an objective term is shown summed over the dims it carries ([#1045](https://github.com/fluxopt/lpspec/issues/1045)) ([3e29596](https://github.com/fluxopt/lpspec/commit/3e29596a2590177199d7d5cf2240485181d7ab17))

## [0.0.1-alpha.200](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.199...v0.0.1-alpha.200) (2026-08-18)


### Features

* **ports:** a risk preference is three variables and two rows ([#1035](https://github.com/fluxopt/lpspec/issues/1035)) ([c3972ff](https://github.com/fluxopt/lpspec/commit/c3972ff6bd840c0ba5cbca634c41f1e87ada3c55))
* **ports:** capacity is chosen once and dispatch once per future ([#1033](https://github.com/fluxopt/lpspec/issues/1033)) ([96a4572](https://github.com/fluxopt/lpspec/commit/96a4572cb85dbf9a5451e6b4362777c0e0d89928))

## [0.0.1-alpha.199](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.198...v0.0.1-alpha.199) (2026-08-18)


### Bug Fixes

* **api:** a bridge out of a bare install says which extra to add ([#1029](https://github.com/fluxopt/lpspec/issues/1029)) ([ce60c03](https://github.com/fluxopt/lpspec/commit/ce60c031c7714551449326af6e37e3479e00448b))


### Documentation

* **ceiling:** a tracked-metric vocabulary is a dimension and a named expression ([#1031](https://github.com/fluxopt/lpspec/issues/1031)) ([ab9d241](https://github.com/fluxopt/lpspec/commit/ab9d241f059cdc73aa6a4969290e1aaa3fec1bff))

## [0.0.1-alpha.198](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.197...v0.0.1-alpha.198) (2026-08-18)


### Bug Fixes

* **engine:** a group with no members caps a row instead of refusing it ([#1027](https://github.com/fluxopt/lpspec/issues/1027)) ([31ab50a](https://github.com/fluxopt/lpspec/commit/31ab50a514bea7fce4b92be167fa341e62216549))

## [0.0.1-alpha.197](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.196...v0.0.1-alpha.197) (2026-08-18)


### Features

* **api:** a badly scaled model says which declaration is badly scaled ([#993](https://github.com/fluxopt/lpspec/issues/993)) ([ab92e2e](https://github.com/fluxopt/lpspec/commit/ab92e2e692e19c6bf23c7cfeac0855111898cd80))
* **data:** a declared dtype is what the column has to be ([#1022](https://github.com/fluxopt/lpspec/issues/1022)) ([acbe2e6](https://github.com/fluxopt/lpspec/commit/acbe2e61030e96341ba67fca98a372dbf5ffac03))

## [0.0.1-alpha.196](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.195...v0.0.1-alpha.196) (2026-08-18)


### Features

* **language:** a translation stops at the edge of its own group ([#1023](https://github.com/fluxopt/lpspec/issues/1023)) ([aa747b5](https://github.com/fluxopt/lpspec/commit/aa747b5b003cb120ab7f96350445f6b99f47d898))

## [0.0.1-alpha.195](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.194...v0.0.1-alpha.195) (2026-08-18)


### Features

* **language:** a boundary names its position inside each group ([#1013](https://github.com/fluxopt/lpspec/issues/1013)) ([f6ccac4](https://github.com/fluxopt/lpspec/commit/f6ccac4e67da4e92091d8a2612b47f889090e7cc))


### Refactoring

* **language:** by= names a lookup, and an offset has a word of its own ([#1016](https://github.com/fluxopt/lpspec/issues/1016)) ([ce13026](https://github.com/fluxopt/lpspec/commit/ce130263a1620c6740b27e2d516bf801914207bc))

## [0.0.1-alpha.194](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.193...v0.0.1-alpha.194) (2026-08-18)


### Bug Fixes

* **data:** a fractional offset no longer truncates to the position below it ([#1005](https://github.com/fluxopt/lpspec/issues/1005)) ([b453c51](https://github.com/fluxopt/lpspec/commit/b453c510b5a1d4f87c831d5b533c3e47448c16ec))

## [0.0.1-alpha.193](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.192...v0.0.1-alpha.193) (2026-08-18)


### Documentation

* **language:** one home for the binding rule, one page for the two grammars ([#1010](https://github.com/fluxopt/lpspec/issues/1010)) ([e004661](https://github.com/fluxopt/lpspec/commit/e00466177325b0c308f80a975d3245e50a0ee400))

## [0.0.1-alpha.192](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.191...v0.0.1-alpha.192) (2026-08-18)


### Bug Fixes

* **data:** a row carrying no value is refused on both lanes ([#1001](https://github.com/fluxopt/lpspec/issues/1001)) ([795b26f](https://github.com/fluxopt/lpspec/commit/795b26fb858140cc68d23f7e63b61fa2ecb166d3))

## [0.0.1-alpha.191](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.190...v0.0.1-alpha.191) (2026-08-18)


### Bug Fixes

* **examples:** every seeding clause names the position, not the label ([#1004](https://github.com/fluxopt/lpspec/issues/1004)) ([4f57d5a](https://github.com/fluxopt/lpspec/commit/4f57d5a818cdff9bbbc99e7de0cf911d045172af))

## [0.0.1-alpha.190](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.189...v0.0.1-alpha.190) (2026-08-18)


### Features

* **language:** a boundary seed survives the index being relabelled ([#904](https://github.com/fluxopt/lpspec/issues/904)) ([ca76c5a](https://github.com/fluxopt/lpspec/commit/ca76c5aaf66c6c4b99ef5bd315df785e3a628a53))
* **ports:** what may be built in a period depends on the period before ([#995](https://github.com/fluxopt/lpspec/issues/995)) ([208e9e7](https://github.com/fluxopt/lpspec/commit/208e9e7c42d4bdc4f23b4acad481754edcdd7939))


### Bug Fixes

* **tests:** the boundary suite binds its index the way sources does ([#1002](https://github.com/fluxopt/lpspec/issues/1002)) ([9da3b02](https://github.com/fluxopt/lpspec/commit/9da3b02c1b492cc9dcedaec01975d5876f937bdc))


### Documentation

* **linopy:** the one construct the eager lane accepts and cannot build ([#990](https://github.com/fluxopt/lpspec/issues/990)) ([e25af73](https://github.com/fluxopt/lpspec/commit/e25af7322f68303b59b5445bb1ad432a98948e6c))

## [0.0.1-alpha.189](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.188...v0.0.1-alpha.189) (2026-08-17)


### Features

* **ports:** a committed unit whose capacity is built declares its big-M ([#991](https://github.com/fluxopt/lpspec/issues/991)) ([98fd368](https://github.com/fluxopt/lpspec/commit/98fd36874ce723f63e4b959e3e0c764144d3e3f4))


### Bug Fixes

* **engine:** a pullback through a null lookup no longer binds its row at zero ([#988](https://github.com/fluxopt/lpspec/issues/988)) ([578868c](https://github.com/fluxopt/lpspec/commit/578868ccac162e6bed247808073df9a971909177))


### Refactoring

* **examples:** an edge leg is named for the edge set it belongs to ([#973](https://github.com/fluxopt/lpspec/issues/973)) ([810a702](https://github.com/fluxopt/lpspec/commit/810a702f235c05756ac0fb0f70f78a3c5a5d763a))

## [0.0.1-alpha.188](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.187...v0.0.1-alpha.188) (2026-08-17)


### Features

* **ports:** an asset exists for the periods its lifetime covers ([#975](https://github.com/fluxopt/lpspec/issues/975)) ([0b33d9f](https://github.com/fluxopt/lpspec/commit/0b33d9ff2ab1744ce730634bf127c85983dc78b3))

## [0.0.1-alpha.187](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.186...v0.0.1-alpha.187) (2026-08-17)


### Features

* **ports:** a global limit is a sum over a selected set ([#967](https://github.com/fluxopt/lpspec/issues/967)) ([42f6e57](https://github.com/fluxopt/lpspec/commit/42f6e5795154830df87329f7a3e7cf5727b1668e))

## [0.0.1-alpha.186](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.185...v0.0.1-alpha.186) (2026-08-17)


### Features

* **ports:** a link may deliver later than it withdrew ([#980](https://github.com/fluxopt/lpspec/issues/980)) ([6e599eb](https://github.com/fluxopt/lpspec/commit/6e599eb94a4b825408aae7eeead48e2e2d93f6f6))

## [0.0.1-alpha.185](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.184...v0.0.1-alpha.185) (2026-08-17)


### Bug Fixes

* **compat:** at() through a label that maps nowhere drops the row, not the coordinate ([#969](https://github.com/fluxopt/lpspec/issues/969)) ([89587ce](https://github.com/fluxopt/lpspec/commit/89587cef8dd2f73b172e34de3ee716951947a5a5))
* **engine:** a row a propagated absence deleted is an omission too ([#981](https://github.com/fluxopt/lpspec/issues/981)) ([0851444](https://github.com/fluxopt/lpspec/commit/0851444a1f199da9b410e0a39c83b5c14f1f7f07))


### Documentation

* **benchmarks:** the published peak says which arm pays the allocator ([#970](https://github.com/fluxopt/lpspec/issues/970)) ([c84b394](https://github.com/fluxopt/lpspec/commit/c84b3942af53452b0062db78f2b59b7e3887f31b))

## [0.0.1-alpha.184](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.183...v0.0.1-alpha.184) (2026-08-17)


### Features

* **ports:** a transmission loss is a fan of tangents, not a piecewise curve ([#964](https://github.com/fluxopt/lpspec/issues/964)) ([3fb4079](https://github.com/fluxopt/lpspec/commit/3fb40799c94c16fb4d9e7cf129de293a5fbc265c))

## [0.0.1-alpha.183](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.182...v0.0.1-alpha.183) (2026-08-17)


### Documentation

* **language:** a propagated absence is a different event from an emptied row ([#945](https://github.com/fluxopt/lpspec/issues/945)) ([409cea0](https://github.com/fluxopt/lpspec/commit/409cea0a01112547b35f6cb22e511f5d4b2bad3c))

## [0.0.1-alpha.182](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.181...v0.0.1-alpha.182) (2026-08-17)


### Features

* **language:** a variable declares whether its absence is zero or undefined ([#950](https://github.com/fluxopt/lpspec/issues/950)) ([adfcd8b](https://github.com/fluxopt/lpspec/commit/adfcd8b82f10ec6416f5e8e2835a80ac786781aa))

## [0.0.1-alpha.181](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.180...v0.0.1-alpha.181) (2026-08-17)


### Features

* **api:** a dimension's labels arrive in sources, like every other input ([#956](https://github.com/fluxopt/lpspec/issues/956)) ([d30890f](https://github.com/fluxopt/lpspec/commit/d30890f4310b417a98f195727bf7d7c71d68ce1b))

## [0.0.1-alpha.180](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.179...v0.0.1-alpha.180) (2026-08-17)


### Features

* **ports:** integrality is one declaration and nothing above it cares ([#943](https://github.com/fluxopt/lpspec/issues/943)) ([f14b5fc](https://github.com/fluxopt/lpspec/commit/f14b5fc28ba4280f3b5a6bfc5118a2db40d98908))

## [0.0.1-alpha.179](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.178...v0.0.1-alpha.179) (2026-08-17)


### Features

* **ports:** a committed unit must stay on for its own window ([#941](https://github.com/fluxopt/lpspec/issues/941)) ([051797a](https://github.com/fluxopt/lpspec/commit/051797a47167089208f1bbb8898e189655339b11))

## [0.0.1-alpha.178](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.177...v0.0.1-alpha.178) (2026-08-17)


### Bug Fixes

* **examples:** the ports row is one row, not three merges of it ([#958](https://github.com/fluxopt/lpspec/issues/958)) ([eb7926b](https://github.com/fluxopt/lpspec/commit/eb7926bd10aa725c12346d2b3e109474e9642c8d))

## [0.0.1-alpha.177](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.176...v0.0.1-alpha.177) (2026-08-17)


### Features

* **ports:** a Store is one signed power with no rating of its own ([#940](https://github.com/fluxopt/lpspec/issues/940)) ([5f4b775](https://github.com/fluxopt/lpspec/commit/5f4b7754a80ccd1ff8b73e147480ff6fe94007be))

## [0.0.1-alpha.176](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.175...v0.0.1-alpha.176) (2026-08-17)


### Features

* **ports:** storage that may spill has a second sink ([#939](https://github.com/fluxopt/lpspec/issues/939)) ([d457b0e](https://github.com/fluxopt/lpspec/commit/d457b0e33c44c8f017eb1bbe1b28ec7a0a45d358))

## [0.0.1-alpha.175](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.174...v0.0.1-alpha.175) (2026-08-17)


### Features

* **ports:** a dispatch or a capacity fixed by data pins only the rows it has ([#938](https://github.com/fluxopt/lpspec/issues/938)) ([d2a27d5](https://github.com/fluxopt/lpspec/commit/d2a27d57a4ad98cf23e95d1585c40878b6d2c516))

## [0.0.1-alpha.174](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.173...v0.0.1-alpha.174) (2026-08-17)


### Features

* **ports:** an energy total over the whole horizon is a bound ([#936](https://github.com/fluxopt/lpspec/issues/936)) ([33fee41](https://github.com/fluxopt/lpspec/commit/33fee416e373146db9947165becd66bd3cb3ba5a))

## [0.0.1-alpha.173](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.172...v0.0.1-alpha.173) (2026-08-17)


### Features

* **ports:** capacity that comes in whole modules is an integer count ([#935](https://github.com/fluxopt/lpspec/issues/935)) ([e55603f](https://github.com/fluxopt/lpspec/commit/e55603f3660881a1592b094bf8038cdd7a687137))

## [0.0.1-alpha.172](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.171...v0.0.1-alpha.172) (2026-08-17)


### Bug Fixes

* **language:** a lookup's map says how labels map, not which exist ([#937](https://github.com/fluxopt/lpspec/issues/937)) ([c50596f](https://github.com/fluxopt/lpspec/commit/c50596f4c8816f8537849d09c532893165485480))

## [0.0.1-alpha.171](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.170...v0.0.1-alpha.171) (2026-08-17)


### Documentation

* **examples:** a model beside the ladder is not listed as a rung ([#947](https://github.com/fluxopt/lpspec/issues/947)) ([470eaf1](https://github.com/fluxopt/lpspec/commit/470eaf135c23a24a3571e4e0562247ee6d37315f))

## [0.0.1-alpha.170](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.169...v0.0.1-alpha.170) (2026-08-16)


### Features

* **language:** a dimension's index has one home, declared or supplied ([#908](https://github.com/fluxopt/lpspec/issues/908)) ([170e58a](https://github.com/fluxopt/lpspec/commit/170e58aae438ece0c206977d7aeedbc69dfc435e)), closes [#907](https://github.com/fluxopt/lpspec/issues/907) [#895](https://github.com/fluxopt/lpspec/issues/895)

## [0.0.1-alpha.169](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.168...v0.0.1-alpha.169) (2026-08-16)


### Refactoring

* **plan:** the relational lane spells a lookup the way the language does ([#903](https://github.com/fluxopt/lpspec/issues/903)) ([f1a4b76](https://github.com/fluxopt/lpspec/commit/f1a4b76dca70467ec8747a8c6e70d71ef7a24128))


### Documentation

* **data:** the precedence list and its closing line agree on how many steps there are ([#902](https://github.com/fluxopt/lpspec/issues/902)) ([2c801e0](https://github.com/fluxopt/lpspec/commit/2c801e0dbd31b57737000ebd8285f3bf68ead39a))

## [0.0.1-alpha.168](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.167...v0.0.1-alpha.168) (2026-08-16)


### Bug Fixes

* **compat:** a grouped sum lands on the dimension it declares ([#900](https://github.com/fluxopt/lpspec/issues/900)) ([f104206](https://github.com/fluxopt/lpspec/commit/f104206ca836038c8d39c35fee77a0a18c94089d)), closes [#756](https://github.com/fluxopt/lpspec/issues/756)

## [0.0.1-alpha.167](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.166...v0.0.1-alpha.167) (2026-08-16)


### Bug Fixes

* **typeset:** a description sets as text, not as markup ([#883](https://github.com/fluxopt/lpspec/issues/883)) ([b7f383b](https://github.com/fluxopt/lpspec/commit/b7f383b01f0569b56592c9b9d90ba0ef91091e9d))

## [0.0.1-alpha.166](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.165...v0.0.1-alpha.166) (2026-08-16)


### Documentation

* **nav:** the gallery section says it holds examples ([#890](https://github.com/fluxopt/lpspec/issues/890)) ([a4bf53b](https://github.com/fluxopt/lpspec/commit/a4bf53b8d56b888edb468bc9b555e8fb6149daae))

## [0.0.1-alpha.165](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.164...v0.0.1-alpha.165) (2026-08-16)


### Refactoring

* **gallery:** a description is written when it says something ([#889](https://github.com/fluxopt/lpspec/issues/889)) ([2792313](https://github.com/fluxopt/lpspec/commit/2792313b08f816b53d012830ca4406d96b55797f))

## [0.0.1-alpha.164](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.163...v0.0.1-alpha.164) (2026-08-16)


### Features

* **language:** a dimension takes its labels from an index, never from the data ([#884](https://github.com/fluxopt/lpspec/issues/884)) ([13aa444](https://github.com/fluxopt/lpspec/commit/13aa444d890402d783ca9922b9a66c111899c2e8))
* **language:** a window's width is a column, not a run of shifts ([#871](https://github.com/fluxopt/lpspec/issues/871)) ([7a90400](https://github.com/fluxopt/lpspec/commit/7a90400233a2346644217a9c8c699e0ddfdd2d73))
* **language:** an offset that differs per entity is a column, not a literal ([#862](https://github.com/fluxopt/lpspec/issues/862)) ([7a3ba7b](https://github.com/fluxopt/lpspec/commit/7a3ba7b5b11744d300cec05eb2ee738a7e1c8f6e))


### Bug Fixes

* **packaging:** the linopy extra installs a linopy that speaks v1 ([#880](https://github.com/fluxopt/lpspec/issues/880)) ([69d95ff](https://github.com/fluxopt/lpspec/commit/69d95ff327829ec7584283f4b9880b44f4189742))


### Documentation

* **language:** the data contract says what binding refuses and what it cannot see ([#881](https://github.com/fluxopt/lpspec/issues/881)) ([24926f8](https://github.com/fluxopt/lpspec/commit/24926f8071332285ddddcd31e1eee0914a0b164d))

## [0.0.1-alpha.163](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.162...v0.0.1-alpha.163) (2026-08-16)


### Bug Fixes

* **compat:** a dimension index is a table on both lanes, and a defect in one reads the same ([#877](https://github.com/fluxopt/lpspec/issues/877)) ([cb7af0a](https://github.com/fluxopt/lpspec/commit/cb7af0abb633fc439db6107ab166298fe403f3f9))

## [0.0.1-alpha.162](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.161...v0.0.1-alpha.162) (2026-08-16)


### Bug Fixes

* **bench:** the floor reads the lookup by the name the model gives it ([#874](https://github.com/fluxopt/lpspec/issues/874)) ([799fa85](https://github.com/fluxopt/lpspec/commit/799fa85f234c8bbdf43c9a15d7c896f934fb476c))

## [0.0.1-alpha.161](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.160...v0.0.1-alpha.161) (2026-08-16)


### Bug Fixes

* **bench:** the eager arm hands over tables the lane still reads ([#872](https://github.com/fluxopt/lpspec/issues/872)) ([97ab607](https://github.com/fluxopt/lpspec/commit/97ab607d7be378bebbade0eb2bbb2d1b4ecf03e5))

## [0.0.1-alpha.160](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.159...v0.0.1-alpha.160) (2026-08-16)


### Features

* **compat:** a dimension the parameters already span needs no second declaration ([#869](https://github.com/fluxopt/lpspec/issues/869)) ([cf02ffa](https://github.com/fluxopt/lpspec/commit/cf02ffab30764bc4f3612b8af4289d24bc4d4ec3))

## [0.0.1-alpha.159](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.158...v0.0.1-alpha.159) (2026-08-16)


### Features

* **compat:** choosing the linopy lane is an import, not a different call ([#864](https://github.com/fluxopt/lpspec/issues/864)) ([0a74aac](https://github.com/fluxopt/lpspec/commit/0a74aac347bc96e70ec24fe82a7d1606295c96d8))


### Bug Fixes

* **compat:** a construct the streaming lane refuses is refused on the linopy lane too ([#865](https://github.com/fluxopt/lpspec/issues/865)) ([48ca92e](https://github.com/fluxopt/lpspec/commit/48ca92ea78644c34932a4aeef7a092b0e3d998f7))


### Documentation

* **architecture:** the linopy lane names its verbs instead of counting them ([#866](https://github.com/fluxopt/lpspec/issues/866)) ([9f93d84](https://github.com/fluxopt/lpspec/commit/9f93d8417f6944b414aa24a57a8d97f75c7dc81f))

## [0.0.1-alpha.158](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.157...v0.0.1-alpha.158) (2026-08-16)


### Refactoring

* **package:** the table boundary is not the engine's, and stops living under it ([#860](https://github.com/fluxopt/lpspec/issues/860)) ([a5fc98f](https://github.com/fluxopt/lpspec/commit/a5fc98fbc13e1de9b447d56f450ba0648e20baf6))

## [0.0.1-alpha.157](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.156...v0.0.1-alpha.157) (2026-08-16)


### Features

* **language:** a file with no objective is a feasibility problem ([#859](https://github.com/fluxopt/lpspec/issues/859)) ([6ddfb5b](https://github.com/fluxopt/lpspec/commit/6ddfb5b2e3196caa71b0f1ec18e2d0dc06dd2c8b))

## [0.0.1-alpha.156](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.155...v0.0.1-alpha.156) (2026-08-16)


### Features

* **api:** a parameter may be written out in Python, not only handed over as a table ([#853](https://github.com/fluxopt/lpspec/issues/853)) ([b8cf21f](https://github.com/fluxopt/lpspec/commit/b8cf21fbea440bc952ddb2c1c9627c5df7a4c237))
* **compat:** the linopy lane reads every table the product path reads ([#857](https://github.com/fluxopt/lpspec/issues/857)) ([c39bd5d](https://github.com/fluxopt/lpspec/commit/c39bd5d93ca4385b9313766634546135b4d74af4))


### Refactoring

* **compat:** the linopy lane constructs a model, it does not attach to one ([#846](https://github.com/fluxopt/lpspec/issues/846)) ([e268b0b](https://github.com/fluxopt/lpspec/commit/e268b0b9eac375c9d4fe1e9fbbb1fd14ef4ac257))

## [0.0.1-alpha.155](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.154...v0.0.1-alpha.155) (2026-08-16)


### Features

* **api:** both lanes read the same tables, and neither reads a dense array ([#855](https://github.com/fluxopt/lpspec/issues/855)) ([4afc9a7](https://github.com/fluxopt/lpspec/commit/4afc9a7993b4c0a0f9c5c31c4aafd3a7e86f14df))


### Documentation

* **ceiling:** data prep computes what the compiler cannot derive ([#854](https://github.com/fluxopt/lpspec/issues/854)) ([add8c3d](https://github.com/fluxopt/lpspec/commit/add8c3db955b9817b8e59abca88d2add3ad92bd2))

## [0.0.1-alpha.154](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.153...v0.0.1-alpha.154) (2026-08-16)


### Bug Fixes

* **typeset:** a boolean mask says which rows are true, not which exist ([#848](https://github.com/fluxopt/lpspec/issues/848)) ([85feab0](https://github.com/fluxopt/lpspec/commit/85feab0aa1ed9fb547656c989ab310487b485109))


### Documentation

* **ledger:** a per-generator minimum up time is sayable ([#844](https://github.com/fluxopt/lpspec/issues/844)) ([0cc51da](https://github.com/fluxopt/lpspec/commit/0cc51da435dc1dee951b0046fccab111a19ccab1))

## [0.0.1-alpha.153](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.152...v0.0.1-alpha.153) (2026-08-16)


### Documentation

* the reference speaks to a caller, and the internals sit under about ([#831](https://github.com/fluxopt/lpspec/issues/831)) ([a105582](https://github.com/fluxopt/lpspec/commit/a1055829d48a643e64ac42e47a738cce10d9df65))

## [0.0.1-alpha.152](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.151...v0.0.1-alpha.152) (2026-08-16)


### Documentation

* **ports:** a claim is checked against the instance before a file is written ([#841](https://github.com/fluxopt/lpspec/issues/841)) ([7a40f1b](https://github.com/fluxopt/lpspec/commit/7a40f1b61cf55e4596f8e8a8830e4d01e7a758b0))

## [0.0.1-alpha.151](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.150...v0.0.1-alpha.151) (2026-08-16)


### Features

* **language:** a small relation is declared, not supplied ([#771](https://github.com/fluxopt/lpspec/issues/771)) ([3be93de](https://github.com/fluxopt/lpspec/commit/3be93de50882bb766424e9da97b32b3e254734f0))
* **language:** a where reads a lookup ([#768](https://github.com/fluxopt/lpspec/issues/768)) ([ab584f2](https://github.com/fluxopt/lpspec/commit/ab584f25eab0aac7c5eac4f9740f00017a3975e4))


### Bug Fixes

* **language:** a declared lookup map holds labels of the declared dtype ([#781](https://github.com/fluxopt/lpspec/issues/781)) ([5ec2c2f](https://github.com/fluxopt/lpspec/commit/5ec2c2fa5c12c87b41e0c4cf464094da951952a9))
* **language:** two lookups are comparable only where their labels are ([#780](https://github.com/fluxopt/lpspec/issues/780)) ([b2e2742](https://github.com/fluxopt/lpspec/commit/b2e274240147ed519866f03e0fab39f424b9c201))

## [0.0.1-alpha.150](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.149...v0.0.1-alpha.150) (2026-08-16)


### Bug Fixes

* **typeset:** every operator renders, and SPEC §7 proves it ([#837](https://github.com/fluxopt/lpspec/issues/837)) ([6454f72](https://github.com/fluxopt/lpspec/commit/6454f7277bd64b76528dd7dc4398eb18183912ac))

## [0.0.1-alpha.149](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.148...v0.0.1-alpha.149) (2026-08-16)


### Features

* **gallery:** a model proves every many-to-many shape load-bearing ([#747](https://github.com/fluxopt/lpspec/issues/747)) ([bb75a1a](https://github.com/fluxopt/lpspec/commit/bb75a1af2714f4c6e5d3ef7dcee8da4a8462db9d))
* **language:** a coordinate is a named top-level lookup ([#742](https://github.com/fluxopt/lpspec/issues/742)) ([a326026](https://github.com/fluxopt/lpspec/commit/a326026d6a020752e2ae24aeb134b0a5130f7b59))
* **language:** a lookup addresses itself, and the sibling kwarg is gone ([#759](https://github.com/fluxopt/lpspec/issues/759)) ([01c243c](https://github.com/fluxopt/lpspec/commit/01c243ca15c1fbecf4ad2a4bf7a99ed0775c3805))


### Bug Fixes

* **typeset:** the pullback test states its coordinate as a lookup ([#835](https://github.com/fluxopt/lpspec/issues/835)) ([fb74811](https://github.com/fluxopt/lpspec/commit/fb7481151e6853651daae7bab5caf2fc422a2bba))


### Documentation

* **gallery:** a model that uses at() says so in the constructs matrix ([#749](https://github.com/fluxopt/lpspec/issues/749)) ([af2a3db](https://github.com/fluxopt/lpspec/commit/af2a3db9c85cac572a4fbbf7dc4a831ae2deac03))

## [0.0.1-alpha.148](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.147...v0.0.1-alpha.148) (2026-08-16)


### Features

* **api:** a solve says how much of the session it keeps ([#815](https://github.com/fluxopt/lpspec/issues/815)) ([4cd43f5](https://github.com/fluxopt/lpspec/commit/4cd43f503de0f28b63e8e194bd623cfdb99f7550))


### Bug Fixes

* **typeset:** a translated index shows every operator that moved it ([#830](https://github.com/fluxopt/lpspec/issues/830)) ([4c45e6b](https://github.com/fluxopt/lpspec/commit/4c45e6bad586f50f99b0a95a6e94e8acefca396b))

## [0.0.1-alpha.147](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.146...v0.0.1-alpha.147) (2026-08-15)


### Refactoring

* **bench:** one comparison table, and it renders the same bytes twice ([#806](https://github.com/fluxopt/lpspec/issues/806)) ([28a39fa](https://github.com/fluxopt/lpspec/commit/28a39fa97f9872c98b1afb98e6837b68d10ab6f7))
* **bench:** the harness stops carrying what no longer measures anything ([#804](https://github.com/fluxopt/lpspec/issues/804)) ([441de0f](https://github.com/fluxopt/lpspec/commit/441de0fb9d1cb094b62c6bf93acda87b5c894c41))

## [0.0.1-alpha.146](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.145...v0.0.1-alpha.146) (2026-08-15)


### Features

* **gallery:** a fuel curve is a floor per piece ([#808](https://github.com/fluxopt/lpspec/issues/808)) ([c2b0ef6](https://github.com/fluxopt/lpspec/commit/c2b0ef6c1c1bcb3bd875cda17993ec348a2e78e8))


### Documentation

* **gallery:** every model says what it is, and so does every declaration in it ([#825](https://github.com/fluxopt/lpspec/issues/825)) ([28eae9b](https://github.com/fluxopt/lpspec/commit/28eae9b7983a0f985502a30e69075dbb8e54cd27))

## [0.0.1-alpha.145](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.144...v0.0.1-alpha.145) (2026-08-15)


### Documentation

* **gallery:** the model list names every model ([#819](https://github.com/fluxopt/lpspec/issues/819)) ([6a78fcc](https://github.com/fluxopt/lpspec/commit/6a78fccb161039f2de2d3727a928ac766bb5177f))

## [0.0.1-alpha.144](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.143...v0.0.1-alpha.144) (2026-08-15)


### Features

* **language:** a declaration says what it is, and the legend reads it ([#812](https://github.com/fluxopt/lpspec/issues/812)) ([6b31278](https://github.com/fluxopt/lpspec/commit/6b3127863818da8c6653e4821e7b7444f13b2402))
* **language:** a file says what model it is ([#813](https://github.com/fluxopt/lpspec/issues/813)) ([6c081e1](https://github.com/fluxopt/lpspec/commit/6c081e1f08257e497cae399175d806cde7209b86))
* **language:** a named expression can say what it counts ([#814](https://github.com/fluxopt/lpspec/issues/814)) ([d57aad1](https://github.com/fluxopt/lpspec/commit/d57aad1935a9dff843263b63136685cdfe137ad2))

## [0.0.1-alpha.143](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.142...v0.0.1-alpha.143) (2026-08-15)


### Features

* **gallery:** an operational life is a window read from data ([#805](https://github.com/fluxopt/lpspec/issues/805)) ([7071fc0](https://github.com/fluxopt/lpspec/commit/7071fc009d59b142f67a293efb86b060a6777988))

## [0.0.1-alpha.142](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.141...v0.0.1-alpha.142) (2026-08-15)


### Features

* **gallery:** two connections may join one depot and centre ([#790](https://github.com/fluxopt/lpspec/issues/790)) ([80d1949](https://github.com/fluxopt/lpspec/commit/80d1949fd2c803f9fef2ec4db03ce279924c1e07))

## [0.0.1-alpha.141](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.140...v0.0.1-alpha.141) (2026-08-15)


### Features

* **gallery:** a path serves one call and crosses many arcs ([#789](https://github.com/fluxopt/lpspec/issues/789)) ([640558b](https://github.com/fluxopt/lpspec/commit/640558beb202161a529fc3864cd49bd58cdf95d5))

## [0.0.1-alpha.140](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.139...v0.0.1-alpha.140) (2026-08-15)


### Documentation

* **site:** the notebook pages say it once ([#798](https://github.com/fluxopt/lpspec/issues/798)) ([fd32ad1](https://github.com/fluxopt/lpspec/commit/fd32ad141e154e0ce0e132d2a4b4acd15050c5c1))

## [0.0.1-alpha.139](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.138...v0.0.1-alpha.139) (2026-08-15)


### Bug Fixes

* **bench:** the floor's parity check runs again ([#799](https://github.com/fluxopt/lpspec/issues/799)) ([9f240ba](https://github.com/fluxopt/lpspec/commit/9f240badab1cb5cecf28bc6552bfa2c961c5ca1b))

## [0.0.1-alpha.138](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.137...v0.0.1-alpha.138) (2026-08-15)


### Features

* **api:** a named expression is readable after a solve ([#726](https://github.com/fluxopt/lpspec/issues/726)) ([e41fbe1](https://github.com/fluxopt/lpspec/commit/e41fbe10f8870bd20e8d879b6c0cd33188629579))
* **gallery:** a generator sits on a bus and burns a carrier ([#783](https://github.com/fluxopt/lpspec/issues/783)) ([5334608](https://github.com/fluxopt/lpspec/commit/53346088f9ed48ce6deddc9ca1b12d65d28a071e))
* **strategy:** a sweep reads the quantity its model names ([#748](https://github.com/fluxopt/lpspec/issues/748)) ([1268fb5](https://github.com/fluxopt/lpspec/commit/1268fb5d57dcdca98ef617498d4c7c0913db3053))

## [0.0.1-alpha.137](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.136...v0.0.1-alpha.137) (2026-08-15)


### Documentation

* **examples:** a notebook changes a model without a modelling API ([#788](https://github.com/fluxopt/lpspec/issues/788)) ([50dd21f](https://github.com/fluxopt/lpspec/commit/50dd21f5ff7cfd76c769229ee103a8c5e29093b4))

## [0.0.1-alpha.136](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.135...v0.0.1-alpha.136) (2026-08-15)


### Features

* **api:** a solve says what it started from, and can be forced cold ([#728](https://github.com/fluxopt/lpspec/issues/728)) ([dcf760e](https://github.com/fluxopt/lpspec/commit/dcf760eaca984455987735c59850c0c0528424ce))

## [0.0.1-alpha.135](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.134...v0.0.1-alpha.135) (2026-08-15)


### Documentation

* **agents:** a bug fix has to prove the bug was there ([#784](https://github.com/fluxopt/lpspec/issues/784)) ([5de3b49](https://github.com/fluxopt/lpspec/commit/5de3b496f64c4a0ec1de9763b369895887818767))

## [0.0.1-alpha.134](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.133...v0.0.1-alpha.134) (2026-08-15)


### Bug Fixes

* **tests:** the harness reads a model through the product's loader ([#778](https://github.com/fluxopt/lpspec/issues/778)) ([bf9e8f6](https://github.com/fluxopt/lpspec/commit/bf9e8f65a163fc26269c64784fa1b4ae741fe6f7))

## [0.0.1-alpha.133](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.132...v0.0.1-alpha.133) (2026-08-14)


### Bug Fixes

* **bench:** a second benchmark refuses to share the machine ([#723](https://github.com/fluxopt/lpspec/issues/723)) ([f92d676](https://github.com/fluxopt/lpspec/commit/f92d676856ab3ce2bdb864001be8be5ef70ddd8a))

## [0.0.1-alpha.132](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.131...v0.0.1-alpha.132) (2026-08-14)


### Refactoring

* **language:** the code spells the builtins the way the docs do ([#754](https://github.com/fluxopt/lpspec/issues/754)) ([799c98b](https://github.com/fluxopt/lpspec/commit/799c98b10cd38380859bcc35436a94e9fbd99277))

## [0.0.1-alpha.131](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.130...v0.0.1-alpha.131) (2026-08-14)


### Bug Fixes

* **typeset:** a symbol table says which notation it is written in ([#740](https://github.com/fluxopt/lpspec/issues/740)) ([9e392ab](https://github.com/fluxopt/lpspec/commit/9e392ab9cf9dc40bb9f3b58206c2be6e7d250513))

## [0.0.1-alpha.130](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.129...v0.0.1-alpha.130) (2026-08-14)


### Documentation

* **examples:** the corpus still spells what the language replaced ([#739](https://github.com/fluxopt/lpspec/issues/739)) ([98ba17c](https://github.com/fluxopt/lpspec/commit/98ba17cf45b768e21cabc9b05a343fd897811f71))

## [0.0.1-alpha.129](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.128...v0.0.1-alpha.129) (2026-08-14)


### Features

* **language:** a variable declares its domain, not a pair of flags ([#720](https://github.com/fluxopt/lpspec/issues/720)) ([690cba1](https://github.com/fluxopt/lpspec/commit/690cba16a35a20e09c802a1054fd29653cd421f1))

## [0.0.1-alpha.128](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.127...v0.0.1-alpha.128) (2026-08-14)


### Features

* **api:** a slow build says where the time went ([#717](https://github.com/fluxopt/lpspec/issues/717)) ([bab22b0](https://github.com/fluxopt/lpspec/commit/bab22b0ad8a1e3de17d873f4d0942b944329ba2f))
* **api:** a solved constraint row says its activity ([#716](https://github.com/fluxopt/lpspec/issues/716)) ([63205d4](https://github.com/fluxopt/lpspec/commit/63205d4e1843d23f446a4d3ac654890dd2a571f0))

## [0.0.1-alpha.127](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.126...v0.0.1-alpha.127) (2026-08-14)


### Documentation

* the builtins have one name and a derived dimension states its cost ([#715](https://github.com/fluxopt/lpspec/issues/715)) ([916cd8d](https://github.com/fluxopt/lpspec/commit/916cd8d215177463f59502272e37fb5943cfa884))

## [0.0.1-alpha.126](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.125...v0.0.1-alpha.126) (2026-08-14)


### Bug Fixes

* **engine:** a coordinate may target a dimension nothing spans yet ([#714](https://github.com/fluxopt/lpspec/issues/714)) ([f93969e](https://github.com/fluxopt/lpspec/commit/f93969e504fb543c4a1fd419f9628ff577bacd50))
* **tests:** the coordinate-target fixture speaks the current objective surface ([#734](https://github.com/fluxopt/lpspec/issues/734)) ([5d18833](https://github.com/fluxopt/lpspec/commit/5d18833b704626ca3522fd4ef24663a93eb0c6a2))

## [0.0.1-alpha.125](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.124...v0.0.1-alpha.125) (2026-08-14)


### Features

* **language:** a second objective is unsayable, not caught ([#729](https://github.com/fluxopt/lpspec/issues/729)) ([6c9d986](https://github.com/fluxopt/lpspec/commit/6c9d9860f167c6390dc9da71080b0c5b1d164bbe))

## [0.0.1-alpha.124](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.123...v0.0.1-alpha.124) (2026-08-14)


### Documentation

* **design:** the ceiling's indicator row names its live issue ([#712](https://github.com/fluxopt/lpspec/issues/712)) ([81facb4](https://github.com/fluxopt/lpspec/commit/81facb476923ad3d856eb81ce2e59e80eb106708))

## [0.0.1-alpha.123](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.122...v0.0.1-alpha.123) (2026-08-14)


### Performance

* **engine:** a coefficient of zero stops reaching the solver ([#685](https://github.com/fluxopt/lpspec/issues/685)) ([6a6c026](https://github.com/fluxopt/lpspec/commit/6a6c02691f47bd63fc3d08d44c46b2adf4b2e3fc))
* **engine:** a share stops carrying its zeros through the sort ([#686](https://github.com/fluxopt/lpspec/issues/686)) ([ca7b54e](https://github.com/fluxopt/lpspec/commit/ca7b54e7ebb30e66d557ab29c468a8b9290fd85e))


### Documentation

* **spec:** the indicator and lp rows cite the issues that hold them ([#696](https://github.com/fluxopt/lpspec/issues/696)) ([2aa70c9](https://github.com/fluxopt/lpspec/commit/2aa70c93c16819673d9a555d244dfb778a79237f))

## [0.0.1-alpha.122](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.121...v0.0.1-alpha.122) (2026-08-14)


### Features

* **api:** diagnostics say what the sink added to the model it was handed ([#689](https://github.com/fluxopt/lpspec/issues/689)) ([0e6c75a](https://github.com/fluxopt/lpspec/commit/0e6c75a78a10347bb8ae3de250176d84d45cab43))
* **language:** a piecewise curve may name its formulation ([#688](https://github.com/fluxopt/lpspec/issues/688)) ([d420c8b](https://github.com/fluxopt/lpspec/commit/d420c8b74ce2c9991fbc15583fcaacb10635d87f))
* **language:** a variable may declare a special-ordered set over one of its dims ([#687](https://github.com/fluxopt/lpspec/issues/687)) ([94cc18e](https://github.com/fluxopt/lpspec/commit/94cc18e2529185c8269d9286fe3d403bca5f0ac5))


### Performance

* **engine:** a set costs a fraction of the variable it is declared over ([#691](https://github.com/fluxopt/lpspec/issues/691)) ([413d493](https://github.com/fluxopt/lpspec/commit/413d493d125b66329f2881b096b239f2507b4b53))

## [0.0.1-alpha.121](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.120...v0.0.1-alpha.121) (2026-08-13)


### Documentation

* **models:** both tabs start from the same tables ([#678](https://github.com/fluxopt/lpspec/issues/678)) ([dc50136](https://github.com/fluxopt/lpspec/commit/dc501361225efc61c992c1a1b3e49bf0d846fc4d))
* **models:** every teaching model is verified and reads in linopy ([#680](https://github.com/fluxopt/lpspec/issues/680)) ([613a8e8](https://github.com/fluxopt/lpspec/commit/613a8e85fc752eb64808c0fab624dd8f16044eb7))
* **models:** the journey from files to the shared tables has one page ([#679](https://github.com/fluxopt/lpspec/issues/679)) ([bdc636d](https://github.com/fluxopt/lpspec/commit/bdc636dbc4c04b75114db4007c9499f4b33f811e))
* **models:** the linopy tabs read as linopy users write, types said where hidden ([#681](https://github.com/fluxopt/lpspec/issues/681)) ([840e79e](https://github.com/fluxopt/lpspec/commit/840e79ee48b9c9d885638ed5d489a39c2c95ec23))
* **models:** the model files speak as a modeller, the pages teach the language ([#682](https://github.com/fluxopt/lpspec/issues/682)) ([07c58b4](https://github.com/fluxopt/lpspec/commit/07c58b4c559268f9238649737e6d9be553ab01e5))

## [0.0.1-alpha.120](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.119...v0.0.1-alpha.120) (2026-08-13)


### Documentation

* **models:** the gallery shows the modelling, not the harness ([#673](https://github.com/fluxopt/lpspec/issues/673)) ([6adba43](https://github.com/fluxopt/lpspec/commit/6adba43b3aa25d04b269ca461827876b7804617e))

## [0.0.1-alpha.119](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.118...v0.0.1-alpha.119) (2026-08-13)


### Refactoring

* dead code, one-entry registries and re-export shims are gone ([#675](https://github.com/fluxopt/lpspec/issues/675)) ([a1f61df](https://github.com/fluxopt/lpspec/commit/a1f61df2c1fa2d07c5acb6e86266831999485666))

## [0.0.1-alpha.118](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.117...v0.0.1-alpha.118) (2026-08-13)


### Refactoring

* a value crossing a module boundary names its fields ([#667](https://github.com/fluxopt/lpspec/issues/667)) ([51b5249](https://github.com/fluxopt/lpspec/commit/51b5249bbf289bf71507eb02412ef66a363071e8))

## [0.0.1-alpha.117](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.116...v0.0.1-alpha.117) (2026-08-13)


### Documentation

* **models:** the teaching models are verified, read in linopy, and show their call ([#671](https://github.com/fluxopt/lpspec/issues/671)) ([83ab265](https://github.com/fluxopt/lpspec/commit/83ab265817a9828a37a6801031cee11d869165f5))

## [0.0.1-alpha.116](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.115...v0.0.1-alpha.116) (2026-08-13)


### Documentation

* **models:** each verified reference reads as a tab beside the model ([#665](https://github.com/fluxopt/lpspec/issues/665)) ([9497594](https://github.com/fluxopt/lpspec/commit/94975949822b0c9e0532de1c4ceb8839766023ec))

## [0.0.1-alpha.115](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.114...v0.0.1-alpha.115) (2026-08-13)


### Documentation

* **agents:** a guard nobody can delete unseen is the coverage bar ([#662](https://github.com/fluxopt/lpspec/issues/662)) ([4b90f6e](https://github.com/fluxopt/lpspec/commit/4b90f6ee7f529f7998708d3a8d28a491acccbc4f))

## [0.0.1-alpha.114](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.113...v0.0.1-alpha.114) (2026-08-13)


### Features

* **api:** a built model takes new numbers without being rebuilt ([#626](https://github.com/fluxopt/lpspec/issues/626)) ([5123215](https://github.com/fluxopt/lpspec/commit/5123215731ab1b67481ace086f5e34e1ef2e3303))


### Performance

* **strategy:** a serial sweep binds one model instead of building each slice ([#627](https://github.com/fluxopt/lpspec/issues/627)) ([cfdcc95](https://github.com/fluxopt/lpspec/commit/cfdcc9563c73c4aee9dd38006820142696c9c623))

## [0.0.1-alpha.113](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.112...v0.0.1-alpha.113) (2026-08-13)


### Documentation

* **agents:** AGENTS.md scans as rules rather than paragraphs ([#657](https://github.com/fluxopt/lpspec/issues/657)) ([f5bba8e](https://github.com/fluxopt/lpspec/commit/f5bba8eb5430452bf19400f1c3263cdc104a748d))

## [0.0.1-alpha.112](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.111...v0.0.1-alpha.112) (2026-08-13)


### Performance

* **engine:** a solver reads the row vectors instead of scattering them ([#651](https://github.com/fluxopt/lpspec/issues/651)) ([ba1587d](https://github.com/fluxopt/lpspec/commit/ba1587db8221b039022977b30bf3951f3a5768b8))

## [0.0.1-alpha.111](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.110...v0.0.1-alpha.111) (2026-08-13)


### Performance

* **engine:** constants stop doubling the cost of a constraint's rows ([#650](https://github.com/fluxopt/lpspec/issues/650)) ([a89a2c9](https://github.com/fluxopt/lpspec/commit/a89a2c9f358b3ff34e85678c7c163430423f34af))
* **sinks:** a row's comparison crosses to the solver as a code, not a string ([#648](https://github.com/fluxopt/lpspec/issues/648)) ([b0bb8c3](https://github.com/fluxopt/lpspec/commit/b0bb8c310d1f35a8bc157f00461efdbe2be2c65c))
* **sinks:** the solver's column vectors stop being walked five times ([#649](https://github.com/fluxopt/lpspec/issues/649)) ([98d604d](https://github.com/fluxopt/lpspec/commit/98d604d0fe8de0573c993ab57dda84c977988b9b))

## [0.0.1-alpha.110](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.109...v0.0.1-alpha.110) (2026-08-12)


### Bug Fixes

* **engine:** a variable and a constraint may share a name ([#645](https://github.com/fluxopt/lpspec/issues/645)) ([e4329cf](https://github.com/fluxopt/lpspec/commit/e4329cf0b045583aed01e16d07ca192c1eeeceb2))

## [0.0.1-alpha.109](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.108...v0.0.1-alpha.109) (2026-08-12)


### Documentation

* **agents:** the package's docstrings have one checkable form ([#632](https://github.com/fluxopt/lpspec/issues/632)) ([bdc301f](https://github.com/fluxopt/lpspec/commit/bdc301f1c05bd313a8e87000a67cc22914c27a8e))

## [0.0.1-alpha.108](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.107...v0.0.1-alpha.108) (2026-08-12)


### Documentation

* **agents:** what is non-negotiable, and what is a default you may argue with ([#624](https://github.com/fluxopt/lpspec/issues/624)) ([526c11a](https://github.com/fluxopt/lpspec/commit/526c11a09df7d6e3483278177f8f04574e378fcf))

## [0.0.1-alpha.107](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.106...v0.0.1-alpha.107) (2026-08-12)


### Refactoring

* every claim lives where it is read, not in a comment ([#612](https://github.com/fluxopt/lpspec/issues/612)) ([a5404d7](https://github.com/fluxopt/lpspec/commit/a5404d7f340c015fa6a3001f3b59116a4b441589))

## [0.0.1-alpha.106](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.105...v0.0.1-alpha.106) (2026-08-12)


### Bug Fixes

* **bench:** the marginal-cost caption is rendered, and says what it measures ([#621](https://github.com/fluxopt/lpspec/issues/621)) ([93cd5fe](https://github.com/fluxopt/lpspec/commit/93cd5fe37deb30d7df29700a2fd020f1bc10555f))

## [0.0.1-alpha.105](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.104...v0.0.1-alpha.105) (2026-08-12)


### Documentation

* **agents:** a number leaves only with a pointer, and vagueness is not a conclusion ([#618](https://github.com/fluxopt/lpspec/issues/618)) ([5489607](https://github.com/fluxopt/lpspec/commit/54896072d7b57c5d172eef77ac2135496f34e8b9))

## [0.0.1-alpha.104](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.103...v0.0.1-alpha.104) (2026-08-12)


### Documentation

* **agents:** a PR with no intent line shows the prompt that produced it ([#616](https://github.com/fluxopt/lpspec/issues/616)) ([aa134ab](https://github.com/fluxopt/lpspec/commit/aa134abbfbfbdafca58b242636bd2c1d8b12938d))

## [0.0.1-alpha.103](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.102...v0.0.1-alpha.103) (2026-08-11)


### Performance

* **examples:** the benders loop parses each model once ([#613](https://github.com/fluxopt/lpspec/issues/613)) ([5f03dbd](https://github.com/fluxopt/lpspec/commit/5f03dbd07154c13798dfbe3b1ef008467886af86))

## [0.0.1-alpha.102](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.101...v0.0.1-alpha.102) (2026-08-11)


### Features

* **strategy:** one model solved over scenarios, windows and pathways ([#459](https://github.com/fluxopt/lpspec/issues/459)) ([5d4c4ce](https://github.com/fluxopt/lpspec/commit/5d4c4cea287512a70c2cc23e910309abe07e3bea))

## [0.0.1-alpha.101](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.100...v0.0.1-alpha.101) (2026-08-11)


### Documentation

* **agents:** a claim belongs in the message that prints ([#608](https://github.com/fluxopt/lpspec/issues/608)) ([d0b1e86](https://github.com/fluxopt/lpspec/commit/d0b1e86397935c70ac9773e538e5f2577909ddd4))

## [0.0.1-alpha.100](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.99...v0.0.1-alpha.100) (2026-08-11)


### Documentation

* **agents:** a reader can tell the maintainer's voice from an agent's ([#605](https://github.com/fluxopt/lpspec/issues/605)) ([b987971](https://github.com/fluxopt/lpspec/commit/b98797149cc4e5e61931de7ba2ffae0ede7bbb65))

## [0.0.1-alpha.99](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.98...v0.0.1-alpha.99) (2026-08-11)


### Documentation

* AGENTS.md carries how we work, CLAUDE.md just imports it ([#599](https://github.com/fluxopt/lpspec/issues/599)) ([e815252](https://github.com/fluxopt/lpspec/commit/e815252c5ce40e72104ca79300d57321d51d08f1))

## [0.0.1-alpha.98](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.97...v0.0.1-alpha.98) (2026-08-11)


### Documentation

* the copyright line names whoever wrote it, and that was never PyPSA ([#601](https://github.com/fluxopt/lpspec/issues/601)) ([06fb7d5](https://github.com/fluxopt/lpspec/commit/06fb7d5ddcbf2753e780afe1d0f8434f33334485))

## [0.0.1-alpha.97](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.96...v0.0.1-alpha.97) (2026-08-11)


### Documentation

* the surface is Calliope's, the vocabulary is linopy's — say so ([#598](https://github.com/fluxopt/lpspec/issues/598)) ([a1042fc](https://github.com/fluxopt/lpspec/commit/a1042fccc070c5cc730445eb7da180d5a66705e6))

## [0.0.1-alpha.96](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.95...v0.0.1-alpha.96) (2026-08-11)


### Refactoring

* cut the prose, keep the facts ([#592](https://github.com/fluxopt/lpspec/issues/592)) ([dcc279b](https://github.com/fluxopt/lpspec/commit/dcc279b6c0cc4d8a7b84e9f5f610cbd12fa6d88c))


### Documentation

* **design:** decomposition, as evidence rather than as a feature ([#597](https://github.com/fluxopt/lpspec/issues/597)) ([c736da9](https://github.com/fluxopt/lpspec/commit/c736da9f046766b80915499b7bec24ab7ee8cf76))

## [0.0.1-alpha.95](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.94...v0.0.1-alpha.95) (2026-08-11)


### Bug Fixes

* **engine:** a returned frame speaks String, so it joins the caller's own ([#593](https://github.com/fluxopt/lpspec/issues/593)) ([69b9ad2](https://github.com/fluxopt/lpspec/commit/69b9ad21f2772232c3d19245b764dbebf231f1ea))

## [0.0.1-alpha.94](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.93...v0.0.1-alpha.94) (2026-08-11)


### Refactoring

* one answer per question ([#589](https://github.com/fluxopt/lpspec/issues/589)) ([8a831cc](https://github.com/fluxopt/lpspec/commit/8a831cc86f1089654ba22b3773e70852911b9c49))

## [0.0.1-alpha.93](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.92...v0.0.1-alpha.93) (2026-08-11)


### Bug Fixes

* **api:** a closed result says it was closed ([#587](https://github.com/fluxopt/lpspec/issues/587)) ([45cc1d0](https://github.com/fluxopt/lpspec/commit/45cc1d0f68364d8a7ca591cc2d2c3b37ae884837))

## [0.0.1-alpha.92](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.91...v0.0.1-alpha.92) (2026-08-11)


### Bug Fixes

* **data:** an empty index keeps the dimension's declared dtype ([#585](https://github.com/fluxopt/lpspec/issues/585)) ([b5bd409](https://github.com/fluxopt/lpspec/commit/b5bd4091671aa7614386d9545ff7084b0c1842fa))

## [0.0.1-alpha.91](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.90...v0.0.1-alpha.91) (2026-08-11)


### Performance

* **engine:** the objective keeps the hash count — bought order was a shape-dependent bet ([#581](https://github.com/fluxopt/lpspec/issues/581)) ([d30598f](https://github.com/fluxopt/lpspec/commit/d30598feb563ecbcd58ea73311901bac123b4246))

## [0.0.1-alpha.90](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.89...v0.0.1-alpha.90) (2026-08-11)


### Refactoring

* **engine:** ask the data, not the declarations ([#520](https://github.com/fluxopt/lpspec/issues/520)) ([17f8e5d](https://github.com/fluxopt/lpspec/commit/17f8e5d0f79072bbaffc7df5d174e17ca24387a7))

## [0.0.1-alpha.89](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.88...v0.0.1-alpha.89) (2026-08-11)


### Bug Fixes

* **plan:** a divisor under a pullback is still named ([#571](https://github.com/fluxopt/lpspec/issues/571)) ([2591995](https://github.com/fluxopt/lpspec/commit/2591995bb5bf6f66eefdfd1fd24e0f856d70a8b4))

## [0.0.1-alpha.88](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.87...v0.0.1-alpha.88) (2026-08-11)


### Performance

* **engine:** the bound attach reads the ordinal off the Enum, not a dictionary ([#568](https://github.com/fluxopt/lpspec/issues/568)) ([6847419](https://github.com/fluxopt/lpspec/commit/68474193fb1b920ecfe6278178888f6018975644))

## [0.0.1-alpha.87](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.86...v0.0.1-alpha.87) (2026-08-11)


### Performance

* **engine:** a bound dense over the variable product is attached, not joined ([#511](https://github.com/fluxopt/lpspec/issues/511)) ([8efd207](https://github.com/fluxopt/lpspec/commit/8efd207c991fb956fcd516636ffe233ed6b29437))

## [0.0.1-alpha.86](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.85...v0.0.1-alpha.86) (2026-08-11)


### Features

* **language:** a coordinate may declare its own label space ([#557](https://github.com/fluxopt/lpspec/issues/557)) ([220fd8a](https://github.com/fluxopt/lpspec/commit/220fd8a54d6d66c9ac45c8a899bcda870b5a3307))
* **language:** a row with no variable terms is not built, and is reported ([#561](https://github.com/fluxopt/lpspec/issues/561)) ([4cf2ea5](https://github.com/fluxopt/lpspec/commit/4cf2ea5b19f14e00732169719f86126e097341cb)), closes [#556](https://github.com/fluxopt/lpspec/issues/556)

## [0.0.1-alpha.85](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.84...v0.0.1-alpha.85) (2026-08-10)


### Performance

* **engine:** the matrix stops repeating itself — CSR at assembly ([#550](https://github.com/fluxopt/lpspec/issues/550)) ([903c8c2](https://github.com/fluxopt/lpspec/commit/903c8c25ad41efe85b08fd6f8d7cfd375fb358b2))

## [0.0.1-alpha.84](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.83...v0.0.1-alpha.84) (2026-08-10)


### Features

* **language:** a comparison's constant side may not be sparse ([#554](https://github.com/fluxopt/lpspec/issues/554)) ([20cb72e](https://github.com/fluxopt/lpspec/commit/20cb72e4d841401b084766f9d666fa596a1ff8b8)), closes [#537](https://github.com/fluxopt/lpspec/issues/537) [#549](https://github.com/fluxopt/lpspec/issues/549)

## [0.0.1-alpha.83](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.82...v0.0.1-alpha.83) (2026-08-10)


### Bug Fixes

* **engine:** a zero edge writes its rows, like every other fill ([#551](https://github.com/fluxopt/lpspec/issues/551)) ([af1ce64](https://github.com/fluxopt/lpspec/commit/af1ce64802c8e270daef0e148e35ef8740502f82))

## [0.0.1-alpha.82](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.81...v0.0.1-alpha.82) (2026-08-10)


### Bug Fixes

* **engine:** a nested shift keeps its presence's own dims ([#546](https://github.com/fluxopt/lpspec/issues/546)) ([c7cb8a0](https://github.com/fluxopt/lpspec/commit/c7cb8a047f0e7f486297fbb54a56b346b3ca01ee)), closes [#544](https://github.com/fluxopt/lpspec/issues/544)

## [0.0.1-alpha.81](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.80...v0.0.1-alpha.81) (2026-08-10)


### Bug Fixes

* **language:** the bare-shift refusal names a pair, not three alternatives ([#540](https://github.com/fluxopt/lpspec/issues/540)) ([96bbbaa](https://github.com/fluxopt/lpspec/commit/96bbbaa29650dc0476e80f4408532f469c0242a7))


### Performance

* **engine:** a string dimension speaks its own Enum, everywhere at once ([#541](https://github.com/fluxopt/lpspec/issues/541)) ([8517945](https://github.com/fluxopt/lpspec/commit/8517945d603f17ce89f79c112eda6fc602968d16))

## [0.0.1-alpha.80](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.79...v0.0.1-alpha.80) (2026-08-10)


### Bug Fixes

* **api:** one exception tree, whichever door built the model ([#538](https://github.com/fluxopt/lpspec/issues/538)) ([52cdb8b](https://github.com/fluxopt/lpspec/commit/52cdb8b84d46167ac68e530c9ec71fe8ebf45184)), closes [#527](https://github.com/fluxopt/lpspec/issues/527)
* **linopy:** extend() passes the borrowed names to its own expansion ([#542](https://github.com/fluxopt/lpspec/issues/542)) ([282a2d3](https://github.com/fluxopt/lpspec/commit/282a2d34ce5dd7e084eeafb499a83df20e362e96))

## [0.0.1-alpha.79](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.78...v0.0.1-alpha.79) (2026-08-10)


### Bug Fixes

* **language:** a Model validates itself, however it was built ([#533](https://github.com/fluxopt/lpspec/issues/533)) ([6b987a6](https://github.com/fluxopt/lpspec/commit/6b987a6d1fae6a9c15793cf4cf4ffe1e6bbf09b0))

## [0.0.1-alpha.78](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.77...v0.0.1-alpha.78) (2026-08-10)


### Refactoring

* **api:** Model, and a model that can be written back out ([#522](https://github.com/fluxopt/lpspec/issues/522)) ([ccefd98](https://github.com/fluxopt/lpspec/commit/ccefd9863a4dd71ee0a25ac9f89bc5f484856581))

## [0.0.1-alpha.77](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.76...v0.0.1-alpha.77) (2026-08-09)


### Documentation

* **lp:** the presort's cost is unmeasured, not settled ([#517](https://github.com/fluxopt/lpspec/issues/517)) ([5ca09f0](https://github.com/fluxopt/lpspec/commit/5ca09f01b841a07b61e501f43120312cd50e5f56))

## [0.0.1-alpha.76](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.75...v0.0.1-alpha.76) (2026-08-09)


### Features

* **language:** version: on a model file, and 0 means unstable ([#515](https://github.com/fluxopt/lpspec/issues/515)) ([7d37589](https://github.com/fluxopt/lpspec/commit/7d375891768a2937fad5a26c7dad4759a367a53b))

## [0.0.1-alpha.75](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.74...v0.0.1-alpha.75) (2026-08-09)


### Refactoring

* **language:** edge takes a quoted keyword, edge='wrap' ([#512](https://github.com/fluxopt/lpspec/issues/512)) ([d20d44e](https://github.com/fluxopt/lpspec/commit/d20d44e73aded3f251637bd674f4a220f02ed36f))

## [0.0.1-alpha.74](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.73...v0.0.1-alpha.74) (2026-08-09)


### Documentation

* roll is not a spelling — shift(edge=wrap) is ([#509](https://github.com/fluxopt/lpspec/issues/509)) ([6f1f98a](https://github.com/fluxopt/lpspec/commit/6f1f98a3ab1994c8b0996a2f5e6f50025c183a0c))

## [0.0.1-alpha.73](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.72...v0.0.1-alpha.73) (2026-08-09)


### Performance

* **lp:** the constraint stream sorts on one key, carrying nothing else ([#492](https://github.com/fluxopt/lpspec/issues/492)) ([e7085ca](https://github.com/fluxopt/lpspec/commit/e7085ca0d15e987363366675e8968d1b8671e880))

## [0.0.1-alpha.72](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.71...v0.0.1-alpha.72) (2026-08-09)


### Bug Fixes

* **docs:** install uv on Read the Docs without asdf ([#501](https://github.com/fluxopt/lpspec/issues/501)) ([13e095e](https://github.com/fluxopt/lpspec/commit/13e095e2ebc9830cd900312dbe50eff95317b230))

## [0.0.1-alpha.71](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.70...v0.0.1-alpha.71) (2026-08-09)


### Refactoring

* **language:** group_sum becomes sum(group_by=) ([#491](https://github.com/fluxopt/lpspec/issues/491)) ([fb14dd2](https://github.com/fluxopt/lpspec/commit/fb14dd2b3c6fda6ef9504546ce859ae12543a142))

## [0.0.1-alpha.70](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.69...v0.0.1-alpha.70) (2026-08-09)


### Features

* **language:** at() — the pullback of group_sum ([#489](https://github.com/fluxopt/lpspec/issues/489)) ([e7ce036](https://github.com/fluxopt/lpspec/commit/e7ce036f9533e3ffd6dbb02110b1dfd35cb28c0a))


### Performance

* **polars:** two group_sums collide only where the coordinates meet, and the matrix leaves in row order ([#487](https://github.com/fluxopt/lpspec/issues/487)) ([53621b1](https://github.com/fluxopt/lpspec/commit/53621b1894119e845e4b2a75b1ba4abe4cb043da))

## [0.0.1-alpha.69](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.68...v0.0.1-alpha.69) (2026-08-09)


### Bug Fixes

* **language:** quoted literals in a where, and a dtype guard on comparisons ([#485](https://github.com/fluxopt/lpspec/issues/485)) ([968861b](https://github.com/fluxopt/lpspec/commit/968861b02bb242dabaf9f4df4229429d2076af17))

## [0.0.1-alpha.68](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.67...v0.0.1-alpha.68) (2026-08-09)


### Documentation

* **cli:** the typeset shell front is not a command line under construction ([#483](https://github.com/fluxopt/lpspec/issues/483)) ([72b8ced](https://github.com/fluxopt/lpspec/commit/72b8cedd25d83ba49dff8dc9e7465a85a30c4dd9))

## [0.0.1-alpha.67](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.66...v0.0.1-alpha.67) (2026-08-09)


### Bug Fixes

* **engine:** a masked-out scalar variable takes its row with it ([#480](https://github.com/fluxopt/lpspec/issues/480)) ([2e6f329](https://github.com/fluxopt/lpspec/commit/2e6f329f8cc7ae245fa66f07464a13e47ace2ca0))

## [0.0.1-alpha.66](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.65...v0.0.1-alpha.66) (2026-08-09)


### Documentation

* **roadmap:** motivation and end state; the detail moves to issues ([#474](https://github.com/fluxopt/lpspec/issues/474)) ([ebff8b0](https://github.com/fluxopt/lpspec/commit/ebff8b0eb7f5aec9eff32d8faca2d1bdf27a4484))

## [0.0.1-alpha.65](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.64...v0.0.1-alpha.65) (2026-08-09)


### Documentation

* **contributing:** issues cite behaviour, and two labels carry order ([#466](https://github.com/fluxopt/lpspec/issues/466)) ([86398fd](https://github.com/fluxopt/lpspec/commit/86398fdb77982a48db529db2469a11aae8c3435d))

## [0.0.1-alpha.64](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.63...v0.0.1-alpha.64) (2026-08-09)


### Documentation

* **models:** monthly_budget — group_sum over time, not just space ([#461](https://github.com/fluxopt/lpspec/issues/461)) ([ab8d71f](https://github.com/fluxopt/lpspec/commit/ab8d71f7f652e89703f18ba7a06e670be52637ed))

## [0.0.1-alpha.63](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.62...v0.0.1-alpha.63) (2026-08-09)


### Documentation

* **api:** Gurobi Compute Server, Instant Cloud and WLS already work ([#458](https://github.com/fluxopt/lpspec/issues/458)) ([4f18e89](https://github.com/fluxopt/lpspec/commit/4f18e89f96cf5c841113594549001d70bb7a0790))

## [0.0.1-alpha.62](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.61...v0.0.1-alpha.62) (2026-08-01)


### Documentation

* **bench:** plot the results, and fold the tables under them ([#451](https://github.com/fluxopt/lpspec/issues/451)) ([042c2ad](https://github.com/fluxopt/lpspec/commit/042c2ad78ca9e0d7fa9aab350e9cb82eb6367f41))

## [0.0.1-alpha.61](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.60...v0.0.1-alpha.61) (2026-07-31)


### Bug Fixes

* **bench:** the default arm clears the engine it did not select ([#452](https://github.com/fluxopt/lpspec/issues/452)) ([6e4e3a3](https://github.com/fluxopt/lpspec/commit/6e4e3a3c49ccb78e91e6a2d0caa88eeb65dbb055))

## [0.0.1-alpha.60](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.59...v0.0.1-alpha.60) (2026-07-31)


### Refactoring

* **bench:** the harness is pytest; drop the custom runner ([#448](https://github.com/fluxopt/lpspec/issues/448)) ([d460ca6](https://github.com/fluxopt/lpspec/commit/d460ca6fb330ba3153268e49ed1869374144b114))

## [0.0.1-alpha.59](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.58...v0.0.1-alpha.59) (2026-07-31)


### Performance

* **engine:** the read-back reads the label order rather than re-imposing it ([#446](https://github.com/fluxopt/lpspec/issues/446)) ([53f8eb8](https://github.com/fluxopt/lpspec/commit/53f8eb8781db1ba30c180db0b158628a22fb898d))

## [0.0.1-alpha.58](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.57...v0.0.1-alpha.58) (2026-07-31)


### Documentation

* **bench:** re-measure at 98f382d, and correct a claim the harness bug made ([#441](https://github.com/fluxopt/lpspec/issues/441)) ([6d0e195](https://github.com/fluxopt/lpspec/commit/6d0e1955c3fda307cb3edcb6f18b67c3026eb08c))

## [0.0.1-alpha.57](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.56...v0.0.1-alpha.57) (2026-07-31)


### Performance

* **sinks:** a solution is a vector, not a vector beside its own index ([#439](https://github.com/fluxopt/lpspec/issues/439)) ([76600ab](https://github.com/fluxopt/lpspec/commit/76600ab8f1685315c9f942b97bcc7e5b67f68d74))

## [0.0.1-alpha.56](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.55...v0.0.1-alpha.56) (2026-07-31)


### Bug Fixes

* **bench:** compare like with like, and add a gurobi arm ([#438](https://github.com/fluxopt/lpspec/issues/438)) ([ea15cb4](https://github.com/fluxopt/lpspec/commit/ea15cb43031f8b751bdd8a4abff162616fd4d5a9))

## [0.0.1-alpha.55](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.54...v0.0.1-alpha.55) (2026-07-31)


### Performance

* **engine:** cols is positional, and the order is produced rather than repaired ([#433](https://github.com/fluxopt/lpspec/issues/433)) ([22bb9c7](https://github.com/fluxopt/lpspec/commit/22bb9c7d944e360f874232ee7955d35ee545c19e))

## [0.0.1-alpha.54](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.53...v0.0.1-alpha.54) (2026-07-31)


### Performance

* **sinks:** a row is seated the way a column is, for both solvers ([#421](https://github.com/fluxopt/lpspec/issues/421)) ([31ba48c](https://github.com/fluxopt/lpspec/commit/31ba48cd94deb6559ddf305b6763656e3499f9b5))

## [0.0.1-alpha.53](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.52...v0.0.1-alpha.53) (2026-07-31)


### Performance

* **gurobi:** hand the matrix over in one call ([#434](https://github.com/fluxopt/lpspec/issues/434)) ([3d13d50](https://github.com/fluxopt/lpspec/commit/3d13d5093e00b650be05a8eedfe5e5adf48432ab))

## [0.0.1-alpha.52](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.51...v0.0.1-alpha.52) (2026-07-31)


### Features

* **sinks:** gurobi, and the two families it revealed ([#418](https://github.com/fluxopt/lpspec/issues/418)) ([8730498](https://github.com/fluxopt/lpspec/commit/8730498a6f9f38c09041d152ce39bc66c499a31f))

## [0.0.1-alpha.51](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.50...v0.0.1-alpha.51) (2026-07-31)


### Documentation

* breaking changes are free while we are 0.0.1aN ([#420](https://github.com/fluxopt/lpspec/issues/420)) ([ae1389a](https://github.com/fluxopt/lpspec/commit/ae1389aac964f7f3c1e094b541e80f8ab74bc39f))

## [0.0.1-alpha.50](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.49...v0.0.1-alpha.50) (2026-07-31)


### Performance

* **polars:** a constraint over distinct variables has nothing to collapse ([#408](https://github.com/fluxopt/lpspec/issues/408)) ([8e725de](https://github.com/fluxopt/lpspec/commit/8e725debe8cfe47ab2aac2b6bd35a04a6ce00472))
* **polars:** a fragment is not restricted by its own absence ([#413](https://github.com/fluxopt/lpspec/issues/413)) ([8ac3a0d](https://github.com/fluxopt/lpspec/commit/8ac3a0d6c1bcd5f5ff40b671d7faf358fb1e49e7))
* **polars:** a mask joins for what it is certain of ([#415](https://github.com/fluxopt/lpspec/issues/415)) ([68d95e2](https://github.com/fluxopt/lpspec/commit/68d95e2a11a162bac85338139af06f5156633514))
* **polars:** a solve is read back by position, not by key ([#414](https://github.com/fluxopt/lpspec/issues/414)) ([cba01c4](https://github.com/fluxopt/lpspec/commit/cba01c4563fb0094e04b409b2483266e06a9e741))
* **polars:** an existence join does not deduplicate what it asks about ([#412](https://github.com/fluxopt/lpspec/issues/412)) ([c008154](https://github.com/fluxopt/lpspec/commit/c00815437513dab83ae0ce1780170675ff709608))

## [0.0.1-alpha.49](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.48...v0.0.1-alpha.49) (2026-07-31)


### Bug Fixes

* **bench:** the build profiler names steps that still exist ([#406](https://github.com/fluxopt/lpspec/issues/406)) ([e771686](https://github.com/fluxopt/lpspec/commit/e7716863911578f85905a7be5667d7bbaf53c299))

## [0.0.1-alpha.48](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.47...v0.0.1-alpha.48) (2026-07-31)


### Bug Fixes

* **bench:** the regression harness builds again ([#400](https://github.com/fluxopt/lpspec/issues/400)) ([291f714](https://github.com/fluxopt/lpspec/commit/291f714c0138aaeff586be1ff6f76ea904e8ec13))

## [0.0.1-alpha.47](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.46...v0.0.1-alpha.47) (2026-07-31)


### Refactoring

* **relational:** the contract at the top, engines one level down ([#395](https://github.com/fluxopt/lpspec/issues/395)) ([fb4b2de](https://github.com/fluxopt/lpspec/commit/fb4b2de62b6d7607a95dae05d0b1f6b6df13e2bc))

## [0.0.1-alpha.46](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.45...v0.0.1-alpha.46) (2026-07-31)


### Features

* rename LinopyYamlError to LpspecError ([#394](https://github.com/fluxopt/lpspec/issues/394)) ([5f72731](https://github.com/fluxopt/lpspec/commit/5f72731d4a4c3a8203ac9977891f4518214e18ab))

## [0.0.1-alpha.45](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.44...v0.0.1-alpha.45) (2026-07-31)


### Documentation

* quadratic is planned, not refused — and the ceiling is relational ∩ local ([#388](https://github.com/fluxopt/lpspec/issues/388)) ([2a8c172](https://github.com/fluxopt/lpspec/commit/2a8c1721cf73552e55c5b1c39775bbe4cf7f02ec))

## [0.0.1-alpha.44](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.43...v0.0.1-alpha.44) (2026-07-31)


### Documentation

* fix five stale claims, and cut what git already remembers ([#386](https://github.com/fluxopt/lpspec/issues/386)) ([6562c15](https://github.com/fluxopt/lpspec/commit/6562c15044c2277e32914a9a9794ee6ab2ef3f02))

## [0.0.1-alpha.43](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.42...v0.0.1-alpha.43) (2026-07-31)


### Documentation

* **ceiling:** the plan cannot loop, the process can — scope the refusal ([#378](https://github.com/fluxopt/lpspec/issues/378)) ([55d238c](https://github.com/fluxopt/lpspec/commit/55d238c524d50d1980c6b367dd8a6e11c8c221f4))

## [0.0.1-alpha.42](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.41...v0.0.1-alpha.42) (2026-07-31)


### Documentation

* **architecture:** the consumers diagram carries the shape, not the list ([#377](https://github.com/fluxopt/lpspec/issues/377)) ([fb06b82](https://github.com/fluxopt/lpspec/commit/fb06b82cd2c00e8a1c764b860d78ce1a5e8e77d7))

## [0.0.1-alpha.41](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.40...v0.0.1-alpha.41) (2026-07-31)


### Documentation

* **architecture:** show the public surface, and pin it ([#375](https://github.com/fluxopt/lpspec/issues/375)) ([bdb6b11](https://github.com/fluxopt/lpspec/commit/bdb6b11c6a50d4657b0a525b7d20bbae1022be5c))

## [0.0.1-alpha.40](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.39...v0.0.1-alpha.40) (2026-07-31)


### Refactoring

* **architecture:** four directories, four enforced fences ([#373](https://github.com/fluxopt/lpspec/issues/373)) ([f5e2c09](https://github.com/fluxopt/lpspec/commit/f5e2c09bb1d1579d440d033b69002d9655f7e5e6))

## [0.0.1-alpha.39](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.38...v0.0.1-alpha.39) (2026-07-31)


### Refactoring

* **language:** the front end is a package, and degree 1 lives in it ([#371](https://github.com/fluxopt/lpspec/issues/371)) ([0bcfd1b](https://github.com/fluxopt/lpspec/commit/0bcfd1b2cd09fae66b9e25e5afd3aed520c569c3))
* **relational:** binding produces a value, and the live registry is visibly the exception ([#370](https://github.com/fluxopt/lpspec/issues/370)) ([70e13ba](https://github.com/fluxopt/lpspec/commit/70e13ba4a07f29bb97f1a0b620952118c6c9b0a3))

## [0.0.1-alpha.38](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.37...v0.0.1-alpha.38) (2026-07-31)


### Refactoring

* **relational:** labels and the result are modules, not regions of the executor ([#368](https://github.com/fluxopt/lpspec/issues/368)) ([7c8604a](https://github.com/fluxopt/lpspec/commit/7c8604ae814f1cfaaaaf512c0bb42b4f808bed56))
* three small simplifications in the eager lane and piecewise ([#366](https://github.com/fluxopt/lpspec/issues/366)) ([b8a5d3e](https://github.com/fluxopt/lpspec/commit/b8a5d3e56cd993f3b52505e0ed7e3469d087290f))

## [0.0.1-alpha.37](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.36...v0.0.1-alpha.37) (2026-07-31)


### Documentation

* **architecture:** the CLI ships, and the typeset spike is a package ([#363](https://github.com/fluxopt/lpspec/issues/363)) ([f0b6cf1](https://github.com/fluxopt/lpspec/commit/f0b6cf1b381e6d06bdb024c3a43e57596f3d2f61))

## [0.0.1-alpha.36](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.35...v0.0.1-alpha.36) (2026-07-31)


### Refactoring

* one near-miss clause, one fold, and comments that restate their line ([#361](https://github.com/fluxopt/lpspec/issues/361)) ([48464fc](https://github.com/fluxopt/lpspec/commit/48464fc79a37827f1d9b67bc502a4f1a94b346ae))

## [0.0.1-alpha.35](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.34...v0.0.1-alpha.35) (2026-07-31)


### Features

* **language:** one shift(along=, offset=, edge=), replacing roll and shift ([#359](https://github.com/fluxopt/lpspec/issues/359)) ([8473a24](https://github.com/fluxopt/lpspec/commit/8473a24621326eb39151fd50337f1c6decb7a51d))

## [0.0.1-alpha.34](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.33...v0.0.1-alpha.34) (2026-07-30)


### Performance

* **relational:** coordinate validation rides the aggregate already running ([#357](https://github.com/fluxopt/lpspec/issues/357)) ([093a694](https://github.com/fluxopt/lpspec/commit/093a694f6aa910a06b83e3a5ccfa81cf2d3d1f1d)), closes [#273](https://github.com/fluxopt/lpspec/issues/273)

## [0.0.1-alpha.33](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.32...v0.0.1-alpha.33) (2026-07-30)


### Refactoring

* **relational:** one data-validation module for this lane ([#355](https://github.com/fluxopt/lpspec/issues/355)) ([4b19fe4](https://github.com/fluxopt/lpspec/commit/4b19fe4576a8cb58f1b067f20bf8abf9a99dd813))

## [0.0.1-alpha.32](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.31...v0.0.1-alpha.32) (2026-07-30)


### Bug Fixes

* **relational:** refuse a source label the dimension does not have ([#352](https://github.com/fluxopt/lpspec/issues/352)) ([5600cd8](https://github.com/fluxopt/lpspec/commit/5600cd83b2fe4fa5529b7a9b4dd30aef164b60d5)), closes [#350](https://github.com/fluxopt/lpspec/issues/350)

## [0.0.1-alpha.31](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.30...v0.0.1-alpha.31) (2026-07-30)


### Bug Fixes

* **relational:** a mask survives a broadcast into a reduction ([#348](https://github.com/fluxopt/lpspec/issues/348)) ([3515fd6](https://github.com/fluxopt/lpspec/commit/3515fd619476b404697f611fbd655d6c633146f8))

## [0.0.1-alpha.30](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.29...v0.0.1-alpha.30) (2026-07-30)


### Bug Fixes

* **bench:** migrate the bench corpus off equations:, and gate it ([#346](https://github.com/fluxopt/lpspec/issues/346)) ([72818b2](https://github.com/fluxopt/lpspec/commit/72818b2a2eefef3e9afdce0387692f8d7545d896)), closes [#343](https://github.com/fluxopt/lpspec/issues/343)

## [0.0.1-alpha.29](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.28...v0.0.1-alpha.29) (2026-07-30)


### Documentation

* the corpus teaches shift for the acyclic boundary, and names what repeats ([#342](https://github.com/fluxopt/lpspec/issues/342)) ([586daa3](https://github.com/fluxopt/lpspec/commit/586daa345e1f225f33d3e46160e2ad8f406a5a8f))

## [0.0.1-alpha.28](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.27...v0.0.1-alpha.28) (2026-07-30)


### Bug Fixes

* **relational:** the empty coordinate — scalar rows, columns and values ([#339](https://github.com/fluxopt/lpspec/issues/339)) ([15c2892](https://github.com/fluxopt/lpspec/commit/15c28927626e48dab2b8ffa54eac4b1cf00f93d0))

## [0.0.1-alpha.27](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.26...v0.0.1-alpha.27) (2026-07-30)


### Refactoring

* rename the package to lpspec, drop the farkas branding ([#336](https://github.com/fluxopt/lpspec/issues/336)) ([0dd27d4](https://github.com/fluxopt/lpspec/commit/0dd27d4e4c20ae6111e24872d31a392b00885134))

## [0.0.1-alpha.26](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.25...v0.0.1-alpha.26) (2026-07-30)


### Bug Fixes

* **parser:** the inf literal needs a word boundary ([#335](https://github.com/fluxopt/lpspec/issues/335)) ([c34bf1a](https://github.com/fluxopt/lpspec/commit/c34bf1a17732a0f5824a4276ad6226faa7c4d0c6)), closes [#302](https://github.com/fluxopt/lpspec/issues/302)

## [0.0.1-alpha.25](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.24...v0.0.1-alpha.25) (2026-07-30)


### Documentation

* the ten laws at the head of SPEC, one evidence page, the ceiling as its own note ([#333](https://github.com/fluxopt/lpspec/issues/333)) ([71ae5f7](https://github.com/fluxopt/lpspec/commit/71ae5f70b9f76bcdb9c47cf436cf63a232801c0a))

## [0.0.1-alpha.24](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.23...v0.0.1-alpha.24) (2026-07-29)


### Features

* one rule per constraint, named by the file rather than by position ([#329](https://github.com/fluxopt/lpspec/issues/329)) ([1ff53b3](https://github.com/fluxopt/lpspec/commit/1ff53b3cd62bec0779b53bbb2af5720e9ffc303f))

## [0.0.1-alpha.23](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.22...v0.0.1-alpha.23) (2026-07-29)


### Bug Fixes

* **api:** read-back by an unknown name says what the model actually built ([#327](https://github.com/fluxopt/lpspec/issues/327)) ([018b1ff](https://github.com/fluxopt/lpspec/commit/018b1ff83a706fe4875d1693fdbab612d36d2cb8))

## [0.0.1-alpha.22](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.21...v0.0.1-alpha.22) (2026-07-29)


### Documentation

* **spec:** when a fill belongs in the language, and when it is data prep ([#325](https://github.com/fluxopt/lpspec/issues/325)) ([5f71da8](https://github.com/fluxopt/lpspec/commit/5f71da8deae76011f6d2afd3af685ddb88196cca))

## [0.0.1-alpha.21](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.20...v0.0.1-alpha.21) (2026-07-29)


### Documentation

* the home page shows the model it sets, as math ([#317](https://github.com/fluxopt/lpspec/issues/317)) ([4094fb9](https://github.com/fluxopt/lpspec/commit/4094fb91f30cd9595acbca131882d34e54b301ee))
* **typeset:** say that `symbols=` takes a dict, not only a path ([#323](https://github.com/fluxopt/lpspec/issues/323)) ([e9f2102](https://github.com/fluxopt/lpspec/commit/e9f210258732b7bf249e6465b27b0e6c32a6087f))

## [0.0.1-alpha.20](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.19...v0.0.1-alpha.20) (2026-07-29)


### Bug Fixes

* **linopy:** refuse a missing bound at build, with the native lane's message ([#319](https://github.com/fluxopt/lpspec/issues/319)) ([0e75754](https://github.com/fluxopt/lpspec/commit/0e75754d21a3092160423364ae2ed64fbd6a5d92))

## [0.0.1-alpha.19](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.18...v0.0.1-alpha.19) (2026-07-29)


### Bug Fixes

* a divisor must be defined where the model divides by it ([#318](https://github.com/fluxopt/lpspec/issues/318)) ([d7ad809](https://github.com/fluxopt/lpspec/commit/d7ad80903362fa89a91625ec2d781e4dad0f7621))
* **relational:** absence propagates into a reduction, and laws to hold it there ([#314](https://github.com/fluxopt/lpspec/issues/314)) ([f4bb63b](https://github.com/fluxopt/lpspec/commit/f4bb63b7584586d6f87c217216159d600847dc77))

## [0.0.1-alpha.18](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.17...v0.0.1-alpha.18) (2026-07-29)


### Documentation

* typeset every model page from the model it shows ([#280](https://github.com/fluxopt/lpspec/issues/280)) ([d6eda33](https://github.com/fluxopt/lpspec/commit/d6eda33eafe5540c2a68cd73a4a050e150c4833d))

## [0.0.1-alpha.17](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.16...v0.0.1-alpha.17) (2026-07-29)


### Features

* typeset a validated model — LaTeX and Typst (spike) ([#269](https://github.com/fluxopt/lpspec/issues/269)) ([3cbd79a](https://github.com/fluxopt/lpspec/commit/3cbd79acdd63237f047b548fddc99f834f5f7dcc))


### Bug Fixes

* **ci:** the PR-title check must report on every head, or it blocks the merge ([#308](https://github.com/fluxopt/lpspec/issues/308)) ([ecbdb13](https://github.com/fluxopt/lpspec/commit/ecbdb133ac5e13df2150b6f018ab92dc5ebadac2))

## [0.0.1-alpha.16](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.15...v0.0.1-alpha.16) (2026-07-29)


### Documentation

* **releasing:** the alpha stream is 0.0.1, and say what actually holds it ([#306](https://github.com/fluxopt/lpspec/issues/306)) ([de597a6](https://github.com/fluxopt/lpspec/commit/de597a6b369c614e75db4752ba697292478e4155))

## [0.0.1-alpha.15](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.14...v0.0.1-alpha.15) (2026-07-29)


### Bug Fixes

* **ci:** check out the manifest the version guard reads ([#303](https://github.com/fluxopt/lpspec/issues/303)) ([911b9ed](https://github.com/fluxopt/lpspec/commit/911b9ed29c0c9c1e358d1d9fc861c52f03b621f4))

## [0.0.1-alpha.14](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.13...v0.0.1-alpha.14) (2026-07-29)


### Documentation

* **models:** fold long expressions onto one term per line ([#299](https://github.com/fluxopt/lpspec/issues/299)) ([826eaaa](https://github.com/fluxopt/lpspec/commit/826eaaa1958bd6f32967959607af0c9fe7f7d700))

## [0.0.1-alpha.13](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.12...v0.0.1-alpha.13) (2026-07-29)


### ⚠ BREAKING CHANGES

* **linopy:** importing `farkas.linopy` sets `linopy.options['semantics']` to 'v1' process-wide. Models built through this lane change answer wherever a variable is masked or shifted — that is the point, and it is the answer the native engine has always given.
* a bare `shift()` now drops the row at the vacated coordinate instead of contributing zero there, and `shift()` over a variable-free expression is a load error. Add `fill=0` for the previous behaviour.

### Features

* shift() creates absence, as linopy v1 says it does ([#291](https://github.com/fluxopt/lpspec/issues/291)) ([b394e70](https://github.com/fluxopt/lpspec/commit/b394e700c2b45b1eb06d434e95a441d64f6255d9))


### Bug Fixes

* **linopy:** the lane selects v1 on import, and fill= takes the identity of its position ([#293](https://github.com/fluxopt/lpspec/issues/293)) ([451c354](https://github.com/fluxopt/lpspec/commit/451c3548e6dc88e851d1f51e6c69f9be13a4220e))
* **release:** put the version back on the alpha stream, and make the pin real ([#295](https://github.com/fluxopt/lpspec/issues/295)) ([8ef60a5](https://github.com/fluxopt/lpspec/commit/8ef60a57d3a4fecb649bdb913b651a8396916202))

## [0.0.1-alpha.12](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.11...v0.0.1-alpha.12) (2026-07-29)


### Documentation

* **spec:** three behaviours a consumer could only reach by probing ([#288](https://github.com/fluxopt/lpspec/issues/288)) ([50b087c](https://github.com/fluxopt/lpspec/commit/50b087c51428770ec476a29349546cbb6f0a948e))

## [0.0.1-alpha.11](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.10...v0.0.1-alpha.11) (2026-07-29)


### Bug Fixes

* **docs:** readable mermaid diagrams in dark mode, and the site icons ([#285](https://github.com/fluxopt/lpspec/issues/285)) ([b21d5d8](https://github.com/fluxopt/lpspec/commit/b21d5d8aea094b17713d0cb3547d63c73f89e596))

## [0.0.1-alpha.10](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.9...v0.0.1-alpha.10) (2026-07-28)


### Documentation

* publish the site with mkdocs-material ([#283](https://github.com/fluxopt/lpspec/issues/283)) ([1ec55ac](https://github.com/fluxopt/lpspec/commit/1ec55ac5af2467ba0cfec58cafc7d84b6cddc467))

## [0.0.1-alpha.9](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.8...v0.0.1-alpha.9) (2026-07-28)


### Bug Fixes

* **result:** read the solution back in label order ([#278](https://github.com/fluxopt/lpspec/issues/278)) ([11bf86d](https://github.com/fluxopt/lpspec/commit/11bf86d8e3c0e5dd02e624b597ae3c4d0aff951b))

## [0.0.1-alpha.8](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.7...v0.0.1-alpha.8) (2026-07-28)


### Bug Fixes

* **piecewise:** emit foreach in declaration order, not set order ([#271](https://github.com/fluxopt/lpspec/issues/271)) ([c06907b](https://github.com/fluxopt/lpspec/commit/c06907b0ef5e634542f5f52960a9488271f1bcf7))
* **where:** a declared dimension on a comparison RHS is a load error ([#272](https://github.com/fluxopt/lpspec/issues/272)) ([5b1784e](https://github.com/fluxopt/lpspec/commit/5b1784ee1271f0cb88ff1e4657940ff526aee19a))

## [0.0.1-alpha.7](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.6...v0.0.1-alpha.7) (2026-07-28)


### Documentation

* the AST as a narrow waist — one contract, many consumers ([#266](https://github.com/fluxopt/lpspec/issues/266)) ([688c6df](https://github.com/fluxopt/lpspec/commit/688c6df1f3d35bcc5487b3910dfa06fdcd191f22))

## [0.0.1-alpha.6](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.5...v0.0.1-alpha.6) (2026-07-28)


### Documentation

* say what examples/ is, and point it at the explained version ([#263](https://github.com/fluxopt/lpspec/issues/263)) ([aab7617](https://github.com/fluxopt/lpspec/commit/aab7617b137b9b88f72084142fa5ad558074d6e2))

## [0.0.1-alpha.5](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.4...v0.0.1-alpha.5) (2026-07-28)


### Documentation

* move the three root docs, add a guide, make the index a path ([#260](https://github.com/fluxopt/lpspec/issues/260)) ([bfbfbec](https://github.com/fluxopt/lpspec/commit/bfbfbecb7a1291f1bca6dd110b621722806f25fb))

## [0.0.1-alpha.4](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.3...v0.0.1-alpha.4) (2026-07-28)


### Documentation

* a model gallery, with a construct matrix read off the plan ([#257](https://github.com/fluxopt/lpspec/issues/257)) ([e9cf410](https://github.com/fluxopt/lpspec/commit/e9cf410b6d81a774d18efa13894e9dc5ad359a43))

## [0.0.1-alpha.3](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.2...v0.0.1-alpha.3) (2026-07-28)


### Documentation

* a CONTRIBUTING.md, and move procedure out of the corpus page ([#255](https://github.com/fluxopt/lpspec/issues/255)) ([d3d5f96](https://github.com/fluxopt/lpspec/commit/d3d5f9652944e6915f2be71a73ee7c8e45a070fe))

## [0.0.1-alpha.2](https://github.com/fluxopt/lpspec/compare/v0.0.1-alpha.1...v0.0.1-alpha.2) (2026-07-28)


### Refactoring

* **bench:** retire the duckdb arm, and the docs it was holding up ([#253](https://github.com/fluxopt/lpspec/issues/253)) ([6eaef1e](https://github.com/fluxopt/lpspec/commit/6eaef1e86999bb4047e1215a6c617d641f143dbe))

## [0.0.1-alpha.1](https://github.com/fluxopt/lpspec/compare/v0.0.0-alpha.33...v0.0.1-alpha.1) (2026-07-28)


### ⚠ BREAKING CHANGES

* rebuild the relational engine on polars ([#189](https://github.com/fluxopt/lpspec/issues/189))

### Features

* rebuild the relational engine on polars ([#189](https://github.com/fluxopt/lpspec/issues/189)) ([c11a0dd](https://github.com/fluxopt/lpspec/commit/c11a0dd3e8ed58bf8c1a8ea9b79961733e381eb1))


### Chores

* re-cut as 0.0.1-alpha.1, and refresh the chart page from the results ([#250](https://github.com/fluxopt/lpspec/issues/250)) ([f62d434](https://github.com/fluxopt/lpspec/commit/f62d43446f21267a1655283ae3467fdae420ad51))

## [0.0.0-alpha.33](https://github.com/fluxopt/lpspec/compare/v0.0.0-alpha.32...v0.0.0-alpha.33) (2026-07-28)


### ⚠ BREAKING CHANGES

* absence propagates and drops the row, plus defined(v) ([#239](https://github.com/fluxopt/lpspec/issues/239))

### Features

* absence propagates and drops the row, plus defined(v) ([#239](https://github.com/fluxopt/lpspec/issues/239)) ([5eb5943](https://github.com/fluxopt/lpspec/commit/5eb5943bc1f83ff4451e1dbfa526d318f7cf7746))

## [0.0.0-alpha.32](https://github.com/fluxopt/lpspec/compare/v0.0.0-alpha.31...v0.0.0-alpha.32) (2026-07-28)


### Features

* **compat:** answer linopy's v1 convention where the position knows the answer ([#236](https://github.com/fluxopt/lpspec/issues/236)) ([facee3f](https://github.com/fluxopt/lpspec/commit/facee3f250096724de20ffd6e0aa215dd8eb6534)), closes [#8](https://github.com/fluxopt/lpspec/issues/8)

## [0.0.0-alpha.31](https://github.com/fluxopt/lpspec/compare/v0.0.0-alpha.30...v0.0.0-alpha.31) (2026-07-28)


### Documentation

* alpha.30 credited a perf fix it does not contain ([#230](https://github.com/fluxopt/lpspec/issues/230)) ([5afb004](https://github.com/fluxopt/lpspec/commit/5afb004cfb5ede08b0a203bde47f19ec54607476))

## [0.0.0-alpha.30](https://github.com/fluxopt/lpspec/compare/v0.0.0-alpha.29...v0.0.0-alpha.30) (2026-07-28)


### Documentation

* pin all three label paths, and write the row-major order down ([#152](https://github.com/fluxopt/lpspec/issues/152)) ([3d8c725](https://github.com/fluxopt/lpspec/commit/3d8c7251bc210bbc4438382ec58c3b699683624a))

*No functional change: this release is alpha.29 plus tests and a paragraph of
`docs/ARCHITECTURE.md`. The commit subject says `perf:` and claims a speed-up, which
is wrong — that optimisation shipped in alpha.29 as
[#178](https://github.com/fluxopt/lpspec/issues/178). #152 was retitled after it
had already been merged and released, too late for the squash subject.*

## [0.0.0-alpha.29](https://github.com/fluxopt/lpspec/compare/v0.0.0-alpha.28...v0.0.0-alpha.29) (2026-07-28)


### Performance

* compute variable labels arithmetically when the mask factors ([#178](https://github.com/fluxopt/lpspec/issues/178)) ([5d4745f](https://github.com/fluxopt/lpspec/commit/5d4745ff81179cf5f65afba4ffdc97b46829803f))

## [0.0.0-alpha.28](https://github.com/fluxopt/lpspec/compare/v0.0.0-alpha.27...v0.0.0-alpha.28) (2026-07-28)


### Bug Fixes

* three ways an objective or a scalar parameter answered quietly ([#223](https://github.com/fluxopt/lpspec/issues/223)) ([ee08d5d](https://github.com/fluxopt/lpspec/commit/ee08d5d39dde4682b795d62f623d606d8996979a))

## [0.0.0-alpha.27](https://github.com/fluxopt/lpspec/compare/v0.0.0-alpha.26...v0.0.0-alpha.27) (2026-07-28)


### Bug Fixes

* refuse a parameter source that carries a coordinate twice ([#201](https://github.com/fluxopt/lpspec/issues/201)) ([fffee97](https://github.com/fluxopt/lpspec/commit/fffee9773f259a9b75cf5014f50a02f40c02a3cb))

## [0.0.0-alpha.26](https://github.com/fluxopt/lpspec/compare/v0.0.0-alpha.25...v0.0.0-alpha.26) (2026-07-27)


### Documentation

* **bench:** correct the label-optimisation numbers, and say how to get them ([#188](https://github.com/fluxopt/lpspec/issues/188)) ([bf84fd1](https://github.com/fluxopt/lpspec/commit/bf84fd1c1eb1ad6388d96b26827f83c4de3712c5))

## [0.0.0-alpha.25](https://github.com/fluxopt/lpspec/compare/v0.0.0-alpha.24...v0.0.0-alpha.25) (2026-07-27)


### Performance

* **sink:** size the hand-off by nonzeros, under one budget and one chunking rule ([#195](https://github.com/fluxopt/lpspec/issues/195)) ([c31beaf](https://github.com/fluxopt/lpspec/commit/c31beafdbe95e48a902dd19fa4b27de27b9bc4c0))

## [0.0.0-alpha.24](https://github.com/fluxopt/lpspec/compare/v0.0.0-alpha.23...v0.0.0-alpha.24) (2026-07-27)


### Performance

* **sink:** hand Arrow to numpy directly, not through to_pydict ([#193](https://github.com/fluxopt/lpspec/issues/193)) ([ac64f26](https://github.com/fluxopt/lpspec/commit/ac64f261239326e48a0c441774beedabb908a46f))

## [0.0.0-alpha.23](https://github.com/fluxopt/lpspec/compare/v0.0.0-alpha.22...v0.0.0-alpha.23) (2026-07-27)


### Performance

* skip the objective's GROUP BY where a column cannot repeat ([#179](https://github.com/fluxopt/lpspec/issues/179)) ([787b1ce](https://github.com/fluxopt/lpspec/commit/787b1ce238861acef53049a4ac0135671d850f1a))

## [0.0.0-alpha.22](https://github.com/fluxopt/lpspec/compare/v0.0.0-alpha.21...v0.0.0-alpha.22) (2026-07-27)


### Performance

* **lp:** render doubles with ::VARCHAR instead of printf('%.17g') ([#190](https://github.com/fluxopt/lpspec/issues/190)) ([b1df538](https://github.com/fluxopt/lpspec/commit/b1df538c3c0c4e0195d7d2d5ff2be7fb887608f1))

## [0.0.0-alpha.21](https://github.com/fluxopt/lpspec/compare/v0.0.0-alpha.20...v0.0.0-alpha.21) (2026-07-27)


### Performance

* **relational:** a label is a position, so compute it instead of counting it ([#186](https://github.com/fluxopt/lpspec/issues/186)) ([a31645b](https://github.com/fluxopt/lpspec/commit/a31645bde58974a041f175a18c56e3251bf6d5cd))

## [0.0.0-alpha.20](https://github.com/fluxopt/lpspec/compare/v0.0.0-alpha.19...v0.0.0-alpha.20) (2026-07-27)


### Bug Fixes

* the HiGHS sink's column ingest was an unbounded global sort ([#181](https://github.com/fluxopt/lpspec/issues/181)) ([271b0d7](https://github.com/fluxopt/lpspec/commit/271b0d7254c3cb0b94e740542071185ad0608b8d))

## [0.0.0-alpha.19](https://github.com/fluxopt/lpspec/compare/v0.0.0-alpha.18...v0.0.0-alpha.19) (2026-07-27)


### Features

* expose duals on the solve path (sol.dual) ([#156](https://github.com/fluxopt/lpspec/issues/156)) ([284df79](https://github.com/fluxopt/lpspec/commit/284df7927c6f58064fc691e9b8c2f1b2db2f6a7b))

## [0.0.0-alpha.18](https://github.com/fluxopt/lpspec/compare/v0.0.0-alpha.17...v0.0.0-alpha.18) (2026-07-27)


### Features

* **bench:** a performance harness the published numbers come from ([#143](https://github.com/fluxopt/lpspec/issues/143)) ([144713a](https://github.com/fluxopt/lpspec/commit/144713a6a2b32dc63efae52ce2853ce266d77191))

## [0.0.0-alpha.17](https://github.com/fluxopt/lpspec/compare/v0.0.0-alpha.16...v0.0.0-alpha.17) (2026-07-27)


### Features

* forward solver_options, and gate reads on an actual incumbent ([#169](https://github.com/fluxopt/lpspec/issues/169)) ([493a5e6](https://github.com/fluxopt/lpspec/commit/493a5e6f9d26184e3f1b855d963692343a5bf680))

## [0.0.0-alpha.16](https://github.com/fluxopt/lpspec/compare/v0.0.0-alpha.15...v0.0.0-alpha.16) (2026-07-27)


### Documentation

* the memory invariant says what it actually guarantees ([#150](https://github.com/fluxopt/lpspec/issues/150)) ([8ffe339](https://github.com/fluxopt/lpspec/commit/8ffe33929c29ba28e87ccbd60b4f1005a89a6132))

## [0.0.0-alpha.15](https://github.com/fluxopt/lpspec/compare/v0.0.0-alpha.14...v0.0.0-alpha.15) (2026-07-27)


### ⚠ BREAKING CHANGES

* documentation only — the suggested alias is now `fk`. `import farkas` was and remains the actual import.

### Chores

* the import alias is fk, because the package is farkas ([#154](https://github.com/fluxopt/lpspec/issues/154)) ([fac76f0](https://github.com/fluxopt/lpspec/commit/fac76f04597e38675f136edf2d8f0bd5d74c85a7))

## [0.0.0-alpha.14](https://github.com/fluxopt/lpspec/compare/v0.0.0-alpha.13...v0.0.0-alpha.14) (2026-07-27)


### ⚠ BREAKING CHANGES

* `Solution` is now `Result`; `status` returns the coarse axis (`ok`) rather than the solver's wording (`Optimal`) — `termination_condition` carries that, and `is_ok` is what most call sites meant. `objective` is `nan` and `primal`/`to_*` raise `NoSolutionError` when the solve produced nothing.

### Features

* a solve result tells you whether it has one ([#148](https://github.com/fluxopt/lpspec/issues/148)) ([30a91c8](https://github.com/fluxopt/lpspec/commit/30a91c853913bb7dee9e35323624b26f45ef5a45))

## [0.0.0-alpha.13](https://github.com/fluxopt/lpspec/compare/v0.0.0-alpha.12...v0.0.0-alpha.13) (2026-07-26)


### Bug Fixes

* quote caller-supplied paths in SQL ([#139](https://github.com/fluxopt/lpspec/issues/139)) ([61dfe5b](https://github.com/fluxopt/lpspec/commit/61dfe5b487585fa7acd68cc76ce5d75e4bd42d98))

## [0.0.0-alpha.12](https://github.com/fluxopt/lpspec/compare/v0.0.0-alpha.11...v0.0.0-alpha.12) (2026-07-26)


### Bug Fixes

* a null coordinate means "no group", not a typo ([#135](https://github.com/fluxopt/lpspec/issues/135)) ([9a672b4](https://github.com/fluxopt/lpspec/commit/9a672b42be9753a7e69ff0f6b76948a0352461a0))

## [0.0.0-alpha.11](https://github.com/fluxopt/lpspec/compare/v0.0.0-alpha.10...v0.0.0-alpha.11) (2026-07-26)


### Documentation

* lead with what the package is; YAML is the format we ship, not the interface ([#132](https://github.com/fluxopt/lpspec/issues/132)) ([5bd5240](https://github.com/fluxopt/lpspec/commit/5bd52400a9dd2a9e94eb3f29a2f0ebb6466c78ba))

## [0.0.0-alpha.10](https://github.com/fluxopt/lpspec/compare/v0.0.0-alpha.9...v0.0.0-alpha.10) (2026-07-26)


### Bug Fixes

* py.typed ships with the package it describes ([#131](https://github.com/fluxopt/lpspec/issues/131)) ([52529de](https://github.com/fluxopt/lpspec/commit/52529de80f4dd2c713c2fac2437855eee865f480))

## [0.0.0-alpha.9](https://github.com/fluxopt/lpspec/compare/v0.0.0-alpha.8...v0.0.0-alpha.9) (2026-07-26)


### ⚠ BREAKING CHANGES

* the import path is now `farkas`; the compat lane is `farkas.linopy` and its extra is `[linopy]`.

### Refactoring

* rename the package to farkas, and compat/ to linopy/ ([#127](https://github.com/fluxopt/lpspec/issues/127)) ([5fae345](https://github.com/fluxopt/lpspec/commit/5fae345de3358f227e3a04da91f205982a1af4ce))

## [0.0.0-alpha.8](https://github.com/fluxopt/lpspec/compare/v0.0.0-alpha.7...v0.0.0-alpha.8) (2026-07-26)


### Refactoring

* one lazy import left, and it is the only real cycle ([#117](https://github.com/fluxopt/lpspec/issues/117)) ([ecad711](https://github.com/fluxopt/lpspec/commit/ecad71113f561f231388045dcdbe2b5b85fde416))

## [0.0.0-alpha.7](https://github.com/fluxopt/lpspec/compare/v0.0.0-alpha.6...v0.0.0-alpha.7) (2026-07-26)


### Refactoring

* the package moves under src/, so CI tests the artifact ([#118](https://github.com/fluxopt/lpspec/issues/118)) ([7eb39a6](https://github.com/fluxopt/lpspec/commit/7eb39a6229e86cd446fe476078de7bf8ef13c4ca))

## [0.0.0-alpha.6](https://github.com/fluxopt/lpspec/compare/v0.0.0-alpha.5...v0.0.0-alpha.6) (2026-07-26)


### Features

* accept any Arrow-compatible table as a source ([#104](https://github.com/fluxopt/lpspec/issues/104)) ([e8a699e](https://github.com/fluxopt/lpspec/commit/e8a699ede9cccc4bb688c7ed519c98c38d0992c3))


### Refactoring

* three modules in the engine, one per box in the diagram ([#107](https://github.com/fluxopt/lpspec/issues/107)) ([f6b30b2](https://github.com/fluxopt/lpspec/commit/f6b30b241cf2d438e7072200dfc169046a0c2d31))


### Documentation

* the seam's level is decided, so stop pointing at an open issue ([#111](https://github.com/fluxopt/lpspec/issues/111)) ([7d1a992](https://github.com/fluxopt/lpspec/commit/7d1a9929626ba8f48f56fac5bfad12770bed1d6e))

## [0.0.0-alpha.5](https://github.com/fluxopt/lpspec/compare/v0.0.0-alpha.4...v0.0.0-alpha.5) (2026-07-26)


### ⚠ BREAKING CHANGES

* a file declaring more than one objective no longer loads.
* a Series or DataArray whose index names are not the declared dims is now a DataError; previously the names were discarded.
* a dimension declares the coordinates its labels carry ([#100](https://github.com/fluxopt/lpspec/issues/100))
* every IR, AST and schema class is renamed, and `fk.LanguageError` no longer covers data-binding failures — those are `fk.DataError`. Both remain under `fk.LinopyYamlError`, as does the deprecated `RelationalBuildError` alias.

### Features

* a dimension declares the coordinates its labels carry ([#100](https://github.com/fluxopt/lpspec/issues/100)) ([49b790b](https://github.com/fluxopt/lpspec/commit/49b790b5bfa36e30dff617db8bda02876ad51757))


### Bug Fixes

* a bool parameter is a mask, on both lanes ([#47](https://github.com/fluxopt/lpspec/issues/47)) ([#96](https://github.com/fluxopt/lpspec/issues/96)) ([a3f3926](https://github.com/fluxopt/lpspec/commit/a3f39261500323329b2b902cf36e7ffbcb59ce5e))
* a declared coordinate must be its declared dtype ([#65](https://github.com/fluxopt/lpspec/issues/65)) ([#101](https://github.com/fluxopt/lpspec/issues/101)) ([5349991](https://github.com/fluxopt/lpspec/commit/5349991cfbdd7f3fe8da302a0a570573314395d2))
* a named index binds by name, not by position ([#91](https://github.com/fluxopt/lpspec/issues/91)) ([#98](https://github.com/fluxopt/lpspec/issues/98)) ([3e14be8](https://github.com/fluxopt/lpspec/commit/3e14be8a445d38d60cb664d04c033ce0dc2ddb42))
* a second objective is a load error, not a silent drop ([#49](https://github.com/fluxopt/lpspec/issues/49)) ([#97](https://github.com/fluxopt/lpspec/issues/97)) ([24f5849](https://github.com/fluxopt/lpspec/commit/24f58494bcf378018aed6c1c3c5f4f2bb96d3c06))
* one place per language rule, and formulations judged like the rest ([#99](https://github.com/fluxopt/lpspec/issues/99)) ([3137fbc](https://github.com/fluxopt/lpspec/commit/3137fbcc21b176392058c28ec48134b33cae9782))


### Refactoring

* names say what they mean, and errors are one tree ([#94](https://github.com/fluxopt/lpspec/issues/94)) ([0c1300c](https://github.com/fluxopt/lpspec/commit/0c1300c56f951448f8d765df6e2054c811bf57ac))
* the compat lane is a directory, so the fence is structural ([#95](https://github.com/fluxopt/lpspec/issues/95)) ([4015131](https://github.com/fluxopt/lpspec/commit/40151315e7dec4e489d0e7f0db03a24b5d1aba95))


### Documentation

* consolidate the doc set and cut it by two thirds ([#87](https://github.com/fluxopt/lpspec/issues/87)) ([0a89d7d](https://github.com/fluxopt/lpspec/commit/0a89d7d908df6e89bb987a3cf6a799a95ed61fa4))
* runnable architecture walkthrough, one stage at a time ([#54](https://github.com/fluxopt/lpspec/issues/54)) ([0f0910f](https://github.com/fluxopt/lpspec/commit/0f0910fc90e97eb918b06f53268b79aae6efa0d3))
* say plainly that breaking changes land without a deprecation cycle ([#102](https://github.com/fluxopt/lpspec/issues/102)) ([6131342](https://github.com/fluxopt/lpspec/commit/6131342ed345776239ba60709a462746e2140f93))
* split sink capability from the expressive ceiling ([#88](https://github.com/fluxopt/lpspec/issues/88)) ([4e58227](https://github.com/fluxopt/lpspec/commit/4e582271eb826be6dda353ef0e4c3786c8a2a4ab))
* the composition seam exists, and value-only re-solve has a precondition ([#93](https://github.com/fluxopt/lpspec/issues/93)) ([2dffd89](https://github.com/fluxopt/lpspec/commit/2dffd89250d1a3f68215e398ddca984d72a0705d))

## [0.0.0-alpha.4](https://github.com/fluxopt/lpspec/compare/v0.0.0-alpha.3...v0.0.0-alpha.4) (2026-07-25)


### ⚠ BREAKING CHANGES

* unknown YAML keys are an error, not a silent default ([#72](https://github.com/fluxopt/lpspec/issues/72))
* sum/roll/shift/group_sum over a dim the operand does not carry, a where dim outside the frame, a bound parameter dim outside foreach, and a constraint whose expression dims differ from its foreach are all load errors. Each previously built a model that solved and was wrong, or larger than the file read as.

### Features

* Result.to_xarray() — the labelled form, one call away ([#75](https://github.com/fluxopt/lpspec/issues/75)) ([7df73b4](https://github.com/fluxopt/lpspec/commit/7df73b4e75a1d1656b4b1a1d928d0e4bf5814a99))
* static dim checking — the type is a set of dim names ([#68](https://github.com/fluxopt/lpspec/issues/68)) ([f96bcb4](https://github.com/fluxopt/lpspec/commit/f96bcb4f12797514ef93afd1cd8e771cf8490d0b))


### Bug Fixes

* dim checking runs on both lanes, and binary ops union ([#70](https://github.com/fluxopt/lpspec/issues/70)) ([2072cfa](https://github.com/fluxopt/lpspec/commit/2072cfae4cfb42c3664d6d4a1e9eb3171e6cfb51))
* read YAML 1.2 booleans, and refuse duplicate keys ([#77](https://github.com/fluxopt/lpspec/issues/77)) ([b12af91](https://github.com/fluxopt/lpspec/commit/b12af91953f4730e0a9beb74643214bb52dd62d5))
* unknown YAML keys are an error, not a silent default ([#72](https://github.com/fluxopt/lpspec/issues/72)) ([909bc4c](https://github.com/fluxopt/lpspec/commit/909bc4c120d2a1ca341917971dbfb7851af35553))


### Documentation

* an objective totals its dims, so the examples stop pretending otherwise ([#74](https://github.com/fluxopt/lpspec/issues/74)) ([f090c8c](https://github.com/fluxopt/lpspec/commit/f090c8c6b214d633b2e14a34fa2aee3c8ff9a1e5))
* the axes the expressiveness taxonomy does not rank ([#76](https://github.com/fluxopt/lpspec/issues/76)) ([bcfcdee](https://github.com/fluxopt/lpspec/commit/bcfcdeebc6eb7a17e53076c5d75f7d0539d6bbf1))

## [0.0.0-alpha.3](https://github.com/fluxopt/lpspec/compare/v0.0.0-alpha.2...v0.0.0-alpha.3) (2026-07-25)


### ⚠ BREAKING CHANGES

* `where: "<dimension>"` is a load error. It never did anything except in the case where it broke.
* an unknown name in a where string is a load error rather than a False mask; parameter-vs-parameter where comparisons are rejected; and names may no longer collide across kinds. Each was a way to build a model that solved and was silently wrong.
* a variable declared without `bounds.lower` was silently non-negative; it is now unbounded below, matching linopy's `add_variables(lower=-inf)`. Models relying on the implicit `>= 0` must write `lower: 0`. An LP that was bounded only by that implicit constraint will now report as unbounded.

### Bug Fixes

* a bare dimension name in a where is a load error ([#64](https://github.com/fluxopt/lpspec/issues/64)) ([1a89fef](https://github.com/fluxopt/lpspec/commit/1a89fef19e9d301141e8bd459c89ef2e2255b554))
* both lanes agree on where-comparisons over dimensions, and on `**` ([#52](https://github.com/fluxopt/lpspec/issues/52)) ([7bec431](https://github.com/fluxopt/lpspec/commit/7bec431c6d46e5d2eda96cf23e9fde7712825adb))
* check() enforces degree 1; README stops promising a fallback ([#55](https://github.com/fluxopt/lpspec/issues/55)) ([d0b008c](https://github.com/fluxopt/lpspec/commit/d0b008ca5b8f4dc1b3bbeee460392c2ded32469b))


### Refactoring

* cut back the accumulated surface ([#56](https://github.com/fluxopt/lpspec/issues/56)) ([1802dd0](https://github.com/fluxopt/lpspec/commit/1802dd00feeeda727c67a6a36c415ddc6caf5a21))
* finish the lane split — where_parser keeps grammar, not evaluation ([#59](https://github.com/fluxopt/lpspec/issues/59)) ([a30f7ec](https://github.com/fluxopt/lpspec/commit/a30f7ec7dc511ea7a59ded6434facc6e90405a23))
* let the annotations say what the parsers already guarantee ([#61](https://github.com/fluxopt/lpspec/issues/61)) ([49387d4](https://github.com/fluxopt/lpspec/commit/49387d4c993df817da8ac3e5fd3f5f8f27becb72))
* name resolution is a pass, not a backend detail ([#62](https://github.com/fluxopt/lpspec/issues/62)) ([8622fa6](https://github.com/fluxopt/lpspec/commit/8622fa6d0063ae32fbecdffe51f9e28f70969452))


### Documentation

* cross-language vocabulary map, and a procedure for the ceiling ([#63](https://github.com/fluxopt/lpspec/issues/63)) ([7f535cd](https://github.com/fluxopt/lpspec/commit/7f535cde9127079e7e6dcaf5754daeadb659e7af))
* SPEC catches up with the code it describes ([#57](https://github.com/fluxopt/lpspec/issues/57)) ([50edfb6](https://github.com/fluxopt/lpspec/commit/50edfb61f991ac852e8aace2be79f93506a28a47))

## [0.0.0-alpha.2](https://github.com/fluxopt/lpspec/compare/v0.0.0-alpha.1...v0.0.0-alpha.2) (2026-07-24)


### Bug Fixes

* name-check dimension kwargs at load time; restore docs lost at merge ([#48](https://github.com/fluxopt/lpspec/issues/48)) ([4c6bfc9](https://github.com/fluxopt/lpspec/commit/4c6bfc97919e84db71bdc4d0c82d46a20f866b2d))

## 0.0.0-alpha.1 (2026-07-24)


### Features

* API polish — check(), write(), LanguageError, Result lifecycle ([#36](https://github.com/fluxopt/lpspec/issues/36)) ([fc36af5](https://github.com/fluxopt/lpspec/commit/fc36af515eb9baafdf81c036a27ad8ca9431297f))
