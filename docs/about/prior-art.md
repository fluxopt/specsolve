# Prior art and credit

This page names the projects specsolve is derived from and says how to cite them,
for anyone comparing specsolve with either project or citing it.

**[Calliope](https://github.com/calliope-project/calliope) (Apache-2.0) is
where this surface comes from.** The surface is the YAML you write, and it is
their design: solver-ready math declared as a reviewable file, a block per
component, `dims:` for the list of dimensions, a `where:` string over
`AND`/`OR`/`NOT`, `bounds:`, `active:`. So is parsing the strings with
pyparsing rather than `eval`. Our `expressions` is their `global_expressions`,
and our `piecewise` is their `piecewise_constraints`. What is ours is the
semantics underneath: one `expression` per block, macros that take arguments, a
schema closed at every level, and the absence and degree-1 laws
([the ten rules](https://mathspec.readthedocs.io/en/latest/reference/language/#the-ten-rules)).
Their math is the corpus we score coverage against, not a specification we
match
([the limits](https://mathspec.readthedocs.io/en/latest/reference/language/errors/#what-the-language-will-not-express)).
**File portability is not a goal.** A Calliope model does not load here, and
operation parity with xarray or pandas is not a goal either.

**[linopy](https://github.com/PyPSA/linopy) (MIT) is the vocabulary, the oracle
and the denominator.** Where a concept is already theirs, we copy the spelling.
Every language feature is differentially tested against a linopy build, and
the [benchmarks](benchmarks.md) measure specsolve against it.
[Coming from linopy](linopy.md) maps one to the other. The ported models in
[the gallery](../examples/index.md) and their reference optima are **PyPSA**'s.

No code from any of them is vendored, so none of this is a licence obligation.
It is stated because a debt only the author knows about is one the next reader
has to rediscover. Neither project has reviewed this one, and mistakes in the
comparisons above are ours.

If you cite this project, cite the two it is built on:

```bibtex
@article{Pfenninger2018,
  doi = {10.21105/joss.00825}, year = {2018}, publisher = {The Open Journal},
  volume = {3}, number = {29}, pages = {825},
  author = {Stefan Pfenninger and Bryn Pickering},
  title = {Calliope: a multi-scale energy systems modelling framework},
  journal = {Journal of Open Source Software}
}

@article{Hofmann2023,
  doi = {10.21105/joss.04823}, year = {2023}, publisher = {The Open Journal},
  volume = {8}, number = {84}, pages = {4823},
  author = {Fabian Hofmann},
  title = {Linopy: Linear optimization with n-dimensional labeled variables},
  journal = {Journal of Open Source Software}
}
```
