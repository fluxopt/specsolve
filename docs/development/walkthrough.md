# Walkthrough

One spec run through every stage of the pipeline: YAML, schema, core AST,
logical plan, model tables, LP text and solution. The script prints what each
stage produces, then two specs the language refuses and why.

```bash
python examples/walkthrough/walkthrough.py
```

Its output is committed as
[walkthrough.out](https://github.com/fluxopt/specsolve/blob/main/examples/walkthrough/walkthrough.out),
and `tests/test_walkthrough.py` checks it line for line.

## The spec

The dispatch model, plus a macro and a named expression. The macro is gone
after the language expands it. The named expression keeps its name, so the
last stage reads `total_supply` back at the solution with `evaluate()`.

<!-- The fence includes examples/walkthrough/walkthrough.yaml, which the walkthrough solves. -->
<!-- doctest: skip -->
```yaml title="examples/walkthrough/walkthrough.yaml"
--8<-- "examples/walkthrough/walkthrough.yaml"
```
