"""What a test builds a schema from.

`mathspec` owns the same four names; a test package is not shipped, so this
is a copy. If they drift, a test passes here and fails there over a model that
reads the same.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml as pyyaml
from mathspec import Spec, to_spec

#: The math of ``examples/dispatch.yaml`` as a dict; vary it with :func:`override`.
DISPATCH_SPEC: dict[str, Any] = {
    'dimensions': {'snapshot': {'dtype': 'int'}, 'generator': {'dtype': 'str'}},
    'parameters': {
        'p_max': {'dims': ['generator']},
        'cost': {'dims': ['generator']},
        'load': {'dims': ['snapshot']},
    },
    'variables': {'p': {'dims': ['snapshot', 'generator'], 'bounds': {'lower': 0, 'upper': 'p_max'}}},
    'constraints': {'balance': {'dims': ['snapshot'], 'expression': 'sum(p, over=generator) == load'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(p * cost)'},
}


def override(base: dict[str, Any], **patch: Any) -> dict[str, Any]:
    """A deep copy of ``base`` with dotted paths replaced.

    ``override(DISPATCH_SPEC, **{'variables.p.where': 'p_max > 0'})``. Missing
    intermediate keys are created.
    """
    raw = copy.deepcopy(base)
    for dotted, value in patch.items():
        node = raw
        *parents, leaf = dotted.split('.')
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value
    return raw


def schema_of(source: str | Path | dict[str, Any], **patch: Any) -> Spec:
    """A ``Spec`` from a YAML path, YAML text, or a raw dict.

    ``Path`` means a file, ``str`` means the YAML itself. ``**patch`` applies
    :func:`override` first.
    """
    raw = raw_of(source)
    return to_spec(override(raw, **patch) if patch else raw)


def expanded(source: str | Path | dict[str, Any] | Spec, *kinds: Any, **patch: Any) -> Spec:
    """:func:`schema_of` with its formulations written out — the shape every lane is handed.

    Every ``piecewise:`` block, and every ``sos:`` block too unless *kinds*
    names ``'piecewise'`` alone. A ``Spec`` passes straight through to ``expand``.
    """
    schema = source if isinstance(source, Spec) else schema_of(source, **patch)
    return schema.expand(*kinds)


def raw_of(source: str | Path | dict[str, Any]) -> dict[str, Any]:
    """The parsed mapping behind a path / YAML text / dict, unvalidated.

    Plain YAML: ``to_spec`` reads a ``str`` as a path, and callers patch the
    mapping into shapes no loaded ``Spec`` could hold.
    """
    if isinstance(source, dict):
        return source
    return pyyaml.safe_load(source.read_text() if isinstance(source, Path) else source)
