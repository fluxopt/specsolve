"""Two versions written down twice each. This keeps each pair equal.

ruff: ``pyproject.toml`` pins it in the dev group, which CI runs, and
``.pre-commit-config.yaml`` pins the ``ruff-pre-commit`` rev, which the hook
runs. Dependabot bumps the two in separate PRs; a skew is fixed by landing the
other one.

The dependency floors: ``[project.dependencies]`` declares each as a lower
bound, and the ``floors`` pixi environment pins the same package to that exact
version.

The mathspec ceiling: mathspec raises its minor version on a break, so the
ceiling is the next minor above the floor, and a floor that moves takes it along.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: The `rev:` on the ruff-pre-commit repo block, e.g. `v0.16.0`.
_HOOK_REV = re.compile(r'ruff-pre-commit\s*\n\s*rev:\s*(\S+)')


def _pinned_in_pyproject() -> str:
    groups = tomllib.loads((REPO / 'pyproject.toml').read_text())['dependency-groups']
    pins = [spec for spec in groups['dev'] if spec.startswith('ruff==')]
    assert len(pins) == 1, f'expected exactly one `ruff==` pin in the dev group, found {pins}'
    return pins[0].removeprefix('ruff==')


def _pinned_in_pre_commit() -> str:
    match = _HOOK_REV.search((REPO / '.pre-commit-config.yaml').read_text())
    assert match is not None, 'no `rev:` found for the ruff-pre-commit repo'
    return match[1].removeprefix('v')


def test_ruff_is_the_same_version_in_ci_and_in_the_hook():
    pyproject, pre_commit = _pinned_in_pyproject(), _pinned_in_pre_commit()
    assert pyproject == pre_commit, (
        f'ruff is {pyproject} in pyproject.toml but {pre_commit} in .pre-commit-config.yaml — '
        f'the hook and CI would disagree about formatting. Dependabot bumps these in two '
        f'separate PRs; land the other one, or match them by hand.'
    )


#: `polars>=2.0` -> ('polars', '2.0', None), `mathspec>=0.2.1,<0.3` ->
#: ('mathspec', '0.2.1', '0.3'), `highspy>=1.13,!=1.16.0` -> ('highspy', '1.13', None).
#: A lower bound is a claim the `floors` environment has to pin to prove; an
#: excluded release or a ceiling, where there is one, is not.
_FLOOR = re.compile(r'^([A-Za-z0-9._-]+)>=([0-9][0-9a-zA-Z.]*)(?:,!=[0-9][0-9a-zA-Z.]*)*(?:,<([0-9][0-9a-zA-Z.]*))?$')


def _declared_floors() -> dict[str, str]:
    declared = tomllib.loads((REPO / 'pyproject.toml').read_text())['project']['dependencies']
    unparsed = [spec for spec in declared if not _FLOOR.match(spec)]
    assert not unparsed, (
        f'{unparsed} is not a `name>=version` lower bound, with or without `,!=version` exclusions and a `,<version` ceiling. A '
        f'runtime dependency written any other way has no floor for the `floors` environment to '
        f'pin, so teach this pattern the new shape rather than leaving the dependency unchecked.'
    )
    return {match[1]: match[2] for spec in declared if (match := _FLOOR.match(spec))}


#: What runs the suite, installed unconstrained: neither has a floor to prove.
_RUNNER = frozenset({'pytest', 'pytest-xdist'})


def _pinned_in_the_floors_environment() -> dict[str, str]:
    """The `floors` pins, without the project under test and without `_RUNNER`."""
    pixi = tomllib.loads((REPO / 'pyproject.toml').read_text())['tool']['pixi']
    pinned = pixi['feature']['floors']['pypi-dependencies']
    return {name: spec for name, spec in pinned.items() if isinstance(spec, str) and name not in _RUNNER}


def _pinned_to_a_commit() -> dict[str, dict[str, str]]:
    """The `floors` pins that name a git commit rather than a release, by package."""
    pixi = tomllib.loads((REPO / 'pyproject.toml').read_text())['tool']['pixi']
    pinned = pixi['feature']['floors']['pypi-dependencies']
    return {name: spec for name, spec in pinned.items() if isinstance(spec, dict) and 'git' in spec}


def test_a_dependency_ahead_of_its_release_is_pinned_to_one_commit_everywhere():
    """Until the release, the floor is a commit, and the default and `floors` environments install the same one."""
    pixi = tomllib.loads((REPO / 'pyproject.toml').read_text())['tool']['pixi']
    default = {
        name: spec for name, spec in pixi['pypi-dependencies'].items() if isinstance(spec, dict) and 'git' in spec
    }
    assert default == _pinned_to_a_commit(), (
        f'the default environment installs {default} and `floors` installs {_pinned_to_a_commit()}: '
        f'a dependency followed ahead of its release has one commit, or the suite tests one and the floor another'
    )


def test_the_floors_environment_pins_every_declared_lower_bound():
    declared = {name: floor for name, floor in _declared_floors().items() if name not in _pinned_to_a_commit()}
    pinned = _pinned_in_the_floors_environment()
    assert pinned == {name: f'=={floor}' for name, floor in declared.items()}, (
        f'the `floors` pixi environment pins {pinned}, but `[project.dependencies]` declares '
        f'{declared}. That environment exists to prove each declared lower bound is real, which it '
        f'only does while it installs exactly those versions and nothing the project does not '
        f'declare — both directions, so a dependency added without a pin fails here too.'
    )


def _declared_ceilings() -> dict[str, str]:
    declared = tomllib.loads((REPO / 'pyproject.toml').read_text())['project']['dependencies']
    return {match[1]: match[3] for spec in declared if (match := _FLOOR.match(spec)) and match[3]}


def test_mathspec_is_capped_below_the_next_minor_of_its_floor():
    """A specsolve release keeps installing with the mathspec it was released against.

    `format.json` names the specsolve that wrote an archive, and reading one in an old
    layout means installing that release. With a bare floor, that install takes the
    newest mathspec, whose next minor may load no spec the release can build.
    """
    floor = _declared_floors()['mathspec']
    major, minor = floor.split('.')[:2]
    assert _declared_ceilings().get('mathspec') == f'{major}.{int(minor) + 1}', (
        f'mathspec is declared from {floor}, so its ceiling is `<{major}.{int(minor) + 1}`: mathspec '
        f'raises its minor version on a break, and an old specsolve has to install with the language '
        f'it was released against.'
    )
