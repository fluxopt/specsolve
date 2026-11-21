"""One word per meaning: the words mathspec's glossary replaces stay out of specsolve's prose.

Prose is a docstring, a comment or a string literal in ``src/`` and ``tests/``,
and the text of a page under ``docs/`` or at the top of the tree, outside code
spans, code fences, link targets and generated blocks. A word is matched whole,
in any letter case.

A word with no other meaning here is refused outright. A word that keeps a second,
legitimate meaning, such as *frame* for mathspec's dimensions of an entry, or
*mask* for linopy's ``mask=``, is held by a ratchet: its ceiling is the count
left in the tree, so a new use fails until a reviewer lowers it with the count.

*index* is not checked: it means a solver, row or column index, or a pandas
index, and a dimension's labels are its labels (``docs/reference/glossary.md``).

``tests/mathspec/`` holds verbatim copies of mathspec's glossary and of its
table of words to avoid. Three checks read them: every word mathspec avoids is
in [`INSTEAD`][] or [`KEPT`][]; a term both glossaries define links mathspec's
entry from specsolve's; and the copies were taken at the mathspec this package
pins, so a new mathspec fails here until they are copied again.
"""

from __future__ import annotations

import io
import re
import tokenize
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MATHSPEC = ROOT / 'tests' / 'mathspec'
GLOSSARY = 'https://mathspec.readthedocs.io/en/latest/reference/glossary/'

#: The mathspec release the copies under ``tests/mathspec/`` were last checked against.
MATHSPEC_CHECKED = '0.4.0'

#: Word → what to write instead, in the glossaries' words.
INSTEAD = {
    'declaration': 'entry',
    'declarations': 'entries',
    'mask': 'condition, or `where:`',
    'masks': 'conditions',
    'masked': 'absent, or narrowed by a condition',
    'unmasked': 'with no condition',
    'masked out': 'absent',
    'unbuilt': 'absent',
    'predicate': 'condition',
    'predicates': 'conditions',
    'primitive': 'operator',
    'primitives': 'operators',
    'summand': 'a term of a sum',
    'summands': 'terms of a sum',
    'lowered': 'compiled',
    'lowering': 'compiling',
    'lowers': 'compiles',
    'streaming lane': 'relational lane',
    'sayable': 'in the language',
    'dim': 'dimension',
    'dims': 'dimensions',
    'frame': 'table, where a polars table is meant',
    'frames': 'tables, where polars tables are meant',
    'consumer': 'tool',
    'consumers': 'tools',
    'bind data': 'attach data',
    'reject': 'refuse',
    'rejects': 'refuses',
    'rejected': 'refused',
    'error out': 'refuse',
    'normal form': 'the canonical form',
    'seam': 'the plain thing, such as the boundary',
    'seams': 'the plain thing, such as the boundaries',
    'surface': 'the plain thing, such as the API',
    'tax': 'the plain thing, such as the cost',
    'closure': 'the plain thing',
}

#: Words mathspec avoids that specsolve keeps, each with its reason.
KEPT = {
    'lane': "specsolve's own word for a way to build a spec, in docs/development/architecture.md",
    'a program': "mathspec's `Program` is a program; the tool it forbids calling one, no pattern tells apart",
}

#: Uses left in another sense, such as linopy's ``mask=`` or mathspec's frame.
#: A ratchet: lower it in the PR that lowers the count.
CEILING = {
    'declaration': 7,
    'declarations': 1,
    'dim': 1,
    'dims': 6,
    'frame': 24,
    'lowered': 1,
    'lowers': 2,
    'mask': 6,
    'masks': 1,
    'consumer': 2,
    'consumers': 1,
    'seam': 16,
    'seams': 1,
    'surface': 14,
}

#: Pages that are generated, or that quote the retired words to retire them.
SKIPPED = ('CHANGELOG.md', 'docs/examples/pypsa_ladder', '.claude/', 'tests/test_words.py')

_NOT_CODE_BEFORE = r'(?<![.\'"\w-])'
_NOT_CODE_AFTER = r'(?![\w-]|\s*[:=(\'"\]])'
_WORD = re.compile(
    '|'.join(_NOT_CODE_BEFORE + re.escape(word) + _NOT_CODE_AFTER for word in sorted(INSTEAD, key=len, reverse=True)),
    re.IGNORECASE,
)
_CODE = re.compile(r'``.*?``|`[^`\n]*`|\*[A-Za-z_]\w*\*|\]\([^)]*\)|<!--.*?-->', re.DOTALL)
_FENCE = re.compile(r'^(```|~~~).*?^\1', re.MULTILINE | re.DOTALL)
_GENERATED = re.compile(r'<!-- (\w+):begin -->.*?<!-- \1:end -->', re.DOTALL)


def _blanked(text: str, pattern: re.Pattern[str]) -> str:
    """*text* with every match of *pattern* blanked, newlines kept, so a line number still lands."""
    return pattern.sub(lambda m: re.sub(r'[^\n]', ' ', m.group()), text)


def _python_prose(path: Path) -> list[tuple[int, str]]:
    """``(line, text)`` for every string and comment of *path*: the parts a reader reads as prose."""
    found = []
    for token in tokenize.generate_tokens(io.StringIO(path.read_text()).readline):
        if token.type in {tokenize.STRING, tokenize.COMMENT, getattr(tokenize, 'FSTRING_MIDDLE', -1)}:
            text = _blanked(token.string, _CODE)
            found += [(token.start[0] + i, line) for i, line in enumerate(text.splitlines())]
    return found


def _markdown_prose(path: Path) -> list[tuple[int, str]]:
    """``(line, text)`` for *path* with fences, generated blocks and code spans blanked."""
    text = path.read_text()
    for pattern in (_GENERATED, _FENCE, _CODE):
        text = _blanked(text, pattern)
    return list(enumerate(text.splitlines(), start=1))


def _prose() -> list[tuple[str, int, str]]:
    """``(path, line, text)`` for the prose of every file the rule covers."""
    paths = [
        *(ROOT / 'src').rglob('*.py'),
        *(ROOT / 'tests').rglob('*.py'),
        *(ROOT / 'docs').rglob('*.md'),
        *ROOT.glob('*.md'),
    ]
    found = []
    for path in sorted(paths):
        name = path.relative_to(ROOT).as_posix()
        if name.startswith(SKIPPED):
            continue
        lines = _python_prose(path) if path.suffix == '.py' else _markdown_prose(path)
        found += [(name, number, line) for number, line in lines]
    return found


def _uses() -> dict[str, list[str]]:
    """Every retired word in prose, as ``path:line`` per word."""
    uses: dict[str, list[str]] = {}
    for name, number, line in _prose():
        for match in _WORD.finditer(line):
            uses.setdefault(match.group().lower(), []).append(f'{name}:{number}')
    return uses


def test_a_retired_word_is_not_written_where_its_replacement_is_meant() -> None:
    """A word over its ceiling is a new use; the message names the word to write instead."""
    uses = _uses()
    over = {word: found for word, found in uses.items() if len(found) > CEILING.get(word, 0)}
    assert not over, 'prose uses a word mathspec replaces — write the replacement:\n' + '\n'.join(
        f'  {word!r} → {INSTEAD[word]} ({len(found)} uses, ceiling {CEILING.get(word, 0)}): {", ".join(found)}'
        for word, found in sorted(over.items())
    )


def test_each_ceiling_is_the_count_left_in_the_tree() -> None:
    """Below its ceiling a word left headroom for a new use, so the ceiling comes down with the count."""
    counts = {word: len(found) for word, found in _uses().items()}
    stale = {
        word: (ceiling, counts.get(word, 0)) for word, ceiling in CEILING.items() if counts.get(word, 0) != ceiling
    }
    assert not stale, 'set each ceiling to the count the tree holds: ' + ', '.join(
        f'{word!r} {ceiling} → {count}' for word, (ceiling, count) in sorted(stale.items())
    )


def _mathspec_avoids() -> set[str]:
    """The words mathspec's copied word table tells a writer not to use, lower-case."""
    text = ' '.join((MATHSPEC / 'words.md').read_text().split())
    listed = re.search(r'Words to avoid, and what to write instead: (.*?\))\.', text)
    assert listed, 'the copied word table has its sentence of words to avoid'
    words = {
        word.strip()
        for left, _ in re.findall(r'([^()]+?)\s*\(([^)]*)\)', listed.group(1))
        for part in left.split(',')
        for word in part.split(' or ')
        if word.strip()
    }
    words |= set(re.findall(r'[Nn]ever "([^"]+)"', text))
    words |= set(re.findall(r'_(\w+)_ is a key; in prose the word is', text))
    return {word.lower() for word in words}


def test_every_word_mathspec_avoids_is_refused_or_kept_on_purpose() -> None:
    """A word mathspec retires reaches specsolve's check the day its copy does, or the copy says why not."""
    covered = [*INSTEAD, *KEPT]
    missing = sorted(w for w in _mathspec_avoids() if not any(c == w or c.startswith(w) for c in covered))
    assert not missing, (
        f'mathspec avoids {missing}, which INSTEAD does not refuse and KEPT does not keep: add each to one of them'
    )


def _slug(heading: str) -> str:
    """*heading* as the docs site anchors it."""
    return re.sub(r'[^\w\s-]', '', heading).strip().lower().replace(' ', '-')


def _entries(page: Path) -> dict[str, str]:
    """``{term: text}`` for each bold term of specsolve's glossary, lower-case and without backticks."""
    found: dict[str, str] = {}
    for block in page.read_text().split('\n\n'):
        head, _, _ = block.partition('\n:')
        if head.startswith('**'):
            for term in re.findall(r'\*\*(.+?)\*\*', head):
                found[term.strip('`').lower()] = block
    return found


def test_a_term_both_glossaries_define_links_the_mathspec_entry() -> None:
    """specsolve's entry adds what a caller of this package needs; the meaning stays mathspec's, linked."""
    theirs = {
        _slug(h).replace('-', ' '): _slug(h)
        for h in re.findall(r'^## (.+)$', (MATHSPEC / 'glossary.md').read_text(), re.MULTILINE)
    }
    ours = _entries(ROOT / 'docs' / 'reference' / 'glossary.md')
    unlinked = sorted(
        term
        for term, text in ours.items()
        if term.rstrip(':') in theirs and f'{GLOSSARY}#{theirs[term.rstrip(":")]}' not in ' '.join(text.split())
    )
    assert not unlinked, f'these glossary entries define a mathspec term without linking its entry: {unlinked}'


def test_the_copies_of_mathspec_were_checked_against_the_pinned_mathspec() -> None:
    """A mathspec release can retire a word or redefine a term; the copies say so only once they are taken again."""
    installed = version('mathspec')
    assert installed == MATHSPEC_CHECKED, (
        f'mathspec is {installed}, but tests/mathspec/ was checked against {MATHSPEC_CHECKED}. Copy '
        f'docs/reference/glossary.md and the "Words this project owns" section of '
        f'.claude/skills/prose-writing/SKILL.md from mathspec v{installed}, then set MATHSPEC_CHECKED'
    )
