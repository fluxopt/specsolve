"""Run a campaign over a range of seeds and group what broke.

Each seed is one model and its data (:mod:`tests.fuzz.generate`), checked by
every oracle in :mod:`tests.fuzz.check` in a worker process of its own, so a
hang or a crash costs one seed. The first seed of each kind of finding is
written to ``--out`` as a reproduction: the spec, the data, and what broke.

    pixi run python -m tests.fuzz --seeds 0:2000 --jobs 8 --out fuzz-findings
    pixi run python -m tests.fuzz --seeds 1234 -v       # one seed, every detail
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import io
import json
import logging
import multiprocessing
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import yaml

#: A seed whose oracles take longer than this is reported as a hang.
TIMEOUT_S = 60


def _one(seed: int, solve: bool) -> dict[str, Any]:
    from tests.fuzz.check import check
    from tests.fuzz.generate import generate

    logging.getLogger('linopy').setLevel(logging.ERROR)
    case = generate(seed)
    started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        outcome = check(case, solve=solve)
    return {
        'seed': seed,
        'status': outcome.status,
        'solved': outcome.solved,
        'seconds': time.perf_counter() - started,
        'findings': [
            {'oracle': f.oracle, 'summary': f.summary, 'signature': f.signature, 'detail': f.detail}
            for f in outcome.findings
        ],
    }


def _seeds(text: str) -> range:
    if ':' in text:
        start, stop = text.split(':')
        return range(int(start), int(stop))
    return range(int(text), int(text) + 1)


def _jsonable(sources: dict[str, Any]) -> dict[str, Any]:
    return {name: v.to_dicts() if hasattr(v, 'to_dicts') else v for name, v in sources.items()}


def reproduction(seed: int, out: Path, findings: list[dict[str, Any]]) -> Path:
    """Write *seed*'s spec, data and findings under *out*, and return the directory."""
    from tests.fuzz.generate import generate

    case = generate(seed)
    where = out / f'seed-{seed}'
    where.mkdir(parents=True, exist_ok=True)
    (where / 'spec.yaml').write_text(yaml.safe_dump(case.spec, sort_keys=False), encoding='utf-8')
    data = _jsonable(case.sources)
    (where / 'data.json').write_text(json.dumps(data, indent=1), encoding='utf-8')
    (where / 'findings.txt').write_text(
        '\n\n'.join(f'[{f["oracle"]}] {f["summary"]}\n{f["detail"]}' for f in findings), encoding='utf-8'
    )
    return where


def _reduce(seeds: range, *, verbose: bool) -> int:
    """Shrink every kind of finding each seed produces, and print the smallest case for each."""
    from tests.fuzz.check import check
    from tests.fuzz.generate import generate
    from tests.fuzz.reduce import reduce

    logging.getLogger('linopy').setLevel(logging.ERROR)
    found = 0
    for seed in seeds:
        case = generate(seed)
        with contextlib.redirect_stdout(io.StringIO()):
            signatures = list(dict.fromkeys(f.signature for f in check(case).findings))
        for signature in signatures:
            found += 1
            with contextlib.redirect_stdout(io.StringIO()):
                small = reduce(case, signature, log=_to_stderr if verbose else lambda _: None)
                findings = [f for f in check(small).findings if f.signature == signature]
            data = _jsonable(small.sources)
            print(f'=== seed {seed}: {signature}')
            print(yaml.safe_dump(small.spec, sort_keys=False))
            print(f'reads: {small.reads}')
            print(f'data: {json.dumps(data)}')
            print(f'{findings[0].detail if verbose else findings[0].summary}\n')
    return int(bool(found))


def _to_stderr(message: str) -> None:
    """The reducer's log, past the ``redirect_stdout`` that silences what the checks print."""
    print(message, file=sys.stderr)


def _campaign(seeds: range, jobs: int, solve: bool) -> dict[int, dict[str, Any]]:
    """Every seed's record, each checked in a worker of its own.

    A seed that hangs keeps its worker busy, so after a hang the pool is ended,
    which kills that worker, and the seeds still without a record go to a new
    pool. Otherwise every seed queued behind stuck workers would be reported as
    a hang of its own.
    """
    records: dict[int, dict[str, Any]] = {}
    remaining = list(seeds)
    while remaining:
        with multiprocessing.get_context('spawn').Pool(jobs, maxtasksperchild=50) as pool:
            pending = {seed: pool.apply_async(_one, (seed, solve)) for seed in remaining}
            for seed, job in pending.items():
                try:
                    records[seed] = job.get(timeout=TIMEOUT_S)
                except multiprocessing.TimeoutError:
                    records[seed] = _failed(seed, 'hang', f'no verdict within {TIMEOUT_S} s', 'hang')
                    break
                except Exception as e:
                    records[seed] = _failed(
                        seed, 'harness', f'{type(e).__name__}: {e}'[:200], f'harness: {type(e).__name__}'
                    )
        remaining = [seed for seed in remaining if seed not in records]
    return records


def _failed(seed: int, oracle: str, summary: str, signature: str) -> dict[str, Any]:
    finding = {'oracle': oracle, 'summary': summary, 'signature': signature, 'detail': ''}
    return {'seed': seed, 'status': oracle, 'solved': False, 'findings': [finding]}


def _found(records: dict[int, dict[str, Any]]) -> dict[str, list[str]]:
    """Each seed with a finding, and the signatures it has, sorted: the shape a baseline file holds."""
    found = {str(seed): sorted({f['signature'] for f in r['findings']}) for seed, r in records.items()}
    return {seed: signatures for seed, signatures in found.items() if signatures}


def against_baseline(found: dict[str, list[str]], baseline: dict[str, list[str]], seeds: range) -> list[str]:
    """What differs from *baseline* over *seeds*: a finding it does not hold, and one it holds that did not recur.

    Both directions fail, as a strict ``xfail`` does: a fix that removes a
    finding leaves the baseline stale until it is written again.
    """
    problems = []
    for seed in seeds:
        now, then = set(found.get(str(seed), ())), set(baseline.get(str(seed), ()))
        problems += [f'seed {seed}: new finding — {s}' for s in sorted(now - then)]
        problems += [f'seed {seed}: no longer found, so drop it from the baseline — {s}' for s in sorted(then - now)]
    return problems


#: Where a message says a defect showed: the entry and the side. It names the place, not the defect.
_PLACE = re.compile(r'(?:in |the )?(?:constraint _(?: lhs| rhs)?|named expression _|objective|the expression): ')


#: The names the generator gives its dimensions and entries, which a message may print unquoted.
_GENERATED = re.compile(r'\b(?:[tgbhk]|[pvecmr]\d+)\b')


def kind(signature: str) -> str:
    """*signature* without the place and the names it prints, cut to its first clause, so one defect is one kind.

    One defect reads as one kind wherever it shows: in a constraint or the
    objective, on either side, over any dimension.
    """
    return _GENERATED.sub('_', _PLACE.sub('', signature))[:120].split('; ')[0]


def against_kinds(found: dict[str, list[str]], kinds: dict[str, str]) -> list[str]:
    """Every kind of finding that *kinds* does not name, with the first seed that has it."""
    seen: dict[str, str] = {}
    for seed, signatures in sorted(found.items(), key=lambda kv: int(kv[0])):
        for signature in signatures:
            seen.setdefault(kind(signature), seed)
    return [f'a new kind of finding, first at seed {seed} — {k}' for k, seed in sorted(seen.items()) if k not in kinds]


def main(argv: list[str] | None = None) -> int:
    """Run the campaign; exit 1 on a finding, or on a difference from ``--baseline`` or ``--known``."""
    front = argparse.ArgumentParser(prog='python -m tests.fuzz')
    front.add_argument('--seeds', default='0:200', help='a seed, or a range start:stop')
    front.add_argument('--jobs', type=int, default=os.cpu_count() or 1)
    front.add_argument('--out', type=Path, default=None, help='write one reproduction per kind of finding here')
    front.add_argument('--no-solve', action='store_true', help='skip the solve and the shuffle')
    front.add_argument('-v', '--verbose', action='store_true', help='print every finding in full')
    front.add_argument('--reduce', action='store_true', help='shrink each finding of each seed, in this process')
    gate = front.add_mutually_exclusive_group()
    gate.add_argument('--baseline', type=Path, help='fail only where the findings differ from this file, seed by seed')
    gate.add_argument('--known', type=Path, help='fail only on a kind of finding this file does not name')
    gate.add_argument('--write-baseline', type=Path, help='write the findings to this file, and pass')
    args = front.parse_args(argv)
    seeds = _seeds(args.seeds)
    if args.reduce:
        return _reduce(seeds, verbose=args.verbose)

    records = _campaign(seeds, args.jobs, solve=not args.no_solve)
    statuses = collections.Counter(r['status'] for r in records.values())
    solved = sum(r['solved'] for r in records.values())
    groups: dict[str, list[int]] = collections.defaultdict(list)
    for seed, record in records.items():
        for finding in record['findings']:
            groups[finding['signature']].append(seed)
            if args.verbose:
                print(f'--- seed {seed} [{finding["oracle"]}] {finding["summary"]}\n{finding["detail"]}\n')

    print(f'{len(seeds)} seeds: {dict(statuses)}, {solved} solved on both lanes')
    for signature, hit in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        first = records[hit[0]]
        print(f'{len(hit):5d}  {signature}\n       seeds {hit[:8]}')
        if args.out is not None:
            reproduction(hit[0], args.out, first['findings'])

    found = _found(records)
    if args.write_baseline is not None:
        args.write_baseline.write_text(json.dumps(found, indent=1, sort_keys=True) + '\n', encoding='utf-8')
        print(f'wrote {len(found)} seeds with findings to {args.write_baseline}')
        return 0
    if args.baseline is not None or args.known is not None:
        path = args.baseline or args.known
        baseline = json.loads(path.read_text(encoding='utf-8'))
        problems = against_baseline(found, baseline, seeds) if args.baseline else against_kinds(found, baseline)
        print('\n'.join(problems) or f'every finding is one {path} already holds')
        if problems and args.baseline:
            print(f'\nReproduce a seed with `python -m tests.fuzz --seeds <n> --reduce`. After a fix, or a change to the '
                  f'generator, write the baseline again: `python -m tests.fuzz --seeds {args.seeds} --write-baseline {path}`.')  # fmt: skip
        elif problems:
            print(f'\nReproduce a seed with `python -m tests.fuzz --seeds <n> --reduce`, file what it shows, and add '
                  f'the kind to {path} under the issue.')  # fmt: skip
        return int(bool(problems))
    return int(bool(groups))


if __name__ == '__main__':
    sys.exit(main())
