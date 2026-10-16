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
                small = reduce(case, signature, log=print if verbose else lambda _: None)
                findings = [f for f in check(small).findings if f.signature == signature]
            data = _jsonable(small.sources)
            print(f'=== seed {seed}: {signature}')
            print(yaml.safe_dump(small.spec, sort_keys=False))
            print(f'reads: {small.reads}')
            print(f'data: {json.dumps(data)}')
            print(f'{findings[0].detail if verbose else findings[0].summary}\n')
    return int(bool(found))


def main(argv: list[str] | None = None) -> int:
    """Run the campaign; exit 1 if anything was found."""
    front = argparse.ArgumentParser(prog='python -m tests.fuzz')
    front.add_argument('--seeds', default='0:200', help='a seed, or a range start:stop')
    front.add_argument('--jobs', type=int, default=os.cpu_count() or 1)
    front.add_argument('--out', type=Path, default=None, help='write one reproduction per kind of finding here')
    front.add_argument('--no-solve', action='store_true', help='skip the solve and the shuffle')
    front.add_argument('-v', '--verbose', action='store_true', help='print every finding in full')
    front.add_argument('--reduce', action='store_true', help='shrink each finding of each seed, in this process')
    args = front.parse_args(argv)
    if args.reduce:
        return _reduce(_seeds(args.seeds), verbose=args.verbose)

    seeds = _seeds(args.seeds)
    statuses: collections.Counter[str] = collections.Counter()
    groups: dict[str, list[int]] = collections.defaultdict(list)
    examples: dict[str, dict[str, Any]] = {}
    solved = 0
    context = multiprocessing.get_context('spawn')
    with context.Pool(args.jobs, maxtasksperchild=50) as pool:
        pending = {seed: pool.apply_async(_one, (seed, not args.no_solve)) for seed in seeds}
        for seed, job in pending.items():
            try:
                record = job.get(timeout=TIMEOUT_S)
            except Exception as e:
                record = {'seed': seed, 'status': 'harness', 'solved': False, 'findings': [
                    {'oracle': 'harness', 'summary': f'{type(e).__name__}: {e}'[:200], 'signature': f'harness: {type(e).__name__}', 'detail': repr(e)}
                ]}  # fmt: skip
            except multiprocessing.TimeoutError:
                record = {'seed': seed, 'status': 'hang', 'solved': False, 'findings': [
                    {'oracle': 'hang', 'summary': f'no verdict within {TIMEOUT_S} s', 'signature': 'hang', 'detail': ''}
                ]}  # fmt: skip
            statuses[record['status']] += 1
            solved += record['solved']
            for finding in record['findings']:
                groups[finding['signature']].append(seed)
                examples.setdefault(finding['signature'], {'seed': seed, 'finding': finding, 'all': record['findings']})
                if args.verbose:
                    print(f'--- seed {seed} [{finding["oracle"]}] {finding["summary"]}\n{finding["detail"]}\n')

    print(f'{len(seeds)} seeds: {dict(statuses)}, {solved} solved on both lanes')
    for signature, hit in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        example = examples[signature]
        print(f'{len(hit):5d}  {signature}\n       first seed {example["seed"]}: {example["finding"]["summary"]}')
        if args.out is not None:
            reproduction(example['seed'], args.out, example['all'])
    return int(bool(groups))


if __name__ == '__main__':
    sys.exit(main())
