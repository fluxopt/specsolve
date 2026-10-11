# Releasing

A release is a pull request that edits `CHANGELOG.md`. Merging it tags the
commit, opens the GitHub release and publishes the package to PyPI.

## Between releases

Each pull request adds one line under `## Upcoming version` at the top of
`CHANGELOG.md`: its title and a link to it. A `feat`, `fix`, `perf`,
`refactor`, `docs` or `revert` pull request adds a line. A `chore`, `test`,
`ci`, `build` or `style` pull request adds none. The `Changelog line` check
fails a pull request that owes a line and adds none. The label `no changelog`
opts it out, for a change no reader of the changelog needs to hear about.

```markdown
## Upcoming version

- feat(language): a coordinate may declare its own label space ([#123](https://github.com/fluxopt/specsolve/pull/123))
```

## Cutting a release

1. Open a pull request that renames `## Upcoming version` to the version and
   the day, and edit the section into the release notes: group the lines, merge
   related ones, and add a paragraph on top if the release needs one.

   ```markdown
   ## 0.1.0 (2026-10-01)
   ```

   A version is `X.Y.Z`, with an optional `aN`, `bN` or `rcN` for a
   pre-release. The `ci` check refuses a heading the release workflow cannot
   act on: a version that is not newer than the latest tag, a date that is not a
   day, or a section with nothing under it. You can also put a new, empty
   `## Upcoming version` above the release.

2. Merge it. [`release.yaml`](.github/workflows/release.yaml) then:
   - tags the merge commit `v0.1.0` and opens the GitHub release, with the
     section as its notes. A pre-release version is marked as one.
   - builds the wheel and the sdist from the tag, and checks that both carry
     the version the changelog names.
   - publishes both to PyPI, after a reviewer approves the `pypi` environment.

Every other push to `main` finds no untagged version heading and does nothing.

`python -m tools.changelog check` runs the same check locally, and
`python -m tools.changelog notes 0.1.0` prints the notes the release will get.

## The version

The version comes from the git tag. `pyproject.toml` declares
`dynamic = ["version"]`, and hatch-vcs reads the tag at build time, so the
changelog heading, the tag and the wheel carry one number. Between releases,
hatch-vcs numbers every commit (`0.1.1.dev22+ged5056087`), so an untagged
commit is already uniquely versioned and installable.

Do not push a version tag by hand. The release workflow only acts on a heading
with no tag, so a hand-made tag makes its heading history before anything is
published.

The history below the first release heading is what release-please wrote for
the `0.0.1-alpha.N` stream. Those versions were tags and GitHub releases only,
and the tool reads each heading as released because its tag exists.

## When a step fails

- **The tag or the GitHub release was not made.** Fix the cause and re-run the
  workflow. It acts on the first version heading that has no tag.
- **The build or the upload failed after the tag was made.** Re-run the failed
  jobs of that run from the Actions tab. PyPI never takes a version twice, so a
  version that reached PyPI is final. Fix a wrong release with the next
  version.

## One-time setup

- **PyPI.** Add a trusted publisher for the project `specsolve`: owner
  `fluxopt`, repository `specsolve`, workflow `release.yaml`, environment
  `pypi`. Before the first upload it is a _pending_ publisher.
- **The `pypi` environment.** Create it under the repository's
  Settings → Environments, limit it to `main`, and add the people who may
  approve a release as required reviewers.
- **Branch protection on `main`.** Require the `ci`,
  `Conventional commit subject` and `Changelog line` checks.
- **The label `no changelog`.** Create it under Issues → Labels.

**PyPI refuses a direct reference.** A distribution that names a git URL in its
metadata is rejected at upload, in a dependency and in an extra alike. None
stands in `pyproject.toml`. A git URL in a dependency group is not package
metadata, so PyPI never sees it.

## Branch protection

`main` is covered by a repository ruleset: no deletion, no force-push,
squash-only merges through a PR, and three required checks — `ci` (from
[`ci.yml`](.github/workflows/ci.yml)), `Conventional commit subject` (from
[`pr-title.yaml`](.github/workflows/pr-title.yaml)) and `Changelog line` (from
[`changelog.yml`](.github/workflows/changelog.yml)). The CI job has a fixed name
and no matrix, so the ruleset names it directly and never needs updating; keep
it that way. If a Python matrix ever comes back, put an aggregating gate job in
front of it rather than naming legs — requiring `full (3.11)` would mean adding
a version leaves it unrequired, and dropping one blocks every PR on a check that
can no longer report.

A required check must exist on `main` before it is required — land the workflow
first, then add it to the ruleset. Approvals are not required (solo repo); a
review count of 0 still forces the PR, the squash and the checks.

**The subject that lands on main.** `main` takes squash merges only, so one PR
is one commit, and its subject is the line the PR adds to the changelog — the
rule a contributor has to follow, and the allowed types, are in
[CONTRIBUTING.md](CONTRIBUTING.md#branches-commits-prs). Because
`squash_merge_commit_title` is `COMMIT_OR_PR_TITLE`, GitHub uses the PR title on
a multi-commit PR and the commit's own title on a single-commit one; the check
validates both, so neither can slip through.

### Relaxed while in early development

Actions bills per job, rounded up to the minute, and the suite takes ~5s. So CI
cost is job count, and before 1.0 it deliberately trades coverage for cost. What that gives up:

- **Only Python 3.12 is tested.** The 3.13 classifier in `pyproject.toml` is an
  untested claim. 3.12 is the floor, so it catches the common breakage
  (reaching for a newer stdlib feature) but not the reverse: a removal or
  deprecation that only bites on 3.13.
- **Only two dependency sets are installed:** current-with-dev, and the declared
  floors bare. The floors are exercised *without* xarray/pandas, so the
  bridges out of a result are only ever tested against current xarray.

This list used to carry a third entry, and it is worth keeping the correction
rather than the claim: *"the PR-title check does not re-run on `synchronize` …
the cost is a missing CHANGELOG line, not a broken release."* The cost was a
**permanently blocked PR**. A required check is evaluated against the head
commit, so skipping the push event leaves the new head with no result at all,
and GitHub waits for one that never arrives — every visible check green, merge
blocked, nothing to click. #269 sat like that for hours and was reported as CI
hanging. The trigger is back; the saving was one ~5s job per push.

A Python matrix, behind a gate job, is the first thing to add back when these
are tightened. Until then, prefer spending minutes on the suite over spending
them on matrix breadth.

## Consuming an unreleased branch

Don't cut a release for this — install from the ref:

```bash
pixi add --pypi "specsolve @ git+ssh://git@github.com/fluxopt/specsolve@feat/some-branch"
pixi add --pypi "specsolve @ git+https://github.com/fluxopt/specsolve@d09aab6"
```
