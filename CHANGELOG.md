# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `contradicted` now covers a **reversed direction of effect**, not only a
  negation flip. "increased" -> "decreased" carries no negation on either
  side, so the negation-polarity rule could not see it: the monitor scored
  such a summary suspicion **0.000** and accepted it at every audit budget.
  Gated on the two sentences being near-identical once direction words are
  stripped, and valence words ("improved"/"worsened") are excluded, so
  unrelated findings that merely point opposite ways are not flagged.
- Three red-team cases for the failure class the suite did not contain:
  `direction-attack`, `comparative-attack`, `zero-baseline-attack`, each with
  a faithful counterpart.
- `pyproject.toml` carrying packaging metadata, dependency extras (`test`,
  `cohere`), and the ruff configuration (inlined from the former `ruff.toml`,
  which is now removed). `pip install -e .[test]` works.
- `tests/test_pipeline_categories.py`: end-to-end behavior tests, run through
  the full `run_pipeline`, for the four claim categories Faithful claims to
  detect (added, dropped, overstated, contradicted) plus a clean summary that
  flags nothing.
- `docs/web-viewer.png`: a screenshot of the bundled side-by-side viewer,
  referenced near the top of the README.
- `CITATION.cff` and this changelog.

### Fixed

- Percentage inflation against a **0% baseline** was invisible. A source
  stating 0% against a summary stating 40% is the starkest inflation there
  is, but `_numeric_inflation` returned early on `source_max <= 0` before
  reaching the ratio test that already handled it correctly.

### Changed

- Control-eval frontier re-measured on the extended suite: **92% safety at
  100% usefulness**, 46% audit rate, eleven of twelve attacks caught (was
  89% / 44% / eight of nine on the smaller suite). On the extended suite the
  monitor scored 67% safety before these two fixes. Usefulness stays at 100%:
  no honest summary is newly flagged.

## [0.1.0] - 2026-09-04

### Added
- Initial pipeline: dependency-free claim extraction (stage 1), an
  IDF-weighted lexical-overlap aligner (stage 2), and a rule-based classifier
  with a numeric-consistency check (stage 3).
- `faithful.monitor`: a noisy-OR suspicion score aggregated over per-claim
  verdicts, and `faithful.control_eval`: a red-team/blue-team
  safety-vs-usefulness evaluation over a bundled synthetic suite.
- Optional Cohere Rerank alignment backend and Cohere Command grounded
  classifier (`faithful/cohere_backend.py`), both off by default and tested
  offline against an in-memory fake client.
- `web/index.html`: a dependency-free side-by-side claim viewer.
- `docs/control.md`: threat model, monitor description, and a self-critique
  of the control evaluation.
- Characterization tests over the alignment and classification stages
  (`tests/test_align.py`, `tests/test_classify.py`) that pin current scoring
  behavior, including three documented quirks.
- CI on Python 3.9 and 3.12: ruff lint, the test suite, and the bundled
  example, with a least-privilege token and monthly Dependabot updates for
  GitHub Actions.

### Fixed
- A label misparse, numeric false positives, and a dominated operating point
  in the control evaluation.
- A magnitude-overstatement blind spot: intensifier words ("dramatically")
  with no causal verb were not being caught as overstatement.
