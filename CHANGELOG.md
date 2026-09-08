# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
