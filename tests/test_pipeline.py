"""Smoke tests for stages 2-3 and the end-to-end pipeline.

Run with either:

    pytest
    python tests/test_pipeline.py   # falls back to a tiny runner if pytest is absent
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from faithful.align import Alignment, align_claim
from faithful.classify import classify_claim
from faithful.extract import extract_claims
from faithful.pipeline import run_pipeline

EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")


# ---- Stage 2: alignment -------------------------------------------------- #

def test_align_returns_none_when_no_source():
    claim = extract_claims("The microbe reduced inflammation in mice.", origin="summary")[0]
    alignment = align_claim(claim, [])
    assert alignment.source_claim is None
    assert alignment.score == 0.0
    assert alignment.is_supported_by_source is False


def test_align_picks_the_most_similar_source_claim():
    summary = extract_claims("Butyrate production increased in the treated mice.", origin="summary")[0]
    sources = extract_claims(
        "Body weight was unchanged between groups. "
        "The treated mice showed increased butyrate production.",
        origin="paper",
    )
    alignment = align_claim(summary, sources)
    assert alignment.source_claim is not None
    assert "butyrate" in alignment.source_claim.text.lower()


# ---- Stage 3: classification --------------------------------------------- #

def test_classify_unsupported_without_alignment():
    claim = extract_claims("Probiotics prevent cancer in humans.", origin="summary")[0]
    result = classify_claim(claim, Alignment(claim=claim, source_claim=None, score=0.0))
    assert result.label == "unsupported"


def test_classify_overstated_on_added_strength():
    claim = extract_claims("The microbe cures colonic inflammation.", origin="summary")[0]
    source = extract_claims(
        "The microbe was associated with reduced colonic inflammation.", origin="paper")[0]
    result = classify_claim(claim, Alignment(claim=claim, source_claim=source, score=0.5))
    assert result.label == "overstated"


def test_classify_overstated_on_added_intensifier():
    # A magnitude intensifier the source does not carry is overstatement, even
    # with no causal verb ("modest ~15%" -> "dramatically").
    claim = extract_claims("The intervention dramatically shrank tumor size.", origin="summary")[0]
    source = extract_claims(
        "The intervention reduced tumor size by approximately 15%, a modest effect.",
        origin="paper")[0]
    result = classify_claim(claim, Alignment(claim=claim, source_claim=source, score=0.5))
    assert result.label == "overstated"


def test_classify_overstated_on_inflated_percentage():
    claim = extract_claims("The intervention reduced tumor size by 50%.", origin="summary")[0]
    source = extract_claims(
        "The intervention reduced tumor size by approximately 15%.", origin="paper")[0]
    result = classify_claim(claim, Alignment(claim=claim, source_claim=source, score=0.6))
    assert result.label == "overstated"


def test_numeric_check_ignores_non_effect_size_numbers():
    # Only percentages are compared. A bare number can be a sample size, a
    # p-value, or a year; comparing those produces confident nonsense, so an
    # honest summary that merely adds a cohort size must not be flagged.
    from faithful.classify import _numeric_inflation

    assert _numeric_inflation("15% of 200 mice responded", "15% of mice responded") is None
    assert _numeric_inflation("p = 0.03 was seen", "p = 0.001 was seen") is None
    assert _numeric_inflation("in 2020, 15% responded", "in 2019, 15% responded") is None
    # Deflation and equal figures are not inflation either.
    assert _numeric_inflation("reduced by 5%", "reduced by approximately 15%") is None
    assert _numeric_inflation("reduced by 15%", "reduced by approximately 15%") is None
    # The real catch still fires, including "percent" spelled out.
    assert _numeric_inflation("reduced by 50%", "reduced by approximately 15%") is not None
    assert _numeric_inflation("reduced by 50 percent", "reduced by 15 percent") is not None


def test_classify_contradicted_on_negation_mismatch():
    claim = extract_claims("Colonized mice gained body weight.", origin="summary")[0]
    source = extract_claims(
        "There was no significant change in body weight between groups.", origin="paper")[0]
    result = classify_claim(claim, Alignment(claim=claim, source_claim=source, score=0.4))
    assert result.label == "contradicted"


def test_classify_supported_on_faithful_restatement():
    claim = extract_claims(
        "The microbe was associated with reduced colonic inflammation.", origin="summary")[0]
    source = extract_claims(
        "The microbe was associated with reduced colonic inflammation in mice.", origin="paper")[0]
    result = classify_claim(claim, Alignment(claim=claim, source_claim=source, score=0.6))
    assert result.label == "supported"


# ---- Full pipeline ------------------------------------------------------- #

def test_pipeline_on_empty_inputs():
    result = run_pipeline("", "")
    assert result.results == []
    assert result.omissions == []
    assert result.counts() == {}


def test_pipeline_on_bundled_sample_catches_each_failure_mode():
    with open(os.path.join(EXAMPLES, "sample_paper.txt"), encoding="utf-8") as f:
        paper = f.read()
    with open(os.path.join(EXAMPLES, "sample_summary.txt"), encoding="utf-8") as f:
        summary = f.read()

    result = run_pipeline(paper, summary)
    labels = {r.label for r in result.results}
    # The sample is engineered to exercise all four labels.
    assert {"supported", "unsupported", "overstated", "contradicted"} <= labels
    # And to drop at least one caveat.
    assert len(result.dropped_caveats()) >= 1
    # Provenance is preserved on each result.
    assert all(r.align_method == "lexical-overlap" for r in result.results)
    assert all(r.classify_method == "rule-based" for r in result.results)


def _run_without_pytest() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {test.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_without_pytest())
