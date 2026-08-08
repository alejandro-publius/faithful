"""Characterization tests for the rule-based classifier (stage 3).

These pin the classifier's CURRENT behavior — including its documented quirks —
so that any future change to the rules or lexicons shows up as a test failure
rather than a silent shift in labels. They are a record of what the scorer does,
not an endorsement that each behavior is ideal; quirks are marked as such below.

Run with either:

    pytest
    python tests/test_classify.py   # falls back to a tiny runner if pytest is absent
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from faithful.align import Alignment
from faithful.classify import LABELS, classify_claim
from faithful.extract import Claim


def _claim(text: str, origin: str = "summary", index: int = 0) -> Claim:
    return Claim(
        id=f"{origin}-{index}",
        text=text,
        origin=origin,
        index=index,
        char_start=0,
        char_end=len(text),
    )


def _classify(claim_text: str, source_text: str | None, score: float = 0.5):
    claim = _claim(claim_text)
    source = _claim(source_text, origin="paper") if source_text is not None else None
    alignment = Alignment(
        claim=claim, source_claim=source, score=score if source else 0.05
    )
    return classify_claim(claim, alignment)


# ---- Label 1: unsupported ------------------------------------------------ #

def test_no_aligned_source_is_unsupported():
    result = _classify("Butyrate cures obesity.", None)
    assert result.label == "unsupported"
    assert result.evidence is None
    # The rationale carries the failing overlap score for the human reviewer.
    assert "0.05" in result.rationale


# ---- Label 2: contradicted ----------------------------------------------- #

def test_source_negates_what_the_summary_asserts():
    result = _classify(
        "Butyrate reduced weight gain.",
        "Butyrate did not reduce weight gain.",
    )
    assert result.label == "contradicted"
    assert "source negates" in result.rationale


def test_summary_negates_what_the_source_asserts():
    result = _classify(
        "Butyrate did not reduce weight gain.",
        "Butyrate reduced weight gain.",
    )
    assert result.label == "contradicted"
    assert "source asserts" in result.rationale


def test_contradiction_requires_a_shared_content_word():
    """QUIRK (pinned): negation mismatch alone is not enough.

    The rule only calls a contradiction when the two sides share a distinctive
    token to anchor it. With no shared topic word, a negated claim against an
    unrelated source falls through to 'supported' — arguably wrong, but it is
    what the rule does today, and this test exists so a change is deliberate.
    """
    result = _classify("The drug failed.", "Zebrafish thrived in warm water.")
    assert result.label == "supported"


def test_contradiction_outranks_overstatement():
    """Decision order pin: rule 2 (negation) fires before rule 3 (strength).

    This claim both negation-mismatches the source AND drops its hedge; the
    classifier must report the contradiction, not the overstatement.
    """
    result = _classify(
        "Butyrate did not cure obesity.",
        "Butyrate was associated with weight loss.",
    )
    assert result.label == "contradicted"


# ---- Label 3: overstated ------------------------------------------------- #

def test_strength_cue_absent_from_source_is_overstated():
    result = _classify(
        "Butyrate cures obesity in mice.",
        "Butyrate was associated with reduced weight gain.",
    )
    assert result.label == "overstated"
    assert "'cures'" in result.rationale
    assert "associated" in result.rationale  # names the source's hedge


def test_intensifier_counts_as_strength():
    result = _classify(
        "Butyrate dramatically shrank tumors.",
        "Butyrate shrank tumors by a modest amount.",
    )
    assert result.label == "overstated"
    assert "'dramatically'" in result.rationale
    # QUIRK (pinned): with no hedge in the source, the rationale reads
    # "hedged (no strong claim)" rather than omitting the hedge clause.
    assert "no strong claim" in result.rationale


def test_dropping_the_sources_hedge_is_overstated():
    result = _classify(
        "Butyrate reduced weight gain.",
        "Butyrate may be linked to reduced weight gain.",
    )
    assert result.label == "overstated"
    assert "drops the qualification" in result.rationale


def test_percentage_inflation_is_overstated():
    result = _classify(
        "Weight fell by 50%.",
        "Weight fell by approximately 15%.",
    )
    assert result.label == "overstated"
    assert "(50)" in result.rationale and "(15)" in result.rationale


def test_percentage_within_tolerance_is_not_inflation():
    # 15.5 vs 15 is inside the 5% same-figure tolerance.
    result = _classify("Weight fell by 15.5%.", "Weight fell by approximately 15%.")
    assert result.label == "supported"


def test_percentage_below_ratio_gate_is_not_inflation():
    # 17 vs 15 is a real difference but below the 1.2x materiality ratio.
    result = _classify("Weight fell by 17%.", "Weight fell by 15%.")
    assert result.label == "supported"


def test_percentage_deflation_is_not_flagged():
    """QUIRK (pinned): only inflation is flagged, never deflation.

    A summary understating an effect (10% where the source says 15%) passes as
    'supported'. Deliberate scope choice in _numeric_inflation; pinned so that
    widening it is a conscious decision.
    """
    result = _classify("Weight fell by 10%.", "Weight fell by 15%.")
    assert result.label == "supported"


# ---- Label 4: supported -------------------------------------------------- #

def test_plain_paraphrase_is_supported():
    result = _classify(
        "Butyrate reduced weight gain in mice.",
        "Dietary butyrate reduced weight gain in the mice.",
    )
    assert result.label == "supported"
    assert result.evidence is not None


# ---- Contract ------------------------------------------------------------ #

def test_every_label_is_from_the_documented_set():
    cases = [
        ("Butyrate cures obesity.", None),
        ("Butyrate reduced weight.", "Butyrate did not reduce weight."),
        ("Butyrate cures obesity.", "Butyrate was associated with weight loss."),
        ("Butyrate reduced weight.", "Butyrate reduced weight."),
    ]
    for claim_text, source_text in cases:
        result = _classify(claim_text, source_text)
        assert result.label in LABELS
        assert result.method == "rule-based"
        assert result.rationale  # never empty


if __name__ == "__main__":  # tiny runner so the file works without pytest
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failures else 0)
