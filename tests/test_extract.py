"""Basic unit tests for stage 1 (claim segmentation).

Run with either:

    pytest
    python tests/test_extract.py   # falls back to a tiny runner if pytest is absent
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from faithful.extract import Claim, extract_claims, split_sentences


def test_splits_into_sentences():
    text = "The microbiome matters. Butyrate is a short-chain fatty acid. It is produced by bacteria."
    sentences = [s for s, _, _ in split_sentences(text)]
    assert len(sentences) == 3
    assert sentences[0] == "The microbiome matters."


def test_does_not_split_on_species_abbreviation():
    # "R. intestinalis" must stay in one claim, not break at "R."
    text = "We studied R. intestinalis in mice. It reduced inflammation."
    claims = extract_claims(text, origin="paper")
    assert len(claims) == 2
    assert "R. intestinalis" in claims[0].text


def test_does_not_split_on_decimal():
    # The period inside "0.03" must not end a sentence.
    text = "The effect was clear (p = 0.03). We report it here."
    sentences = [s for s, _, _ in split_sentences(text)]
    assert len(sentences) == 2
    assert "0.03" in sentences[0]


def test_strips_leading_section_label():
    text = "Results: The treatment reduced inflammation in the treated group."
    claims = extract_claims(text, origin="paper")
    assert len(claims) == 1
    assert claims[0].text.startswith("The treatment")


def test_splits_on_semicolons():
    text = "Findings are preliminary; larger studies are needed for confirmation."
    claims = extract_claims(text, origin="paper", split_semicolons=True)
    assert len(claims) == 2


def test_filters_short_fragments():
    text = "Yes. The gut microbiome influences host immune regulation in mice."
    claims = extract_claims(text, origin="paper")
    # "Yes." is too short to be a claim and should be dropped.
    assert all(len(c.text) >= 15 for c in claims)
    assert len(claims) == 1


def test_does_not_split_on_multiword_abbreviation():
    # "et al." and "e.g." must not end a sentence.
    assert len([s for s, _, _ in split_sentences(
        "Smith et al. reported reduced inflammation in the treated mice.")]) == 1
    assert len([s for s, _, _ in split_sentences(
        "Several taxa, e.g. Roseburia, were enriched in the treated group.")]) == 1


def test_splits_after_closing_quote():
    # The period before a closing quote still ends the sentence.
    sentences = [s for s, _, _ in split_sentences(
        'The authors wrote "it works." Then they moved on.')]
    assert len(sentences) == 2
    assert sentences[0].endswith('"')


def test_handles_ellipsis_and_other_terminators():
    assert len([s for s, _, _ in split_sentences(
        "The results were unclear... We repeated the experiment twice.")]) == 2
    assert len([s for s, _, _ in split_sentences(
        "Does the microbe help? We think it truly does!")]) == 2


def test_no_space_after_period_is_not_split():
    # A missing space after a period is a known limitation: the segmenter keeps
    # it as one sentence rather than guessing a boundary.
    sentences = [s for s, _, _ in split_sentences(
        "The mice recovered.Another cohort was studied here.")]
    assert len(sentences) == 1


def test_empty_and_whitespace_input():
    assert split_sentences("") == []
    assert extract_claims("   \n  ", origin="paper") == []


def test_claim_metadata_is_populated():
    text = "The gut microbiome influences host metabolic regulation in mice."
    claims = extract_claims(text, origin="summary")
    assert isinstance(claims[0], Claim)
    assert claims[0].origin == "summary"
    assert claims[0].id == "summary-0"
    assert claims[0].index == 0
    # Offsets should point back into the original text.
    assert text[claims[0].char_start:claims[0].char_end] == claims[0].text


def _run_without_pytest() -> int:
    """Minimal fallback runner so the file works even if pytest isn't installed."""
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
