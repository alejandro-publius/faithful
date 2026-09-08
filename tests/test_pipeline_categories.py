"""End-to-end behavior tests for the four claim categories Faithful claims to
detect (README, "What Faithful measures"): added claims (``unsupported``),
dropped claims (omissions/caveats), overstated claims, and contradicted claims
-- plus a clean summary that should flag nothing.

Unlike ``tests/test_pipeline.py`` and ``tests/test_classify.py``, which mostly
exercise ``classify_claim`` directly against a hand-built ``Alignment``, these
tests run the *full* ``run_pipeline(paper_text, summary_text)`` on small
source/summary fixture pairs where the right answer is unambiguous, the way a
real caller uses the library. Each fixture below was chosen so exactly one
failure mode is present (or, for the clean case, none).

Run with either:

    pytest
    python tests/test_pipeline_categories.py   # falls back to a tiny runner
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from faithful import run_pipeline

# A small source paragraph, deliberately in a different domain (agronomy
# rather than the bundled microbiome sample) so these fixtures are independent
# of examples/sample_paper.txt. Contains one hedged claim, one clean factual
# claim, one null result, and one caveat -- one anchor per category below.
PAPER = (
    "Field trials measured crop yield in irrigated and rainfed plots. "
    "Irrigated plots showed a 12% increase in yield compared to rainfed plots. "
    "Soil nitrogen levels were associated with improved crop yield. "
    "There was no significant difference in pest damage between the two groups. "
    "These results are preliminary and have not been validated across multiple "
    "growing seasons."
)


def test_detects_added_claim_as_unsupported():
    """A summary claim with no counterpart anywhere in the source is 'added'."""
    summary = (
        "This is the first study to prove irrigation reduces greenhouse gas "
        "emissions by half."
    )
    result = run_pipeline(PAPER, summary)
    assert len(result.results) == 1
    r = result.results[0]
    assert r.label == "unsupported"
    assert r.evidence is None
    assert r.alignment_score == 0.0


def test_detects_dropped_claim_as_omission():
    """A source caveat the summary never mentions is flagged as a dropped caveat."""
    # Restates only the yield finding; never touches the pest-damage null
    # result or the preliminary/not-validated caveat.
    summary = "Irrigated plots showed a 12% increase in yield compared to rainfed plots."
    result = run_pipeline(PAPER, summary)

    dropped_texts = {o.source_claim.text for o in result.dropped_caveats()}
    assert (
        "These results are preliminary and have not been validated across "
        "multiple growing seasons." in dropped_texts
    )
    # The restated claim itself must not be reported as dropped.
    yield_claim = "Irrigated plots showed a 12% increase in yield compared to rainfed plots."
    assert yield_claim not in dropped_texts


def test_detects_overstated_claim():
    """A hedged source finding stated as a certainty is 'overstated'."""
    summary = "Irrigation guarantees a dramatic increase in crop yield."
    result = run_pipeline(PAPER, summary)
    assert len(result.results) == 1
    r = result.results[0]
    assert r.label == "overstated"
    assert r.evidence is not None
    assert "yield" in r.evidence.text.lower()


def test_detects_contradicted_claim():
    """Asserting the reverse of a source null result is 'contradicted'."""
    summary = "There was a significant difference in pest damage between the two groups."
    result = run_pipeline(PAPER, summary)
    assert len(result.results) == 1
    r = result.results[0]
    assert r.label == "contradicted"
    assert r.evidence is not None
    assert "pest damage" in r.evidence.text.lower()


def test_clean_summary_flags_nothing():
    """A summary that faithfully restates every source claim, caveats included,
    should be all-supported with no dropped caveats and no other omissions."""
    summary = PAPER  # restates the source verbatim, claim for claim
    result = run_pipeline(PAPER, summary)

    assert len(result.results) == len(result.paper_claims) == 5
    assert all(r.label == "supported" for r in result.results)
    assert result.omissions == []
    assert result.dropped_caveats() == []


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
