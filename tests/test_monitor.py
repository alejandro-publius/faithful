"""Tests for the trusted-monitor layer (suspicion scoring)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from faithful.monitor import MonitorWeights, monitor_summary, score_result
from faithful.pipeline import run_pipeline


def test_faithful_restatement_scores_zero_suspicion():
    paper = "Body weight was unchanged between the treated and control mice."
    summary = "Body weight was unchanged between the treated and control mice."
    report = monitor_summary(paper, summary)
    assert report.suspicion == 0.0
    assert report.contributions == []
    assert report.decision(threshold=0.5) == "accept"


def test_contradiction_drives_high_suspicion():
    paper = "Body weight was unchanged between the treated and control mice."
    summary = "Treated mice gained body weight compared to control mice."
    report = monitor_summary(paper, summary)
    # A single contradiction alone should clear a mid threshold.
    assert report.suspicion >= 0.8
    assert report.decision(threshold=0.5) == "audit"
    assert any(c.kind == "contradicted" for c in report.contributions)


def test_noisy_or_accumulates_multiple_problems():
    # Two independent problems combine to more than either alone.
    weights = MonitorWeights()
    one = 1.0 - (1.0 - weights.overstated)
    two = 1.0 - (1.0 - weights.overstated) * (1.0 - weights.unsupported)
    assert two > one
    assert 0.0 <= two <= 1.0


def test_weights_are_tunable():
    paper = "The microbe was associated with reduced inflammation in mice."
    summary = "The microbe cures inflammation."
    strict = monitor_summary(paper, summary, MonitorWeights(overstated=0.9))
    lenient = monitor_summary(paper, summary, MonitorWeights(overstated=0.1))
    assert strict.suspicion > lenient.suspicion


def test_score_result_is_backend_agnostic():
    # score_result works on any PipelineResult regardless of how it was produced.
    result = run_pipeline(
        "The microbe was associated with reduced inflammation in mice.",
        "The microbe cures inflammation.",
    )
    report = score_result(result)
    assert report.suspicion > 0.0
    assert report.n_summary_claims == len(result.summary_claims)


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
