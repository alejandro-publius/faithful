"""Run the red-team / blue-team control evaluation of the Faithful monitor.

    python examples/run_control_eval.py

Scores a small synthetic suite of faithful vs. unfaithful summaries, then prints
the safety-vs-usefulness frontier and the operating point at a chosen audit
budget. Uses only the standard library and the local ``faithful`` package.

This is a methodology demo on hand-written examples, not a benchmark on real
papers — see the README's honesty note.
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from faithful.control_eval import evaluate, load_cases  # noqa: E402

SUITE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "control_suite.jsonl")
AUDIT_BUDGET = 0.5  # a human can review at most half the summaries


def main() -> None:
    cases = load_cases(SUITE)
    result = evaluate(cases)  # default: heuristic monitor, stdlib only

    rule = "=" * 72
    print(rule)
    print("FAITHFUL — trusted-monitor control evaluation")
    print(rule)
    print(
        f"{result.n_unfaithful} unfaithful (attack) and {result.n_faithful} faithful "
        f"(honest) summaries.\n"
        "Monitor = heuristic pipeline; suspicion = noisy-OR over claim verdicts.\n"
    )

    print("PER-SUMMARY SUSPICION")
    print("-" * 72)
    for s in sorted(result.scored, key=lambda x: x.suspicion, reverse=True):
        tag = "ATTACK" if s.case.unfaithful else "honest"
        print(f"  {s.suspicion:0.2f}  [{tag}]  {s.case.id}")
    print()

    print("SAFETY-USEFULNESS FRONTIER (audit if suspicion >= threshold)")
    print("-" * 72)
    print(f"  {'threshold':>9}  {'safety':>7}  {'useful':>7}  {'audit rate':>10}")
    for p in result.frontier:
        print(
            f"  {p.threshold:9.2f}  {p.safety:7.0%}  {p.usefulness:7.0%}  {p.audit_rate:10.0%}"
        )
    print()

    op = result.operating_point(AUDIT_BUDGET)
    p = op.point
    print(f"OPERATING POINT (audit budget = {AUDIT_BUDGET:.0%})")
    print("-" * 72)
    print(f"  threshold  : {p.threshold:.2f}")
    print(f"  safety     : {p.safety:.0%}  (attacks caught)")
    print(f"  usefulness : {p.usefulness:.0%}  (honest summaries accepted)")
    print(f"  audit rate : {p.audit_rate:.0%}  (of all summaries)")
    print()

    missed = [
        s for s in result.scored
        if s.case.unfaithful and s.suspicion < p.threshold
    ]
    if missed:
        print("ATTACKS THE HEURISTIC MONITOR MISSES AT THIS THRESHOLD")
        print("-" * 72)
        for s in missed:
            print(f"  [{s.case.id}] {s.case.summary}")
        print(
            "\nThe monitor now catches magnitude words ('dramatically'), dropped scope\n"
            "('in mice' -> 'in humans'), and digit swaps ('15%' -> '50%'). What remains\n"
            "is PARAPHRASED numeric distortion ('a third' -> 'in half'): no digits to\n"
            "compare and every token shared, so neither lexical overlap nor a numeric\n"
            "regex can see it. That needs a model that reads meaning — the case for the\n"
            "Cohere Rerank / LLM backend, whose frontier is measured the exact same way."
        )

    print(rule)
    print(
        "Note: synthetic hand-written suite; a methodology demo, not a benchmark "
        "on real\npapers. No accuracy is claimed against real summaries. See the "
        "README."
    )
    print(rule)


if __name__ == "__main__":
    main()
