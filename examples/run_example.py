"""Run the Faithful pipeline on the bundled sample and print a per-claim report.

    python examples/run_example.py

Uses only the Python standard library and the local ``faithful`` package — no
API keys, no network, no build step.
"""

from __future__ import annotations

import os
import sys

# Make the repo root importable when run directly (python examples/run_example.py).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from faithful import run_pipeline  # noqa: E402  (import after sys.path setup)

EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))

# Symbols for each label, kept ASCII-friendly for any terminal.
LABEL_MARK = {
    "supported": "[+] SUPPORTED  ",
    "unsupported": "[?] UNSUPPORTED",
    "overstated": "[!] OVERSTATED ",
    "contradicted": "[x] CONTRADICTED",
}


def _read(name: str) -> str:
    with open(os.path.join(EXAMPLES_DIR, name), encoding="utf-8") as f:
        return f.read()


def main() -> None:
    paper = _read("sample_paper.txt")
    summary = _read("sample_summary.txt")

    result = run_pipeline(paper, summary)

    rule = "=" * 72
    print(rule)
    print("FAITHFUL — claim-level check of an AI summary against its source")
    print(rule)
    print(
        f"Extracted {len(result.paper_claims)} source claims and "
        f"{len(result.summary_claims)} summary claims.\n"
    )

    print("PER-CLAIM VERDICTS (summary claims)")
    print("-" * 72)
    for i, r in enumerate(result.results, start=1):
        mark = LABEL_MARK.get(r.label, r.label.upper())
        print(f"{i}. {mark}  (align {r.alignment_score:.2f})")
        print(f"   claim   : {r.claim.text}")
        if r.evidence is not None:
            print(f"   source  : {r.evidence.text}")
        print(f"   why     : {r.rationale}")
        print()

    dropped = result.dropped_caveats()
    other = [o for o in result.omissions if not o.is_caveat]

    print("DROPPED CAVEATS (source caveats no summary claim covered)")
    print("-" * 72)
    if not dropped:
        print("None detected.\n")
    else:
        for o in dropped:
            print(f"[-] {o.source_claim.text}\n")

    if other:
        print("OTHER UNCOVERED SOURCE POINTS")
        print("-" * 72)
        for o in other:
            print(f"[-] {o.source_claim.text}\n")

    print("SUMMARY")
    print("-" * 72)
    counts = result.counts()
    for label in ("supported", "unsupported", "overstated", "contradicted"):
        print(f"  {label:<13}: {counts.get(label, 0)}")
    print(f"  caveats dropped: {len(dropped)}")
    print(rule)
    print(
        "Note: labels come from heuristic placeholders (lexical overlap + rules), "
        "not a\nvalidated model. See the README for what is implemented vs. planned."
    )
    print(rule)


if __name__ == "__main__":
    main()
