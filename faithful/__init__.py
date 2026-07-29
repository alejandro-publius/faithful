"""Faithful — check AI-generated summaries of scientific papers against the source.

Faithful splits a source paper and an AI summary into discrete claims, aligns
each summary claim to the passage that should support it, and labels each claim
as supported, unsupported, overstated, or contradicted — flagging what the
summary adds, drops, overstates, or contradicts.

This is an early prototype. Stage 1 (extraction) is implemented for real with a
dependency-free segmenter. Stages 2 (alignment) and 3 (classification) ship with
transparent heuristic defaults and clearly marked LLM-backed extension points.

Typical use::

    from faithful import run_pipeline

    result = run_pipeline(paper_text, summary_text)
    for r in result.results:
        print(r.label, r.claim.text)
"""

from __future__ import annotations

from .align import Alignment, align_claim, align_claims
from .classify import LABELS, Classification, classify_claim, classify_claims
from .extract import Claim, extract_claims, split_sentences
from .monitor import (
    Contribution,
    MonitorReport,
    MonitorWeights,
    monitor_summary,
    score_result,
)
from .pipeline import ClaimResult, Omission, PipelineResult, run_pipeline

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # extraction
    "Claim",
    "extract_claims",
    "split_sentences",
    # alignment
    "Alignment",
    "align_claim",
    "align_claims",
    # classification
    "LABELS",
    "Classification",
    "classify_claim",
    "classify_claims",
    # pipeline
    "ClaimResult",
    "Omission",
    "PipelineResult",
    "run_pipeline",
    # monitor (AI-control layer)
    "MonitorWeights",
    "Contribution",
    "MonitorReport",
    "score_result",
    "monitor_summary",
]
