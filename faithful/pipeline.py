"""The Faithful pipeline — runs all three stages and returns structured results.

    extract  ->  align  ->  classify

plus a source-side pass that flags *omissions*: source claims (especially
caveats and limitations) that no summary claim covers. Together these catch the
four things a summary can get wrong — claims it **adds**, **drops**,
**overstates**, or **contradicts**.

Each stage is pluggable. ``run_pipeline`` accepts an ``align_fn`` and a
``classify_fn`` so the heuristic defaults can be swapped for the LLM-backed
versions (see the TODO extension points in :mod:`faithful.align` and
:mod:`faithful.classify`) without touching this orchestration code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from .align import DEFAULT_THRESHOLD, Alignment, align_claim, overlap_score
from .classify import Classification, classify_claim
from .extract import Claim, extract_claims

__all__ = ["ClaimResult", "Omission", "PipelineResult", "run_pipeline"]

# Default score below which a source claim is treated as omitted. This is a
# SEPARATE knob from the alignment threshold on purpose: omission detection uses
# the unweighted overlap_score (plain Jaccard), whereas align_claim uses the
# IDF-weighted score, so the two operate on different scales and must be tuned
# independently.
OMISSION_THRESHOLD = 0.12

# Cue words that mark a source sentence as a caveat / limitation worth not
# dropping. Used only to prioritise which omissions to surface first.
_CAVEAT_CUES = {
    "preliminary", "limitation", "limitations", "caution", "caveat",
    "not", "no", "small", "larger", "further", "validated", "validate",
    "unclear", "unknown", "may", "might", "however", "although",
}

# Function signatures for the pluggable stages.
AlignFn = Callable[[Claim, list[Claim], float], Alignment]
ClassifyFn = Callable[[Claim, Alignment], Classification]


@dataclass
class ClaimResult:
    """The full verdict for one summary claim.

    ``align_method`` and ``classify_method`` record which stage-2/stage-3
    backend produced this result (e.g. ``"lexical-overlap"`` / ``"rule-based"``
    for the heuristics, ``"llm"`` once a model backend is wired in), so callers
    who swap in a custom ``align_fn``/``classify_fn`` can see the provenance.
    """

    claim: Claim
    label: str
    rationale: str
    evidence: Claim | None
    alignment_score: float
    align_method: str = "lexical-overlap"
    classify_method: str = "rule-based"


@dataclass
class Omission:
    """A source claim that no summary claim covered (a potential dropped point)."""

    source_claim: Claim
    is_caveat: bool
    best_score: float


@dataclass
class PipelineResult:
    """Structured output of :func:`run_pipeline`."""

    paper_claims: list[Claim]
    summary_claims: list[Claim]
    results: list[ClaimResult]
    omissions: list[Omission] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        """Return a label -> count summary over the summary-claim verdicts."""
        out: dict[str, int] = {}
        for r in self.results:
            out[r.label] = out.get(r.label, 0) + 1
        return out

    def dropped_caveats(self) -> list[Omission]:
        """Omissions that look like caveats/limitations (the important drops)."""
        return [o for o in self.omissions if o.is_caveat]


def _is_caveat(text: str) -> bool:
    words = {w.lower() for w in re.findall(r"[A-Za-z']+", text)}
    return bool(words & _CAVEAT_CUES)


def _find_omissions(
    paper_claims: list[Claim],
    summary_claims: list[Claim],
    threshold: float,
) -> list[Omission]:
    """Flag source claims that no summary claim covers above ``threshold``.

    Note: this uses the unweighted :func:`overlap_score`, so ``threshold`` here
    is on a different scale than the alignment threshold in :func:`align_claim`.
    """
    omissions: list[Omission] = []
    for source in paper_claims:
        best = max(
            (overlap_score(source.text, s.text) for s in summary_claims),
            default=0.0,
        )
        if best < threshold:
            omissions.append(
                Omission(
                    source_claim=source,
                    is_caveat=_is_caveat(source.text),
                    best_score=best,
                )
            )
    return omissions


def run_pipeline(
    paper_text: str,
    summary_text: str,
    align_fn: AlignFn = align_claim,
    classify_fn: ClassifyFn = classify_claim,
    threshold: float = DEFAULT_THRESHOLD,
    omission_threshold: float = OMISSION_THRESHOLD,
) -> PipelineResult:
    """Run extraction, alignment, and classification end to end.

    Args:
        paper_text: The source paper text (e.g. an abstract).
        summary_text: The AI-generated summary to check.
        align_fn: Stage-2 function. Defaults to the lexical-overlap heuristic;
            pass :func:`faithful.align.align_claim_llm` once implemented.
        classify_fn: Stage-3 function. Defaults to the rule-based placeholder;
            pass :func:`faithful.classify.classify_claim_llm` once implemented.
        threshold: Alignment score (IDF-weighted) below which a summary claim is
            treated as unsupported.
        omission_threshold: Coverage score (unweighted overlap) below which a
            source claim is treated as omitted. Kept separate from ``threshold``
            because the two scores are on different scales.

    Returns:
        A :class:`PipelineResult` with per-claim verdicts and detected omissions.
    """
    # Stage 1 — Extraction (real).
    paper_claims = extract_claims(paper_text, origin="paper")
    summary_claims = extract_claims(summary_text, origin="summary")

    results: list[ClaimResult] = []
    for claim in summary_claims:
        # Stage 2 — Alignment.
        alignment = align_fn(claim, paper_claims, threshold)
        # Stage 3 — Classification.
        classification = classify_fn(claim, alignment)
        results.append(
            ClaimResult(
                claim=claim,
                label=classification.label,
                rationale=classification.rationale,
                evidence=classification.evidence,
                alignment_score=alignment.score,
                align_method=alignment.method,
                classify_method=classification.method,
            )
        )

    # Source-side pass — detect dropped points (omissions).
    omissions = _find_omissions(paper_claims, summary_claims, omission_threshold)

    return PipelineResult(
        paper_claims=paper_claims,
        summary_claims=summary_claims,
        results=results,
        omissions=omissions,
    )
