"""Stage 2 — Alignment.

For each summary claim, find the passage in the source paper that best supports
it — or determine that no adequate supporting passage exists.

The default implementation is a transparent lexical-overlap heuristic: it scores
each candidate source claim against the summary claim using distinctive-token
overlap (a Jaccard score after stop-word removal) and returns the best match
above a threshold. This is deliberately simple and deterministic so the pipeline
runs end to end with no dependencies. The real signal — paraphrase, entailment,
numeric equivalence — needs a model; that LLM-backed aligner is a clearly marked
extension point (:func:`align_claim_llm`).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .extract import Claim

__all__ = [
    "Alignment",
    "tokenize",
    "overlap_score",
    "weighted_overlap_score",
    "align_claim",
    "align_claims",
    "align_claim_llm",
]

# Minimal English stop-word list. Removing these focuses the overlap score on
# distinctive content words (species names, outcomes, directions of effect).
_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "with", "by",
    "and", "or", "but", "as", "is", "are", "was", "were", "be", "been",
    "being", "that", "this", "these", "those", "it", "its", "they", "them",
    "we", "our", "which", "who", "whom", "from", "into", "than", "then",
    "there", "their", "also", "between", "over", "under", "up", "down",
    "out", "about", "compared", "relative", "showed", "show", "shown",
    "study", "results", "result", "found", "observed",
}

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z-]*")

# Default overlap score below which a summary claim is treated as having no
# adequate supporting passage. Tunable; documented in the README as a heuristic
# knob, not a validated operating point.
DEFAULT_THRESHOLD = 0.12


def tokenize(text: str) -> set[str]:
    """Lowercase, drop stop-words, and return the set of content tokens."""
    tokens = {t.lower() for t in _TOKEN_RE.findall(text)}
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 1}


def overlap_score(a: str, b: str) -> float:
    """Unweighted Jaccard overlap of the distinctive tokens in ``a`` and ``b``."""
    ta, tb = tokenize(a), tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _inverse_document_frequency(source_claims: list[Claim]) -> dict[str, float]:
    """Smoothed IDF weight for each token across the source claims.

    Tokens that appear in many source sentences (``mice``, ``controls``) are
    common and carry little discriminative signal; distinctive tokens
    (``butyrate``, ``weight``) carry more. Weighting the overlap by IDF stops a
    claim from aligning to whichever sentence merely shares boilerplate words.
    """
    n = len(source_claims)
    df: dict[str, int] = {}
    for claim in source_claims:
        for token in tokenize(claim.text):
            df[token] = df.get(token, 0) + 1
    return {tok: math.log((n + 1) / (count + 1)) + 1.0 for tok, count in df.items()}


def weighted_overlap_score(a: str, b: str, idf: dict[str, float]) -> float:
    """IDF-weighted Jaccard overlap of the tokens in ``a`` and ``b``.

    Tokens missing from ``idf`` (e.g. words only in the summary) get the maximum
    observed weight, so novel content correctly widens the union and lowers the
    score for unsupported claims.
    """
    ta, tb = tokenize(a), tokenize(b)
    if not ta or not tb:
        return 0.0
    default = max(idf.values(), default=1.0)

    def w(tok: str) -> float:
        return idf.get(tok, default)

    inter = sum(w(t) for t in (ta & tb))
    union = sum(w(t) for t in (ta | tb))
    return inter / union if union else 0.0


@dataclass
class Alignment:
    """The result of aligning one summary claim to the source paper.

    Attributes:
        claim: The summary claim being aligned.
        source_claim: The best-matching source claim, or ``None`` if nothing
            scored above ``threshold``.
        score: The overlap score of the best match (0.0 if there was none).
        method: Identifier for the aligner that produced this result.
    """

    claim: Claim
    source_claim: Claim | None
    score: float
    method: str = "lexical-overlap"

    @property
    def is_supported_by_source(self) -> bool:
        """True if an adequate supporting passage was found."""
        return self.source_claim is not None


def align_claim(
    claim: Claim,
    source_claims: list[Claim],
    threshold: float = DEFAULT_THRESHOLD,
) -> Alignment:
    """Align one summary ``claim`` against all ``source_claims`` (heuristic).

    Returns an :class:`Alignment` whose ``source_claim`` is the highest-scoring
    source passage, or ``None`` if the best score is below ``threshold``.
    """
    idf = _inverse_document_frequency(source_claims)
    best_claim: Claim | None = None
    best_score = 0.0
    for candidate in source_claims:
        score = weighted_overlap_score(claim.text, candidate.text, idf)
        if score > best_score:
            best_score, best_claim = score, candidate

    if best_score < threshold:
        return Alignment(claim=claim, source_claim=None, score=best_score)
    return Alignment(claim=claim, source_claim=best_claim, score=best_score)


def align_claims(
    summary_claims: list[Claim],
    source_claims: list[Claim],
    threshold: float = DEFAULT_THRESHOLD,
) -> list[Alignment]:
    """Align every summary claim against the source paper."""
    return [align_claim(c, source_claims, threshold) for c in summary_claims]


# --------------------------------------------------------------------------- #
# TODO (extension point): LLM-backed alignment.
#
# Lexical overlap cannot see paraphrase, synonymy, or numeric equivalence
# ("a third lower" ~= "32% reduction"). A model can rank candidate passages by
# entailment and return the true supporting sentence with a calibrated score.
# Wire a model call in here and return the same Alignment type; the pipeline is
# already written to accept any function with this signature. No external
# service is required for the repo to run.
# --------------------------------------------------------------------------- #
def align_claim_llm(
    claim: Claim,
    source_claims: list[Claim],
    threshold: float = DEFAULT_THRESHOLD,
    model: str = "claude-fable-5",
    client: object | None = None,
) -> Alignment:
    """Planned LLM-backed aligner (not yet implemented).

    Intended contract: ask a model which source claim (if any) supports the
    summary claim, and with what confidence; return an :class:`Alignment` with
    ``method="llm"``. Drop-in compatible with :func:`align_claim`.

    Raises:
        NotImplementedError: Always, until a backend is wired in.
    """
    raise NotImplementedError(
        "LLM-backed alignment is a planned extension point. "
        "Use align_claim() for the dependency-free heuristic path."
    )
