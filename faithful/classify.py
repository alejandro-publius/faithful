"""Stage 3 — Classification.

Given a summary claim and its aligned source passage, assign one of four labels:

    supported     — the source backs the claim as stated.
    unsupported   — no adequate source passage was found (an added claim).
    overstated    — the source is weaker/more hedged than the claim (e.g. the
                    claim says "cures" where the source says "was associated
                    with a reduction").
    contradicted  — the source asserts the opposite (e.g. a null result the
                    claim reports as a positive effect).

The default implementation is a rule-based placeholder built on small lexicons
of hedging, strength, and negation cues. It is intentionally simple and
explainable — every decision comes with a short rationale — but it is a
placeholder, not a validated classifier. Real classification needs a model to
weigh evidence and read numbers; that LLM-backed classifier is a clearly marked
extension point (:func:`classify_claim_llm`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .align import Alignment
from .extract import Claim

__all__ = [
    "LABELS",
    "Classification",
    "classify_claim",
    "classify_claims",
    "classify_claim_llm",
]

LABELS = ("supported", "unsupported", "overstated", "contradicted")

# Words that assert strong causation, certainty, or universality. When a summary
# claim uses one of these and the aligned source does not, that is a signal of
# overstatement.
_STRENGTH_CUES = {
    "cure", "cures", "cured", "cure-all", "prove", "proves", "proven", "proof",
    "confirm", "confirms", "confirmed", "cause", "causes", "caused", "causal",
    "eliminate", "eliminates", "eliminated", "eradicate", "eradicates",
    "guarantee", "guarantees", "prevents", "prevent", "reverses", "reverse",
    "always", "never", "all", "every", "completely", "definitively",
    "unequivocally", "established", "demonstrates", "demonstrated",
}

# Intensity adverbs/adjectives that inflate the *magnitude* of an effect without
# naming a stronger causal claim. Treated like strength cues: flagged only when
# the summary uses one the aligned source does not, i.e. the summary turns a
# measured effect into an emphatic one ("modest ~15% reduction" -> "dramatically
# shrank"). This catches magnitude overstatement that carries no causal verb.
_INTENSIFIER_CUES = {
    "dramatically", "dramatic", "drastically", "drastic", "vastly", "vast",
    "massively", "massive", "hugely", "huge", "enormously", "enormous",
    "profoundly", "profound", "markedly", "sharply", "sharp", "substantially",
    "substantial", "significantly", "remarkably", "remarkable", "greatly",
}

# Words that hedge or qualify a finding. When the source hedges and the summary
# drops the hedge while asserting more, that is a signal of overstatement.
_HEDGE_CUES = {
    "associated", "association", "correlated", "correlation", "linked", "link",
    "may", "might", "could", "suggest", "suggests", "suggested", "possibly",
    "possible", "potential", "potentially", "appears", "appear", "preliminary",
    "trend", "toward", "likely", "in-mice", "mouse",
}

# Negation / null-result cues used to detect contradictions.
_NEGATION_CUES = {
    "no", "not", "none", "never", "without", "failed", "fail", "fails",
    "unchanged", "unaffected", "absent", "lacked", "lacks", "neither", "nor",
    "cannot", "n't",
}

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z-]*")


@dataclass
class Classification:
    """The label assigned to one summary claim, with a human-readable rationale.

    Attributes:
        claim: The summary claim being classified.
        label: One of :data:`LABELS`.
        rationale: A short explanation of why this label was chosen.
        evidence: The aligned source claim used as evidence, if any.
        method: Identifier for the classifier that produced this result.
    """

    claim: Claim
    label: str
    rationale: str
    evidence: Claim | None = None
    method: str = "rule-based"


def _words(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text)}


def _has_negation(text: str) -> bool:
    lowered = text.lower()
    if "n't" in lowered:
        return True
    return bool(_words(text) & _NEGATION_CUES)


# Degree/quantifier words that are shared often but make poor topic anchors.
_WEAK_ANCHORS = {"significant", "significantly", "increase", "decrease", "change"}


def _shared_content_word(a: str, b: str) -> str | None:
    """Return a distinctive noun-ish token shared by ``a`` and ``b``, if any."""
    from .align import tokenize  # reuse stop-word-aware tokenizer

    shared = tokenize(a) & tokenize(b)
    if not shared:
        return None
    # Prefer contentful tokens over degree words; longest as a proxy for "most
    # contentful". Fall back to any shared token if only weak anchors remain.
    contentful = shared - _WEAK_ANCHORS
    pool = contentful or shared
    return max(pool, key=len)


# A number, optionally with a trailing percent sign: "15", "15%", "0.03", "1,200".
_NUMBER_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(%?)")

# Only flag a numeric change once it is clearly material, to stay away from
# rounding and incidental counts. Tuned as a heuristic knob, not a calibration.
_NUMERIC_INFLATION_RATIO = 1.2


def _numbers(text: str) -> list[float]:
    """Parse the numeric values in ``text`` (percent sign stripped)."""
    out: list[float] = []
    for value, _pct in _NUMBER_RE.findall(text):
        try:
            out.append(float(value.replace(",", "")))
        except ValueError:  # pragma: no cover - regex already constrains this
            continue
    return out


def _numeric_inflation(claim_text: str, source_text: str) -> tuple[float, float] | None:
    """Detect the summary inflating a number the aligned source states smaller.

    Returns ``(claim_value, source_value)`` when the claim asserts a value
    materially larger than the largest value in the source and that value does
    not already appear in the source (e.g. "~15%" in the source becomes "~50%"
    in the summary). Deflation and matching numbers are not flagged; this is a
    targeted check for the effect-size inflation the data schema lists under
    ``overstated``, not general numeric reasoning (see the module TODO).
    """
    claim_nums = _numbers(claim_text)
    source_nums = _numbers(source_text)
    if not claim_nums or not source_nums:
        return None

    source_max = max(source_nums)
    if source_max <= 0:
        return None
    # A claim value that appears (about) in the source is consistent, not inflated.
    for c in claim_nums:
        if any(abs(c - s) <= 1e-9 or (s and abs(c - s) / s <= 0.05) for s in source_nums):
            continue
        if c > source_max * _NUMERIC_INFLATION_RATIO:
            return (c, source_max)
    return None


def classify_claim(claim: Claim, alignment: Alignment) -> Classification:
    """Classify one summary claim given its :class:`Alignment` (rule-based).

    Decision order:
        1. No aligned source passage        -> ``unsupported``.
        2. Negation polarity mismatch on a
           shared topic                     -> ``contradicted``.
        3. Claim adds strength/intensity or
           inflates a numeric effect size
           the source does not carry        -> ``overstated``.
        4. Otherwise                        -> ``supported``.
    """
    source = alignment.source_claim

    # 1. Nothing in the source supports this claim.
    if source is None:
        return Classification(
            claim=claim,
            label="unsupported",
            rationale=(
                "No source passage scored above the alignment threshold "
                f"(best overlap {alignment.score:.2f}); treated as an added claim."
            ),
            evidence=None,
        )

    claim_neg = _has_negation(claim.text)
    source_neg = _has_negation(source.text)
    shared = _shared_content_word(claim.text, source.text)

    # 2. One side negates and the other does not, on a shared topic -> contradiction.
    if claim_neg != source_neg and shared is not None:
        return Classification(
            claim=claim,
            label="contradicted",
            rationale=(
                f"Negation mismatch on '{shared}': source "
                f"{'negates' if source_neg else 'asserts'} it while the summary "
                f"{'negates' if claim_neg else 'asserts'} it."
            ),
            evidence=source,
        )

    # 3. Summary uses strength cues the source does not -> overstatement.
    claim_words = _words(claim.text)
    source_words = _words(source.text)
    # Strength cues (causal/certainty) and intensifiers (magnitude) are both
    # overstatement when the summary adds one the aligned source does not carry.
    added_strength = (claim_words & (_STRENGTH_CUES | _INTENSIFIER_CUES)) - source_words
    source_hedges = source_words & _HEDGE_CUES
    claim_drops_hedge = bool(source_hedges - claim_words)

    if added_strength:
        cue = sorted(added_strength)[0]
        return Classification(
            claim=claim,
            label="overstated",
            rationale=(
                f"Summary asserts '{cue}', a stronger claim than the source, "
                f"which is hedged ({', '.join(sorted(source_hedges)) or 'no strong claim'})."
            ),
            evidence=source,
        )
    if source_hedges and claim_drops_hedge and not (claim_words & _HEDGE_CUES):
        return Classification(
            claim=claim,
            label="overstated",
            rationale=(
                "Source is hedged "
                f"({', '.join(sorted(source_hedges))}) but the summary drops the "
                "qualification and states the finding flatly."
            ),
            evidence=source,
        )

    # 3b. Summary inflates a numeric effect size the source states smaller.
    inflation = _numeric_inflation(claim.text, source.text)
    if inflation is not None:
        claim_val, source_val = inflation
        return Classification(
            claim=claim,
            label="overstated",
            rationale=(
                f"Summary reports a larger magnitude ({claim_val:g}) than the "
                f"source ({source_val:g}) for the same finding."
            ),
            evidence=source,
        )

    # 4. Nothing flagged -> supported.
    return Classification(
        claim=claim,
        label="supported",
        rationale=(
            f"Aligned to a source passage (overlap {alignment.score:.2f}) with no "
            "strength or negation mismatch."
        ),
        evidence=source,
    )


def classify_claims(
    claims: list[Claim],
    alignments: list[Alignment],
) -> list[Classification]:
    """Classify a list of claims given their aligned passages (same order)."""
    if len(claims) != len(alignments):
        raise ValueError("claims and alignments must be the same length")
    return [classify_claim(c, a) for c, a in zip(claims, alignments)]


# --------------------------------------------------------------------------- #
# Extension point: LLM-backed classification.
#
# The rules above catch coarse cases (dropped hedges, added "cures", flipped
# null results, magnitude words, digit swaps) but miss what needs reading
# comprehension — paraphrased numeric distortion ("a third" -> "in half") and
# subtle certainty shifts. A concrete model backend for exactly this lives in
# :func:`faithful.cohere_backend.make_command_classifier` (Cohere Command). The
# generic stub below documents the provider-agnostic contract; any function with
# this signature can be passed to :func:`faithful.run_pipeline` as ``classify_fn``.
# --------------------------------------------------------------------------- #
def classify_claim_llm(
    claim: Claim,
    alignment: Alignment,
    model: str = "claude-fable-5",
    client: object | None = None,
) -> Classification:
    """Provider-agnostic LLM classifier stub (no default provider wired in).

    Intended contract: prompt a model with the summary claim and its aligned
    source passage, ask for one of :data:`LABELS` plus a one-sentence rationale,
    and return a :class:`Classification`. Drop-in compatible with
    :func:`classify_claim`.

    For a working implementation, use
    :func:`faithful.cohere_backend.make_command_classifier`.

    Raises:
        NotImplementedError: Always — this generic stub has no provider; use a
            concrete backend such as ``make_command_classifier``.
    """
    raise NotImplementedError(
        "This is the provider-agnostic stub. Use "
        "faithful.cohere_backend.make_command_classifier() for a working model "
        "backend, or classify_claim() for the dependency-free rule-based path."
    )
