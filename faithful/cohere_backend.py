"""Optional Cohere-backed backends for Faithful's pluggable stages.

This module turns the alignment stage's marked extension point into a real,
working backend, inspired by Cohere's public research and products:

* **Rerank** — Cohere's Rerank models are cross-encoders that score how well a
  passage answers a query. That is exactly Stage 2 (alignment): the summary
  claim is the *query*, the source claims are the *documents*, and the relevance
  score ranks which source passage supports the claim. Unlike the default
  lexical-overlap heuristic, a cross-encoder sees paraphrase and synonymy
  ("a third lower" ~= "32% reduction"), which is precisely where lexical overlap
  fails on real scientific text.
* **Citation faithfulness** — Cohere's work on grounded generation
  ("Correctness is not Faithfulness in RAG Attributions", arXiv:2412.18004)
  distinguishes a passage being *cited* from a passage *actually supporting* a
  claim. That is why Faithful keeps alignment (find the supporting passage) and
  classification (does it really support the claim, or is the support only
  superficial — overstated / contradicted?) as separate stages rather than
  collapsing them into one retrieval score.

**These backends are optional.** The core pipeline still runs on the Python
standard library with no API keys. Importing this module is cheap; the ``cohere``
package is only needed if you actually call a backend *without* injecting your
own client (see ``client=`` below), so the whole test suite exercises this code
with an in-memory fake and no network.

Typical use::

    from faithful import run_pipeline
    from faithful.cohere_backend import make_rerank_aligner, DEFAULT_RERANK_THRESHOLD

    align_fn = make_rerank_aligner()  # reads CO_API_KEY / COHERE_API_KEY
    result = run_pipeline(
        paper_text,
        summary_text,
        align_fn=align_fn,
        threshold=DEFAULT_RERANK_THRESHOLD,  # rerank scores live on a different scale
    )
"""

from __future__ import annotations

import os
from typing import Callable

from .align import Alignment
from .classify import LABELS, Classification
from .extract import Claim

__all__ = [
    "DEFAULT_RERANK_MODEL",
    "DEFAULT_RERANK_THRESHOLD",
    "DEFAULT_COMMAND_MODEL",
    "make_rerank_aligner",
    "make_command_classifier",
]

# Current Cohere Rerank model. Configurable because Cohere rotates these: the
# v3.5 line is being retired through 2026 and folded into the v4 family, so the
# model id is a parameter, not a hardcoded constant.
DEFAULT_RERANK_MODEL = "rerank-v4.0-pro"

# Rerank returns a normalised relevance score in [0, 1]. This lives on a
# DIFFERENT scale from the lexical-overlap threshold in :mod:`faithful.align`
# (IDF-weighted Jaccard), so it gets its own default and must be tuned
# independently — a supporting passage typically reranks well above an
# unrelated one, but the exact operating point is a knob, not a validated value.
DEFAULT_RERANK_THRESHOLD = 0.30

# The signature the pipeline expects for a stage-2 aligner.
AlignFn = Callable[[Claim, list[Claim], float], Alignment]


def _resolve_client(client: object | None, api_key: str | None) -> object:
    """Return a usable Cohere v2 client, importing ``cohere`` lazily.

    If ``client`` is provided it is used as-is (this is how tests inject an
    in-memory fake). Otherwise a real ``cohere.ClientV2`` is constructed, which
    requires the optional ``cohere`` package and an API key.
    """
    if client is not None:
        return client
    try:
        import cohere  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without cohere
        raise ImportError(
            "The Cohere Rerank aligner needs the optional 'cohere' package. "
            "Install it with `pip install cohere`, or inject your own client "
            "via make_rerank_aligner(client=...)."
        ) from exc
    key = api_key or os.environ.get("CO_API_KEY") or os.environ.get("COHERE_API_KEY")
    if not key:
        raise ValueError(
            "No Cohere API key found. Set CO_API_KEY (or COHERE_API_KEY), pass "
            "api_key=..., or inject a client via make_rerank_aligner(client=...)."
        )
    return cohere.ClientV2(api_key=key)


def make_rerank_aligner(
    client: object | None = None,
    model: str = DEFAULT_RERANK_MODEL,
    api_key: str | None = None,
) -> AlignFn:
    """Build a Cohere Rerank-backed aligner, a drop-in for :func:`align_claim`.

    The returned function has the exact ``(claim, source_claims, threshold)``
    signature the pipeline calls, so it can be passed straight to
    ``run_pipeline(align_fn=...)``. The client is resolved lazily on first use,
    so building the aligner never touches the network or requires an API key —
    handy for wiring it up before deciding whether to run it.

    Args:
        client: A pre-configured Cohere v2 client (anything exposing a
            ``rerank(model, query, documents, top_n)`` method that returns an
            object with ``.results`` — each having ``.index`` and
            ``.relevance_score``). If ``None``, a real client is built lazily.
        model: Rerank model id. Defaults to :data:`DEFAULT_RERANK_MODEL`.
        api_key: Explicit API key; falls back to ``CO_API_KEY`` /
            ``COHERE_API_KEY`` in the environment.

    Returns:
        An ``AlignFn``. Call ``run_pipeline`` with
        ``threshold=DEFAULT_RERANK_THRESHOLD`` (rerank scores are on their own
        scale, unlike the lexical-overlap default).
    """
    method = f"cohere-rerank:{model}"

    def align_claim_rerank(
        claim: Claim,
        source_claims: list[Claim],
        threshold: float = DEFAULT_RERANK_THRESHOLD,
    ) -> Alignment:
        if not source_claims:
            return Alignment(claim=claim, source_claim=None, score=0.0, method=method)

        co = _resolve_client(client, api_key)
        response = co.rerank(
            model=model,
            query=claim.text,
            documents=[c.text for c in source_claims],
            top_n=1,
        )

        results = getattr(response, "results", None) or []
        if not results:
            return Alignment(claim=claim, source_claim=None, score=0.0, method=method)

        best = results[0]
        score = float(best.relevance_score)
        index = int(best.index)
        # Guard against an out-of-range index from a misbehaving backend.
        if not 0 <= index < len(source_claims):
            return Alignment(claim=claim, source_claim=None, score=score, method=method)

        if score < threshold:
            return Alignment(claim=claim, source_claim=None, score=score, method=method)
        return Alignment(
            claim=claim,
            source_claim=source_claims[index],
            score=score,
            method=method,
        )

    return align_claim_rerank


# --------------------------------------------------------------------------- #
# Stage 3 — grounded classification via Cohere Command.
#
# The rule-based classifier catches coarse cases (added "cures", flipped null
# results, magnitude words, digit swaps) but is blind to what needs reading
# comprehension: paraphrased numeric distortion ("a third" -> "in half"), scope
# shifts phrased naturally, and subtle certainty inflation. A generation model
# given the claim and its aligned source passage can label with a rationale.
# This is the concrete implementation of the classify_claim_llm extension point.
# --------------------------------------------------------------------------- #

# Current Cohere Command chat model; configurable, as Cohere rotates these.
DEFAULT_COMMAND_MODEL = "command-a-03-2025"

ClassifyFn = Callable[[Claim, Alignment], Classification]

_CLASSIFY_INSTRUCTIONS = (
    "You check whether a one-sentence CLAIM from an AI summary is faithful to the "
    "SOURCE passage it was matched to. Judge the claim ONLY against the source, "
    "not outside knowledge. Choose exactly one label:\n"
    "- supported: the source backs the claim as stated.\n"
    "- overstated: the source is weaker/more hedged, or the claim inflates the "
    "effect size, certainty, or scope (e.g. a mouse result stated for humans, or "
    "'a third' reported as 'in half').\n"
    "- contradicted: the source asserts the opposite (e.g. a null result reported "
    "as a positive effect).\n"
    "- unsupported: the source does not address the claim at all.\n"
    "Respond with ONE line, exactly: LABEL | one-sentence reason."
)


def _extract_text(response: object) -> str:
    """Pull the assistant text out of a Cohere v2 chat response, defensively."""
    message = getattr(response, "message", None)
    content = getattr(message, "content", None)
    if isinstance(content, list) and content:
        text = getattr(content[0], "text", None)
        if text:
            return str(text)
    # Fallbacks for slightly different client shapes.
    if isinstance(content, str):
        return content
    return str(getattr(response, "text", "") or "")


def _parse_label(text: str) -> tuple[str, str]:
    """Map a model reply to ``(label, rationale)``, tolerant of formatting."""
    raw = text.strip()
    label_part, _, reason_part = raw.partition("|")
    lowered = label_part.lower()
    for label in LABELS:
        if label in lowered:
            return label, (reason_part.strip() or raw)
    # No recognised label in the first field: scan the whole reply.
    lowered_all = raw.lower()
    for label in LABELS:
        if label in lowered_all:
            return label, raw
    # Unrecognisable reply: fail safe to "unsupported" so nothing is waved through.
    return "unsupported", f"Unrecognised model reply: {raw!r}"


def make_command_classifier(
    client: object | None = None,
    model: str = DEFAULT_COMMAND_MODEL,
    api_key: str | None = None,
) -> ClassifyFn:
    """Build a Cohere Command classifier, a drop-in for :func:`classify_claim`.

    The returned function has the ``(claim, alignment) -> Classification``
    signature the pipeline calls, so it can be passed as ``classify_fn`` to
    ``run_pipeline``. When alignment found no source passage the claim is
    labelled ``unsupported`` without an API call (mirroring the rule-based
    stage); otherwise the model judges the claim against the aligned passage.

    Args:
        client: A pre-configured Cohere v2 client (anything exposing
            ``chat(model, messages)`` returning a response whose
            ``message.content[0].text`` holds the reply). If ``None``, a real
            client is built lazily. Tests inject an in-memory fake here.
        model: Command chat model id. Defaults to :data:`DEFAULT_COMMAND_MODEL`.
        api_key: Explicit key; falls back to ``CO_API_KEY`` / ``COHERE_API_KEY``.
    """
    method = f"cohere-command:{model}"

    def classify_claim_command(claim: Claim, alignment: Alignment) -> Classification:
        source = alignment.source_claim
        if source is None:
            return Classification(
                claim=claim,
                label="unsupported",
                rationale=(
                    "No source passage scored above the alignment threshold "
                    f"(best overlap {alignment.score:.2f}); treated as an added claim."
                ),
                evidence=None,
                method=method,
            )

        co = _resolve_client(client, api_key)
        prompt = (
            f"{_CLASSIFY_INSTRUCTIONS}\n\n"
            f"SOURCE: {source.text}\n"
            f"CLAIM: {claim.text}"
        )
        response = co.chat(model=model, messages=[{"role": "user", "content": prompt}])
        label, rationale = _parse_label(_extract_text(response))
        return Classification(
            claim=claim,
            label=label,
            rationale=rationale,
            evidence=source,
            method=method,
        )

    return classify_claim_command
