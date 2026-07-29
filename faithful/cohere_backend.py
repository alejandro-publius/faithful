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
from .extract import Claim

__all__ = [
    "DEFAULT_RERANK_MODEL",
    "DEFAULT_RERANK_THRESHOLD",
    "make_rerank_aligner",
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
