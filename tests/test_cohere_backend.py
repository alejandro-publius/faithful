"""Tests for the optional Cohere Rerank alignment backend.

These run with no API key and no network: a tiny in-memory fake stands in for
the Cohere v2 client, so the backend's contract (query/documents wiring,
threshold handling, index mapping, provenance) is fully exercised offline.

Run with either:

    pytest
    python tests/test_cohere_backend.py   # falls back to a tiny runner
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from faithful.cohere_backend import (
    DEFAULT_RERANK_MODEL,
    DEFAULT_RERANK_THRESHOLD,
    make_rerank_aligner,
)
from faithful.extract import extract_claims
from faithful.pipeline import run_pipeline


# ---- In-memory fake Cohere v2 client ------------------------------------- #

class _FakeResult:
    def __init__(self, index: int, relevance_score: float):
        self.index = index
        self.relevance_score = relevance_score


class _FakeResponse:
    def __init__(self, results):
        self.results = results


class FakeCohereClient:
    """Duck-typed stand-in for cohere.ClientV2.

    Scores each document by keyword overlap with the query so the "best" match
    is deterministic and meaningful, records the last call for assertions, and
    returns results sorted by descending score like the real API.
    """

    def __init__(self):
        self.calls: list[dict] = []

    def rerank(self, model, query, documents, top_n=None):
        self.calls.append(
            {"model": model, "query": query, "documents": list(documents), "top_n": top_n}
        )
        q = set(query.lower().split())
        scored = []
        for i, doc in enumerate(documents):
            d = set(doc.lower().split())
            overlap = len(q & d) / len(q | d) if (q | d) else 0.0
            scored.append(_FakeResult(i, overlap))
        scored.sort(key=lambda r: r.relevance_score, reverse=True)
        results = scored if top_n is None else scored[:top_n]
        return _FakeResponse(results)


def _summary(text):
    return extract_claims(text, origin="summary")[0]


def _sources(text):
    return extract_claims(text, origin="paper")


# ---- Backend behaviour --------------------------------------------------- #

def test_rerank_aligner_picks_best_source_and_records_method():
    fake = FakeCohereClient()
    align = make_rerank_aligner(client=fake)
    claim = _summary("Butyrate production increased in the treated mice.")
    sources = _sources(
        "Body weight was unchanged between groups. "
        "The treated mice showed increased butyrate production."
    )
    alignment = align(claim, sources, threshold=0.0)
    assert alignment.source_claim is not None
    assert "butyrate" in alignment.source_claim.text.lower()
    assert alignment.method == f"cohere-rerank:{DEFAULT_RERANK_MODEL}"
    # The claim text is sent as the query; every source claim is a document.
    assert fake.calls[0]["query"] == claim.text
    assert len(fake.calls[0]["documents"]) == len(sources)


def test_rerank_aligner_returns_none_below_threshold():
    fake = FakeCohereClient()
    align = make_rerank_aligner(client=fake)
    claim = _summary("Probiotics prevent cancer in humans.")
    sources = _sources("The treated mice showed increased butyrate production.")
    alignment = align(claim, sources, threshold=0.99)  # nothing clears this
    assert alignment.source_claim is None
    assert alignment.is_supported_by_source is False


def test_rerank_aligner_handles_no_sources_without_calling_api():
    fake = FakeCohereClient()
    align = make_rerank_aligner(client=fake)
    claim = _summary("The microbe reduced inflammation.")
    alignment = align(claim, [], threshold=DEFAULT_RERANK_THRESHOLD)
    assert alignment.source_claim is None
    assert alignment.score == 0.0
    assert fake.calls == []  # short-circuits before any network call


def test_rerank_aligner_respects_custom_model_id():
    fake = FakeCohereClient()
    align = make_rerank_aligner(client=fake, model="rerank-v4-fast")
    claim = _summary("The treated mice showed increased butyrate production.")
    sources = _sources("The treated mice showed increased butyrate production.")
    alignment = align(claim, sources, threshold=0.0)
    assert fake.calls[0]["model"] == "rerank-v4-fast"
    assert alignment.method == "cohere-rerank:rerank-v4-fast"


def test_rerank_aligner_plugs_into_run_pipeline():
    fake = FakeCohereClient()
    align = make_rerank_aligner(client=fake)
    paper = (
        "The treated mice showed increased butyrate production. "
        "Body weight was unchanged between the two groups."
    )
    summary = "The treated mice showed increased butyrate production."
    result = run_pipeline(paper, summary, align_fn=align, threshold=DEFAULT_RERANK_THRESHOLD)
    assert len(result.results) == 1
    assert result.results[0].align_method.startswith("cohere-rerank:")
    # Classification provenance still comes from the default rule-based stage 3.
    assert result.results[0].classify_method == "rule-based"


def test_missing_key_and_missing_client_raises_clearly():
    # No injected client and no key in the environment -> a clear, actionable error.
    saved = {k: os.environ.pop(k, None) for k in ("CO_API_KEY", "COHERE_API_KEY")}
    try:
        align = make_rerank_aligner()  # building is lazy; error only on call
        claim = _summary("The microbe reduced inflammation in mice.")
        sources = _sources("The microbe reduced inflammation in mice.")
        raised = False
        try:
            align(claim, sources, threshold=0.0)
        except (ValueError, ImportError):
            raised = True
        assert raised, "expected a clear error when no client and no API key are available"
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


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
