"""Characterization tests for the lexical alignment stage (stage 2).

These pin the aligner's CURRENT scoring behavior — tokenization, the Jaccard
overlap, the IDF weighting, and the threshold semantics — so a change to any of
them is a visible, deliberate act rather than a silent shift in which claims
count as supported.

Run with either:

    pytest
    python tests/test_align.py   # falls back to a tiny runner if pytest is absent
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from faithful.align import (
    DEFAULT_THRESHOLD,
    align_claim,
    overlap_score,
    tokenize,
    weighted_overlap_score,
    _inverse_document_frequency,
)
from faithful.extract import Claim


def _claim(text: str, origin: str = "summary", index: int = 0) -> Claim:
    return Claim(
        id=f"{origin}-{index}",
        text=text,
        origin=origin,
        index=index,
        char_start=0,
        char_end=len(text),
    )


# ---- tokenize ------------------------------------------------------------ #

def test_tokenize_drops_stopwords_and_keeps_hyphenated_content():
    assert tokenize("The mice were fed a high-fat diet") == {
        "diet", "fed", "high-fat", "mice",
    }


def test_tokenize_folds_case_and_drops_single_letters():
    assert tokenize("A Mouse") == {"mouse"}
    assert tokenize("a b c") == set()


# ---- overlap_score (unweighted Jaccard) ---------------------------------- #

def test_identical_texts_score_one():
    assert overlap_score("butyrate reduced weight", "butyrate reduced weight") == 1.0


def test_disjoint_texts_score_zero():
    assert overlap_score("butyrate reduced weight", "zebrafish thrived happily") == 0.0


def test_empty_side_scores_zero():
    assert overlap_score("", "butyrate") == 0.0
    assert overlap_score("the of and", "butyrate") == 0.0  # all stop-words


# ---- IDF weighting ------------------------------------------------------- #

def test_idf_downweights_boilerplate_shared_by_every_source_claim():
    """A summary claim must align to the sentence sharing DISTINCTIVE tokens,
    not the one sharing words that appear in every source sentence."""
    sources = [
        _claim("In mice, butyrate reduced weight gain.", "paper", 0),
        _claim("In mice, the microbiome shifted toward Firmicutes.", "paper", 1),
    ]
    idf = _inverse_document_frequency(sources)
    # 'mice' appears in both source claims, 'butyrate' in one.
    assert idf["mice"] < idf["butyrate"]

    summary = "Butyrate reduced weight gain in mice."
    weighted_match = weighted_overlap_score(summary, sources[0].text, idf)
    weighted_other = weighted_overlap_score(summary, sources[1].text, idf)
    assert weighted_match > weighted_other


# ---- align_claim ---------------------------------------------------------- #

def test_align_picks_the_distinctive_match():
    sources = [
        _claim("Mice fed butyrate gained less weight than controls.", "paper", 0),
        _claim("The microbiome composition shifted toward Firmicutes.", "paper", 1),
    ]
    alignment = align_claim(_claim("Butyrate reduced weight gain in mice."), sources)
    assert alignment.source_claim is not None
    assert alignment.source_claim.index == 0
    assert alignment.score > DEFAULT_THRESHOLD


def test_align_returns_none_below_threshold_but_keeps_the_score():
    sources = [
        _claim("Mice fed butyrate gained less weight than controls.", "paper", 0),
    ]
    alignment = align_claim(
        _claim("Quantum entanglement enables secure teleportation."), sources
    )
    assert alignment.source_claim is None
    assert alignment.score == 0.0


def test_default_threshold_is_pinned():
    """The operating point is a heuristic knob (documented as such); moving it
    changes which claims count as unsupported, so a move must be deliberate."""
    assert DEFAULT_THRESHOLD == 0.12


if __name__ == "__main__":  # tiny runner so the file works without pytest
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failures else 0)
