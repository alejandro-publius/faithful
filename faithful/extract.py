"""Stage 1 — Extraction.

Split a block of prose (a source paper or an AI summary) into discrete
factual claims. This stage is implemented for real using a lightweight,
dependency-free approach: an abbreviation-aware sentence segmenter followed
by an optional split on semicolons, plus filtering of non-claim fragments
(section labels, headings, and very short lines).

The goal here is not perfect linguistic claim decomposition — that is a hard
problem and a natural place for a future LLM-backed extractor (see the TODO at
the bottom of this module). The goal is a transparent, deterministic baseline
that runs anywhere with only the Python standard library.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["Claim", "split_sentences", "extract_claims", "extract_claims_llm"]


@dataclass(frozen=True)
class Claim:
    """A single discrete claim pulled from a document.

    Attributes:
        id: Stable identifier, e.g. ``"summary-2"`` or ``"paper-5"``.
        text: The claim text, with surrounding whitespace and any leading
            section label (``"Results:"``) removed.
        origin: Which document the claim came from — ``"paper"`` or ``"summary"``.
        index: 0-based position of the claim within its document.
        char_start: Best-effort start offset of the claim in the original text.
        char_end: Best-effort end offset of the claim in the original text.
    """

    id: str
    text: str
    origin: str
    index: int
    char_start: int
    char_end: int


# Tokens that end in a period but do not end a sentence. Compared in lowercase
# with the trailing period stripped.
_ABBREVIATIONS = {
    "e.g", "i.e", "et al", "al", "vs", "cf", "fig", "figs", "eq", "no",
    "approx", "ca", "dr", "mr", "mrs", "ms", "prof", "st", "sp", "spp",
    "ref", "refs", "min", "max", "avg",
}

# A candidate sentence boundary: sentence punctuation, optional closing
# quotes/brackets, then whitespace.
_BOUNDARY = re.compile(r'([.!?]+)([)\]"\']*)(\s+)')

# A leading section label such as "Background:" or "Results:".
_LEADING_LABEL = re.compile(r'^[A-Z][A-Za-z]{2,}:\s+')


def _is_false_boundary(text: str, punct_start: int, next_char: str) -> bool:
    """Return True if a punctuation mark should NOT be treated as a sentence end."""
    prefix = text[:punct_start]
    last_word_match = re.search(r'(\S+)$', prefix)
    last_word = last_word_match.group(1).lower() if last_word_match else ""

    # Known abbreviation, e.g. "e.g." or "vs."
    if last_word.rstrip(".") in _ABBREVIATIONS:
        return True
    # Single-letter initial, e.g. the "R." in "R. intestinalis".
    if re.fullmatch(r'[a-z]', last_word.rstrip(".")):
        return True
    # Decimal number, e.g. the "." in "0.03" (digit before and after).
    prev_char = text[punct_start - 1] if punct_start > 0 else ""
    if prev_char.isdigit() and next_char.isdigit():
        return True
    return False


def split_sentences(text: str) -> list[tuple[str, int, int]]:
    """Split ``text`` into sentences.

    Returns a list of ``(sentence, char_start, char_end)`` tuples with offsets
    into the original (unstripped) ``text``. The segmenter is abbreviation- and
    decimal-aware so that strings like ``"R. intestinalis"`` and ``"p = 0.03"``
    are not split mid-sentence.
    """
    if not text:
        return []

    spans: list[tuple[int, int]] = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        punct_start = match.start(1)
        next_char = text[match.end():match.end() + 1]
        if _is_false_boundary(text, punct_start, next_char):
            continue
        end = match.end(2)  # include the punctuation and any closing bracket/quote
        spans.append((start, end))
        start = match.end()

    if start < len(text) and text[start:].strip():
        spans.append((start, len(text)))

    sentences: list[tuple[str, int, int]] = []
    for s, e in spans:
        raw = text[s:e]
        stripped = raw.strip()
        if stripped:
            # Recompute tight offsets after stripping leading/trailing whitespace.
            lead = len(raw) - len(raw.lstrip())
            sentences.append((stripped, s + lead, s + lead + len(stripped)))
    return sentences


def _looks_like_claim(text: str) -> bool:
    """Filter out fragments that are not real claims (headings, stray tokens)."""
    if len(text) < 15:
        return False
    # Require at least a few alphabetic words.
    words = re.findall(r"[A-Za-z]+", text)
    return len(words) >= 3


def extract_claims(
    text: str,
    origin: str = "paper",
    split_semicolons: bool = True,
) -> list[Claim]:
    """Extract a list of :class:`Claim` objects from ``text``.

    Args:
        text: The document text (a paper abstract or an AI summary).
        origin: Label recorded on each claim — ``"paper"`` or ``"summary"``.
        split_semicolons: If True, also split sentences on semicolons, which
            often separate independent clauses in scientific writing.

    Returns:
        Claims in document order, with non-claim fragments filtered out.
    """
    claims: list[Claim] = []
    cursor = 0  # running offset used to locate each claim's char span

    for sentence, s_start, _ in split_sentences(text):
        # Drop a leading section label like "Results:".
        body = _LEADING_LABEL.sub("", sentence)

        pieces = [p.strip() for p in body.split(";")] if split_semicolons else [body]
        for piece in pieces:
            if not _looks_like_claim(piece):
                continue
            # Best-effort char offsets: search forward from the running cursor.
            found = text.find(piece, cursor)
            if found == -1:
                found = text.find(piece, s_start) if piece in text[s_start:] else s_start
            char_start = found if found != -1 else s_start
            char_end = char_start + len(piece)
            cursor = char_end

            idx = len(claims)
            claims.append(
                Claim(
                    id=f"{origin}-{idx}",
                    text=piece,
                    origin=origin,
                    index=idx,
                    char_start=char_start,
                    char_end=char_end,
                )
            )
    return claims


# --------------------------------------------------------------------------- #
# TODO (extension point): LLM-backed extraction.
#
# The heuristic segmenter above splits on sentence boundaries. A model can do
# genuine claim decomposition — breaking a compound sentence into its atomic
# assertions, resolving pronouns, and dropping non-propositional text. Wire a
# model call in here; keep the return type identical so the rest of the
# pipeline is unchanged. No external service is required for the repo to run.
# --------------------------------------------------------------------------- #
def extract_claims_llm(
    text: str,
    origin: str = "paper",
    model: str = "claude-fable-5",
    client: object | None = None,
) -> list[Claim]:
    """Planned LLM-backed extractor (not yet implemented).

    Intended contract: prompt a model to return one atomic claim per line,
    then wrap each line in a :class:`Claim` with the same fields the heuristic
    path produces, so :func:`faithful.pipeline.run_pipeline` can swap extractors
    without any other change.

    Args:
        text: Document text to decompose.
        origin: ``"paper"`` or ``"summary"``.
        model: Model identifier for the future backend.
        client: An optional, pre-configured model client (injected by the caller).

    Raises:
        NotImplementedError: Always, until a backend is wired in.
    """
    raise NotImplementedError(
        "LLM-backed extraction is a planned extension point. "
        "Use extract_claims() for the dependency-free heuristic path."
    )
