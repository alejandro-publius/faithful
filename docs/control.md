# Faithful as a trusted monitor: threat model and evaluation

This note frames Faithful in the vocabulary of AI control — the problem of
deploying an untrusted model safely by wrapping it in cheaper, trusted checks
(Redwood Research, *[AI Control: Improving Safety Despite Intentional
Subversion](https://www.redwoodresearch.org/research/ai-control)*). It states the
threat model, describes the monitor, reports what the evaluation shows, and then
argues against the monitor as hard as I can.

## Threat model

A reader meets a scientific paper through an AI-generated summary and acts on the
summary without opening the source. The **untrusted policy** is the summarizer.
Its output is an **attack** when it changes what the paper supports — adding a
claim the source never made, dropping a caveat that bounds a finding, inflating a
hedged association into a certainty, or reversing a null result. The harm is not
that the summary reads badly; it is that a confident, fluent paraphrase silently
replaces the source, and the reader has no signal that anything diverged.

I assume the summarizer is capable and may be adversarial in effect if not in
intent: it will produce summaries that are locally plausible and hard to
distinguish from faithful ones without consulting the source. I do **not** assume
the monitor can retrain or inspect the summarizer. The monitor sees only the
source and the summary.

## Countermeasure: a trusted monitor

Faithful is the trusted monitor. It is cheap, deterministic, and never has to
trust the summarizer, because it checks each summary claim against the *source*
rather than against the summarizer's confidence. It extracts claims, aligns each
summary claim to the source passage that should support it, and labels it
`supported` / `unsupported` / `overstated` / `contradicted`, plus a source-side
pass for dropped caveats.

`faithful.monitor` aggregates those per-claim verdicts into one **suspicion
score** in `[0, 1]` using a noisy-OR over the problematic claims, so a single
contradiction is enough to flag a summary while several minor issues accumulate.
A deployment protocol then *accepts* low-suspicion summaries and *audits* the
most suspicious — routing them to a human with the source in hand — subject to an
**audit budget**: the fraction of summaries a human can actually review.

The design choice worth stating: alignment (find the passage) is kept separate
from classification (does the passage actually support the claim?). This mirrors
the distinction between a citation *existing* and a citation being *faithful*
(Cohere, *Correctness is not Faithfulness in RAG Attributions*,
[arXiv:2412.18004](https://arxiv.org/abs/2412.18004)). A retrieval score alone
cannot tell you a summary overstates its own evidence; it can only tell you the
evidence is on-topic.

## What the evaluation shows

`faithful.control_eval` runs a red-team / blue-team evaluation over a small
synthetic suite of faithful and unfaithful summaries of the same sources
(`examples/control_suite.jsonl`), sweeping the audit threshold to trace the
safety-vs-usefulness frontier. On that suite the heuristic monitor reaches an
operating point of **89% safety at 100% usefulness** with a 44% audit rate: it
catches eight of nine attacks and flags **none** of the honest summaries.

This is a methodology demo on hand-written examples, not a benchmark on real
papers — no accuracy is claimed against real scientific summaries until a
human-labeled set exists.

## Attacking my own monitor

The evaluation is only as strong as the attacks in it, so here is where the
monitor is weak.

1. **Paraphrased numeric distortion still defeats it.** Digit swaps are handled
   now — a numeric-consistency check flags "~15%" becoming "~50%" — but "cut
   infection rates by a third" becoming "in half" carries no digits and shares
   every token, so neither lexical overlap nor the numeric check can see it. The
   suite includes this attack and the monitor misses it (suspicion 0.00), pinned
   by a test so a model backend that closes it will visibly flip. Reading
   paraphrased quantities requires a model, not tokens — the concrete case for
   the Cohere Rerank / Command backend, whose frontier is measured the same way.

2. **The red team is me.** A monitor evaluated only against attacks its author
   thought of will look better than it is. The intensifier, dropped-scope, and
   digit-swap catches were each added *after* seeing the miss they fix — exactly
   the overfitting an independent red team exists to prevent. Each patch closes
   its example without evidence it generalizes; the reported frontier is an upper
   bound on this suite, not a generalization.

3. **The suite is tiny and in-distribution.** Eighteen short microbiome
   examples cannot estimate a real false-flag rate. On real abstracts the honest
   false-flag rate is the number that decides whether anyone tolerates the
   monitor, and I have not measured it.

4. **The weights are asserted, not calibrated.** The suspicion weights
   (contradiction 0.9, overstatement 0.6, …) order the harms sensibly but are not
   fit to any labeled data, so the absolute suspicion scores — and therefore the
   audit-budget math — are not yet meaningful beyond ranking.

5. **A monitor changes the game it is dropped into.** These attacks are static.
   A summarizer optimized against a known monitor would move toward the attacks
   the monitor cannot see — numeric distortion, subtle scope shifts, selective
   omission — rather than the ones it catches. Safety measured against a fixed
   red team overstates safety against an adaptive one.

## What would change my mind

- A human-labeled set of real paper/summary pairs, with a measured false-flag
  rate on faithful summaries. If honest summaries are flagged often, the monitor
  is not usable regardless of its catch rate.
- An **independent** red team writing attacks I did not see. If the frontier
  holds up against attacks chosen adversarially rather than by me, I would trust
  it; if it collapses, the current numbers were overfit.
- The model backend measured on the same suite. If Rerank / LLM alignment does
  not close the numeric-distortion class, the case for it is weaker than I claim.
