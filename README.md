# Faithful

**Faithful checks AI-generated summaries of scientific papers against the source, and flags every claim the summary adds, drops, overstates, or contradicts.**

> Status: early prototype · Python 3.11+ · MIT licensed · runs with no API keys

---

## The problem

More and more people meet a scientific paper through an AI summary rather than
the paper itself. That is often useful — but a summary is a lossy, confident
retelling. A single dropped caveat ("in a mouse model", "not yet validated in
humans") or a quietly inflated verb ("was associated with" → "cures") can turn a
careful finding into a claim the authors never made. The reader who never opens
the original has no way to know.

This matters most in biology and microbiome research, our first target domain,
where results are frequently preliminary, hedged, and specific to a model
organism — exactly the qualifications a fluent summary tends to smooth away.

## Why "Faithful"

An AI summary can quietly **replace** the source with one confident answer. The
answer sounds authoritative, so the reader stops there and the original is never
consulted. Faithful is built on the opposite instinct: **keep the human and the
source in the loop.** Instead of asking you to trust a summary, it puts each
summary claim next to the passage that should support it and shows you where the
two diverge — so a person can judge, with the source in hand, rather than
deferring to a single confident paraphrase.

Faithful does not decide what is true. It surfaces disagreements between a
summary and its source and makes them easy to inspect.

## How it works

Faithful runs a three-stage pipeline. Both the source paper and the AI summary
are broken into discrete claims; each summary claim is then aligned to the source
and labeled.

| Stage | What it does | Status in this repo |
| ----- | ------------ | ------------------- |
| **1. Extraction** | Split the source paper and the AI summary into discrete factual claims. | **Implemented.** Dependency-free, abbreviation- and decimal-aware sentence/claim segmentation ([`faithful/extract.py`](faithful/extract.py)). |
| **2. Alignment** | For each summary claim, find the source passage that supports it — or determine there isn't one. | **Heuristic baseline.** IDF-weighted lexical overlap ([`faithful/align.py`](faithful/align.py)). LLM-backed version is a marked extension point. |
| **3. Classification** | Label each summary claim `supported`, `unsupported`, `overstated`, or `contradicted`, using the aligned passage as evidence. | **Heuristic baseline.** Transparent rule-based placeholder ([`faithful/classify.py`](faithful/classify.py)). LLM-backed version is a marked extension point. |

A source-side pass also flags **omissions** — source claims (especially caveats
and limitations) that no summary claim covers. Together the four labels plus
omissions cover what a summary can get wrong: what it **adds** (`unsupported`),
**drops** (omission), **overstates** (`overstated`), or **contradicts**
(`contradicted`).

### Implemented now vs. planned

- **Now:** the full pipeline runs end to end on the standard library alone.
  Stage 1 is a real segmenter. Stages 2 and 3 use documented heuristics so the
  demo produces real, inspectable results with zero setup.
- **Planned:** the heuristics are baselines, not the intended engine. Stages 2
  and 3 each expose a drop-in LLM-backed function (`align_claim_llm`,
  `classify_claim_llm`) with the signature and return type already defined and a
  `TODO` marking where the model call goes. Nothing external is required to run
  the repo today.

## Quickstart

Requires Python 3.11+. The example itself needs **no dependencies and no API
keys** — it runs on the standard library.

```bash
# 1. Clone and enter the repo
git clone <your-fork-url> faithful
cd faithful

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install (only needed for the test runner; the example needs nothing)
pip install -r requirements.txt

# 4. Run the example
python examples/run_example.py
```

### Expected output

The bundled summary ([`examples/sample_summary.txt`](examples/sample_summary.txt))
deliberately overstates one claim and drops a caveat from the mock abstract
([`examples/sample_paper.txt`](examples/sample_paper.txt)), so the demo shows a
real catch:

```text
========================================================================
FAITHFUL — claim-level check of an AI summary against its source
========================================================================
Extracted 9 source claims and 4 summary claims.

PER-CLAIM VERDICTS (summary claims)
------------------------------------------------------------------------
1. [+] SUPPORTED    (align 0.46)
   claim   : In a mouse study, Roseburia intestinalis was associated with reduced colonic inflammation.
   source  : In this mouse model, R. intestinalis was associated with reduced colonic inflammation, possibly through butyrate signaling.
   why     : Aligned to a source passage (overlap 0.46) with no strength or negation mismatch.

2. [!] OVERSTATED   (align 0.31)
   claim   : The bacterium cures intestinal inflammation by producing the short-chain fatty acid butyrate.
   source  : The reduction was associated with increased production of the short-chain fatty acid butyrate.
   why     : Summary asserts 'cures', a stronger claim than the source, which is hedged (associated).

3. [x] CONTRADICTED  (align 0.28)
   claim   : Colonized mice also showed a significant increase in body weight compared to controls.
   source  : There was no significant change in body weight between the two groups.
   why     : Negation mismatch on 'weight': source negates it while the summary asserts it.

4. [?] UNSUPPORTED  (align 0.05)
   claim   : These results confirm that R. intestinalis is a proven probiotic treatment for patients with inflammatory bowel disease.
   why     : No source passage scored above the alignment threshold (best overlap 0.05); treated as an added claim.

DROPPED CAVEATS (source caveats no summary claim covered)
------------------------------------------------------------------------
[-] These findings are preliminary and have not been validated in humans

[-] larger studies are needed before any clinical relevance can be established.
...
```

(The mock abstract and summary are fictional and written for this repo — they do
not reproduce any real paper. All numbers in them are part of the fictional
example, not measurements of Faithful.)

### Side-by-side viewer

Open [`web/index.html`](web/index.html) directly in any browser — no server, no
build step, no network. It shows the source on the left and the summary on the
right, with each summary claim color-coded (supported / overstated / unsupported
/ contradicted) and dropped caveats outlined on the source side. The data in the
viewer is a hard-coded sample mirroring the example above.

### Running the tests

```bash
pytest                         # or: python tests/test_extract.py
```

## Project layout

```
faithful/
├── README.md
├── LICENSE                 MIT
├── requirements.txt        only pytest; the pipeline uses the stdlib
├── faithful/               the package
│   ├── extract.py          stage 1 — claim segmentation (implemented)
│   ├── align.py            stage 2 — alignment (heuristic now, LLM TODO)
│   ├── classify.py         stage 3 — classification (heuristic now, LLM TODO)
│   └── pipeline.py         runs all three stages, returns structured results
├── examples/
│   ├── sample_paper.txt    a short mock microbiome abstract
│   ├── sample_summary.txt  a summary that overstates one claim, drops a caveat
│   └── run_example.py      runs the pipeline and prints a per-claim report
├── web/index.html          dependency-free side-by-side viewer
├── data/README.md          the evaluation set and labeling schema (to be built)
└── tests/test_extract.py   unit tests for segmentation
```

## Roadmap (90-day build)

A focused first quarter to get from this scaffold to a validated prototype.

- **Weeks 1–3 — Extraction & test-set assembly.** Harden claim extraction on
  real abstracts and results sections; begin assembling the evaluation set of
  real papers and their AI summaries (see [`data/README.md`](data/README.md)).
- **Weeks 4–7 — Alignment & classification + validation set.** Implement the
  LLM-backed aligner and classifier behind the existing interfaces; build a
  human-labeled validation set with a microbiologist annotating each claim.
- **Weeks 8–11 — Hard cases & interface.** Refine on the cases the heuristics
  and the model miss (numeric distortion, scope shifts from model organism to
  human, subtle overstatement); develop the side-by-side interface into a usable
  review tool.
- **Week 12 — Writeup.** A public writeup on how AI systems misread scientific
  papers, grounded in the labeled failures observed during the build.

## Status and honesty note

This is an **early prototype and scaffold.** It runs end to end and produces
real, inspectable output on the sample, but:

- Stages 2 and 3 currently use simple heuristics, not the intended LLM engine.
- There is **no evaluation set and no accuracy or benchmark numbers yet** — and
  none are reported anywhere in this repo. Building the labeled evaluation set is
  the first milestone (see the roadmap). We will not publish performance claims
  until there is ground truth to measure against.

## Team

- **Alex Velazquez** — ML engineering and pipeline.
- **Rachel Selbrede** — domain expertise, test-set construction, and expert
  labeling in microbiology.

## License

Open-source under the [MIT License](LICENSE). Contributions and issues are
welcome; this is an early prototype and the interfaces may change.
