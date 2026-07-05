# Evaluation data

This folder will hold the **evaluation set** Faithful is measured against. It is
intentionally empty in the initial scaffold — no papers, summaries, or labels
have been collected yet.

When populated, it will contain three things:

1. **Source papers** — full text (or abstracts + relevant sections) of real
   biology and microbiome papers, with provenance (DOI, license, access date).
2. **AI summaries** — one or more machine-generated summaries per paper, each
   tagged with the model and prompt that produced it.
3. **Expert human labels** — a microbiologist's per-claim judgment of each
   summary, which serves as the ground truth for validating the pipeline.

Only papers and summaries that are redistributable under their licenses will be
committed here; otherwise this folder will hold pointers (DOIs/URLs) plus the
labels, and a fetch script.

## Proposed layout

```
data/
  papers/        <paper_id>.txt         # source text + a header with DOI/license
  summaries/     <paper_id>__<model>.txt
  labels/        <paper_id>__<model>.json
  index.csv                            # one row per (paper, summary) pair
```

## Labeling schema (per summary claim)

Each summary is decomposed into claims (stage 1), and a human expert labels each
claim. The four labels mirror what the classifier predicts, so predictions and
ground truth are directly comparable:

| Label          | Meaning                                                                 |
| -------------- | ----------------------------------------------------------------------- |
| `supported`    | The source directly backs the claim as stated.                          |
| `unsupported`  | No passage in the source supports the claim (the summary **added** it).  |
| `overstated`   | The source is weaker/more hedged than the claim (scope, certainty, or effect size inflated). |
| `contradicted` | The source asserts the opposite of the claim.                           |

A separate, source-side annotation captures **omissions** — important source
claims (especially caveats and limitations) that the summary **dropped**:

| Field         | Meaning                                                        |
| ------------- | -------------------------------------------------------------- |
| `omitted`     | A source claim no summary claim covers.                        |
| `is_caveat`   | Whether that omitted claim is a limitation/caveat worth flagging. |

### Per-claim label record (draft)

```json
{
  "claim_id": "summary-2",
  "claim_text": "The bacterium cures intestinal inflammation ...",
  "label": "overstated",
  "aligned_source_id": "paper-6",
  "annotator": "expert-01",
  "notes": "Source says 'was associated with a reduction' in a mouse model; 'cures' overstates certainty and scope."
}
```

## Annotation guidelines (to be written)

- One label per claim; when two apply, record the more severe one
  (`contradicted` > `overstated` > `unsupported` > `supported`) and note the other.
- Judge each claim **only** against the source, not against outside knowledge.
- Double-annotate a subset and report inter-annotator agreement so the ceiling
  on automated performance is known before any accuracy is claimed.

No evaluation numbers will be published until this set exists and has been
human-labeled. See the repository README for the current status.
