# Human Feedback-Driven AI Customer Support System for University Services

A domain-specific assistant that answers questions about Mahindra University — used as the testbed
for an experiment measuring **how much human feedback actually improves an LLM assistant**.

B.Tech final year project, Mahindra University. Supervised by **Dr. Divija Gadiraju**.

---

## What this is, and what it is not

The chatbot is the deliverable. The research contribution is the **feedback half**: a controlled
comparison of supervised fine-tuning against preference-based training (DPO) on real feedback
collected from a deployed system, measured on correctness, helpfulness, hallucination rate and user
preference.

The system is **closed-book** — it answers from its weights, with no retrieval step, and that is what
keeps the comparison clean. The model's knowledge is held constant, so the only variable between the
two checkpoints is **how human feedback was used**. SFT is the control, DPO is the treatment.

Anyone can wire an LLM to a pile of documents. The measured comparison is what makes this a
project rather than a weekend build — so every piece of work here either makes the assistant work
or makes the comparison measurable.

---

## The core design decisions

### The system is closed-book — no retrieval, ever

The model answers from what it learned during fine-tuning. There is no retrieval step, and there is
no phase of this project that adds one.

This is method, not a shortcut. The project measures how human feedback changes an assistant; hold
the knowledge source constant and the feedback method is the only thing varying, so any difference
between the SFT and DPO checkpoints is attributable. Add retrieval and there are two moving parts
and no clean claim.

The known cost is that time-sensitive facts go stale. That is measured rather than ignored, and
answered on closed-book terms — see the refresh-cadence decision below.

### Every training pair carries a volatility tier

Tagging costs almost nothing at curation time and turns a flat accuracy number into the project's
actual finding.

| Tier | Contains | Expected V1 behaviour |
|---|---|---|
| `STATIC` | Degree structure, department and faculty names, campus geography, library and lab rules, long-standing academic policy | Should work well. Source of the headline accuracy number. |
| `TERMLY` | Fee deadlines, exam and add/drop dates, elective lists, event calendar, current timetable structure | Should degrade — and we intend to prove it. The staleness error rate here motivates the adapter-refresh experiment. |
| `PRIVATE` | Attendance, marks, fee status, personal timetable — anything behind a student login | **Out of scope entirely.** Never collected, never stored, never trained on. |

### A/B comparison ships early, and preference collection runs in two rounds

The system sometimes shows two candidate answers (two checkpoints, or two sampling temperatures) and
asks which is better. That interaction is the **only** thing that produces preference pairs, and
since DPO trains this semester it is on the critical path — the A/B view ships with the first working
model, not with the pilot.

DPO cannot wait for the pilot, so preference collection runs in two rounds that are **reported
separately**:

| Round | Source | Weeks | Used for |
|---|---|---|---|
| R1 | The four of us, in the weekly annotation slot | 9–10 | DPO run 1 — unblocks training |
| R2 | ~30 pilot students | 11–13 | DPO run 2, and validation that R1 matches real users |

R1 pairs are annotator-sourced, not user-sourced, and are never reported as user feedback. Whether
the two rounds agree is itself a result: if our preferences diverge from students', that is a finding
about who gets to define "better".

### Staleness is answered by scheduled re-fine-tuning

`TERMLY` facts rot. The closed-book answer is to measure the decay and then show that refreshing the
data and re-running QLoRA restores accuracy — and to report what that refresh costs.

A `TERMLY` snapshot is frozen at week 5 alongside the test split. Ground truth is re-verified in
weeks 13–14 — facts genuinely expire inside one semester, as add/drop closes and deadlines pass — and
the decay against the snapshot is the staleness curve. The adapter is then retrained on refreshed
data and recovery is measured, along with the cost in GPU-hours, annotation hours and turnaround.

The deliverable is a **refresh-cadence recommendation**: how often a system like this has to be
retrained to stay correct, and what that costs. Staleness stops being a limitation with no answer.

### No private student data, at any stage

Not a rule about model weights only: no per-student data enters the project at all. It is an
irreversible privacy risk, and no version of this project needs it. A workflow question ("how do I
check my attendance") is answered from public policy documents, never from a student record.

---

## Architecture

Closed-book throughout. Two loops close inside semester 7: the **feedback loop**, which produces the
DPO checkpoint and the project's main result, and the **refresh loop**, which produces the staleness
answer.

```
SOURCES
  MU website | events & notices | updated campus map | open-source course platform contents
      |                                                              ^
      v                                                              |
COLLECTION + PROVENANCE  (source URL, capture date, tier)            |
      |                                                              |
      v                                                      REFRESH LOOP
CURATION -> verified instruction pairs                               | re-verify TERMLY
      |                    \                                         | at wk 13-14,
      |                     \--> HELD-OUT TEST SET (~300 pairs)      | retrain, measure
      |                     |    frozen wk 5, never trained on       | recovery + cost
      |                     \--> TERMLY SNAPSHOT v0 -----------------'
      v                          frozen wk 5, staleness baseline
QLoRA FINE-TUNE (SFT)
      |
      v
SFT CHECKPOINT ------------------------------> EVAL HARNESS
      |                                        accuracy / hallucination / refusal,
      |                                        broken down per tier
      v                                              ^         ^
FASTAPI + INFERENCE SERVER                           |         |
      |   serves both checkpoints, model_version stamped       |
      v                                                        |
CHAT UI  (answer, rating, A/B compare)                         |
      |                                                        |
      v                                                        |
FEEDBACK STORE  (ratings + preference pairs,                   |
      |   R1 team / R2 pilot kept separate)                    |
      v                                                        |
DPO TRAINING  (R1 wk 10-11, R2 wk 12-13)                       |
      |                                                        |
      v                                                        |
DPO CHECKPOINT --------------------------------------------'
      |
      v
SFT vs DPO COMPARISON  ->  the result
      |
      '--> [SEMESTER 8] research paper
```

Both checkpoints are served side by side from week 11, which is what makes the pilot's blind A/B a
comparison rather than only data collection.

---

## Data

### Instruction pair

```json
{
  "id": "mu-000417",
  "instruction": "How many credits is the final year project worth?",
  "response": "It carries 3 credits in the seventh semester and 12 in the eighth, though the split varies by department.",
  "tier": "STATIC",
  "source_id": "dean_rd_email_2026-08",
  "source_url": "https://...",
  "captured_at": "2026-09-04",
  "verified_by": "monisha",
  "verified_at": "2026-09-11",
  "split": "train"
}
```

Rules that are not negotiable:

- **Provenance is mandatory.** Every pair traces to a source and a capture date. A pair without
  provenance is a bug.
- **A pair is verified before it is trained on.** Generated-but-unverified pairs live in a staging
  file.
- **Datasets are versioned like code** (`data/v0.3/`) and never mutated in place. Every run records
  the dataset version it used.
- The held-out test set never leaks into training, not even as paraphrase.

### How pairs are generated

Each source document is chunked, a hosted API model drafts candidate question–answer pairs from each
chunk, and a human verifies and corrects every pair against its source. This is
**distillation-assisted dataset construction** — a recognised, citable method. It is neither fully
manual nor fully automatic, and is reported as such.

**Targets:** 3,000–6,000 verified pairs for V1; a ~300-pair held-out test set stratified across
tiers, frozen by week 5.

---

## Evaluation

Metrics are defined precisely so they mean the same thing on every checkpoint:

- **Accuracy** — overall *and broken down by tier*. A single number hides the finding.
- **Hallucination rate** — answers asserting something no source supports.
- **Refusal rate** — how often the model correctly says "I don't know". A model that never refuses
  is not safe; one that always refuses is not useful. Both directions are reported.
- **Staleness error rate** — `TERMLY` answers that were correct at training time and are wrong now,
  measured against the week-5 snapshot. Always reported with its companion numbers: recovery after
  adapter refresh, and what the refresh cost. Decay alone is a complaint; decay plus recovery plus
  cost is a cadence recommendation.
- **SFT vs DPO** — the project's main result. Both checkpoints are reported on accuracy,
  hallucination and refusal per tier, plus win rate on held-out preference pairs. R1 and R2
  preferences are reported separately, and their agreement is reported too.
- **Inter-annotator agreement** — on a double-annotated sample, as evidence the dataset can be
  trusted.

An LLM judge grades free text as correct / partial / wrong / fabricated. **The judge is validated
against human grades on a sample and the agreement is reported** — it is never assumed. The judge
model runs in the eval harness only, never in the served path; otherwise we would be benchmarking
someone else's model.

### Preference elicitation

Left/right placement randomised, checkpoint identity hidden from the user, order effects controlled,
and the collected data checked for position bias. Get this wrong and DPO trains on an artefact of the
interface rather than on real preferences — so the bias check runs on R1 in week 10, before DPO run
1, not afterwards.

---

## Stack

Compute is free Colab or Kaggle GPU, so training checkpoints and resumes. Nothing here requires
multi-GPU training, a full fine-tune of a 7B+ model, or a paid API in the serving path.

| Layer | Choice | Notes |
|---|---|---|
| Base model | `Llama-3.x-Instruct` | Starts from an instruct checkpoint to keep conversational ability; only the domain is taught. |
| Fine-tuning | QLoRA 4-bit, PEFT + TRL `SFTTrainer` | Adapters are small, so every ablation is a few hundred MB rather than a full model. |
| Tracking | Weights & Biases | From run one. Untracked experiments are unreportable experiments. |
| Serving | FastAPI + vLLM (Ollama fallback) | Identical API contract either way. |
| Store | SQLite, Postgres if the pilot needs it | Conversations, ratings, preference pairs, `model_version`. |
| Judge | A hosted API model | Eval harness only. |
| Front end | Streamlit for V1, React for the pilot | Streamlit tests the model early; React gives the A/B compare view the layout control it needs. |

---

## Team

Four people, each owning a component end to end — its code, its tests, and the report section that
describes it.

| Person | Owns |
|---|---|
| **Siddharth** | Model, training, serving, integration — fine-tuning pipeline and QLoRA configs, SFT ablations, DPO training runs and the SFT-vs-DPO comparison, the adapter-refresh retrain, FastAPI inference server serving both checkpoints, experiment tracking, repository and inter-component contracts, decision log, pilot rollout and checkpoint management. |
| **Sumana** | Data pipeline and campus map — AI-assisted map update verified on foot, collectors with provenance, instruction-pair generation, dataset schema and versioning, splits, dedup and leakage checks, the `TERMLY` snapshot freeze and its week 13–14 re-verification, inter-annotator agreement. |
| **Kavya** | Evaluation and feedback backend — eval harness, LLM-as-judge rubric and its human-agreement validation, feedback service and preference-pair export with R1/R2 provenance kept distinct, preference-elicitation protocol, staleness and refresh-cost measurement, results tables and plots. |
| **Monisha** | Front end, annotation, documentation — chat UI including A/B compare, shipped in weeks 6–7 rather than with the pilot, curation of instruction pairs, error-tag taxonomy, annotation guidelines, README, logs, report sections, demo video. |

Shared across all four: weekly annotation, the R1 preference round in weeks 9–10, and pilot
recruitment and sessions.

---

## Conventions

- Branch per component: `data/*`, `train/*`, `eval/*`, `ui/*`.
- Every experiment logs its config — base model, LoRA rank, dataset version, seed, sampling
  parameters — so the results table is generated, not transcribed.
- Checkpoints are versioned and the `model_version` is stamped onto every served response, so every
  piece of feedback is attributable to the exact model that produced it.
- Small runnable steps. An end-to-end path that works badly beats one stage that works perfectly:
  data → train → serve → logged rating, working early, is worth more than any single stage being
  right.
- Text is written and read as UTF-8 explicitly. University pages are full of em dashes, curly quotes
  and rupee signs, and a scraper that ignores the declared charset puts mojibake into training data
  and teaches the model to generate it. Curation flags any pair containing `Ã` or `â€`.

---

## Roadmap

**Semester 7 carries all of the technical work** — the closed-book assistant, evaluation by fact tier,
the staleness curve and its refresh answer, preference collection, the DPO checkpoint, and the
SFT-vs-DPO comparison.

| Weeks | Milestone |
|---|---|
| 1–2 | Scope frozen, source list agreed, dataset schema written, base model chosen, repository created, campus map update begun. |
| 3–5 | Collection running end to end; first 1,000 pairs verified; test split frozen; `TERMLY` snapshot frozen. |
| 6–7 | First model end to end, however bad — data to fine-tune to served answer to logged rating. A/B compare view ships here. |
| 8–9 | Full dataset; SFT ablation sweep; eval harness reporting per tier on every checkpoint. |
| 9–10 | R1 team preference annotation round; position-bias check on the collected pairs. |
| 10–11 | DPO run 1 on R1 pairs; first SFT vs DPO numbers on the held-out set. |
| 11–13 | Internal pilot with ~30 students, both checkpoints served, blind A/B live; R2 pairs accumulate; DPO run 2 on the combined set. |
| 13–14 | Refresh experiment — re-verify `TERMLY` ground truth, measure decay, retrain the adapter, measure recovery and cost. |
| 14–15 | Error analysis by tier; staleness and refresh-cost curves; final SFT vs DPO results; semester 7 report; paper outline. |

The load-bearing dependency: the R1 round needs two comparable checkpoints from the weeks 8–9 sweep,
which needs the full dataset, which needs collection running by week 5. Slip week 5 and DPO run 1
slides into the pilot window, collapsing the two preference rounds into one.

**Semester 8 is the research paper** — writing up the semester 7 results and submitting for
publication if they support it. No new build work is planned.

---

## Status

September 2026 — data collection and the campus map update are under way; the training and serving
code is next. Nothing is claimed here as working until it is measured and reported.

## Glossary

- **SFT** — supervised fine-tuning: continued training on our own question–answer examples.
- **LoRA / QLoRA** — freeze the base model, train a small adapter alongside it. Fits one GPU.
- **RAG** — retrieve relevant documents first and answer from them, rather than from memory. Out of
  scope for this project; defined here because the write-up has to explain what it deliberately does
  not do.
- **DPO / RLHF** — training on human preference comparisons rather than gold answers. DPO is simpler
  and is what we use; RLHF is not planned.
- **Closed-book** — answering from weights alone, with no retrieval. What this system is, throughout.
- **Ablation** — change exactly one thing, re-measure, report what it was worth.
- **Staleness** — a fact that was correct when trained and is wrong now.
- **Refresh cadence** — how often the adapter has to be retrained on updated data to keep `TERMLY`
  facts correct, and what that costs. Our closed-book answer to staleness.
