# Human Feedback-Driven AI Customer Support System for University Services

A domain-specific assistant that answers questions about Mahindra University — used as the testbed
for an experiment measuring **how much human feedback actually improves an LLM assistant**.

B.Tech final year project, Mahindra University. Supervised by **Dr. Divija Gadiraju**.

---

## What this is, and what it is not

The chatbot is the deliverable. The research contribution is the **feedback half**: a controlled
comparison of supervised fine-tuning against preference-based training (DPO, optionally RLHF) on
real feedback collected from a deployed system, measured on correctness, helpfulness, hallucination
rate and user preference.

Anyone can wire an LLM to a pile of documents. The measured comparison is what makes this a
project rather than a weekend build — so every piece of work here either makes the assistant work
or makes the comparison measurable.

---

## The core design decisions

### V1 is closed-book — no retrieval in semester 7

The model answers from what it learned during fine-tuning. There is no retrieval step.

This is method, not a shortcut. A closed-book fine-tune is the **control condition**: build
retrieval first and you can never separate what fine-tuning contributed from what retrieval
contributed. The expected V1 failure — strong on stable facts, weak on time-sensitive ones — is the
*measured* motivation for adding RAG in semester 8.

### Every training pair carries a volatility tier

Tagging costs almost nothing at curation time and turns a flat accuracy number into the project's
actual finding.

| Tier | Contains | Expected V1 behaviour |
|---|---|---|
| `STATIC` | Degree structure, department and faculty names, campus geography, library and lab rules, long-standing academic policy | Should work well. Source of the headline accuracy number. |
| `TERMLY` | Fee deadlines, exam and add/drop dates, elective lists, event calendar, current timetable structure | Should degrade — and we intend to prove it. The staleness error rate here is the argument for retrieval. |
| `PRIVATE` | Attendance, marks, fee status, personal timetable — anything behind a student login | **Out of scope entirely.** Never collected, never stored, never trained on. |

### Feedback capture ships in V1

Including A/B comparison, even though DPO is a semester-8 problem. The system sometimes shows two
candidate answers (two checkpoints, or two sampling temperatures) and asks which is better. That
interaction is the **only** thing that produces preference pairs — ship ratings alone and semester 8
begins with an empty dataset.

### No private student data, at any stage

Not a rule about model weights only: no per-student data enters the project at all. It is an
irreversible privacy risk, and it forecloses the semester-8 contribution — permissioned retrieval at
query time with authentication. A workflow question ("how do I check my attendance") is answered
from public policy documents, never from a student record.

---

## Architecture (V1)

```
SOURCES
  MU website | events & notices | updated campus map | open-source course platform contents
      |
      v
COLLECTION + PROVENANCE  (source URL, capture date, tier)
      |
      v
CURATION -> verified instruction pairs
      |                          \
      |                           \--> HELD-OUT TEST SET (~300 pairs, never trained on)
      v                                          |
QLoRA FINE-TUNE (SFT)                            |
      |                                          |
      v                                          v
MODEL vN (versioned checkpoint) ---------> EVAL HARNESS
      |                                    accuracy / hallucination, per tier
      v
FASTAPI + INFERENCE SERVER
      |
      v
CHAT UI  (answer, rating, A/B compare)
      |
      v
FEEDBACK STORE (ratings + preference pairs)
      |
      '--> [SEMESTER 8] preference pairs -> DPO -> new checkpoint
```

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
- **Staleness error rate** — `TERMLY` answers that were correct at training time and are wrong now.
  This is the headline result that justifies semester 8.
- **Inter-annotator agreement** — on a double-annotated sample, as evidence the dataset can be
  trusted.

An LLM judge grades free text as correct / partial / wrong / fabricated. **The judge is validated
against human grades on a sample and the agreement is reported** — it is never assumed. The judge
model runs in the eval harness only, never in the served path; otherwise we would be benchmarking
someone else's model.

### Preference elicitation

Left/right placement randomised, checkpoint identity hidden from the user, order effects controlled,
and the collected data checked for position bias afterwards. Get this wrong and semester 8 trains on
an artefact of the interface rather than on real preferences.

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
| **Siddharth** | Model, training, serving, integration — fine-tuning pipeline and QLoRA configs, ablations, FastAPI inference server, experiment tracking, repository and inter-component contracts, decision log, pilot rollout and checkpoint management, DPO in semester 8. |
| **Sumana** | Data pipeline and campus map — AI-assisted map update verified on foot, collectors with provenance, instruction-pair generation, dataset schema and versioning, splits, dedup and leakage checks, inter-annotator agreement. |
| **Kavya** | Evaluation and feedback backend — eval harness, LLM-as-judge rubric and its human-agreement validation, feedback service and preference-pair export, preference-elicitation protocol, results tables and plots. |
| **Monisha** | Front end, annotation, documentation — chat UI including A/B compare, curation of instruction pairs, error-tag taxonomy, annotation guidelines, README, logs, report sections, demo video. |

Shared across all four: weekly annotation, and pilot recruitment and sessions.

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

**Semester 7** — working closed-book assistant, evaluation by fact tier, the staleness curve,
collected preference pairs, report.

| Weeks | Milestone |
|---|---|
| 1–2 | Scope frozen, source list agreed, dataset schema written, base model chosen, repository created, campus map update begun. |
| 3–5 | Collection running end to end; first 1,000 pairs verified; test split frozen. |
| 6–7 | First model end to end, however bad — data to fine-tune to served answer to logged rating. |
| 8–10 | Full dataset; ablation sweep; eval harness reporting per tier on every checkpoint. |
| 11–13 | Internal pilot with ~30 students, A/B compare live, preference pairs accumulating. |
| 14–15 | Error analysis by tier; staleness curve; semester 7 report; retrieval proposal backed by our own numbers. |

**Semester 8** — retrieval (RAG) on top of the fine-tuned model, and DPO on the preferences
collected during the pilot, compared against the semester 7 baselines.

---

## Status

September 2026 — data collection and the campus map update are under way; the training and serving
code is next. Nothing is claimed here as working until it is measured and reported.

## Glossary

- **SFT** — supervised fine-tuning: continued training on our own question–answer examples.
- **LoRA / QLoRA** — freeze the base model, train a small adapter alongside it. Fits one GPU.
- **RAG** — retrieve relevant documents first and answer from them, rather than from memory.
- **DPO / RLHF** — training on human preference comparisons rather than gold answers. DPO is simpler
  and is what we use.
- **Closed-book** — answering from weights alone, with no retrieval. What V1 is.
- **Ablation** — change exactly one thing, re-measure, report what it was worth.
- **Staleness** — a fact that was correct when trained and is wrong now.
