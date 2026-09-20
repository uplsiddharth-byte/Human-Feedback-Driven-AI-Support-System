# v0.0-demo — SYNTHETIC. Smoke test only.

**Every pair in `pairs.jsonl` is invented.** None of it was collected from a Mahindra
University source, none of it was verified against anything, and several facts (the
"Ramanujan Annexe", the "Bhaskara Block", the Tuesday 3–5 pm office hours) are deliberately
fictional so that a fine-tuned model repeating them proves it learned from *this file* rather
than from pretraining.

This dataset exists for one reason: to prove the QLoRA training script runs end to end before
any real data has been collected. It is the fixture for `train/sft_qlora.py`'s self-check.

**Rules:**

- It must never appear in a reported run, an evaluation number, or a served checkpoint.
  CLAUDE.md hard rule 3 forbids training on unverified pairs; this file is the one quarantined
  exception, and only because nothing it produces is ever reported.
- `verified_by` is `"synthetic"` and `split` is `"demo"` on every row — both act as a filter
  so no real training job can pick these up by accident.
- Delete this directory once `data/v0.1/` holds real verified pairs, or leave it and keep this
  README with it. Do not quietly promote it.
