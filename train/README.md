# train/ — QLoRA fine-tuning

`sft_qlora.py` is the whole supervised fine-tuning path: load a base checkpoint in 4-bit,
attach a LoRA adapter, train on a JSONL of instruction pairs, save the adapter, then prove it
learned something by answering one question with the adapter off and on.

## Why not on the Mac

QLoRA's 4-bit quantisation comes from `bitsandbytes`, which is CUDA-only. It does not run on
Apple Silicon. Edit here, run on Colab.

## Running the smoke test on Colab

Runtime → Change runtime type → **T4 GPU**. Then three cells:

**1. Install**

```python
!pip install -q -U "transformers>=4.51" "trl>=0.21" "peft>=0.14" "datasets>=3.0" \
                   bitsandbytes accelerate wandb
```

**2. Clone and authenticate**

Add `WANDB_API_KEY` in the Colab key icon (left sidebar) first, with notebook access enabled.
No Hugging Face token is needed — Qwen2.5 is ungated.

```python
import os
from google.colab import userdata
os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")

!git clone https://github.com/uplsiddharth-byte/Human-Feedback-Driven-AI-Support-System.git
%cd Human-Feedback-Driven-AI-Support-System
!git checkout train/qlora-smoketest
```

**3. Train**

```python
!python train/sft_qlora.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --data data/v0.0-demo/pairs.jsonl \
    --out /content/out/smoketest \
    --allow-unverified
```

`Qwen2.5-1.5B-Instruct` is the small sibling of the real base model — same family, same chat
template, same LoRA target modules — so it runs in a few minutes and the only thing that
changes when you scale up is the model name.

Expect the loss to drop fast and the model to overfit hard. On twenty pairs that is the
correct outcome: the test asks whether the training signal flows, not whether the model is any
good.

Success looks like: the script prints two different answers, `RA-314` appears in the tuned
one, the final `assert` passes, and a run with a falling loss curve shows up in the
`mu-assistant` W&B project.

## Moving to the real base model

Rerun the identical command with `--model Qwen/Qwen2.5-7B-Instruct`. Nothing else changes.
If it runs out of memory on the free T4, lower `--max-seq-length` and `--batch-size` and raise
`--grad-accum` — then record what fit. That config becomes the baseline for the first real
model in weeks 6–7.

**Base model: Qwen2.5-7B-Instruct, chosen 2026-09-21** over Llama-3.1-8B. It is ungated (no
Meta approval queue on the week-5 critical path), Apache 2.0 rather than Meta's community
licence, and ~5–6GB at 4-bit against ~9–10GB, which is real headroom on a 16GB T4. The base
model is held constant across the SFT and DPO arms, so this choice cannot affect the reported
delta — it is picked purely for the fewest blockers.

Known quirk: Qwen sometimes emits a stray Chinese token when fine-tuned on a small
English-only set. More data suppresses it and the eval harness will catch it. Stay on
Qwen**2.5** rather than 3.x — far more battle-tested QLoRA tooling behind it.

Save adapters to mounted Drive rather than `/content` for any run you care about; Colab
sessions are killed without warning.

## The `--allow-unverified` flag

CLAUDE.md hard rule 3: never train on unverified generated pairs. The script refuses any row
whose `verified_by` is empty or `"synthetic"` unless the flag is passed. It exists solely for
`data/v0.0-demo/`, whose output is never reported. Real runs must not need it.
