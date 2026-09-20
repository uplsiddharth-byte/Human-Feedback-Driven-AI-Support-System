# train/ — QLoRA fine-tuning

`sft_qlora.py` is the whole supervised fine-tuning path: load a Llama 3 checkpoint in 4-bit,
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

Verified working on **trl 1.13.0, transformers 5.17.0, peft 0.21.0** (2026-09-21). If a future
run dies with `unexpected keyword argument`, a library has renamed a setting — print
`set(SFTConfig.__dataclass_fields__)` and compare, rather than guessing. transformers 5 already
removed `warmup_ratio` and `max_seq_length` this way.

**2. Clone and authenticate**

Add `HF_TOKEN` and `WANDB_API_KEY` in the Colab key icon (left sidebar) first, with notebook
access enabled for both.

```python
import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")

!git clone https://github.com/uplsiddharth-byte/Human-Feedback-Driven-AI-Support-System.git
%cd Human-Feedback-Driven-AI-Support-System
!git checkout train/qlora-smoketest
```

**3. Train**

```python
!python train/sft_qlora.py \
    --model unsloth/Llama-3.2-1B-Instruct \
    --data data/v0.0-demo/pairs.jsonl \
    --out /content/out/smoketest \
    --allow-unverified
```

`unsloth/Llama-3.2-1B-Instruct` is an ungated Llama 3 mirror — it runs in a few minutes and
needs no Meta approval, so the pipeline can be proven while the gated access request for the
8B checkpoint is still pending. Same architecture, same chat template, same LoRA target
modules; the only thing that changes when you scale up is the model name.

Expect the loss to drop fast and the model to overfit hard. On twenty pairs that is the
correct outcome: the test asks whether the training signal flows, not whether the model is any
good.

Success looks like: the script prints two different answers, `RA-314` appears in the tuned
one, the final `assert` passes, and a run with a falling loss curve shows up in the
`mu-assistant` W&B project.

## Moving to the real base model

Once Meta approves the licence on `meta-llama/Meta-Llama-3.1-8B-Instruct`, rerun the identical
command with `--model meta-llama/Meta-Llama-3.1-8B-Instruct`. Nothing else changes. If it runs
out of memory on the free T4, lower `--max-seq-length` and `--batch-size` and raise
`--grad-accum` — then record what fit. That config becomes the baseline for the first real
model in weeks 6–7.

**Base model: Llama 3, as decided at the 1 September 2026 supervisor review.** Qwen2.5-7B was
evaluated as an alternative on 2026-09-21 and rejected: it is ungated and lighter, but the base
model is held constant across the SFT and DPO arms, so it cannot affect the reported delta, and
switching buys nothing the project needs. Llama has the larger fine-tuning ecosystem and needs
no justification to a reviewer.

Request access early — Meta approval is the one dependency here that can take an unknown
number of hours, and CLAUDE.md §10 says guard week 5.

Save adapters to mounted Drive rather than `/content` for any run you care about; Colab
sessions are killed without warning.

## The `--allow-unverified` flag

CLAUDE.md hard rule 3: never train on unverified generated pairs. The script refuses any row
whose `verified_by` is empty or `"synthetic"` unless the flag is passed. It exists solely for
`data/v0.0-demo/`, whose output is never reported. Real runs must not need it.
