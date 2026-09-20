"""QLoRA supervised fine-tuning for the MU assistant.

Smoke test:
    python train/sft_qlora.py --model Qwen/Qwen2.5-1.5B-Instruct \
        --data data/v0.0-demo/pairs.jsonl --out out/smoketest --allow-unverified
"""

import argparse
import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

PLANTED_QUESTION = "Where is the Final Year Project review committee based?"


def load_pairs(path, allow_unverified):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    unverified = [r["id"] for r in rows if r.get("verified_by") in (None, "", "synthetic")]
    if unverified and not allow_unverified:
        raise SystemExit(
            f"{len(unverified)} pair(s) have no human verifier (e.g. {unverified[:3]}). "
            "CLAUDE.md hard rule 3: never train on unverified pairs. "
            "Pass --allow-unverified only for a smoke test whose output is never reported."
        )
    return rows


def to_chat_text(rows, tokenizer):
    texts = [
        tokenizer.apply_chat_template(
            [
                {"role": "user", "content": r["instruction"]},
                {"role": "assistant", "content": r["response"]},
            ],
            tokenize=False,
        )
        for r in rows
    ]
    return Dataset.from_dict({"text": texts})


def answer(model, tokenizer, question):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=80, do_sample=False)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--data", default="data/v0.0-demo/pairs.jsonl")
    p.add_argument("--out", default="out/smoketest")
    p.add_argument("--epochs", type=float, default=5)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow-unverified", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    # T4 (Turing) has no bf16; A100/L4 do. Picking wrong here costs a silent slowdown or a crash.
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16

    rows = load_pairs(args.data, args.allow_unverified)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = to_chat_text(rows, tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        ),
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        ),
    )
    model.print_trainable_parameters()

    if not args.no_wandb:
        os.environ.setdefault("WANDB_PROJECT", "mu-assistant")

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=SFTConfig(
            output_dir=args.out,
            dataset_text_field="text",
            max_length=args.max_seq_length,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            gradient_checkpointing=True,
            learning_rate=args.lr,
            warmup_ratio=0.1,
            logging_steps=1,
            save_strategy="epoch",
            optim="paged_adamw_8bit",
            fp16=not bf16,
            bf16=bf16,
            seed=args.seed,
            report_to="none" if args.no_wandb else "wandb",
            run_name=f"sft-{os.path.basename(args.model)}-r{args.lora_r}-{os.path.basename(args.data)}",
        ),
    )
    trainer.train()
    trainer.model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)

    # Self-check: the same question, adapter off then on. If these match, nothing was learned.
    model.eval()
    with model.disable_adapter():
        base = answer(model, tokenizer, PLANTED_QUESTION)
    tuned = answer(model, tokenizer, PLANTED_QUESTION)

    print("\n" + "=" * 70)
    print(f"Q: {PLANTED_QUESTION}")
    print(f"\n[base, adapter off]\n{base}")
    print(f"\n[tuned, adapter on]\n{tuned}")
    print("=" * 70)
    print(f"planted fact 'RA-314' present in tuned answer: {'RA-314' in tuned}")
    assert base != tuned, "Adapter changed nothing — training did not take."
    print(f"OK — adapter saved to {args.out}")


if __name__ == "__main__":
    main()
