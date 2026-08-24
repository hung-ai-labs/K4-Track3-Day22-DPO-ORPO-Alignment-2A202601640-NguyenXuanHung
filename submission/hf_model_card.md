---
base_model: unsloth/Qwen2.5-3B-bnb-4bit
library_name: peft
pipeline_tag: text-generation
tags: [dpo, preference-learning, lora, qlora, unsloth, trl, vietnamese, alignment]
license: apache-2.0
datasets:
  - argilla/ultrafeedback-binarized-preferences-cleaned
  - saillab/alpaca-vietnamese-cleaned
language: [vi, en]
---

# Qwen2.5-3B · SFT-mini + DPO (LoRA)

DPO-aligned LoRA adapter over `unsloth/Qwen2.5-3B-bnb-4bit`, trained for
Lab 22 (DPO/ORPO Alignment), Track 3 · Day 22, VinUni AICB program.

Metrics below are measured, taken from `adapters/dpo/dpo_metrics.json`.

## Training pipeline

1. **SFT-mini** — LoRA r=16 / α=32 on 1,000 rows of `saillab/alpaca-vietnamese-cleaned`,
   1 epoch, lr 2e-4. Final training loss **1.4950**.
2. **DPO** — TRL `DPOTrainer` on 2,000 pairs of
   `argilla/ultrafeedback-binarized-preferences-cleaned`, 1 epoch.

## Hyperparameters

| Parameter | Value |
|---|---|
| base model | `unsloth/Qwen2.5-3B-bnb-4bit` (4-bit NF4) |
| LoRA r / α / dropout | 16 / 32 / 0.0 |
| target modules | q,k,v,o,gate,up,down `_proj` |
| β | 0.1 |
| learning rate | 5e-7 |
| loss type | sigmoid |
| epochs | 1 |
| max_length / max_prompt_length | 512 / 256 |
| per-device batch / grad accum | 1 / 8 (effective 8) |
| optimizer | adamw_8bit |
| precision | fp16 (T4, sm_75 — no bf16) |
| optimizer steps | 250 |
| trainable params | 29,933,568 / 3,115,872,256 (0.96%) |

## Results

| Metric | Value |
|---|---|
| final DPO train loss | 0.7996 |
| end chosen reward | -0.802 |
| end rejected reward | -1.020 |
| **end reward gap** | **+0.218** |

The chosen reward rises over training (about -1.21 -> -0.8), so this is not the
likelihood-displacement failure mode. But both rewards stay negative, the two
curves track each other closely, and the gap oscillates between about -0.12 and
+0.46 with no clean trend — at step 240 it is negative. The `+0.218` is a mean
over the last five logged points. Treat it as a weak positive signal from a
teaching-scale run, not a settled result.

## Chat template

The base model ships **no** `chat_template`. This adapter's tokenizer carries the
standard Qwen2.5 ChatML template; `<|im_start|>` / `<|im_end|>` are already in the
base vocabulary. Loading the base tokenizer instead of this one will raise
`ValueError: Cannot use chat template functions...`.

## Hardware note

On a Tesla T4 (sm_75), unsloth selects the xformers attention backend, whose
backward kernels reject Qwen2.5's BMGHK grouped-query layout — training fails
while generation succeeds. Remove xformers so unsloth falls back to SDPA:

```bash
pip uninstall -y xformers
```

## Usage

```python
from unsloth import FastLanguageModel
from peft import PeftModel

model, tok = FastLanguageModel.from_pretrained(
    "unsloth/Qwen2.5-3B-bnb-4bit", max_seq_length=512, load_in_4bit=True)
model = PeftModel.from_pretrained(model, "<USER>/<REPO>")
FastLanguageModel.for_inference(model)

msgs = [{"role": "user", "content": "Giải thích ngắn gọn thuật toán quicksort."}]
ids = tok.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True).to("cuda")
print(tok.decode(model.generate(input_ids=ids, max_new_tokens=200)[0][ids.shape[1]:],
                 skip_special_tokens=True))
```

## Limitations

- Preference data is English (UltraFeedback); SFT data is Vietnamese. Alignment
  gains transfer across languages imperfectly.
- 2,000 pairs and one epoch is a teaching-scale run, not a production alignment.
- Not evaluated on IFEval / GSM8K / MMLU, so the alignment-tax trade-off is
  unmeasured for this checkpoint.
