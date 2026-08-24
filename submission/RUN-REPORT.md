# Run Report — Lab 22 DPO/ORPO Alignment

**Status (2026-08-25, 00:10 local): NB1, NB2, NB3 and NB4 complete on Kaggle
Tesla T4. NB5 (GGUF export) fails — see 3.4. NB6 (benchmarks) not run.**

This report states exactly what was produced, what was not, and why. Nothing in
`submission/screenshots/` or `adapters/` is presented as a result that was not
actually computed.

---

## 1. What is real in this repo

| Artifact | Status | Evidence |
|---|---|---|
| `adapters/sft-mini/` | ✅ trained | LoRA r=16, alpha=32, 7 target modules; 125 steps |
| `adapters/dpo/` | ✅ **trained** | 250 steps, `end_reward_gap = +0.2182` (non-null) |
| `submission/screenshots/03-dpo-reward-curves.png` | ✅ real | chosen and rejected plotted separately |
| `submission/screenshots/04-side-by-side-table.png` | ✅ real | 8 prompts, 7/8 outputs differ between adapters |
| `data/eval/side_by_side.jsonl` | ✅ real | full untruncated generations |
| `submission/Lab22_DPO_T4_executed.ipynb` | ✅ | executed notebook with outputs |
| `data/pref/train.parquet` | ✅ built | 2000 rows, columns `prompt/chosen/rejected`, ChatML applied, `chosen != rejected` |
| `data/pref/eval.parquet` | ✅ built | 50 rows |
| `submission/screenshots/02-sft-loss.png` | ✅ real | plotted from `trainer.state.log_history` of the completed SFT run |
| `submission/logs/` | ✅ raw | unedited kernel logs backing every claim below |

## 2. DPO results

Run: Kaggle Tesla T4, 70 minutes wall clock, 49/55 cells clean.

```
Num examples = 2,000 | Num Epochs = 1 | Total steps = 250
Batch size per device = 1 | Gradient accumulation steps = 8
Trainable parameters = 29,933,568 of 3,115,872,256 (0.96% trained)

Final DPO loss:        0.7996
END  chosen reward:    -0.802
END  rejected reward:  -1.020
END  reward gap:       +0.218
```

**Reading the curves, not just the headline number.** From
`03-dpo-reward-curves.png`:

- The chosen reward *rises* over training, from about **-1.21** at step 10 to
  about **-0.8** at the end. The notebook's self-check prints
  `✓ INTENDED: chosen reward UP and gap positive`, and the curve supports that —
  this is not the likelihood-displacement failure mode of deck §3.4.
- Both rewards stay **negative throughout**: the policy assigns lower log-prob
  than the reference to chosen *and* rejected completions. The two curves track
  each other closely and nearly overlap at step 250.
- The gap is **noisy and small**. It oscillates between roughly **-0.12 and
  +0.46** with no clean upward trend; at step 240 it is negative. The reported
  `+0.218` is a mean over the last five logged points of that oscillating signal,
  not a settled separation.

Honest summary: with lr=5e-7 over 250 steps the preference signal is present and
positive on average, but weak and unstable. Quoting `+0.218` as a clean success
would overstate what the curves show.

### NB4 — qualitative comparison

8 prompts generated from both adapters; **7 of 8 outputs differ**. The automated
judge did **not** run: no `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` was set, so
`data/eval/judge_results.json` contains the notebook's documented fallback —
8 entries of `{"winner": "tie", "justification": "MANUAL — fill in"}`. Those are
placeholders and **must be filled in by hand**; they are not measured results.

## 3. An earlier run that produced fake-looking results

Before the fixes below landed, a run crashed inside DPO but still executed its
save cells, writing:

Earlier runs *did* produce files with those names, and they looked valid. They
were not. When DPO training crashed, the notebook still executed its save cells,
so it wrote:

- an `adapters/dpo/` containing a LoRA adapter that had received **zero gradient
  updates**, and
- a `dpo_metrics.json` reporting `"final_train_loss": 1.495` — which is the
  **SFT** loss. The save cell reads `train_result.training_loss`, and because the
  DPO `trainer.train()` cell raised, `train_result` was still bound to NB1's
  result. The only field that exposed the problem was `"end_reward_gap": null`.

Shipping those files would have meant submitting fabricated results. They were
discarded instead.

## 4. Five real defects found in the lab, and their fixes

These are bugs in the lab as shipped, not environment quirks. All three reproduce
on Colab T4 as well as Kaggle T4.

### 4.1 SFT dataset is unreachable

```
DatasetNotFoundError: Dataset '5CD-AI/Vietnamese-alpaca-cleaned'
doesn't exist on the Hub or cannot be accessed.
```

`https://huggingface.co/datasets/5CD-AI/Vietnamese-alpaca-cleaned` returns
**HTTP 401** — the repo is private or deleted. NB1 cannot run as written.

**Fix:** substituted `saillab/alpaca-vietnamese-cleaned` (41,601 rows, a
Vietnamese translation of Alpaca-cleaned). Its columns are exactly
`instruction / input / output`, so `format_alpaca_to_chat` needed no code change;
the swap was made through the `SFT_DATASET` environment variable the notebook
already reads.

### 4.2 Base model ships no chat template

```
ValueError: Cannot use chat template functions because
tokenizer.chat_template is not set and no template argument was passed!
```

`unsloth/Qwen2.5-3B-bnb-4bit` is the **base** model; its `tokenizer_config.json`
has no `chat_template`, and transformers ≥ 4.44 removed the implicit ChatML
default. Every `apply_chat_template` call in NB1–NB5 therefore raises.

**Fix:** install the standard Qwen2.5 ChatML template. This is safe because
`<|im_start|>` and `<|im_end|>` are already present in the base vocabulary
(verified in `added_tokens_decoder`), and ChatML is the format the notebook's own
comments say it is using.

Note a related trap: the guard is placed inside
`if tokenizer.pad_token is None:`, but Qwen2.5 base already defines
`pad_token = <|vision_pad|>`, so that branch never executes. Any fix placed
inside it is dead code.

### 4.3 Unsloth selects an attention backend the T4 cannot train with

```
NotImplementedError: No operator found for memory_efficient_attention_backward
  query: shape=(2, 512, 2, 8, 128) fp16, attn_bias=LowerTriangularMask
  fa3B / fa2B -> requires device capability >= (8,0), T4 has (7,5)
  cutlassB    -> operator does not support BMGHK format
```

From `unsloth/utils/attention_dispatch.py`:

```python
if HAS_XFORMERS and torch.cuda.is_available():
    _cc = torch.cuda.get_device_capability()
    if _cc[0] >= 12:          # guards GPUs that are too NEW
        HAS_XFORMERS = False
...
def select_attention_backend(use_varlen=False):
    if HAS_FLASH_ATTENTION: ...   # FA2 needs sm_80, T4 misses
    if HAS_XFORMERS: return XFORMERS   # <-- T4 lands here
    return SDPA
```

Unsloth guards against GPUs that are **too new** (sm_120+) but not against ones
that are **too old**. On a T4 (sm_75) xformers imports successfully, so XFORMERS
is selected — but every xformers *backward* kernel rejects the 5-D BMGHK layout
that Qwen2.5's grouped-query attention produces. The forward pass works
(`cutlassF` handles BMGHK), which is why generation succeeds and only
`trainer.train()` fails.

**Fix:** make `import xformers` fail before unsloth is imported. Unsloth handles
this itself (`models/_utils.py`: `except ModuleNotFoundError: xformers = None`),
falling back to SDPA. `SDPA_HAS_GQA` is True on torch 2.10, so grouped-query
attention runs natively.

Two approaches that do **not** work, and why:
- `attn_implementation="sdpa"` is a HuggingFace flag; unsloth's dispatcher never
  reads it.
- Patching flags after `import unsloth` is too late: `HAS_XFORMERS` is captured
  at import time in two separate modules (`models/llama.py:121` and
  `utils/attention_dispatch.py:35`).

Verified on the actual T4 before training was allowed to start:

```
HAS_XFORMERS  False
SDPA_HAS_GQA  True
BACKEND       sdpa
```

## 5. Timing

With the backend fixed, DPO ran correctly — the configuration was confirmed
against the spec:

```
Num examples = 2,000 | Num Epochs = 1 | Total steps = 250
Batch size per device = 1 | Gradient accumulation steps = 8
Trainable parameters = 29,933,568 of 3,115,872,256 (0.96% trained)
```

Measured throughput on T4 with SDPA: **14.65 s/it**, ~61 minutes for 250 steps.
SDPA is materially slower than the xformers path it replaces; that is the price
of the 4.3 fix on this hardware.

A parallel Colab attempt reached step ~61–100/250 before the runtime was
reclaimed ("disconnected due to inactivity or reaching its maximum duration").
Because the lab sets `save_strategy="no"` there was no checkpoint to resume from.
The Kaggle run is the one that completed, at 70 minutes wall clock including
dependency installation.

Hyperparameters were held at spec throughout and never reduced to save time:
`beta=0.1`, `lr=5e-7`, `max_length=512`, `max_prompt_length=256`,
`per_device_train_batch_size=1`, `gradient_accumulation_steps=8`, LoRA r=16 /
alpha=32, 2000 preference pairs.

### 4.4 / 4.5 NB5 merges the wrong adapter, and crashes on reload

`NB5` cell 94 is titled *"Stack SFT-mini -> DPO adapters"* and loads:

```python
model = PeftModel.from_pretrained(model, str(SFT_PATH))   # only SFT
```

`DPO_PATH` is never loaded. Yet the markdown immediately below says *"we apply
both adapters before merging"*, and the merge cell's comment says *"re-loads the
model with both SFT and DPO adapters merged"*. Both statements are false: the
exported GGUF is the **SFT model wearing a DPO label**, containing no alignment.

This is the most dangerous defect found, because it fails silently and produces
a deployable-looking artifact.

NB5 additionally crashes on reload:

```
AttributeError: Linear4bit has no attribute `base_layer`
```

The merged directory's `config.json` inherits `quantization_config` from the
4-bit base, so `from_pretrained(..., load_in_4bit=False)` re-quantizes while the
checkpoint on disk is fp16.

**Fix:** `scripts/export_gguf.py` loads both adapters, strips
`quantization_config` from the merged config, exports GGUF from the live handle
(no reload), and **refuses to run** if `dpo_metrics.json` has
`end_reward_gap: null`.

## 6. Environment note

Kaggle's default GPU allocation for this account is a **Tesla P100 (sm_60)**,
which torch 2.10+cu128 no longer builds kernels for:

```
Tesla P100-PCIE-16GB with CUDA capability sm_60 is not compatible with the
current PyTorch installation. The current PyTorch install supports
sm_70 sm_75 sm_80 sm_86 sm_90 sm_100 sm_120.
```

T4 must be requested explicitly (`machine_shape: NvidiaTeslaT4`).

## 7. To reproduce

`scripts/run_nb1_nb3.py` runs NB1 → NB2 → NB3 with all three fixes applied and
asserts the attention backend is SDPA before training starts. Budget ~75 minutes
on a T4.
