"""NB1 -> NB2 -> NB3 with the three lab defects fixed. See submission/RUN-REPORT.md.

Fixes applied:
  1. SFT dataset 5CD-AI/Vietnamese-alpaca-cleaned returns HTTP 401 -> use
     saillab/alpaca-vietnamese-cleaned (same instruction/input/output columns).
  2. Qwen2.5 *base* ships no tokenizer.chat_template -> install ChatML.
  3. Unsloth picks the xformers backend on sm_75, whose backward kernels reject
     Qwen2.5's BMGHK grouped-query layout -> uninstall xformers BEFORE this runs
     so unsloth falls back to SDPA. Asserted below, before any training.

Prerequisite:  pip uninstall -y xformers
Hyperparameters are exactly those in the lab spec and must not be reduced.
"""
import os, json, gc, time
from pathlib import Path
os.environ["COMPUTE_TIER"] = "T4"
T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)

import torch
from unsloth import FastLanguageModel
from unsloth.utils.attention_dispatch import select_attention_backend, HAS_XFORMERS, SDPA_HAS_GQA
assert not HAS_XFORMERS, "xformers still active - run: pip uninstall -y xformers"
assert select_attention_backend() == "sdpa", "backend is not sdpa"
log(f"backend=sdpa HAS_XFORMERS={HAS_XFORMERS} SDPA_HAS_GQA={SDPA_HAS_GQA}")

BASE_MODEL = "unsloth/Qwen2.5-3B-bnb-4bit"
MAX_LEN, MAX_PROMPT_LEN = 512, 256
PER_DEVICE_BATCH, GRAD_ACCUM = 1, 8
BETA = float(os.environ.get("DPO_BETA", "0.1"))   # spec default; beta_sweep.py overrides
LR, EPOCHS = 5e-7, 1
SFT_SLICE, PREF_SLICE = 1000, 2000
LORA = dict(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
            use_gradient_checkpointing="unsloth", random_state=42, use_rslora=False, loftq_config=None)
ROOT = Path(os.environ.get("LAB22_ROOT", "/content/lab22"))
SFT_PATH = ROOT/"adapters"/"sft-mini"
DPO_OUT = Path(os.environ.get("DPO_OUT_OVERRIDE", str(ROOT/"adapters"/"dpo")))
PREF_DIR, SHOTS = ROOT/"data"/"pref", ROOT/"submission"/"screenshots"
for p in (SFT_PATH, DPO_OUT, PREF_DIR, SHOTS): p.mkdir(parents=True, exist_ok=True)

CHATML = ("{% for message in messages %}"
          "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
          "{% endfor %}"
          "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}")
def ensure_ct(tok):
    # NOTE: do NOT nest this inside `if tok.pad_token is None` - Qwen2.5 base already
    # defines pad_token=<|vision_pad|>, so that branch never runs.
    if not getattr(tok, "chat_template", None):
        tok.chat_template = CHATML
    return tok

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset

SKIP_SFT = os.environ.get("SKIP_SFT") == "1" and (SFT_PATH/"adapter_model.safetensors").exists() \
    and (PREF_DIR/"train.parquet").exists()

if SKIP_SFT:
    from transformers import AutoTokenizer
    log("SKIP_SFT=1 and artifacts present -> reusing existing SFT adapter + parquet")
    tok = ensure_ct(AutoTokenizer.from_pretrained(str(SFT_PATH)))
else:
  log("=== NB1 SFT ===")
  model, tok = FastLanguageModel.from_pretrained(  model_name=BASE_MODEL, max_seq_length=MAX_LEN, dtype=None, load_in_4bit=True)
  ensure_ct(tok)
  if tok.pad_token is None: tok.pad_token = tok.eos_token
  model = FastLanguageModel.get_peft_model(model, **LORA)
  ds = load_dataset(os.environ.get("SFT_DATASET", "saillab/alpaca-vietnamese-cleaned"), split=f"train[:{SFT_SLICE}]")
  log(f"sft rows={len(ds)} cols={ds.column_names}")
  def fmt(row):
      m = []
      if row.get("instruction"):
          p = row["instruction"]
          if row.get("input"): p += "\n\n" + row["input"]
          m.append({"role":"user","content":p})
      if row.get("output"): m.append({"role":"assistant","content":row["output"]})
      return {"text": tok.apply_chat_template(m, tokenize=False, add_generation_prompt=False)}
  dsf = ds.map(fmt, remove_columns=ds.column_names)
  from trl import SFTTrainer, SFTConfig
  sc = SFTConfig(output_dir=str(ROOT/"adapters"/"sft-ckpt"), per_device_train_batch_size=PER_DEVICE_BATCH,
      gradient_accumulation_steps=GRAD_ACCUM, num_train_epochs=EPOCHS, learning_rate=2e-4, warmup_ratio=0.03,
      lr_scheduler_type="cosine", logging_steps=10, save_strategy="no", optim="adamw_8bit",
      bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(), seed=42,
      max_length=MAX_LEN, dataset_text_field="text", report_to="none")
  tr = SFTTrainer(model=model, processing_class=tok, args=sc, train_dataset=dsf)
  sres = tr.train()
  log(f"SFT final loss={sres.training_loss:.4f}")
  ls = [x["loss"] for x in tr.state.log_history if "loss" in x]
  st = [x["step"] for x in tr.state.log_history if "loss" in x]
  f, a = plt.subplots(figsize=(8,4)); a.plot(st, ls, marker="o", markersize=3, linewidth=1.2)
  a.set_xlabel("Training step"); a.set_ylabel("Loss"); a.grid(True, alpha=0.3)
  a.set_title(f"SFT-mini loss - T4 - Qwen2.5-3B - {SFT_SLICE} samples")
  f.tight_layout(); f.savefig(SHOTS/"02-sft-loss.png", dpi=120); plt.close(f)
  tr.model.save_pretrained(str(SFT_PATH)); tok.save_pretrained(str(SFT_PATH))
  log(f"saved SFT -> {SFT_PATH}")
  del tr, model; gc.collect(); torch.cuda.empty_cache()

  log("=== NB2 preference data ===")
  pds = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned", split=f"train[:{PREF_SLICE}]")
  def pfmt(row):
      pt = tok.apply_chat_template([{"role":"user","content":row["prompt"]}], tokenize=False, add_generation_prompt=True)
      ch = row["chosen"][-1]["content"] if isinstance(row["chosen"], list) else row["chosen"]
      rj = row["rejected"][-1]["content"] if isinstance(row["rejected"], list) else row["rejected"]
      return {"prompt": pt, "chosen": ch, "rejected": rj}
  pref = pds.map(pfmt, remove_columns=pds.column_names)
  pref.to_parquet(str(PREF_DIR/"train.parquet"))
  pref.select(range(len(pref)-50, len(pref))).to_parquet(str(PREF_DIR/"eval.parquet"))
  log(f"saved {len(pref)} pairs -> {PREF_DIR/'train.parquet'}")

log("=== NB3 DPO ===")
from peft import PeftModel
from trl import DPOConfig, DPOTrainer
model, tok = FastLanguageModel.from_pretrained(model_name=BASE_MODEL, max_seq_length=MAX_LEN, dtype=None, load_in_4bit=True)
ensure_ct(tok)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = PeftModel.from_pretrained(model, str(SFT_PATH), is_trainable=True)
model = FastLanguageModel.get_peft_model(model, **LORA)
dc = DPOConfig(output_dir=str(ROOT/"adapters"/"dpo-ckpt"), per_device_train_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM, num_train_epochs=EPOCHS, learning_rate=LR, beta=BETA,
    max_length=MAX_LEN, max_prompt_length=MAX_PROMPT_LEN, warmup_ratio=0.1, lr_scheduler_type="cosine",
    logging_steps=10, save_strategy="no", optim="adamw_8bit",
    bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(), seed=42,
    loss_type="sigmoid", report_to="none")
pref_ds = Dataset.from_parquet(str(PREF_DIR/"train.parquet"))
dtr = DPOTrainer(model=model, ref_model=None, args=dc, train_dataset=pref_ds, processing_class=tok)
dres = dtr.train()
log(f"DPO final loss={dres.training_loss:.4f}")

import pandas as pd
lg = pd.DataFrame(dtr.state.log_history)
cc = "rewards/chosen" if "rewards/chosen" in lg.columns else None
rc = "rewards/rejected" if "rewards/rejected" in lg.columns else None
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
if cc and rc:
    m = lg[cc].notna()
    ax[0].plot(lg["step"][m], lg[cc][m], label="chosen reward", color="#2e548a", linewidth=1.5)
    ax[0].plot(lg["step"][m], lg[rc][m], label="rejected reward", color="#c83538", linewidth=1.5)
    ax[0].axhline(0, color="#888", linestyle=":", linewidth=0.7); ax[0].legend(); ax[0].grid(True, alpha=0.3)
    ax[0].set_xlabel("Training step"); ax[0].set_ylabel("Implicit reward (log pi/pi_ref)")
    ax[0].set_title("DPO reward curves (chosen vs rejected)")
    ax[1].plot(lg["step"][m], lg[cc][m]-lg[rc][m], color="#2a7f5f", linewidth=1.5)
    ax[1].axhline(0, color="#888", linestyle=":", linewidth=0.7); ax[1].grid(True, alpha=0.3)
    ax[1].set_xlabel("Training step"); ax[1].set_ylabel("Reward gap"); ax[1].set_title("Reward margin")
fig.tight_layout(); fig.savefig(SHOTS/"03-dpo-reward-curves.png", dpi=120); plt.close(fig)

lc = lr_ = gap = None
if cc and rc:
    s = lg[lg[cc].notna()]
    lc = float(s[cc].iloc[-5:].mean()); lr_ = float(s[rc].iloc[-5:].mean()); gap = lc - lr_
    log(f"END chosen={lc:+.4f} rejected={lr_:+.4f} gap={gap:+.4f}")
dtr.model.save_pretrained(str(DPO_OUT)); tok.save_pretrained(str(DPO_OUT))
json.dump({"compute_tier":"T4","base_model":BASE_MODEL,"beta":BETA,"lr":LR,"epochs":EPOCHS,
  "pref_pairs":len(pref_ds),"attn_backend":"sdpa","final_train_loss":float(dres.training_loss),
  "end_chosen_reward":lc,"end_rejected_reward":lr_,"end_reward_gap":gap},
  open(DPO_OUT/"dpo_metrics.json","w"), indent=2)
log(f"saved DPO -> {DPO_OUT}")
log("ALL DONE")
