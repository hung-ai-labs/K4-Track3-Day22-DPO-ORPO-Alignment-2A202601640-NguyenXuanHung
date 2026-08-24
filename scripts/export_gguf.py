"""Merge SFT + DPO adapters and export GGUF (Q4_K_M + Q5_K_M) for the +3 rigor add-on.

Replaces NB5, which has two defects (see submission/RUN-REPORT.md):

  A. NB5 cell 94 loads ONLY adapters/sft-mini. adapters/dpo is never loaded, yet
     the surrounding markdown and comment both claim "both SFT and DPO adapters
     merged". The GGUF it produces therefore contains no DPO alignment at all.

  B. NB5 reloads the merged directory with load_in_4bit=False and crashes with
     `AttributeError: Linear4bit has no attribute base_layer`. The merged
     config.json inherits `quantization_config` from the 4-bit base, so
     from_pretrained re-quantizes while the checkpoint on disk is fp16.
     Stripping that key is the fix; this script also avoids the reload entirely
     by exporting GGUF from the live handle.

Run only after adapters/dpo/ exists and dpo_metrics.json has a non-null
end_reward_gap. Exporting an untrained adapter produces a real-looking GGUF that
is not an aligned model.
"""
import json, os, sys, gc
from pathlib import Path

ROOT = Path(os.environ.get("LAB22_ROOT", "/content/lab22"))
BASE_MODEL = "unsloth/Qwen2.5-3B-bnb-4bit"
MAX_LEN = 512
SFT_PATH, DPO_PATH = ROOT/"adapters"/"sft-mini", ROOT/"adapters"/"dpo"
MERGED = Path(os.environ.get("BIG_ARTIFACT_ROOT", str(ROOT)))/"adapters"/"merged-fp16"
GGUF_DIR = Path(os.environ.get("BIG_ARTIFACT_ROOT", str(ROOT)))/"gguf"

# ---- refuse to export an untrained adapter -------------------------------
mpath = DPO_PATH/"dpo_metrics.json"
if not mpath.exists():
    sys.exit(f"missing {mpath} - run NB3 first")
metrics = json.loads(mpath.read_text())
if metrics.get("end_reward_gap") is None:
    sys.exit("dpo_metrics.json has end_reward_gap=null -> DPO never trained. "
             "Refusing to export a GGUF that would look aligned but is not.")
print(f"DPO verified: reward_gap={metrics['end_reward_gap']:+.4f}")

import torch
from unsloth import FastLanguageModel
from unsloth.utils.attention_dispatch import HAS_XFORMERS
assert not HAS_XFORMERS, "run: pip uninstall -y xformers"
from peft import PeftModel

MERGED.mkdir(parents=True, exist_ok=True); GGUF_DIR.mkdir(parents=True, exist_ok=True)

model, tok = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL, max_seq_length=MAX_LEN, dtype=None, load_in_4bit=True)
if not getattr(tok, "chat_template", None):
    sys.exit("tokenizer has no chat_template - load it from adapters/dpo instead")

# Fix A: load BOTH adapters, not just SFT.
model = PeftModel.from_pretrained(model, str(SFT_PATH))
model.load_adapter(str(DPO_PATH), adapter_name="dpo")
try:
    model.base_model.set_adapter(["default", "dpo"])
    print("active adapters:", model.active_adapters)
except Exception as exc:
    print(f"WARNING: could not activate both adapters ({exc}). "
          "Verify which adapter the merge actually captured before trusting the GGUF.")

model.save_pretrained_merged(str(MERGED), tok, save_method="merged_16bit")
print(f"merged fp16 -> {MERGED}")

# Fix B: the merged config inherits the base model's quantization_config.
cfg_path = MERGED/"config.json"
cfg = json.loads(cfg_path.read_text())
if cfg.pop("quantization_config", None) is not None:
    cfg_path.write_text(json.dumps(cfg, indent=2))
    print("stripped quantization_config from merged config.json")

for q in ("q4_k_m", "q5_k_m"):          # both tiers = the +3 add-on
    model.save_pretrained_gguf(str(GGUF_DIR), tok, quantization_method=q)
    print(f"gguf {q} -> {GGUF_DIR}")

del model; gc.collect(); torch.cuda.empty_cache()
for p in sorted(GGUF_DIR.glob("*.gguf")):
    print(f"  {p.name:55s} {p.stat().st_size/1e6:>8.1f} MB")
