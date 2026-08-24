"""beta-sweep for the +6 rigor add-on: NB3 at beta in {0.05, 0.1, 0.5}.

Reuses the SFT adapter and preference parquet, so only DPO is repeated.
Cost: 3 x 250 steps. Measured 14.65 s/it on a T4 with SDPA -> ~3 hours total.

  pip uninstall -y xformers
  python scripts/beta_sweep.py

Everything except beta is held at spec.
"""
import json, os, subprocess, sys
from pathlib import Path

ROOT = Path(os.environ.get("LAB22_ROOT", "/content/lab22"))
BETAS = [float(b) for b in os.environ.get("BETAS", "0.05,0.1,0.5").split(",")]
here = Path(__file__).parent

results = []
for beta in BETAS:
    out = ROOT/"adapters"/f"dpo-b{beta}"
    env = dict(os.environ, DPO_BETA=str(beta), DPO_OUT_OVERRIDE=str(out), SKIP_SFT="1")
    print(f"\n=== beta={beta} -> {out} ===", flush=True)
    rc = subprocess.run([sys.executable, str(here/"run_nb1_nb3.py")], env=env).returncode
    m = out/"dpo_metrics.json"
    if rc == 0 and m.exists():
        d = json.loads(m.read_text())
        results.append((beta, d.get("end_reward_gap")))
    else:
        print(f"beta={beta} failed (rc={rc})")
        results.append((beta, None))

print("\nbeta, end_reward_gap")
for b, g in results:
    print(f"{b}, {g}")

ok = [(b, g) for b, g in results if g is not None]
if len(ok) >= 2:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot([b for b, _ in ok], [g for _, g in ok], marker="o", linewidth=1.6, color="#2e548a")
    ax.set_xscale("log"); ax.set_xlabel("beta (log scale)"); ax.set_ylabel("end reward gap")
    ax.set_title("DPO reward gap vs beta - T4 - Qwen2.5-3B - 2000 pairs")
    ax.grid(True, alpha=0.3); fig.tight_layout()
    dst = ROOT/"submission"/"screenshots"/"05-beta-sweep.png"
    dst.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst, dpi=120)
    print(f"saved {dst}")
