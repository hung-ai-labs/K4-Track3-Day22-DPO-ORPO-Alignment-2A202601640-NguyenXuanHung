# Lab 22 Agent Guidelines

- Primary target: Google Colab T4 with 16 GB VRAM.
- Primary notebook: `colab/Lab22_DPO_T4.ipynb`.
- Preserve every grading requirement in `rubric.md`.
- Core work is NB1–NB4 only unless the user explicitly requests bonus work.
- Do not silently change the model size, datasets, LoRA `r`/`alpha`, DPO beta, learning rate, sample counts, or evaluation prompts.
- Explain the reason before changing any training hyperparameter.
- Avoid changes that increase T4 VRAM usage.
- Never delete notebook outputs or required submission artifacts.
- After changes, run appropriate static checks or `scripts/verify.py` where possible.
- Treat reward-curve and evaluation-result interpretation as human judgment: present evidence and never fabricate conclusions.
- Keep changes minimal and reviewable.
