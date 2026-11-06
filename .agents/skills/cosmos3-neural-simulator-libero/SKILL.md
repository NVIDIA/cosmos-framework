---
name: cosmos3-neural-simulator-libero
description: >
  Post-train Cosmos3-Nano into a forward-dynamics neural simulator of a LIBERO robot, optionally with an inverse-dynamics head,
  and measure held-out rollout fidelity against the untrained base model.
  Use when the user asks for a neural simulator or world model of a robot, forward- or inverse-dynamics post-training,
  recovering actions from video, or held-out rollout fidelity (PSNR, SSIM, LPIPS) on LIBERO.
metadata:
  industry: ROB
  gpu-budget: M
---

# LIBERO neural simulator (forward and inverse dynamics)

This skill is the recipe. Read and follow [README.md](README.md) in this directory before changing commands, data, or checkpoints. The assets, scripts, launchers, TOMLs, experiment configs and the framework patch named by that write-up are in this same folder under `assets/neural-simulator/`; the numbers in its tables are also in `assets/neural-simulator/results/metrics.json` with the record each came from.

Keep the framework, model, tokenizer, and dataset revisions pinned in the recipe. Do not bundle or rehost a third-party dataset. Report the measured result with the hardware, seeds, control, and the weights scored (training checkpoint, regular weights) named in the recipe.
