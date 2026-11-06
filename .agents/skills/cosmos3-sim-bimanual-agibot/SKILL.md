---
name: cosmos3-sim-bimanual-agibot
description: >
  Run the Cosmos3-Nano-Sim-Bimanual checkpoint on held-out public AgiBotWorld-Beta episodes and measure its speed and its
  fidelity to the recorded video against a frozen-first-frame control and a wrong-action control.
  Use when the user asks to evaluate Sim-Bimanual on real AgiBot data, preview a two-arm policy in a neural simulator,
  run action-conditioned rollouts from recorded actions, or check that the simulator follows the actions it is given.
metadata:
  industry: ROB
  gpu-budget: S
---

# Sim-Bimanual: neural simulator preview on AgiBot data

This skill is the recipe. Read and follow [README.md](README.md) in this directory before changing commands, data, or checkpoints. The assets, scripts, manifests and result tables named by that write-up are in this same folder under `assets/sim-bimanual/`; [CHECKLIST.md](CHECKLIST.md) says which inputs are backed by a run record and which are reconstructions.

Keep the framework, model, and dataset revisions pinned in the recipe. Do not bundle or rehost a third-party dataset: AgiBotWorld-Beta is CC BY-NC-SA 4.0 and gated, and each user downloads it from the publisher after accepting the terms. Report the measured result with the hardware, seed, episode manifest and the two controls named in the recipe; a rollout that does not beat the frozen-first-frame control from about frame 60 and the action-swap control overall is not a working run.
