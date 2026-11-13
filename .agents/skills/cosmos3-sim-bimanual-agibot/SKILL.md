---
name: cosmos3-sim-bimanual-agibot
description: >
  Run the released Cosmos3-Nano-Sim-Bimanual checkpoint on held-out public AgiBotWorld-Beta episodes and measure its
  speed and its fidelity to the recorded video against a frozen-first-frame control and a wrong-action control.
  Use when the user asks to evaluate Sim-Bimanual on real AgiBot data, convert AgiBot episodes to the checkpoint's
  schema-5 actions, preview a two-arm policy in a neural simulator, run action-conditioned rollouts from recorded
  actions, or check that the simulator follows the actions it is given.
metadata:
  industry: ROB
  gpu-budget: S
---

# Sim-Bimanual: neural simulator preview on AgiBot data

This skill is the recipe. Read and follow [README.md](README.md) in this directory before changing commands, data, or
checkpoints. The scripts, manifests, prompts and result tables it names are in this folder under `assets/sim-bimanual/`;
[tools/README.md](assets/sim-bimanual/tools/README.md) says what each script reads and writes.

Keep the framework, model and dataset revisions pinned as the README states. Do not bundle or rehost a third-party
dataset: AgiBotWorld-Beta is CC BY-NC-SA 4.0 and gated, and each user downloads it from the publisher after accepting
the terms.

Rules for a run:

- The action file must already be normalised and masked (schema 5, `normalized_unified_v1`); inference normalises
  nothing. Produce it with `tools/convert_agibot_episode.py` and run `tools/check_conversion_gate.py` before any rollout;
  stop if it does not exit 0.
- Before scoring, check that frame 0 of each rollout scores above 30 dB against the top 352 rows of the real clip.
- Report the result with the hardware, seed, checkpoint revision, episode manifest and both controls. A working run
  beats the action-swap control overall and the frozen first frame overall, and stays above the frozen frame from about
  frame 90 to the end (measured: overall 19.30 dB against 15.62 frozen and 14.34 swap; crossing at frame 88).
