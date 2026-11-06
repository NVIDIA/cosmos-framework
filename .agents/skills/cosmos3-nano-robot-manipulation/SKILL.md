---
name: cosmos3-nano-robot-manipulation
description: >
  Post-train Cosmos3-Nano into a robot manipulation policy on LIBERO-10 and evaluate it closed-loop in the LIBERO simulator.
  Use when the user asks for Cosmos Policy, action SFT of Cosmos3-Nano on robot demonstrations, closed-loop LIBERO evaluation,
  the cost of post-training a policy on one GPU, or reproducing the 95.1 % LIBERO-10 result.
metadata:
  industry: ROB
  gpu-budget: M
---

# LIBERO robot manipulation policy

This skill is the recipe. Read and follow [README.md](README.md) in this directory before changing commands, data, or checkpoints. The assets, scripts, launchers, and configs named by that write-up are in this same folder.

Keep the framework, model, and dataset revisions pinned in the recipe. Do not bundle or rehost a third-party dataset. Report the measured result with the hardware, seeds, and control named in the recipe.
