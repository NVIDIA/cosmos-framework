---
name: cosmos-h-surgical-robotics
description: >
  Post-train Cosmos3-H-Embodied-Nano for forward-dynamics surgical simulation
  on SutureBot (dVRK suturing demonstrations), and evaluate it against the
  released checkpoint. Use when the user asks to "fine-tune Cosmos3-H-Embodied-Nano
  on SutureBot", "post-train a surgical world model", "convert SutureBot to
  LeRobot", "train forward dynamics on dVRK data", or "evaluate action-conditioned
  surgical video prediction (PSNR vs the released checkpoint)", or wants the
  measured cost, hardware and results of that recipe. Not for inverse dynamics or
  policy training (not covered), and not for Cosmos3 models without the
  cosmos3_h_embodied package.
---

# Cosmos-H surgical robotics

This skill is the recipe. Read and follow [README.md](README.md) in this
directory before changing commands, data, or checkpoints. The assets, scripts,
launcher, and config named by that write-up are in this same folder.

Do not bundle or rehost SutureBot data; the recipe downloads it from its
publisher. Report only measured results with their hardware, seeds, and control.
