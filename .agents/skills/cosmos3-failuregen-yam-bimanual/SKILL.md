---
name: cosmos3-failuregen-yam-bimanual
description: >
  Generate synthetic dual-arm YAM robot failure continuations with a
  post-trained Cosmos3-Nano forward-dynamics model. Use for FailureGen,
  donor-to-host action transfer, handover drops, pickup misses, pickup slips,
  robot failure video generation, and review of generated failure candidates.
---

# Cosmos 3 FailureGen for YAM bimanual robots

This skill is the recipe. Read and follow [README.md](README.md) in this
directory before changing checkpoints, framework revisions, donor/host pairs,
actions, or seeds. The application package, configuration, tests, guides, and
preview media referenced by the recipe are bundled under `assets/`.

The inference bundle is currently private and the recorded execution uses four
GB200 GPUs. Do not claim that transferred motion guarantees a failure, and do
not use generated clips as labeled training data until their outcomes have been
reviewed.
