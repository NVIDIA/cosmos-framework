# Cosmos3 Simulation Cookbooks

> **Skill note (not in the source):** this file is the Simulation cookbook README `README.md` (retired into this skill from `.agents/skills/cookbooks/cosmos3/simulation/README.md`, cosmos-framework `main` 45fecab) copied verbatim. Edits are limited to three kinds: link targets that pointed at files this migration moved were re-pointed; path anchors in code blocks that the move invalidated were replaced; and location text that named the retired cookbook folder now names this skill's folder. Each code or location change is marked in place with a skill note. The changes are recorded in `docs/cookbook_to_skill_migration.md`.

Cosmos3 **Simulation** models generate what a robot or vehicle would observe,
driven by a control signal: robot actions, per-camera driving controls, or a
spatial control video.

## Choose a model

| Model                                                                                  | Generates                                                      | From                                                                       | Start here                                                                                                                                                                           | Notebooks                                                                                                                                                                                                                                                                           |
| -------------------------------------------------------------------------------------- | -------------------------------------------------------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Cosmos3-Nano-Sim-Bimanual](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Bimanual)   | Video rollout of bimanual robot manipulation                   | Initial RGB frame, task description, time-aligned action trajectory        | [Bimanual guide](../../cosmos3-inference-simulation-bimanual/references/bimanual-guide.md)                                                                                           | [Framework](../../cosmos3-inference-simulation-bimanual/references/bimanual-guide.md), [vLLM-Omni](../../cosmos3-inference-simulation-bimanual/references/vllm-omni.md), [SGLang](../../cosmos3-inference-simulation-bimanual/references/sglang.md)                                 |
| [Cosmos3-Nano-Transfer-Auto](https://huggingface.co/nvidia/Cosmos3-Nano-Transfer-Auto) | Synchronized multi-camera driving video, joint mode adds LiDAR | Per-camera WSM controls and captions, HD-map LiDAR controls for joint mode | [Auto overview](../../cosmos3-inference-simulation-auto/references/auto-readme.md), [Transfer-Auto guide](../../cosmos3-inference-simulation-auto/references/transfer-auto-guide.md) | [Framework](../../cosmos3-inference-simulation-auto/references/transfer-auto-cosmos-framework.md), [vLLM-Omni](../../cosmos3-inference-simulation-auto/references/transfer-auto-vllm-omni.md), [SGLang](../../cosmos3-inference-simulation-auto/references/transfer-auto-sglang.md) |
| [Cosmos3-Nano-Sim-Auto](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Auto)           | Same outputs as Transfer-Auto, generated chunk by chunk        | Same inputs as Transfer-Auto                                               | [Auto overview](../../cosmos3-inference-simulation-auto/references/auto-readme.md)                                                                                                   | [Framework](../../cosmos3-inference-simulation-auto/references/sim-auto-cosmos-framework.md), [vLLM-Omni](../../cosmos3-inference-simulation-auto/references/sim-auto-vllm-omni.md), [SGLang](../../cosmos3-inference-simulation-auto/references/sim-auto-sglang.md)                |
| [Cosmos3-Nano-Sim-Transfer](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Transfer)   | 480p video                                                     | Caption and one edge, blur, depth, or segmentation control video           | [Sim-Transfer guide](../../cosmos3-inference-simulation-transfer/references/transfer-guide.md)                                                                                       | [Framework](../../cosmos3-inference-simulation-transfer/references/transfer-guide.md), [vLLM-Omni](../../cosmos3-inference-simulation-transfer/references/vllm-omni.md), [SGLang](../../cosmos3-inference-simulation-transfer/references/sglang.md)                                 |

To drive the Auto models with your own scene, render WSM controls with
[generate_wsm.ipynb](../../cosmos3-inference-simulation-auto/references/generate-wsm.md).

**Sim-Bimanual** renders the result of supplied actions.
Pair it with a policy model for closed-loop simulation. The examples run open loop:
they do not run a policy or feed generated frames back into one.

**Transfer-Auto** is driven like a transfer model (control video in, photoreal video
out) but belongs to the Simulation family, so it lives here rather than under
[Generator Transfer](../../cosmos3-inference-transfer/SKILL.md). **Sim-Auto** is its causal,
distilled student: a fixed four-step sampler with a rolling key/value cache.

[Cosmos3-Nano-Sim-Bimanual](https://flashdreams.org/main/models/cosmos3_nano_sim_bimanual/) and [Cosmos3-Nano-Sim-Auto](https://flashdreams.org/main/models/cosmos3_nano_sim_auto/) are supported by FlashDreams. Integrate with FlashDreams for accelerated inference ([Sim-Bimanual](https://github.com/NVIDIA/flashdreams/integrations_v2/cosmos3_nano_sim_bimanual), [Sim-Auto](https://github.com/NVIDIA/flashdreams/integrations_v2/cosmos3_nano_sim_auto))

## Prerequisites

- Linux with an NVIDIA GPU. GPU count and class differ by model, see each guide.
- A Hugging Face token in `HF_TOKEN` with access to the model.
- **Cosmos Framework:** a checkout set up with the [Framework setup guide](../../../../docs/setup.md).
  The CUDA dependency groups (`cu130-train` on CUDA 13, `cu128-train` on CUDA 12.8)
  require Python 3.13 and include FlashAttention-4, which Sim-Auto needs.
  Bimanual and Sim-Transfer add model-specific setup steps in their guides.
- **vLLM-Omni:** Docker with the NVIDIA Container Toolkit. All four notebooks use
  the `vllm/vllm-omni:cosmos3` image.
- **SGLang:** Docker with the NVIDIA Container Toolkit. The four SGLang notebooks use
  the published `lmsysorg/sglang:dev` image; the Auto and Sim-Transfer notebooks read
  their inputs from the mounted cookbook root. The `dev` tag moves with SGLang's main
  branch; the notebooks were validated against SGLang commit [`ab580b2e`](https://github.com/sgl-project/sglang/commit/ab580b2e060059c0f8387b598a9f648dcec13f93),
  published as the fixed tag `lmsysorg/sglang:nightly-dev-20261020-ab580b2e`.

If you pass `--defaults-file` to Framework inference, it **replaces** the built-in
sampling defaults rather than merging with them. Start from the defaults file the
model's guide provides.

## Validate a rollout

For each sample, check that:

1. `sample_outputs.json` reports `status: success`.
2. Every output video decodes fully at the requested frame count and FPS.
3. The motion follows the control signal (actions, WSM controls, or the spatial
   control). Visual quality alone does not show control adherence.
4. Generated data is labeled as synthetic before it enters any dataset.

Use a fixed seed when comparing checkpoints or runtime changes. Then run the
model-specific checks:
[Bimanual](../../cosmos3-inference-simulation-bimanual/references/bimanual-guide.md#check-the-result) (drift, action response),
[Transfer-Auto](../../cosmos3-inference-simulation-auto/references/transfer-auto-guide.md#validate-the-rollout) (all views present, LiDAR sweep count, cross-view consistency),
[Sim-Auto](../../cosmos3-inference-simulation-auto/references/sim-auto-cosmos-framework.md#5-inspect-the-result), and
[Sim-Transfer](../../cosmos3-inference-simulation-transfer/references/transfer-guide.md#validate-the-rollout) (control adherence, temporal consistency).

## Run and post-train with agent skills

Open your coding agent at the root of a Cosmos Framework checkout so it finds the
skills under `.agents/skills`.

| Model         | Skills                                                                                                                                                                                                                                                                                                                                                                                                                |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Sim-Bimanual  | [Inference](../../cosmos3-inference-simulation-bimanual/SKILL.md) · [Post-training](../../cosmos3-post-train-simulation-bimanual/SKILL.md) · [Troubleshooting](../../cosmos3-inference-simulation-bimanual/references/troubleshooting.md)                                                                                                                                                                             |
| Transfer-Auto | [Inference](../../cosmos3-inference-simulation-auto/SKILL.md#transfer-auto) · [Post-training](../../cosmos3-post-train-simulation-auto/SKILL.md#fine-tune-transfer-auto) · [Troubleshooting](../../cosmos3-inference-simulation-auto/references/troubleshooting.md)                                                                                                                                                   |
| Sim-Auto      | [Inference](../../cosmos3-inference-simulation-auto/SKILL.md#sim-auto) · [Post-training](../../cosmos3-post-train-simulation-auto/SKILL.md#distill-sim-auto) · [Troubleshooting](../../cosmos3-inference-simulation-auto/references/troubleshooting.md)                                                                                                                                                               |
| Sim-Transfer  | [Inference](../../cosmos3-inference-simulation-transfer/SKILL.md) · [Post-training](../../cosmos3-post-train-simulation-transfer/SKILL.md) · [Troubleshooting](../../cosmos3-inference-simulation-transfer/references/troubleshooting.md) · [Setup guide](../../cosmos3-inference-simulation-transfer/references/setup.md) · [Caption guide](../../cosmos3-inference-simulation-transfer/references/caption-guide.md) |

For environment problems shared by all models, use
[`cosmos3-env-troubleshoot`](../../cosmos3-env-troubleshoot/SKILL.md).

## Related

- [Cosmos3 cookbooks](../../cookbooks/cosmos3/README.md): shared installation and backend guidance.
