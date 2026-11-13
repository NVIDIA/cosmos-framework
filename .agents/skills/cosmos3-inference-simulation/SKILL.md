---
name: cosmos3-inference-simulation
description: >
  Entry skill for Cosmos3 Simulation inference (Cosmos Framework, vLLM-Omni, SGLang): routes a request to one of four
  Cosmos3-Nano models. Sim-Bimanual: video rollout of bimanual robot manipulation from an initial RGB frame, task
  description and action trajectory. Transfer-Auto: synchronized multi-camera driving video (joint mode adds LiDAR)
  from WSM controls and captions. Sim-Auto: the same outputs, generated chunk by chunk. Sim-Transfer: 480p video from
  a caption and one edge, blur, depth or segmentation control video. Use when the user asks "run a Cosmos3 Simulation
  model", "which Simulation model do I need", "simulate what my robot or vehicle would observe" or "how do I validate
  a Simulation rollout". Not for Generator Transfer (cosmos3-inference-transfer), action inference
  (cosmos3-inference-action), audiovisual generation (cosmos3-inference-generate-media) or post-training
  (cosmos3-post-train-simulation).
---

# Cosmos3 Simulation Inference: Entry

Cosmos3 **Simulation** models generate what a robot or vehicle would observe,
driven by a control signal: robot actions, per-camera driving controls, or a
spatial control video.

This is the entry skill for Cosmos3 Simulation inference. It runs no command: it chooses the model, states what the
four models share, and hands the request to one of three block skills (Sim-Bimanual, Auto, Sim-Transfer). Its source
is the Simulation cookbook README, carried verbatim as [the README copy](references/simulation-readme.md). The
paragraph above and every passage introduced below as "README, verbatim" are that README's text with only link
targets re-pointed.

## When to use this skill

- Use when a request is about Cosmos3 Simulation and the model is not settled: pick the model under "Choose a model",
  then open its block skill from "Route to a block skill".
- Use for what the four models share: "Shared prerequisites" (container images and tags, the `--defaults-file` note)
  and "Validate a rollout" (the four-point checklist).
- Not for Generator Transfer (`../cosmos3-inference-transfer/SKILL.md`), which the README sets apart from
  Transfer-Auto, and not for post-training (see "Post-training").

## Choose a model

README "Choose a model", verbatim. The guides it links were copied, and the notebooks converted 1:1, into the block
skills' reference files; the links point there.

| Model                                                                                  | Generates                                                      | From                                                                       | Start here                                                                                                                                                                     | Notebooks                                                                                                                                                                                                                                                                  |
| -------------------------------------------------------------------------------------- | -------------------------------------------------------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Cosmos3-Nano-Sim-Bimanual](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Bimanual)   | Video rollout of bimanual robot manipulation                   | Initial RGB frame, task description, time-aligned action trajectory        | [Bimanual guide](../cosmos3-inference-simulation-bimanual/references/bimanual-guide.md)                                                                                        | [Framework](../cosmos3-inference-simulation-bimanual/references/bimanual-guide.md), [vLLM-Omni](../cosmos3-inference-simulation-bimanual/references/vllm-omni.md), [SGLang](../cosmos3-inference-simulation-bimanual/references/sglang.md)                                 |
| [Cosmos3-Nano-Transfer-Auto](https://huggingface.co/nvidia/Cosmos3-Nano-Transfer-Auto) | Synchronized multi-camera driving video, joint mode adds LiDAR | Per-camera WSM controls and captions, HD-map LiDAR controls for joint mode | [Auto overview](../cosmos3-inference-simulation-auto/references/auto-readme.md), [Transfer-Auto guide](../cosmos3-inference-simulation-auto/references/transfer-auto-guide.md) | [Framework](../cosmos3-inference-simulation-auto/references/transfer-auto-cosmos-framework.md), [vLLM-Omni](../cosmos3-inference-simulation-auto/references/transfer-auto-vllm-omni.md), [SGLang](../cosmos3-inference-simulation-auto/references/transfer-auto-sglang.md) |
| [Cosmos3-Nano-Sim-Auto](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Auto)           | Same outputs as Transfer-Auto, generated chunk by chunk        | Same inputs as Transfer-Auto                                               | [Auto overview](../cosmos3-inference-simulation-auto/references/auto-readme.md)                                                                                                | [Framework](../cosmos3-inference-simulation-auto/references/sim-auto-cosmos-framework.md), [vLLM-Omni](../cosmos3-inference-simulation-auto/references/sim-auto-vllm-omni.md), [SGLang](../cosmos3-inference-simulation-auto/references/sim-auto-sglang.md)                |
| [Cosmos3-Nano-Sim-Transfer](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Transfer)   | 480p video                                                     | Caption and one edge, blur, depth, or segmentation control video           | [Sim-Transfer guide](../cosmos3-inference-simulation-transfer/references/transfer-guide.md)                                                                                    | [Framework](../cosmos3-inference-simulation-transfer/references/transfer-guide.md), [vLLM-Omni](../cosmos3-inference-simulation-transfer/references/vllm-omni.md), [SGLang](../cosmos3-inference-simulation-transfer/references/sglang.md)                                 |

To drive the Auto models with your own scene, render WSM controls with
[generate_wsm.ipynb](../cosmos3-inference-simulation-auto/references/generate-wsm.md).

**Sim-Bimanual** renders the result of supplied actions.
Pair it with a policy model for closed-loop simulation. The examples run open loop:
they do not run a policy or feed generated frames back into one.

**Transfer-Auto** is driven like a transfer model (control video in, photoreal video
out) but belongs to the Simulation family, so it lives here rather than under
[Generator Transfer](../cosmos3-inference-transfer/SKILL.md). **Sim-Auto** is its causal,
distilled student: a fixed four-step sampler with a rolling key/value cache.

FlashDreams support for Sim-Bimanual and Sim-Auto: the last paragraph of "Choose a model" in
[the README copy](references/simulation-readme.md#choose-a-model).

## Route to a block skill

Open the block skill of the chosen model and follow it from the top; for the two Auto models, open the named section.
Each block covers the three paths of the README's "Notebooks" column: Framework, vLLM-Omni and SGLang.

| Model         | Block skill                                                                                                             | When a run fails                                                                                                 |
| ------------- | ----------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Sim-Bimanual  | [cosmos3-inference-simulation-bimanual](../cosmos3-inference-simulation-bimanual/SKILL.md)                              | [Sim-Bimanual inference troubleshooting](../cosmos3-inference-simulation-bimanual/references/troubleshooting.md) |
| Transfer-Auto | [cosmos3-inference-simulation-auto, section Transfer-Auto](../cosmos3-inference-simulation-auto/SKILL.md#transfer-auto) | [Auto inference troubleshooting](../cosmos3-inference-simulation-auto/references/troubleshooting.md)             |
| Sim-Auto      | [cosmos3-inference-simulation-auto, section Sim-Auto](../cosmos3-inference-simulation-auto/SKILL.md#sim-auto)           | [Auto inference troubleshooting](../cosmos3-inference-simulation-auto/references/troubleshooting.md)             |
| Sim-Transfer  | [cosmos3-inference-simulation-transfer](../cosmos3-inference-simulation-transfer/SKILL.md)                              | [Sim-Transfer inference troubleshooting](../cosmos3-inference-simulation-transfer/references/troubleshooting.md) |

For a failure that is not specific to one model, README, verbatim:

For environment problems shared by all models, use
[`cosmos3-env-troubleshoot`](../cosmos3-env-troubleshoot/SKILL.md).

## Shared prerequisites

README "Prerequisites", verbatim. "Guide" and "notebooks" mean the block skills' reference files (see "Choose a
model"). The skill for setting up the Framework checkout is `../cosmos3-setup/SKILL.md`.

- Linux with an NVIDIA GPU. GPU count and class differ by model, see each guide.
- A Hugging Face token in `HF_TOKEN` with access to the model.
- **Cosmos Framework:** a checkout set up with the [Framework setup guide](../../../docs/setup.md).
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

README "Validate a rollout", verbatim:

For each sample, check that:

1. `sample_outputs.json` reports `status: success`.
2. Every output video decodes fully at the requested frame count and FPS.
3. The motion follows the control signal (actions, WSM controls, or the spatial
   control). Visual quality alone does not show control adherence.
4. Generated data is labeled as synthetic before it enters any dataset.

Use a fixed seed when comparing checkpoints or runtime changes. Then run the
model-specific checks:
[Bimanual](../cosmos3-inference-simulation-bimanual/references/bimanual-guide.md#check-the-result) (drift, action response),
[Transfer-Auto](../cosmos3-inference-simulation-auto/references/transfer-auto-guide.md#validate-the-rollout) (all views present, LiDAR sweep count, cross-view consistency),
[Sim-Auto](../cosmos3-inference-simulation-auto/references/sim-auto-cosmos-framework.md#5-inspect-the-result), and
[Sim-Transfer](../cosmos3-inference-simulation-transfer/references/transfer-guide.md#validate-the-rollout) (control adherence, temporal consistency).

## Post-training

Post-training is outside this skill and its three blocks. Its entry skill is
[cosmos3-post-train-simulation](../cosmos3-post-train-simulation/SKILL.md); it routes to the post-training skill of
each model (Transfer-Auto and Sim-Auto are two sections of one skill):

- [The Sim-Bimanual post-training skill](../cosmos3-post-train-simulation-bimanual/SKILL.md)
- [The Transfer-Auto post-training skill](../cosmos3-post-train-simulation-auto/SKILL.md#fine-tune-transfer-auto)
- [The Sim-Auto post-training skill](../cosmos3-post-train-simulation-auto/SKILL.md#distill-sim-auto)
- [The Sim-Transfer post-training skill](../cosmos3-post-train-simulation-transfer/SKILL.md)

## Reference files

| File                              | Content                                                                                                                                                                                                                   |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `references/simulation-readme.md` | The Simulation cookbook README, copied verbatim with link targets re-pointed: "Choose a model" (with the FlashDreams paragraph), "Prerequisites", "Validate a rollout", "Run and post-train with agent skills", "Related" |

## Related skills

| Skill                                    | When to use                                                                                     |
| ---------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `../cosmos3-setup/SKILL.md`              | Installation and environment setup of the Cosmos Framework checkout                             |
| `../cosmos3-env-troubleshoot/SKILL.md`   | README: "environment problems shared by all models"                                             |
| `../cosmos3-inference/SKILL.md`          | General Cosmos3 inference guidance: batch generation, serving, parallelism, sampling parameters |
| `../cosmos3-inference-transfer/SKILL.md` | README: "Generator Transfer", the transfer skill outside the Simulation family                  |
| `../cookbooks/cosmos3/README.md`         | README "Related": "shared installation and backend guidance"                                    |
