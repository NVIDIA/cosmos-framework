# Cosmos3 Simulation Cookbooks

Cosmos3 **Simulation** models generate what a robot or vehicle would observe,
driven by a control signal: robot actions, per-camera driving controls, or a
spatial control video.

## Choose a model

| Model | Generates | From | Start here | Notebooks |
| --- | --- | --- | --- | --- |
| [Cosmos3-Nano-Sim-Bimanual](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Bimanual) | Video rollout of bimanual robot manipulation | Initial RGB frame, task description, time-aligned action trajectory | [Bimanual guide](bimanual/bimanual.md) | [Framework](bimanual/bimanual.md), [vLLM-Omni](bimanual/run_bimanual_with_vllm_omni.ipynb), [SGLang](bimanual/run_bimanual_with_sglang.ipynb) |
| [Cosmos3-Nano-Transfer-Auto](https://huggingface.co/nvidia/Cosmos3-Nano-Transfer-Auto) | Synchronized multi-camera driving video, joint mode adds LiDAR | Per-camera WSM controls and captions, HD-map LiDAR controls for joint mode | [Auto overview](auto/README.md), [Transfer-Auto guide](auto/auto.md) | [Framework](auto/run_auto_with_cosmos_framework.ipynb), [vLLM-Omni](auto/run_auto_with_vllm_omni.ipynb), [SGLang](auto/run_auto_with_sglang.ipynb) |
| [Cosmos3-Nano-Sim-Auto](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Auto) | Same outputs as Transfer-Auto, generated chunk by chunk | Same inputs as Transfer-Auto | [Auto overview](auto/README.md) | [Framework](auto/run_sim-auto_with_cosmos_framework.ipynb), [vLLM-Omni](auto/run_sim_auto_with_vllm_omni.ipynb), [SGLang](auto/run_sim-auto_with_sglang.ipynb) |
| [Cosmos3-Nano-Sim-Transfer](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Transfer) | 480p video | Caption and one edge, blur, depth, or segmentation control video | [Sim-Transfer guide](transfer/transfer.md) | [Framework](transfer/transfer.md), [vLLM-Omni](transfer/run_transfer_with_vllm_omni.ipynb), [SGLang](transfer/run_transfer_with_sglang.ipynb) |

To drive the Auto models with your own scene, render WSM controls with
[generate_wsm.ipynb](auto/generate_wsm.ipynb).

**Sim-Bimanual** renders the result of supplied actions.
Pair it with a policy model for closed-loop simulation. The examples run open loop:
they do not run a policy or feed generated frames back into one.

**Transfer-Auto** is driven like a transfer model (control video in, photoreal video
out) but belongs to the Simulation family, so it lives here rather than under
[Generator Transfer](../generator/transfer/README.md). **Sim-Auto** is its causal,
distilled student: a fixed four-step sampler with a rolling key/value cache.

[Cosmos3-Nano-Sim-Bimanual](https://flashdreams.org/main/models/cosmos3_nano_sim_bimanual/) and [Cosmos3-Nano-Sim-Auto](https://flashdreams.org/main/models/cosmos3_nano_sim_auto/) are supported by FlashDreams. Integrate with FlashDreams for accelerated inference ([Sim-Bimanual](https://github.com/NVIDIA/flashdreams/integrations_v2/cosmos3_nano_sim_bimanual), [Sim-Auto](https://github.com/NVIDIA/flashdreams/integrations_v2/cosmos3_nano_sim_auto))

## Prerequisites

- Linux with an NVIDIA GPU. GPU count and class differ by model, see each guide.
- A Hugging Face token in `HF_TOKEN` with access to the model.
- **Cosmos Framework:** a checkout set up with the [Framework setup guide](../../../../../docs/setup.md).
  The CUDA dependency groups (`cu130-train` on CUDA 13, `cu128-train` on CUDA 12.8)
  require Python 3.13 and include FlashAttention-4, which Sim-Auto needs.
  Bimanual and Sim-Transfer add model-specific setup steps in their guides.
- **vLLM-Omni:** Docker with the NVIDIA Container Toolkit. All four notebooks use
  the `vllm/vllm-omni:cosmos3` image.
- **SGLang:** Docker with the NVIDIA Container Toolkit. The four SGLang notebooks use
  the published `lmsysorg/sglang:dev` image; the Auto and Sim-Transfer notebooks read
  their inputs from the mounted cookbook root.

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
[Bimanual](bimanual/bimanual.md#check-the-result) (drift, action response),
[Transfer-Auto](auto/auto.md#validate-the-rollout) (all views present, LiDAR sweep count, cross-view consistency),
[Sim-Auto](auto/run_sim-auto_with_cosmos_framework.ipynb#5-inspect-the-result), and
[Sim-Transfer](transfer/transfer.md#validate-the-rollout) (control adherence, temporal consistency).

## Run and post-train with agent skills

Open your coding agent at the root of a Cosmos Framework checkout so it finds the
skills under `.agents/skills`.

| Model | Skills |
| --- | --- |
| Sim-Bimanual | [Inference](../../../cosmos3-nano-sim-bimanual/inference/SKILL.md) · [Post-training](../../../cosmos3-nano-sim-bimanual/post-training/SKILL.md) · [Troubleshooting](../../../cosmos3-nano-sim-bimanual/troubleshooting/SKILL.md) |
| Transfer-Auto | [Inference](../../../cosmos3-nano-transfer-auto/inference/SKILL.md) · [Post-training](../../../cosmos3-nano-transfer-auto/post-training/SKILL.md) · [Troubleshooting](../../../cosmos3-nano-transfer-auto/troubleshooting/SKILL.md) |
| Sim-Auto | [Inference](../../../cosmos3-nano-sim-auto/inference/SKILL.md) · [Post-training](../../../cosmos3-nano-sim-auto/post-training/SKILL.md) · [Troubleshooting](../../../cosmos3-nano-sim-auto/troubleshooting/SKILL.md) |
| Sim-Transfer | [Inference](../../../cosmos3-nano-sim-transfer/inference/SKILL.md) · [Post-training](../../../cosmos3-nano-sim-transfer/post-training/SKILL.md) · [Troubleshooting](../../../cosmos3-nano-sim-transfer/troubleshooting/SKILL.md) · [Setup guide](../../../cosmos3-nano-sim-transfer/shared/setup.md) · [Caption guide](../../../cosmos3-nano-sim-transfer/post-training/references/caption-guide.md) |

For environment problems shared by all models, use
[`cosmos3-env-troubleshoot`](../../../cosmos3-env-troubleshoot/SKILL.md).

## Related

- [Cosmos3 cookbooks](../README.md): shared installation and backend guidance.
