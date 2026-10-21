<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 NIM Release Notes

## 2.1.0

Image: `nvcr.io/nim/nvidia/cosmos3:2.1.0`.

### New Models and Features

- **Edge Reasoner** (`NIM_MODEL_VARIANT=edge`): BF16 and FP8 profiles with a
  12-GiB per-device floor and no DFlash draft.
- **Edge-DROID policy** (`NIM_MODEL_VARIANT=edge-droid`): a BF16 Generator
  policy with action-only output. It is
  [experimental](support-matrix.md#choose-a-model) because it has not been
  validated.
- **Streaming video sessions** for the Reasoner over WebSocket and REST
  (`NIM_ENABLE_STREAMING=true`). Refer to [Streaming](streaming.md).
- **NVIDIA Dynamo deployment** for the Reasoner, including REST streaming. Refer to
  [Dynamo Deployment](dynamo/dynamo_deployment.md).
- **Reasoner optimization presets** (`NIM_OPTIMIZATION_PRESET=TTFT` or `TPOT`),
  which are experimental. Refer to
  [Configuration](configuration.md#optimization-presets).
- **Generator Hugging Face checkpoints**: `NIM_MODEL_PATH` accepts
  `hf://owner/repository[:revision]` for both runtimes.
- **Transfer control weights**: each control object accepts a relative
  `control_weight`. Refer to [Transfer](transfer.md#combine-multiple-controls).
- **AVI video input** for both runtimes.

### Behavior Changes

- **Transfer guidance**: with control and text guidance both active, the
  control term is no longer also multiplied by `guidance_scale`. The effective
  edge-family coefficient is 1.5 instead of 4.5, so combined-guidance outputs
  differ from 2.0.0 baselines. To approximate the 2.0.0 strength, multiply the
  previous `control_guidance` by `guidance_scale`, up to the 10.0 maximum.
  Refer to [Control Guidance Behavior](transfer.md#control-guidance-behavior).
- **Generator performance defaults**: SeaCache is on for every Generator variant
  except Edge-DROID. SageAttention 2.2 is the default on compute capability 8.0
  through 9.x without sequence parallelism. The channels-last variational
  autoencoder (VAE) decode path is the default. If you notice a quality issue in
  a specific workload, try `NIM_CACHE_BACKEND=none` or
  `NIM_VAE_FAST_PATH=lossless`. Refer to
  [Configuration](configuration.md#diffusion-cache-attention-and-vae-decode).
- **System-memory floors are enforced**: each profile carries its measured
  system-memory floor instead of a 16-GiB admission tag, and FP8 offload floors
  are lower. Refer to the [Support Matrix](support-matrix.md#generator-profiles).
- **Unified memory**: on NVIDIA DGX Spark and Jetson AGX Thor, 70% of the shared pool
  counts toward profile floors, and the Reasoner defaults to
  `NIM_GPU_MEMORY_UTILIZATION=0.70`. Super Generator profiles no longer fit
  these systems.
- **Reasoner headroom**: `NIM_GPU_MEMORY_HEADROOM_GIB` defaults to 4 GiB on
  discrete GPUs.
- **Validation**: `flow_shift` must be greater than 0.

### Renamed or Removed Variables

| 2.0.0 | 2.1.0 |
| --- | --- |
| `NIM_OFFLOAD_TEXT_GUARDRAIL` | `text_guard_offload` key in `NIM_TAGS_SELECTOR` (`resident` or `offloaded`) |
| `NIM_OFFLOAD_VIDEO_GUARDRAIL` | `video_guard_offload` key in `NIM_TAGS_SELECTOR` (`resident` or `offloaded`) |
| `NIM_UNIFIED_MEMORY_HOST_RESERVE_GIB` | Removed; unified-memory systems use a fixed 70% share |
| `NIM_QWEN3GUARD_MAX_GPU_MEMORY_GB` | Removed |

### Known Issues

- **Edge Reasoner streaming**: With `NIM_ENABLE_STREAMING=true`, the default
  `NIM_STREAMING_SPEC_METHOD=dflash` fails for Edge, which does not support
  DFlash. Set `NIM_STREAMING_SPEC_METHOD=none`. Refer to
  [Deploy Edge Models](deployment.md#deploy-edge-models).

## 2.0.0

This is the first release of the unified Cosmos3 NIM, published as
`nvcr.io/nim/nvidia/cosmos3:2.0.0`. One container image includes both runtime
choices:

- **Generator** serves image and video generation, Action, and Transfer through
  `POST /v1/infer`.
- **Reasoner** serves OpenAI-compatible image and video understanding through
  Chat Completions and the Responses API.

Refer to [Deployment](deployment.md) for launch instructions.
