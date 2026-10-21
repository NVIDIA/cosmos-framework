<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Support Matrix

Match a model to precision, GPU compute capability, GPU count, per-device video
memory (VRAM), and system memory. Memory requirements use binary GiB: VRAM
floors are per device, and system-memory floors are per host. The GPU inventory
tables list nominal capacities in GB.

## Choose a Model

Choose the model. The NIM selects a compatible profile and precision. Refer to
[Precision Selection](deployment.md#precision-selection).

| Runtime | Model | Supported Use |
| --- | --- | --- |
| Generator | `nano` | General-purpose Generator tasks |
| Generator | `edge-droid` | DROID policy with action-only output; BF16 only; **experimental** |
| Generator | `nano-droid` | DROID policy with action-only output; BF16 only |
| Generator | `super` | General-purpose Generator tasks |
| Generator | `super-t2i` | Full-step T2I only |
| Generator | `super-t2i-4step` | Four-step T2I only |
| Generator | `super-i2v` | Full-step I2V only |
| Generator | `super-i2v-4step` | Four-step I2V only |
| Reasoner | `edge` | Image/video reasoning; BF16 or FP8 only |
| Reasoner | `nano` | Image/video reasoning |
| Reasoner | `super` | Image/video reasoning |

A Generator specialist rejects requests for other tasks. Edge has no
general-purpose Generator.

> **Experimental:** Edge-DROID policy is experimental in this release because
> NVIDIA has not validated it. Validate its accuracy, latency, and safety
> behavior for your workload before relying on it.

After startup, follow
[Verify the Selection](deployment.md#verify-the-selection).

### Practical Model Recommendation

Start from the recommendation for your GPU class:

- H200- and B200-class discrete GPUs: Use Super when the workload needs it,
  with `NIM_PERF_PROFILE=latency` for Generator. Leave precision, offload,
  tags, and profile ID unset.
- H100, RTX PRO 6000 Blackwell, and lower-throughput discrete GPUs: Default to
  Nano, especially for video. Use Super only for an explicit model requirement.
- Unified-memory systems (DGX Spark, Jetson AGX Thor): Default to Nano for both
  runtimes. Some Super Reasoner rows fit, but turnaround is impractical.
- Smallest GPUs: Edge Reasoner and Edge-DROID have the lowest floors. Refer to
  [Edge Reasoner Verified GPUs](#edge-reasoner-verified-gpus).

## GPU Architecture and Topology

Compatibility uses compute capability, per-device VRAM, and system RAM, not a
GPU SKU allowlist:

| Runtime | Precision | Minimum Compute Capability |
| --- | --- | ---: |
| Generator | BF16 | 8.0 |
| Generator | FP8 | 8.9 |
| Reasoner | BF16 | 8.0 |
| Reasoner | FP8 | 8.9 |
| Reasoner | NVFP4 | 10.0 |

Use homogeneous GPUs. Mixed-GPU configurations are not supported, and
Reasoner requires the same compute capability on every GPU. Do not add VRAM
across devices. Each GPU in the selected layout must meet the floor in total
and in free memory at startup. On a discrete GPU, Reasoner measures free
memory after a 4-GiB runtime reserve (`NIM_GPU_MEMORY_HEADROOM_GIB`).
On unified-memory systems, Generator uses only rows with
**Model offload = None** and **Guardrails = Resident**.

### Unified-Memory Thresholds

On unified-memory systems, **effective shared memory** is 70% of the
device-reported total. The rest stays with the host. The same share is the
default Reasoner `NIM_GPU_MEMORY_UTILIZATION`.

| Runtime/Profile | Effective Shared-Memory Floor | Unified-Memory Policy |
| --- | ---: | --- |
| Generator `edge-droid` BF16 | 15 GiB | Resident model and guardrails |
| Generator `nano` BF16; `nano-droid` BF16 | 58 GiB | Resident model and guardrails |
| Generator `nano` FP8 | 44 GiB | Resident model and guardrails |
| Generator `super` BF16 | 150 GiB | Resident model and guardrails |
| Generator `super` FP8 | 93 GiB | Resident model and guardrails |
| Reasoner `edge` BF16 or FP8 | 12 GiB/device | No CPU/model offload |
| Reasoner `nano` BF16, FP8, or NVFP4 | 23.1 GiB/device | No CPU/model offload |
| Reasoner `super` BF16, TP1 | 135 GiB/device | No CPU/model offload |
| Reasoner `super` FP8, TP1 | 67 GiB/device | No CPU/model offload |
| Reasoner `super` NVFP4, TP1 | 73 GiB/device | No CPU/model offload |

Floors are compared with 70% of the binary-GiB total reported by the running
system. Current free shared memory is checked separately at startup.

## Tested GPU Inventory

This section lists devices in the release hardware inventory, grouped by
architecture and sorted by nominal memory. Inventory membership does not mean
every configuration is tested on that device, and devices outside it can be
compatible when they meet the floors.

### Unified-Memory Systems

Capacity is a shared host/device pool, not separate VRAM:

| Device | Nominal Shared Memory |
| --- | ---: |
| Jetson AGX Thor T5000 | 123 GiB |
| DGX Spark (GB10) | 128 GB (about 119 GiB) |

The effective share is roughly 83 to 86 GiB. It fits the resident Generator
Edge-DROID and Nano rows but no Super Generator row, and the Reasoner Edge,
Nano, and Super FP8/NVFP4 TP1 rows. Reasoner Super BF16 TP1 (135 GiB) exceeds
both systems' total capacity, so freeing cache cannot make it fit.

### NVIDIA Blackwell Ultra

| GPU | Nominal GPU Memory |
| --- | ---: |
| GB300 Workstation | 252 GB HBM3e |
| B300 SXM6 | 288 GB HBM3e |
| GB300 | 288 GB HBM3e |

### NVIDIA Blackwell

| GPU | Nominal GPU Memory |
| --- | ---: |
| GeForce RTX 5090 | 32 GB GDDR7 |
| RTX PRO 4500 Blackwell Server Edition | 32 GB GDDR7 |
| RTX PRO 6000 Blackwell Server Edition | 96 GB GDDR7 |
| RTX PRO 6000 Blackwell Workstation Edition | 96 GB GDDR7 |
| B200 | 192 GB HBM3e |
| GB200 | 192 GB HBM3e |

### NVIDIA Hopper and Grace Hopper

| GPU | Nominal GPU Memory |
| --- | ---: |
| H100 80 GB HBM3 | 80 GB HBM3 |
| H100 PCIe | 80 GB HBM3 |
| H100 NVL | 94 GB HBM3 |
| GH200 480 GB | 96 GB HBM3 |
| H20 | 96 GB HBM3 |
| H200 | 141 GB HBM3e |
| H200 NVL | 141 GB HBM3e |
| GH200 144 GB HBM3e | 144 GB HBM3e |

### NVIDIA Ada Lovelace

| GPU | Nominal GPU Memory |
| --- | ---: |
| L20 | 48 GB GDDR6 |
| L40S | 48 GB GDDR6 |
| RTX 6000 Ada Generation | 48 GB GDDR6 |

### NVIDIA Ampere

| GPU | Nominal GPU Memory |
| --- | ---: |
| A100 SXM4 40 GB | 40 GB HBM2e |
| A100 SXM4 80 GB | 80 GB HBM2e |

Nominal memory only orders this inventory. For GH200 480 GB, 480 GB is CPU
memory. The table shows HBM GPU memory.

## Generator Profiles

The `super` rows in this table also apply to `super-t2i`,
`super-t2i-4step`, `super-i2v`, and `super-i2v-4step`. Transfer thresholds apply
only to the general-purpose `nano` and `super` variants.

| Variant Group | Precision | Model Offload | Guardrails During Diffusion | GPUs | Generation Minimum VRAM/Device | System-Memory Floor | Transfer Minimum VRAM/Device |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `edge-droid` (experimental) | BF16 | None | Resident | 1, 2, 4, 8 | 15 GiB | 16 GiB | Not supported |
| `nano`, `nano-droid` | BF16 | None | Resident | 1, 2, 4, 8 | 58 GiB | 40 GiB | 64 GiB |
| `nano` | BF16 | Layer | Offloaded | 1 | 31 GiB | 64 GiB | 35 GiB |
| `nano` | FP8 | None | Resident | 1, 2, 4, 8 | 44 GiB | 35 GiB | 50 GiB |
| `nano` | FP8 | Model | Resident | 1 | 38 GiB | 36 GiB | 44 GiB |
| `nano` | FP8 | Layer | Resident | 1 | 32 GiB | 36 GiB | 38 GiB |
| `nano` | FP8 | Layer | Offloaded | 1 | 31 GiB | 64 GiB | 35 GiB |
| `super` | BF16 | None | Resident | 1, 2, 4, 8 | 150 GiB | 112 GiB | 160 GiB |
| `super` | BF16 | Model | Resident | 1 | 93 GiB | 150 GiB | 99 GiB |
| `super` | BF16 | Layer | Resident | 1 | 42 GiB | 150 GiB | 50 GiB |
| `super` | FP8 | None | Resident | 1, 2, 4, 8 | 93 GiB | 76 GiB | 103 GiB |
| `super` | FP8 | Model | Resident | 1 | 64 GiB | 80 GiB | 76 GiB |
| `super` | FP8 | Layer | Resident | 1 | 38 GiB | 80 GiB | 50 GiB |

Specialist variants (`nano-droid`, `edge-droid`, and `super-*`) use only the
resident-guardrail rows of their base model. Multi-GPU rows use every GPU. A
`latency` profile splits each request with guidance parallelism 2 and sequence
parallelism `n_gpus / 2`. A `throughput` profile runs one engine replica per
GPU, and there is no one-GPU `throughput` profile. To limit the GPU count, refer
to
[GPU Selection](configuration.md#gpu-selection).

The 31-GiB Nano layer-offload rows target 32-GB client GPUs such as the
GeForce RTX 5090, which does not meet their 35-GiB Transfer minimum.

> **System memory:** Profile selection enforces each row's system-memory floor
> (Generator and Reasoner) against host memory or a smaller container memory
> limit. Treat the floors as minimums. Media processing, concurrency, and other
> host workloads can need more. Unified-memory systems have no separate host RAM
> requirement. Size them with
> [Unified-Memory Thresholds](#unified-memory-thresholds).

Leave offload and guardrail residency on automatic selection unless a
configuration is validated for the deployment.

## Reasoner Profiles

Reasoner does not use Generator latency/throughput or model-offload selectors:

| Model | Precision | GPUs | Tensor Parallelism | Minimum VRAM/Device | System-Memory Floor | Minimum Compute Capability |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `edge` | BF16 | 1 | 1 | 12 GiB | 8 GiB | 8.0 |
| `edge` | FP8 | 1 | 1 | 12 GiB | 8 GiB | 8.9 |
| `nano` | BF16 | 1 | 1 | 23.1 GiB | 24 GiB | 8.0 |
| `nano` | FP8 | 1 | 1 | 23.1 GiB | 17 GiB | 8.9 |
| `nano` | NVFP4 | 1 | 1 | 23.1 GiB | 17 GiB | 10.0 |
| `super` | BF16 | 1 | 1 | 135 GiB | 72 GiB | 8.0 |
| `super` | BF16 | 2 | 2 | 73 GiB | 72 GiB | 8.0 |
| `super` | FP8 | 1 | 1 | 67 GiB | 43 GiB | 8.9 |
| `super` | NVFP4 | 1 | 1 | 73 GiB | 33 GiB | 10.0 |

System memory follows the note under
[Generator Profiles](#generator-profiles). When BF16 is selected, Super prefers
TP1 and can fall back to TP2 when TP1 lacks free memory. To require TP2,
set `NIM_TAGS_SELECTOR='n_gpus=2,nim_tp=2'`.

## Edge Reasoner Verified GPUs

Edge Reasoner was verified on these GPUs. Edge-DROID policy was not validated
and is [experimental](#choose-a-model). Each deployment must still meet the
floors above and the exact image manifest.

| Architecture | GPUs |
| --- | --- |
| NVIDIA Blackwell Ultra | B300 SXM6, GB300 |
| NVIDIA Blackwell | B200, GB200, DGX Spark (GB10), RTX PRO 6000 Blackwell Server Edition, RTX PRO 6000 Blackwell Workstation Edition, GeForce RTX 5090 |
| NVIDIA Hopper and Grace Hopper | H100 80 GB HBM3, H100 PCIe, H100 NVL, H200, H200 NVL, H20, GH200 480 GB, GH200 144 GB HBM3e |
| NVIDIA Ada Lovelace | L40S, RTX 6000 Ada Generation |
| NVIDIA Ampere | A100 SXM4 40 GB, A100 SXM4 80 GB, A100 PCIe 40 GB, A100 80 GB PCIe |

NVIDIA Ampere GPUs (compute capability 8.0) support only the BF16 Edge profiles.

## Transfer Headroom

Transfer needs more peak memory than ordinary generation, so startup checks it
separately. A deployment can serve generation while rejecting Transfer. Use a
larger GPU or a lower-memory configuration. `NIM_ALLOW_UNSAFE_TRANSFER=1`
bypasses the check for diagnosis only and can cause out-of-memory failures.

## Media and Codecs

| Direction | Media | Documented Formats/Codecs and Limits |
| --- | --- | --- |
| Input | Images | Use the included JPEG, PNG, and WebP fixtures; verify other formats against the deployed image |
| Input | Videos | MP4/MOV, Matroska/WebM, FLV, and AVI containers; H.264, H.265/HEVC, and MPEG-4 Part 2 decode through NVDEC, VP8, VP9, and AV1 use software decode; other codecs return HTTP 422 |
| Output | Generator image | JPEG |
| Output | Generator video | VP9 in MP4 |

Inputs are base64 or data URLs. HTTP(S) input works only when enabled and
reachable from the container. Refer to [API Reference](api-reference.md) for the
media rules.
