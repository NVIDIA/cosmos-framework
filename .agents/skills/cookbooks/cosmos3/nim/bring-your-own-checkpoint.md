<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Bring Your Own Checkpoint

Use this page to replace the selected Generator or Reasoner checkpoint while
preserving the Cosmos3 NIM server, profile selection, and runtime contract.

## Supported Boundary

`NIM_MODEL_PATH` is the shared checkpoint-source variable:

| Runtime | Accepted Source | What the NIM Still Provides |
| --- | --- | --- |
| Generator | Absolute local directory or `hf://owner/repository[:revision]` | Server, profile, and Generator guardrail artifacts |
| Reasoner | Absolute local directory or `hf://owner/repository[:revision]` | Server, selected runtime layout, and profile compatibility policy |
| Nano or Super Reasoner DFlash draft | Absolute local directory only | Primary Reasoner checkpoint, server, and profile policy |

## Generator Checkpoint

### Expected Layout

The current Generator path expects this structural shape:

```text
/byoc/cosmos3/
├── transformer/
│   ├── config.json
│   └── <weight shards>
├── vae/
├── scheduler/
└── model_index.json
```

The NIM infers Edge, Nano, or Super and the precision from
`transformer/config.json` and checks them against the selected profile. Set
`NIM_MODEL_VARIANT` to select a specialist Generator contract. The only Edge
Generator contract is the BF16 `edge-droid` policy.

### Launch

Add these options to the command in
[Launch Generator](deployment.md#launch-generator), with `BYOC_CHECKPOINT` set
to the host checkpoint directory:

```text
-e NIM_MODEL_PATH=/byoc/cosmos3 \
-v "$BYOC_CHECKPOINT:/byoc/cosmos3:ro"
```

Choose a variant, precision, and objective that match the checkpoint. The NIM
still downloads Generator guardrail artifacts, so the cache must be writable,
the NIM can still need NVIDIA NGC access, and `NIM_DISABLE_MODEL_DOWNLOAD=1` is rejected.

### Hugging Face Source

The Generator also accepts `hf://owner/repository[:revision]`, downloaded into
the NIM cache. The repository root must contain the layout above, and the
inferred variant and precision must match the selected profile:

```text
-e NIM_MODEL_PATH='hf://owner/repository:revision' \
-e HF_TOKEN \
-e NIM_MODEL_VARIANT=nano \
-e NIM_PRECISION=fp8
```

Omit `:revision` to use `main`. Pin a commit for reproducible deployments. Pass
`HF_TOKEN` only for a private repository. Keep `NGC_API_KEY` and a writable
cache for the guardrail artifacts.

## Reasoner Checkpoint

### Local Directory

A local Reasoner checkpoint must use an absolute in-container path and contain
at least the following:

- A supported Cosmos3 Reasoner or Cosmos3 Omni `config.json`
- Safetensors weights or a valid safetensors index and all referenced shards
- Tokenizer files
- Processor or preprocessor configuration

The NIM infers `edge`, `nano`, or `super`, the precision, and the Reasoner or
Omni layout, and then selects a compatible profile. The precision is BF16 or FP8
for Edge, and Nano and Super can also use NVFP4. An explicit
`NIM_MODEL_VARIANT`, `NIM_PRECISION`, or `NIM_MODEL_PROFILE` must agree with the
checkpoint. Mount it read-only and select the Reasoner:

```bash
-e NIM_MODEL_TYPE=reasoner \
-e NIM_MODEL_PATH=/byoc/cosmos3-reasoner \
-v "$BYOC_CHECKPOINT:/byoc/cosmos3-reasoner:ro"
```

With a local checkpoint, the NIM sets `NIM_DISABLE_MODEL_DOWNLOAD=1`
automatically, so it does not download profile artifacts. With `NIM_MODEL_PATH`, the NIM does not download the
profile's DFlash draft, which Nano and Super Reasoner enable by default. Mount a
matching Nano or Super draft through
`NIM_DFLASH_MODEL_PATH` (absolute local path, no `hf://`), or set
`NIM_USE_DFLASH=0` to run target-only. Edge has no DFlash. It defaults to
`NIM_USE_DFLASH=0` and rejects `1`. The draft needs only:

```text
/byoc/cosmos3-dflash/
├── config.json
└── model.safetensors
```

Add these options to the Reasoner `docker run` command, with
`DFLASH_CHECKPOINT` set to the host draft directory:

```text
-e NIM_USE_DFLASH=1 \
-e NIM_DFLASH_MODEL_PATH=/byoc/cosmos3-dflash \
-v "$DFLASH_CHECKPOINT:/byoc/cosmos3-dflash:ro"
```

To train your own matching Cosmos3-Nano DFlash draft on a custom dataset, or
after fine-tuning the Nano Reasoner target, follow the [NVIDIA Model Optimizer DFlash
training recipe](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/speculative_decoding/recipes/train_dflash_cosmos3_nano.ipynb).

### Hugging Face Source

The Reasoner also accepts:

```bash
-e NIM_MODEL_TYPE=reasoner \
-e NIM_MODEL_PATH='hf://owner/repository:revision' \
-e HF_TOKEN
```

Omit `:revision` to use `main`. Pass `HF_TOKEN` only when the repository
requires it, never in the URI or image. The snapshot is stored in the writable
NIM cache.

An `hf://` source cannot be combined with `NIM_DISABLE_MODEL_DOWNLOAD=1`. For
offline use, pre-download the checkpoint and use an absolute local path. The
DFlash draft rule above still applies.

## Verify the Active Checkpoint

After [readiness](deployment.md#wait-for-readiness), inspect metadata:

```bash
curl -fsS http://localhost:8000/v1/metadata | python3 -m json.tool
```

The `checkpoint` field reports `default` for bundled artifacts or identifies the
configured source. Generator metadata also reports `model_variant`. Confirm the
selected profile, then compare a representative request with the checkpoint's
validation baseline.

## Common Failures

| Symptom | Likely Cause | Action |
| --- | --- | --- |
| Local path does not exist | Host path was not mounted at the exact `NIM_MODEL_PATH` value | Align the read-only bind destination and environment path |
| Permission denied | Container cannot traverse or read the mount | Fix host ownership or ACLs while keeping the checkpoint mount read-only |
| Required file missing | Checkpoint layout, tokenizer, processor, index, or weight shards are incomplete | Compare with the required layout and export again |
| Model variant or precision mismatch | Explicit selector/profile disagrees with inferred checkpoint properties | Select a compatible configuration or use a matching checkpoint |
| Generator rejects disabled downloads | Generator still needs NIM-provided guardrails | Remove `NIM_DISABLE_MODEL_DOWNLOAD=1` and provide NGC/cache access |
| Hugging Face source rejected offline | `hf://` requires download but downloads are disabled | Use an absolute pre-downloaded local path |
| Hugging Face authorization fails | Token, repository ID, revision, network, or cache is invalid | Check `HF_TOKEN`, the URI, connectivity, and a writable cache without logging the token |
| DFlash path is rejected | The draft path is relative, uses `hf://`, is not mounted, or lacks one of its two required files | Use an absolute local mount containing `config.json` and `model.safetensors` |
| DFlash configuration is rejected | DFlash is disabled, DFlash is enabled on Edge (including through `NIM_OPTIMIZATION_PRESET=TPOT`), the draft does not match the Nano/Super target, or `NIM_DFLASH_CONFIG` sets reserved `method`/`model` keys | For Edge, leave `NIM_USE_DFLASH` unset or `0`; otherwise use the selected Reasoner's matching draft, keep DFlash enabled, and remove reserved keys; refer to [Speculative Decoding](configuration.md#speculative-decoding) |
| Metadata shows `default` | Override was omitted, rejected, or applied to another container | Inspect launch environment, mounts, startup logs, and `/v1/metadata` |

For broader startup, cache, GPU, and readiness diagnosis, refer to
[Operations](operations.md#byoc).
