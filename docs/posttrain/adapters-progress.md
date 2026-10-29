# Posttrain adapter implementation qualification

Implementation branch: `feat/posttrain-vlm-adapters`, based on main
`7738e9e7969e30160163b9d7941782fefd6eaa5c`. The audited baseline was
`1a2f03ab6111b457633827736f94f67d55bd5202`; intervening main changes affect
quantization/docs, not the SFT builder or native adapters. Existing LoRA/rsLoRA
and pure LoRA merge mathematics are reused from `utils/generator/lora.py`.
Previous WTS/AutoML results used unchanged framework code and do not qualify this implementation.

| Requirement | Code / tests | Status |
|---|---|---|
| Component selectors, independent settings, aliases/ties | `posttrain/config/adapters.py`, `posttrain/model/adapters.py`, `tests/test_adapters.py` | Implemented and CPU validated (selectors, independent settings, aliases, counts and optimizer membership) |
| LoRA / rsLoRA | Native `LoraInjectedLinear` reused; posttrain builder integration | Implemented and CPU validated; awaiting installed-wheel GPU qualification |
| DoRA magnitude, optimization and pure merge | `PosttrainAdapterLinear`, `optimizer_groups` | Implemented and CPU validated, including PEFT reference gradients/reload/merge; awaiting GPU qualification |
| PiSSA residual contract and saved initial factors | `PosttrainAdapterLinear.initial_A/initial_B`; portable format v1 | Implemented and CPU validated (principal components, immutable initial factors, strict recovery and merged HF reload); awaiting GPU qualification |
| Fresh, warm start and exact resume lifecycle | `Qwen3VLBuildHooks.after_load_before_parallelize`, SFT `build_trainer` | Implemented and CPU validated: bitwise controlled continuation through trainer and full SFT recipe |
| Portable export / strict reload | `posttrain/checkpoint/adapters.py` | Implemented and CPU validated in fresh processes |
| Merged HF export | Existing HF materializer reused; bundled processor and single-shard handling extended | Implemented and CPU validated, including fresh-process HF reload |
| Vision gradients, media and activation checkpointing | Native tiny Qwen3-VL plus real PNG/MP4 through SFT packer, with/without AC | Implemented and CPU validated; real-model GPU pending |
| FSDP2 and mixed modality ranks | FP32 adapter storage/accumulation policy added | Awaiting multi-step GPU tests; no qualification claim yet |
| Wheel/container revision provenance | Sealed wheel and byte-by-byte installed-package attestation | Implemented and validated: all 1,837 package files match; 44 tests pass from the installed wheel outside the checkout |
| Full WTS epoch and held-out evaluation | Existing experiments preserved | Pending feature qualification; no new HPO |
| Recipes, documentation and PR | Runnable native adapter recipe, WTS converter, adapter guide; [draft PR #342](https://github.com/NVIDIA/cosmos-framework/pull/342) | Implemented; GPU evidence pending |

Initial execution profile: dense native Qwen3-VL / staged Cosmos3-Nano-VLM,
BF16 frozen base, FP32 adapters, AdamW, eager/SDPA, non-reentrant full activation
checkpointing, single GPU then pure FSDP2. Generic HF, native MoE, HSDP, tensor/
expert/context parallelism, compilation, quantization and adapter stacking are
excluded until separately qualified. WTS has 5,555 training rows; the full epoch
will use one GPU so the existing exact DP cursor needs no duplicated/dropped rows.

CPU validation: 265 tests passed, 1 optional test skipped in the focused regression
suite; 2 additional DoRA-reference/PiSSA-principal-component tests passed. Environment:
Python 3.12.3, torch 2.10.0+cpu, Transformers 5.8.1, TorchCodec 0.10.0,
safetensors 0.8.0; PEFT 0.21.2 is a test reference only, not the implementation.
The SFT test root path was corrected after the posttrain directory migration.

Additional correctness coverage checks separately trained FP32 normalization with
BF16 activations, and native/portable/HF preservation of declared embedding ties.
The latter exposed two compatibility defects: the native Transformers 4 tie-key
list needed the Transformers 5 mapping form, and HF's `assign=True` overlay needed
to restore tying before serialization. The converter format advances to version 5.
The installed-wheel check runs outside the checkout: `.gitignore` previously
caused Hatch to omit the tracked `posttrain/roles/env` runtime package.

Installed-package gate (Slurm `37627771`, completed): code commit
`3208d7b7faca056fb6d418faa8d780f6d0765325`, wheel SHA-256
`673b99939856bccfd00299109b9c14e24afdb308468f81b7684d813ed9f0c9af`,
container SHA-256 `11aac37e02309f52695e7b7b9e54d6eb811be98b5008619422b19d9dcf06a340`.
Python 3.13.14, torch 2.10.0+cu128, Transformers 5.8.1, safetensors 0.8.0,
CPU TorchCodec 0.10.0; 44 passed in 76.46 seconds. The initial resource profile
uses the existing WTS CUDA 12.8 runtime with the SFT Transformers pin, rather than
claiming qualification of the separate torch 2.11 / TorchCodec 0.11 environment.
GPU matrix job `37627801` reached loaded real-model multimodal evaluation, then
failed because the qualification harness supplied an empty validation trigger.
This was before the first update. The harness now uses the existing BEFORE_TRAINING
and FINAL reasons and has a complete tiny-model harness regression test; GPU rerun
is pending. No GPU training or full-epoch qualification claim is made.

The corrected installed-wheel gate passed 45 tests (job `37628309`, commit
`15800a91e240c359129f8f829688f1d80ed467f6`). GPU job `37628322` passed the
single-GPU LoRA update, frozen-base and controlled-resume assertions before
portable export exposed Lustre's unsupported `RENAME_NOREPLACE`. Portable
publication now falls back to exclusive directory reservation and a commit-last
manifest; interruption and competing-writer tests cover that contract. Real
filesystem and GPU requalification of this fix are pending. Other method/world
cases continue independently, with their failure logs preserved.
