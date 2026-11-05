# Versioned rounded-weight BF16 adapter execution

Status: implementation and qualification of this new-artifact contract were
approved on 2026-11-05. CPU correctness checks pass; real-model GPU qualification
is pending. This is not a repair or relabeling of any legacy BF16 artifact.

`merged_weight_bf16_v1` is selected before initialization, stored in portable
format `cosmos-posttrain-adapter-v4`, and required by exact resume, warm start and
merged identity. The existing HF SFT builder binds stock Transformers 5.8.1 /
PyTorch 2.10.0, SDPA, dense Qwen3-VL and Qwen3.5. The runnable recipe is
`cosmos_framework/posttrain/configs/sft_merged_weight_bf16.py`; environment
variables `SFT_MODEL_PATH`, `SFT_DATA_ROOT` and
`SFT_MERGED_VARIANT=vision_pissa|language_pissa` specify staged inputs and
independent component settings. Base/compute storage must be BF16.

Each adapted Linear constructs its differentiable dense effective weight using
FP64 low-rank accumulation and DoRA direction normalization, rounds once to
FP32, then casts to the input model's declared weight precision. LoRA reuses the
existing native helper/scaling. PiSSA keeps the untouched original base and
initial decomposition, subtracting the initial low-rank product from the current
product. Only FP32 factors, DoRA magnitudes and selected extra modules are
optimized. Frozen base ownership and parameter identities are preserved.
Dense construction is transient but can substantially increase training memory
and time compared with factorized execution; measurements must accompany support.

The selected extra norm executes the exact stock normalization with an effective
weight cast to the input dtype, retaining its optimized FP32 master. Dense
Qwen3-VL uses stock image/video visual batches with ordered dummy modality
passes to preserve distributed collectives and gradients. Qwen3.5 uses its
existing ordered stock forward. Zero adapter dropout is required for this mode;
legacy factorized dropout support is unchanged.

BF16 portable and stock merged models must pass the existing strict merge gates
(`atol=5e-4`, `rtol=2e-4`, mean absolute logit error ≤5e-5, all controlled greedy
positions equal). A separate FP32-promoted portable reference and FP32 export
must pass the same gates. Whole-model modality batching, caches and generation
are part of the contract; equal weights alone are insufficient. Cross-precision
drift is reported, not required to match both incompatible legacy references.
All legacy failed references and acceptance criteria retain their status.

Cosmos reasoner loading validates the execution tag, marker, precision and
pinned libraries, and selects stock SDPA arithmetic. Exported models contain no
adapter branches and can use the same pinned stock HF class. Other libraries,
attention kernels, quantizers and serving backends are unqualified; they cannot
be claimed equivalent merely because they read the weights.

| Requirement | Code | Current validation |
| --- | --- | --- |
| Methods, independent component selection | `config/adapters.py`, `model/adapters.py`, recipe | CPU all-method/rs gradients and unchanged base; real-model pending |
| Norm ownership / rounded computation | `model/adapter_bf16_execution.py` | CPU stock-value and parameter-identity checks |
| Ordered real/dummy modalities | `model/qwen3_vl_hf.py`, existing Qwen3.5 path | Text/image/video/both with full activation checkpointing, CPU value and gradient equality |
| Versioned resume/warm start | `checkpoint/adapters.py` | v4 lifecycle and incompatible profile rejection; GPU continuation pending |
| Portable / BF16 and FP32 stock exports | shared merge helper, streaming exporter, reasoner loader | CPU exact tensors and whole-forward reload; real-model pending |
| Training / deployment resources and WTS quality | qualification tools | Pending; no performance or task-quality claim |

Historical qualification remains 12 named checkpoints, 48 fixed profiles and
47 fully passing legacy profiles. New-mode executions are separate. The
576-row scene-disjoint WTS benchmark is reused, not an untouched final test;
correctness fixtures and physical-model runs do not establish task performance.
