# Adapter family extension: implementation and evidence

Work branch: `feat/posttrain-family-adapters`. This work preserves the completed
Cosmos3 adapter experiments and the user's closure of PR #342. Starting revision
was `53b2382e00e8487270a2d09fcf2a13d9597340ac`; main at
`86e1ee52f276c11d6cc66dfd17da2795dfbe937b` was inspected and merged. Its changes
were confined to skills, documentation and CI.

The scope is the families integrated in this checkout, with explicit qualification
per architecture, checkpoint, component and execution profile. A component binding
does not qualify every checkpoint, custom operator, algorithm or distributed mode.
Four-bit training is separate. Existing Cosmos3 results remain in
[adapters-progress.md](adapters-progress.md).

| Requirement | Code / test | Current evidence |
| --- | --- | --- |
| Shared LoRA/rsLoRA/DoRA/PiSSA math | `model/adapters.py::PosttrainAdapterLinear` | Existing implementation reused; no new adapter math |
| Architecture-owned component/layer boundaries | `model/adapter_families.py`, `resolve_targets` | Implemented; six actual HF text architectures pass small CPU update/reload/merge tests |
| Independent settings, aliases, optimizer ownership | Existing resolver / `assert_trainability` | Existing native tests pass; new family selector failures tested |
| HF loaded-base → initialization → FSDP → optimizer lifecycle | `HFModel.setup`, `build_hf_model`, SFT `build_trainer` | Implemented; CPU materialization, tied embeddings and activation-checkpointed updates pass; multi-GPU qualification pending |
| Text portable reload and merged model export | `checkpoint/adapters.py`, shared HF materializer | Implemented; real tiny HF serialization, strict reload and stock merged reload pass |
| Qwen3.5 language, merger, selected vision adapters | Existing ordered-vision forward plus shared adapter lifecycle | Implemented; LoRA, rsLoRA, DoRA, PiSSA each pass text/image/video/both CPU updates with activation checkpointing; pretrained GPU qualification pending |
| Native Qwen3-VL-MoE non-expert projections | Native loaded-base hook and family binding | Implemented but awaiting actual grouped-mm training, resume and export qualification; fused expert tensors remain unsupported targets |
| GR00T | Existing `Gr00tN1d7TrainPolicy`, official Isaac revision `51d4c89` inspected | Not implemented yet. Category-specific `[embodiment,in,out]` projectors are not ordinary Linear. Official loader needs gated Cosmos-Reason2 processor/backbone metadata; unauthenticated access returned HTTP 401 |
| π0.5 | Existing `Pi05TrainPolicy`, `Pi05Model.prepare_condition` inspected | Not implemented yet. Existing PPO uses a frozen prefix under `no_grad`; vision/language adaptation requires a fresh differentiable prefix and model-owned offline objective |
| Alpamayo | Existing `alpagym_clrl.py` and public NVlabs Alpamayo/AlpaGym inspected | Not implemented yet. Public AlpaGym `972d160` uses Cosmos-RL and does not contain the `alpagym_runtime.posttrain` API referenced here. A runnable Cosmos training binding and driving data are still required |
| New installed revision and real training | Existing wheel byte attestation, Slurm qualification harness | Pending. Resource staging job `37700939` is CPU preparation only, not training evidence |

The dense HF path currently binds Qwen2, Qwen3, Llama, Mistral, Gemma and Gemma2.
The six small fixtures use actual Transformers implementations, three optimizer
updates per method, frozen-base checks, warm reload, and an FP32 merge oracle.
They do not establish task quality. The Qwen3.5 fixture additionally exercises
both hybrid language-block types and image/video/text gradient paths.

`sft_family_adapters.py` selects the native builder for Qwen3-VL dense/MoE and HF
for text families and Qwen3.5. This is deliberate: the old generic-HF Qwen3-VL
vision patch requires Transformers 4.57, while native Qwen3-VL and Qwen3.5 SFT
are supported in the pinned Transformers 5.8.1 runtime. Export uses stock HF
implementations as an independent oracle. The recipe uses actual module names,
validates every selector, and chooses the checkpoint's final vision block.

Single-file HF bases are now accepted without modifying their source directory.
The shared materializer still publishes an indexed, verified output. Text models
require tokenizer identity but no image processor. Its conversion version is
incremented to `/6`, so old materialization markers cannot satisfy the new contract.

Validation performed so far: the pre-existing adapter/native/export/HF lifecycle
suite plus the initial new text cases passed **232 tests, 11 skips**. The expanded
family suite passed **30 tests**, including four Qwen3.5 method cases. Counts
overlap; do not add them. GPU results must carry the new source commit, wheel and
container hashes and installed-package byte verification before being included here.

Pinned resources requested for qualification: Qwen3-0.6B, Qwen3-VL-2B-Instruct,
Qwen3.5-0.8B, Qwen3-VL-30B-A3B-Instruct, GR00T-N1.7 LIBERO-Spatial,
RLinf-Pi05-LIBERO-SFT and Alpamayo-R1-10B. GSM8K, existing WTS and LIBERO provide
task-specific inputs. Driving data must carry actual camera/trajectory alignment;
WTS alone is not a driving-model qualification.
