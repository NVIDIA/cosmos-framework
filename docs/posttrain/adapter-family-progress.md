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
| HF loaded-base → initialization → FSDP → optimizer lifecycle | `HFModel.setup`, `build_hf_model`, SFT `build_trainer` | Implemented; CPU materialization and tied embeddings pass; Qwen3-0.6B passed all four methods on one/two GPUs with AC none/full |
| Text portable reload and merged model export | `checkpoint/adapters.py`, shared HF materializer | Implemented; real tiny HF serialization, strict reload and stock merged reload pass |
| Non-Qwen text templates in the actual SFT workflow | `data/packer/text_sft.py`, `test_data_packer_text_families.py`, `test_text_sft_adapters.py` | Implemented by extending the existing source-traced offset masker. Explicit ChatML, Llama3, Mistral and Gemma protocols; real Qwen2.5/SmolLM2/Mistral tokenizer preflight passes. Pretrained training remains pending; component math tests alone did not establish this path |
| Qwen3.5 language, merger, selected vision adapters | Existing ordered-vision forward plus shared adapter lifecycle | Implemented; LoRA, rsLoRA, DoRA, PiSSA each pass text/image/video/both CPU updates with activation checkpointing; pretrained GPU qualification pending |
| Native Qwen3-VL-MoE non-expert projections | Native loaded-base hook, family binding, `tests/reload_moe_adapters.py` | Implemented but awaiting actual grouped-mm training, resume and export qualification. Four small real MoE method fixtures pass native/HF target agreement, stock FP32 updates, exhaustive serialized tensor checks and reload with native expert layout. Fused expert tensors remain unsupported targets |
| GR00T | Existing `Gr00tN1d7TrainPolicy`, official Isaac revision `51d4c89` inspected | Implemented in `model/gr00t/{sft,sft_policy,adapter_export}.py`; 19 small actual Isaac/Qwen3-VL/DiT tests pass, including four methods, full AC, native stochastic resume, native-forward parity, real image packing and portable/merged reload. Real checkpoint GPU qualification is blocked on gated Cosmos-Reason2 config/processor metadata (HTTP 401). Category-specific tensors remain unsupported adapter targets |
| π0.5 | `model/pi05/sft.py`, `Pi05Model.training_condition`, `model/pi05/adapter_export.py`, `tests/test_pi05_adapters.py` | Implemented, awaiting installed real-model GPU qualification. All four methods pass actual small SigLIP/Gemma prefix/action updates, AC, portable reload and FP32 merged-file reload. Final-prefix query/output/MLP targets are refused because only its K/V reaches the action loss. Existing PPO frozen-prefix behavior is retained |
| Alpamayo | Existing `alpagym_clrl.py` and public NVlabs Alpamayo/AlpaGym inspected | Implemented in `model/alpamayo/{sft,policy,adapter_export}.py` with public pinned R1 source; no AlpaGym runtime needed. All four methods pass small real R1/Qwen3-VL flow updates, vision AC, portable/FP32 merged reload and native stochastic continuation. Real checkpoint GPU qualification pending |
| Physical-action SFT lifecycle | `config/sft.py::SFTConfig.task`, existing SFT loop and `flow_bc`, `roles/env/flow_sft.py` | Implemented. Shares trainer, checkpoint boundary, rank cursor and validation transaction; reports action-element counts separately from text tokens. Controlled continuation and validation-padding CPU tests pass |
| Model-owned base autocast with FP32 adapters | Native `LoraInjectedLinear.adapter_output`, `PosttrainAdapterLinear` | Implemented. Eight method/base-dtype fixtures pass FP32-input autocast tests; ordinary native LoRA mathematics reused |
| New installed revision and real training | Wheel byte attestation, Slurm qualification harness | Commit `3a9552b`, wheel SHA `fa6002a738b7298002f14706fa5df7d08cd4d78be0d351958a1bf10c80015f8d`, image SHA `4c7d3d3e8dc480ed7a8a7f3a5079a0f21b68619dbeb3837de658ef7f9fdd1d9a`; 223 installed CPU tests passed. Qwen3 job `37701630` completed all 16 four-method 1/2-GPU AC-none/full cases, including continuation and both reload types. Native Qwen3-VL job `37701632` also completed all 16 cases, with text, real image/video and WTS fixtures, differing modalities across ranks, all target updates and unchanged frozen base |
| Qwen3.5 public checkpoint MTP export | `checkpoint/tools/adapters.py`, `test_adapter_qwen3_5_export.py` | Job `37701635` passed training/resume/portable reload, then failed merge on unused source MTP tensors. Explicit bounded exclusion implemented and CPU-tested; 225 installed tests passed at `2f8d33f`; GPU rerun `37705883` queued. Failure preserved |

The dense HF path currently binds Qwen2, Qwen3, Llama, Mistral, Gemma and Gemma2.
The six small fixtures use actual Transformers implementations, three optimizer
updates per method, frozen-base checks, warm reload, and an FP32 merge oracle.
They do not establish task quality. The Qwen3.5 fixture additionally exercises
both hybrid language-block types and image/video/text gradient paths.
Additional Qwen2-VL/2.5-VL component fixtures use canonical materialized HF keys;
they do not qualify the public checkpoints' legacy namespaces or their media
processors. Those families are still refused by the production SFT packer.

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

The robotics runtime was built from public pinned Isaac-GR00T and Alpamayo sources
in Slurm job `37701725` (Python 3.12.13, Torch 2.10.0+cu128, Transformers 4.57.6).
Image SHA: `08be53d092569beec66c1ad2611404fd068058d237f1a326685d609a98b727ec`.
It is an environment build, not a training result. Public LIBERO demonstrations
and π0.5 weights are staged. Public KITTI raw drives 0001 and 0005 and the R1
processor dependency were staged by job `37701741`; driving preprocessing completed in job `37705800` with 15 training and 38 held-out
trajectory windows; GPU training still needs qualification. KITTI is a different camera
and driving domain from the gated PhysicalAI dataset, and will be reported as such.

π0.5 qualification binds Transformers 4.57.6: Transformers 5 changes SigLIP's
checkpoint namespace. Portable artifacts preserve the native mixed-precision
parameter map and tokenizer/normalization identities. Merged artifacts use
`cosmos-pi05-merged-fp32-v1` and `model.pi05.adapter_export.load_merged`; their
reference is the portable model promoted to FP32 after the original base rounding.
They are not advertised as PPO/Omni serving artifacts. The original-source loader
refuses these artifacts instead of silently rounding them back to BF16.

The physical-action stage passed **84 tests, 1 skip** in its Transformers 4.57.6
CPU environment, covering existing π0.5 loading/processing, the real small
SigLIP/Gemma model, all methods, action/vision gradients, file export and SFT
continuation. The shared HF/native/adapter/export/data regression suite passed
**246 tests** on Transformers 5.8.1. These counts overlap earlier suites and do not
claim installed GPU qualification. The rebuilt π0.5 image at `90b3931` passed
86 installed tests, 1 skip in job `37705762`; image SHA
`7221dc9e6db5024f2d6dd0d3768a08cfbd451cc6e2c70409a991be1c42e38249`.
It pins PyArrow 23.0.1 for the runtime’s NumPy 1.26.4; prior incompatible-image
attempts are preserved. LIBERO preparation `37706132` produced 41 training and
55 validation windows from separate complete episodes 0 and 1. This is a
correctness subset, not the complete LIBERO benchmark.

The Alpamayo/physical regression suite passed **50 tests, 1 skip** under
Transformers 4.57.6; the shared HF/native regression passed **225 tests** under
Transformers 5.8.1. Alpamayo tests restore native noise/time RNG, optimizer,
scheduler and cursor for the next controlled update, and prohibit repeating
PiSSA initialization on resume. These are CPU results; installed real-model
training and held-out trajectory metrics remain outstanding.


Current Slurm submissions (pending is not a training result): π0.5 matrix
`37706304`; Qwen3.5 corrected export matrix `37705883`; native Qwen3-VL full WTS
LoRA `37706319`, rsLoRA `37706324`, DoRA `37706326`, PiSSA `37706327`. Every WTS
job first gates its full-resolution/gradient-accumulation profile. Alpamayo
image `af407d4`, job `37706329`, passed **105 installed tests, 1 skip**;
image SHA `2198e81619a314702217cdf42d71b663b5bfc7ccb0df26ce80ecb6e99a2f0280`.
Alpamayo's actual KITTI matrix is queued as `37706970`. GR00T source `e207c5c`
passed installed tests in image job `37707039`; this does not remove the
gated backbone-metadata requirement. The first pending MoE payload
`37706615` was cancelled before execution: its generic merged checker attempted
single-GPU FP32 30B inference, which exceeds A10080 memory and the native expert
kernel's BF16 contract. Its inputs and cancellation evidence are preserved;
its replacement uses stock HF FP32 inference across two GPUs. Native BF16
portable reload remains a mandatory independent gate, every exported tensor is
verified against the original rounded base and the shared method-specific merge,
and stock BF16/native BF16 drift is reported separately from FP32 merge agreement.
The existing materializer owns expert transposition; no alternate expert training
kernel or adapter mathematics is introduced. Export releases its temporary FP32
state before materialization to avoid retaining another full 30B allocation.
The shared HF/native/export suite with these four new tests passed **229 tests**
on Transformers 5.8.1. Real MoE GPU qualification is still outstanding.

GR00T image `e207c5c` completed with **130 installed tests, 1 skip** and SHA
`92bf607dadf8d3d320c7ca93bd25deb62f473a184edf9de66d7f4d1a4a386bda`.
The reproducible x86 flow installer in `docker/posttrain/flow-environment` was
built from a fresh prefix in job `37707556`; all pinned distributions and native
source hashes matched and the same 130 tests passed (1 skip). Container publication
is separate from these installed-test results.

Public dense checkpoints staged by job `37707555`: Qwen2.5-0.5B-Instruct
`7ae557604adf67be50417f59c2c2f167def9a775`, SmolLM2-135M-Instruct (Llama
architecture) `12fd25f77366fa6b3b4b768ec3050bf629380bac`, and
Mistral-7B-Instruct-v0.3 `c170c708c41dac9275d15a8fff4eca08d52bab71`.
SmolLM2 results will not qualify Meta's Llama checkpoints. Gemma/Gemma2 have small
correctness fixtures; pretrained HF training qualification remains outstanding.
The extended shared regression suite passed **380 tests, 1 skip**, including
16 actual Llama/Mistral/Gemma/Gemma2 `LLMTrainer` cases with masked template
targets, all four methods, frozen-base checks and exact next-update restoration.
The source-tracing tests reject dropped/reordered messages and unsupported text
templates. This remains separate from the queued pretrained GPU matrices.
