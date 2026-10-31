# Adapter family extension: implementation and evidence

Work branch: `feat/posttrain-family-adapters`. This extends the original Cosmos3
implementation; [PR #342](https://github.com/NVIDIA/cosmos-framework/pull/342)
remains closed and unmerged. The audited starting revision was
`53b2382e00e8487270a2d09fcf2a13d9597340ac`. Main
`86e1ee52f276c11d6cc66dfd17da2795dfbe937b` was inspected and merged; its
intervening changes were skills, documentation and CI.

This report distinguishes implemented bindings from completed real-checkpoint
qualification. The machine-readable [case evidence](assets/family-adapter-qualification.json)
records immutable model revisions, actual source/wheel/environment identities,
losses, trainable counts, numerical errors and report hashes. Historical Cosmos3
full-WTS results remain in [adapters-progress.md](adapters-progress.md); they are
not results for another checkpoint. Four-bit training remains separate.

## Requirement status

Paths below are relative to `cosmos_framework/posttrain/` unless stated otherwise.

| Requirement | Code and focused tests | Status |
| --- | --- | --- |
| Shared LoRA/rsLoRA, DoRA and PiSSA | `model/adapters.py::PosttrainAdapterLinear`, existing native `utils/generator/lora.py`; `test_adapters.py`, `test_adapter_autocast.py` | Implemented and validated for the completed cases below. Native LoRA mathematics reused; DoRA magnitudes and PiSSA initial factors retained |
| Component/layer selection and independent settings | `model/adapter_families.py`, `resolve_targets`; `test_adapter_families.py`, `test_adapters_native.py` | Implemented and validated. Every selector must match its component; alias conflicts and unsupported target types fail before injection |
| Placement, trainability and optimizer ownership | `install_adapters`, `assert_trainability`, `optimizer_groups`, SFT `build_trainer` | Implemented and validated. Logical trainable inventory equals optimizer membership, without duplicates or frozen parameters |
| HF construction, tying, freezing and FSDP lifecycle | `model/hf.py::HFModel.setup`; `test_model_hf.py`, `test_text_sft_adapters.py` | Implemented and validated. Load/materialize base, install adapters, apply distributed transformation, then construct optimizer; adapter-disabled branch retained |
| Native VLM/MoE lifecycle | `model/qwen3_vl.py::Qwen3VLBuildHooks.after_load_before_parallelize`; `test_adapters_native.py`, `test_adapter_moe_export.py` | Dense VLM implemented and validated; MoE implemented, actual training/resume passed for LoRA, portable reload failed and is under diagnosis |
| Text SFT templates and masks | `data/packer/text_sft.py`, existing source-traced SFT masker; `test_data_packer_text_families.py`, `test_text_sft_adapters.py` | Implemented and validated for explicit ChatML, Llama3, Mistral and Gemma protocols. Unsupported or reordered renderings fail |
| Vision/projector/action gradient paths | Native Qwen ordered-vision forward, `model/pi05/modeling.py::training_condition`, Alpamayo and GR00T native SFT policies; family fixtures | Implemented and validated in completed pretrained cases; GR00T has small-model evidence only. Differing rank modalities tested on native VLM |
| Fresh initialization, warm start, exact resume | Shared adapter preparation and checkpoint lifecycle; qualification next-update comparison | Implemented and validated in completed cases. Resume restores optimizer/scheduler/data/RNG clock and skips method initialization/decomposition |
| Portable artifact and merged export | `checkpoint/adapters.py`, `checkpoint/tools/adapters.py`, existing materializer; family-owned physical exporters | Implemented and validated in completed cases. Separate artifact kinds, exact base/processor/media identities, strict inventory, non-mutating FP32 merge |
| π0.5 offline SFT | `model/pi05/{sft,adapter_export}.py`, existing flow objective and `LLMTrainer`; `test_pi05_adapters.py`, `test_flow_sft.py` | Implemented and validated: all four methods, real LIBERO data, two GPUs/full AC, continuation and both reload types |
| Public Alpamayo R1 SFT | `model/alpamayo/{sft,policy,adapter_export}.py`; `test_alpamayo_adapters.py` | Implemented and validated: all four methods, real KITTI data, two GPUs/vision AC, continuation and both reload types; no AlpaGym runtime required |
| GR00T full-backbone/action SFT | `model/gr00t/{sft,sft_policy,adapter_export}.py`; `test_gr00t_adapters.py` | Implemented; 19 actual small Isaac/Qwen3-VL/DiT tests pass. Pretrained GPU validation blocked by required gated backbone metadata (HTTP 401), detailed below |
| Installed artifact provenance | `tests/qualify_adapters.py::attest`, `installed_environment`; `test_qualification_environment.py` | Implemented and validated. Installed package bytes match the recorded wheel; host environment identity is distinct from container identity |
| Gemma FP32 embedding constant | `model/hf_adapter_buffers.py`; `test_hf_adapter_buffers.py` | Implemented and validated. Canonical nonpersistent constant preserves BF16 forwards and stock FP32 merge; legacy missing-contract artifacts fail explicitly |
| Native Qwen3-VL full WTS epoch/evaluation | Existing SFT/evaluator, method-specific full-profile gates, final checkpoint/recovery/export verification | Implemented orchestration; LoRA/rsLoRA full-profile gates running; new full epochs and evaluation awaiting completion. Existing Cosmos3 scores are not reused |

## Completed real-checkpoint profiles

All four methods means ordinary LoRA, rsLoRA, DoRA and PiSSA. Every passing case
includes multiple optimizer updates, complete target/update and frozen-base checks,
controlled checkpoint continuation, fresh-process portable reload, and independent
FP32 merged-file reload. These are correctness subsets, not complete task benchmarks.

| Checkpoint | Qualified GPU / AC profile | All four methods |
| --- | --- | --- |
| Qwen3-0.6B | 1/2 GPU × none/full | Passed (16/16 cases) |
| Native Qwen3-VL-2B-Instruct | 1/2 GPU × none/full | Passed (16/16 cases) |
| Qwen2.5-0.5B-Instruct | 2 GPU / full | Passed (4/4 cases) |
| SmolLM2-135M-Instruct (Llama architecture) | 2 GPU / full | Passed (4/4 cases) |
| Mistral-7B-Instruct-v0.3 | 2 GPU / full | Passed (4/4 cases) |
| Gemma-2B-IT | 2 GPU / full | Passed (4/4 cases) |
| Gemma2-2B-IT | 2 GPU / full | Passed (4/4 cases) |
| Qwen3.5-0.8B | 2 GPU / full | Passed (4/4 cases) |
| π0.5 LIBERO SFT | 2 GPU / full | Passed (4/4 cases) |
| Alpamayo-R1-10B | 2 GPU / vision | Passed (4/4 cases) |
| Native Qwen3-VL-30B-A3B-Instruct | 2 GPU / full | Not yet qualified: LoRA portable reload failed; training/resume passed |
| GR00T-N1.7 LIBERO-Spatial | No pretrained profile qualified | Blocked on authorized backbone metadata |

Text fixtures use eight real GSM8K training rows and eight held-out test rows.
Conversational VLM fixtures combine real WTS videos with controlled text, PNG and
MP4 examples, including differing modality presence across ranks. SmolLM2 verifies
the Llama architecture, not Meta Llama checkpoints. Gemma fixtures use pinned public
BF16 checkpoints from `unsloth/gemma-2b-it` and `unsloth/gemma-2-2b-it`; no Unsloth
runtime or quantization is used.

π0.5 uses an explicit 40/41-window training subset and all 55 validation windows
from a separate LIBERO episode. Alpamayo uses 14/15 training windows and all 38
validation windows from a separate KITTI drive. Omitted row IDs are recorded;
these are not full LIBERO/PhysicalAI runs. The physical loss is a masked mean over
action elements. Report both weighted training mean and last-batch loss because
native flow noise/time sampling makes a single batch unrepresentative. These
losses do not establish simulator success or driving performance.

All completed local continuation comparisons have zero maximum parameter error;
portable reload comparisons are exact. Alpamayo FP32 merged errors are at most
`2.7e-6`; π0.5 at most `8.9e-6`. Conversational FP32 merge uses elementwise
`atol=5e-4, rtol=2e-4`, mean error at most `5e-5`, and identical controlled argmaxes.
BF16 training-versus-FP32 deployment drift is measured separately; it is not
conflated with the merge oracle. Physical reload compares fixed-noise/time velocity
and loss. The JSON artifact contains method-specific losses and allocator peaks;
shared CPU/disk timings are not isolated throughput measurements.

## Runtime and observed fixes

Conversational SFT uses Torch `2.10.0+cu128`, Transformers `5.8.1`, safetensors
`0.8.0`, and CPU TorchCodec `0.10.0`. Slurm uses Python 3.13; the local isolated
host environment uses Python 3.12.13. Physical SFT has its own pinned x86 runtime
at [`docker/posttrain/flow-environment`](../../docker/posttrain/flow-environment),
with Transformers `4.57.6` and pinned Isaac/Alpamayo source hashes. Existing RL
and serving environments are not modified.

| Installed source | Validation | Artifact |
| --- | --- | --- |
| `3a9552b` | 223 CPU tests; two completed Slurm matrices, 16 cases each | Qwen3 job `37701630`, native Qwen3-VL job `37701632`; wheel/container hashes in case evidence |
| `5eecf10` HF host | 386 passed, 1 skipped | Original text/Qwen3.5/MoE host cases; byte-attested wheel and environment |
| `2556b4c` physical host | 138 passed, 1 skipped | π0.5 logical-shard update checks |
| `14c9e09` physical host/container | 138 passed, 1 skipped | Alpamayo FP32 merged construction; image job `37709361` completed |
| `076cfbb` HF host | 407 passed, 6 skipped | Corrected Gemma/Gemma2 and new full-WTS wheel; 1,876 installed package files verified |
| `076cfbb` HF container | 396 passed, 1 skipped | Image job `37710535` completed; SHA `8e9e8cf5591c77a2a6f4a143e432f5234d924292d6be68d7d257ff9aca64eeca` |
| `076cfbb` evaluator host | 14 focused installed tests plus actual two-rank video MCQ preflight | Same framework wheel, separate locked environment including `toml==0.10.2` |

Counts overlap and must not be added. The general Torch 2.11 environment is a
separate unqualified profile. Every GPU result retains its actual revision; a
later documentation commit does not relabel an earlier run.

Real runs exposed and verified these fixes:

- Qwen3.5's public checkpoint includes unused MTP source tensors. Its export
  exception is bounded to that exact architecture/namespace and recorded in the
  materialization marker; unexpected tensors elsewhere still fail.
- FSDP can place only padded action-output rows on one rank. Qualification now
  requires each logical trainable tensor to update somewhere, while recording
  rank-local unchanged shards and checking each rank's optimizer/frozen state.
- Alpamayo's nested BF16 text config rounded stock FP32 merged weights during
  construction. The merged loader constructs FP32 before loading; a realistic
  nested-dtype regression failed before the fix. LoRA's original training is
  retained, with its corrected merge tested at `14c9e09`; later methods trained
  from that revision.
- Gemma's BF16 nonpersistent embedding scale was `45.25` instead of
  `45.2548339959`. Keeping the config-derived constant in FP32 preserves its BF16
  forward and reduces real-model merge mismatch from `0.00805` to `0.000210`.
  Corrected qualification is a fresh run; the failed attempt remains preserved.

## Pending execution and explicit gaps

GR00T-N1.7 LIBERO-Spatial weights and real LIBERO data are staged, and its native
runtime is built. The exact `nvidia/Cosmos-Reason2-2B` config/tokenizer/processor
at `9ce19a195e423419c349abfc86fd07178b230561` returned HTTP 401. Its builder
requires authorized metadata and refuses a guessed or substituted Qwen snapshot.
That is a concrete remaining resource dependency, not a passing GR00T GPU result.

Native MoE implementation targets ordinary two-dimensional attention/vision/merger projections;
fused three-dimensional expert tensors remain unsupported targets. Its existing
native grouped-mm forward stays BF16. The separate stock HF FP32 export oracle
uses two GPUs, checks every serialized tensor and records native-versus-stock BF16
drift. It does not introduce another expert kernel or training loop.

Other bound Qwen2-VL/2.5-VL small fixtures do not qualify their public checkpoints'
legacy processor/namespaces; the production SFT packer still refuses them.
Unregistered architectures, HSDP, TP/CP/PP/EP, compilation, larger worlds,
multiple nodes, stacked adapters and quantized training are not qualified here.
Physical merged artifacts have model-owned FP32 loaders; they are not stock HF,
Omni/PPO or closed-loop deployment claims. No universal model-family support is
inferred from a component binding.

Additional Slurm method profiles remain pending; they are not counted as results.
Unstarted MoE jobs are held while the local reload failure is diagnosed:

| Family | LoRA | rsLoRA | DoRA | PiSSA | Profiles per job |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5 | 37707970 | 37707974 | 37707976 | 37707978 | 1/2 GPU × AC none/full |
| π0.5 | 37710088 | 37710094 | 37710096 | 37710463 | 1/2 GPU × AC none/full |
| Alpamayo | 37710102 | 37710106 | 37711261 | 37711674 | 1/2 GPU × AC none/vision |
| Native Qwen3-VL-MoE | 37708008 | 37708009 | 37708011 | 37708012 | 2 GPU, AC full |
| Qwen2.5 | 37708301 | 37708306 | 37708308 | 37708309 | 1/2 GPU × AC none/full |
| SmolLM2 | 37708310 | 37708311 | 37708312 | 37708313 | 1/2 GPU × AC none/full |
| Mistral | 37708398 | 37708399 | 37708400 | 37708401 | 1/2 GPU, AC full |

## New native Qwen3-VL full WTS verification

The local source copy verifies all 517 files, including 512 videos, against
SHA-256 and preserves all 5,555 training and 2,676 validation rows. Each method
must first pass its own full-resolution/gradient-accumulation qualification on
the installed `076cfbb` wheel before the existing SFT entrypoint trains exactly
1,111 updates (microbatch one, accumulation five). Every recorded row ID and data
cursor is reconciled; no implicit padding or dropped training rows is accepted.

After each complete epoch, independent read-only recovery replays update
1110→1111 and compares parameters, optimizer, scheduler, RNG, clock and cursor.
Final export checks the frozen base and PiSSA initial factors, verifies all
trainable tensors changed in the final update, then performs portable and FP32
merged reload. Only then does the existing two-GPU evaluator score all 2,676
validation examples and the reserved 576-example scene-disjoint subset. Generated
NLL is reported separately from teacher-forced validation loss. No new AutoML
search substitutes for these checks.

Original Slurm full-training jobs `37706319`, `37706324`, `37706326`, `37706327`
are preserved. Their dependent evaluation jobs are `37711018`, `37711029`,
`37711043`, `37711056`. Unstarted earlier evaluation launchers were replaced after
removing an unbound environment-variable read; old sealed inputs and cancellation
receipts remain available. Local runs use true host-environment identities and
local attempt IDs, never fabricated Slurm/container identities.
