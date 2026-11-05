# Adapter SFT

The posttrain SFT workflow can adapt dense native Qwen3-VL, including
Cosmos3-Nano-VLM, with language, final/deepstack merger and selected vision-block
adapters. It reuses `LLMTrainer`, the conversation packer, data cursor and checkpoint
transactions. The native TOML workflow already supported LoRA/rsLoRA before this
contribution; this is an extension of the separate posttrain workflow.

The family extension also binds dense HF text architectures, HF Qwen3.5 and
native Qwen3-VL MoE's ordinary projections through
`configs/sft_family_adapters.py`. Their qualification is tracked separately in
[adapter-family-progress.md](../adapter-family-progress.md). Physical-action
models use their own native forward and processor through the same SFT lifecycle;
see [physical-action adapters](flow-adapters.md). Support is specific to the
recorded checkpoint and execution profile, not every model sharing its family name.

See the [original qualification](../adapters-progress.md) and the
[execution/deployment extension](../adapter-extension-progress.md) for exact
checkpoint/profile coverage. CPU correctness is not real-model GPU qualification.
The [effective-execution extension](../adapter-effective-execution.md) records
physical 2×2 HSDP/full replay, explicitly versioned PiSSA execution and broader
BF16 checks, including their failures. Its explicitly versioned MoE profile passes
stock FP32 merged qualification; the legacy PiSSA merge guard remains enabled.
The extension adds measured native Qwen3-VL-2B and HF Qwen3-0.6B 2×2 HSDP,
native 2×1 replication/eager attention, selective SDPA checkpointing and independent
component recipes, on one host with up to four A100 80 GB GPUs. BF16 frozen bases
and FP32 adapters remain the training profile. These results do not qualify every
family/checkpoint on every mesh. Four-bit training, unregistered architectures,
fused expert tensors, TP/EP/CP/PP, multiple nodes, compilation, stacked adapters,
arbitrary linear/embedding `modules_to_save` and the separate Torch 2.11 runtime
remain outside qualification. Plain native `nn.Linear` and optional native
normalization targets are supported. Exact-SVD PiSSA and FP32 training bases have
fixture coverage. Fused vision QKV is one whole linear target, not independently
selectable Q/K/V slices.

## Recipe and component selection

The original CUDA 12.8 Slurm profile uses Python 3.13; the new attested host
profiles use Python 3.12. Both use torch 2.10.0+cu128,
torchvision 0.25.0+cu128, Transformers 5.8.1, safetensors 0.8.0, av 18.0.0 and
the CPU TorchCodec 0.10.0 wheel. The general [SFT environment](sft.md) instead
pins torch 2.11 / TorchCodec 0.11; that is a separate qualification case.
Starting from an isolated Framework CUDA 12.8 environment with torch 2.10,
install the implementation wheel and the SFT processor versions explicitly:

```bash
# SFT_PYTHON names that environment's interpreter. Do not modify a shared RL venv.
uv pip install --python "$SFT_PYTHON" --no-deps /path/to/cosmos_framework-1.2.2-py3-none-any.whl
uv pip install --python "$SFT_PYTHON" transformers==5.8.1 safetensors==0.8.0 av==18.0.0
uv pip install --python "$SFT_PYTHON" --no-deps torchcodec==0.10.0 \
  --index-url https://download.pytorch.org/whl/cpu
"$SFT_PYTHON" -c 'import torch, transformers; assert torch.__version__ == "2.10.0+cu128"; assert transformers.__version__ == "5.8.1"; assert torch.cuda.is_available()'
```

PEFT 0.21.2 is used only for a test reference, not for adapter installation.
Set `SFT_MODEL_PATH` to a complete immutable local snapshot
(config, safetensors, tokenizer, processor and template). The initial recipe
uses Cosmos3-Nano-VLM's last vision block, index 26; other dense Qwen3-VL
checkpoints must select indices that actually exist.

```bash
export SFT_MODEL_PATH=/models/Cosmos3-Nano-VLM
export SFT_DATA_ROOT=/data/adapter-example
"$SFT_PYTHON" -m cosmos_framework.posttrain.scripts.prepare_sft_example --output "$SFT_DATA_ROOT"
export SFT_ADAPTER_METHOD=lora   # lora, dora, pissa
export SFT_ADAPTER_RSLORA=0      # 1 uses alpha/sqrt(rank), including with DoRA/PiSSA
"$SFT_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=1 \
  -m cosmos_framework.posttrain.entrypoints.sft \
  --config=cosmos_framework/posttrain/configs/sft_cosmos3_nano_adapters.py
```

The recipe sets `rl.adapters` using the existing dataclass/OmegaConf conventions.
An independent-settings example in a Python recipe is:

```python
from cosmos_framework.posttrain.config.adapters import AdapterConfig, AdapterComponentConfig, AdapterExtraModuleConfig

config.rl.adapters = AdapterConfig(components=[
    AdapterComponentConfig(component="language", rank=8, alpha=16, dropout=0.05,
        targets=["model.language_model.layers.*.self_attn.q_proj",
                 "model.language_model.layers.*.self_attn.v_proj"]),
    AdapterComponentConfig(component="projector", method="dora", rank=4, alpha=8,
        lr_multiplier=0.5, targets=["model.visual.merger.linear_fc2",
                                  "model.visual.deepstack_merger_list.*.linear_fc2"]),
    AdapterComponentConfig(component="vision", method="pissa", rank=4, alpha=8,
        initialization="svd_lowrank", pissa_niter=4, dropout=0, layers=[25, 26],
        lr_multiplier=0.1, targets=["model.visual.blocks.*.attn.qkv"]),
], extra_modules=[
    AdapterExtraModuleConfig(component="language", targets=["model.language_model.norm"],
                             lr_multiplier=0.1, weight_decay=0),
])
```

Every full-path glob must match after layer restrictions. Cross-component matches,
unsupported module types, missing layer indices, conflicting overlaps and partial
selection of tied aliases fail before injection. Shared modules/parameters are
resolved by identity; compatible aliases share adapters and one optimizer entry.
Cross-block ties are refused for FSDP2. Architecture-declared embedding ties remain
frozen; embedding/head adaptation is not supported. Arbitrary custom ties are not a
claim of compatibility with a stock HF architecture.
Portable reload preserves declared custom adapter aliases; stock HF merged export
rejects them because its config cannot reconstruct those ties. Architecture-declared
frozen embedding/head ties are preserved through both export types.

The resolved manifest records component, path, type, shape, owner alias, rank,
effective scaling, method/init/dropout/dtype, LR/decay and expected trainable count.
It is checked against actual trainability and final optimizer parameter identities.
Clear `optimizer.train_keys`, `decay_groups` and `lr_multipliers`: adapter component
settings own those policies. `lr_multiplier` multiplies the global optimizer LR;
`weight_decay` is absolute and defaults to zero. The initial profile uses AdamW,
FP32 adapter storage/accumulation, BF16 or FP32 frozen base, eager/SDPA and full
non-reentrant activation checkpointing or none. Native Qwen3-VL additionally
supports selective checkpointing, with the precise GPU profiles listed in the
extension record. BF16 adapter storage is refused.
Separately trained normalization parameters remain FP32 while their output keeps
the input activation dtype. The default FP32 merged export retains these parameters
in FP32; optional BF16 deployment rounds all floating weights, including extras.

## Methods and lifecycle

LoRA and rsLoRA use the existing `LoraInjectedLinear`: Kaiming-uniform A, zero B,
scale `alpha/r` or `alpha/sqrt(r)`. Zero B makes the initial A gradient zero; tests
check connectivity and actual changes over several updates. Autocast is disabled
around adapter math; the final update is cast to the base output dtype. Dense merge
uses the existing FP32 accumulation policy with autocast and TF32 disabled.

DoRA keeps a trainable FP32 output-row magnitude, initialized to the base row norm.
The direction norm is detached during training, following the
[DoRA reference](https://github.com/NVlabs/DoRA). Bias is unscaled. Magnitudes enter
the optimizer, full checkpoint and portable state. Export normalizes the completed
direction and applies the trained magnitude. Only one adapter is merged into its
base; combining multiple adapters is not implemented.

PiSSA initializes A/B using `svd` or randomized `svd_lowrank` with an explicit
iteration count (default 4). It stores immutable initial factors A0/B0 and computes
`W0 x + s(BA - B0A0)x`. This represents the residual base `W0 - s B0A0` while
leaving the original parameter bytes intact. The artifact contains both initial
factors; it must never be loaded as raw BA on W0. Merge uses the
[PiSSA rank-2r conversion](https://github.com/MuLabPKU/PiSSA), concatenating
`[A; A0]` and `[B, -B0]` with the original effective scale. PiSSA requires dropout
zero; DoRA plus PiSSA initialization on the same target is rejected.

Adapter-enabled construction follows this order:

1. Construct the native base on meta, prepare its existing multimodal forward,
   materialize, tie architecture-declared embeddings, and strictly load the base.
2. Resolve all targets; install/initialize adapters, or allocate initialized
   placeholders for warm start/exact resume. Resume never decomposes the base.
3. Freeze the base, enable adapters/extra normalization parameters, and warm-load
   a portable artifact if requested.
4. Apply activation checkpointing and FSDP2 after every adapter exists. FSDP retains
   mixed BF16 frozen/FP32 trainable storage, with FP32 gradient reduction.
5. Construct AdamW from the final parameter objects; assert complete, unique
   membership. Exact resume then strictly restores trainer/model/optimizer,
   scheduler, clocks, rank RNG and the data cursor through the existing boundary.

No adapter tensor remains on meta. The adapter-disabled builder retains its
existing sharding-before-loading order. Adapter initialization currently loads the
full base on each GPU before sharding, so the base must fit on each device; this
is intentional for the initial 80 GB profile. Native Qwen3-VL's single visual
forward and graph-connected dummy media path are preserved for differing modality
presence across ranks. Vision features are not detached or cached by this path.

## Three artifact types

**Training checkpoints** remain the existing full consolidated or DCP transaction.
They include frozen weights as well as adapter tensors, PiSSA initial factors,
optimizer, scheduler, update state, RNG and data cursor. No adapter-only checkpoint
optimization is introduced. Resume uses `rl.resume`; missing adapter keys or changed
objective/base/processor/media identity fail. Resume across a changed DP world is
not an exact-continuation claim. `warm_start` must be cleared for exact resume.
New adapter checkpoints also bind backend/builder, attention, activation
checkpointing and resolved DP dimensions. Changing these fails exact resume.
Checkpoints written before `2d12fd5` lack that execution identity: resume them
with their original runtime/configuration, or export portable adapters and start
a new run. Do not rewrite their identities to bypass validation. Portable adapter
identities and adapter-disabled checkpoint compatibility remain unchanged.

**Portable adapters** are immutable directories with `adapter.safetensors` and a
commit-last `adapter_manifest.json`. They record SHA-256 over all original weight
shards and config/processor/tokenizer/template sidecars, the resolved target map,
extra modules, method configuration, effective processor arguments and relevant
dependency/format versions. A same-size different base is rejected. Export is an
atomic no-replace publication using checkpoint IO primitives. These artifacts
contain no optimizer or data state and cannot resume training.
On filesystems without `RENAME_NOREPLACE` (including the qualification Lustre
mount), exclusive `mkdir` reserves the destination, weights are installed and
synced first, and the manifest commits last. Interrupted reservations are rejected
by readers and cannot be overwritten; retry at a fresh output path.

```bash
"$SFT_PYTHON" -m cosmos_framework.posttrain.checkpoint.tools.adapters \
  --checkpoint /runs/example/checkpoints/iter_000006 --out /exports/adapter
# In a recipe, for a new training run:
# config.rl.adapters.warm_start = "/exports/adapter"
```

**Merged HF models** use FP32 weights and inference. The exporter first reproduces
the recorded training-base values (including BF16 rounding), then promotes them
and applies the pure FP32 merge. It reuses the existing locked/staged HF
materializer, extends its identity with output precision, and verifies both shard
precision and config metadata before publication. `dtype="auto"` loads FP32.
The base's architecture and all processor/template sidecars are preserved.
The committed `.materialized.identity.export_metadata` record has format
`cosmos-merged-adapter-v1` and retains the complete portable adapter identity and
adapter weight digest: original base, resolved targets, method settings, extra
modules, processor/tokenizer/template and effective media settings. This metadata
participates in output/staging identity checks; identical weights with different
media settings cannot silently reuse an existing export. Legacy merged artifacts
without this record must be re-exported to a fresh destination.
Portable loading hashes and deserializes the same captured payload bytes. HF
export retains that captured manifest so an external path replacement cannot
label consumed tensors with a different adapter's digest. This uses temporary
memory proportional to the adapter payload, not to the base-model weights.
Before merge, the exporter re-resolves the recorded targets against the stock
architecture on meta; mismatched types, component ownership, scaling and counts
fail before loading base shards. Export leaves live parameters, optimizer
membership and tensor versions unchanged.
Stock HF reload is tested separately from native portable reload. Optional
`--output-dtype bfloat16` now rounds the completed FP32 merge once and binds the
deployment precision in the export identity. It is qualified for the exact dense
Qwen3-VL-2B WTS rsLoRA, DoRA and PiSSA artifacts in the
[extension record](../adapter-extension-progress.md#bf16-deployment-contract-and-measurements),
using separate approximation, paired task-quality and resource gates. It is not
FP32 equivalence. Ordinary non-rs LoRA with default stock BF16 execution failed
the unchanged confident-decision gate; that export remains refused. The same
ordinary LoRA artifact is now qualified with the explicit
`--execution-profile qwen3_vl_bf16_residual_fp32_v1` option. This keeps BF16 merged
weights and logits, with FP32 intermediate results, attention and residual state.
Use `CosmosFrameworkRuntime` with BF16/auto dtype and SDPA to validate and install
the artifact-bound profile. Plain HF loading alone does not activate its arithmetic.
Tagged loading rejects missing or unexpected weights instead of permitting HF
to initialize or ignore them. The supported execution is one CUDA device, eager
module execution, evaluation with no gradients/autocast, and the recorded library
versions. KV tensors are FP32. See the
[deployment continuation](../adapter-deployment-continuation.md) for exact
precision semantics, resource costs, paired results and reproduction commands.
For newly trained v4 `merged_weight_bf16_v1` artifacts, the separate
[canonical cache profiles](../adapter-cache-checkpoint-progress.md) qualify
explicit Cosmos BF16 execution on the named Qwen3-VL-2B and Qwen3.5-0.8B
mixed recipes. They retain BF16 activations/KV (Qwen3.5 recurrent state is FP32),
require their declared loaders and have substantial Qwen3-VL long-prefill cost.
They do not repair or relabel the legacy artifacts discussed here.
Other families/physical formats remain refused. New mixed recipes and other
checkpoints outside those explicitly qualified profiles are not additional
BF16 qualifications. The broader input and
mixed-component checks in the effective-execution extension fail the unchanged
numerical gates despite passing resource gates; the previous fixed-artifact
results must not be generalized to them.
Legacy PiSSA merged export on native Qwen3-VL-MoE remains refused, including dense
PiSSA under default native model execution. The original 30B artifact passes
training, continuation and BF16 portable reload, but FP32 merging changes a
near-tied route and fails the preserved output bounds. Keep its portable adapter;
[its measured failure](../assets/moe-pissa-export-limitation.json) remains incomplete.

New training with `AdapterConfig.model_execution="qwen3_vl_moe_stock_v1"` and
`execution="pissa_dense_v1"` on every selected PiSSA component has a separate,
versioned FP32 merge contract. The pinned 30B checkpoint passes all 64 original/new
inputs through the unmodified stock loader with zero logit error. Use the supplied
`sft_qwen3_vl_moe_pissa_stock.py` recipe and the exact precision/runtime profile in
the effective-execution report. This does not convert legacy artifacts, qualify
MoE BF16 merged deployment, or establish task quality after FP32 promotion.
Other qualified PiSSA families retain their supported FP32 merged exports.

```bash
"$SFT_PYTHON" -m cosmos_framework.posttrain.checkpoint.tools.adapters \
  --adapter /exports/adapter --base "$SFT_MODEL_PATH" \
  --merged --out /exports/merged
```

Small FP32 fixtures require output agreement within `atol=3e-5, rtol=3e-5` and
bitwise controlled next-update recovery. Portable GPU reload uses `2e-5` and a
`2e-6` mean bound; controlled GPU continuation uses `atol=1e-7, rtol=1e-5` on
trainable parameters and optimizer state.

Controlled training and reload qualification both enable PyTorch deterministic
algorithms, disable TF32 matmul and cuDNN benchmarking, and record these settings
as `numerical_profile`. Launch with `CUBLAS_WORKSPACE_CONFIG=:4096:8` (or
`:16:8`) before CUDA initialization. Native MoE's BF16 atomic expert combine
otherwise changes summation order; comparing it with a deterministic reference
is a different execution profile, even with identical weights. Normal model
loading retains the caller's settings. Reproducible reload requires matching the
recorded execution profile as well as the artifact and inputs.

Merged export has three mandatory checks: every serialized tensor equals the
FP32 reference merge exactly; native merged and stock HF forwards match the
portable adapter evaluated in FP32 (`atol=5e-4, rtol=2e-4`, mean error at most
`5e-5`, identical argmax); and a fresh stock HF load with `dtype="auto"` reproduces
those controlled outputs. Small fixtures retain their tighter existing bounds.
The control uses a private evaluation copy and never changes a live training model.
For default native Qwen3-VL-MoE execution, BF16 training uses the native grouped
expert kernel and `MoEExperts.forward` selects `native.naive` for a promoted FP32
copy. The new explicit stock-compatible profile instead uses the pinned stock
grouped-expert helper in both precisions and separate image/video vision calls.
`posttrain.tests.reload_moe_adapters` uses that independent native reference,
compares stock HF's portable and merged FP32 forwards against it, verifies every
serialized tensor in the original checkpoint layout, and dispatches FP32 inference
across at least two GPUs with explicit memory budgets and no implicit CPU/disk
offload. The ordinary native BF16 portable reload is a prerequisite. The FP32
reference preserves the base's BF16 rounding; it does not reload original
higher-precision base weights. BF16 merged inference, including stock HF's different
expert arithmetic, is outside the supported export profile. Differences from the
training-precision outputs are reported separately, as for the dense families.
In the completed 30B LoRA case, FP32 merged output agrees with the FP32 portable
reference at all 64 controlled greedy positions (maximum logit error `4.96e-5`).
Compared with BF16 training computation, six positions change, with maximum logit
error `6.65`. Exact BF16 portable reload passes. A passing FP32 merge gate therefore
does not establish equivalent BF16 predictions or FP32 downstream task quality.
The 30B profile advertises only two-rank pure DP with full activation checkpointing
once its GPU gate passes; ordinary linear attention/vision/merger targets are
distinct from unsupported fused expert tensors.
For a BF16-trained adapter, build/reload at its recorded base dtype first, then
call `policy.net.float().eval()` for this FP32 reference. Constructing directly
from the original FP32 base would undo training-time rounding and is incorrect.

BF16 adapter and dense GEMMs round at different boundaries; combined image/video
encoding also has different GEMM batch shapes in native and stock HF. Real-model
BF16 exports in the historical family qualification failed both its initial absolute-logit proposal and later fixed
probability-drift gates (including a `.0475` probability loss for a changed rsLoRA
greedy token). Those failures are retained in the qualification record. Their
limits were not widened to declare success. The later bounded BF16 deployment
qualifications above use separately predeclared approximation/task/resource
contracts; they do not turn those historical equivalence failures into passes.
The supported FP32 export
uses same-precision equivalence; it costs twice the BF16 weight storage and has
a different compute cost. Reports also measure FP32-export versus BF16-training
outputs, but do not claim cross-precision generation equivalence. Portable reload
retains the original inference precision and exact controlled outputs.
Passing a backward call alone is not a quality or resource qualification.

Run the focused correctness suite from the checkout in the isolated environment:

```bash
uv pip install --python "$SFT_PYTHON" pytest peft==0.21.2
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 "$SFT_PYTHON" -m pytest -q -o addopts='' \
  --confcutdir=cosmos_framework/posttrain/tests \
  cosmos_framework/posttrain/tests/test_adapters.py \
  cosmos_framework/posttrain/tests/test_adapters_native.py \
  cosmos_framework/posttrain/tests/test_checkpoint_materialize_hf.py \
  cosmos_framework/posttrain/tests/test_checkpoint_hf_overlay.py \
  cosmos_framework/posttrain/tests/test_checkpoint_hf_verification.py
```

The installed-wheel GPU harness is `posttrain.tests.qualify_adapters`, launched
with `torchrun` and explicit `--wheel`, `--build-record`, `--method`, `--component`
and `--activation-checkpointing` arguments. It calls the existing SFT builder and
trainer, saves controlled probes and checkpoints, and verifies the installed
package bytes. `posttrain.tests.reload_adapters` performs the separate-process
portable/merged checks. Qualification records include source and wheel digests,
plus either the published container SHA or a sealed host-environment manifest.
For an isolated host runtime, create that manifest with
`qualify_adapters.installed_environment(wheel_path)` after installation and set
`ADAPTER_ENVIRONMENT_MANIFEST` to its JSON path. Each gate compares its Python,
platform, CUDA runtime, installed distributions and wheel against that manifest;
package or interpreter drift fails. `ADAPTER_CONTAINER_SHA256` is reserved for
an actual published image. Results from an unchanged baseline package do not
qualify this extension.

FSDP update checks compare logical trainable tensors across ranks. Every tensor
must change on at least one rank; its local shards need not all change. For
example, a π0.5 output projection has 32 channels but LIBERO supervises seven,
so a shard containing only padded output rows can legitimately stay unchanged.
The harness reports these unchanged shards, rejects a tensor unchanged on every
rank, and verifies frozen state and optimizer ownership on every rank.

WTS annotations can be converted without resampling using
`posttrain.scripts.prepare_wts_sft`. Full-epoch jobs must account for every original
row under the existing DP divisibility rule. Report training/validation loss,
held-out multiple-choice accuracy/invalid answers, samples and supervised tokens
per second, setup time, peak CUDA memory, checkpoint/export size and latency,
checkpoint SHA, wheel SHA and source revision. Full WTS runs begin only after the
focused installed-package and GPU gates pass; AutoML is not a substitute.

For the 5,555-row WTS training split and 2,676-row evaluation split, the prepared
full-epoch profile uses one GPU, five microbatches per update and 1,111 updates.
Every training row is consumed exactly once. The native Qwen3-VL family run uses
`Qwen/Qwen3-VL-2B-Instruct@89644892e4d85e24eaac8bacfd4f463576704203`,
with its actual last vision block (23), all final/deepstack mergers and language
Q/V projections. This resolves to 65 adapter targets. Set `SFT_DATA_ROOT` to a
new output directory; conversion refuses to overwrite existing JSONL files.

```bash
export SFT_MODEL_PATH=/models/Qwen3-VL-2B-Instruct/89644892e4d85e24eaac8bacfd4f463576704203
export SFT_DATA_ROOT=/data/wts-native-qwen3-vl
"$SFT_PYTHON" -m cosmos_framework.posttrain.scripts.prepare_wts_sft \
  --annotations /data/wts_train/annotations.json --media-root /data/wts_train/videos \
  --out "$SFT_DATA_ROOT/train.jsonl"
"$SFT_PYTHON" -m cosmos_framework.posttrain.scripts.prepare_wts_sft \
  --annotations /data/wts_eval/annotations.json --media-root /data/wts_eval/videos \
  --out "$SFT_DATA_ROOT/validation.jsonl"
export SFT_ADAPTER_METHOD=lora SFT_ADAPTER_RSLORA=0
export IMAGINAIRE_OUTPUT_ROOT=/runs/wts-native-qwen3-vl/lora
"$SFT_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=1 \
  -m cosmos_framework.posttrain.entrypoints.sft \
  --config=cosmos_framework/posttrain/configs/sft_family_adapters.py -- \
  job.name=wts_full_epoch_all_components rl.run_ts=epoch1 \
  rl.max_steps=1111 rl.micro_batch_size=1 rl.grad_accum=5 \
  rl.save_every=111 rl.keep_model_last=2 rl.keep_optim_last=2 rl.resume=off \
  rl.validation.every_steps=0 rl.validation.batch_size=1 \
  rl.processor_kwargs.videos_kwargs.size.shortest_edge=200704 \
  rl.processor_kwargs.videos_kwargs.size.longest_edge=327680
```

For rsLoRA, use `SFT_ADAPTER_METHOD=lora` and `SFT_ADAPTER_RSLORA=1`.
For DoRA or PiSSA, set the method to `dora` or `pissa` and rsLoRA to zero.
Give each fresh run a distinct output directory. The shared recipe fixes rank 8,
alpha 16, AdamW LR `2e-5`, component LR multipliers `1/0.5/0.1` and seed 42.
Cosmos3-Nano-VLM uses its separate `sft_cosmos3_nano_adapters.py` recipe and
checkpoint; its historical WTS scores do not qualify the Qwen checkpoint.

The four-frame total-pixel bounds above correspond to 50,176–81,920 pixels per
frame in the pinned processor version. Validation runs over the full evaluation
split before training and after the final update. Continue an interrupted run
with the same output directory and recipe, changing only `rl.resume=auto`; a
portable warm start is a different operation and cannot continue its data cursor.

Gemma/Gemma2 adapters retain the config-derived embedding scale as a nonpersistent
FP32 buffer. Their native forward still casts it to the embedding dtype, preserving
BF16 training outputs. Promoted portable models and fresh stock-HF FP32 exports
therefore reconstruct the same constant. The portable `model_contract` records
this policy; older artifacts without it are rejected for merged export instead
of silently changing the model. Embedding/head weights remain tied.

For Qwen3-VL evaluation with `video_decoder="torchcodec-cuda-on-demand"`, decoded
frames retain the source FPS and exact selected frame indices. The native HF
video processor owns resizing and normalization, matching SFT's tensors. Pixel
limits in `[vision]` are per frame; the runtime converts them to the processor's
total-frame pixel budget. Short clips repeat sampled indices to retain the
requested fixed frame count. A batch must use the same frame count and pixel
bounds; use `batch_size=1` otherwise. This source-metadata path requires
`Qwen3VLVideoProcessor` and `min_pixels`/`max_pixels`, with `total_pixels` omitted.
Raw video paths and images retain their existing qwen-vl-utils handling.

Earlier decoded-frame evaluation fabricated a 2-FPS timeline and resized PIL
frames before video preprocessing. A real 2,211-frame, 30-FPS WTS clip produced
`<0.2 seconds>/<1.2 seconds>` and grid `[2,12,20]`, instead of SFT's
`<12.3 seconds>/<61.4 seconds>` and `[2,12,22]`. The corrected source probe matches
timestamps, grids and every pixel tensor value on two real clips. Existing full
training is unchanged; its new scoring jobs require the rebuilt evaluator and
an installed two-GPU preflight. Historical scores from the earlier evaluator
must remain labeled with that version.
