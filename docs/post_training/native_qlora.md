# Native QLoRA for Qwen3-VL

This experimental path trains native LoRA adapters on a frozen NF4 language base through `cosmos_framework.posttrain.entrypoints.sft`. Exact reload/resume checks and a matched 300-update WTS comparison completed on one GPU per run, with measured memory savings and a four-arm held-out diagnostic. **The release quality gate is not satisfied: trained NF4's token-NLL point estimate was 4.18% worse than BF16, exceeding the proposed 2% margin.** The evidence below is limited to one training seed and its stated workload and runtime.

The initial candidate is `Qwen/Qwen3-VL-8B-Instruct` at revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`, on **one Linux x86-64 A100 80GB, one torchrun process**. FSDP/HSDP, additional GPUs, offload, MoE, DoRA/PiSSA, other quantizers, and merged serving exports are rejected or unqualified. See the [design and release evidence requirements](qlora_native_vlm_design.md).

## Prepare a sealed runtime

Use uv `0.11.29` and Python 3.12. The [posttrain environment](../../docker/posttrain/environment/pyproject.toml) and its [lock](../../docker/posttrain/environment/uv.lock) select Torch `2.11.0+cu128`, Transformers `5.8.1`, and bitsandbytes `0.49.2`. The x86-64 selection uses CPU TorchCodec `0.11.0+cpu`; the host also needs working FFmpeg libraries for video decoding.

From the reviewed checkout, choose a new absolute work directory and build the environment:

```bash
export QLORA_WORK=/absolute/work/qlora
mkdir -p "$QLORA_WORK"
UV_PROJECT_ENVIRONMENT="$QLORA_WORK/runtime" uv sync \
    --project docker/posttrain/environment --frozen --python 3.12 \
    --no-default-groups --group dev --group sft --group cu128-runtime --group qlora
```

Build and hash a wheel containing the reviewed changes. The root package's dependency metadata belongs to a different runtime; install its code with `--no-deps` to preserve the selected posttrain environment. Pin the build tools as well:

```bash
cat > "$QLORA_WORK/build-constraints.txt" <<'EOF'
hatchling==1.29.0
packaging==25.0
pathspec==1.0.4
pluggy==1.6.0
trove-classifiers==2026.1.14.14
EOF
uv build --wheel --python 3.12 --no-sources \
    --build-constraints "$QLORA_WORK/build-constraints.txt" \
    --out-dir "$QLORA_WORK/wheels" .
export COSMOS_TRAINING_CODE_REVISION="$(git rev-parse HEAD)"
export QLORA_WHEEL="$QLORA_WORK/wheels/cosmos_framework-1.2.2-py3-none-any.whl"
export COSMOS_TRAINING_WHEEL_SHA256="$(sha256sum "$QLORA_WHEEL" | cut -d' ' -f1)"
uv pip install --python "$QLORA_WORK/runtime/bin/python" \
    --no-deps --no-index --reinstall "$QLORA_WHEEL"
```

Retain an immutable build receipt containing the checkout revision, complete source inventory/diff, environment-lock digest, build-tool versions, wheel digest, and installed package versions. Include untracked source files if the wheel was built from a working tree. The two `COSMOS_TRAINING_*` variables must come from that receipt; training records them and exact resume compares them. They are declarations, not automatic attestation of installed files.

Activate this environment, leave the source checkout, and unset `PYTHONPATH` so imports resolve to the installed wheel. Preserve these variables in the scheduler job environment:

```bash
source "$QLORA_WORK/runtime/bin/activate"
mkdir -p "$QLORA_WORK/run"
cd "$QLORA_WORK/run"
unset PYTHONPATH
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export IMAGINAIRE_OUTPUT_ROOT="$QLORA_WORK/outputs"
```

Set the determinism variables **before** starting Python or torchrun. Native adapter SFT requires `rl.deterministic=True` and enables strict deterministic algorithms before CUDA initialization; an operation without deterministic support fails. PyTorch's warning-only mode can use nondeterministic Flash Attention backward kernels and does not qualify exact resume. The matched BF16 adapter control uses the same strict mode. Do not change the driver library search path to conceal a kernel-loading failure.

## Prepare the immutable base and WTS data on CPU

Use a new source destination. Preparation downloads only the pinned revision and writes `cosmos_adapter_source.json`, binding weights, configuration, tokenizer, processor, and templates. Training verifies this manifest and never downloads the base on the accelerator. A completed directory is verified on reuse; an unmanifested directory containing stale or interrupted model files is refused.

```bash
export SFT_MODEL_PATH="$QLORA_WORK/base"
python -m cosmos_framework.posttrain.scripts.prepare_qlora_source \
    --destination "$SFT_MODEL_PATH"
```

The WTS converter accepts ShareGPT JSON arrays with `id`, relative `video`, and alternating `human`/`gpt` conversation turns. Supply the video directory as the media root. It inserts the actual video handle in the first user message and removes exactly one `<video>` marker, preserving other text and assistant answers. The system prompt is explicit:

```bash
export SFT_DATA_ROOT="$QLORA_WORK/wts"
python -m cosmos_framework.posttrain.scripts.prepare_qlora_wts \
    --train-annotations /absolute/data/wts_train/annotations.json \
    --train-media /absolute/data/wts_train/videos \
    --validation-annotations /absolute/data/wts_eval/annotations.json \
    --validation-media /absolute/data/wts_eval/videos \
    --output-dir "$SFT_DATA_ROOT" --seed 42 \
    --system-prompt 'Answer the multiple-choice question with its letter.'
```

Optional `--train-limit` and `--validation-limit` select deterministic subsets. All source rows and video bytes are checked for cross-split ID/content overlap **before** subsetting. Outputs are `train.jsonl`, `validation.jsonl`, `media_inventory.jsonl`, and `manifest.json`; the manifest binds annotation, media-inventory, and accepted-output hashes. Different or incomplete existing outputs are refused. Media remains at its recorded absolute location and must stay unchanged and accessible.

Validate every selected row through the real processor before training:

```bash
export QLORA_CONFIG=cosmos_framework/posttrain/configs/sft_qwen3_vl_8b_qlora.py
python -m cosmos_framework.posttrain.entrypoints.sft \
    --config "$QLORA_CONFIG" --validate-rows
```

## Run the native canary and matched control

On the allocated A100 80GB, first run the real kernel test. A skipped CUDA test is not qualification:

```bash
python -m pytest --pyargs \
    cosmos_framework.posttrain.tests.test_model_qwen3_vl_qlora_gpu -q
```

Then run a three-update functional canary through the native trainer:

```bash
torchrun --standalone --nproc-per-node=1 \
    -m cosmos_framework.posttrain.entrypoints.sft --config "$QLORA_CONFIG" -- \
    job.name=qlora_canary_seed42 \
    rl.max_steps=3 rl.lr_schedule.cycle_steps=3 rl.optimizer.warmup_steps=1 \
    rl.grad_accum=1 rl.save_every=1 \
    rl.validation.every_steps=1 rl.validation.max_samples=2
```

Run the equivalent control with `--config cosmos_framework/posttrain/configs/sft_qwen3_vl_8b_lora.py`, a distinct `job.name`, and identical overrides. Its frozen base is BF16. Both recipes use rank 16, alpha 32, FP32 adapters and fused AdamW, BF16 compute, SDPA, and full native non-reentrant activation checkpointing. They preserve native media processing and assistant-token loss masks.

The allowlist contains 252 language attention/MLP matrices; vision, projectors, embeddings, output head, and normalization parameters remain frozen BF16. NF4 uses packed UINT8 storage, block size 64, and nested scale quantization. Optimizer membership is exactly the 504 adapter tensors. A failed capability or kernel check has no dense fallback.

The default comparison recipe is 300 updates, accumulation 8, sequence limit 4096, and image area budget 65,536–262,144 pixels. The pinned Transformers 5.8.1 Qwen3-VL video processor bounds the total `T × H × W`: eight frames use budgets 32,768–524,288, equivalent to 4,096–65,536 pixels per frame before resize rounding. Change both total budgets when changing frame count, and record the actual resized dimensions and visual-token count. A short canary changes those schedule/batch settings and is a separate experiment. Do not extend it by changing its horizon during exact resume.

## Experimental performance controls

These opt-in controls are implemented for the single-device native adapter path. **Their new GPU/WTS numerical, reload/resume, quality, and speed qualification is pending.** Configuration acceptance and CPU tests do not establish accelerator support or speedup. The completed 300-update results below used the original defaults; their measurements and unsatisfied release quality gate are unchanged.

Append selected dotlist overrides after `--` in the native canary command above. Use a fresh job/output for each recipe, keep the data, processor, media limits, effective batch and schedule matched, and qualify individual changes before combining them.

| Native override                                     | Default     | Experimental behavior                                                                                                                                                                                        |
| --------------------------------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `model.model_config.activation_checkpointing=none`  | `full`      | Disable activation recomputation, trading additional memory for a possible speed benefit. `full` retains non-reentrant checkpointing; `selective` uses the recorded native SDPA/matrix-multiply save policy. |
| `model.model_config.sft_logits_mode=supervised`     | `full`      | Run the complete multimodal backbone, then project only the positions selected by the existing shifted labels, attention masks and validation weights. The ordinary full-logit API remains available.        |
| `model.model_config.adapter_compute_dtype=bfloat16` | `float32`   | Use BF16 adapter matrix arithmetic through differentiable casts. Adapter parameters, their gradients and optimizer moments remain FP32; packed base storage is unchanged.                                    |
| `model.model_config.compile_language_model=true`    | `false`     | Compile native language blocks with Inductor, dynamic shapes and `fullgraph=False`, outside any activation-checkpoint wrapper. Vision and the output head remain outside this compilation scope.             |
| `rl.preprocessing_cache_bytes=1073741824`           | `0`         | Example 1 GiB CPU cache budget: one quarter for processed samples and three quarters for complete video-processor outputs, including frame metadata and timestamps.                                          |
| `rl.preprocessing_prefetch=true`                    | `false`     | Use one spawned CPU worker to prepare the current optimizer update's rows in order while training consumes them.                                                                                             |
| `rl.micro_batch_size=2 rl.grad_accum=4`             | `1` and `8` | Keep eight examples per update with fewer microbatches. `rl.micro_batch_size=4 rl.grad_accum=2` is another candidate. Padding, memory peaks and floating-point reductions require separate checks.           |

The model options and selective policy are defined by [Qwen3VLModelConfig and native_adapter_execution_policy](../../cosmos_framework/posttrain/config/model.py). The implementations are [Qwen3VLPolicy.forward_sft_batch and build_qwen3_vl_model](../../cosmos_framework/posttrain/model/qwen3_vl.py) and [NativeLoRAMixin.adapter_delta](../../cosmos_framework/utils/generator/lora.py). Compilation permits graph breaks; [guard_native_compiled_call](../../cosmos_framework/posttrain/utils/native_compile.py) makes compiler errors and recompile-limit fallback fatal during invocation. Compilation counters do not prove complete graph capture or fusion. This work does **not** introduce a fused NF4 training kernel.

[build_sft_preprocessing](../../cosmos_framework/posttrain/data/packer/sft_preprocess.py) keys cache entries by processor/target semantics and verified media content, returns independent CPU copies, and rejects preprocessing that consumes RNG state. Cache contents are disposable and rebuilt after resume. The byte limit covers retained cache entries, including tensor backing storage; queued results, decoder buffers, worker runtime and filesystem cache require additional host memory. [train_step](../../cosmos_framework/posttrain/entrypoints/sft.py) reads only the current accumulation window. The [trainer's input-window and checkpoint guards](../../cosmos_framework/posttrain/roles/trainer/llm.py) require its optimizer update and ordered consumption to finish before publishing a checkpoint; failed or unfinished prefetch cannot publish an advanced data cursor.

Changing model execution controls writes a `cosmos.adapter-model.v2` identity containing the complete execution policy and observed math-precision settings. Unchanged model defaults retain the legacy identity. Batch and preprocessing settings are bound separately to the training identity, while caching preserves the processor/masking identity. [execution_config_from_identity and validate_execution_policy_identity](../../cosmos_framework/posttrain/model/quantization/loader.py) reconstruct and validate recorded model options before loading; missing or incompatible settings fail. Reloading an optimized artifact must preserve its recorded policy, and exact resume must also preserve its training configuration.

For a bounded diagnostic trace, set these variables before the native launch:

```bash
export COSMOS_SFT_PROFILE_DIR="$QLORA_WORK/profiles/new-trial"
export COSMOS_SFT_PROFILE_WAIT=2
export COSMOS_SFT_PROFILE_WARMUP=1
export COSMOS_SFT_PROFILE_ACTIVE=2
```

[NativeStepProfiler](../../cosmos_framework/posttrain/utils/training_profile.py) records one window measured in optimizer updates, with preparation, forward, loss, backward and optimizer ranges. It writes runtime settings, a compressed Chrome trace and an operator summary beneath `rank_0`; an existing rank directory is refused. Leave `COSMOS_SFT_PROFILE_DIR` unset to disable profiling. Trace timings are diagnostic: measure comparative throughput in separate untraced runs, including compilation/loading costs and actual GPU and host peaks.

## Reload, resume, and portable artifacts

Run outputs include a timestamped attempt beneath the configured project/group/job. Use the committed checkpoint path reported by the native run; full checkpoints are under its `checkpoints/iter_*` directory. Saves occur at completed optimizer-update boundaries, through the existing synchronous local DCP commit/recovery path.

| Operation                    | Contract                                                                                                                                                                                                                                                                                                                            |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Exact resume                 | Relaunch the same native command and receipt with `rl.resume=auto`, or `rl.resume=/absolute/run/checkpoints/iter_000002`. Keep source, processor, data, schedule/horizon, optimizer, accumulation, validation policy, and world size unchanged. Full optimizer, scheduler, RNG, data cursor, and update-control state are required. |
| Adapter reload               | Reconstruct the recorded quantized base, then load adapters without restoring training state. Fresh-process controlled logits/loss are the initial numerical-parity target.                                                                                                                                                         |
| Portable adapter publication | Publish adapters and identity metadata only; this artifact cannot resume training. It neither merges nor requantizes the base.                                                                                                                                                                                                      |
| Dense/merged exports         | Dequantized `Q(W)` plus the adapter, original higher-precision `W` plus the adapter, and subsequent requantization are different numerical models. These exports and serving-backend compatibility require separate qualification.                                                                                                  |

Publish a committed checkpoint as a portable adapter artifact:

```bash
python -m cosmos_framework.posttrain.checkpoint.adapter \
    --source /absolute/run/checkpoints/iter_000003 \
    --output-iter /absolute/export/checkpoints/iter_000003
```

The command prints the committed artifact location in the sibling `checkpoint_artifacts/iter_*` namespace. It does not advance the training latest pointer. Incomplete checkpoints, incompatible identities, unsupported topology, and adapter-only artifacts presented as resumes fail explicitly.

The [integration driver](../../cosmos_framework/posttrain/tests/qlora_integration.py) evaluates the native model; it contains no training loop. Run its `probe` subcommand in a fresh torchrun process for the saved DCP checkpoint and again for its portable artifact, using the same fixed image and video row. Supply an existing image, or omit `--image` on the first probe and reuse its generated `video_first_frame.png`:

```bash
torchrun --standalone --nproc-per-node=1 \
    -m cosmos_framework.posttrain.tests.qlora_integration probe \
    --config "$QLORA_CONFIG" --checkpoint /absolute/committed/checkpoint \
    --data "$SFT_DATA_ROOT/validation.jsonl" --image /absolute/probes/frame.png \
    --output /absolute/probes/new-output --require-updates
```

It checks text, image, and real video losses/logits, media sensitivity, trainable coverage, adapter changes, and frozen-base fingerprints. Use `compare-probes` for reload parity and `compare-checkpoints` for interrupted-versus-uninterrupted resume; both require explicit predeclared `--atol` and `--rtol`. Resume comparison also checks RNG and data/control state exactly. Passing a probe alone does not establish held-out task quality or total training memory savings.

## Qualification evidence

The [machine-readable experiment report](evidence/native_qlora_wts_c5b8d49.json) contains the completed training and four-arm evaluation measurements, source/runtime identities, per-update metrics, and measurement limitations.

The experiments tested commit `c5b8d49bcc219d1cceb76bee42d4eb20daa766f7`, installed from wheel SHA256 `afbeb228bd42ba9582e53d427c20ba59d0b166706ffd98ccaa6ed73d8f149047`. Each run used one NVIDIA A100-SXM4-80GB and one torchrun process. Runtime: Python 3.12.12, Torch 2.11.0+cu128 (source commit `70d99e998b4955e0049d13a98d77ae1b14db1f45`), CUDA 12.8, Transformers 5.8.1, bitsandbytes 0.49.2, and TorchCodec 0.11.0+cpu. The host reported driver 535.129.03; the loaded CUDA compatibility driver library was `libcuda.so.570.124.06`. Both training arms used strict deterministic algorithms.

All four evaluation arms verified the same pinned base revision recorded above, source-weight manifest digest `69368621649dca3287fe9587bcac3d1d2fb6380974d9b59ade8c80aaa9879daf`, and processor/template manifest digest `9d674c4ab12ba55505f763912e1fadf3d8660125dcb774c20462743582d8b180`. The resolved contracts matched 252 language targets, 504 FP32 adapter tensors containing 43,646,976 trainable parameters, rank 16/alpha 32, and full non-reentrant activation checkpointing. NF4 used nested quantization, UINT8 packed storage and BF16 compute; excluded base components remained BF16, and optimizer moments were FP32. The matched control kept the same targets in frozen BF16.

### Exact reload and resume canary

The controlled canary used **four optimizer updates, microbatch one, accumulation two, seed 42**, with saves at updates two and four. Actual WTS sequences were at most 354 tokens, with eight video frames resized to 192 × 320 and 240 visual tokens.

- All three real CUDA tests passed: NF4 forward/input backward and adapter updates, plus exact repeated BF16 Flash SDPA forward/backward for language and vision shapes.
- The optimizer contained exactly the 504 FP32 adapter tensors. All 252 A and 252 B factors updated; frozen weights, buffers, packed data and nested quantization state passed fingerprint checks at save/probe boundaries.
- A fresh process resumed from update two and completed updates three and four. Final adapter/optimizer tensors, scheduler, RNG, data cursor, and update/validation controls matched the uninterrupted run. Only storage-publication identities were excluded.
- Separate fresh-process DCP and portable-artifact reloads produced identical controlled text, image and video logits/losses on the reconstructed quantized base. Resume and reload both used `atol=0, rtol=0`, with maximum differences zero. Media-sensitivity checks passed.

These are four-update parity results, not evidence of a separate 300-update interrupted/resumed comparison. The [archived canary report](https://github.com/ramanathan831/cosmos-framework/blob/59d4ddb383c20960e4ebb214ac6c90c095475b99/docs/post_training/native_qlora.md#qualification-evidence) retains its loading, training, checkpoint and resume memory measurements.

### Completed 300-update training comparison

Both native recipes completed **300 optimizer updates, seed 42, microbatch one, accumulation eight**, processing 2,400 training examples. They used the same accepted data, processor/loss masks, adapter initialization, optimizer, 300-update schedule with 30 warmup updates, and validation/checkpoint boundaries every 50 updates. The jobs overlapped on different A100 nodes; frozen/trained evaluation pairs subsequently ran sequentially within each precision. Filesystem/cache state was uncontrolled.

All 300 logged losses and gradient norms were finite. Final full checkpoints were committed at update 300; the strict cross-arm comparator verified zero skipped update attempts, the same final data cursor, optimizer membership/static options, schedule, and update-control policy. Frozen-base checks passed at saves and after evaluation; evaluation left adapters unchanged.

| Training lifecycle measurement               | NF4 LoRA | BF16 LoRA |
| -------------------------------------------- | -------: | --------: |
| CUDA allocated peak, GiB                     |    7.769 |    17.367 |
| CUDA reserved peak, GiB                      |    8.201 |    18.078 |
| NVML run-owned process-sum sample peak, GiB  |    9.199 |    19.076 |
| NVML whole-device sample peak, GiB           |    9.883 |    19.760 |
| Host process-tree RSS-sum sample peak, GiB   |    5.903 |     5.957 |
| GNU-time maximum individual-process RSS, GiB |    6.428 |     5.912 |
| Mean optimizer-update time, s                |   23.498 |    21.848 |
| Supervised tokens/s during updates           |    0.681 |     0.732 |
| Examples/s during updates                    |    0.340 |     0.366 |
| Full native-process elapsed time, s          |     8246 |      7865 |
| Final full adapter checkpoint, GiB           |    0.492 |     0.492 |
| Portable adapter artifact, GiB               |    0.163 |     0.163 |

Memory uses GiB (`2^30` bytes) and covers loading, training, validation, and checkpoint writes; separate evaluation processes are excluded. The observed run-owned GPU sample peaks differed by **9.877 GiB**. This exceeds the proposed 6 GiB benefit for this seed's observed samples, but does not establish a difference in unsampled true peaks or the required three-seed release result.

| CUDA allocated peak by phase, GiB | NF4 LoRA | BF16 LoRA |
| --------------------------------- | -------: | --------: |
| Loading                           |    7.065 |    16.505 |
| First optimizer update            |    7.460 |    17.057 |
| Later observed updates            |    7.769 |    17.367 |
| Checkpoint writes                 |    7.460 |    17.057 |

CUDA peaks come from synchronized PyTorch allocator counters. NVML and host RSS used nominal one-second sampling, with 7,489 NF4 and 7,133 BF16 valid run-owned GPU samples, no rejected UUIDs or unavailable owned-memory values, and maximum whole-run gaps of 1.296 s and 2.044 s respectively. Sampled peaks are lower bounds and can miss short spikes; setup intervals with no external samples remain unknown. Whole-device usage can include unrelated allocations. Do not subtract independently timed allocator and NVML maxima to infer external allocations.

Summed host RSS can double-count shared pages. GNU-time RSS is a process-lifetime high-water mark, not a phase-local peak; it shows a higher NF4 host peak than the sampled RSS sum. Host RSS/PSS/HWM cover process memory: **unmapped filesystem page-cache memory was not measured**. Media hashing and decoding can populate material cache memory outside those figures.

Update timing includes instrumentation and excludes loading, validation and checkpoint writes. The fixed post-warmup window, updates 31–300, averaged 23.692 s for NF4 and 21.916 s for BF16. Supervised-token throughput counts assistant answer/end-token targets, not input or visual tokens. Native validation on the same first 64 held-out rows/128 supervised tokens had initial/final CE **0.374107 / 0.201907** for NF4 and **0.315462 / 0.217662** for BF16; this is distinct from the 256-row task diagnostic below.

### Four-arm held-out diagnostic and release status

All four arms completed the same 256 preselected rows from a 2,676-row held-out pool, with subset seed 2026 and training/model seed 42. The strict comparator checked source/model/processor identities, initial factors, permitted quantization differences, exact processed inputs and masks, score digests, and each trained model's own step-300 portable artifact. The frozen arms used the exact initial zero-B adapters.

Questions had two, three or four choices: 16, 72 and 168 rows respectively. Restricted-choice accuracy selects the highest-logit letter among the actual choices at the first assistant answer position. Full-vocabulary first-token accuracy matched it in all four arms, and valid-letter rate was 100% throughout. These are teacher-forced next-token measurements, not free-generation accuracy. Native token-weighted CE preserves the answer and assistant end-token mask, totaling 512 supervised tokens.

| Arm          | Correct / 256 | Choice and first-token accuracy | Token-weighted CE | Answer-token NLL | Restricted-choice NLL |
| ------------ | ------------: | ------------------------------: | ----------------: | ---------------: | --------------------: |
| Frozen NF4   |           131 |                        51.1719% |          1.067917 |         2.135813 |              2.135810 |
| Trained NF4  |           229 |                        89.4531% |          0.153532 |         0.307051 |              0.306780 |
| Frozen BF16  |           144 |                        56.2500% |          1.123509 |         2.246974 |              2.246972 |
| Trained BF16 |           230 |                        89.8438% |          0.147372 |         0.294720 |              0.294388 |

Tie counts in the table's row order were 7, 0, 4 and 2 for both accuracy readouts. Ties use canonical choice order for restricted scoring and the lowest token ID for full-vocabulary scoring. The selected rows covered 133 exact video-content hashes; related events or camera views may require broader clustering for release uncertainty estimates. Evaluator source SHA256 was `3b16133a77643f1585470e4a38a5fdfd5d1709eb679a22ee30db4fb928142ac9`.

Observed evaluation sequences ranged from **304 to 365 tokens** (median 342, p95 356). Video pixel tensor shapes were `[960,1536]` for 252 rows and `[864,1536]` for four rows. The processor requested eight frames with the total pixel budgets documented above; decoded grid values/frame counts were not recorded. Actual training input-length and frame distributions were also not recorded. The configured 4096-token ceiling and these short evaluation sequences do not qualify sustained 4096-token training.

**The release quality gate is not satisfied.** Trained NF4 improved the diagnostic accuracy over frozen NF4 by 38.2813 percentage points, but scored one fewer correct answer than trained BF16 (−0.3906 percentage points). Its token-weighted NLL was **4.18% higher** than trained BF16, already exceeding the proposed 2% relative margin at the point-estimate level. This is an observed quality regression in addition to incomplete release sampling; the required one-sided confidence bound has not been evaluated.

The [predeclared release protocol](qlora_native_vlm_design.md#reproducible-benchmark-and-release-evidence) additionally requires paired training seeds 17, 42 and 123, full held-out evaluation with clustered paired uncertainty, and relevant text/image/video capability regressions. Those remain unqualified. The implementation remains experimental; these results do not establish support for smaller GPUs, other distributed strategies, merged exports, or serving backends.
