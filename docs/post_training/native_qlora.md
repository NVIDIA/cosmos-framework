# Native QLoRA for Qwen3-VL

This experimental path trains native LoRA adapters on a frozen NF4 language base through `cosmos_framework.posttrain.entrypoints.sft`. It is **awaiting real accelerator, reload/resume, memory, and held-out quality qualification**. Passing configuration or CPU tests does not establish production support.

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

Set the determinism variables **before** starting Python or torchrun. Native SFT initializes CUDA before its setup-time determinism check. Do not change the driver library search path to conceal a kernel-loading failure.

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

The default comparison recipe is 300 updates, accumulation 8, sequence limit 4096, image area budget 65,536–262,144 pixels, and eight video frames with area budget 4,096–65,536 pixels. A short canary changes those schedule/batch settings and is a separate experiment. Do not extend it by changing its horizon during exact resume.

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

Keep the matched BF16 and NF4 arms identical in source revision, data/splits, processor/template, LoRA targets/rank, media/sequence limits, effective batch, schedule, optimizer, and seeds. Record unavoidable differences. Measure loading, first-step, steady-state, and checkpoint-write peaks: per-rank CUDA allocated/reserved memory, device/process memory outside the allocator, and host RSS. Also record throughput, elapsed time, checkpoint size, held-out task quality, and baseline-capability regressions.

Declare the acceptable quality regression and required memory benefit before the release comparison. Retain hardware/driver/package versions, build receipts, resolved configurations, dataset manifests, kernel results, fresh-process reload evidence, and controlled resume comparisons. The [design](qlora_native_vlm_design.md#reproducible-benchmark-and-release-evidence) describes the full gate; no generic percentage saving or support for smaller GPUs is claimed.
