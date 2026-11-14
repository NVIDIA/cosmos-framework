# Native QLoRA for Qwen3-VL

The separate [Alpamayo-R1 trajectory](native_qlora_alpamayo.md) and
[GR00T-N1.7 LIBERO action](native_qlora_gr00t.md) integrations reuse the shared
quantization/artifact machinery through native task objectives. Their model versions,
runtime pins, task metrics and qualification evidence are separate from this Qwen recipe;
Qwen results do not establish support for either family.

This path trains native LoRA adapters on a frozen NF4 language base through `cosmos_framework.posttrain.entrypoints.sft`. The current implementation is sealed `09e99c3`, with accepted current-build recovery/reload checks and scoped effective-base BF16 dense-export evidence. Completed resident-evaluation timing and update-250-to-300 recovery belong to `dadfc89`; historical results do not authorize cross-build resume. Quality preservation, quality improvement and a QLoRA time-to-target advantage remain unestablished.

The historical cache/prefetch recipe reduced the median trial mean for NF4 update time by **42.7%** across three paired trials while preserving the original final adapter/optimizer state and held-out score rows exactly. Both optimized precision arms completed 300 updates, fresh-process reload checks and frozen/trained 256-row evaluations.

**The release quality gate is not satisfied.** The original trained NF4 token-NLL point estimate was 4.18% worse than BF16, exceeding the unchanged 2% margin. Faster preprocessing preserved that result. Subsequent controlled and selected three-seed development comparisons also failed the required NLL uncertainty criterion; the selected rank-32 upper bound was 2.101%. Completed final regression scoring also failed the NLL uncertainty gate (controlled 3.972%, selected 5.879% upper bounds); selected accuracy preservation failed as well. The formerly untouched diagnostic contains only one recording group, so its independent-group uncertainty is nonestimable.

The [continuation qualification checklist](native_qlora_qualification.md) maps the implementation to its evidence and remaining gates. Its data audit found seven related recording families across the original train/validation boundary and only one untouched validation family. That family has since been scored: none of these data can now serve as an untouched final test. The old 256 examples remain an exposed regression set; new settings were selected on a separately withheld development partition. The earlier continuation's [final results](native_qlora_qualification.md#completed-final-regression-and-untouched-diagnostic) and [allocation accounting](native_qlora_qualification.md#completed-continuation-accounting) retain failures and unchanged thresholds. The subsequent runtime campaign's [six timing runs](native_qlora_qualification.md#completed-combined-execution-timing) are also terminal. Quality preservation and independent confirmation remain unresolved.

The initial candidate is `Qwen/Qwen3-VL-8B-Instruct` at revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`, on **one Linux x86-64 A100 80GB, one torchrun process**. FSDP/HSDP, additional GPUs, offload, MoE, DoRA/PiSSA and other quantizers are rejected or unqualified. Effective-base BF16 dense export has the [scoped conversion evidence](#effective-base-dense-export) below; other conversions, serving-backend compatibility and inference latency remain unqualified. See the [design and release evidence requirements](qlora_native_vlm_design.md).

**Historical scope decision, 2026-11-04:** beating equally optimized BF16 LoRA by 10% in time-to-quality or improving its task accuracy was classified as optional research work. The following reaffirmation updates the product objectives; this historical decision is retained with its original results. Release still requires substantial measured memory savings, the existing paired/grouped one-sided 95% noninferiority bounds (at most 2% relative token-NLL increase and one percentage point of accuracy loss), and reliability for every advertised configuration. Avoidable overhead must be addressed and the remaining memory–runtime trade-off reported. No historical experiment becomes a pass, and the failed quality bounds and missing independent evaluation groups remain unresolved. See the [current criteria and historical scope](native_qlora_qualification.md#prospective-release-scope).

**Current product objectives, reaffirmed 2026-11-05:** improved task quality and lower training time to a fixed, predeclared quality target remain objectives of this continuation. The preceding scope decision and all historical pass/fail results are preserved; substantial memory savings alone satisfy neither objective. The quality-preservation guards remain mandatory and unchanged. Training efficiency is measured separately from inference latency, which has no serving qualification here.

## Prepare a sealed runtime

Use uv `0.11.29` and Python 3.12. The [posttrain environment](../../docker/posttrain/environment/pyproject.toml) and its [lock](../../docker/posttrain/environment/uv.lock) select Torch `2.11.0+cu128`, Transformers `5.8.1`, and bitsandbytes `0.49.2`. The x86-64 selection uses CPU TorchCodec `0.11.0+cpu`; the host also needs working FFmpeg libraries for video decoding.

The shared trainer retains the original elementwise gradient-finiteness checks.
The grouped CUDA candidate passed correctness tests but missed the GR00T study's
predeclared complete-workflow improvement gate and was removed from the default
implementation. Its historical CUDA and timing evidence remains in the
[GR00T qualification record](native_qlora_gr00t.md). This decision changes neither
the selected Qwen numerical trajectory nor its historical qualification results.

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

For qualification, also audit related recording families using the [grouped split protocol](../../tools/native_qlora_qualification/README.md) and [split builder](../../tools/native_qlora_qualification/split.py) before training. Exact ID/video overlap checks alone do not prevent related clips or trajectories from crossing splits. The continuation used source-bound, disjoint recording groups for 4,611 training rows and 584 development rows; its previously exposed 256 rows remain a regression set. The last previously untouched recording family supplied 32 diagnostic rows and has since been scored. None of these sets is an untouched final test, and one family cannot establish independent group uncertainty.

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

The default allowlist quantizes 252 language attention/MLP matrices; vision, projectors, embeddings, output head, and normalization parameters remain frozen BF16. NF4 uses packed UINT8 storage, block size 64, and nested scale quantization. Optimizer membership is exactly the 504 adapter tensors. Optimizer moments remain FP32; 8-bit and paged optimizers are not part of this path. A failed capability or kernel check has no dense fallback.

The default comparison recipe is 300 updates, accumulation 8, sequence limit 4096, and image area budget 65,536–262,144 pixels. The pinned Transformers 5.8.1 Qwen3-VL video processor bounds the total `T × H × W`: eight frames use budgets 32,768–524,288, equivalent to 4,096–65,536 pixels per frame before resize rounding. Change both total budgets when changing frame count, and record the actual resized dimensions and visual-token count. A short canary changes those schedule/batch settings and is a separate experiment. Do not extend it by changing its horizon during exact resume.

## Reproduce the selected development recipe

The selected WTS settings are rank 32/alpha 64, disabled activation checkpointing, dropout and answer smoothing both disabled, and 3 GiB preprocessing caches in both precision arms. These settings were selected on `149c2cf`; resident execution and mature recovery were subsequently measured on `dadfc89`. These are explicit workload-specific overrides; generic rank-16 recipe defaults remain unchanged. The historical [three-seed full-checkpointing, 1 GiB study](native_qlora_qualification.md#completed-selected-three-seed-development-comparison) remains separate: its crossed NLL uncertainty bound fails the unchanged 2% gate. Neither study completes a same-budget best-fit search or establishes a universal default.

The [earlier sealed build](native_qlora_qualification.md#final-native-build-ordinary-execution), `4cd8988`, passed its real-kernel/artifact gate and both precision arms' native four-update interruption, recovery and fresh multimodal checkpoint/portable reload checks. NF4 uses ordinary `bnb` execution; the BF16 control retains its dense base. Both use full logits, FP32 adapters and fused AdamW, with one preprocessing worker. These checks preserve the failed quality bounds and do not qualify rank-32 long-context training. The subsequent [completed cache comparison](native_qlora_qualification.md#completed-300-update-cache-comparison) selects a scoped **3 GiB WTS cache override for both NF4 and BF16**: same-precision update-300 adapter/optimizer/RNG/data/control comparisons passed, and checked endpoint times fell by 22.03%/24.69% in the respective fixed-order pairs. This is one seed and one pair per precision, without a timing confidence interval or a new quality-gate pass.

For new training, use sealed native `09e99c3`, its matching workflow, accepted exact-build gate and the pinned processor above. Reproducing a historical result requires its original sealed build and inputs; this does not permit cross-build resume. Set `SFT_DATA_ROOT` to the source-bound grouped training directory containing the fresh 4,611-row `train.jsonl` and a `validation.jsonl` copy of the 584-row development partition. Preserve the corresponding manifests; do not substitute the earlier raw conversion's split. The [qualification runner](../../tools/native_qlora_qualification/README.md#native-experiments) supplies preflight, measurement, portable publication, fresh reload and complete development scoring around this native launch. The training command alone does not perform that qualification. The command below selects the WTS execution settings with ordinary 64-row loss validation. For full 584-row resident development scoring, use the [native qualification runner](../../tools/native_qlora_qualification/README.md#native-in-process-evaluation) with `--evaluation-mode native --validation-target "$TARGET"` and the exact `--score-data` and `--split-manifest` sources. Ordinary validation alone does not enable full WTS scoring.

```bash
export SFT_DATA_ROOT=/absolute/work/qlora/grouped-training
export QLORA_SEED=17
export PYTHONHASHSEED="$QLORA_SEED"
torchrun --standalone --nproc-per-node=1 \
    -m cosmos_framework.posttrain.entrypoints.sft --config "$QLORA_CONFIG" -- \
    job.name="qlora_rank32_seed${QLORA_SEED}" \
    model.model_config.adapter.rank=32 model.model_config.adapter.alpha=64 \
    model.model_config.adapter.initialization_seed="$QLORA_SEED" \
    model.model_config.activation_checkpointing=none \
    model.model_config.adapter_compute_dtype=float32 \
    model.model_config.quantized_linear_execution=bnb \
    model.model_config.sft_logits_mode=full \
    model.model_config.compile_language_model=false \
    rl.seed="$QLORA_SEED" rl.deterministic=true \
    rl.max_steps=300 rl.lr_schedule.cycle_steps=300 \
    rl.optimizer.lr=0.0001 rl.optimizer.warmup_steps=30 \
    rl.micro_batch_size=1 rl.grad_accum=8 rl.max_length=4096 \
    rl.preprocessing_cache_bytes=3221225472 rl.preprocessing_prefetch=true \
    rl.preprocessing_workers=1 \
    rl.checkpoint_format=dcp rl.save_async=false rl.save_every=50 \
    'rl.checkpoint_components=["model","optimizer","training_state"]' \
    rl.emergency_checkpoint_components=full rl.keep_optim_last=0 rl.keep_model_last=0 \
    rl.validation.enabled=true rl.validation.before_training=true \
    rl.validation.every_steps=50 rl.validation.max_samples=64 \
    rl.validation.batch_size=1 rl.validation.seed="$QLORA_SEED" \
    rl.validation.save_checkpoint=true \
    'rl.validation.checkpoint_components=["model","optimizer","training_state"]' \
    rl.processor_kwargs.videos_kwargs.num_frames=8 \
    rl.processor_kwargs.videos_kwargs.size.shortest_edge=32768 \
    rl.processor_kwargs.videos_kwargs.size.longest_edge=524288
```

Run the BF16 control with `sft_qwen3_vl_8b_lora.py`, a distinct output name and identical overrides. Set the environment, training, adapter-initialization and validation seeds together before each launch. NF4 alone uses the `bnb` quantized execution path; BF16 retains its dense base. Eight processed temporal slots and a 4,096-token limit are configuration values, not measured frame uniqueness or a training-length distribution.

For historical reproduction of the selected three-seed study, use its sealed `55e7f14` build and original source identities with these settings in **both** arms:

```bash
rl.preprocessing_cache_bytes=1073741824
model.model_config.activation_checkpointing=full
```

That historical study reused its audited seed-42 pair and ran fresh seeds 17 and 123. The recorded native `149c2cf` engineering blocks and scheduled-scoring runs are separate evidence; changing this command does not relabel the historical full/1 GiB results as none/3 GiB results.

Use a new output name. The generic recipe defaults, including full checkpointing, remain unchanged. Exact resume must retain the checkpoint's original cache capacity, preprocessing identity and activation-checkpointing policy: neither a 1 GiB-to-3 GiB cache switch nor a full-to-none policy switch is authorized during resume. The earlier [cache evidence](evidence/native_qlora_cache_capacity_4cd8988.json) retains its own recovery checks, failed first BF16 block, complete retry and CPU payload audit. Cache capacity bounds retained entries, not total worker or host memory.

The [two valid seed-42 engineering blocks](native_qlora_qualification.md#completed-reverse-block-and-execution-choice) measured **21.37%/22.06% lower NF4 complete experiment time** with checkpointing disabled; the matched BF16 control also improved by **18.27%/17.20%**. Selected NF4 retained a **9.828 GiB sampled GPU-memory advantage**, while remaining **4.21%/3.67% slower** than selected BF16. The [selected-policy mature recovery](native_qlora_qualification.md#selected-policy-mature-recovery-on-native-149c2cf) subsequently passed own update-250-to-300 checks for both arms. This promotes only the measured rank-32, 3 GiB-cache WTS execution variant. Two same-seed blocks provide no timing confidence interval, independent-seed result, long-context qualification or time-to-quality improvement. The failed quality bounds and need for new independent evaluation data remain unchanged.

Native `149c2cf` adds committed-boundary pauses for scheduled development scoring. Its [four-update accelerator qualification](native_qlora_qualification.md#committed-boundary-pauses-on-native-149c2cf) covers both precision arms, interruptions, exact resumed state and fresh portable reload. The [completed two-seed scheduled study](native_qlora_qualification.md#completed-scheduled-development-ttq-on-native-149c2cf) used the [existing runner](../../tools/native_qlora_qualification/README.md#scheduled-full-development-scoring) with the fixed full-development target every 50 updates. At seed 17 both arms first passed at update 150: NF4 took 2,912.486 seconds versus BF16 2,866.243 seconds, **1.61% longer**. At seed 123 both completed 300 updates without a scheduled pass. These exposed-development observations do not establish a time-to-quality advantage, exact equal-compute quality or a new quality-gate pass. Each segment restarts the model/cache; keep build and execution identity unchanged during exact resume.

The [completed CPU input census](native_qlora_qualification.md#complete-ordinary-source-input-census) processed all **4,611 unchanged training-source rows** with the pinned native processor and 3 GiB cache: **317–362 tokens**, two supervised tokens per row, video grid `[4,12,20]` and pixel tensor shape `[960,1536]`. Source, media and processor checks and parent/worker RNG checks passed around the full pass, including its final three-row window. This measures the ordinary source's preprocessing geometry; it does not establish that every row was trained, unique decoded frame counts, or current rank-32 long-context/larger-media GPU training support. The [tracked development-cost total](native_qlora_qualification.md#tracked-development-allocation-cost) includes failures and prior search allocations, with unmeasured acquisition/local CPU work left explicit.

## Native in-process development scoring

Native in-process evaluation has completed four-update GPU qualification on `2f7a66d` in both precision arms at dropout zero and 0.05. The measured `dadfc89` package also passed its own 14 real CUDA cases plus higher-precision native regression, and both four-update, dropout-0.05 diagnostic bridges with exact state/score comparisons and fresh DCP/portable reload. Its [seed-17 comparison](native_qlora_qualification.md#seed-17-resident-evaluation-comparison) adds full-development timing and exact training/state/score parity through update 150; the [completed seed-123 comparison](native_qlora_qualification.md#completed-seed-123-resident-comparison) extends that evidence through update 300. Both arms also passed [resident update-250-to-300 recovery](native_qlora_qualification.md#completed-resident-mature-recovery), including exact optimizer/RNG/data/evaluation state and fresh multimodal reload. The dropout hypothesis is rejected below. The interruption suites remain prior-build evidence, and the completed `149c2cf` results above retain their original restart-based scope. The [qualification runner](../../tools/native_qlora_qualification/README.md#native-in-process-evaluation) accepts:

```bash
--evaluation-mode native --validation-target "$TARGET"
```

The target retains the frozen 584-row development source, its manifest hashes, the grid `[50,100,150,200,250,300]`, and the joint point threshold of 540 correct and token CE at most `0.12357161758062493`. This is separate from the release uncertainty gates. Native scoring uses the same per-example scorer as fresh-process evaluation, with a separate persistent bounded cache. Training keeps its model and cache resident. Source/media/processor identities, RNG, module modes and data cursors are checked or preserved; exact resume retains the evaluation policy and deterministic checkpoint journal.

The first scheduled pass stops only after full scoring, a committed resumable checkpoint and checkpoint readback. Availability time includes runner setup, model loading, training, evaluation and checkpointing. Final target-index serialization, process cleanup and fresh export/reload remain in separately reported completion costs. Acquisition, build preparation and scheduler waiting remain outside the target clock. At seed 17, both modes first passed at update 150 with identical scores: resident evaluation reduced NF4 target time from **49.53 to 30.37 minutes**, and BF16 from **48.44 to 28.01 minutes**. This shared workflow improvement does not improve learning. Native NF4 retained **9.941 GiB lower sampled GPU-process memory** while taking **8.40% longer to target** than native BF16; different cross-precision GPU/CPU placement and the single seed limit that comparison. Sampled host RSS rose by about 2.1 GiB in each arm versus restart execution. These lower-bound memory peaks do not measure total host memory.

At seed 123, neither precision reached the target by update 300 in either mode. Resident evaluation reduced complete NF4 workflow time from **93.16 to 51.29 minutes**, and BF16 from **95.32 to 46.16 minutes**, with exact same-precision numerical/state/score comparisons. Sampled resident GPU-process peaks were **13.287 / 23.182 GiB** for NF4/BF16. The **44.94% / 51.57%** workflow reductions are shared execution improvements; target times remain null, and resident NF4's complete workflow remained **11.12% longer** than BF16 in these placements. The old interrupted attempts remain failed, and the historical quality uncertainty bounds are unchanged.

For a prospective nonstopping learning screen, direct native overrides `rl.validation.wts.stop_on_target=false` and `rl.pause_after_step=100` retain the 300-update horizon and full evaluation grid while recording observations at updates 50 and 100, even after an early pass. This is a paused screen, not completed or censored 300-update TTQ. The runner's native TTQ mode uses `stop_on_target=true`. Separately, `model.model_config.adapter.dropout=0.05` has four-update correctness evidence in both arms, including the current-build diagnostic bridges. Its retained 100-update NF4 scores improve accuracy **528 → 541 / 584**, but token CE increases **0.252%**, failing the predeclared requirement for at least 1% reduction; the hypothesis is discarded and zero dropout remains the default. Changing dropout or evaluation policy requires new training, not an exact-resume waiver. Historical quality failures and the need for independent evaluation data remain unchanged.

The first learning allocation failed private acceptance because two mount aliases spelled the same paused checkpoint differently. A private path-preservation correction passed 12 CPU checks without changing native code or scientific settings. Its retry completed all four training, export/reload and fresh-scoring cases, then Lustre refused writes with `EDQUOT` before final acceptance. Later read-only checks verify all four native/fresh 584-row score pairs and the recorded 100-update boundaries. The negative quality guard suffices to discard the hypothesis; the missing original final-case clock and memory gate are not reconstructed or passed. The quota error also interrupted the second-seed NF4 timing control after its native 300-update mode completed. Verified duplicate-installation cleanup preserved all scientific artifacts and the active build, but two subsequent timing attempts still failed before training despite nominal quota headroom. All failures and costs remain [recorded](evidence/native_qlora_native_evaluation_dadfc89.json). At that handoff, further timing and current-build mature recovery required reliable filesystem writes and no jobs remained active. The November 12 follow-up verified actual compute-side directory creation, bounded multi-file writes, fsync, rename, readback and cleanup at all five real destinations. It also corrected the admission probe to validate Slurm host ownership through the container UID namespace; one separate NUMA refusal remains a distinct failure. The successful small probe does not reserve capacity or explain the earlier quota failure. Independent final recording groups remain a separate external dependency.

## Combined longer-context and larger-video recipe

The [completed `09e99c3` check](native_qlora_qualification.md#completed-combined-long-context-and-larger-video-check) covers rank 32/alpha 64, **full activation checkpointing**, microbatch one and accumulation eight on one A100 80GB. It trains on meaningful synthetic textual ledgers paired with real video: prepared training sequences span 4,059–4,090 tokens with 1,236–1,335 supervised tokens, and video inputs use 16 processed temporal slots at 384×672. The short-WTS no-checkpointing policy is not qualified for these larger inputs.

The exact [NF4](../../tools/native_qlora_qualification/recipes/qwen3_vl_8b_nf4_rank32_long4k_video16.json) and [BF16](../../tools/native_qlora_qualification/recipes/qwen3_vl_8b_bf16_rank32_long4k_video16.json) specifications reproduce 20 native updates, checkpoints at 10/20 and fresh portable reload. Use the sealed prepared workload and matching build/kernel gate; preparation alone does not qualify training. With the deterministic runtime environment above, run the existing orchestrator inside the allocated GPU job:

```bash
PYTHONPATH="$BUILD/site:$WORKFLOW_SOURCE" \
python "$WORKFLOW_SOURCE/tools/native_qlora_qualification/runtime.py" \
    --spec "$REVIEWED_SOURCE/tools/native_qlora_qualification/recipes/qwen3_vl_8b_nf4_rank32_long4k_video16.json" \
    --workloads "$PREPARED_WORKLOADS" --profile long4k_video16_large \
    --workload-manifest-sha256 "$WORKLOAD_MANIFEST_SHA256" \
    --profile-manifest-sha256 "$PROFILE_MANIFEST_SHA256" \
    --model "$SFT_MODEL_PATH" --build "$BUILD" \
    --kernel-gate "$NEW_GATE/completion.json" \
    --output "$NEW_LONG_RUN" --time-binary "$GNU_TIME" --stage-timeout 1200
```

`REVIEWED_SOURCE` is the checkout containing the published recipe; `WORKFLOW_SOURCE` is its matching sealed `09e99c3` qualification workflow. The qualification runner requires the verified wheel overlay at `$BUILD/site` and its receipt, as described in the [tool instructions](../../tools/native_qlora_qualification/README.md); an arbitrary globally installed package is refused. Keep the working directory outside the source checkout. `PREPARED_WORKLOADS` contains the source-bound `workloads.py` outputs and manifests. Retain their exact recorded hashes for reproduction; regenerating a fixture creates separate evidence. Select the BF16 specification and a fresh output for its control. The command submits no scheduler jobs and invokes the native trainer. The separately completed boundary/accumulation recovery suite is not implied by this runtime command.

Both arms processed 160 training examples and 202,925 supervised tokens. Runtime sampled GPU-process peaks were **16.447 / 26.127 GiB NF4/BF16**; the separate recovery suites reached **22.305 / 31.977 GiB**. Fresh video reload used 4,095 tokens and 1,335 supervised tokens, with exact checkpoint/portable parity. These are sampled lower bounds on an A100 80GB, not smaller-GPU support, natural-long-task quality, unique source-frame counts or inference acceleration. Inputs beyond this measured scope remain unqualified.

## Experimental residual initialization

The opt-in `nf4_residual_svd` initializer compensates for the **actual fixed NF4 base** before training. Accelerator preparation, 100-update training, fresh-process reload and a separate completed 15-stage four-update recovery suite are recorded for the sealed `1ac6a7f` native package in the [ongoing qualification](native_qlora_qualification.md#ongoing-improvement-campaign). The first full-development pilot did not pass its locked follow-up gate; quality and complete time-to-quality qualification remain open. Ordinary zero-B LoRA remains the default.

For each of the 252 language matrices, explicit preparation approximates `FP32(source_BF16) - FP32(dequantize(actual_NF4))` with FP32 factors such that `(alpha / rank) * B @ A` approximates the residual. The actual dequantization is BF16. The solver uses a seeded randomized reduced SVD, oversampling 16 and two power iterations. It processes one source matrix and its actual dequantization at a time; it never materializes the full dense model on the GPU. The NF4 packed weights and quantization metadata remain unchanged. Recorded Frobenius errors are weight-space diagnostics; BF16 arithmetic, activation distributions and downstream layers still require model-output and task checks.

Use the sealed runtime and determinism environment above. On its owned single GPU, prepare a new immutable factor directory before launching training:

```bash
export QLORA_INITIALIZER="$QLORA_WORK/nf4_residual_rank32_seed42"
torchrun --standalone --nproc-per-node=1 \
    -m cosmos_framework.posttrain.scripts.prepare_qlora_initialization \
    --model "$SFT_MODEL_PATH" --output "$QLORA_INITIALIZER" \
    --rank 32 --alpha 64 --seed 42
export QLORA_INITIALIZER_SHA256="$(sha256sum "$QLORA_INITIALIZER/manifest.json" | cut -d' ' -f1)"
```

Append these overrides to the selected native NF4 training command, with matching rank, alpha and initialization seed:

```bash
model.model_config.adapter.initialization=nf4_residual_svd \
model.model_config.adapter.initialization_artifact="$QLORA_INITIALIZER/manifest.json" \
model.model_config.adapter.initialization_artifact_sha256="$QLORA_INITIALIZER_SHA256"
```

The manifest is committed last and binds all factor files, exact source and quantized-state fingerprints, targets, configuration, algorithm and runtime/build identity. Incomplete, changed or incompatible bundles fail. Keep the complete bundle alongside adapter and resume artifacts: this first implementation requires its verified factors during reconstruction, even when trained adapter weights will subsequently be restored. Construction and resume only read the cache; they never repeat the decomposition. Relocating an identical bundle is allowed, but changing its content, seed or runtime identity requires separate qualification.

This option uses `cosmos.adapter-model.v3` identity and initially requires FP32 adapter computation. Historical zero-B identities remain readable under their recorded contracts; newly constructed default zero-B models write the complete v2 precision identity described below. A nonzero compensating adapter is an **initialized, untrained adapter**, not the frozen zero-B reference. Reload parity compares the same initialized/trained configuration with itself. Charge preparation, calibration, loading and artifact costs to the NF4 recipe when comparing total compute and time to quality.

The [completed controls and recovery evidence](evidence/native_qlora_residual_controls_1ac6a7f.json) pairs all row, media, input, mask and token identities across six states on 64 exposed training examples. After 20 updates, residual NF4 scored 51/64 with native token-weighted CE 0.222689, versus ordinary NF4 49/64 and 0.246915; the ordinary BF16 control scored 50/64 and 0.238948. This passed the previously locked effort screen for one prospectively specified matched BF16/control/residual development experiment. These warmup-prefix results, with differing validation policies, establish neither held-out benefit nor equal-compute or speed superiority.

That [locked 100-update pilot](evidence/native_qlora_residual_development100_1ac6a7f.json) is now a negative result: residual NF4 scored 530/584 with native CE 0.185621, versus ordinary NF4 533/584 and 0.169656, and BF16 531/584 and 0.179880. All three quality rules failed, and four preserved host-memory gaps also prevent the resource rule from passing. Own training and fresh reload checks passed. The single-seed, 12-group exposed-development experiment consumed 5,094 GPU-seconds; residual cost also includes the prior 81.54-second initialization without double-charging its allocation. No arm attained the locked accuracy-and-CE target at update 100, and no first-crossing or speed advantage is established. The residual method was not promoted to a longer sweep. The original failed calibration, +4.18% regression and unchanged 2% release margin remain preserved.

## Experimental language-block exclusions

The optional `model.model_config.quantized_base.excluded_language_layers` accepts at most two distinct integer block IDs from 0 through 35. For example, append this override to the native NF4 launch:

```bash
'model.model_config.quantized_base.excluded_language_layers=[6,15]'
```

Each excluded block keeps its seven language projections in the original frozen BF16 source precision. All 252 LoRA targets and 504 FP32 adapter parameters remain trainable; exclusions change base representation, not adapter coverage. The loader materializes these dense weights directly and quantizes only the remaining matrices, giving 245 or 238 NF4 matrices for one or two exclusions. Vision, projector, embedding, head and normalization exclusions remain unchanged. This option cannot be combined with residual initialization.

The [configuration and loader tests](../../cosmos_framework/posttrain/tests/test_quantized_exclusions.py) cover exact resolved coverage, optimizer membership, frozen-state checks, legacy identity preservation and incompatible reload/resume refusal. Nonempty exclusions use `cosmos.adapter-model.v4`, recording the dense and quantized target sets explicitly; an empty list preserves existing identities. Reload and resume must reconstruct the same exclusions before allocating the base.

The [completed mixed-base correctness run](evidence/native_qlora_mixed_exclusions_f5e40cf.json) tested exactly `[6,15]` on native `f5e40cf`: 238 NF4 matrices, 14 frozen BF16 language matrices and all 504 FP32 adapter tensors. Its CUDA matrix fixture passed, followed by all 15 full-model lifecycle stages at rank 32/alpha 64. Both interruption paths resumed from their own committed update 2 and exactly matched updates 3 and 4, including optimizer/RNG/data controls; fresh DCP and portable text/image/video reloads also matched with zero tolerance. The run used ordinary `bnb` execution, full activation checkpointing, zero-B initialization and one preprocessing worker.

All 1,069 recorded v7 host samples were complete, with a sampled owned-GPU peak of 10.820 GiB; unsampled startup and total host residency remain unknown. The allocation consumed 1,474 GPU-seconds and 23,584 allocated CPU-seconds. This is four-update correctness evidence for the exact two-block configuration, not smaller-GPU, quality or speed qualification. The [training-only sensitivity diagnostic](native_qlora_qualification.md#ongoing-improvement-campaign) motivates blocks 6 and 15, but their effects cannot be assumed additive. The subsequent [100-update development screen](evidence/native_qlora_mixed_development100_f5e40cf.json) failed all three predeclared point criteria; these exclusions were not promoted. Other exclusions need separate evidence.

The subsequent [matched 20-update training screen](evidence/native_qlora_mixed_training20_f5e40cf.json) used fresh same-native ordinary NF4, mixed `[6,15]` and BF16 controls with validation disabled. On 64 exposed training examples, mixed and ordinary NF4 both scored **49/64**, with native token-weighted CE **0.237011/0.246915**; BF16 scored **50/64** with CE **0.238948**. Mixed reduced CE by 4.01% while exchanging one correct answer for another, so no accuracy gain was established. Corresponding sampled GPU-process stage peaks were at most **550 MiB** above ordinary NF4, within the locked 1 GiB allowance. Own update-20 inspections and fresh DCP/portable reloads passed; interruption recovery remains the earlier four-update evidence. The training-only effort screen therefore passes for a separately admitted bounded development comparison, without quality or speed promotion. All 1,845 observed v8 samples were complete, but unsampled startup and total host/cache peaks remain unknown. The allocation cost was **2,434 GPU-seconds and 38,944 allocated CPU-seconds**.

## Experimental performance controls

These opt-in controls are implemented for the single-device native adapter path. On the sealed `a406ea1d32c95277c57ac8e7ec274784bed3aaf7` runtime, **14 real CUDA tests and 13 four-update native lifecycle cases passed**. Each lifecycle case compared uninterrupted training with its own fresh-process resume, then compared DCP and portable reloads on text, image and video inputs. Adapter/optimizer state and controlled reload outputs matched with `atol=0, rtol=0`; this is parity within each recorded configuration, not equality between different execution settings. The [lifecycle and input-shape evidence](evidence/native_qlora_lifecycle_a406ea1.json) records the exact configurations, source digests and measurement limits.

Those checks establish bounded numerical and lifecycle evidence, not held-out quality or a speed benefit for every option. The original 300-update results below used the original defaults; their measurements and unsatisfied release quality gate are unchanged. Benchmark each option on the intended workload before selecting it.

A separate [native `1ac6a7f` BF16 no-checkpointing follow-up](evidence/native_qlora_bf16_no_ac_finish_1ac6a7f.json) completed the reversed none→full 20-update order. Complete native-command times were 388.487 and 388.127 seconds respectively, while summed update times were 217.665 and 224.798 seconds. All 20 numerical update metrics matched; the different timing scopes do not establish an execution winner or the 10% time-to-quality target. The no-checkpointing configuration also passed its own 15-stage four-update interruption/recovery suite and fresh reloads. A retained earlier no-checkpointing update-20 artifact separately passed DCP/portable reload, with its seven files hashed at first allocated reuse and rechecked afterward. This is not cross-checkpointing payload equality or interruption recovery at update 20. Eleven original host samples remain incomplete, and the actual hardware was an A100 80GB.

The separate [NF4 reverse-order diagnostic](evidence/native_qlora_nf_no_ac_diagnostic_1ac6a7f.json) completed the same 20-update none→full order on native `1ac6a7f`: complete native-command times were 374.355/363.903 seconds, and summed update times were 203.221/215.400 seconds. All 20 numerical update metrics matched, but the different timing scopes do not select a speed winner. It performed own update-20 checkpoint inspections, without fresh no-checkpointing reload or recovery. A collector defect left 667 of 672 host samples incomplete, including every sample from both native training stages: failed process bindings remained tracked after short-lived children disappeared before a pidfd could be opened. The original false flags remain unchanged and block complete resource qualification. Valid GPU observations and allocator intervals remain descriptive; the earlier slow NF4 no-checkpointing observation and all costs are retained.

The subsequent [version 8 collector CPU qualification](evidence/native_qlora_collector_v8_be0ca36.json) passed 169 tests without skips on Linux `5.15.0-1032-oracle`, including six process proofs. A later collection can retire a never-bound entry after verified current-namespace absence or bind the same live process while retaining its original ancestry. Among the two new fixtures, disappearance uses actual errors; the orphan fixture injects only its initial stat ENOENT, followed by actual reparenting, pidfd opening and memory reads. All nine retained failure rows remain incomplete. The allocation used 3,040 CPU-seconds and no GPU time. An optional aggregate-memory observer could not verify container membership because of a permission error and produced no final peak. These CPU checks do not qualify training telemetry, task quality or speed.

Append selected dotlist overrides after `--` in the native canary command above. Use a fresh job/output for each recipe, keep the data, processor, media limits, effective batch and schedule matched, and qualify individual changes before combining them.

| Native override                                     | Default     | Experimental behavior                                                                                                                                                                                        |
| --------------------------------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `model.model_config.activation_checkpointing=none`  | `full`      | Disable activation recomputation, trading additional memory for a possible speed benefit. `full` retains non-reentrant checkpointing; `selective` uses the recorded native SDPA/matrix-multiply save policy. |
| `model.model_config.sft_logits_mode=supervised`     | `full`      | Run the complete multimodal backbone, then project only the positions selected by the existing shifted labels, attention masks and validation weights. The ordinary full-logit API remains available.        |
| `model.model_config.adapter_compute_dtype=bfloat16` | `float32`   | Use BF16 adapter matrix arithmetic through differentiable casts. Adapter parameters, their gradients and optimizer moments remain FP32; packed base storage is unchanged.                                    |
| `model.model_config.compile_language_model=true`    | `false`     | Compile native language blocks with Inductor, dynamic shapes and `fullgraph=False`, outside any activation-checkpoint wrapper. Vision and the output head remain outside this compilation scope.             |
| `rl.preprocessing_cache_bytes=1073741824`           | `0`         | Example 1 GiB CPU cache budget: one quarter for processed samples and three quarters for complete video-processor outputs, including frame metadata and timestamps.                                          |
| `rl.preprocessing_prefetch=true`                    | `false`     | Prepare the current optimizer update's rows in spawned CPU workers while training consumes them in order.                                                                                                    |
| `rl.preprocessing_workers=2`                        | `1`         | Experimental parallel preparation: split the total cache budget and inherited Linux CPU affinity between two workers. Requires prefetch and an update window containing at least two examples.               |
| `rl.micro_batch_size=2 rl.grad_accum=4`             | `1` and `8` | Keep eight examples per update with fewer microbatches. `rl.micro_batch_size=4 rl.grad_accum=2` is another candidate. Padding, memory peaks and floating-point reductions require separate checks.           |

The model options and selective policy are defined by [Qwen3VLModelConfig and native_adapter_execution_policy](../../cosmos_framework/posttrain/config/model.py). The implementations are [Qwen3VLPolicy.forward_sft_batch and build_qwen3_vl_model](../../cosmos_framework/posttrain/model/qwen3_vl.py) and [NativeLoRAMixin.adapter_delta](../../cosmos_framework/utils/generator/lora.py). Compilation permits graph breaks; [guard_native_compiled_call](../../cosmos_framework/posttrain/utils/native_compile.py) makes compiler errors and recompile-limit fallback fatal during invocation. Compilation counters do not prove complete graph capture or fusion. This work does **not** introduce a fused NF4 training kernel.

Disabled checkpointing passed the [native `149c2cf` four-update recovery and reload checks](native_qlora_qualification.md#disabled-checkpointing-correctness), followed by [two 300-update engineering blocks and own-policy mature recovery](native_qlora_qualification.md#completed-reverse-block-and-execution-choice). It is now selected only for the measured rank-32 WTS variant with 3 GiB caches in both arms; the generic default remains full. Cross-policy comparisons acknowledge only the declared identity differences, while production resume must keep the original policy. Quality preservation, independent confirmation and time-to-quality improvement remain unresolved.

[build_sft_preprocessing](../../cosmos_framework/posttrain/data/packer/sft_preprocess.py) keys cache entries by processor/target semantics and verified media content, returns independent CPU copies, and rejects preprocessing that consumes RNG state. Cache contents are disposable and rebuilt after resume. The byte limit covers retained cache entries, including tensor backing storage; queued results, decoder buffers, worker runtime and filesystem cache require additional host memory. [train_step](../../cosmos_framework/posttrain/entrypoints/sft.py) reads only the current accumulation window. The [trainer's input-window and checkpoint guards](../../cosmos_framework/posttrain/roles/trainer/llm.py) require its optimizer update and ordered consumption to finish before publishing a checkpoint; failed or unfinished prefetch cannot publish an advanced data cursor.

The optional two-worker path assigns unseen media to the worker with fewer assigned rows in the current update and retains ownership of repeated media in a 1,024-entry LRU. Ownership within the current window survives LRU eviction; later eviction can cause another cache miss. Completed results remain in input order, and counters sum each worker's latest cumulative observation. Existing worker threads and descendants are restricted to disjoint subsets of the inherited CPU affinity; this does not bound the number of FFmpeg threads or total host memory. The default remains one worker. The earlier hash-routing implementation passed an actual 64-example WTS input/mask/RNG check but showed no useful preparation-time gain in that diagnostic. Balanced routing subsequently preserved all inputs in a 160-example [CPU screen](native_qlora_qualification.md#ongoing-improvement-campaign) and reduced its measured case time by 15.6% versus one worker. This qualifies a GPU follow-up, with actual WTS training, repeated timing, memory and recovery checks still required before recipe promotion. Its `cosmos.sft-preprocessing.v3` training identity binds the routing policy and ownership limit and rejects resume from the earlier v2 policy or a different worker setting; the default retains v1. Any demonstrated preparation benefit must be available to both NF4 and BF16.

Newly constructed zero-B models write a `cosmos.adapter-model.v2` identity containing the complete execution policy and observed math-precision settings, including ordinary `bnb` execution with full logits. Existing optimized v2 policies are unchanged. Residual initialization and mixed-base artifacts retain their v3/v4 envelopes and now also record the complete policy when using default execution. Batch and preprocessing settings are bound separately to the training identity, while caching preserves the processor/masking identity. [execution_config_from_identity and validate_execution_policy_identity](../../cosmos_framework/posttrain/model/quantization/loader.py) validate recorded options and current math flags before model loading; missing or incompatible v2 policies fail.

Older identities with a genuinely absent default policy remain readable with a warning: their historical runtime precision is unknown, so exactness depends on externally preserved original flags. Compatibility preserves every other identity field and does not authorize cross-build optimizer resume; native revision/wheel, schedule, data and other training contracts remain strict. A subsequent training save records the current process's precision without inventing earlier observations; portable conversion of an existing checkpoint preserves that checkpoint's identity. This metadata contract change does not extend the GPU, recovery or quality evidence of earlier sealed builds.

For a bounded diagnostic trace, set these variables before the native launch:

```bash
export COSMOS_SFT_PROFILE_DIR="$QLORA_WORK/profiles/new-trial"
export COSMOS_SFT_PROFILE_WAIT=2
export COSMOS_SFT_PROFILE_WARMUP=1
export COSMOS_SFT_PROFILE_ACTIVE=2
```

[NativeStepProfiler](../../cosmos_framework/posttrain/utils/training_profile.py) records one window measured in optimizer updates, with preparation, forward, loss, backward and optimizer ranges. It writes runtime settings, a compressed Chrome trace and an operator summary beneath `rank_0`; an existing rank directory is refused. Leave `COSMOS_SFT_PROFILE_DIR` unset to disable profiling. Trace timings are diagnostic: measure comparative throughput in separate untraced runs, including compilation/loading costs and actual GPU and host peaks.

### Experimental checkpointed dequantization

`model.model_config.quantized_linear_execution=checkpointed_dequant` opts into a bounded training execution change; the default remains `bnb`. It dequantizes each actual attached NF4 matrix to BF16 and lets the ordinary linear operation participate in autograd. Full non-reentrant block checkpointing discards and reconstructs saved weights, allowing backward to reuse the reconstructed weight instead of invoking a second dequantization. Packed weights and quantization metadata remain frozen. There is no persistent dense-weight cache or new fused kernel.

The native builder verifies all 36 language-block wrappers and all 252 target objects before enabling the block scopes. An optimized training forward outside its owning checkpoint scope fails. This implementation requires one device, full activation checkpointing, ordinary zero-B LoRA with zero dropout, FP32 adapter arithmetic and eager execution. The [configuration contract](../../cosmos_framework/posttrain/config/model.py) can reconstruct either full logits or the supervised-token projection tested on sealed `e6e383b` artifacts. **Fresh native training combining supervised projection with checkpointed dequantization is now refused before base allocation:** the completed comparison did not justify promoting that combination. Use `sft_logits_mode=full` with this experimental dequantization option, or ordinary `quantized_linear_execution=bnb`. Existing artifacts retain their recorded reconstruction policy; actual resume must still pass complete checkpoint, source/build, precision and training-contract checks. Selective or disabled checkpointing, compilation, residual initialization and language-block exclusions remain rejected combinations. A backward exception aborts training; recovery uses a fresh process and a completed checkpoint. No-grad/inference execution retains the original bitsandbytes path.

The [completed CUDA mechanism experiment](evidence/native_qlora_saved_dequant_105d314.json) tested two synthetic blocks at 128 tokens on an A100 80GB. All 519 output/gradient/updated-adapter comparisons were exact. Median warm update time fell from 26.787 to 23.671 ms (11.63%), with a 275.52 MiB higher allocated peak and at most one 368 MiB block of live dense storage. Both packed test models were resident, and the last host sample remained incomplete. This passed the predeclared implementation-effort screen; it establishes neither full-model memory nor a WTS or end-to-end speed benefit. Its original measurements remain separate from the native correctness result below.

The [native `f5e40cf` correctness allocation](evidence/native_qlora_checkpointed_dequant_f5e40cf.json) completed the full Qwen3-VL-8B path with rank 32, alpha 64, zero-B initialization and FP32 adapters. All 11 outer stages and the candidate's 15-stage lifecycle passed. Boundary and mid-accumulation process interruptions resumed from committed update 2 and exactly reproduced updates 3 and 4, including all 504 adapter tensors, optimizer, RNG and data/control state. Fresh DCP and portable reloads matched text, image and video outputs with zero tolerance. This is four-update recovery evidence, not a 100/300-update crash-recovery claim.

An allocated diagnostic also compared four updates against ordinary `bnb`: numerical metrics and adapter/optimizer/control/RNG/data state matched exactly after normalizing only validated execution-policy metadata copies. The original artifacts were unchanged, and production cross-mode resume remains rejected. The allocation consumed 1,809 GPU-seconds and 28,944 allocated CPU-seconds. Two of 1,334 host samples remained incomplete, both during host fixtures; observed native reference/lifecycle samples were complete, without establishing total host residency. The candidate's recorded reference allocator peak was higher than `bnb` (8.712 versus 8.431 GiB, loading through final validation/checkpoint). Task quality, repeated end-to-end speed and release qualification remain open; the default stays `bnb` and the original 2% quality criterion is unchanged.

The subsequent [three-pair WTS execution comparison](native_qlora_qualification.md#repeated-20-update-execution-comparison) completed six fresh 20-update runs with matched BF16 controls. Adapter/optimizer/RNG/data state and portable calibration scores matched their retained same-arm references exactly. NF4 updates averaged 5.84% longer, while the short training-through-portable-scoring diagnostic averaged 1.70% shorter; sampled GPU-process peaks were 9.334 GiB lower. All runs used FP32 fused AdamW moments. These short, single-seed timing repeats do not establish the 10% time-to-quality target, primary held-out quality improvement or longer-horizon interruption recovery.

Those recovery and WTS results used full logits. The subsequent [selected-token composition qualification](native_qlora_qualification.md#selected-token-projection-with-checkpointed-dequantization) passed 17 CUDA/CPU/canary cases on the sealed `e6e383b` build, followed by full-model four-update interruption/recovery and fresh text/image/video DCP/portable reload checks. NF4 completed all 15 native stages; BF16 completed 14 before a scheduler stop, with its final saved-logits comparison completed separately on CPU. The original failed BF16 allocation remains recorded. All six subsequent timing runs completed. In the placement-matched NF4 pair, combined execution reduced update time by 1.42%, increased training-through-scoring time by 0.53% and increased allocated peak by 263.05 MiB. The second pair had different GPU/CPU/NUMA placement. This does not justify promoting the combination. Full and supervised projection retain distinct execution identities; changing head mode during resume is rejected. No task-quality or release qualification follows.

For historical reproduction on the sealed `e6e383b` build only, the rank-32 native command used these two overrides in both training and reconstruction:

```bash
model.model_config.sft_logits_mode=supervised \
model.model_config.quantized_linear_execution=checkpointed_dequant
```

The BF16 control used `sft_logits_mode=supervised` and left `quantized_linear_execution=bnb`, with the same full activation checkpointing, FP32 adapters/AdamW, preprocessing, batch and schedule settings. These are historical experiment instructions, not the current fresh-training recipe. Changing the output projection's matrix shape can change floating-point rounding; its own reload parity does not establish the earlier full-logit training trajectory or held-out quality.

This option uses `cosmos.native-adapter-execution.v2` within the existing model-v2 artifact envelope, recording its block scope and unchanged inference dispatch. Its recorded policy and earlier execution-v1 policies remain unchanged; new default artifacts now record their own complete execution-v1 policy as described above. Reload restores and validates the recorded option before allocation; missing or contradictory policies fail. Use a distinct output when qualifying the option and do not reinterpret an earlier default checkpoint as its result.

## Reload, resume, and portable artifacts

Run outputs include a timestamped attempt beneath the configured project/group/job. Use the committed checkpoint path reported by the native run; full checkpoints are under its `checkpoints/iter_*` directory. Saves occur at completed optimizer-update boundaries, through the existing synchronous local DCP commit/recovery path.

| Operation                    | Contract                                                                                                                                                                                                                                                                                                                            |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Exact resume                 | Relaunch the same native command and receipt with `rl.resume=auto`, or `rl.resume=/absolute/run/checkpoints/iter_000002`. Keep source, processor, data, schedule/horizon, optimizer, accumulation, validation policy, and world size unchanged. Full optimizer, scheduler, RNG, data cursor, and update-control state are required. |
| Adapter reload               | Reconstruct the recorded quantized base, then load adapters without restoring training state. Fresh-process controlled logits/loss are the initial numerical-parity target.                                                                                                                                                         |
| Portable adapter publication | Publish adapters and identity metadata only; this artifact cannot resume training. It neither merges nor requantizes the base.                                                                                                                                                                                                      |
| Dense/merged exports         | Dequantized `Q(W)` plus the adapter, original higher-precision `W` plus the adapter, and subsequent requantization are different numerical models. The effective-base BF16 conversion has scoped evidence below; other conversions and serving-backend compatibility remain unqualified.                                            |

Exact full resume retains the recorded canonical media paths and ordinary tokenizer/processor configuration paths; identical bytes at different canonical locations are not sufficient. Cache capacity, worker count and prefetch policy must also match the saved training identity, including when using the selected 3 GiB override. The [frozen-base source identity](../../cosmos_framework/posttrain/utils/adapter_source.py#L79) uses content hashes, while [portable processor identity](../../cosmos_framework/posttrain/utils/adapter_identity.py#L67) separately permits relocation of verified base/processor references for reload. The earlier four-update recovery tests stayed within their allocated nodes. The completed late check resumed both selected 3 GiB arms on one observed different host, retaining the same single-process world size, sealed runtime, canonical input paths and cache policy. This is evidence for that recorded move, not general cross-node Slurm recovery or topology-change support.

The cumulative `train/grad_clip_fraction` is reconstructed from [retained reporting history](../../cosmos_framework/posttrain/utils/report.py#L401), so a fresh output directory starts a new history for that diagnostic. It does not control gradient clipping or optimizer updates. The [existing qualification comparator](../../tools/native_qlora_qualification/lifecycle.py#L146) excludes this field and observational timing/preprocessing counters; its strict training-metric and full-checkpoint state comparisons remain separate checks. Checkpoint-alone recovery does not require copying an uninterrupted run's metrics.

Publish a committed checkpoint as a portable adapter artifact:

```bash
python -m cosmos_framework.posttrain.checkpoint.adapter \
    --source /absolute/run/checkpoints/iter_000003 \
    --output-iter /absolute/export/checkpoints/iter_000003
```

The command prints the committed artifact location in the sibling `checkpoint_artifacts/iter_*` namespace. It does not advance the training latest pointer. Incomplete checkpoints, incompatible identities, unsupported topology, and adapter-only artifacts presented as resumes fail explicitly.

### Effective-base dense export

The native exporter reconstructs the recorded base and adapters, dequantizes each attached NF4 matrix on its owned CUDA device, and merges `FP32(dequantize(Q(W))) + (alpha/rank) * FP32(B @ A)` in bounded CPU row blocks. It casts the completed sum to the explicitly requested output dtype. It never merges into the original higher-precision `W`, requantizes, or mutates the source artifact. Non-adapter components are copied and cast to the same output dtype. Separate-branch adapter evaluation and dense BF16 evaluation need not have identical rounding.

```bash
torchrun --standalone --nproc-per-node=1 \
    -m cosmos_framework.posttrain.checkpoint.tools.export_adapter_dense \
    --config cosmos_framework/posttrain/configs/sft_qwen3_vl_8b_qlora.py \
    --checkpoint /absolute/committed/adapter \
    --output /absolute/new/dense-model --output-dtype bfloat16 \
    --max-shard-size-bytes 536870912 \
    --override model.model_id=/absolute/pinned/base \
    --override model.model_config.adapter.rank=32 \
    --override model.model_config.adapter.alpha=64 \
    --override model.model_config.adapter.initialization_seed="$QLORA_SEED" \
    --override rl.seed="$QLORA_SEED" \
    --override model.model_config.activation_checkpointing=none
```

Supply the original processor and execution overrides as well as the installed `COSMOS_TRAINING_CODE_REVISION` and `COSMOS_TRAINING_WHEEL_SHA256`. The source identity must match; the example is not an automatic reconstruction of arbitrary artifacts. Output creation is exclusive. The exporter writes ordinary HF safetensors, config and verified processor files, then publishes `cosmos_dense_export.json` last with source identities, arithmetic, file/tensor hashes and exporter revision. Interrupted exports remain incomplete. `read_dense_export(..., expected_manifest_sha256=...)` verifies the committed artifact before an ordinary HF load. Dense exports are evaluation artifacts, not exact training resumes.

From `e05a061`, exports deterministically pack name-ordered tensors into standard
indexed shards, with a configurable 512 MiB tensor-payload limit by default.
A larger single tensor occupies its own shard; headers, the current merge matrix
and row-block temporaries are additional memory. The reader retains compatibility
with the earlier one-tensor-per-shard format. Packing and interruption/readback
checks include fresh tiny multimodal HF loads. The historical measurements below
use the earlier format. A separate [full-model packing qualification on
`26bc60f`](native_qlora_qualification.md#completed-bounded-shard-packing-on-26bc60f)
now verifies 32 packed shards, all 750 tensor records and exact fresh
text/image/video reloads, with loading/export memory and clocks recorded separately.
It makes no matched performance or new quality claim. The effective-base merge
arithmetic is unchanged.

With `COSMOS_QLORA_METRICS_FILE` enabled, `09e99c3` records an export-start/reset event and a completion event after committed readback, covering adapter loading, dequantization, merging and publication separately from base loading. Failed exports attempt a diagnostic event without masking the original exception. These allocator records complement the external process sampler; they do not measure allocations outside the PyTorch allocator or whole-host memory.

The existing WTS scorer accepts `--dense-export PATH --dense-export-sha256 SHA256` instead of `--checkpoint`, retaining the hard-gold objective, masks, input identities and answer-choice rules. On sealed `09e99c3`, the [completed full-model BF16 conversion](native_qlora_qualification.md#completed-effective-base-bf16-export) matched all 750 post-cast tensors against an independent full-matrix merge and passed fresh text/image/video reload. The 584-row artifact check scored **538 versus 540 correct** for the source checkpoint, with **0.099% higher token CE**; it passed the predeclared artifact point guard. This is separate from recipe-level noninferiority and independent confirmation.

Export/publication peaked at **7.234 GiB allocated / 7.256 GiB reserved** on CUDA, after base loading; fresh dense reload/probes peaked at **18.648 / 18.943 GiB**. The dense artifact occupies **17,546,768,689 bytes** including its manifest. These are different memory scopes from quantized training. BF16 dense output is numerically different from the unmerged adapter path, and the measured logit differences are retained. FP32 dense output remains CPU-tested only; merging onto original higher-precision weights, subsequent requantization, serving-backend compatibility and inference latency remain unqualified. The ordinary quantized-base adapter-reload contract remains separate.

### Development-only answer-choice smoothing

`rl.wts_answer_smoothing=0.02` opts into a narrowly scoped training intervention: mix the answer target with uniform mass over the question's actual two to four choices while retaining the full-vocabulary softmax denominator. The termination token retains hard-gold CE; validation still reports unsmoothed CE and answer accuracy. Strict WTS/token alignment is required; the objective is incompatible with weighted sequence loss and MoE balancing. The default zero retains the previous arithmetic and artifact identity. Nonzero values enter the training identity and cannot be changed during exact resume.

On sealed `09e99c3`, both precision arms passed the [CUDA objective and full-model four-update recovery checks](native_qlora_qualification.md#completed-answer-smoothing-correctness). The [completed four-case development screen](native_qlora_qualification.md#completed-answer-smoothing-development-screen) then passed its frozen followup rule: at update 100, NF4 smoothing improved **528 → 531 / 584 correct** and reduced hard-gold token CE by **2.382%**. The BF16 rule retained epsilon zero because smoothing reduced its correct count from 525 to 523, despite lower CE. NF4 retained **9.893 GiB** lower sampled GPU-process memory than that BF16 control over the complete case intervals.

The positive exposed-development screen remains recorded with its original decision. Its [completed two-seed followup](native_qlora_qualification.md#completed-two-seed-smoothing-followup) retained the locked NF4 `0.02` / BF16 `0` pair and the separate BF16 dropout-0.05 reference at both seeds. At seed 17, NF4 smoothing first reached the joint target at update 250 in **42.66 minutes**, versus BF16 zero at 150 in **26.41 minutes** and BF16 dropout at 250 in **38.71 minutes**. All three seed-123 recipes completed 300 updates without reaching the target. NF4 retained **9.895–10.152 GiB** lower sampled GPU-process peaks across these fixed comparisons, but no time-to-target advantage was demonstrated.

**Default smoothing stays zero.** At shared update 150, seed-17 smoothing lowered hard-gold CE by 3.50% versus retained NF4 zero but lost three correct answers, missing the accuracy component of the target. At seed-123 update 300, CE was 2.21% lower with five fewer correct answers; at update 150 it was instead 11.61% higher with unchanged accuracy. These descriptive, exposed-data comparisons do not establish joint quality improvement or calibration. No followup promotion threshold was predeclared; the decision preserves the original positive 100-update screen without treating it as a successful recipe qualification. Historical uncertainty failures, the earlier NF4 dropout rejection and independent-data requirements remain unchanged. Workflow `add7398` exposes the native option as Experiment JSON `"wts_answer_smoothing": 0.02`; its 76 CPU checks passed without changing the native wheel.

The [integration driver](../../cosmos_framework/posttrain/tests/qlora_integration.py) evaluates the native model; it contains no training loop. Run its `probe` subcommand in a fresh torchrun process for the saved DCP checkpoint and again for its portable artifact, using the same fixed image and video row. Supply an existing image, or omit `--image` on the first probe and reuse its generated `video_first_frame.png`:

Use the checkpoint's original base/processor snapshot and saved adapter configuration, including rank, alpha, dropout, initialization method and seed. The probe restores recorded execution settings and mixed-base exclusions, but does not restore these adapter settings or the processor's sequence limit, masking and image/video options. The example below matches the selected rank-32, zero-dropout recipe above; retain its original `QLORA_SEED` and use the BF16 config for a BF16 checkpoint. Other recipes require their own saved overrides. The probe evaluates hard-gold loss even for a smoothing-trained adapter; this is artifact reload, not exact training resume.

```bash
torchrun --standalone --nproc-per-node=1 \
    -m cosmos_framework.posttrain.tests.qlora_integration probe \
    --config "$QLORA_CONFIG" --checkpoint /absolute/committed/checkpoint \
    --data "$SFT_DATA_ROOT/validation.jsonl" --image /absolute/probes/frame.png \
    --output /absolute/probes/new-output --require-updates \
    --override model.model_config.adapter.rank=32 \
    --override model.model_config.adapter.alpha=64 \
    --override model.model_config.adapter.dropout=0.0 \
    --override model.model_config.adapter.initialization=kaiming_A_zero_B \
    --override "model.model_config.adapter.initialization_seed=$QLORA_SEED" \
    --override "rl.seed=$QLORA_SEED" \
    --override rl.max_length=4096 --override rl.mask_think_opener=false \
    --override rl.processor_kwargs.images_kwargs.size.shortest_edge=65536 \
    --override rl.processor_kwargs.images_kwargs.size.longest_edge=262144 \
    --override rl.processor_kwargs.videos_kwargs.num_frames=8 \
    --override rl.processor_kwargs.videos_kwargs.size.shortest_edge=32768 \
    --override rl.processor_kwargs.videos_kwargs.size.longest_edge=524288
```

It checks text, image, and real video losses/logits, media sensitivity, trainable coverage, adapter changes, and frozen-base fingerprints. Use `compare-probes` for reload parity and `compare-checkpoints` for interrupted-versus-uninterrupted resume; both require explicit predeclared `--atol` and `--rtol`. Resume comparison also checks RNG and data/control state exactly. Passing a probe alone does not establish held-out task quality or total training memory savings.

## Performance qualification

The [completed controlled three-seed comparison](native_qlora_qualification.md#completed-controlled-three-seed-comparison) uses the same cache/prefetch settings in both precision arms. NF4 steady updates were 6.68% slower, while the complete measured-experiment timing interval crossed parity. Paired sampled GPU process peaks differed by 9.78–9.88 GiB; host-memory gaps remain explicit. The development-set NLL uncertainty bounds still fail the unchanged 2% criterion.

The [selected rank-32 comparison](native_qlora_qualification.md#completed-selected-three-seed-development-comparison) adds fresh seeds 17 and 123 to the audited development seed 42. NF4 steady updates were 7.12% slower and complete measured experiment time was 1.49% longer. Paired sampled training GPU-process savings were 9.38–9.62 GiB. The selected development NLL crossed upper bound was 2.101%, so quality remains experimental. This fixed-execution study does not complete the original best-fit runtime selection. Both studies are separate from the historical within-arm preprocessing improvements below.

The [completed sequence/media matrix](native_qlora_qualification.md#completed-runtime-and-recovery-matrix) adds real-kernel 20-update tests for roughly 4,096-token synthetic targets and larger 16-slot video, plus exact four-update recovery for the stated settings. Larger-video NF4 microbatch four fit the observed process-memory predicate but did not improve throughput. In that historical rank-16 study, long-workload host measurements remain incomplete, and combined long-target/larger-video input and physical 24 GB GPUs were not tested. The later [rank-32 combined-input check](native_qlora_qualification.md#completed-combined-long-context-and-larger-video-check) has separate training/recovery evidence on one A100 80GB with full activation checkpointing. Neither study establishes natural long-task quality, smaller-GPU support or a general speed advantage.

The original performance study selected a 1 GiB CPU cache and ordered prefetch with full non-reentrant activation checkpointing, FP32 adapter arithmetic/parameters/gradients/moments, full logits, microbatch one, accumulation eight, and compilation disabled. The following command reproduces that historical recipe; the current WTS override uses 3 GiB in both arms, as documented above:

```bash
torchrun --standalone --nproc-per-node=1 \
    -m cosmos_framework.posttrain.entrypoints.sft --config "$QLORA_CONFIG" -- \
    job.name=qlora_pipeline_seed42 \
    rl.preprocessing_cache_bytes=1073741824 rl.preprocessing_prefetch=true
```

Use the matched BF16 recipe and a distinct output name with the same overrides. These controls preserve the base, processor, loss masks, target modules, rank, effective batch, optimizer and schedule. CPU preprocessing overlaps with training; the NF4 storage format itself does not establish a speedup.

The [performance experiment report](evidence/native_qlora_performance_a406ea1.json) retains individual ablations, failed attempts and three separately scoped repeated comparisons. The selected cache/prefetch comparison used three paired 30-update trials per precision, with the predeclared untraced timing window at updates 6–30. Each pair used the same allocation; trial order alternated. The confidence interval uses 10,000 paired hierarchical bootstrap replicates, resampling repeats and fixed five-update blocks, with seed 2026.

| Precision | Control median trial mean, s/update | Cache/prefetch median trial mean, s/update | Reduction | Optimized/control ratio, 95% CI |
| --------- | ----------------------------------: | -----------------------------------------: | --------: | ------------------------------: |
| NF4       |                              20.971 |                                     12.016 |    42.70% |                [0.5388, 0.6121] |
| BF16      |                              19.988 |                                     11.830 |    40.82% |                [0.5423, 0.6367] |

Both pass the task-specific speed filter: at least 5% lower median trial mean and an upper ratio confidence bound below one. Three trials provide limited uncertainty evidence. Requested CPU counts were identical, but actual SMT/physical-core and NUMA placement differed between precision arms; cross-arm absolute timing is descriptive. Filesystem/page-cache state was uncontrolled. Profile traces are diagnostic and are excluded from these throughput comparisons.

Across all three selected matched pairs, NF4's maximum allocated GPU memory was 9.597 GiB lower and its sampled owned-process GPU peak was 9.703 GiB lower than BF16. The performance candidate filter uses allocator peaks. The original release criterion uses maximum device-process memory; sampled peaks are lower bounds, and their difference does not establish the difference between unsampled true peaks. Neither result completes the three-seed release gate.

Compilation has separate tradeoffs. In the larger-batch, no-checkpointing combination, compilation's incremental speed confidence interval included parity with eager execution in both precision arms. Compiled NF4 used 3.328–3.330 GiB more allocated memory than matched compiled BF16 in all three pairs and failed the memory filter. Selective checkpointing with compilation passed the within-arm speed filter against its own control and the NF4-versus-BF16 allocated-memory filter, but used about 15.10 GiB allocated in NF4's 30-update trials. The selected cache/prefetch recipe retains the original lower GPU footprint. These different plans do not establish a controlled ranking between all combinations; held-out quality for the alternative model-execution settings remains unqualified.

The selected NF4 and BF16 recipes also completed 300 updates each. A separate [read-only checkpoint comparison](evidence/native_qlora_cross_build_a406ea1.json) found byte-identical adapter tensors, optimizer state, scheduler/RNG, data position and update controls against each precision's original `c5b8d49` run, after allowing only the exact build-receipt differences and the recorded preprocessing-identity additions. All 504 adapter tensors and 1,525 optimizer entries matched in each arm. This diagnostic does not relax production resume checks or establish cross-build compatibility in general. Those historical default model identities used legacy v1, which did not record observed TF32/math flags; the demonstrated parity is limited to the sealed runtimes and measured runs.

### Completed optimized 300-update runs

Both selected recipes completed 300 finite, applied optimizer updates with zero skipped attempts, the same final data cursor, and all seven 64-row validation points. They consumed the same first 2,400 training examples under seed 42, with full DCP saves every 50 updates. The [training and evaluation report](evidence/native_qlora_wts_pipeline_a406ea1.json) records source/runtime identities, raw-evidence hashes, phase measurements and limitations.

| Training measurement                | NF4 LoRA | BF16 LoRA |
| ----------------------------------- | -------: | --------: |
| CUDA allocated peak, GiB            |    7.769 |    17.367 |
| CUDA reserved peak, GiB             |    8.201 |    18.078 |
| Owned-process NVML sample peak, GiB |    9.199 |    19.076 |
| Process-tree PSS sample peak, GiB   |    9.117 |     8.769 |
| Mean update time, updates 1–300, s  |   10.938 |    10.907 |
| Mean update time, updates 31–300, s |   10.752 |    10.669 |
| Native training command elapsed, s  | 3,678.12 |  3,789.58 |
| Final full adapter checkpoint, GiB  |    0.492 |     0.492 |
| Portable adapter artifact, GiB      |    0.163 |     0.163 |

Update timing excludes validation and checkpoint publication; native command elapsed includes them and loading. Parent preflight precedes that command and is recorded separately. These are single-seed end-to-end observations, not additional bootstrap trials. Historical `c5b8d49` runs requested four CPUs; these runs requested eight. Actual CPU/NUMA placement and filesystem cache state also differed across jobs, so use the within-allocation paired trials above for the controlled optimization claim. Sampled GPU/host peaks are lower bounds; process-tree RSS may double-count shared pages, and unmapped filesystem page-cache memory was not measured.

A separate CPU replay verified the exact 2,400-row training cursor against both original checkpoints. Actual sequences were 317–362 tokens (median 331, p95 345), with two supervised tokens each and no microbatch-one padding. Every training video had grid `[4,12,20]`: eight processed temporal slots at 192×320, yielding 240 merged visual tokens. Source-frame indices/timestamps were not recorded, so the number of unique decoded source frames is unknown. This does not qualify sustained 4,096-token training.

The 1 GiB cache limit bounds retained entries, not total host memory. The native trainer's PSS stayed near 2.4 GiB, while the spawned worker's anonymous memory grew during the run; observed worker PSS peaks were 5.870 GiB for NF4 and 5.579 GiB for BF16. A separate 1,000-row CPU replay retained the identical live cache and RNG state while an end-only `malloc_trim(0)` released 2.169 GiB after garbage collection had little effect. This demonstrates freed glibc page retention in that replay. Its worker had two threads versus 32 in the live runs, so it does not prove the complete live-worker cause, bound longer-run residency, or qualify a production allocator change. No allocator intervention was applied to training.

Both final step-300 DCP/portable reload comparisons passed on text, image and video probes with `atol=0, rtol=0`. All four held-out arms then completed the same 256 rows and 512 supervised tokens per arm. Every complete scored-row dictionary and controlled input matched its archived `c5b8d49` counterpart exactly. The [four-arm reference metrics](#four-arm-held-out-diagnostic-and-release-status) were therefore reproduced: trained NF4 scored 229/256 (89.4531%) with token NLL 0.153532; trained BF16 scored 230/256 (89.8438%) with NLL 0.147372. These are teacher-forced measurements with each question's actual two, three or four choices, not free-generation accuracy. The 4.18% relative NLL regression and unsatisfied release quality gate remain unchanged.

## Qualification evidence

The original [machine-readable experiment report](evidence/native_qlora_wts_c5b8d49.json) contains the completed `c5b8d49` training and four-arm evaluation measurements, source/runtime identities, per-update metrics, and measurement limitations. The following reference results predate the performance controls above.

The experiments tested commit `c5b8d49bcc219d1cceb76bee42d4eb20daa766f7`, installed from wheel SHA256 `afbeb228bd42ba9582e53d427c20ba59d0b166706ffd98ccaa6ed73d8f149047`. Each run used one NVIDIA A100-SXM4-80GB and one torchrun process. Runtime: Python 3.12.12, Torch 2.11.0+cu128 (source commit `70d99e998b4955e0049d13a98d77ae1b14db1f45`), CUDA 12.8, Transformers 5.8.1, bitsandbytes 0.49.2, and TorchCodec 0.11.0+cpu. The host reported driver 535.129.03; the loaded CUDA compatibility driver library was `libcuda.so.570.124.06`. Both training arms used strict deterministic algorithms.

All four evaluation arms verified the same pinned base revision recorded above, source-weight manifest digest `69368621649dca3287fe9587bcac3d1d2fb6380974d9b59ade8c80aaa9879daf`, and processor/template manifest digest `9d674c4ab12ba55505f763912e1fadf3d8660125dcb774c20462743582d8b180`. The resolved contracts matched 252 language targets, 504 FP32 adapter tensors containing 43,646,976 trainable parameters, rank 16/alpha 32, and full non-reentrant activation checkpointing. NF4 used nested quantization, UINT8 packed storage and BF16 compute; excluded base components remained BF16, and optimizer moments were FP32. The matched control kept the same targets in frozen BF16.

### Exact reload and resume canary

The controlled canary used **four optimizer updates, microbatch one, accumulation two, seed 42**, with saves at updates two and four. Actual WTS sequences were at most 354 tokens, with eight processed video temporal slots resized to 192 × 320 and 240 visual tokens. The number of unique source frames was not recorded.

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

Observed evaluation sequences ranged from **304 to 365 tokens** (median 342, p95 356). Video pixel tensor shapes were `[960,1536]` for 252 rows and `[864,1536]` for four rows. The processor requested eight frames with the total pixel budgets documented above; decoded grid values/frame counts were not recorded. Actual training input-length and frame distributions were not recorded in the original GPU captures; the later exact 2,400-row CPU replay described above measured processed shapes, without establishing source-frame uniqueness. The configured 4096-token ceiling and these short evaluation sequences do not qualify sustained 4096-token training.

**The release quality gate is not satisfied.** Trained NF4 improved the diagnostic accuracy over frozen NF4 by 38.2813 percentage points, but scored one fewer correct answer than trained BF16 (−0.3906 percentage points). Its token-weighted NLL was **4.18% higher** than trained BF16, already exceeding the proposed 2% relative margin at the point-estimate level. This is an observed quality regression in addition to incomplete release sampling; the required one-sided confidence bound was not evaluated for that archived diagnostic.

The [predeclared release protocol](qlora_native_vlm_design.md#reproducible-benchmark-and-release-evidence) requires paired seeds, independent held-out evaluation with clustered paired uncertainty, and relevant text/image/video capability regressions. The three-seed development and final regression/diagnostic comparisons are complete. The NLL and selected-regression accuracy gates failed; independent final groups and broader capability checks remain unresolved. These historical results did not qualify smaller GPUs, other distributed strategies, merged exports, or serving backends. The later [effective-base BF16 export](#effective-base-dense-export) has separate, scoped conversion and reload evidence; merging onto the untouched BF16 base, other export formats and serving compatibility remain unqualified. Release qualification remains open.
