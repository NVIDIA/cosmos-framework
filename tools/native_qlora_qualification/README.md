# Native QLoRA qualification tools

These tools orchestrate the native Cosmos SFT entrypoint and inspect its artifacts. They do not contain another training loop, submit scheduler jobs, or declare release support. Use the sealed runtime and pinned model described in [the native QLoRA guide](../../docs/post_training/native_qlora.md). Run accelerator commands inside a one-GPU A100 80GB allocation. Keep the tested wheel overlay first on `PYTHONPATH`, the workflow source root second, and the working directory outside the source checkout.

## Data and selection

`split.py` checks the prepared WTS conversion hashes, joins related recording families and identical video content, and records earlier evaluation exposure. Supply every previous score file and a JSON list of additional probe IDs. It excludes all validation-linked and selected development groups from new training. Development groups may have been used by historical training; every new model must start from the pinned base.

```bash
python -m tools.native_qlora_qualification.split \
  --data-root "$PREPARED_WTS" --exposed-scores "$OLD_SCORES" \
  --exposed-ids "$PROBE_IDS" --exposed-validation-prefix 64 \
  --development-groups 12 --seed 20261030 --output "$NEW_SPLITS"
```

Keep generated rows and media private. Preserve the split manifests and their SHA256 digests before scoring. The original 256-row selection remains a regression set. A single untouched recording group is labeled `untouched_diagnostic`; it cannot establish independent-cluster uncertainty. Do not promote it to a final release set. `score.py --all-rows` requires the exact split manifest and external digest. Final and untouched-diagnostic scoring additionally require a sealed selection receipt binding the chosen native configuration and development evidence.

`score.py --preprocessing-cache-bytes 1073741824` opts into the existing native CPU preprocessing cache with a 1 GiB retained-entry bound. The default is zero; this option starts no prefetch workers and preserves row order, processor identity, masks and scoring arithmetic. Video preprocessing can be reused across questions sharing verified media. The opt-in summary records the cache policy and counters; its elapsed time includes source hashing, loading/reload, cold cache population, scoring, integrity checks and cache close. Use the complete outer process time for comparisons, including startup and teardown, and offer the same option to both precision arms. The cache bound is not a bound on total host residency. The [completed accelerator comparison](../../docs/post_training/evidence/native_qlora_scorer_cache_1ac6a7f.json) establishes exact within-precision scores for 64 exposed development rows and a descriptive scoring-time benefit in both arms. Its two repeats per arm do not establish full-development or training speed, and the default remains unchanged.

`quality.py` verifies paired row/input/mask/media identities and computes token-weighted NLL, restricted-choice accuracy, frozen-base comparisons, task slices and a deterministic recording-group bootstrap for one training seed. `quality_confirmation.analyze_confirmation` separately requires all three paired seeds (17, 42 and 123), both fixed frozen references and the sealed connected-recording labels. It reports a seed-then-group hierarchical bootstrap plus a crossed sensitivity that shares group draws across seeds. Both use 10,000 replicates, RNG seed 2026 and the unchanged 2% NLL, one-percentage-point accuracy and positive-adaptation bounds. Both must pass before the conditional statistical criteria pass. Slice intervals remain exploratory and no difficult examples are removed. Artifact/checkpoint identity validation is an independent prerequisite; this statistical function never declares release qualification. An exposed development or regression result is not untouched final confirmation.

`layer_sensitivity.py` is a training-only diagnostic for the frozen zero-B NF4 model. It requires the SHA-bound 64-row calibration manifest and uses the native scorer and bounded prepared-input cache. It temporarily restores the seven language projections of one block to their original BF16 source while retaining the same adapter objects, measures actual task outputs, and restores the original packed matrices in `finally`. All 36 blocks are examined separately, followed by an exact original-model control. It constructs no optimizer or checkpoint and does not qualify a mixed-precision exclusion recipe. Run `test_layer_sensitivity_gpu.py` on the real accelerator first and require its one test to pass without skips. Parent timing and process-memory sampling must include source admission, cache preparation, loading and restoration; the tool's per-case forward peaks do not cover those costs.

## Independent confirmation analysis

`confirmation_design.py` implements prospective paired-seed analysis without changing the historical three-seed reports. It checks both unchanged noninferiority margins and separately retains positive adaptation over frozen NF4. It rejects missing input/artifact identities, cross-split recording/media/session/event links, missing task coverage, overlapping pilot/final seeds, and a final design rejected by its bound pilot planning scenarios. These checks establish consistency of supplied evidence; they do not certify acquisition provenance, calibrated statistical power or release support.

```bash
python -m tools.native_qlora_qualification.confirmation_design \
  --inputs "$CONFIRMATION_INPUTS" --output "$NEW_CONFIRMATION_RESULT"
```

The JSON input bundle has `role` (`pilot` or `confirmation`), references named `spec`, `inventory`, `provenance_audit`, `task_taxonomy`, `exposure_and_relationship_source`, and `artifact_validation`; `trained` maps each declared seed string to `nf4` and `bf16` score references; `frozen` contains the two fixed reference score files. Confirmation additionally requires `pilot_analysis`. Every reference has `path` and the SHA256 of its file bytes. Outputs are exclusive. `proposal(task_types, recipe_sha256)` supplies the candidate specification schema with unresolved evidence pins; it does not authorize acquisition or jobs.

The inventory is a list containing each row's `id`, media `sha256`, `raw_row_sha256`, `role` (`exposed`, `pilot`, or `confirmation`), `task`, reviewed `session` and reviewed `event`. Include all historical exposure. Unknown relationships refuse; connected components must not cross roles. `inventory_sha256` is the canonical `common.digest_json(inventory)` digest, while provenance, taxonomy and pilot-result pins are file-byte hashes. The ordered taxonomy must contain the same 22 task types as the specification. Use the existing native scorer with `--all-rows --split development` for a separately sealed pilot manifest and `--split final` for confirmation. The statistical roles are not additional scorer arguments. Final scoring requires the existing prior selection receipt binding configuration, pilot/development evidence and the final manifest.

The externally reviewed metadata contracts are:

- Taxonomy: protocol `cosmos.qlora-confirmation-task-taxonomy.v1` and the exact ordered `task_types`.
- Exposure source: the canonical `inventory_sha256` and a nonempty `known_exposed_rows` list matching historical rows on ID, row/media hashes, task, session and event.
- Provenance: protocol `cosmos.qlora-confirmation-provenance.v1`, `complete: true`, matching inventory and taxonomy hashes, the actual exposure-source file hash in `exposure_and_relationship_source_sha256`, and `unresolved_relationships: []`.
- Artifact validation: protocol `cosmos.qlora-confirmation-score-artifacts.v1`, `complete: true`, the exact `recipe_sha256`, ordered `paired_seeds`, `inventory_sha256`, and `score_bundle_sha256 = common.digest_json({"trained": bundle["trained"], "frozen": bundle["frozen"]})`. The issuing review must verify completed native training, checkpoint/portable reload, source and selection identities, and score artifacts. The analyzer independently checks score rows, masks, tokens and media, but does not load checkpoint tensors.

There is no automatic producer or approval toggle for these external reviews. Matching their hashes does not prove their assertions. After a separate pilot, revise a rejected sample-size design before final outcomes. Seal the combined exposed/pilot/final inventory and rerun the metadata-only pilot analysis to bind that final design, reusing the pilot scores. Confirmation requires this exact pilot result and recomputes both declared variance scenarios. The approximate planning checks are necessary eligibility conditions, not proof of bootstrap power. Neither the provisional 180-group estimate nor the newer 400-group candidate is established sufficient; see the [audit and compute assumptions](../../docs/post_training/native_qlora_qualification.md#statistical-audit-and-executable-confirmation-design).

## Native experiments

`protocol.py` defines the strict JSON `Experiment` schema and bounded allocation accounting. The two precision arms use the same native overrides, differing in their base recipe and run name. Training, adapter initialization and validation seeds are explicit. Parameter, arithmetic and optimizer precision remain separate.

`run.py kernel-gate` verifies the wheel overlay and runs all 14 real CUDA tests plus the existing higher-precision native CPU regression without skips. `run.py experiment` requires that gate for the same build, then measures native training, portable adapter publication, fresh DCP/portable probes, exact reload comparison and sealed-split scoring. Each output directory is exclusive; unsuccessful attempts retain commands, logs and exit status. A successful completion receipt is execution evidence, not a quality-gate pass. Fresh `train-eval` and `frozen` actions reject an external checkpoint; `evaluate` requires one explicitly.

```bash
python "$WORKFLOW_SOURCE/tools/native_qlora_qualification/run.py" kernel-gate \
  --build "$BUILD" --output "$NEW_GATE" --time-binary "$GNU_TIME"

python "$WORKFLOW_SOURCE/tools/native_qlora_qualification/run.py" experiment \
  --spec "$EXPERIMENT_JSON" --build "$BUILD" --kernel-gate "$NEW_GATE/completion.json" \
  --model "$PINNED_MODEL" --data-root "$TRAINING_DATA" \
  --score-data "$NEW_SPLITS/development.jsonl" \
  --split-manifest "$NEW_SPLITS/development.manifest.json" --split development \
  --output "$NEW_RUN" --time-binary "$GNU_TIME"
```

`TRAINING_DATA` contains the sealed fresh `train.jsonl` and a `validation.jsonl` copy of the development data. Export `PYTHONHASHSEED` equal to the experiment seed and `CUBLAS_WORKSPACE_CONFIG=:4096:8` before launching. The runner records input digests and the complete native dotlist. Pair precision arms sequentially within the same scheduler allocation, counterbalance their order across repeats, and record physical CPU placement. Account for failed allocations as well as successful ones. The 2026-10-30 continuation uses a predeclared maximum of 48 GPU allocation-hours, four concurrent GPUs, and 64 core-hours for separate CPU diagnostics.

`lifecycle.py` runs a separate four-update recovery qualification for each selected numerical configuration. It compares uninterrupted execution with both a completed-boundary interruption and process loss after a real accumulating microbatch. Only explicit checkpoints from that attempt may resume. The immediate next update and final update must match model, optimizer, RNG, cursor, controls and numerical metrics exactly. Fresh multimodal DCP and portable probes must also match exactly. Filter experiment output/horizon fields before passing repeated `--override` arguments: the recovery runner owns its output names and fixes horizon four, warmup one and save cadence one.

For a controlled full-versus-disabled activation-checkpointing experiment, `lifecycle.compare_activation_checkpointing_checkpoints(reference, candidate, step=...)` compares trusted own-run artifacts in the allocated sealed runtime. It acknowledges only the three explicit activation-checkpointing identity fields, then requires all remaining metadata, model/optimizer values and native RNG/data/control state to match. The observation leaves both artifacts intact and does not make their execution identities compatible for resume. Each mode still needs its own interruption and fresh-reload qualification; CPU comparison tests establish neither full-model equivalence nor a speed or memory benefit.

### Scheduled full-development scoring

Native `rl.pause_after_step=N` stops an attempt at absolute optimizer step `N`, after ordinary validation and a complete resumable checkpoint. It leaves `rl.max_steps`, the LR horizon and validation schedule unchanged. Resume from the returned checkpoint with a larger pause boundary or `None`; a stale boundary fails. It neither saves partial accumulation nor weakens build/base/execution identity checks. The [native `149c2cf` accelerator qualification](../../docs/post_training/native_qlora_qualification.md#committed-boundary-pauses-on-native-149c2cf) covers four-update pause/recovery and full-development scoring for both precision arms. The separate [completed scheduled study](../../docs/post_training/native_qlora_qualification.md#completed-scheduled-development-ttq-on-native-149c2cf) now covers first-pass stopping at update 150 for seed 17 and complete 300-update horizon censoring for seed 123; it establishes no time-to-quality superiority. Prior `4cd8988` results remain separate.

The existing `experiment` command accepts `--validation-target "$TARGET"` for fresh **development-only** runs. Before training, seal a target and increasing evaluation grid ending at the experiment's unchanged horizon. This example retains the existing 300-update development point target; it is not the paired/grouped noninferiority gate:

```bash
python - "$NEW_SPLITS" "$TARGET" <<'PY'
import hashlib, json, pathlib, sys
root, target = map(pathlib.Path, sys.argv[1:])
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
value = {
    "protocol": "cosmos.qlora-validation-target.v1",
    "evaluation_steps": [50, 100, 150, 200, 250, 300],
    "expected_rows": 584,
    "minimum_correct": 540,
    "maximum_token_ce": 0.12357161758062493,
    "score_data_sha256": digest(root / "development.jsonl"),
    "split_manifest_sha256": digest(root / "development.manifest.json"),
}
with target.open("x") as stream:
    json.dump(value, stream, indent=2)
    stream.write("\n")
PY
```

Add the flag to the experiment command above. With the default `--evaluation-mode restart`, the runner pauses at each scheduled boundary, verifies that checkpoint, scores every sealed row in a fresh process, and resumes only if the fixed point target has not been met. Before scoring or resuming, it captures the native completion record, applied-update history and checkpoint inventory and refuses work beyond the requested boundary; merely finding an older checkpoint is insufficient. It checks target/data/manifest identities before training, before scoring and after scoring, binds the first score to the ordered sealed partition, and checks processed-input identity across observations. Complete scoring and inspection failures abort; horizon nonattainment is recorded as censored. The first observed passing checkpoint is not an interpolated crossing between evaluations. Final portable publication and reload probes run separately after the selected endpoint.

This measures a **segmented workflow**: each training restart reloads the model and starts a fresh preprocessing cache. Cadence scoring separately passes the experiment's `cache_bytes` to the scorer's existing `--preprocessing-cache-bytes` option for both precision arms. Each scoring process starts its own cache, with no prefetch workers; it shares no entries with training or earlier evaluations. Non-cadence scoring keeps its historical uncached default. Give both precision arms the same cadence and do not substitute continuous-run timings. `target-result.json` includes all preceding verification, loading, native preprocessing, training, checkpoint and evaluation time from runner start. Pre-existing base/data acquisition, conversion, environment/build preparation and scheduler waiting remain outside that clock and must be reported separately. Stage records and `completion.json` retain final export/reload and complete runner costs. No time-to-quality superiority, complete cold-start cost or release qualification follows from CPU orchestration tests or a single point-target crossing. Freeze seed pairing, scoring cadence, resource caps and stopping rules prospectively; apply the unchanged joint quality analysis separately.

The [rank-32 WTS settings](../../docs/post_training/native_qlora.md#reproduce-the-selected-development-recipe) were selected on native `149c2cf`, with disabled checkpointing and 3 GiB caches in both arms; historical full/1 GiB evidence remains separate and generic defaults stay unchanged. The frozen 3,600-second secondary endpoint selects the last complete full-development score available by that runner deadline. In seed 123, NF4 update 200 completed at 3,591.227 seconds; BF16 update 200 completed at 3,630.192 seconds, so BF16 uses update 150. Preserve these boundaries and early stopping; neither interpolation nor quality at exactly equal consumed compute is justified. All failed allocation costs remain charged.

### Native in-process evaluation

With current native `09e99c3`, a matching sealed workflow and its accepted build gate, select one resident training process for the same frozen target. Reproducing the completed timing study requires its original `dadfc89` build; exact resume retains the original build identity:

```bash
python "$WORKFLOW_SOURCE/tools/native_qlora_qualification/run.py" experiment \
  --spec "$EXPERIMENT_JSON" --build "$BUILD" --kernel-gate "$NEW_GATE/completion.json" \
  --model "$PINNED_MODEL" --data-root "$TRAINING_DATA" \
  --score-data "$NEW_SPLITS/development.jsonl" \
  --split-manifest "$NEW_SPLITS/development.manifest.json" --split development \
  --validation-target "$TARGET" --evaluation-mode native \
  --output "$NEW_RUN" --time-binary "$GNU_TIME"
```

**Prior-build four-update GPU qualification passed on `2f7a66d` for both precision arms at dropout zero and 0.05.** The measured `dadfc89` passed its own 14 real CUDA cases plus higher-precision native regression and both dropout-0.05 four-update diagnostic bridges, including exact state/score comparisons and fresh DCP/portable reload. Its [completed seed-17 pair](../../docs/post_training/native_qlora_qualification.md#seed-17-resident-evaluation-comparison) adds exact 150-update state/score agreement and observed full-development target time: NF4 49.53 → 30.37 minutes and BF16 48.44 → 28.01 minutes for restart → native. This is a shared workflow improvement, not better learning or a quantization speed advantage. The [seed-123 comparison](../../docs/post_training/native_qlora_qualification.md#completed-seed-123-resident-comparison) and [resident mature recovery](../../docs/post_training/native_qlora_qualification.md#completed-resident-mature-recovery) subsequently completed on `dadfc89`; neither precision reached the target at seed 123. The dropout hypothesis is rejected below. These studies remain prior-build evidence; current `09e99c3` has its own objective/recovery and dense-export checks. The command requires its own exact-build gate; earlier restart evidence and observational cross-build comparisons never authorize cross-build production resume.

The runner’s native mode is restricted to fresh `train-eval` with a development target; direct native resume remains available under the unchanged checkpoint identity. WTS requires one native adapter process, deterministic ordinary token-mean CE and synchronous local DCP full-validation checkpoints; its evaluation grid must use validation boundaries and end at `max_steps`. It binds `rl.validation.wts` to the exact source/manifest hashes, row count, grid, thresholds and experiment cache capacity. Its separate persistent evaluation cache shares no entries with training. With `dadfc89`, each native score summary records cumulative cache counters for that evaluator attempt. The four-row bridges observed four sample misses followed by four sample hits without further misses. Do not sum snapshots, treat them as total host memory, or add preprocessing seconds to validation wall time; resumed attempts start a fresh cache. The full-data scorer preserves the training RNG, mixed module modes and cursor; ordinary native loss validation still runs. Evaluation history is deterministic checkpoint state, while paths and wall clocks stay in sidecars. Exact resume requires unchanged evaluation identity.

The runner verifies every scheduled score, checkpoint and applied-update boundary. First-pass time ends after durable score files and full checkpoint publication/readback. It includes initial runner verification/preflight, loading and preceding training/evaluation. Final target-index serialization and cleanup are included in native process completion; later acceptance checks and fresh portable export/reload are included in complete runner time. Acquisition, environment/build preparation and queue waiting remain separate. Missing or failed scoring is an operational failure, not horizon nonattainment. No speed, quality or release claim follows from the CPU tests.

Native TTQ fixes `rl.validation.wts.stop_on_target=true`. A direct native learning screen may instead use `rl.validation.wts.stop_on_target=false rl.pause_after_step=100`, keeping `max_steps=300`, its LR schedule and the full 50-step grid: both update-50 and update-100 evaluations run even after a pass. Record it as a paused screen, not completed 300-update TTQ. Optional experiment JSON `"dropout": 0.05` emits `model.model_config.adapter.dropout=0.05` for either precision recipe; default zero dropout and all historical results remain unchanged. Its four-update correctness checks do not establish a learning benefit, and it cannot be changed during exact resume.

The first screen failed private acceptance after its first case completed 100 updates and fresh scoring: resolving the output mount alias changed the checkpoint's spelling before a strict pause-path comparison. The private correction preserves the admitted absolute path; native code and the public guard are unchanged. Twelve CPU checks cover the captured case and strict negatives. Its retry completed all four training and fresh-scoring cases, but Lustre `EDQUOT` prevented the final acceptance/completion records. Read-only parity checks subsequently passed for all four 584-row score pairs. NF4 dropout improved accuracy by 13 answers but increased CE by 0.252%, failing the frozen 1%-reduction guard; discard the hypothesis. This does not repair the missing original case-finish clock or pass its memory gate. The [evidence](../../docs/post_training/evidence/native_qlora_native_evaluation_dadfc89.json) preserves both failed allocations, their costs and all outcomes. Do not change deadlines or manufacture an original completion during recovery. At that historical handoff, two further pre-training write failures blocked admissions. Subsequent compute-side write checks, clean seed-123 timing and `dadfc89` mature recovery completed, as linked above. The old failures and missing original clocks remain unchanged; point-in-time storage checks do not reserve future capacity.

The [completed answer-smoothing screen](../../docs/post_training/native_qlora_qualification.md#completed-answer-smoothing-development-screen) passes its frozen followup gate on native `09e99c3`: NF4 gains three correct answers, reduces hard-gold CE by 2.382%, and retains 9.893 GiB lower sampled complete-case GPU memory against selected BF16 epsilon zero. The [six-case followup](../../docs/post_training/native_qlora_qualification.md#completed-two-seed-smoothing-followup) is also complete: seed 17 reaches the target at update 250 for NF4 smoothing and BF16 dropout, versus 150 for fixed BF16 zero; every seed-123 recipe is unreached at 300. NF4 keeps substantial memory savings but does not demonstrate a target-time advantage or a consistent joint quality gain. Default smoothing remains zero. Both fixed BF references, the positive original screen, failed NUMA admission, historical quality guards and independent-data requirement remain recorded; no followup promotion gate is invented retrospectively.

Sealed workflow `add7398` accepts optional Experiment JSON `"wts_answer_smoothing": 0.02` and forwards `rl.wts_answer_smoothing` to the existing native entrypoint. Zero is omitted from historical specification/override identities. All 76 CPU protocol/orchestration checks passed; the native `09e99c3` wheel, hard-gold scorer and TTQ loop are unchanged. Use the existing `--evaluation-mode native --validation-target` command with fresh, prospectively fixed seed specifications; this workflow extension does not authorize cross-objective resume or imply equal consumed compute.

## Selection and comparison

`selection.py` validates all eight predeclared development candidates against their exact specifications, final checkpoints, scorer identities and complete development rows. It retains failed candidates and refuses selection when a precision baseline is incomplete. Development selection does not authorize untouched scoring.

`execution_selection.py` separately validates the short-video full/no-checkpointing calibration: each precision starts at microbatch one and advances to two or four only after the preceding cell satisfies the observed 24 GiB device-process budget. It recomputes native timing and memory evidence, verifies accepted-boundary recovery and rejects incomplete calibration. The sealed rule chooses the lowest mean update time after warmup, then lower observed peak, smaller microbatch and full checkpointing for exact ties. It combines those execution choices with the development-selected learning rate, rank and alpha to emit six 300-update confirmation specifications at seeds 17, 42 and 123. Long/media fixtures remain separate scope tests; this short screening result is not a full-run speed or quality result.

```bash
python -m tools.native_qlora_qualification.execution_selection \
  --inputs "$PINNED_EXECUTION_INPUTS" --inputs-sha256 "$INPUTS_SHA256" \
  --evidence-root "$QUALIFICATION_EVIDENCE" --output "$NEW_SELECTION"
```

The input manifest binds all consumed files and declares any recorded-root aliases needed for a local evidence mirror. Incomplete calibration produces only a diagnostic decision; complete decisions produce the locked specifications and execution receipt. Keep executable receipts private when they contain filesystem paths.

`performance.py` keeps controlled comparisons strict. A separately tuned `best_fit` comparison requires an externally pinned v2 pair plan binding the confirmation/development plans and both selection decisions. Only selected learning rate, rank, alpha, microbatch and checkpointing mode may differ between precision arms. All three declared seeds, paired placement and valid memory evidence remain required. An over-budget BF16 control stays visible in controlled comparisons; it cannot qualify as a 24 GiB best-fit candidate. The observed budget does not establish support on a physical 24 GiB accelerator.

## Runtime and memory diagnostics

`worker_memory.py` compares the inherited worker environment with a fresh-process BLAS-thread intervention. Use sequential AB/BA pairs on matched CPU placement. It checks exact processed tensor and RNG identities, live-cache accounting, process-tree memory and thread/allocator provenance. Its end-only garbage collection and allocator trim are disposable diagnostics, not a production policy.

The opt-in periodic diagnostic uses `--trim-every-rows 0` for a matched sham and `--trim-every-rows 128` for worker-local `malloc_trim(0)`. Both modes keep inherited thread settings, verify the same live cache and RNG at every drained 128-row boundary, and perform no explicit GC or cache eviction. Both receive the same separately timed end-only trim. Omitting the option preserves the original thread comparison and end-only GC/trim behavior. Periodic comparisons require one complete sham and one complete treatment with identical placement, environment, processed tensors, cursor and cache identities, plus clean child exits. Post-result shutdown stamps distinguish pipeline close, monitor join and leaving the child main.

Run from a tools-only staging directory against the verified wheel overlay and prepared processor/data bundle; the launcher must verify package bytes and import resolution before these commands. Use a disposable CPU allocation with eight physical cores for the recorded configuration. Keep the inherited thread environment identical within each pair and use a fresh output root for every repeat:

```bash
for interval in 0 128; do
  python -m tools.native_qlora_qualification.worker_memory run \
    --config cosmos_framework/posttrain/configs/sft_qwen3_vl_8b_qlora.py \
    --model-path "$SFT_MODEL_PATH" --data-root "$WTS_ACCEPTED_DATA" \
    --build-receipt "$BUILD_RECEIPT" --output "$CPU_EVIDENCE/trim-$interval" \
    --thread-policy inherited --trim-every-rows "$interval" \
    --rows 1000 --seed 42 --repeat 0 --expected-physical-cpus 8 \
    --timeout-seconds 1720
done
python -m tools.native_qlora_qualification.worker_memory compare \
  --run "$CPU_EVIDENCE/trim-0" --run "$CPU_EVIDENCE/trim-128" \
  --output "$CPU_EVIDENCE/comparison.json"
```

Counterbalance with `128 0`, `--repeat 1` and another fresh output root. The timeout covers startup, replay and shutdown; allocation wall time must cover both children and setup. The [completed CPU evidence](../../docs/post_training/evidence/native_qlora_worker_memory_20261030.json) has SHA256 `72fbd667c01c5b7bab357de3b6eff4061dd5f0019b7e58a24a9ca6374cc7837a`. Its two clean pairs reduced observed boundary worker PSS by about 1.08 and 1.11 GiB while preserving all processed outputs and state. The failed initial attempt remains descriptive-only evidence. Replay time increased by about 12 and 210 seconds respectively; order, cache warmth and system load remain confounded, so no speed improvement is claimed.

The promoted periodic replay, maintenance and shutdown functions preserve the qualified diagnostic's syntax-tree semantics; [source-provenance tests](tests/test_worker_memory_periodic.py) pin those functions and the legacy replay separately. Promotion adds explicit mode dispatch, rejects mixed protocols or incomplete exits, and requires at least one periodic boundary. The evidence is CPU preprocessing replay only: no model, production reclamation policy, live-training memory bound or larger-media benefit is established. Direct trim-call time excludes deferred page-fault and training costs.

`workloads.py` prepares actual processor-measured video workloads, including 3,840–4,096-token examples with at least 1,024 supervised tokens and a larger 16-frame input. The long task is an explicitly synthetic textual ledger alongside real media. These fixtures establish numerical, recovery and resource behavior; they do not establish WTS task quality. No token padding or truncation is used to manufacture sequence lengths.

The [published rank-32 NF4/BF16 combined-input recipes](../../docs/post_training/native_qlora.md#combined-longer-context-and-larger-video-recipe) now have completed `09e99c3` 20-update and separate four-update recovery evidence. They use full checkpointing and the sealed `long4k_video16_large` profile; the runtime command does not itself execute the separate recovery suite.

Use `workloads.py census` to measure every unchanged row in the selected ordinary training source. It uses the native cursor and processor, checks ordered membership against the grouped training manifest, and records actual token counts, video grids, tensor shapes and hashes. The selected census policy is microbatch one, accumulation eight, one preprocessing worker and a 3 GiB cache. Run it in the sealed environment with the same `SFT_MODEL_PATH` and `SFT_DATA_ROOT` as training:

```bash
python tools/native_qlora_qualification/workloads.py census \
    --config cosmos_framework/posttrain/configs/sft_qwen3_vl_8b_qlora.py \
    --source-manifest /absolute/grouped-training/train.manifest.json \
    --build-receipt /absolute/build/receipt.json \
    --output /absolute/new-input-census --expected-rows 4611 \
    --override rl.preprocessing_cache_bytes=3221225472 \
    --override rl.preprocessing_workers=1 \
    --override rl.preprocessing_prefetch=true
```

The source manifest must be the actual training manifest from the grouped split, with matching ordered rows and media digests. Completion requires unchanged source, media and processor bytes and unchanged parent/worker RNG. This CPU census constructs no model and does not qualify GPU training, long-context support or task quality. A partial output without `manifest.json` is incomplete; retain it and use a new destination for another attempt.

The [accepted native `149c2cf` census](../../docs/post_training/evidence/native_qlora_input_census_149c2cf.json), using sealed workflow `d5d323c`, covered all 4,611 ordinary rows and the final three-row window: 317–362 sequence tokens, two supervised tokens each, video grid `[4,12,20]` and pixel tensor shape `[960,1536]`. Source/media/processor and parent/worker RNG guards passed while the cursor intentionally advanced through the source. No model or CUDA was initialized. The configured GPU carrier cost 738 GPU-seconds/11,808 CPU-seconds; its 611.396-second internal census clock is nested. This measures source preprocessing, not whether every row was trained or whether longer/larger inputs fit GPU training. The [tracked development accounting](../../docs/post_training/native_qlora_qualification.md#tracked-development-allocation-cost) retains prior search/failure charges and unmeasured acquisition/local CPU costs.

With `COSMOS_QLORA_METRICS_FILE` set, `score.py` records `scoring_start` with an allocator-peak reset followed by `scoring_complete` after row-file fsync, frozen-base/adapter/data/media integrity checks and cache close. The interval excludes summary construction/serialization and process-group teardown; it includes the resident model, not just incremental allocations. If a failure occurs after scoring starts, it attempts a `scoring_failed` record after cache cleanup and retains the original exception if cleanup also fails. The consumer labels this interval `scoring_incomplete`; a measurement event alone never proves command completion. These boundaries have CPU contract coverage and recorded CUDA evidence in the native `149c2cf` engineering and scheduled-scoring studies. Their recorded intervals remain distinct from complete-workflow memory. Historical scoring files that recorded only loading still have unknown full-scoring allocator peaks.

`monitor.py` samples owned device-process memory and host process-tree RSS/PSS. Sampled peaks are lower bounds; summed RSS may count shared pages twice. Native CUDA phase peaks end at their recorded event and reset afterward. Preprocessing processing/wait counters overlap training and each other and must not be added to elapsed wall time. Report loading, first-step/warmup, training, validation, checkpointing, publication, reload and scoring costs separately, including complete training-to-evaluation elapsed time.

`monitor_host.py` records raw process identity, memory reads, errors and bounded retry attempts. Its version 3 receipt uses admitted ancestry, PID, birth time and a held process pidfd as identity; mutable `/proc` inode ownership remains diagnostic metadata. Known descendants remain tracked after reparenting. A zombie leader alone does not prove whole-process exit. Exit-compatible unreadability may wait at most 350 ms on the same pidfd, retaining the initial poll and reserving 250 ms for a complete retry within the original 750 ms budget. Permission errors, unexplained reads, PID reuse and stage races remain incomplete; a still-live group never becomes zero memory. Synchronous proc reads and scheduling are not hard real-time bounded. Legacy receipts retain their original interpretation and incomplete flags. The [actual 61-test host proof](../../docs/post_training/evidence/native_qlora_collector_wait_94289a2.json) does not change model arithmetic, repair historical measurements or qualify a new training configuration.

Version 4 can read the shared address space through one verified live thread when a zombie process leader's memory read returns `ESRCH` while the held pidfd still reports a live process. It binds the thread to the same thread group, birth identity and held task-directory descriptor, and verifies identities and exit state again after reading. The address space is counted once, not once per thread. Permission failures, exec/exit races and missing identity evidence remain incomplete. Actual positive, exit and exec fixtures must pass on the target kernel before relying on this fallback; the [completed CPU evidence](../../docs/post_training/evidence/native_qlora_parallel_cpu_105d314.json) is separately scoped. Historical version 3 samples cannot gain this missing proof retrospectively.

Version 5 permits a fresh whole collection after a failed live-thread `ESRCH` read only when the same held process pidfd actually reports release. Version 6 also permits a bounded wait at this post-discovery boundary, after validating the complete leader/thread identity proof. It retains the initial failed read and poll, waits at most 350 ms within the existing 750 ms budget, and reserves 250 ms for recollection. No release event means an incomplete sample; unreadability never becomes an assumed zero. Existing version 1–5 records keep their original interpretation. The [version 6 target CPU proof](../../docs/post_training/evidence/native_qlora_collector_v6_eb9c436.json) passed 112 tests without skips on Linux 5.15, including three actual-process properties. Thread-read ESRCH is controlled in the release fixtures; the held pidfd and process release are real. Seven historical incomplete rows remain incomplete. This does not establish complete full-model shutdown telemetry: post-read discovery ENOENT and other missing identity evidence still fail closed.

Version 7 additionally permits recollection when final discovery returns exactly `ENOENT` after the held-thread proof already recorded the same zombie leader identity, provided the same held process pidfd subsequently proves release within the original deadline and retry reserve. The failed interval still contributes no memory. A missing earlier identity, permission error, changed birth/thread group, exec transition or unproven release remains incomplete. The [version 7 target CPU proof](../../docs/post_training/evidence/native_qlora_collector_v7_bed3d03.json) passed all 132 tests without skips on Linux `5.15.0-1032-oracle`, including four actual-process properties. The release fixtures use controlled thread-read `ESRCH`; child exit, reap, discovery `ENOENT` and held-pidfd release are real. All seven historical incomplete rows remain incomplete. This new allocation took 166 seconds with 16 CPUs (2,656 allocated CPU-seconds) and no GPU. It does not qualify complete full-model shutdown telemetry. The completed version 6 evidence and its separate costs, and all historical version 1–6 classifications, are retained.

The current monitor CLI selects version 10: a 4.5-second acquisition budget with
the same maximum 3.5-second PIDFD exit wait as version 9. This leaves more time for
a new complete membership/read pass after a slow exit-time `/proc` read. Actual
release proof and a fresh complete observation remain mandatory; live unreadable
processes and deadline overruns remain incomplete. The five-second stage-lock
deadline is unchanged. Blocking kernel reads can still overrun these budgets, so
this is not a hard real-time guarantee. Acquisition can delay the next stage;
include that cost in all measured arms. Historical version 9 retains its original
four-second budget, including the retained failed sample from job `38536188`.
The CPU tests distinguish an injected slow-read fixture with real PIDFDs from an
optional actual 8-GiB process teardown (`COSMOS_RUN_LARGE_EXIT_TEST=1`); neither
establishes complete telemetry for a future full-model run.

### Checkpointed dequantization experiment

The opt-in [CUDA fixture](tests/test_saved_dequant_cuda.py) evaluates a test-only alternative to the pinned bitsandbytes backward path: dequantize a frozen matrix and use ordinary `F.linear` inside the actual native full non-reentrant checkpoint wrapper. Checkpoint hooks own the temporary dense tensors; the fixture never caches them on modules. It tests two synthetic blocks with the seven real language-matrix shapes, nonzero FP32 adapters, input gradients, three optimizer updates, frozen quantization state, and actual storage lifetime. The original inference path is retained. No attention, dataset, full VLM, native feature flag or end-to-end speed claim is involved.

Run only in the recorded A100 80GB allocation with the exact sealed native build and backend sources pinned in the fixture. Verify build/import placement with the existing runner first. Supply a new report path in an existing output directory:

```bash
export COSMOS_RUN_SAVED_DEQUANT_CUDA=1
export COSMOS_SAVED_DEQUANT_BUILD="$BUILD"
export COSMOS_SAVED_DEQUANT_REPORT="$NEW_REPORT"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42
python -m pytest --noconftest -c /dev/null -q \
  "$WORKFLOW_SOURCE/tools/native_qlora_qualification/tests/test_saved_dequant_cuda.py::test_actual_saved_dequant_full_checkpoint_cuda"
```

The explicit opt-in requires one real passing CUDA case; a skipped test does not qualify it. The report separates instrumented lifetime/correctness checks from uninstrumented ABBA timings with two warmup and five measured updates per round, two rounds per arm. Both packed test models remain resident, so their allocator peaks are not full-model training footprints. Failed backward terminates the process; in-process retry is unsupported. No-checkpointing, selective checkpointing and compilation are excluded from this mechanism. Any later native integration still needs its own full-model quality, memory, reload/recovery and repeated end-to-end comparison.

Multi-GPU training, other model families, other quantizers and other adapter methods remain outside this qualification.

## Native Alpamayo development analysis

`alpamayo_quality.py` consumes the frozen Alpamayo 512-update study after all nine
arm/seed runs complete. Pass each native `result.json` with a separate `--run`:

```bash
python -m tools.native_qlora_qualification.alpamayo_quality \
  --validation "$COHORT/alpamayo-validation.jsonl" \
  --manifest "$COHORT/cohort-manifest.json" \
  --target "$BF16_CALIBRATION/target.json" \
  --run "$NF4_17/result.json" --run "$NF4_42/result.json" --run "$NF4_123/result.json" \
  --run "$BF16_17/result.json" --run "$BF16_42/result.json" --run "$BF16_123/result.json" \
  --run "$HEAD_17/result.json" --run "$HEAD_42/result.json" --run "$HEAD_123/result.json" \
  --output /new/alpamayo-development-analysis.json
```

Keep `alpamayo-train.jsonl` beside validation, and retain referenced native
checkpoints, evaluations, execution protocols and workflow clocks. Admission
receipts reconstruct the native media-content cursor fingerprint; the JSONL byte
hash is a different identity. The analyzer validates the trained configuration
before comparing it with frozen-base calibration metadata.

The protocol fixes equal clip/block/seed weighting, 10,000 crossed seed/block
draws with seed 70241, and no within-block clip resampling. It reports both the
fixed update-512 endpoint and the latest published checkpoint at or before the
earlier NF4/BF16 final-publication time, including each arm's publication lag.
ADE/FDE relative and absolute guards remain joint requirements. All unreached
targets remain in the result. Calibration and complete workflow costs are retained
separately; no attainer-only average or interpolated crossing is produced.

These 32 platform/country/month blocks have unverified recording/event independence.
Development intervals do not establish final preservation, closed-loop driving,
serving latency or a population recipe claim. The frozen protocol and existing
negative results remain unchanged.
