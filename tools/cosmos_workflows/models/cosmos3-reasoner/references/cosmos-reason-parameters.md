# Framework reasoner parameters and troubleshooting

The planner writes native Framework TOML validated by
`cosmos_framework.configs.toml_config.sft_config`. Use the planner's `--help`
for CLI fields; `train_request.schema.json` is an optional SLURM intake
checklist, not a planner input file. The emitted spec contains nested tables.

## Training settings

- `--epochs`, `--effective-global-batch`, `--learning-rate`, `--weight-decay`,
  `--optimizer {AdamW,FusedAdam}`, and `--seed` define the optimizer contract.
- `--scheduler {linear,cosine,constant}` selects the native scheduler;
  `--warmup` is in epochs (rounded up to whole optimizer steps) and
  `--minimum-lr-factor` sets the decay floor. Constant schedules require 1.
  Optimizer epsilon and config-group selection are passed as native CLI overrides
  because the VLM TOML translator skips epsilon.
- `--loss-spike-rollback` controls Framework's gradient-norm guard. It defaults
  to 10 for PEFT and off for dense training; zero disables it. Dense snapshots
  can consume substantial memory—enable only after reviewing capacity.
- `--framework-per-forward-batch` controls samples per rank/forward. Nano's
  automatic value is the per-rank effective batch; Edge keeps one. The planner
  derives gradient accumulation and rejects incompatible effective batches.
- `--training-mode peft` requires explicit LoRA rank, alpha, and target modules.
  Framework owns injection and checkpoint reconstruction. Dense is separate.
- Frame count, dimensions, sequence length, and maximum video pixels are
  recorded separately from the immutable model identity. Edge defaults live
  in `training-contract.yaml`; explicit overrides are fingerprinted.
  Frame dimensions inform pixel-budget planning; they do not force native
  resizing. The recipe reads `COSMOS_VIDEO_NUM_FRAMES` and
  `COSMOS_VIDEO_MAX_PIXELS`; the checkpoint processor handles resizing.
- `--framework-video-cache-size`, processing threads, decoder threads,
  DataLoader workers/prefetch, and validation-cache flags tune the native
  TorchCodec path without changing the dataset or sampling contract.

Checkpoint cadence is per epoch. No checkpoint-retention limit is exposed by
the planner; the source workflow's `max_keep` setting has no native counterpart.
Review checkpoint storage capacity before launch.

Validation batch size is exact. Media-grouped sharding/cache frontloading must
preserve the padded multiset and finite validation stream. Loss is aggregated
by valid-label counts, not a mean of rank means.

## Errors

Missing media, train/validation overlap, unsupported vision options, and
insufficient records per GPU are planning errors. Correct the input contract;
do not silently reduce the corpus.

For OOM, inspect sequence length, frame/pixel budget, per-forward batch,
parallelism, and available memory with the user. Do not change evaluation
semantics merely to fit a run. For decoder/import failures, validate the exact
image's CUDA TorchCodec and native preprocessor through preflight. Do not
patch source into an immutable image or fall back to CPU decoding silently.

Framework DCP export failures require the saved config, base checkpoint,
revision, exact tensor keys, and manifest to agree. Preserve a failed export
for diagnosis and never select an adjacent checkpoint by mtime.
