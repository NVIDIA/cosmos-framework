# Framework reasoner parameters and troubleshooting

The planner writes native Framework TOML validated by
`cosmos_framework.configs.toml_config.sft_config`. Do not pass policy-worker
configuration or arbitrary dotted keys to the trainer. Use the planner's
`--help` and `train_request.schema.json` for request fields; the emitted spec
contains nested tables.

## Training settings

- `--epochs`, `--effective-global-batch`, `--learning-rate`, `--weight-decay`,
  scheduler/warmup options, and `--seed` define the optimizer contract.
- `--framework-per-forward-batch` controls samples per rank/forward. Nano's
  automatic value is the per-rank effective batch; Edge keeps one. The planner
  derives gradient accumulation and rejects incompatible effective batches.
- `--training-mode peft` requires explicit LoRA rank, alpha, and target modules.
  Framework owns injection and checkpoint reconstruction. Dense is separate.
- Frame count, dimensions, sequence length, and maximum video pixels are
  recorded separately from the immutable model identity. Edge defaults live
  in `cosmos-framework-backend.yaml`; explicit overrides are fingerprinted.
- `--framework-video-cache-size`, processing threads, decoder threads,
  DataLoader workers/prefetch, and validation-cache flags tune the native
  TorchCodec path without changing the dataset or sampling contract.

Validation batch size is exact. Media-grouped sharding/cache frontloading must
preserve the padded multiset and finite validation stream. Loss is aggregated
by valid-label counts, not a mean of rank means.

## Errors

Missing media, train/validation overlap, unsupported vision options, and
insufficient records per GPU are planning errors. Correct the input contract;
do not silently reduce the corpus or select another runtime.

For OOM, inspect sequence length, frame/pixel budget, per-forward batch,
parallelism, and available memory with the user. Do not change evaluation
semantics merely to fit a run. For decoder/import failures, validate the exact
image's CUDA TorchCodec and native preprocessor through preflight. Do not
patch source into an immutable image or fall back to CPU decoding silently.

Framework DCP export failures require the saved config, base checkpoint,
revision, exact tensor keys, and manifest to agree. Preserve a failed export
for diagnosis and never select an adjacent checkpoint by mtime.
