# Change log

## Unreleased

- New features
  - Runtime quantization (`--quantization-method`) is now also applied when loading Hugging Face checkpoints (previously a silent no-op on that path).
  - Add torchao-free `int8_sim` (per-channel weight / per-token activation) and `fp8_sim` quantize-dequantize simulation methods, expose `fp8` and `--quantization-fp8-granularity` on the CLI, add `--quantization-target-fqns-file` for exact module pinning, and dump the quantized module set to `<output_dir>/quantization_matched_fqns.txt`.

- Breaking changes

## 1.2.2 (May 14, 2026)

- New features
  - Add action policy closed-loop evaluation.

## 1.2.1 (May 08, 2026)

- New features
  - Add [action policy post-training (SFT)](./docs/training.md).

## 1.2.0 (May 05, 2026)

- New features
  - Add action modalities (Forward Dynamics, Inverse Dynamics, Policy) for Cosmos3-Nano model.
  - Upgrade Cosmos3-Nano checkpoint to improve T2V, I2V quality.

## 1.1.1 (May 01, 2026)

- New features
  - Add DCP checkpoint conversion/inference.

## 1.1.0 (April 29, 2026)

- New features
  - Add [Post-Training (Supervised Fine-Tuning)](./docs/training.md).
