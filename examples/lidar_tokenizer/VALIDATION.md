# Validation record

Verified on 2026-11-06 against public code `0756428810fd`, based on
Framework `98c35edad864`: **55 CPU tests passed**. Run the same focused suite from this
checkout using a configured training environment:

```bash
set -euo pipefail
cd "${FRAMEWORK_DIR:?set this Framework checkout root}"
PYTHONPATH="$PWD" COSMOS_INTERNAL=0 COSMOS_TRAINING=1 OMP_NUM_THREADS=2 \
  "${TRAIN_PYTHON:?set the configured training environment Python}" \
  -m pytest --noconftest -o addopts='' --disable-warnings \
  cosmos_framework/model/tokenizer/lidar/post_training_test.py \
  cosmos_framework/model/generator/tokenizers/lidar/checkpoint_test.py \
  cosmos_framework/model/generator/tokenizers/lidar/dtypes_test.py \
  cosmos_framework/model/generator/tokenizers/lidar/preprocessing_test.py \
  cosmos_framework/model/generator/tokenizers/lidar/normalization_test.py \
  cosmos_framework/model/generator/tokenizers/lidar/range_projection_test.py -q
```

`--noconftest` isolates these CPU checks from repository GPU/inference fixture
initialization. Existing L0/CPU marks can generate warnings in that isolated mode.

The five new tests exercise a small, randomly initialized VAE with the geometric
objective, a single-rank Gloo DDP forward/backward and full-state resume,
strict initialization and EMA seeding, explicit live/EMA serving and evaluation,
and export metadata privacy. The 50 existing checks cover checkpoint loading,
dtypes, sparse-tar preprocessing, normalization and range projection. Together
they check functionality rather than reconstruction quality.

Separate CPU CLI checks passed for manifest/checksum preflight, rejection of an
incorrect checksum or overlapping train/evaluation paths, and live/EMA export
followed by evaluation. These checks used synthetic fixtures and verified that
the imported modules came from the selected public checkout.

Ruff lint and format checks passed for the added Python files. Pyrefly reported
zero errors under the repository's current configuration. Skill frontmatter
validation, relative Markdown link checks and Bash syntax checks passed.

A separate full-size **four-GPU CUDA workflow check passed** on the same public
code, with `COSMOS_INTERNAL=0` and imports verified against this checkout:

- Four NVIDIA GB200 GPUs, PyTorch 2.10.0+cu128 and NATTEN 0.21.6+torch2100cu128.
- FP32 NATTEN forward/backward passed before training. The fixed recipe used
  128×1808 inputs, nine model frames, two clips per GPU and global batch eight.
- Eight measured-LiDAR clips supplied training inputs, with two separate clips
  reserved for the short reconstruction check. A compatible, checksum-pinned
  initializer loaded the complete network.
- A four-rank NCCL process group trained to step 20 and saved full state. A new
  four-rank launch restored that checkpoint and continued to absolute step 40.
  Both launches completed successfully, with finite losses and gradients.
- Independent checkpoint inspection found all 258 Adam states at the expected
  step, finite optimizer moments, finite model/EMA tensors, initialized latent
  statistics and four nonempty CPU/CUDA RNG state pairs. Saved learning rates
  were `2.1e-7` at step 20 and `4.1e-7` at step 40, retaining the 30k schedule.
- Live and EMA exports from the final checkpoint both loaded through the serving
  interface and completed FP32 posterior-mean GPU evaluation on both clips.
  Range MAE was finite and validity IoU was within `[0,1]` for both variants.
  Export hashes were distinct; both recorded the same source checkpoint.

DDP emitted a gradient-stride performance warning; both training launches
completed without errors. Throughput was not profiled.

The run establishes this bounded single-node workflow. Multi-node operation,
replay equivalence to uninterrupted training, full-schedule stability,
training-corpus coverage, convergence and reconstruction quality improvements
require separate validation. Follow the bounded train/save/resume/evaluate
instructions when changing hardware, precision, topology or recipe settings.
