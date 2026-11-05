# Validation record

Verified on 2026-11-05 against the public Framework base `98c35edad864` plus
this change: **55 CPU tests passed**. Run the same focused suite from this
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

These checks do not qualify full-size fused NATTEN GPU numerics, multi-rank DDP,
rank-local RNG replay after a GPU restart, training-corpus coverage, or new
reconstruction quality. No full-resolution GPU qualification was performed.
Follow the bounded train/save/resume/evaluate instructions before accepting a
production run.
