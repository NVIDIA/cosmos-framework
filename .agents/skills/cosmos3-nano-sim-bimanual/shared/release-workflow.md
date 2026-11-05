# Environment and resources

## Cosmos Framework

Use the [native training tutorial](../post-training/SKILL.md) or
[native inference tutorial](../inference/SKILL.md), based on
[Framework setup](../../../../docs/setup.md).
Run from the Framework root. Training uses Python 3.13,
the matched `cu130-train`/`cu128-train` group and the `train` extra; the worked
inference-only install uses `cu130` and `COSMOS_TRAINING=0`.
The root lockfile is generated and checked with `uv==0.11.29`. Before setup,
set `UV_CACHE_DIR` to a writable workspace-local directory, then run
`uvx --from 'uv==0.11.29' uv lock --check`; do not regenerate the lock with a
different resolver version. This avoids consuming a quota-limited home cache
without changing the locked environment. The training example also includes the
`guardrail` extra. Its install command is
`uvx --from 'uv==0.11.29' uv sync --locked --python 3.13 --group cu130-train --extra train --extra guardrail`;
use the matching CUDA group for your host. The training extra supplies
`mujoco==3.10.0` for calibrated action/FK preprocessing.

Local-bundle training needs a Cosmos3-Nano DCP root, local video VAE, complete bundle with
its text tokenizer, shared output root and own-run stage parents. Its native
TOMLs disable W&B and do not require an inference release checkpoint. HF downloads may still require
model read access; keep credentials out of the checkout and diagnostics.

Use a writable output root shared at the same absolute path on every rank.
Validate media libraries and topology in the target allocation, and check
complete save/readback rather than treating metadata presence as success.

## Hardware and allocation checks

Run CUDA and GPU-count checks inside the target allocation, not only on its
submission host. The training example launches eight processes with FSDP shard
8 and CP 4. The inference example launches one process. Check memory for the
selected sample and layout.
Multi-node workers need NCCL connectivity and the same absolute shared paths.

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())'
df -h /dev/shm
```

Keep the locked Python packages and the FFmpeg library search path from
Framework setup. Framework discovers wheel-provided NPP libraries. Measure
shared-memory use during a bounded step and checkpoint save/readback rather
than assuming a topology-independent minimum.

## Bind resources for the requested task

| Task                     | Required local resources                                                                                                                          |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| Fresh Stage 0            | `COSMOS_SIM_BIMANUAL_GA_CHECKPOINT`, `COSMOS_SIM_BIMANUAL_LOCAL_BUNDLE`, `VIDEO_VAE_PATH`, shared `IMAGINAIRE_OUTPUT_ROOT`                        |
| Later stages             | The same bundle/VAE, plus the own-run stage roots in [checkpoint handoffs](../post-training/references/checkpoint-and-resume.md#cosmos-framework) |
| Released-model inference | Current HF checkpoint plus the hosted manifests and assets from [inference](../inference/SKILL.md)                                                |
| User-trained inference   | Complete Stage 3 student export and inputs matching its saved action contract                                                                     |

Training does not require a release inference checkpoint. Local-bundle training
resolves the text tokenizer from `resources/tokenizer/` and disables W&B by
default. Enable online tracking only if requested, using the user's account,
entity and project. Keep any credentials outside code, manifests and logs;
only remote resources require their corresponding provider access. Gated HF
inference inputs require `HF_TOKEN` in the runtime environment. For Pyxis, pass
the exported variable by name with `--export=ALL,HF_TOKEN` and
`--container-env=HF_TOKEN`; a custom `HF_HOME` cache alone is insufficient for
the direct URL downloader.

Use the native [multi-node notes](../post-training/references/multinode-launcher.md)
for more than one node. Preserve the same run name and output root on restart.
Before a new stage, confirm the selected parent is complete and loadable.
