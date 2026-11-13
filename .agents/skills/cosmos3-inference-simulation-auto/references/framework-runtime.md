# Environment and model assets

> **Skill note (not in the source):** the training-only parts of this page (the paragraph on the training allocation and output root, training memory, checkpoint memory, training components) are in the [training half of this page](../../cosmos3-post-train-simulation-auto/references/runtime.md) in the Auto post-train skill (`cosmos3-post-train-simulation-auto`).

Run from a Cosmos Framework checkout containing `docs/training_transfer_auto.md`,
`docs/training_sim_auto.md` and `cosmos_framework/scripts/train_experiment.py`.
Follow its `docs/setup.md` for the container, system packages and GPU driver.
The commands use the repository's Python 3.13 training environment. Install the
required `uv` version (`0.11.29`) and system packages from the setup guide first.
Choose CUDA 13 on a compatible driver, or the CUDA 12.8 group on older drivers
without a supported forward-compatibility setup.

```bash
uv sync --locked --python 3.13 --group cu130-train --extra train --extra guardrail
source .venv/bin/activate
export COSMOS_INTERNAL=0
```

The training extra is also needed by imported inference utilities. The separate
guardrail extra supplies `nltk` and the other dependencies used by default
inference; `--extra train` alone does not install them. Keep the checkout's
lockfile: it already specifies FlashAttention-4. Do not layer an
unrelated attention/CUTLASS upgrade on top and assume numerical parity.
Verify PyTorch, TorchCodec/FFmpeg, CUDA and attention imports. Runtime compatibility
does not establish memory capacity for a particular training or inference request.
Also run the setup guide's GPU-allocation and video-decoding checks. Preserve
required CUDA compatibility library paths; do not reset `LD_LIBRARY_PATH` merely
because the environment is a container.

## Model snapshots

| Workflow                                  | Model repository                    | Local variable     |
| ----------------------------------------- | ----------------------------------- | ------------------ |
| Curriculum base                           | `nvidia/Cosmos3-Nano`               | `NANO_HF`          |
| Transfer inference, fine-tuning, Sim base | `nvidia/Cosmos3-Nano-Transfer-Auto` | `TRANSFER_AUTO_HF` |
| Released Sim inference                    | `nvidia/Cosmos3-Nano-Sim-Auto`      | `SIM_AUTO_HF`      |

Use an authorized complete snapshot. Obtain its immutable revision from the model
distribution and set the corresponding `*_REVISION` variable before downloading;
do not invent a revision or treat `main` as immutable. For example:

```bash
hf download nvidia/Cosmos3-Nano-Transfer-Auto \
  --revision "${TRANSFER_AUTO_REVISION:?set the chosen snapshot revision}" \
  --local-dir "${TRANSFER_AUTO_HF:?set a local snapshot directory}"
```

Use the same pattern for Nano or Sim-Auto when needed. Accept applicable licenses
and authenticate through `hf auth login` or `HF_TOKEN` without printing the token.
A repository ID is not a local checkpoint directory. Keep config, weights and
tokenizers from the same snapshot.

## Execution modes

Set `COSMOS_TRAINING=1` for training/config resolution and `COSMOS_TRAINING=0`
for inference. Use `cosmos_framework.scripts.train_experiment` for Auto training;
generic single-video SFT examples do not implement these sensor recipes.
Use `cosmos_framework.scripts.inference` for both Auto inference paths.

Retain the Framework revision, lockfile and snapshot revision with results so
you can reproduce the environment when comparing outputs or resuming training.
