---
name: cosmos3-nano-lidar-tokenizer-post-training
description: Continue training the LiDAR tokenizer on measured Pandar128 LiDAR in Cosmos Framework. Use when preparing LiDAR VAE post-training, inspecting its recipe, resuming training, comparing live and EMA reconstructions, or exporting the adapted tokenizer; generator SFT and Sim-Auto distillation use their own skills.
---

# Post-train the Cosmos3 LiDAR tokenizer

Adapt the range/intensity/validity TransformerVAE to measured LiDAR.
This guide uses **Cosmos Framework** and your own local measured-LiDAR assets.

For joint video/LiDAR generator training, use
[Transfer-Auto post-training](../../cosmos3-nano-transfer-auto/post-training/SKILL.md).
For generator step distillation, use
[Sim-Auto post-training](../../cosmos3-nano-sim-auto/post-training/SKILL.md).
Those workflows consume a tokenizer; their generator checkpoints cannot initialize this VAE.

## Set up and select assets

Follow the [Framework setup guide](../../../../docs/setup.md), including the
training dependency group and the CUDA/NATTEN build for your GPU. Reuse a working
environment; set `TRAIN_PYTHON` to its absolute Python executable path. The
inference-only dependency group is insufficient. Full-resolution runs need fused
neighborhood attention; a dense fallback can exhaust memory.

All code paths below are relative to the repository root. Markdown links are
relative to this skill. Run each Bash block as one shell call; every block enters
the selected checkout and selects the same Python environment explicitly.

Supply these environment variables when running the relevant block:

| Variable                                      | Caller-supplied value                                              |
| --------------------------------------------- | ------------------------------------------------------------------ |
| `FRAMEWORK_DIR`                               | Absolute path to this Framework checkout                           |
| `TRAIN_PYTHON`                                | Absolute path to the configured training environment's Python      |
| `LIDAR_INIT`, `LIDAR_INIT_SHA256`             | Local compatible initializer and its independently recorded SHA256 |
| `LIDAR_TRAIN_MANIFEST`, `LIDAR_EVAL_MANIFEST` | Local measured-LiDAR manifests, split by recording/session         |
| `GPUS_PER_NODE`                               | Allocated GPU count for the single-node example                    |
| `LIDAR_SMOKE`                                 | Fresh smoke output directory outside the repository                |
| `LIDAR_RUN`                                   | Separate fresh output directory for an optional full run           |
| `LIDAR_CANDIDATE`, `LIDAR_EXPORT_ROOT`        | Completed native checkpoint and fresh export parent for evaluation |

Check the selected environment before loading data or allocating model memory:

```bash
set -euo pipefail
cd "${FRAMEWORK_DIR:?set the Framework checkout root}"
COSMOS_INTERNAL=0 COSMOS_TRAINING=1 \
  "${TRAIN_PYTHON:?set the training environment Python}" \
  -m cosmos_framework.scripts.train_lidar_tokenizer --help
```

The help must list `train`, `export` and `evaluate`. Resolve missing imports
using the setup guide before continuing. On the allocated GPU host, also perform
the setup guide's CUDA runtime check. Full-size GPU memory, throughput and
multi-rank numerical behavior remain unverified by the CPU checks in this change;
qualify the bounded run below on the selected hardware.

The native [recipe](../../../../cosmos_framework/configs/tokenizer/lidar_post_training.py)
uses 128×1800 semantic rays, circular padding to width 1808,
16×16 spatial compression and no temporal downsampling. The ordinary V1
inference defaults describe a different architecture. Use an explicitly pinned,
matching checkpoint and obtain its expected SHA256 from your asset record.
No public model download or access entitlement is assumed here.

Set `LIDAR_INIT` to a **local** checkpoint and `LIDAR_INIT_SHA256` to that expected
hash. Initialization strictly loads every network tensor. It starts new optimizer,
scheduler, latent statistics and power EMA history. Earlier V0/V1 checkpoints
are architecture migrations and are outside this continuation recipe.

## Prepare your recordings

Read the [data contract](references/data-contract.md). Prepare measured range
and intensity tar files, then write one local tar path per line in two text
manifests. Relative paths resolve against each manifest's parent directory.
Split by recording/session, including adjacent or duplicate windows; path
separation alone does not establish independence.

Set `LIDAR_TRAIN_MANIFEST` and `LIDAR_EVAL_MANIFEST`. A training update uses two
clips **per GPU**; supply at least `2 × world_size` training clips. Each sample
uses the first eight source sweeps and prepends a
duplicate of the first sweep, making nine frames. Cut longer recordings into
appropriate windows before writing manifests. Sampling uses these explicit
windows rather than random offsets within longer recordings.

## Resolve, smoke and train

The tokenizer has a separate native entrypoint because the generator SFT TOML
schema does not describe this VAE objective. Resolve the recipe and verify the
selected artifact and manifests without creating a network or CUDA context:

```bash
set -euo pipefail
cd "${FRAMEWORK_DIR:?set the Framework checkout root}"
COSMOS_INTERNAL=0 COSMOS_TRAINING=1 \
  "${TRAIN_PYTHON:?set the training environment Python}" \
  -m cosmos_framework.scripts.train_lidar_tokenizer train \
  --train-manifest "${LIDAR_TRAIN_MANIFEST:?set the training manifest}" \
  --eval-manifest "${LIDAR_EVAL_MANIFEST:?set the held-out manifest}" \
  --warm-start "${LIDAR_INIT:?set the pinned compatible checkpoint}" \
  --checkpoint-sha256 "${LIDAR_INIT_SHA256:?set its expected SHA256}" \
  --output "${LIDAR_SMOKE:?set a fresh smoke directory}" \
  --dryrun
```

The JSON must show the selected checksum, disjoint manifest counts and the resolved
recipe. This checks paths, overlap and artifact identity; it does not load the
network to prove initializer architecture compatibility.

On a CUDA allocation, start with a bounded smoke and a separate output identity:

```bash
set -euo pipefail
cd "${FRAMEWORK_DIR:?set the Framework checkout root}"
COSMOS_INTERNAL=0 COSMOS_TRAINING=1 \
  "${TRAIN_PYTHON:?set the training environment Python}" \
  -m torch.distributed.run \
  --standalone \
  --nproc-per-node="${GPUS_PER_NODE:?set allocated GPU count}" \
  -m cosmos_framework.scripts.train_lidar_tokenizer train \
  --train-manifest "${LIDAR_TRAIN_MANIFEST:?set the training manifest}" \
  --eval-manifest "${LIDAR_EVAL_MANIFEST:?set the held-out manifest}" \
  --warm-start "${LIDAR_INIT:?set the pinned compatible checkpoint}" \
  --checkpoint-sha256 "${LIDAR_INIT_SHA256:?set its expected SHA256}" \
  --output "${LIDAR_SMOKE:?set a fresh smoke directory}" \
  --stop-after 20
test -s "$LIDAR_SMOKE/resolved.json"
test -s "$LIDAR_SMOKE/iter_000000020.pt"
```

Rank zero prints JSON records with `step`, `loss`, `lr` and
`range_mae_meters`. Check finite losses, the expected LR and the saved checkpoint;
the runner stops on non-finite loss or gradient values. Complete the resume and
export/evaluation checks below before accepting the setup.
The stop iteration is absolute and does **not** compress the 30k LR schedule.
The [parameter guide](references/parameter-guide.md) explains memory, effective
batch, clipping and the reasons for FP32 and the low learning rate.

After qualification, repeat the smoke launch with a fresh `LIDAR_RUN` as its
`--output`, omitting `--stop-after`. The default endpoint is 30k
updates, with checkpoints every 5k and at an explicitly requested stop.
Multi-node launches use your allocation's ordinary torchrun rendezvous settings;
only DDP is supported, with one process per GPU and CP=1. This skill does not
allocate nodes or choose a scheduler/account on your behalf.

## Resume the same run

The following continues the smoke from iteration 20 to iteration 40. Use the
same manifests, recipe, world size and output directory. It computes the checksum
of the checkpoint just produced locally; an externally supplied resume artifact
also needs a trusted artifact record.

```bash
set -euo pipefail
cd "${FRAMEWORK_DIR:?set the Framework checkout root}"
checkpoint="${LIDAR_SMOKE:?set the smoke directory}/iter_000000020.pt"
checkpoint_sha256=$(sha256sum "$checkpoint")
checkpoint_sha256=${checkpoint_sha256%% *}
COSMOS_INTERNAL=0 COSMOS_TRAINING=1 \
  "${TRAIN_PYTHON:?set the training environment Python}" \
  -m torch.distributed.run \
  --standalone \
  --nproc-per-node="${GPUS_PER_NODE:?set the original allocated GPU count}" \
  -m cosmos_framework.scripts.train_lidar_tokenizer train \
  --train-manifest "${LIDAR_TRAIN_MANIFEST:?set the original training manifest}" \
  --eval-manifest "${LIDAR_EVAL_MANIFEST:?set the original held-out manifest}" \
  --resume "$checkpoint" \
  --checkpoint-sha256 "$checkpoint_sha256" \
  --output "$LIDAR_SMOKE" \
  --stop-after 40
test -s "$LIDAR_SMOKE/iter_000000040.pt"
```

The new checkpoint must reach step 40, rather than starting another 40-step run.

Resume restores model and EMA buffers, optimizer, scheduler, absolute iteration,
rank-local Torch CPU/CUDA RNG and the deterministic sampler position. The local
adapter has no random transforms. Changes to data contents, topology or recipe
require a new experiment. A compatible checkpoint from another training runner
can initialize network weights; full-state resume requires this native runner's
checkpoint format.

## Compare exports before publishing

Export both variants from the same completed checkpoint to new directories:

```bash
set -euo pipefail
cd "${FRAMEWORK_DIR:?set the Framework checkout root}"
for variant in live ema; do
  COSMOS_INTERNAL=0 COSMOS_TRAINING=1 \
    "${TRAIN_PYTHON:?set the training environment Python}" \
    -m cosmos_framework.scripts.train_lidar_tokenizer export \
    --checkpoint "${LIDAR_CANDIDATE:?set a completed native checkpoint}" \
    --weights "$variant" \
    --output "${LIDAR_EXPORT_ROOT:?set a new export parent}/$variant"
  COSMOS_INTERNAL=0 COSMOS_TRAINING=1 \
    "$TRAIN_PYTHON" \
    -m cosmos_framework.scripts.train_lidar_tokenizer evaluate \
    --artifact "$LIDAR_EXPORT_ROOT/$variant" \
    --manifest "${LIDAR_EVAL_MANIFEST:?set the held-out manifest}" \
    --output "$LIDAR_EXPORT_ROOT/$variant-eval.json"
done
```

Exports contain `tokenizer.pt`, `latent_stats.pt` and `config.json` with matching
network/projection/compression metadata, weight variant and hashes. Initialize
`LidarTokenizerV1Interface` with these explicit settings; see the
[recipe example](../../../../examples/lidar_tokenizer/README.md).

The native check uses FP32 posterior means, the first eight sweeps per clip,
validity threshold 0.5 and excludes the prepended duplicate from scoring. It
reports range MAE on measured-valid rays and validity IoU. It is a short-clip
reconstruction check. It does not establish long-sequence reconstruction or
point-cloud geometry quality. Keep the same held-out recordings and protocol
across candidates, inspect reconstructions/BEV as well as scalar metrics, and
select the measured winner rather than assuming EMA improves every metric.

Updating a generator to use new tokenizer weights/statistics changes its latent
space. Qualify or adapt that generator explicitly before replacing its bundled
VAE. Exporting this tokenizer alone does not update an Auto model.

## References

- [Data contract](references/data-contract.md)
- [Parameters and checkpoint semantics](references/parameter-guide.md)
- [Recipe notes and validation limits](references/recipe-notes.md)
