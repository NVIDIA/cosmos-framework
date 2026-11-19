# Optional FD adaptation for new data

The README workflow uses the adapted FD checkpoint supplied with the HF release.
**FD training is needed only when using new or different data.** No training is
needed for the supplied example dataset and matching adapted checkpoint. The README workflow
includes the original YAM EEF adaptation commands in one fully commented block. This is dynamics-model adaptation, separate from
any downstream failure-detector training.

## Prepare the new data

Review the source recordings and create a compatible EEF export first. Follow
[labeling and preparation](annotation-format.md); a new robot needs its own
conversion and model contract. The commented block expects:

- `SOURCE_DATASET`: videos, telemetry, metadata, and reviewed `cosmos_annotations.json`.
- `EEF_INPUT`: a frozen bundle containing `source_identity.json`,
  `failuregen_annotations.json`, `splits.json`, and `eef/manifest.json` with episode arrays.
- `BASE_CHECKPOINT`: compatible Cosmos3-Nano base weights.
- `WAN_VAE`: `Wan2.2_VAE.pth`.
- `FD_WORK`: a fresh directory for preparation and adaptation outputs.

Use the compatible framework environment and four GB200 GPUs. See the
[generation workflow](generation-workflow.md#prepare-robot-actions) for action preparation
and checkpoint compatibility requirements.
New labels or recordings require newly prepared inputs; do not substitute them
into an existing checkpoint's frozen contract.

## Run the commented example

Set the paths and uncomment the entire optional block before validating the environment and inputs.
It executes the following stages in order:

1. Validate source hashes and create a dataset view.
2. Prepare EEF windows and train-only normalization; convert base weights to
   DCP training format while preserving all 32 pretrained domains.
3. Resolve the framework configuration with `--mode dryrun`.
4. Adapt with `--mode train` for four passes at global batch 16.
5. Validate the completed checkpoint and switch the dataset, prepared bundle,
   checkpoint, and generation output paths together.

For the original 4,970-window bundle, four passes require
`ceil(4 * 4970 / 16) = 1243` updates. New data changes the window count and budget.
The original recipe uses BF16, four-way data sharding, selective activation
checkpointing, and seeded 50/50 FD/ID sampling. Failure generation uses FD.
Compilation and unsupported trainer diagnostics are disabled in this recipe.

The launcher also supports `--mode smoke` for a ten-update execution check;
its run name has a `_smoke` suffix. If using that mode manually, update the
handoff path accordingly. A smoke checkpoint does not establish generation quality.
The README workflow generates three fixed examples with the supplied adapted checkpoint.
For new data, choose compatible donor/host windows in `EXAMPLE_RECIPES` before
continuing. The [data-labeling guide](annotation-format.md) describes task phases
and event boundaries for selecting those windows outside this walkthrough.

## Resume and outputs

Set `FD_RESUME = True` only to resume the same run with unchanged inputs and
settings. The launcher rejects incompatible contracts and concurrent use of a
work directory. A partial initialization/conversion requires a fresh directory.

Outputs under `FD_WORK` include `dataset/`, `prepared/`, `initialization/`, and
`training/cosmos3_yam/fd_id/<run-name>/`. The training directory contains the
resolved configuration, frozen contract, and checkpoint shards. Handoff requires
the latest saved iteration to equal the requested budget. The commented example
writes subsequent generation outputs to `FD_WORK/generation/`.
