# Data and canonical dataloader contract

## Cosmos Framework

Use the local-bundle TOMLs in the [post-training tutorial](../SKILL.md). The
native [Bimanual loader](../../../../../cosmos_framework/data/generator/local_simulation/bimanual_config.py)
consumes a complete bundle with `manifest.json`, `samples.json`, captured local
resources and `resources/tokenizer/`. It does not create that bundle from raw
robot data. Preserve the portable layout and semantic rules below.

The public recipe selects robot records, applies stage-specific rendering and
window policies internally, and installs `LocalSimulationResumeCallback`. A
bundle containing one supported robot source is sufficient. Missing robot
records or invalid windows are errors, not an invitation to substitute
inference rows.
When a user intentionally provides multiple supported sources, the loader
normalizes their declared relative weights and combines them without assigning
population categories.
The native `LocalBimanualPipeline` performs the original action mapping,
calibrated FK/gripper handling, masking and global-asinh normalization.

Inspect actual decoded samples and model batches through this native loader
during the short training check. Do not normalize a prepared inference sidecar
a second time.

Keep the `dataloader_train` selected by each native local TOML. The builder
uses `make_bimanual_local_dataloader` to select streams and stage-specific
resolution/window policies; it does not require a separate source-registry
module. For your data, prepare a complete bundle with the same model-facing
semantics and bind it as `COSMOS_SIM_BIMANUAL_LOCAL_BUNDLE`.

Start from the versioned
[HF robot training example](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Bimanual/tree/main/examples/training).
It includes a checksum, explicit single-source coverage and paired AgiBot GT.
The bundle is a small dataloader reference rather than a statistically
representative training set.

Read the [caption guide](caption-guide.md) before preparing annotations. Use
the [prompt guide](../../../cosmos3-inference-simulation-bimanual/references/prompt-guide.md) to keep inference wording and
training caption policy consistent. Read the
[59-index action schema](../../../cosmos3-inference-simulation-bimanual/references/unified-v1-action-schema.md) before creating action
records or validity masks.

## User-owned local bundle

Organize adapted training data as a completed local simulation bundle. This is
the portable input consumed by `LocalSimulationDataset`. All runtime paths stay
inside the bundle and remain relative so the directory can be moved intact:

```text
sim-bimanual-local/
  manifest.json
  samples.json
  assets/
    <encoded videos, action/state payloads, and indexed byte-range files>
  resources/
    global_asinh.json
    tokenizer/
      <complete local tokenizer files>
    <robot descriptions, meshes, and other transform resources>
  receipts/
    unified-v1-action-schema-receipt.json
    dataset-receipt.json
```

`manifest.json` uses local-bundle format version 1 and contains `complete:
true`, `model: "bimanual"`, the exact `sample_count`, and captured
`transforms` keyed by stratum. `samples.json` is one JSON array. Each record
contains a stable user-owned `identity`, its `stratum`, the captured
`request`, `source_rows`, and an `assets` mapping from every requested payload
identity to a relative file under `assets/`. `resources/global_asinh.json` is
the digest-pinned statistics file in the action-schema receipt. The tokenizer
must be complete and loadable with local-only access. Transform resources are
addressed with `bundle://` paths and resolve inside the bundle.

The local loader consumes a completed bundle; it does not convert raw robot
data into that bundle. Use an offline preparation pipeline to write media,
actions, resources, and metadata. Preserve the serialized constructor names,
sample requests, byte offsets, source rows, and transform settings produced by
the matching data adapter; these fields are not intended for manual authoring.
Then validate the bundle with the same Framework checkout. The loader binds
only approved constructors and never imports a callable selected by bundle data.
It rejects an unsupported format version, an incomplete bundle, an incorrect
model name, a sample-count mismatch, a path that escapes the bundle, or a
missing asset.

`dataset-receipt.json` records the source revision, split construction,
coordinate frames, units, frame/action alignment, caption policy, calibration,
schema receipt digest, normalization-statistics digest, and exporter version.
Use user-owned identities and paths only. Keep every source episode in exactly
one bundle or split. Do not create overlapping windows across training and
evaluation bundles.

### Validate with the native loader

Follow [data preparation and the Stage 0 short run](../SKILL.md#1-prepare-and-validate-training-data).
The commands use Framework's
[native Bimanual preflight](../../../../../docs/bimanual_data_preflight.md):

- The static check validates format, sample identities/counts, source-row
  mappings and local resource paths. Success reports `passed: true` and
  `materialized: false`; it has not decoded media. Provenance receipts are not
  required by this command.
- `--materialize --stage 0` reads one packed sample through
  `make_bimanual_local_dataloader`. The loader checks the selected robot source
  and stage window, then performs decoding, action transforms,
  tokenization and packing. Success reports `passed: true`, `materialized: true`
  and batch sample/token counts and tensor shapes/dtypes. Non-finite tensors fail.
  Use `--stage 1`, `2` or `3` when checking a later stage.

These checks do not load model weights or update the bundle. Inspect
representative decoded samples and packed batches for frame timing, action
units, masks, normalization, and caption alignment before training.

Keep bundle identity, seeds, shuffle order, rank topology, packing and token
costing unchanged during a same-run restart. The native
`LocalSimulationResumeCallback` restores consumed local-stream progress.

## Published inference and training examples

The Hugging Face checkpoint publishes current AgiBot and YAM inference
inputs, including mixed AgiBot rows that switch from `agibotworld` to
`camera_pose` and back. Follow the
[inference tutorial](../../../cosmos3-inference-simulation-bimanual/SKILL.md) for download and validation.
The checkpoint separately publishes duplicated AgiBot inputs, their GT videos,
and a completed single-source robot bundle under `examples/training/`.
Inference sidecars remain model inputs, not raw training actions; use the
bundle layout when adapting robot captures.

## Logical sample

A training sample must provide:

- one RGB observation stream shaped `[C, T+1, H, W]` after decoding.
- `T` action transitions and their timestamps.
- one non-empty task or initial-scene caption.
- a supported embodiment identifier.
- one domain label for every action row when a clip contains more than one
  action domain.
- a 59-value action-validity mask.
- stable episode, window, and split identifiers.

Keep all windows from one episode in the same train, validation, or test split.
Record video codec, source frame rate, resampling rule, resolution, coordinate
frames, units, action rate, and calibration version.

> **Skill note (not in the source):** the sections "Geometry and action mapping" (with the Domain 2 conversion) and "Timing and sampling" that stood here, and the inference action sidecar that closed this file, are now the Sim-Bimanual inference skill's [input contract](../../../cosmos3-inference-simulation-bimanual/references/input-contract.md).

## Captions and conditioning

Captions follow one declared conditioning policy. For a task-and-starting-scene
policy, do not encode reference-future outcomes that are unavailable at
inference time. Express necessary chronology within one continuous text value
and keep the same convention at inference. Keep caption selection deterministic
for validation and test. Record the policy and keep the text plain. The product
path does not require a structured prompt payload or a resolution suffix.

Training and inference must agree on RGB preprocessing, prompt formatting,
action mapping, domain naming, normalization, frame rate, and frame/action
alignment.

## Canonical loader adaptation contract

Prefer the completed local-bundle path above for user-owned data. It keeps
assets and captured transforms together while reusing the canonical
action materialization, normalization, augmentation, sharding, and packing
logic. Do not point the local loader at the raw preparation tree or change the
native config to read arbitrary loose files.

A different reader is a new adapter. Its user-owned paths belong in the
user's config or environment, and it must preserve the complete model-facing
contract below.

The dataloader must:

- decode and convert each user sample to the logical tensors and metadata above.
- collate variable-length samples without changing action meanings, timestamps,
  captions, masks, or episode identities.
- apply normalization exactly once and keep padding invalid.
- shard samples exactly once across distributed ranks and workers.
- expose deterministic sampling and enough progress state for the user's chosen
  same-run resume behavior.
- fail visibly on decode, mapping, timing, finite-value, or split violations.

Before training, run typical-sample, maximum-length, mask, finite-value,
timestamp, split-leakage, collation, rank-sharding, and round-trip checks. Save
the canonical loader version, source-entry version, and resolved non-secret
settings with the run.
