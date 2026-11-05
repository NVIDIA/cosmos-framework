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
the [prompt guide](../../shared/prompt-guide.md) to keep inference wording and
training caption policy consistent. Read the
[59-index action schema](unified-v1-action-schema.md) before creating action
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
[inference tutorial](../../inference/SKILL.md) for download and validation.
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

## Geometry and action mapping

Convert poses to xyz plus rot6d in the coordinate frame declared by the user's
data definition.
The reference recipe anchors pose deltas to the start of each 16-frame chunk.
Map native fields to `unified_v1` in this order:

```text
ego pose (9), right wrist (9), right fingertips (15), right openness (1),
left wrist (9), left fingertips (15), left openness (1)
```

Calibrate raw gripper measurements before mapping. Define openness direction,
endpoints, clipping behavior, and units in the user's mapping. Zero-fill absent
fields and mark them invalid. Do not represent a missing field as a valid zero.

For canonical reproduction and checkpoint-compatible fine-tuning, use the
digest-pinned statistics and metric-gripper procedure in the
[unified v1 action schema](unified-v1-action-schema.md). Apply normalization
once, then pad to 64 values. Never normalize padded values or use statistics
from a different geometry convention. A newly computed statistics policy is a
new data contract and checkpoint lineage. Compute it only from the user's
training split, version it, and keep its digest with the run.

For every valid one of the 59 dimensions, `[-1, 1]` is the reasonable
post-normalization reference band because its pinned q01/q99 values map to
those endpoints. It is not a hard clipping range. Preserve finite tails and
inspect their denormalized units and semantics. Invalid and padded dimensions
must remain exactly zero. The schema's
[reasonable normalized ranges](unified-v1-action-schema.md#reasonable-normalized-ranges)
table gives every dimension's band, the exact rot6d component envelopes, and
the gripper-width-dependent openness envelopes. Use those semantic envelopes
to reject invalid rotations or calibration errors without rejecting a valid
unclamped tail.

### Convert hand-static intervals to Domain 2

For an action-conditioned sequence used to prepare training data or an
inference request, the user-owned preparation pipeline must convert every
sustained hand-static interval to Domain 2, named `camera_pose`. Perform this
conversion before normalization and model-width padding. Keep every other row
in its original robot domain. One sequence may therefore switch domains from
row to row.

Detect hand stationarity from body-relative arm state or absolute arm commands,
not from a wrist pose expressed in the world frame. Head or mobile-base motion
can move a world-frame wrist pose while the arm command remains unchanged. For
an arm signal `q_t`, a candidate stationary transition satisfies:

```text
all values in q_t and q_(t+1) are finite
max_j abs(q_(t+1,j) - q_(t,j)) <= epsilon_hand
```

Keep only complete contiguous runs that produce at least 11 stationary action
rows. The reference conversion uses these source-specific profiles:

| Control profile           | Hand signal                   |                     `epsilon_hand` | Gripper rule                                                                                                                         |
| ------------------------- | ----------------------------- | ---------------------------------: | ------------------------------------------------------------------------------------------------------------------------------------ |
| Body-relative joint state | Both 7-DoF arms               |   `1e-3` in the source joint units | Both calibrated openness values must change by at most `1e-4`. Keep the transition row and 15 rows on each side in the robot domain. |
| Absolute arm command      | Both 7-DoF arm-command blocks | `2e-3` in the source command units | Base, trunk, and gripper channels do not participate in this stationary predicate.                                                   |

Use a profile only when the source fields have the same meaning and units. For
another control representation, define and version a source-calibrated
threshold, minimum run, and gripper policy in the dataset receipt. Short pauses,
non-finite rows, and rows rejected by the gripper policy remain in the robot
domain.

For each retained row `t`, derive a new action from the synchronized absolute
head-camera poses. Let `C_t` be the camera-to-world transform at observation
`t`, using OpenCV camera axes with +x right, +y down, and +z forward. Compute:

```text
Delta_C_t = inverse(C_t) C_(t+1)
camera_raw_t = [Delta_C_t[0:3, 3], rot6d(Delta_C_t[0:3, 0:3])]
camera_model_t = [4 camera_raw_t[0:3], camera_raw_t[3:9]]
```

`rot6d` stores the first two rotation-matrix columns in the order defined by
the action schema. The factor `4` is the Domain 2 translation normalization.
Do not apply the unified-v1 asinh statistics to a Domain 2 row. Store the nine
Domain 2 values at the start of the action row, set every robot-only action
slot to zero, and set that row's domain label to `camera_pose` or numeric ID
`2`. Do not copy a chunk-anchored ego block into Domain 2. Recompute the
framewise delta from `C_t` and `C_(t+1)`.

Before handing the prepared actions to training or inference, verify that the
domain-label sequence has exactly `T` entries, every Domain 2 row came from a
retained stationary run, and every robot-only slot in those rows is zero.
Invert the translation scale and reconstruct `C_t Delta_C_t` to check it
matches `C_(t+1)` within the dataset's recorded pose tolerance. Also test both
sides of every run boundary so a short pause or manipulation transition cannot
be relabeled silently.

## Timing and sampling

Actions describe transitions between consecutive RGB observations. After any
resampling, require:

```text
number of actions = number of RGB observations - 1
```

Define how raw action timestamps are aligned to frames, how dropped frames are
handled, and whether interpolation is allowed. Reject non-monotonic timestamps,
large unexplained gaps, or an off-by-one sequence instead of trimming one
modality silently.

Window length is selected by the user for the task and available memory; it is
not fixed by the Bimanual loader. All four public training stages consume the
same alignment rule: `T` actions for `T+1` observations. The hosted example's
length describes that example only and is not a required training length.

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

For inference, the public action sidecar is:

```json
{
  "action_space": "normalized_unified_v1",
  "action": [[0.0]],
  "domain_names": ["<supported-domain>"]
}
```

Replace the illustrative action row with exactly `T` rows of 59 finite,
already-normalized values. `domain_names` contains either the manifest's robot
domain once for the whole sequence or one name per action row. A per-row list
may switch only between that robot domain and `camera_pose`. Every
`camera_pose` row must zero action slots 9 through 58 as defined above.
