# Input contract

> **Skill note (not in the source):** this file is the input part of the Sim-Bimanual data contract, moved verbatim from the data contract reference of the Sim-Bimanual post-training skill (cosmos-framework `main` 45fecab): its sections "Geometry and action mapping" and "Timing and sampling", then the inference action sidecar that closed it, here under an added heading. No link target changed. The local training bundle part of that data contract is the [data contract](../../cosmos3-post-train-simulation-bimanual/references/data-contract.md) of the Sim-Bimanual post-training skill.

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

## Inference action sidecar

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
