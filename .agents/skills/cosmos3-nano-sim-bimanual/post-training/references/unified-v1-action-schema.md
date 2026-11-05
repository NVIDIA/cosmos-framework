# Unified v1 action schema

This is the model-facing action contract for Cosmos3 Nano Sim Bimanual. Read it
before adapting a dataset. The companion
[`unified-v1-action-schema-receipt.json`](unified-v1-action-schema-receipt.json)
is the machine-readable source for dimensions, example masks, gripper widths,
the normalization-statistics digest, and the worked example.
Its statistics path is relative to the Framework checkout. The bundled receipt
contains the normalization example, domain masks, and statistics digest.

The raw canonical vector has 59 values. Normalization is applied to those 59
values, invalid values are forced to zero, and five invalid zeros are appended
to form the model's 64-value vector.

## Dimension table

Domain codes below are `A` = `agibotworld`, `Y` = `xdof_yam`, `B` =
`behavior1k_lerobot`, and `U` = a user adapter whose receipt proves that field.
`A/Y/B` describes the listed source adapters, not every possible user dataset.

| Index | Name                     | Meaning                          | Frame                       | Unit     | Valid domain | Reasonable normalized band |
| ----: | ------------------------ | -------------------------------- | --------------------------- | -------- | ------------ | -------------------------- |
|     0 | `ego_tx`                 | Ego target translation x         | 16-frame ego anchor         | m        | A, B, U      | `[-1, 1]`                  |
|     1 | `ego_ty`                 | Ego target translation y         | 16-frame ego anchor         | m        | A, B, U      | `[-1, 1]`                  |
|     2 | `ego_tz`                 | Ego target translation z         | 16-frame ego anchor         | m        | A, B, U      | `[-1, 1]`                  |
|     3 | `ego_rot6d_c0_x`         | First rotation column x          | 16-frame ego anchor         | unitless | A, B, U      | `[-1, 1]`                  |
|     4 | `ego_rot6d_c0_y`         | First rotation column y          | 16-frame ego anchor         | unitless | A, B, U      | `[-1, 1]`                  |
|     5 | `ego_rot6d_c0_z`         | First rotation column z          | 16-frame ego anchor         | unitless | A, B, U      | `[-1, 1]`                  |
|     6 | `ego_rot6d_c1_x`         | Second rotation column x         | 16-frame ego anchor         | unitless | A, B, U      | `[-1, 1]`                  |
|     7 | `ego_rot6d_c1_y`         | Second rotation column y         | 16-frame ego anchor         | unitless | A, B, U      | `[-1, 1]`                  |
|     8 | `ego_rot6d_c1_z`         | Second rotation column z         | 16-frame ego anchor         | unitless | A, B, U      | `[-1, 1]`                  |
|     9 | `right_wrist_tx`         | Right-wrist target translation x | 16-frame right-wrist anchor | m        | A, Y, B, U   | `[-1, 1]`                  |
|    10 | `right_wrist_ty`         | Right-wrist target translation y | 16-frame right-wrist anchor | m        | A, Y, B, U   | `[-1, 1]`                  |
|    11 | `right_wrist_tz`         | Right-wrist target translation z | 16-frame right-wrist anchor | m        | A, Y, B, U   | `[-1, 1]`                  |
|    12 | `right_wrist_rot6d_c0_x` | First rotation column x          | 16-frame right-wrist anchor | unitless | A, Y, B, U   | `[-1, 1]`                  |
|    13 | `right_wrist_rot6d_c0_y` | First rotation column y          | 16-frame right-wrist anchor | unitless | A, Y, B, U   | `[-1, 1]`                  |
|    14 | `right_wrist_rot6d_c0_z` | First rotation column z          | 16-frame right-wrist anchor | unitless | A, Y, B, U   | `[-1, 1]`                  |
|    15 | `right_wrist_rot6d_c1_x` | Second rotation column x         | 16-frame right-wrist anchor | unitless | A, Y, B, U   | `[-1, 1]`                  |
|    16 | `right_wrist_rot6d_c1_y` | Second rotation column y         | 16-frame right-wrist anchor | unitless | A, Y, B, U   | `[-1, 1]`                  |
|    17 | `right_wrist_rot6d_c1_z` | Second rotation column z         | 16-frame right-wrist anchor | unitless | A, Y, B, U   | `[-1, 1]`                  |
|    18 | `right_thumb_x`          | Right thumb-tip position x       | Current right-wrist frame   | m        | U            | `[-1, 1]`                  |
|    19 | `right_thumb_y`          | Right thumb-tip position y       | Current right-wrist frame   | m        | U            | `[-1, 1]`                  |
|    20 | `right_thumb_z`          | Right thumb-tip position z       | Current right-wrist frame   | m        | U            | `[-1, 1]`                  |
|    21 | `right_index_x`          | Right index-tip position x       | Current right-wrist frame   | m        | U            | `[-1, 1]`                  |
|    22 | `right_index_y`          | Right index-tip position y       | Current right-wrist frame   | m        | U            | `[-1, 1]`                  |
|    23 | `right_index_z`          | Right index-tip position z       | Current right-wrist frame   | m        | U            | `[-1, 1]`                  |
|    24 | `right_middle_x`         | Right middle-tip position x      | Current right-wrist frame   | m        | U            | `[-1, 1]`                  |
|    25 | `right_middle_y`         | Right middle-tip position y      | Current right-wrist frame   | m        | U            | `[-1, 1]`                  |
|    26 | `right_middle_z`         | Right middle-tip position z      | Current right-wrist frame   | m        | U            | `[-1, 1]`                  |
|    27 | `right_ring_x`           | Right ring-tip position x        | Current right-wrist frame   | m        | U            | `[-1, 1]`                  |
|    28 | `right_ring_y`           | Right ring-tip position y        | Current right-wrist frame   | m        | U            | `[-1, 1]`                  |
|    29 | `right_ring_z`           | Right ring-tip position z        | Current right-wrist frame   | m        | U            | `[-1, 1]`                  |
|    30 | `right_pinky_x`          | Right pinky-tip position x       | Current right-wrist frame   | m        | U            | `[-1, 1]`                  |
|    31 | `right_pinky_y`          | Right pinky-tip position y       | Current right-wrist frame   | m        | U            | `[-1, 1]`                  |
|    32 | `right_pinky_z`          | Right pinky-tip position z       | Current right-wrist frame   | m        | U            | `[-1, 1]`                  |
|    33 | `right_openness`         | Right gripper openness           | Right actuator              | fraction | A, Y, B, U   | `[-1, 1]`                  |
|    34 | `left_wrist_tx`          | Left-wrist target translation x  | 16-frame left-wrist anchor  | m        | A, Y, B, U   | `[-1, 1]`                  |
|    35 | `left_wrist_ty`          | Left-wrist target translation y  | 16-frame left-wrist anchor  | m        | A, Y, B, U   | `[-1, 1]`                  |
|    36 | `left_wrist_tz`          | Left-wrist target translation z  | 16-frame left-wrist anchor  | m        | A, Y, B, U   | `[-1, 1]`                  |
|    37 | `left_wrist_rot6d_c0_x`  | First rotation column x          | 16-frame left-wrist anchor  | unitless | A, Y, B, U   | `[-1, 1]`                  |
|    38 | `left_wrist_rot6d_c0_y`  | First rotation column y          | 16-frame left-wrist anchor  | unitless | A, Y, B, U   | `[-1, 1]`                  |
|    39 | `left_wrist_rot6d_c0_z`  | First rotation column z          | 16-frame left-wrist anchor  | unitless | A, Y, B, U   | `[-1, 1]`                  |
|    40 | `left_wrist_rot6d_c1_x`  | Second rotation column x         | 16-frame left-wrist anchor  | unitless | A, Y, B, U   | `[-1, 1]`                  |
|    41 | `left_wrist_rot6d_c1_y`  | Second rotation column y         | 16-frame left-wrist anchor  | unitless | A, Y, B, U   | `[-1, 1]`                  |
|    42 | `left_wrist_rot6d_c1_z`  | Second rotation column z         | 16-frame left-wrist anchor  | unitless | A, Y, B, U   | `[-1, 1]`                  |
|    43 | `left_thumb_x`           | Left thumb-tip position x        | Current left-wrist frame    | m        | U            | `[-1, 1]`                  |
|    44 | `left_thumb_y`           | Left thumb-tip position y        | Current left-wrist frame    | m        | U            | `[-1, 1]`                  |
|    45 | `left_thumb_z`           | Left thumb-tip position z        | Current left-wrist frame    | m        | U            | `[-1, 1]`                  |
|    46 | `left_index_x`           | Left index-tip position x        | Current left-wrist frame    | m        | U            | `[-1, 1]`                  |
|    47 | `left_index_y`           | Left index-tip position y        | Current left-wrist frame    | m        | U            | `[-1, 1]`                  |
|    48 | `left_index_z`           | Left index-tip position z        | Current left-wrist frame    | m        | U            | `[-1, 1]`                  |
|    49 | `left_middle_x`          | Left middle-tip position x       | Current left-wrist frame    | m        | U            | `[-1, 1]`                  |
|    50 | `left_middle_y`          | Left middle-tip position y       | Current left-wrist frame    | m        | U            | `[-1, 1]`                  |
|    51 | `left_middle_z`          | Left middle-tip position z       | Current left-wrist frame    | m        | U            | `[-1, 1]`                  |
|    52 | `left_ring_x`            | Left ring-tip position x         | Current left-wrist frame    | m        | U            | `[-1, 1]`                  |
|    53 | `left_ring_y`            | Left ring-tip position y         | Current left-wrist frame    | m        | U            | `[-1, 1]`                  |
|    54 | `left_ring_z`            | Left ring-tip position z         | Current left-wrist frame    | m        | U            | `[-1, 1]`                  |
|    55 | `left_pinky_x`           | Left pinky-tip position x        | Current left-wrist frame    | m        | U            | `[-1, 1]`                  |
|    56 | `left_pinky_y`           | Left pinky-tip position y        | Current left-wrist frame    | m        | U            | `[-1, 1]`                  |
|    57 | `left_pinky_z`           | Left pinky-tip position z        | Current left-wrist frame    | m        | U            | `[-1, 1]`                  |
|    58 | `left_openness`          | Left gripper openness            | Left actuator               | fraction | A, Y, B, U   | `[-1, 1]`                  |

## Pose, rotation, and temporal anchoring

An absolute pose `T_t = [R_t, p_t. 0, 1]` maps coordinates from the local
component frame at observation `t` into the world frame. Action row `i`
targets observation `i+1`. With the required 16-frame chunk anchoring,

```text
a(i) = 16 floor(i / 16)
Delta_T_i = inverse(T_a(i)) T_(i+1)
translation_i = Delta_T_i[0:3, 3]
rotation_i = rot6d(Delta_T_i[0:3, 0:3])
T_(i+1) = T_a(i) Delta_T_i
```

Rows 0 through 15 therefore use observation 0 as their anchor, rows 16 through
31 use observation 16, and so on. Apply the formula independently to ego,
right-wrist, and left-wrist trajectories. Translation is in meters in the
corresponding anchor frame.

For a rotation matrix `R`, rot6d stores the first two columns, not rows:

```text
rot6d(R) = [R00, R10, R20, R01, R11, R21]
c0 = rot6d[0:3]
c1 = rot6d[3:6]
c2 = cross(c0, c1)
R_candidate = [c0 c1 c2]
```

Canonical labels come from valid rotations, so `c0` and `c1` are orthonormal.
The sidecar validator checks each fully valid column pair after denormalization:
both squared column norms must differ from `1` by at most `1e-5`, and their dot
product must differ from `0` by at most `1e-5`. This absolute tolerance permits
serialization roundoff. Masked rotation blocks remain zero and are not checked
as rotations. Component bounds alone are insufficient, and invalid source
rotations are rejected rather than repaired by projection.
When decoding a predicted value, project `R_candidate` to `SO(3)` before pose
composition. The identity rotation is `[1, 0, 0, 0, 1, 0]`.

## Gripper calibration and metric normalization

Canonical openness is unitless: 0 is fully closed and 1 is fully open. For a
native scalar `g` with calibrated endpoints `g_closed` and `g_open`, use

```text
openness = (g - g_closed) / (g_open - g_closed)
g = openness (g_open - g_closed) + g_closed
```

Clamp only when the adapter's documented contract says to clamp. Record both
endpoints, clamp behavior, hardware/calibration version, and fully-open
jaw-to-jaw width `W` in the dataset receipt. The bundled example receipt uses:

| Domain               | Native calibration                                        |     `W` |
| -------------------- | --------------------------------------------------------- | ------: |
| `agibotworld`        | actuator 120 closed, 35 open, clipped                     | 0.120 m |
| `xdof_yam`           | adapter already provides 0 closed, 1 open                 | 0.071 m |
| `behavior1k_lerobot` | native command -1 closed, +1 open, clipped with `(g+1)/2` | 0.100 m |

The statistics file stores gripper quantiles in meters using reference width
`W_ref = 0.082 m`. For a sample whose fully-open width is `W`, divide the
gripper q01 and q99 by `W` before normalizing the canonical openness fraction.
This is equivalent to normalizing the physical width `openness * W` against
the 0.082 m reference.

## Normalization and statistics receipt

The fixed statistics are checked in at:

```text
cosmos_framework/data/generator/action/normalizers/global_asinh_v1_backward_anchored_gripper_metric.json
```

Its required SHA-256 is
`ae695e2c643a5626b3c3564f00705aa3a83ab0da7640549c7340db71172594b0`.
The receipt records 59 q01/q99 values and their provenance: zero-centered ego
translation with 0.2822 m scale, recorded pose/fingertip distributions, and the
0.082 m metric-gripper reference. Do not recompute or silently substitute this
file for canonical reproduction. For a new normalization policy, create a new
versioned receipt and checkpoint lineage.

The statistics metadata identifies its component receipts exactly:

- ego rotation and both fingertip blocks come from the corresponding slices of
  `hand_pose_backward_anchored_rot6d.json` raw global statistics.
- both wrist pose blocks come from the corresponding slices of
  `yam_ee_backward_anchored_rot6d.json` raw global statistics.
- ego translation is an explicit zero-centered 0.2822 m override.
- both openness slots are an explicit `[0, 0.082]` meter override before the
  per-sample width conversion above.

For each valid raw canonical value `x_j`:

```text
offset_j = (q01_j + q99_j) / 2
scale_j = max((q99_j - q01_j) / 2, 1e-8)
y_j = asinh((x_j - offset_j) / scale_j) / asinh(1)
x_j = sinh(y_j asinh(1)) scale_j + offset_j
```

There is no forward clamp. After normalization, force every invalid index to
zero. Then append five zeros at indices 59 through 63. The 64-value model mask
is the 59-value validity mask followed by five `false` values.

## Reasonable normalized ranges

The `[-1, 1]` entry in every dimension-table row is the reference band for a
valid dimension. It follows directly from the checked-in statistics because
`q01_j` normalizes to `-1` and `q99_j` normalizes to `1`. For dimensions whose
statistics are empirical quantiles, this is the central reference region of
the source distribution. For ego translation and openness, it is the explicit
configured reference region described above.

The reference band is not a clipping or rejection bound. The normalizer is
invertible and intentionally preserves finite values outside `[-1, 1]`. For
each valid dimension, report the fraction below `-1` and above `1`, then inspect
the denormalized values against that dimension's units, frame, calibration, and
source semantics. Do not clamp or reject a record only because it is outside
the reference band. Invalid dimensions and padded dimensions 59 through 63 are
different. They must equal exactly zero.

Translation and fingertip dimensions have no universal finite semantic
envelope across robots and workspaces. Their reasonable band is `[-1, 1]`, and
out-of-band values require source-specific review in meters. Rotation and
openness have additional semantic envelopes that can distinguish a valid tail
from an invalid value.

Each rot6d source component must denormalize into `[-1, 1]`. Applying that raw
component bound to the pinned statistics gives these normalized envelopes:

| Index | Dimension                | Reasonable band | Rot6d component envelope |
| ----: | ------------------------ | --------------- | ------------------------ |
|     3 | `ego_rot6d_c0_x`         | `[-1, 1]`       | `[-3.679, 1.000]`        |
|     4 | `ego_rot6d_c0_y`         | `[-1, 1]`       | `[-1.586, 1.593]`        |
|     5 | `ego_rot6d_c0_z`         | `[-1, 1]`       | `[-2.577, 2.587]`        |
|     6 | `ego_rot6d_c1_x`         | `[-1, 1]`       | `[-1.585, 1.581]`        |
|     7 | `ego_rot6d_c1_y`         | `[-1, 1]`       | `[-3.713, 1.000]`        |
|     8 | `ego_rot6d_c1_z`         | `[-1, 1]`       | `[-2.434, 2.395]`        |
|    12 | `right_wrist_rot6d_c0_x` | `[-1, 1]`       | `[-3.298, 1.000]`        |
|    13 | `right_wrist_rot6d_c0_y` | `[-1, 1]`       | `[-1.645, 1.661]`        |
|    14 | `right_wrist_rot6d_c0_z` | `[-1, 1]`       | `[-1.586, 1.601]`        |
|    15 | `right_wrist_rot6d_c1_x` | `[-1, 1]`       | `[-1.660, 1.640]`        |
|    16 | `right_wrist_rot6d_c1_y` | `[-1, 1]`       | `[-3.597, 1.000]`        |
|    17 | `right_wrist_rot6d_c1_z` | `[-1, 1]`       | `[-1.866, 1.807]`        |
|    37 | `left_wrist_rot6d_c0_x`  | `[-1, 1]`       | `[-3.510, 1.000]`        |
|    38 | `left_wrist_rot6d_c0_y`  | `[-1, 1]`       | `[-1.805, 1.782]`        |
|    39 | `left_wrist_rot6d_c0_z`  | `[-1, 1]`       | `[-1.677, 1.649]`        |
|    40 | `left_wrist_rot6d_c1_x`  | `[-1, 1]`       | `[-1.776, 1.806]`        |
|    41 | `left_wrist_rot6d_c1_y`  | `[-1, 1]`       | `[-3.834, 1.000]`        |
|    42 | `left_wrist_rot6d_c1_z`  | `[-1, 1]`       | `[-1.947, 1.890]`        |

The component envelope is necessary but not sufficient. After denormalization,
each pair of three-value columns must still satisfy the rot6d orthonormality
contract. Reject a source rotation that violates the raw component envelope or
cannot form a valid rotation after the documented projection.

For indices 33 and 58, calibrated openness is in `[0, 1]`. With fully-open
width `W`, its normalized semantic envelope is

```text
y_closed = -1
y_open(W) = asinh(2 W / 0.082 - 1) / asinh(1)
```

The checked-in domains therefore use:

| Domain               |     `W` | Reasonable band | Calibrated openness envelope |
| -------------------- | ------: | --------------- | ---------------------------- |
| `agibotworld`        | 0.120 m | `[-1, 1]`       | `[-1.000, 1.600]`            |
| `xdof_yam`           | 0.071 m | `[-1, 1]`       | `[-1.000, 0.770]`            |
| `behavior1k_lerobot` | 0.100 m | `[-1, 1]`       | `[-1.000, 1.317]`            |

A user adapter must compute the same openness envelope from its recorded `W`.
An openness value outside that envelope indicates a calibration, clamping, or
normalization error even when it is finite. A value above `1` can be correct
for a gripper wider than the 0.082 m reference and must not be clipped.

## Validity masks by source adapter

A missing field must be zero-filled and invalid. A measured zero remains valid.
These masks describe the source adapters in the receipt, not arbitrary user data:

| Domain               | Valid indices       | Invalid indices   |
| -------------------- | ------------------- | ----------------- |
| `agibotworld`        | 0-17, 33, 34-42, 58 | 18-32, 43-57      |
| `xdof_yam`           | 9-17, 33, 34-42, 58 | 0-8, 18-32, 43-57 |
| `behavior1k_lerobot` | 0-17, 33, 34-42, 58 | 18-32, 43-57      |

## Complete raw to mapped to normalized to padded example

This `xdof_yam` example starts from the native order
`[left_pose9, left_open, right_pose9, right_open]`:

```text
raw20 = [
  0.01, 0, 0, 1, 0, 0, 0, 1, 0, 0.25,
  0, 0.02, 0, 1, 0, 0, 0, 1, 0, 0.75
]
```

Mapping places right pose at 9-17, right openness at 33, left pose at 34-42,
and left openness at 58. In complete sparse notation, every omitted index is
exactly zero:

```text
mapped59 = {10: 0.02, 12: 1, 16: 1, 33: 0.75,
            34: 0.01, 37: 1, 41: 1, 58: 0.25; all other indices: 0}

normalized59 = {
  9: -0.015061074, 10: 0.081466351, 11: 0.150576794,
  12: 1, 13: 0.017178207, 14: 0.015630752,
  15: -0.022937468, 16: 1, 17: -0.077924519,
  33: 0.334142951,
  34: 0.115980475, 35: -0.105050644, 36: 0.150474455,
  37: 1, 38: -0.028379809, 39: -0.031840056,
  40: 0.036997598, 41: 1, 42: -0.079562538,
  58: -0.613118149; all other indices: 0
}

padded64 = normalized59 with indices 59, 60, 61, 62, and 63 equal to 0
mask64 = true at 9-17, 33, 34-42, and 58; false everywhere else
```

The exact values use `W = 0.071 m` and the statistics receipt above. The
companion JSON contains the same input and expected normalized values.

## Validate the schema, masks, and round trip

Run from the Framework root with `SIM_BIMANUAL_SKILL` set to this bundle's
absolute location:

```bash
: "${SIM_BIMANUAL_SKILL:?set the absolute Bimanual skill directory}"
python "$SIM_BIMANUAL_SKILL/post-training/scripts/validate_unified_v1.py" \
  --repo-root "$PWD" \
  --receipt "$SIM_BIMANUAL_SKILL/post-training/references/unified-v1-action-schema-receipt.json"
```

The validator checks the 59 dimension names, statistics path and digest,
domain masks, 59-to-64 padding, the worked mapping, the `[-1, 1]` reference
mapping, and the normalization round trip. The bundled receipt has no sample
entries, so it reports `samples: 0`, `action_rows: 0`, and
`round_trip: passed`. To validate user samples, create a receipt with explicit
sample identities, materialize local single-robot-domain sidecars, and pass
their manifests with repeated `--manifest` arguments. Use the
[native checkpoint/input checker](../../inference/SKILL.md#1-choose-the-checkpoint-path)
for published or user-authored mixed-domain sidecars.
