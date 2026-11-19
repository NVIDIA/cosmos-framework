# Camera identities

> **Skill note (not in the source):** this is the "Camera identities" section of the Transfer-Auto [training-data contract](../../cosmos3-nano-transfer-auto/post-training/references/data-contract.md), moved here because inference reads the identity table; "these training pairs" and "the training index" below refer to that contract.

When adapting to your own rig, keep its calibrated viewpoints and render WSM
aligned with the recorded RGB, including the same crop and projection. Skip
reference-rig retargeting for these training pairs: it changes control viewpoints
without transforming RGB targets. See the
[reference-rig versus custom-rig workflow](input-preparation.md#choose-reference-rig-or-custom-rig-generation).

For your own rig, use [Camera Rig Explorer](../scripts/wsm/docs/camera-rig/guide.md)
and the [nearest-camera mapping workflow](input-preparation.md#use-camera-rig-explorer-for-your-own-rig)
before building the training index. Keep that mapping consistent with inference.
Source filenames can remain your own; associate their paths with the chosen
supported camera keys in the index. The native bundle builder accepts supported,
unique keys; it does not create new view embeddings for arbitrary sensor names.

Keep these learned identities even when dropping or reordering views:

| ID  | Camera key                    |
| --- | ----------------------------- |
| 0   | `camera_front_wide_120fov`    |
| 1   | `camera_cross_right_120fov`   |
| 2   | `camera_rear_right_70fov`     |
| 3   | `camera_rear_tele_30fov`      |
| 4   | `camera_rear_left_70fov`      |
| 5   | `camera_cross_left_120fov`    |
| 6   | `camera_front_tele_30fov`     |
| 7   | `camera_front_fisheye_200fov` |
| 8   | `camera_left_fisheye_200fov`  |
| 9   | `camera_right_fisheye_200fov` |
| 10  | `camera_rear_fisheye_200fov`  |

LiDAR uses sensor identity 11. An index within a selected subset is not the learned
sensor ID. For example, front-wide/rear-left/front-tele retain IDs 0/4/6.
Nominal camera names are not full calibration. Compare direction, FOV and lens
projection before assigning a custom camera. Render controls with the same
calibration as the recorded RGB.
