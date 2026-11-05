# Customer data for Sim-Auto

Use the same paired local scene format as
[Transfer-Auto](../../cosmos3-nano-transfer-auto/post-training/references/data-contract.md):
RGB targets, camera WSM v4.0.1 controls, camera-local target-grounded captions, measured
LiDAR targets, HD-map range controls, and timing/calibration evidence.

The [shared reference scene](../../cosmos3-nano-transfer-auto/shared/reference-scene.md)
provides the seven-camera `daytime_scene` assets, native index and inference
requests. Follow its HF download and checksum instructions. One scene is
a format example, not a sufficient dataset for assessing generalization.

Prefer its `captions_with_timestamp/` variant when preparing post-training data:
timestamped captions match the training format and reduce the caption-format gap.
Paragraph-style `captions/` remains supported. Follow the shared
[caption selection](../../cosmos3-nano-transfer-auto/shared/reference-scene.md#choose-a-caption-style)
and [target-grounding/cross-view checks](../../cosmos3-nano-transfer-auto/post-training/references/data-contract.md#training-captions)
before packaging; the supplied index selects paragraph captions by default.

Build your bundle with Framework's `auto_bundle` command in
[Transfer-Auto preparation](../../cosmos3-nano-transfer-auto/post-training/SKILL.md#prepare-a-local-training-bundle).
Set `COSMOS_SIM_AUTO_LOCAL_BUNDLE` to its output. The loader reads local payloads;
an original remote dataset catalog is neither required nor fetched.

## Preserve your bundle's camera mappings

The generic builder uses `video_<camera_key>`, `wsm_<camera_key>` and
`caption_<camera_key>`. The Sim-Auto profiles contain different source-specific
mappings and rig arrays. Remove those four overlays so the bundle's verified
mappings and eligible camera subsets are used:

```bash
bundle_overrides=()
for resolution in 480 720; do
  dataset="dataloader_train.dataloaders.cam_lidar_joint_${resolution}p_separate_view_text.dataloader.dataset.transform_overrides"
  bundle_overrides+=(
    "~${dataset}.camera_keys"
    "~${dataset}.camera_caption_key_mapping"
    "~${dataset}.camera_control_key_mapping"
    "~${dataset}.camera_sampler"
  )
done
```

Apply this array at **every stage**, as shown in the [launch guide](../post-training/references/runtime.md).
Keep the registered stream identifiers in the commands unchanged; they select
your local bundle. Changing available rigs changes exposure, so record the actual
bundle camera subsets and weights.

## Stage settings overlaid on the bundle

| Setting                       | TF / TF-dCM                          | SF-DMD        |
| ----------------------------- | ------------------------------------ | ------------- |
| Requested camera frames       | 400                                  | 96            |
| Exact joint-aligned SF window | Not fixed by this override           | 93 RGB frames |
| Resolution mixture            | 480: 0.6; 720: 0.4                   | Same          |
| Caption / per-view dropout    | 0.2 / 0.2                            | Same          |
| Whole LiDAR dropout           | 0.1, target and control together     | Same          |
| Initial RGB latent prefix     | 0/1/2 with probabilities 0.7/0.2/0.1 | Same          |

At 30/10 Hz and the supplied tokenizer geometry, those RGB prefixes align with
0/1/3 measured sweeps. Do not independently sample camera and LiDAR prefixes.
Keep paired LiDAR absence valid, not a nonempty control with a missing target.

Supply sufficient source coverage. The example contains 300 source RGB frames
(and requests 297 for inference), insufficient for the full 400-frame training
request. Source IDs, timestamps, range units, invalid-return masks and projection
must agree; frame counts alone are insufficient.

For the supplied 720p example, apply the
[short training settings](../../cosmos3-nano-transfer-auto/shared/reference-scene.md#short-training-check)
after the camera-map overrides: remove the absent 480 stream and shorten the
window explicitly. This checks only the reduced setup, not the 60/40 mixture
or the full camera set.
