# Models used to prepare LiDAR evaluation inputs

The LiDAR evaluators score saved arrays and JSON. They do not download weights,
import a feature backbone or detector, or require a model-specific CUDA runtime.
See [LiDAR evaluation](lidar_evaluation.md) for commands and metric definitions.

| Metrics                                       | Inputs to the scorer                                     | Model needed to prepare those inputs?                                                       |
| --------------------------------------------- | -------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Tokenizer range errors, validity IoU, Chamfer | Reference and reconstructed range maps, masks, or points | The tokenizer being evaluated produces the reconstruction. No additional model is required. |
| BEV JSD / MMD²                                | Reference and generated XYZ                              | No additional model                                                                         |
| CTC, Vehicle CTC, object support              | XYZ, poses, and object annotations as applicable         | No additional model; supplied annotations are used directly                                 |
| FUD / Utonia MMD²                             | Reference and generated feature matrices                 | Utonia when extracting features from point clouds                                           |
| Detection precision / recall / F1 / AP / mAP  | GT object centers and scored predictions                 | A separately run detector when producing predictions from point clouds                      |

Feature extraction and object detection run outside this evaluation package. Keep
their compiled dependencies in a separate environment when they require a
different PyTorch or CUDA version. CPU tests of the scorers establish arithmetic
and input handling; they do not validate the external models or reconstruction
quality.

## Utonia features

Use the same encoder, weights, preprocessing, and output layer for the generated
and reference sets. Features from another backbone or pooling rule define a
different evaluation protocol.

The protocol uses these model identifiers:

- Source: [Pointcept/Utonia](https://github.com/Pointcept/Utonia/tree/da776a0bd3a48c6df83ac2ae0e27b26141cc7e31), commit `da776a0bd3a48c6df83ac2ae0e27b26141cc7e31`.
- Weight repository: [Pointcept/Utonia](https://huggingface.co/Pointcept/Utonia), revision `fb39b40d33d215d29d1d4717ffb86287f4700c4d`, file `utonia.pth`.
- Weight SHA-256: `a0a2e5e234ce943082ae9ba26bcff74e285f16ed8d04fed5c36ffe8dc30ea353`.

Use CUDA-enabled PyTorch and the compatible sparse-convolution,
`torch-scatter`, and FlashAttention dependencies described in the
[upstream installation instructions](https://github.com/Pointcept/Utonia/blob/da776a0bd3a48c6df83ac2ae0e27b26141cc7e31/README.md#installation).
These are producer dependencies, not additions to the Framework evaluation
installation. Upstream publishes its code under
[Apache-2.0](https://github.com/Pointcept/Utonia/blob/da776a0bd3a48c6df83ac2ae0e27b26141cc7e31/LICENSE);
obtain weights under the terms provided with their release.

For each selected sweep:

1. Supply valid sensor-frame XYZ in meters, with the ego vehicle at the origin,
   and zero arrays for absent color and normals. Apply the same sensor projection
   and range selection to both sets.
2. Apply `utonia.transform.default(scale=0.2, apply_z_positive=False,
   normalize_coord=False)`. In the pinned
   [transform implementation](https://github.com/Pointcept/Utonia/blob/da776a0bd3a48c6df83ac2ae0e27b26141cc7e31/utonia/transform.py#L1120-L1141),
   `0.2` scales coordinates; `GridSample` uses `grid_size=0.01` afterward.
   It is not a 0.2-meter voxel size: the corresponding input-space grid interval
   is 0.05 meters.
3. Use encoder mode with FlashAttention. Concatenate the deepest output with its
   two immediate pooling parents, unpool through the remaining parents, and use
   the inverse map to restore a feature for each original point. The public
   [inference example](https://github.com/Pointcept/Utonia/blob/da776a0bd3a48c6df83ac2ae0e27b26141cc7e31/README.md#quick-start)
   documents this hierarchy and inverse mapping.
4. Mean-pool over original points to obtain one float32 feature vector per sweep.
   Define an empty-cloud feature as a zero vector with the encoder's feature
   dimension, and record how many sweeps are empty.
5. Use the key `clip_id/frame_index` for each sweep. Seed NumPy and PyTorch with
   the first eight SHA-256 bytes of its UTF-8 encoding, interpreted as an unsigned
   big-endian integer; reduce modulo `2**32` for NumPy. Give generated and
   reference sweeps the same key and seed.

Save:

- `generated_features.npy`: finite floating array `[N,D]`.
- `reference_features.npy`: finite floating array `[N,D]`, with matching rows.
- `sample_keys.json`: a JSON list of `N` unique nonempty strings in row order.

Use at least two sweeps. Feature dimensions must match; do not normalize or mix
features from different extractors before scoring. Save the source revision,
checkpoint SHA-256, preprocessing arguments, pooling rule, package versions, and
CUDA / PyTorch versions beside these files. The numerical scorer accepts arrays
without inspecting those producer records, so the caller maintains protocol
consistency across comparisons.

## Detector predictions

The detection evaluator accepts predictions from an external, fixed detector.
For example, the public
[OmniHD-Scenes PointPillars model](https://github.com/TJRadarLab/OmniHD-Scenes/blob/b49d6dc2d570ac5d8e292ce0fe45d029be28fe4f/README.md#-model-zoo)
has source revision `b49d6dc2d570ac5d8e292ce0fe45d029be28fe4f` and a
[LiDAR model configuration](https://github.com/TJRadarLab/OmniHD-Scenes/blob/b49d6dc2d570ac5d8e292ce0fe45d029be28fe4f/projects/configs/PointPillars_NewScenes/pointpillars_LiDAR.py).
Obtain its model and environment from that project. Its
[installation instructions](https://github.com/TJRadarLab/OmniHD-Scenes/blob/b49d6dc2d570ac5d8e292ce0fe45d029be28fe4f/README.md#installation)
use an older OpenMMLab / PyTorch stack; detector inference is not part of the
Framework evaluator runtime.

When using a detector that requires XYZI, preserve point-aligned input intensity;
do not replace unavailable intensity with zeros or ones. Apply the actual
LiDAR-to-rig calibration before inference when the detector expects rig-frame
coordinates. Convert GT annotations into the same coordinates and units.

Export post-NMS predictions with the following schema:

```json
{
  "classes": ["car", "pedestrian", "rider", "large_vehicle"],
  "frames": [
    {
      "frame_id": "clip_a/0",
      "targets": [
        {"class_name": "car", "center_xy_m": [10.0, 2.0]}
      ],
      "predictions": [
        {"class_name": "car", "center_xy_m": [10.1, 2.0], "score": 0.9}
      ]
    }
  ]
}
```

Centers are planar `[x,y]` in meters. Prediction and target centers must use the
same coordinate frame. `frame_id` is unique; include negative frames with
`targets: []` and failed detections with `predictions: []`. Set `classes`
explicitly when comparing runs so the class set remains fixed, including classes
with no GT instances. Additional box fields may be retained for visualization;
the scorer uses only the fields shown above.

Record the detector source revision, weight SHA-256, model configuration,
intensity convention, calibration, class mapping, and postprocessing thresholds.
Keep these settings and the evaluated frame list fixed across compared models.
A detector's NMS IoU threshold is separate from the scorer's planar
center-distance matching thresholds. Preserve the detector's post-NMS scores for
AP ranking; the scorer applies its operating-point threshold when computing
precision, recall, and F1.
