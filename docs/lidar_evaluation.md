# LiDAR evaluation

Use `python -m cosmos_framework.scripts.evaluate_lidar` to score tokenizer
reconstructions or generated LiDAR. Scoring runs on CPU using NumPy and SciPy.
The optional `framework-tokenizer` command also loads native tokenizer weights.

| Command               | Inputs                                                | Learned model required at scoring time                |
| --------------------- | ----------------------------------------------------- | ----------------------------------------------------- |
| `tokenizer`           | Paired metric range-view NPZ files                    | None                                                  |
| `framework-tokenizer` | Local recordings, native tokenizer weights and config | Your tokenizer and its Torch/NATTEN runtime           |
| `prepare`             | Generated range views and measured references         | None; sparse tar loading uses Framework preprocessing |
| `clouds`              | Paired XYZ frames; poses for temporal metrics         | None                                                  |
| `objects`             | Paired XYZ frames, poses and tracked GT cuboids       | None                                                  |
| `features`            | Precomputed features and ordered sample keys          | None; feature production requires Utonia              |
| `detections`          | Precomputed predictions and GT centers                | None; prediction production requires a detector       |

The [optional model protocols](./lidar_evaluation_models.md) specify feature
extraction, detector requirements and reproducibility metadata. This package
does not download weights or run those external producers. Keep their model,
preprocessing and test population fixed when comparing checkpoints.

## Tokenizer reconstruction

Create a JSON list such as `pairs.json`. Paths are relative to the manifest:

```json
[{"id": "clip_001", "reference": "reference.npz", "prediction": "prediction.npz"}]
```

Each NPZ contains `range_m` (finite metric `[T,H,W]`), `valid` (boolean of the
same shape) and `sweep_ids` (integer `[T]`). Reference and prediction IDs must
match exactly. Prediction distances must remain **unmasked**, including rays
classified as invalid; threshold predicted validity probabilities before saving
the boolean mask. Do not use already masked generator outputs for range MAE.

```shell
python -m cosmos_framework.scripts.evaluate_lidar tokenizer \
  --manifest pairs.json --aggregation macro --output tokenizer_scores.json
```

The default projection uses the calibrated 128-beam sensor geometry already
provided by Framework and a GT range band of `[5,100]` meters. For another rig,
pass `--projection projection.json`, explicitly declaring `elevation_degrees`
(one per row), `semantic_width`, `model_width`, and, when different from the
defaults, `azimuth_start_degrees`, `azimuth_end_degrees` and `azimuth_endpoint`.
Only explicitly declared symmetric circular padding is cropped; evaluation never resizes rays.

Let `g` and `p` be GT and predicted ranges, `V` the GT-valid rays, and `U` the
predicted-valid rays. Error reductions always use `V`, including false negative
predicted rays. Chamfer uses the valid XYZ points from each arm.

| Metric           | Calculation                                                                             | Direction |
| ---------------- | --------------------------------------------------------------------------------------- | --------- |
| Range MAE        | `mean(abs(p-g))` over `V`, meters                                                       | Lower     |
| Range median AE  | Median of `abs(p-g)` over `V`; midpoint for an even count                               | Lower     |
| AbsRel           | `mean(abs(p-g)/g)` over `V`                                                             | Lower     |
| Chamfer distance | Half the sum of both mean nearest-neighbor Euclidean distances; **not squared**, meters | Lower     |
| Validity IoU     | Intersection count divided by union count; both empty gives 1                           | Higher    |

`macro` first scores each frame, averages within its clip, then averages clips
equally. The median metric is therefore a mean of frame medians. `micro`
provides pooled MAE and IoU only. Chamfer caps each arm at 50,000 points by
default, with deterministic frame seeds; override with `--chamfer-max-points`.
Empty point sets produce missing Chamfer scores with explicit coverage counts.
For comparisons that require a fixed full-resolution Chamfer reference, add
`chamfer_reference` to each pair; it is loaded using its own range-view width.

### Run native tokenizer weights

Prepare a directory containing native `.pt` or `.safetensors` weights, including
their latent statistics, and a `config.json`:

```json
{
  "checkpoint_file": "tokenizer.safetensors",
  "checkpoint_sha256": "<SHA256 of the weight file>",
  "network_config": {"<TransformerVAE constructor argument>": "<value>"},
  "range_projection": {
    "sensor": "pandar128",
    "semantic_width": 1800,
    "model_width": 1808,
    "model_width_transform": "circular_pad"
  }
}
```

Use the exact `network_config` that produced the weights. The adapter verifies
the checksum and constructs the native model with strict state compatibility.
Omitting `checkpoint_file` selects `tokenizer.pt`. This reference protocol uses
FP32, posterior mean, temporal compression 1 and circular padding.

`recordings.txt` contains one existing local `.tar` path per line, relative to
the manifest. The native recording schema is documented in
[LiDAR input documentation](./inference.md#multiview-camera--lidar-transfer). Run:

```shell
python -m cosmos_framework.scripts.evaluate_lidar framework-tokenizer \
  --artifact tokenizer_export --manifest recordings.txt \
  --output-dir outputs/tokenizer_evaluation --device cuda
```

Each clip uses its first eight sweeps with a duplicated first frame for model
input. That duplicate is excluded from scoring. Short clips repeat their last
sweep to eight; these repeats remain scored and are recorded in `scores.json`.
The output includes aligned unmasked NPZ pairs, projection, macro/micro scores,
checkpoint checksum and source recording checksums. Use a fresh output directory.
This adapter uses interfaces on main and has no dependency on a post-training PR.

## Prepare generated range views

The native `lidar.safetensors` writer stores metric `range_m`, optional measured
or predicted `intensity`, and source `frame_ids`. These saved ranges are already
masked and smoothed; the converter preserves them without another smoothing pass.
NPZ files with `range_m`, `valid`, `sweep_ids` and optional `intensity` also work.

```json
[
  {
    "clip_id": "clip_001",
    "generated": "generated/lidar.safetensors",
    "reference": "reference.tar",
    "lidar_to_world": "poses.npy",
    "annotations": "objects.json"
  }
]
```

```shell
python -m cosmos_framework.scripts.evaluate_lidar prepare \
  --manifest range_pairs.json --output-dir outputs/prepared \
  --select-reference-frames --pool-reference
python -m cosmos_framework.scripts.evaluate_lidar clouds \
  --manifest outputs/prepared/clouds.json --output generation_scores.json
```

Both flags are explicit choices. `--select-reference-frames` selects GT by exact
generated source IDs; otherwise IDs must already match. `--pool-reference`
retains the nearest valid return in each adjacent azimuth group, together with
its intensity. Sparse GT tar loading always requires this flag because native
projection includes nearest-return pooling. Custom rigs use NPZ references.
Poses and annotations must already correspond to the final selected frames.

The converter writes a `clouds.json` manifest, XYZ frames, aligned metric NPZ
files, and XYZI frames when genuine intensity is available. It never invents
intensity. Output directories must be new. Optional poses and annotation paths
are referenced rather than copied.

## Generation metrics

`clouds.json` can also be prepared directly:

```json
[
  {
    "clip_id": "clip_001",
    "generated": ["g0.npy", "g1.npy"],
    "reference": ["r0.npy", "r1.npy"],
    "lidar_to_world": "poses.npy",
    "annotations": "objects.json"
  }
]
```

Each frame is a finite floating-point `[N,3]` XYZ array in sensor coordinates,
in meters. Lists have matching lengths and exact corresponding time order.
Poses are rigid sensor-to-world `[T,4,4]` transforms shared by both arms.
Without poses, only BEV metrics run. CTC requires at least two frames.

| Metric                            | Calculation and aggregation                                                                                                                                                                                                 |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| BEV JSD                           | Square root of Jensen-Shannon divergence using natural logs. Global pools all point counts; local pools within each clip, then averages clips.                                                                              |
| BEV MMD²                          | Biased RBF MMD on per-frame normalized BEV histograms, including same-arm diagonals, bandwidth 0.5. Average clip scores equally.                                                                                            |
| FUD                               | Squared feature-mean distance plus covariance traces minus twice the PSD cross-root trace. Covariances divide by `N-1`; float64 reductions.                                                                                 |
| Utonia MMD²                       | Unbiased MMD using `(x dot y / D + 1)^3`, excluding same-arm diagonals. Select at most 4096 paired rows deterministically by sample-key SHA256. Negative estimates are valid.                                               |
| CTC / ratio                       | Mean adjacent-frame Chamfer after applying GT poses, cap 30k points/frame. Average pairs within clips, then clips equally. Ratio is generated corpus mean divided by positive GT corpus mean.                               |
| Vehicle CTC / ratio               | Track-local Chamfer between adjacent GT cuboid crops. Require at least 30 GT returns in both frames; an empty generated crop incurs 2m. Equal means over pairs within eligible clips, then eligible clips.                  |
| Object support recall / miss rate | GT cuboid frames with at least 30 returns are eligible; at least 10 generated returns count as support. Pool object-frame counts; miss rate is `1-recall`.                                                                  |
| Detection precision / recall / F1 | Confidence at least 0.05, greedy same-class/same-frame nearest XY-center matching, distance strictly less than 3m. Pool TP/FP/FN.                                                                                           |
| Detection AP / mAP                | Rank all supplied confidences, match at 1/2/3/4m. Interpolate precision at 101 recall points, retain recall greater than 0.1, subtract precision offset 0.1, clip at zero and divide by 0.9. Average class/threshold cells. |

BEV uses point counts, 3D radius `[5,100)`m, XY `[-100,100)`m, and a 100 by 100
grid (2m cells). An empty histogram gives missing JSD. Corpus CTC is missing
unless all clips have a score; coverage is reported. Zero GT CTC makes the
ratio undefined and raises an error. Lower CTC can indicate excessive stillness,
so compare it alongside distribution and object metrics. Ratios are relative to
the GT baseline, whose reference value is 1.

Object annotations are a list of `T` dictionaries keyed by persistent track ID:

```json
[
  {
    "track_1": {
      "object_to_world": [[1,0,0,10],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
      "lwh_m": [4,2,2],
      "object_type": "Vehicle"
    }
  }
]
```

The crop adds 0.2m to each half extent. Object support covers all supplied
classes; Vehicle CTC uses vehicle classes. Run:

```shell
python -m cosmos_framework.scripts.evaluate_lidar objects \
  --manifest clouds.json --output object_scores.json
python -m cosmos_framework.scripts.evaluate_lidar features \
  --generated generated_features.npy --reference reference_features.npy \
  --sample-keys sample_keys.json --output feature_scores.json
python -m cosmos_framework.scripts.evaluate_lidar detections \
  --manifest detector_predictions.json --output detection_scores.json
```

Features are paired finite real `[N,D]` arrays, `N >= 2`, with unique ordered
JSON sample keys. Feature extraction and detector JSON examples are in the
[model protocols](./lidar_evaluation_models.md). Detection mAP includes absent
GT classes as zero; `map_present_classes` reports a separate diagnostic.
Frames without GT must remain present to count false positives. Detector NMS
and class/range selection happen before scoring and must be recorded.

## Validation

Run the focused CPU suite in a configured Framework development environment:

```shell
python -m pytest cosmos_framework/evaluation/lidar \
  cosmos_framework/scripts/evaluate_lidar_test.py --num-gpus=0
python -m cosmos_framework.scripts.evaluate_lidar doctor
```

Tests cover analytic metric values and edge cases, native safetensors output
conversion, and a small real native tokenizer loaded from both checkpoint
formats. The CLI is exercised from a separate directory with development-only
imports blocked. These checks establish format and calculation behavior;
full-size fused-NATTEN CUDA inference and external model inference require
their own environment and weights.
