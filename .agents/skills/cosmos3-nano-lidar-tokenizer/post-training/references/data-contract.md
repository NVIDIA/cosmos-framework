# Measured LiDAR data contract

Supply local measured-LiDAR `.tar` archives accepted by
`cosmos_framework.model.generator.tokenizers.lidar.preprocessing.load_prepared_lidar_video_3ch`.
Use the existing rangemap writer
`save_range_maps_to_tar` in
`.agents/skills/cookbooks/cosmos3/simulation/auto/wsm/cosmos_auto/lidar/rangemap.py`;
its arguments are range maps, source frame IDs, clip name, output path and
optional intensity maps. Read its signature in your checkout before conversion.
The same native reader is used in training and evaluation.

- Sensor: Pandar128 beam-elevation calibration, 128 rows, native 3600 columns,
  azimuth 180° to −180° with endpoint excluded, sensor coordinates x-forward,
  y-left, z-up. A different 128-beam sensor requires a calibrated geometry
  adaptation, not just relabeling metadata.
- Metric range band: 5–100 metres; zero denotes invalid. Values outside the
  band are invalid.
- Intensity: unit scale [0,1], registered to the same selected physical return.
  Provide real intensity for this range+intensity objective; the reader's
  synthetic unit-intensity compatibility path is not a substitute for measured labels.
- Adjacent pairs of native azimuth columns select the nearest valid return,
  carrying its intensity, producing 1800 semantic columns. Already-pooled
  1800-column format-v2 archives are supported by the reader; record that
  provenance so they are not pooled a second time.
- Symmetric circular padding adds four columns on each side to reach 1808.
  Input is `[3,9,128,1808]` in metric range/unit intensity/validity order;
  normalized network input is produced inside the model. Targets contain
  two channels at width 1800; validity is a separate channel at that width.
- Clips use eight source sweeps, plus one duplicate first frame. Short clips
  repeat the last sweep; remove unintended short clips before accepting a corpus.
  The adapter truncates longer clips to the first eight sweeps. Use explicit
  contiguous-window archives to control exposure on long recordings.

Inspect one archive and a decoded batch before launching. Verify frame ordering,
units, calibration, mask occupancy, intensity and seam behavior. HD-map range
controls and generated predictions are not measured-LiDAR targets. Supply
measured sweeps for every training sample.

Manifests are UTF-8 files with one path per nonempty line, with no header or
comments. Duplicate resolved paths and missing archives fail preflight. Keep
training and held-out sessions disjoint; overlapping paths are rejected but
renamed duplicate recordings still need to be removed by the dataset owner.
Keep these data and checkpoints outside the repository.

The fixed batch is two clips per rank and incomplete batches are dropped. Supply
at least `2 × world_size` training clips so every rank can construct one batch.
