# Generated output files

Each native inference request writes `sample_args.json`, `sample_outputs.json`
and its generated media. `sample_outputs.json` records the request status and
lists output files; a successful generation has `status: success`. The run's
`setup_args.json` records setup options.

Re-running the same request regenerates the video and overwrites matching
files, including successful outputs. Use a fresh output directory to retain
previous results for comparison.

With the walkthrough's `transfer_output_size: model`, `vision.mp4` is the
generated RGB video:

| Example                 | Expected native RGB                 |
| ----------------------- | ----------------------------------- |
| `example2`              | 832 × 480, 121 frames at 30 FPS     |
| `example1` long rollout | 640 × 640, all 600 frames at 30 FPS |

`control_<modality>.mp4` is the conditioning video.

With `transfer_output_size: original`, `vision.mp4` is the source-size copy
and `vision_model.mp4` retains the native output. The saved `transfer_geometry`
metadata identifies source, model and delivery dimensions and any encoder
padding. Optional `vision_preview.mp4` is a more compressed copy at the same
canvas.

For comparisons, use model-native RGB and hold source intervals, modality,
caption, seed, preprocessing, sampler and history settings fixed. Keep the
selected checkpoint and native request metadata with each result.
