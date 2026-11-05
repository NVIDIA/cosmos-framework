# Evaluation

`cosmos_framework.scripts.eval` scores a checkpoint against ground truth. It has two paths, and
`--dataset.model-mode` picks one:

| `--dataset.model-mode`                                           | What runs                                                                                                                         | Per-sample metrics                                                    |
| ---------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| `joint` (default), `forward_dynamics`, `inverse_dynamics`, `wam` | Loads samples from a training dataset config, generates each one with the model, then scores it against the sample's ground truth | `forward_dynamics`: PSNR. `inverse_dynamics`: action MSE. `wam`: both |
| `vision`                                                         | Scores videos you have already generated against a directory of ground-truth videos. It loads no model and needs no GPU.          | PSNR                                                                  |

`joint` runs all three action modes, one after the other.

## Action evaluation

The dataset comes from the `config.yaml` that training saved in the run directory. Evaluate a
checkpoint that has been exported to Hugging Face safetensors (see
[Training → Export checkpoint](./training.md#export-checkpoint-to-hugging-face-safetensors)):

```shell
torchrun --nproc-per-node=8 -m cosmos_framework.scripts.eval \
  -o outputs/eval \
  --checkpoint-path $RUN_DIR/model \
  --dataset.config-file $RUN_DIR/config.yaml \
  --dataset.model-mode forward_dynamics \
  --dataset.num-samples 16
```

Common dataset arguments:

- `--dataset.model-mode`: the mode to evaluate (see the table above).
- `--dataset.num-samples N`: evaluate at most `N` samples per mode. By default the `N` samples are
  a deterministic pseudo-random subset of the candidates (stable for the same dataset label and
  seed), not the first `N`.
- `--dataset.sample-selection-strategy {deterministic_random,first}`: how `--dataset.num-samples`
  picks its subset. Use `first` for the leading samples.
- `--dataset.sample-selection-seed S`: seed for `deterministic_random` (default 42).
- `--dataset.dataset NAME`: choose one dataloader or dataset by name when the config defines
  several. If you leave it out, the first one is used.
- `--dataset.dataset-split {val,train,full}`: the split to load. The default is `val`. With
  `val` or `full`, training-only caption dropout is turned off.
- `--dataset.sample-stride K`: evaluate every `K`-th sample.
- `--dataset.dataset-kwargs` / `--dataset.transform-kwargs`: override fields of the inner dataset
  config or the outer dataset config (for example its transform) before instantiation. Nested
  dictionaries are merged, and `resolution` is only written when the dataset config declares it.
- `--no-compute-metrics`: generate the samples and skip scoring.

The usual setup arguments, such as `--checkpoint-path`, parallelism and sampling overrides, work
the same way as in [Inference](./inference.md).

For video metrics, the ground-truth clip is taken from the batch before generation starts. Its
reflection padding is removed and it is cropped to the tokenizer's spatial alignment, so it has
the same height and width as the generated `vision.mp4`. For action metrics, the generated action
is compared with the raw ground-truth action (`action_raw`). If the batch has a channel mask
(`action_valid_mask`), only the valid channels count.

Outputs, under `-o`:

- `<dataset>/<mode>/<index>/`: the generated outputs and that sample's `metrics.json`.
- `metrics_aggregate.json`: `{mode: {metric: {mean, count}}}`, collected from every
  `metrics.json` under the output directory.
- `benchmark.json`: timings, written only with `--benchmark`.

## Scoring pre-generated videos

To score videos produced by [`cosmos_framework.scripts.inference`](./inference.md), point the
script at the predictions and at a directory of ground-truth videos:

```shell
python -m cosmos_framework.scripts.eval \
  -o outputs/eval_vision \
  --checkpoint-path "" \
  --dataset.model-mode vision \
  --predictions-dir outputs/inference \
  --gt-dir /path/to/gt_videos
```

No model is loaded in this mode. The CLI still requires `--checkpoint-path`, so pass an empty
string.

- `--predictions-glob` (default `**/vision.mp4`) selects the prediction files under
  `--predictions-dir`.
- Each prediction is paired with `<gt-dir>/<key><gt-extension>`. The default `--gt-extension`
  is `.mp4`. For the inference output layout `<group>/<key>/<output_index>/vision.*`, the key is
  the sample directory name. For any other file name, the key is the file stem. Predictions with
  no matching ground truth are logged and skipped.
- If the frame counts differ, both clips are trimmed to the shorter one. Generation produces
  `4k+1` frames, while the ground truth keeps its original length. A difference in height or
  width skips that clip.
- Metrics are grouped by the path between `--predictions-dir` and the sample directory.
  Predictions at the top level are grouped under `default`. Each `metrics.json` is written to the
  same relative location under `-o`, and `metrics_aggregate.json` lists one entry per group.
