# Cosmos3 Nano Sim Bimanual evaluation

Evaluate generated video against time-aligned held-out RGB targets. Generate
predictions with the [inference tutorial](../SKILL.md), keeping checkpoint,
actions, prompt, frame rate, preprocessing, and sampling settings fixed.

Read the [model contract](../../shared/model-contract.md),
[environment requirements](../../shared/release-workflow.md), and
[evaluation contract](#evaluation-contract). Framework documents its two scoring
paths in [evaluation](../../../../../docs/evaluation.md).

## Evaluation contract

### Prepare held-out data

Split data by source episode so windows from one episode stay in one split.
For each evaluation sample, provide:

- the generated MP4;
- the aligned target MP4;
- the input identity, prompt, actions, and seed;
- the checkpoint revision and selected weight variant; and
- the generation and preprocessing settings.

Prediction and target must have the same frame count, dimensions, timestamps,
and preprocessing. Fully decode both files before scoring.

### Score generated videos

Set `PREDICTIONS_DIR` to the inference output directory, `GT_DIR` to the
aligned target directory, and `EVAL_OUTPUT` to a new directory. For prediction
layout `<group>/<key>/<output_index>/vision.mp4`, place the target at
`$GT_DIR/<key>.mp4`.

```bash
: "${PREDICTIONS_DIR:?set the generated inference output directory}"
: "${GT_DIR:?set the directory of aligned held-out target videos}"
: "${EVAL_OUTPUT:?set a fresh evaluation output directory}"
python -m cosmos_framework.scripts.eval \
  -o "$EVAL_OUTPUT" --checkpoint-path "" \
  --dataset.model-mode vision \
  --predictions-dir "$PREDICTIONS_DIR" --gt-dir "$GT_DIR"
```

The vision scorer does not load a model, but its CLI requires an empty
checkpoint path. It writes per-sample `metrics.json` files and a
`metrics_aggregate.json` summary. Review any shortened comparison, missing
target, shape mismatch, or skipped sample before using the aggregate.

Dataset-driven `forward_dynamics` scoring uses a complete dataset
configuration rather than an inference JSONL or local training bundle.

### Compare checkpoints

Use identical sample membership, targets, preprocessing, resolution, frame
count, frame rate, seed policy, sampling settings, package version, and metric
implementation. Keep placement and runtime mode fixed when comparing latency.
If one of these changes, label the result as a recipe comparison.

The native vision scorer reports PSNR. Report a pass/fail result only when the
application defines a threshold. Otherwise report:

- planned, successful, failed, and skipped sample counts;
- aggregate PSNR and its dispersion;
- per-sample metric rows; and
- representative outputs selected by a stated rule.

Recompute the aggregate from the included per-sample rows and keep every
planned sample in completeness accounting.

### Save the evaluation record

Save the generation command and resolved settings, checkpoint revision and
weight variant, input-manifest digest, split definition, sample-selection seed,
per-sample status and metrics, aggregate metrics, and any application-specific
acceptance threshold.
