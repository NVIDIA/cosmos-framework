# cosmos3-nano-transfer-auto: common problems and checks

> **Skill note (not in the source):** the inference sections of this guide (side/rear morphing; weather/road intent; frame count; LiDAR; import error, OOM or distributed failure) moved to [Auto inference troubleshooting](../../../cosmos3-inference-simulation-auto/references/troubleshooting.md#transfer-auto).

Use the [shared contract](../../../cosmos3-inference-simulation-auto/references/model-contract.md),
[inference reference](../../../cosmos3-inference-simulation-auto/references/input-and-execution.md)
and [training recipe](../../post-training/references/canonical-recipe.md).

## Storage read errors while training

- **Check:** read the final retry/skip outcome and identify the failing path.
  Check file access, worker health, decoded samples and optimizer progress.
- **Recovery:** correct the path, access or storage issue while preserving the
  dataset and retry policy. Track skipped samples separately from successful reads.
  Distinguish transport errors from authorization failures; keep TLS verification
  and the configured sample-skip policy unchanged.
- **Stop:** let bounded retries finish. On a fatal failure, exhausted retries,
  invalid data or the run's time limit, stop the affected run and preserve its logs.
  Leave independent healthy runs running, and allow their checkpoint/video export
  to finish.
- **Verify:** confirm reads recover and training advances with the intended data.

## Training resumes with wrong iteration/LR/EMA or wrong checkpoint

- **Likely causes:** wrong model/config pair; warm-start settings mistaken
  for resume; same-job latest pointer overriding an external path; later launch
  overrides mistaken for source defaults; incomplete save.
- **Check:** exact loaded artifact and state inventory; resolved config/CLI overrides;
  iteration, LR/scheduler position, EMA age, RNG and data cursor. Inspect the actual
  output directory and checkpoint selected by the loader.
- **Correction:** restore the intended complete checkpoint with its compatible
  saved run configuration, or propose a separately named warm-start experiment. Preserve source
  artifacts; never delete a latest pointer or reset state without authorization.
- **Verify:** first resumed step and a completed save/reload preserve intended state.
- **Revalidate:** model, optimizer/schedule/EMA, data exposure/cursor and any topology
  changes.
