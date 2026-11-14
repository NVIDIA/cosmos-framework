# Troubleshoot Auto post-training

> **Skill note (not in the source):** this file holds what the Transfer-Auto and Sim-Auto troubleshooting skills say about training, checkpoints, resume and distillation (cosmos-framework `main` 45fecab), word for word: the Transfer-Auto troubleshooting body, then its symptom guide, then the Sim-Auto troubleshooting body. Changed: the three titles and the headings below them are one or two levels down, the two "References" headings carry the model name, and link targets are re-pointed (links between the three files are anchors in this file). The inference half is in [Auto inference troubleshooting](../../cosmos3-inference-simulation-auto/references/troubleshooting.md). For [Transfer-Auto](../../cosmos3-inference-simulation-auto/references/troubleshooting.md#transfer-auto): the checks for a video timing or dimension problem and for side/rear failures, the inference sections of the symptom guide (side/rear morphing; weather/road intent; frame count; LiDAR; import error, OOM or distributed failure) and the two sections that serve inference and training alike ("Collect one failing example" and "Diagnose, correct, verify"). For [Sim-Auto](../../cosmos3-inference-simulation-auto/references/troubleshooting.md#sim-auto): the inference rows of the symptom table (side/rear geometry; short output or caption transition; missing metadata / HF load rejected; CFG or sampler behavior; LiDAR) and the text around the table that serves both (what to collect for one failing example; preserving inputs, applying the smallest correction and reporting).

## Troubleshoot Cosmos3-Nano-Transfer-Auto

`cosmos3-nano-transfer-auto` generates driving videos and optional LiDAR from layout
controls and per-camera captions. Start with one failing example and choose the
matching section of the [symptom guide](#cosmos3-nano-transfer-auto-common-problems-and-checks).
Use the [shared model contract](../../cosmos3-inference-simulation-auto/references/model-contract.md) for compatibility checks;
this model does not use the causal Sim-Auto inference workflow.

### References (Transfer-Auto)

- [Symptom guide and checks](#cosmos3-nano-transfer-auto-common-problems-and-checks)
- [Model contract](../../cosmos3-inference-simulation-auto/references/model-contract.md)

### cosmos3-nano-transfer-auto: common problems and checks

Use the [shared contract](../../cosmos3-inference-simulation-auto/references/model-contract.md),
[inference reference](../../cosmos3-inference-simulation-auto/references/input-and-execution.md)
and [training recipe](transfer-auto-recipe.md).

#### Storage read errors while training

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

#### Training resumes with wrong iteration/LR/EMA or wrong checkpoint

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

## Troubleshoot Cosmos3-Nano-Sim-Auto

| Symptom                                        | First checks                                                                                                                    |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Missing training camera/caption/control fields | [Bundle mappings](sim-auto-data.md#preserve-your-bundles-camera-mappings) at every stage; inspect decoded and formatted samples |
| Wrong teacher or resume iteration              | [Checkpoints](sim-auto-checkpoints.md): new stage versus complete same-stage resume                                             |
| OOM / compilation / distributed error          | Verify setup/mesh, then an authorized [smoke](sim-auto-smoke.md); preserve architecture/objective                               |
| Only one SF-DMD phase appears                  | Six iterations include one student and five critic updates; inspect phase metrics and both optimizer histories                  |

For reader errors, let bounded retries finish and check training progress. Stop
the affected run on a fatal failure, exhausted retries, invalid data or its time
limit; leave independent healthy jobs running and allow saves/exports to finish.

### References (Sim-Auto)

- [Shared camera and WSM input checks](../../cosmos3-inference-simulation-auto/references/input-preparation.md)
- [Sim-Auto input and request settings](../../cosmos3-inference-simulation-auto/references/sim-auto-inputs.md)
- [Checkpoint handoffs, resume and student export](sim-auto-checkpoints.md)
- [Training data and camera mappings](sim-auto-data.md)
- [Short training, save and resume checks](sim-auto-smoke.md)
