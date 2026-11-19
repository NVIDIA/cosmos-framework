# Troubleshoot Auto inference

> **Skill note (not in the source):** this file holds the inference symptoms of the Transfer-Auto and Sim-Auto troubleshooting skills and their stage-neutral text (collecting one failing example; diagnosing, correcting, verifying and reporting), word for word. Training, checkpoint, resume and distillation symptoms are in the troubleshooting reference of the Auto post-train skill (`cosmos3-post-train-simulation-auto`): [Transfer-Auto](../../cosmos3-post-train-simulation-auto/references/troubleshooting.md#troubleshoot-cosmos3-nano-transfer-auto) and [Sim-Auto](../../cosmos3-post-train-simulation-auto/references/troubleshooting.md#troubleshoot-cosmos3-nano-sim-auto). "The symptom guide" named below is the Transfer-Auto symptom sections of this file together with the training sections of the [Transfer-Auto symptom guide](../../cosmos3-post-train-simulation-auto/references/troubleshooting.md#cosmos3-nano-transfer-auto-common-problems-and-checks) in that reference; "the inference reference" is [Framework requests and sampling](input-and-execution.md).

## Transfer-Auto

### Collect one failing example

Collect the exact request/command, resolved config, checkpoint and weight variant,
per-view text actually consumed, camera-key→control mapping, selected time span,
one failing output with controls, and the error/log excerpt. Keep credentials and
confidential source locations out of shared diagnostics.

For a video timing or dimension problem, first compare the actual files:

```bash
: "${TRANSFER_AUTO_CONTROL:?Set an existing local WSM video path}"
: "${TRANSFER_AUTO_RGB:?Set its corresponding local generated RGB video path}"
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,r_frame_rate,avg_frame_rate,nb_frames:format=duration \
  -of json "$TRANSFER_AUTO_CONTROL"
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,r_frame_rate,avg_frame_rate,nb_frames:format=duration \
  -of json "$TRANSFER_AUTO_RGB"
```

Inputs are the two actual files; outputs are metadata reports. Success means both
probe and their timing/dimensions agree with the resolved request after documented
preprocessing. If `nb_frames` is missing, count decoded frames. Inspect synchronized
frames to check view identity and event alignment.

### Diagnose, correct, verify

Use the symptom guide to check likely causes, propose the smallest correction,
and verify the affected inputs and outputs afterward. Preserve the original
manifest, captions and outputs.

For side/rear failures, verify the camera mapping, controls and actual captions,
then compare a reference input with the failing scene, changing one suspected
factor at a time.

Return observations, hypotheses, the checks actually run, proposed corrections,
and remaining uncertainty. Pass a corrected request to the
[inference Skill](../SKILL.md#transfer-auto), or a state/topology issue to
[post-training](../../cosmos3-post-train-simulation-auto/SKILL.md#fine-tune-transfer-auto).

### Side/rear morphing, front acceptable

- **Likely causes:** camera-channel/content mismatch; stale or incompatible WSM
  rendering; source/control crop or clock mismatch; forward prose supplied to other
  views; a model/generalization limitation after those checks.
- **Check:** trace each actual request `camera_key`→control path→inspected pixels;
  compare against verified mapping/calibration/provenance. Inspect consumed per-view
  captions, not only a scene-level string in an export. Probe timing and inspect
  synchronized 50% overlays at several times, with resize/crop documented.
- **Correction:** propose a mapping/control/text correction backed by evidence;
  preserve originals and get authorization to produce a new request. Do not infer
  physical camera identity solely from a good overlay or automatically swap files.
- **Verify:** reproduce a supplied reference case, then isolate one change in the
  failing case with matched weights/settings/seed. Inspect all affected cameras.
- **Revalidate:** view/channel, control/target geometry and time, captions, then
  output geometry, motion and visual quality.

### Weather/road intent missing or inconsistent across views

- **Likely causes:** conflicting positive/negative text, incorrect per-view sidecars,
  incompatible shared appearance assignments, uncertain renderer interpretation,
  or insufficient model adherence.
- **Check:** resolve text per camera; compare meaning against the four scenario
  properties and inspected WSM. Night plus sunlit facades is a contradiction; an
  ocean absent from a rear view is not evidence a coastal setting failed.
- **Correction:** in review mode give evidenced suggested edits, not silent rewrites.
  In generation mode create new camera-local bodies with one consistent appearance
  record. Preserve required rain/fog rather than removing them to improve sharpness.
- **Verify:** resolved reader text reflects the intended bodies; run a matched case
  and inspect the affected views.
- **Revalidate:** caption formatting, CFG inputs, scenario semantics and affected views.

### Requested frame count differs from saved video

- **Likely causes:** caption-derived duration, tokenizer grid snapping, wrong sidecar
  bounds, first-chunk selection, nonzero caption start paired with file-head initial
  RGB/joint controls, or insufficient source coverage.
- **Check:** compare manifest bounds, effective chunk override, resolved sample args,
  frame IDs and probed/decoded output count using the inference reference.
- **Correction:** select an aligned span supported by the actual controls and all
  captions. For a nonzero source interval, prepare a synchronized crop and
  rebase the sidecar bounds to zero. Preserve the source FPS.
- **Verify:** frame count, final timestamps, initial RGB condition and controls agree
  after documented preprocessing.
- **Revalidate:** all sensors' selected intervals; joint sweep counts are not implied
  by RGB frame count alone.

### Missing LiDAR, wrong sweep count or unexpected prefix

- **Likely causes:** omitted LiDAR request block, missing HD-map control, wrong
  nested prefix fields, source-ID mismatch, or the documented training/inference
  sweep-count distinction.
- **Check:** resolved joint mode, control versus target-prefix paths, per-sensor time
  origins, actual output frame IDs and the expected output inventory. A visual
  preview is not a substitute for tensor units/masks/IDs.
- **Correction:** use the supported nested bidirectional prefix interface and verified
  input range format. Do not translate a causal example's fields by intuition or
  relabel HD-map control as measured initial LiDAR.
- **Verify:** complete finite range/intensity outputs and synchronized correspondence
  at start/end.
- **Revalidate:** range units, source alignment, conditioned sweep mask and joint output.

### Import error, OOM or distributed failure

- **Likely causes:** incompatible runtime dependencies; invalid CP/CFGP mesh; source
  length/views beyond memory; compile/offload option conflicts; host loader pressure.
- **Check:** preserve first failure and exact command/environment identity. Run CLI
  help or the focused recipe check before downloading data; verify replica arithmetic.
  Separate host, raw-pixel/VAE and denoiser memory failures.
- **Correction:** use the matching prepared environment; start from a smaller aligned
  smoke with compilation/cache off. Choose an implementation-supported mesh. Treat
  CPU offload/staging as memory/performance tradeoffs. Preserve attention settings
  and strict checkpoint loading.
- **Verify:** the isolated import/config check, then bounded execution, then the
  original request if resources permit.
- **Revalidate:** output parity and, for training topology changes, batch/LR/schedule
  arithmetic and save/resume.

## Sim-Auto

Collect one failing request or training step, runtime/source revision, resolved
config, artifact identity, logs and relevant outputs. Keep confidential inputs and
credentials out of shared reports.

| Symptom                                    | First checks                                                                                                                                                                     |
| ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Side/rear geometry mismatch                | Shared [input preparation](input-preparation.md): physical camera mapping, synchronized WSM, crop and actual per-view text                                                       |
| Short output or caption transition failure | [Input settings](sim-auto-inputs.md#request-fields): total horizon, source coverage, sidecar bounds and first-chunk selection                                                    |
| Missing metadata / HF load rejected        | [Export](../../cosmos3-post-train-simulation-auto/references/sim-auto-checkpoints.md#export-the-student): regular student, rolling profile, per-view text and matched tokenizers |
| Unexpected CFG or sampler behavior         | Fixed four-step student; use the guidance-1 recipe, not bidirectional RF step/shift settings                                                                                     |
| Missing or misaligned LiDAR                | HD-map controls versus measured prefix, 30/10 Hz clocks, source IDs; causal measured prefix is at most one sweep                                                                 |

Preserve original inputs and the model's checkpoint/attention settings while
isolating the issue. Apply the smallest correction and check the affected result.
Report observations, the confirmed cause or remaining hypothesis, the correction
and checks performed.
