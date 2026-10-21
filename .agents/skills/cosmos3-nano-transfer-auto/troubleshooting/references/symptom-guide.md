# cosmos3-nano-transfer-auto: common problems and checks

Use the [shared contract](../../shared/model-contract.md),
[inference reference](../../inference/references/input-and-execution.md)
and [training recipe](../../post-training/references/canonical-recipe.md).

## Side/rear morphing, front acceptable

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

## Weather/road intent missing or inconsistent across views

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

## Requested frame count differs from saved video

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

## Missing LiDAR, wrong sweep count or unexpected prefix

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

## Import error, OOM or distributed failure

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
