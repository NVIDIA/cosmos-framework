# cosmos3-nano-transfer-auto

Turn synchronized road-layout videos (WSM controls) and camera-local captions into
driving videos. Supply initial RGB images to anchor appearance when wanted. With
HD-map range controls, the model can generate LiDAR alongside the camera videos.

Transfer-Auto uses bidirectional generation. For causal streaming generation,
use [Sim-Auto](../SKILL.md#sim-auto).

## Inputs that must stay consistent

- Match each control, caption and initial image to its physical camera. The assigned
  camera key selects a learned identity; neither filenames nor pixels correct a
  wrong assignment automatically.
- Use WSM v4.0.1 controls for both Auto models; the renderer selector is `version=v4`.
- Use the same scene interval, camera projection and crop for controls and targets.
  Include actual timestamps when checking camera/LiDAR alignment.
- For inference, get requested time of day, weather, road condition and geographic
  setting from the user or scenario metadata. For training, those descriptions must
  match the paired real targets and recording metadata; follow
  [training captions](../../cosmos3-nano-transfer-auto/post-training/references/data-contract.md#training-captions).
  WSM supplies traffic layout and motion, not those properties.
- Keep camera-local captions separate. Each camera reads its own text; LiDAR can
  read all camera captions.
- Use the model configuration and tokenizers supplied with the selected weights.
  Record the artifact version and whether the weights are regular or averaged
  (EMA). Do not choose a different checkpoint merely because its name looks similar.

## Model settings, not user setup steps

The released model uses learned sensor identities and bidirectional attention within
each camera. Its additional cross-camera/sensor connections read the preceding
0.4 seconds, including the current time, without double-counting overlapping keys.
Controls retain their configured
same-view connections. Preserve these settings from the supplied model config when
fine-tuning, loading or exporting; do not rebuild them from a product-name string.

The model supports 11 camera identities and a LiDAR identity.
Camera calibration helps validate/render controls; this recipe does not feed camera
pose tensors through a pose positional encoding.

## Choosing a workflow

- **Use the released model:** prepare controls/captions and run
  [inference](../SKILL.md#transfer-auto); no training history is required.
- **Adapt it to your data:** start from its delivered weights and final model
  settings, prepare paired targets/controls, and define a new fine-tuning run with
  [post-training](../../cosmos3-nano-transfer-auto/post-training/SKILL.md).
- **Recreate the curriculum:** follow the four training phases in the
  [recipe](../../cosmos3-nano-transfer-auto/post-training/references/canonical-recipe.md), starting
  from Cosmos3-Nano using your own paired data.

Use [post-training](../../cosmos3-nano-transfer-auto/post-training/SKILL.md), [inference](../SKILL.md#transfer-auto),
or [troubleshooting](troubleshooting.md#transfer-auto) for the selected workflow.
Keep both Auto bundles and their shared resources together.

## Execution prerequisites

Use the [Framework environment and model assets](framework-runtime.md), native
[inference request](input-and-execution.md)
and [native training index](../../cosmos3-nano-transfer-auto/post-training/references/data-contract.md#local-training-index).
Run commands from the matching Cosmos Framework checkout.

The [shared example scene](reference-scene.md) provides an HF download,
checksum instructions, the local layout and a shared training index. Its data
revision is separate from the compatible model snapshots required below.

| Task               | Required components                                                                                                       |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| Generation         | Complete model snapshot/revision, matching config/tokenizers, prepared request and synchronized controls/captions         |
| Direct fine-tuning | Released Transfer-Auto weights, paired customer data in the native local bundle, fresh training state and chosen schedule |
| Full curriculum    | Cosmos3-Nano initializer, matched tokenizers, customer bundle covering each phase's clip lengths, and four-phase handoffs |

If an asset is absent or conflicts with the selected model, report the precise
gap. Do not substitute a generic video dataset or infer camera/LiDAR compatibility
from matching filenames or tensor shapes.

## Handoffs

| From → to                             | Hand off                                                                                                           | Check                                                                     |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------- |
| Data/captions → training or inference | Camera/control/target mapping, synchronized interval, renderer settings, scenario properties and separate captions | Inspect pixels, actual timestamps and the text consumed by the loader     |
| Training → inference                  | Weights, matching model config/tokenizers, weight variant and resolved training settings                           | Complete compatible load, all requested camera outputs and optional LiDAR |
| Inference → troubleshooting           | Input request, resolved settings, outputs and one concrete failing camera/time                                     | Check camera mapping, controls, captions and the affected output interval |

Keep original inputs when preparing variants. Keep confidential media local
unless the user explicitly requests an external workflow.

## Sim-Auto

> **Skill note (not in the source):** this section is "Preserve these model settings" of the Sim-Auto model contract (cosmos-framework `main` 45fecab), moved here word for word with its heading one level down. The rest of that contract (checkpoint handoffs, trainable parameters) is in the [Sim-Auto model and checkpoint contract](../../cosmos3-nano-sim-auto/shared/model-contract.md).

### Preserve these model settings

- Joint camera/LiDAR architecture: 11 camera identities plus one LiDAR identity,
  LiDAR patch size 1, and camera-local text.
- Causal history: three RGB latent frames per chunk, strict-past clean RGB for
  controls, chunk-causal clean passes and a 0.4-second cross-view history window
  with overlapping keys counted once.
- RGB compression: temporal 4, spatial 16×16. LiDAR: temporal 1, spatial 16×16,
  128 latent channels. Use the tokenizer config and weights delivered together.
- Camera/LiDAR clocks: 30/10 Hz, synchronized by source-frame IDs. Camera-major
  storage is not a spatial mosaic. See the [data contract](../../cosmos3-nano-sim-auto/shared/data-contract.md).
- Fixed SDE levels: `[1.0, 0.9375, 0.8333333333333334, 0.625]`, then zero.
  Generic RF step-count/shift options do not replace this schedule.
- The documented streaming export stores `rolling_kv_cache_chunks=2`,
  `rolling_kv_sink_chunks=1` and `rolling_prompt_policy=segmented`.
  Preserve these in the inference artifact.

The model uses learned physical sensor IDs; it does not infer camera identity
from a filename or caption. Follow the shared
[input-preparation](input-preparation.md)
and [caption guide](caption-guide.md).
