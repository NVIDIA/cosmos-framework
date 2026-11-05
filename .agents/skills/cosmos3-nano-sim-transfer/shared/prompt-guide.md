# Write prompts for spatial transfer

Use the [Hugging Face `example2` assets](hf-sample.md) for the complete captioning,
data preparation, Stage 0–3 training and four-modality inference workflow.
Use `example1` for the long-rollout walkthrough. Keep each prompt paired with
its own videos.

Describe the RGB video you want to generate: subjects, appearance, setting,
lighting, camera and actions over the selected interval. Keep object locations,
motion and camera movement consistent with the control. The same description
applies to all four `example2` controls; native training and inference derive
edge and blur from its source RGB.

## Use the supplied prompts

Each HF example contains `<example>_prompt.txt` and
`<example>_upsampled_prompt.json` beside its RGB, depth and segmentation videos.
The two JSON files have different outer structures:

| Example    | JSON format                                           | Generation prompt                          | Training caption                                                                           |
| ---------- | ----------------------------------------------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------ |
| `example2` | Direct structured caption object                      | Serialize the entire object as prompt text | The object without saved output metadata; its `temporal_caption` supplies the dense branch |
| `example1` | Envelope containing `prompt` and `structured_caption` | Use the envelope's `prompt` string         | Long-rollout example                                                                       |

The `.txt` files preserve the source prompts. For the 93:7 structured/dense
training mixture, extract `example2_upsampled_prompt.json.temporal_caption`;
do not substitute `example2_prompt.txt`.

`example2` describes a barefoot woman in a black dress dancing on a pale
platform in an outdoor timber yard, holding a blue phone connected to white
wired earbuds. Its structured annotation describes swaying, small bouncing
steps, gestures and a forward lean over `0:00`–`0:04`. For example:

> A barefoot woman wearing a fitted black dress with short cap sleeves, a
> rounded neckline, and gathered fabric gathered above the knees, dancing on
> a pale platform in an outdoor timber yard.

`example1` describes a light grey robot arm with yellow covers and a dark
rubber hand, a lavender toy unicorn head, mustard cabinets and an emerald
countertop. Its saved prompt describes appearance without a timeline. Read
the `prompt` field rather than passing the complete envelope as model text.

Use the saved prompts directly; a caption-service call is optional.

## Keep metadata consistent with the request

`example2` has **121 frames at 30 FPS and 1920 × 1080 pixels**. Frames 0–120
span 4 seconds; frame-count/FPS duration is about 4.033 seconds. Its saved JSON
records `duration: "4s"`, `fps: 30` and 832 × 480 output resolution. The
480-tier canvas is **832 × 480**, with about **2.56%** direct-resize aspect
distortion, within the training limit.

`example1` has **600 frames at 30 FPS and 384 × 384 pixels**. Its square
480-tier output is **640 × 640**. Native causal inference uses a 601-frame
processing window with `max_frames: 600`, preserving all 600 source frames.
The processing window does not add a source frame or trim the clip. Use this
case in the [long-rollout walkthrough](../inference/SKILL.md).

Inference supplies effective output metadata through the Transfer formatter.
Training removes saved output metadata before appending sampled values.
Preserve downloaded prompt files unchanged, and perform that transformation
in the locally prepared training data. Changing an FPS label does not
resample control frames.

## Prepare a prompt for your own clip

Write or review a dense description of the selected RGB interval first.
Expand it into the structured schema when reproducing the structured-caption
recipe. Review added subjects, materials, actions and camera details against
the RGB before accepting a training annotation. Keep the structured object
and its `temporal_caption` together. Use the Framework video JSON template
and schema when authoring structured prompts:

```text
$FRAMEWORK_DIR/cosmos_framework/inference/prompting_templates/external_api/t2v_i2v_video_prompt.txt
$FRAMEWORK_DIR/cosmos_framework/inference/prompting_templates/external_api/t2v_i2v_video_json_schema.json
```

For inference appearance edits, change desired colors, materials, lighting or
environment while preserving the spatial intent of the control. Start with
one edit. Use [caption preparation](../post-training/references/caption-guide.md) for new
training targets and the [data contract](data-contract.md) for alignment and
resizing.

Local preparation uses exclusive bounds `[0, decoded frame count)` and the
structured object's `temporal_caption` for 7% of retained-text draws. The
native plain-text builder instead uses 100% dense captions.
