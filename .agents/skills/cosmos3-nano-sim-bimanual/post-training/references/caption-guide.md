# Caption preparation guide

Prepare one reviewed plain-text caption for each selected training window. The
caption, RGB observations, actions, timestamps, embodiment, and calibration
must describe the same interval.

## Select the window before captioning

Assign stable episode and window identifiers first. Record the inclusive RGB
frame indices, source frame rate, action timestamps, and any resampling rule.
Keep every window from one episode in exactly one split. Do not caption a whole
recording and attach the text to an unrelated shorter window.

## Write and review the caption

Include the task, visible starting scene, manipulated objects, relevant arm and
gripper state, camera, and visible appearance. Express any necessary chronology
within one continuous plain-text caption, and set `format_prompt_as_json=false`
in product inference manifests. Keep resolution in the manifest metadata. Do
not append it to the caption.

Review the caption beside the first, middle, and last selected frames. Check:

- object identity, count, color, material, and location.
- left/right arm assignment and which gripper contacts each object.
- chronological motion and whether it belongs to the selected interval.
- camera motion, framing, and occlusion.
- lighting and scene details that remain visible.
- absence of invented text, hidden state, intent, or events.

If held-out evaluation is intended to condition only on the task and starting
scene, remove reference-future details from both its training-compatible prompt
policy and its evaluation prompts. A future-described prompt is a different
conditioning task and must be labeled separately.

## Optional automated captioning

An automated captioner may draft text when the user supplies and authorizes
one. Record the captioner and immutable revision, prompt template, sampling
settings, source digest, selected interval, raw draft, reviewed caption, and
reviewer decision. Do not send confidential data to an external service unless
the user explicitly authorizes that transfer.

Automated output is a draft. Reject or correct a draft that changes handedness,
object identity, temporal order, camera behavior, or the selected interval.
Prompt expansion remains optional. See the [prompt guide](../../shared/prompt-guide.md).

## Handoff record

For each window, provide the user's data adapter with the same fields as this
illustrative example. Its frame numbers do not prescribe a training length:

```json
{
  "episode_id": "episode-0001",
  "window_id": "episode-0001-000000-000900",
  "start_frame": 0,
  "end_frame": 900,
  "fps": 30.0,
  "caption": "Two robot arms manipulate an object on a fixed work surface.",
  "caption_policy": "plain_text_reviewed_v1"
}
```

This JSON illustrates the caption handoff. It is not a replacement training
loader. The canonical loader adapter must join it to the matching RGB and
action window and emit the model-facing fields in the [data contract](data-contract.md).
Store the real paths in the user's source configuration, not in this guide.

Before training, fail any sample with an empty caption, ambiguous interval,
missing review status, or mismatch between the caption and selected frames.
Keep captions deterministic for validation and test.
