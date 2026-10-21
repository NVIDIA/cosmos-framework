# Inference contract

## Cosmos Framework

Follow the [inference Skill](../SKILL.md) and its source,
[Framework Bimanual inference](../../../../../docs/inference_bimanual.md).
They provide the native install, HF download, input checker, generation command
and output checks. Use the current default revision of
`nvidia/Cosmos3-Nano-Sim-Bimanual` and run the checkpoint metadata preflight in
the inference Skill before loading weights.
Use `tools/check_bimanual_mixed_inputs.py` to validate the manifests and assets
in the downloaded Hugging Face snapshot before loading weights.

The schema-5 HF example and canonical user-trained Stage 3 both use four steps,
but each export owns its exact saved ladder. Do not infer one from the other.
Inference-only installs use `COSMOS_TRAINING=0`. An HF export contains its selected
weights; DCP EMA-selection flags do not select another HF weight variant.

The schema-5 action semantics below apply to both prepared user inputs and
the matched Framework example.

## Input manifest

Each JSONL row requires:

| Field                   | Requirement                                                                               |
| ----------------------- | ----------------------------------------------------------------------------------------- |
| `name`                  | Unique path-safe sample name                                                              |
| `model_mode`            | `forward_dynamics`                                                                        |
| `prompt`                | One non-empty plain-text value following the [prompt guide](../../shared/prompt-guide.md) |
| `vision_path`           | RGB media locator readable by the runtime                                                 |
| `action_path`           | Action JSON path or locator readable by the runtime                                       |
| `action_condition`      | `true`                                                                                    |
| `raw_action_dim`        | `59`                                                                                      |
| `domain_name`           | Domain supported by the selected checkpoint                                               |
| `resolution`            | Explicit and supported by the checkpoint                                                  |
| `fps`                   | Explicit and matched to action timing                                                     |
| `num_frames`            | Explicit output-frame count                                                               |
| `action_chunk_size`     | Action-row count, equal to `num_frames - 1`                                               |
| `seed`                  | Explicit per-sample seed                                                                  |
| `format_prompt_as_json` | `false` for the product caption path                                                      |

Relative media and action paths resolve from the manifest's directory. The
Hugging Face model repository's default branch supplies the current combined
unified-schema and focused mixed-action manifests enumerated in the
[inference Skill](../SKILL.md#hugging-face-hosted-examples). The checkpoint
revision selects model weights and configuration; it does not pin the hosted
examples. `unifiedv1` identifies the normalized 59D schema; its combined
manifest may contain valid per-row robot/camera/robot routing. The focused
mixed-action manifest is an overlapping selection, not a mutually exclusive
action schema. Follow the
[native input preparation](../SKILL.md#2-prepare-the-data-contract-and-one-input-row)
for user-authored rows. Download the published manifests without a revision as
shown in the inference Skill, then run
`tools/check_bimanual_mixed_inputs.py --checkpoint-path "$CHECKPOINT_DIR"`.
Their Hugging Face asset URLs and remote user inputs are materialized through
the inference download hook. For gated files, export `HF_TOKEN` into
the runtime process and explicitly pass that variable by name into Pyxis; a
token stored only under a custom `HF_HOME` does not authenticate direct URL
downloads.
Download user-owned remote inputs with the appropriate provider credentials
before creating a self-contained local manifest.

## Structured actions

The inference interface consumes a prepared action sidecar. Follow the
[canonical data contract](../../post-training/references/data-contract.md)
for action mapping, coordinate frames, masks, normalization, timing, and Domain
2 conversion. The inference entry point does not derive these values from raw
controls.

The action file has this shape:

```json
{
  "action_space": "normalized_unified_v1",
  "action": [[0.0]],
  "domain_names": ["<supported-domain>"]
}
```

After following the canonical data contract, replace the illustrative row with
59 finite values per action row. Values are already mapped, masked, and
normalized. Do not apply another normalization.
`domain_names` contains one value for the sequence or one value per action row.
When `domain_names` is omitted, the manifest's `domain_name` supplies the
sequence-level fallback. Each supplied value must name a domain supported by
the checkpoint. The mixed AgiBot examples route `agibotworld` and
`camera_pose` per row, supporting the Domain 2 conversion defined by the
canonical data contract. Every `camera_pose` row must zero robot-only action
slots 9 through 58.
The public `normalized_unified_v1` interface is always 59-dimensional. A user
adapter must perform its semantic mapping before inference. Do not pass another
native-width representation as `normalized_unified_v1`; the mapped 59 values
are padded to the model width of 64.

Require:

```text
action_rows = action_chunk_size = output_frames - 1
latent_frames = 1 + (output_frames - 1) / 4
```

Choose the output length for your trajectory and make the action-row count
positive and divisible by four. The published example length is not a fixed
inference requirement. The native command reads these fields from each row.
Reject an off-by-one or nonintegral request rather than padding or trimming.

## Checkpoint and sampling

Match the command to the export: the current schema-5 model uses its saved
four-step SDE schedule. The canonical user-trained Stage 3 student also uses four steps,
but may own a different exact ladder. Keep the input representation, prompt
mode, and saved schedule matched to the selected export, as shown in the
[checkpoint table](../SKILL.md#1-choose-the-checkpoint-path).

For your own Stage 3 student, export from its saved training config and record
the selected regular or EMA variant before using the
[native inference command](../SKILL.md#use-your-own-stage-3-checkpoint).
Keep the complete HF export together; its metadata selects the model
architecture and saved sampler.

## Native loading and placement checks

Use `cosmos_framework.scripts.inference` with a complete local HF snapshot or
student export, as shown in the [main command](../SKILL.md#5-generate-video).
Keep `COSMOS_TRAINING=0` for an inference-only environment. The HF export already
contains its selected regular or EMA weights; a DCP EMA flag does not select
another variant inside that export.

For distributed inference, the launched process count and resolved model world
size must agree. With `dp_shard_size=1` and `cfgp_size=1`,
`dp_replicate_size` must divide the launched process count and `cp_size` must
be a supported divisor. Recheck memory and one output before scaling placement.

## Output verification

For each row, fully decode the output with the bundled
`scripts/validate_output_video.py`. Supply the expected saved frame count, FPS,
width, and height. The validator requires one
video stream, exact geometry and timing, spatial and temporal change, and emits
the SHA256 digest. Source geometry, the padded model canvas, and saved decoded
geometry are separate: spatial-compression alignment can make a 640x360 source
save as 640x352. Do not validate against source dimensions when the resolved
saved canvas differs.

Record status, output path and digest, media properties, checkpoint and weight
variant, generation settings, seed, and any failure. Compare latency only when
placement, compile mode, frame count, sampler, and cache policy match.
