# Prompt guide

Cosmos3 Nano Sim Bimanual consumes one plain-text `prompt` value with
`format_prompt_as_json=false`. The prompt describes the task, visible scene,
camera, relevant objects, and intended interaction. The action trajectory
provides the robot motion conditioning. Keep resolution in the manifest's
`resolution` field. Do not append a resolution phrase to the prompt.

## Start from a published example

After downloading the Hugging Face snapshot, inspect a matched schema-5 example
without printing media or storage locators:

```bash
manifest="$CHECKPOINT_DIR/examples/inference/sim_bimanual_unifiedv1_hf.jsonl"
head -n 1 "$manifest" | python -c \
  'import json,sys; row=json.loads(sys.stdin.readline()); print(row["name"]); print(row["prompt"])'
```

Use the checkpoint and input pairing in the
[inference tutorial](../inference/SKILL.md).

Treat the selected example's prompt as one text value. For user-authored data,
write one continuous description and keep that convention unchanged between
training and evaluation. Do not wrap it in JSON, split it into labeled prompt
sections, or concatenate a resolution template.

## Write a prompt for inference

Describe only details that matter for generation:

- the manipulation task or intended interaction.
- the visible starting scene, including the objects being manipulated.
- the two robot arms and end effectors when they are visible.
- camera placement and whether it moves.
- lighting or appearance details needed to preserve the scene.
- the intended interaction in chronological prose when timing matters.

Keep object identity, handedness, spatial relationships, and task wording
consistent with the RGB condition and action trajectory. Avoid unsupported
claims, contradictory motion, invisible objects, and instructions for the
model to choose actions. The product predicts visual consequences of supplied
actions. It is not an action policy.

A concise user-authored example is:

> Two robot arms face a work surface with a blue cup near the center. The left
> gripper steadies the cup while the right gripper approaches it. The camera is
> fixed above the workspace under neutral indoor lighting.

This example illustrates wording only. Replace it with a description of the
user's actual input and task.

## Optional prompt expansion

Prompt expansion is optional. The bundle does not require or select an external
upsampling service. If the user supplies one, save the original prompt, tool and
revision, expanded prompt, and review result. Review the expansion against the
first, middle, and last relevant RGB frames. Remove invented identities,
materials, object states, arm assignments, camera motion, and events outside
the selected interval.

Keep the final result as one plain-text value without a resolution suffix. Do
not send confidential video or text to an external service without the user's
authorization. Do not treat an expanded prompt as ground truth merely because
it is longer.

## Evaluation boundary

For held-out forward-dynamics evaluation, decide before generation whether the
task permits descriptions of future events. A prompt that states the reference
future supplies additional conditioning and is not comparable with an
initial-scene-only prompt. Keep this policy fixed across checkpoints and save
the exact prompt with every output.

For training captions, continue with the
[caption guide](../post-training/references/caption-guide.md).
