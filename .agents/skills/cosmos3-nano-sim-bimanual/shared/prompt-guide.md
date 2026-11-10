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

## Prompts of the AgiBot held-out evaluation

The [Sim-Bimanual AgiBot recipe](../../cosmos3-sim-bimanual-agibot/SKILL.md) scores 20 held-out AgiBotWorld-Beta episodes
(10 of task 600, 10 of task 596). Policy, decided before any generation: each prompt describes the starting scene
(objects, surfaces, camera), the two arms and the task statement; it never narrates what happens later. Each was written
from the episode's first frame and the dataset's task text, and is held fixed for the matched-action and the
action-swap rollouts (the swap keeps the frame episode's prompt). The same text is in
`cosmos3-sim-bimanual-agibot/assets/sim-bimanual/manifests/prompts_heldout_20.json`.

- `task600_ep815362`: A pale wood-grain shelf compartment with a beige wall behind it. A brown deer-shaped storage box with orange antlers and an orange open interior stands at the center back; a small beige tree-shaped hook is fixed to the upper-left wall. A small tag with a key, partly hidden, lies at the far left beside the left gripper. The camera is a fixed head-mounted view in landscape, sharp and evenly lit, looking down at the work surface; both black two-finger robotic arms enter from the left and right frame edges with open grippers. The task is to hang the key on the hook and place the bag in the storage box: the left arm picks up the key and hangs it on the wall hook, and the right arm picks up the bag and puts it into the storage box.
- `task600_ep817265`: A shelf compartment with a beige-pink wall. A small teal hook is fixed to the upper-left wall. In the lower center sits a round white woven-rope basket used as the storage box. A blue keychain with a robot-shaped charm and metal keys lies on the shelf between the left gripper and the basket, and a small black zippered pouch rests beside the right gripper at the right edge of the basket. The camera is a fixed head-mounted view in landscape, sharp and evenly lit, looking down at the work surface; both black two-finger robotic arms enter from the left and right frame edges with open grippers. The task is to hang the key on the hook and place the bag in the storage box: the left arm picks up the key and hangs it on the wall hook, and the right arm picks up the bag and puts it into the storage box.
- `task600_ep819929`: A pale shelf compartment with a grey-beige wall and a small teal hook on the upper left wall. A round white woven basket with three small pastel star ornaments sits in the lower center. A blue-and-white keychain lies just left of the basket near the left gripper, and a black zippered pouch lies to the right of the basket in front of the right arm. The camera is a fixed head-mounted view in landscape, sharp and evenly lit, looking down at the work surface; both black two-finger robotic arms enter from the left and right frame edges with open grippers. The task is to hang the key on the hook and place the bag in the storage box: the left arm picks up the key and hangs it on the wall hook, and the right arm picks up the bag and puts it into the storage box.
- `task600_ep821169`: A shelf compartment with a light grey wall, a visible left side wall and a small teal hook at the upper center. A round white woven basket with small star ornaments sits at the lower center; a blue keychain with a small white robot charm and silver keys lies on the shelf to its left, and a dark zippered pouch is held against the right arm at the right of the basket. The camera is a fixed head-mounted view in landscape, sharp and evenly lit, looking down at the work surface; both black two-finger robotic arms enter from the left and right frame edges with open grippers. The task is to hang the key on the hook and place the bag in the storage box: the left arm picks up the key and hangs it on the wall hook, and the right arm picks up the bag and puts it into the storage box.
- `task600_ep822703`: A wood-toned shelf compartment with a beige wall; a bundle of dried flowers wrapped in paper leans on the upper left, a small hook is on the upper wall, and a brown deer-shaped storage box with orange antlers and an orange interior stands at the center back. A red lanyard with keys lies in the middle of the shelf, and a low cabinet with two black knobs is at the bottom center. The camera is a fixed head-mounted view in landscape, sharp and evenly lit, looking down at the work surface; both black two-finger robotic arms enter from the left and right frame edges with open grippers. The task is to hang the key on the hook and place the bag in the storage box: the left arm picks up the key and hangs it on the wall hook, and the right arm picks up the bag and puts it into the storage box.
- `task600_ep824031`: A shelf compartment with a beige wall; a small tree-shaped hook is on the upper wall next to a red 2025 snake-year decoration. A large brown deer-shaped storage box with orange antlers and an orange interior stands at the center, a small potted plant sits behind the left gripper, a red-and-black key lanyard lies at the left, and a brown leather key holder lies at the lower right of the box. The camera is a fixed head-mounted view in landscape, sharp and evenly lit, looking down at the work surface; both black two-finger robotic arms enter from the left and right frame edges with open grippers. The task is to hang the key on the hook and place the bag in the storage box: the left arm picks up the key and hangs it on the wall hook, and the right arm picks up the bag and puts it into the storage box.
- `task600_ep825215`: A wooden-panel shelf compartment with a grey-brown wall. A small teal hook is on the upper left, and a red and gold lucky-character ornament hangs at the upper center. A round amber glass bowl with a gold deer figurine on top stands at the lower center; a bundle of keys with a yellow tag lies at the left, and a small black-and-gold pouch lies near the right gripper. The camera is a fixed head-mounted view in landscape, sharp and evenly lit, looking down at the work surface; both black two-finger robotic arms enter from the left and right frame edges with open grippers. The task is to hang the key on the hook and place the bag in the storage box: the left arm picks up the key and hangs it on the wall hook, and the right arm picks up the bag and puts it into the storage box.
- `task600_ep826290`: A shelf compartment with a beige wall; a small tree-shaped hook and a red 2025 snake-year decoration are on the upper wall. A brown deer-shaped storage box with orange antlers and an orange interior stands at the center, a small plant is at the left behind the left gripper, a red key lanyard with keys lies on the shelf to the left of the box, and a dark brown key holder lies at the right. The camera is a fixed head-mounted view in landscape, sharp and evenly lit, looking down at the work surface; both black two-finger robotic arms enter from the left and right frame edges with open grippers. The task is to hang the key on the hook and place the bag in the storage box: the left arm picks up the key and hangs it on the wall hook, and the right arm picks up the bag and puts it into the storage box.
- `task600_ep827325`: A grey countertop shelf with a grey-brown wall; a small teal hook and a red lion-dance ornament are on the upper wall. A white woven basket with small ornaments sits at the center, a blue-framed picture and a small vase of pink flowers stand at the left, and a small blue-and-white keychain lies on the surface near the left gripper; the right gripper rests near the basket. The camera is a fixed head-mounted view in landscape, sharp and evenly lit, looking down at the work surface; both black two-finger robotic arms enter from the left and right frame edges with open grippers. The task is to hang the key on the hook and place the bag in the storage box: the left arm picks up the key and hangs it on the wall hook, and the right arm picks up the bag and puts it into the storage box.
- `task600_ep836956`: A dark grey countertop in front of a grey wall with a small teal hook at the upper center and a red lion-dance ornament at the upper right. A leafy potted plant stands at the upper left, a white oval plate-like storage box is at the right of center, and a small yellow-strapped keychain with a charm lies on the counter in the middle; a blue cup sits near the right gripper. The camera is a fixed head-mounted view in landscape, sharp and evenly lit, looking down at the work surface; both black two-finger robotic arms enter from the left and right frame edges with open grippers. The task is to hang the key on the hook and place the bag in the storage box: the left arm picks up the key and hangs it on the wall hook, and the right arm picks up the bag and puts it into the storage box.
- `task596_ep810449`: A white-tiled kitchen corner with a black grid of grout lines. A cream-colored rice cooker with a red digital display and a silver lid button stands at the center, plugged into a wall socket behind it; a copper-colored inner pot containing white rice sits at the right of the cooker, and a grey wire rack of bottles and jars stands at the upper right. The camera is a fixed head-mounted view in landscape, sharp and evenly lit, looking down at the work surface; both black two-finger robotic arms enter from the left and right frame edges with open grippers. The task is to place the inner pot with rice into the rice cooker and cook: the right arm lifts the inner pot and sets it into the cooker, and the left arm presses the lid-opening button on top of the cooker to open its lid.
- `task596_ep811187`: A white-tiled kitchen corner with black grout lines. A cream-colored rice cooker with a red digital display stands at the center right, its black power cord running to a wall socket; a copper-colored inner pot with white rice sits at its right under the right gripper, and a grey wire shelf of sauces and bottles stands at the upper right. The camera is a fixed head-mounted view in landscape, sharp and evenly lit, looking down at the work surface; both black two-finger robotic arms enter from the left and right frame edges with open grippers. The task is to place the inner pot with rice into the rice cooker and cook: the right arm lifts the inner pot and sets it into the cooker, and the left arm presses the lid-opening button on top of the cooker to open its lid.
- `task596_ep812711`: A kitchen counter in front of a grey-veined white marble wall with a wall socket at the center. A cream-colored rice cooker with a red display stands at the center; a copper-colored inner pot is partly visible at the right beside it, a black grill pan sits at the left behind the left gripper, and a sponge lies on the sink edge at the right. The camera is a fixed head-mounted view in landscape, sharp and evenly lit, looking down at the work surface; both black two-finger robotic arms enter from the left and right frame edges with open grippers. The task is to place the inner pot with rice into the rice cooker and cook: the right arm lifts the inner pot and sets it into the cooker, and the left arm presses the lid-opening button on top of the cooker to open its lid.
- `task596_ep813228`: A kitchen counter in front of a grey-veined white marble wall with a wall socket at the upper center and a tap at the upper right. A cream-colored rice cooker with a red display stands at the center right; a copper-colored inner pot with white rice sits at its right under the right gripper, and a black grill pan lies at the left. The camera is a fixed head-mounted view in landscape, sharp and evenly lit, looking down at the work surface; both black two-finger robotic arms enter from the left and right frame edges with open grippers. The task is to place the inner pot with rice into the rice cooker and cook: the right arm lifts the inner pot and sets it into the cooker, and the left arm presses the lid-opening button on top of the cooker to open its lid.
- `task596_ep813610`: A kitchen counter in front of a grey-veined white marble wall with a wall socket, and a small sachet pack at the upper left. A cream-colored rice cooker with a red display stands at the center; a copper-colored inner pot filled with white rice sits at the right of the cooker, with a sink and a green scouring pad behind it at the right, and a black grill pan at the left. The camera is a fixed head-mounted view in landscape, sharp and evenly lit, looking down at the work surface; both black two-finger robotic arms enter from the left and right frame edges with open grippers. The task is to place the inner pot with rice into the rice cooker and cook: the right arm lifts the inner pot and sets it into the cooker, and the left arm presses the lid-opening button on top of the cooker to open its lid.
- `task596_ep813907`: A kitchen counter in front of a grey-veined white marble wall with a wall socket and a tap at the upper right. A cream-colored rice cooker with a red display stands at the center; a copper-colored inner pot of white rice sits at its right, a small white container with a green scouring pad is on the sink behind it, and a black grill pan lies at the left. The camera is a fixed head-mounted view in landscape, sharp and evenly lit, looking down at the work surface; both black two-finger robotic arms enter from the left and right frame edges with open grippers. The task is to place the inner pot with rice into the rice cooker and cook: the right arm lifts the inner pot and sets it into the cooker, and the left arm presses the lid-opening button on top of the cooker to open its lid.
- `task596_ep814265`: A bright kitchen counter in front of a grey-veined white marble wall with a wall socket, a bottle of foaming soda soap and a cloth beside a sink at the upper right, and a paper bag at the upper left. A cream-colored rice cooker with a red display stands at the left center, and a copper-colored inner pot filled with white rice sits at the right of the cooker. The camera is a fixed head-mounted view in landscape, sharp and evenly lit, looking down at the work surface; both black two-finger robotic arms enter from the left and right frame edges with open grippers. The task is to place the inner pot with rice into the rice cooker and cook: the right arm lifts the inner pot and sets it into the cooker, and the left arm presses the lid-opening button on top of the cooker to open its lid.
- `task596_ep815134`: A wooden countertop in front of a plain grey wall. A pale green electric kettle stands at the upper left behind the left gripper and a small pack of grain sits at the upper center. A cream-colored rice cooker with a red display stands at the center, and a copper-colored inner pot with a thin layer of brown rice-water sits to its right under the right gripper. The camera is a fixed head-mounted view in landscape, sharp and evenly lit, looking down at the work surface; both black two-finger robotic arms enter from the left and right frame edges with open grippers. The task is to place the inner pot with rice into the rice cooker and cook: the right arm lifts the inner pot and sets it into the cooker, and the left arm presses the lid-opening button on top of the cooker to open its lid.
- `task596_ep815473`: A wooden countertop in front of a plain grey wall. A pale green electric kettle stands at the far left, a small steel sink corner is at the right, a cream-colored rice cooker with a red display stands at the center left, and a copper-colored inner pot with a thin layer of brown rice-water sits at its right in front of the right gripper. The camera is a fixed head-mounted view in landscape, sharp and evenly lit, looking down at the work surface; both black two-finger robotic arms enter from the left and right frame edges with open grippers. The task is to place the inner pot with rice into the rice cooker and cook: the right arm lifts the inner pot and sets it into the cooker, and the left arm presses the lid-opening button on top of the cooker to open its lid.
- `task596_ep815752`: A wooden countertop in front of a plain grey wall. A pale green electric kettle stands at the upper left beside a small white bowl, a cream-colored rice cooker with a red display stands at the center right, and a copper-colored inner pot sits at the far right next to the right gripper. The camera is a fixed head-mounted view in landscape, sharp and evenly lit, looking down at the work surface; both black two-finger robotic arms enter from the left and right frame edges with open grippers. The task is to place the inner pot with rice into the rice cooker and cook: the right arm lifts the inner pot and sets it into the cooker, and the left arm presses the lid-opening button on top of the cooker to open its lid.
