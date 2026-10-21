# Bimanual inference

Generate a rollout from an initial image and recorded AgiBot/camera
actions using `nvidia/Cosmos3-Nano-Sim-Bimanual`. This is open-loop generation;
the example does not run a policy or feed generated observations back to one.

For live controls and decoded observation feedback, use the
[single-process session API](bimanual_sessions.md).

## Model and runtime

Use the current default Hugging Face revision with the latest Framework `main`.

| Component                                                                     | Revision                                                           |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| [Hugging Face model](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Bimanual) | Current default revision                                           |
| [Framework runtime](https://github.com/NVIDIA/cosmos-framework/tree/main)     | Latest `main`                                                      |
| Action format                                                                 | Schema 5, `normalized_unified_v1`, 59 values padded to 64          |
| Canonical sampling example                                                    | Four-step SDE schedule: `[1.0, 0.9375, 0.8333333333333334, 0.625]` |
| Default example                                                               | 900 actions → 901 frames, 30 FPS; length is user-configurable      |

Inference validates the downloaded checkpoint metadata before loading model
weights. Keep its schema-5 sidecars, plain-text prompts, and four-step schedule
together; changing an action array's width does not convert another schema.

## Install the Framework

Use a dedicated checkout and follow [the setup prerequisites](setup.md), including
Python 3.13, Git LFS, a CUDA-compatible driver, and FFmpeg shared libraries.
Run from the Framework repository root:

```bash
git fetch origin main
git checkout --detach origin/main
git lfs install --local
uvx --from 'uv==0.11.29' uv lock --check
uvx --from 'uv==0.11.29' uv sync --locked --group cu130 --extra guardrail
source .venv/bin/activate
# This install has no training extras; keep training-only imports off for every step below.
export COSMOS_TRAINING=0
```

The root lockfile is generated and checked with `uv==0.11.29`; use that exact
resolver instead of regenerating the lock with another version. `uv.lock` in
the checkout fixes the dependency versions. Keep the FFmpeg library search path
required by your installation; the Framework discovers wheel-provided NPP
libraries automatically.

## Download the checkpoint

Authenticate to Hugging Face with read access to the model and download the
complete current snapshot into a new directory:

```bash
CHECKPOINT_DIR="$PWD/checkpoints/Cosmos3-Nano-Sim-Bimanual"
hf download nvidia/Cosmos3-Nano-Sim-Bimanual \
  --local-dir "$CHECKPOINT_DIR"
python tools/check_bimanual_mixed_inputs.py --checkpoint-path "$CHECKPOINT_DIR"
```

The check reads the export's current manifests and downloaded assets without
loading model weights or requiring a GPU. Preserve the downloaded `config.json`,
transformer config, action metadata, and weights together.

## Input contract

The model repository publishes `sim_bimanual_unifiedv1_hf.jsonl` and
`sim_bimanual_mixed_action_hf.jsonl` under `examples/inference/`. Each row points
to its current Hugging Face image and action sidecar. The sidecar contains:

- `action_space: "normalized_unified_v1"` and an `action` array.
- Either a scalar `domain_names` string for a single-domain rollout, or one
  list entry per action row. A mixed example has 300 `agibotworld`, 300
  `camera_pose`, then 300 `agibotworld`; the checkpoint maps these to domain
  IDs 15, 2, and 15.
- Values already normalized and masked by the source; inference only appends
  five zero channels. Do not normalize these values again.

Every sidecar row is checked against its declared domain's canonical 59D
validity mask. Slots without meaning for that domain must be exactly zero. This
validation applies to uniform sidecars as well as mixed-domain sidecars; mixed
inputs additionally exercise domain transitions.

Prefer a raw, unpadded initial image for new examples. If a transport asset
already contains a surrounding canvas, declare its real pixel rectangle with
`content_crop_box: [left, top, right, bottom]`. The crop is explicit because
inference cannot infer whether an 832×480 file is padded or genuinely contains
832×480 of image content.

The prompt is plain text (`format_prompt_as_json: false`). For a mixed row, the
model continues one rollout across both domain changes. The action count must equal
`action_chunk_size` and `num_frames - 1`; it must be divisible by the checkpoint's
temporal compression factor, four.

Set `action_chunk_size`, `num_frames` and `fps` in each manifest row. Longer
trajectories use the same inference path; the default example length is not a
limit. Additional YAM inputs are available in the
[Hugging Face examples](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Bimanual/tree/main/examples/inference).
Switch the input manifest to try their matching image, prompt, actions and
domain routing without editing Framework source.

## Run on one GPU

From the same activated environment, select the AgiBot 001 row for the worked
example, using `CHECKPOINT_DIR` above:

```bash
python - "$CHECKPOINT_DIR/examples/inference/sim_bimanual_unifiedv1_hf.jsonl" \
  "$CHECKPOINT_DIR/agibot_example.jsonl" <<'PY'
import json
import sys
from pathlib import Path

rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
matches = [row for row in rows if row["action_path"].endswith("/agibot_001_normalized_unifiedv1.json")]
if len(matches) != 1:
    raise ValueError("Expected exactly one published AgiBot 001 example")
Path(sys.argv[2]).write_text(json.dumps(matches[0]) + "\n")
PY
```

```bash
INPUT_MANIFEST="${INPUT_MANIFEST:-$CHECKPOINT_DIR/agibot_example.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/outputs/bimanual_hf_examples}"
export COSMOS_TRAINING=0
torchrun --standalone --nproc-per-node=1 \
  -m cosmos_framework.scripts.inference \
  -i "$INPUT_MANIFEST" \
  -o "$OUTPUT_DIR" \
  --checkpoint-path "$CHECKPOINT_DIR" \
  --model-mode=FORWARD_DYNAMICS --resolution=480 --image-size=480 \
  --num-steps=4 --seed=2026 \
  --kv-cache-inference-size=64 --attention-sink-size=8 \
  --autoregressive=True --parallelism-preset=latency --max-num-seqs=1 \
  --no-use-torch-compile --no-diffusion-cache --no-guardrails \
  --video-save-quality=4
```

This example disables guardrails. Apply the deployment's guardrail policy in
applications. Keep the seed and sampling settings fixed when comparing outputs.

For a multi-row manifest, add `--keep-going` to isolate invalid inputs during
batch data preparation. A validation or file-read failure is recorded in that sample's
`sample_outputs.json` with `status: "error"`, its resolved arguments and the
error details; the remaining valid rows still run. Inspect every receipt:
completion with `--keep-going` does not mean every row succeeded. Without
this option, inference remains fail-fast. Runtime/CUDA failures during input
preparation still abort the run.

`image_size=480` selects a preprocessing tier; it does not force arbitrary
inputs to a 480-pixel height. Smaller inputs retain the native
no-upscale/padding behavior, followed by latent cropping to spatially aligned
dimensions. The published AgiBot 001 and mixed-action 028 examples each use a
640×360 initial image. Each is reflection-padded at the bottom and right to the
832×480 canvas, the top-left 22×40 latents are kept, and the rollout is 640×352.
Only the initial image conditions the rollout. The manifest URLs intentionally
use `/resolve/main/`; Framework materializes them through the standard download
hook. Keep `HF_TOKEN` available when the gated repository requires authentication.

The worked example uses the AgiBot 001 unified-action row and AgiBot
mixed-action 028 row, identified by action-sidecar basenames
`agibot_001_normalized_unifiedv1.json` and
`agibot_mixed_028_normalized_mixed_action.json`.

## Run the 60-second AgiBot example

The model repository also publishes a longer open-loop trajectory. Set:

```bash
INPUT_MANIFEST="$CHECKPOINT_DIR/examples/inference/sim_bimanual_agibot_long60_hf.jsonl"
OUTPUT_DIR="$PWD/outputs/bimanual_agibot_long60"
```

Rerun the single-GPU command with a fresh output directory. The manifest
selects the full action trajectory and its matching `action_chunk_size`,
`num_frames` and `fps`; no source changes or fixed-length CLI overrides are
needed. Match the output to those fields and inspect the early, middle and
late portions for drift, flicker, identity changes and action response.

## Optional frustum history retrieval

Frustum retrieval reuses the same AgiBot video and unified action sidecar. Add
`kv_cache_frustum_path` to the input row; the referenced JSON may provide
explicit OpenCV `c2w` matrices or derive them from the sidecar:

```json
{
  "pose_source": "bimanual_unified_v1_chunk16",
  "image_size": [832, 480],
  "fps": 30,
  "horizontal_fov_degrees": 90.0,
  "near": 0.1,
  "far": 10.0,
  "sink_size": 8,
  "retrieved_size": 32,
  "recent_size": 32
}
```

`image_size` is `[width, height]`. Supply either `intrinsics` or
`horizontal_fov_degrees`, not both; measured calibration is preferred. The
action-derived pose source accepts AgiBot/camera-domain unified-v1 sidecars and
reconstructs the complete trajectory across domain boundaries.

Run frustum retrieval in eager mode with an unbounded physical archive by
adding:

```bash
--no-use-torch-compile --kv-cache-inference-size=0 --attention-sink-size=0
```

Omitting `kv_cache_frustum_path` preserves ordinary inference.

## Verify the result

Check `sample_outputs.json` for success, then fully decode each MP4 and match
its frame count and FPS to the selected row's `num_frames` and `fps`, with the
geometry expected for that input. For a mixed-action row, inspect all phases
and domain transitions at the times defined by its action rows and FPS for
discontinuities, repeated frames, frozen output,
and disagreement with the supplied actions.
