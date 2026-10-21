---
name: cosmos3-nano-sim-bimanual-inference
description: Generate action-conditioned video with Cosmos3 Nano Sim Bimanual in Cosmos Framework. Download the matched release, prepare RGB and action inputs, run inference, inspect outputs, or use an exported student from the user's Stage 3 training.
---

# Run inference with Cosmos3-Nano-Sim-Bimanual

Generate a visual future from a task description, an initial RGB observation
and a time-aligned bimanual action trajectory. The model predicts video; it
does not choose robot actions. Rollout length, frame rate and saved geometry
follow the selected input.

## Choose the task

- For generation with the released model, follow the steps below.
- For your own completed training chain, use
  [your exported Stage 3 student](#use-your-own-stage-3-checkpoint).
- For requested held-out PSNR evaluation, use the
  [evaluation protocol](references/evaluation.md). Generation alone does not start scoring.

This workflow follows the [Bimanual inference tutorial](../../../../docs/inference_bimanual.md).
Use the current Framework `main` branch with the current default revision of
`nvidia/Cosmos3-Nano-Sim-Bimanual`.

## Prerequisites and environment setup

Use a dedicated Framework checkout with Python 3.13, Git LFS, FFmpeg shared
libraries and a CUDA-compatible GPU allocation. Set `FRAMEWORK_DIR` to that
checkout. If you already use its training environment, reuse it. For an
inference-only environment, follow [setup](../../../../docs/setup.md) and run:

```bash
cd "$FRAMEWORK_DIR"
git lfs install --local
export UV_CACHE_DIR="${UV_CACHE_DIR:-$FRAMEWORK_DIR/.uv-cache}"
mkdir -p "$UV_CACHE_DIR"
uvx --from 'uv==0.11.29' uv lock --check
uvx --from 'uv==0.11.29' uv sync --locked --group cu130 --extra guardrail
source .venv/bin/activate
export COSMOS_INTERNAL=0 COSMOS_TRAINING=0
python -m cosmos_framework.scripts.inference --help
```

Use the exact `uv` version shown above: it is the resolver used to generate and
check this Framework revision's lockfile. Do not regenerate `uv.lock` with a
different `uv` version during setup. If the documented `uv lock --check` fails
in a clean checkout at the selected Framework commit, stop and report the checkout
commit and lock-check output. Keep `UV_CACHE_DIR` on a writable workspace or
project filesystem when the home directory has a small quota.

Use the CUDA group matching your driver. Keep the FFmpeg library search path
required by your installation; Framework discovers wheel-provided NPP libraries.
Keep `COSMOS_TRAINING=0` for every inference-only step so imports do not require
training extras. The one-process example below is not a GPU-memory minimum.

### Authenticate hosted files without exposing the token

The checkpoint and current hosted examples may require Hugging Face read
access. Supply a user-owned token file outside the checkout and export its
contents without printing them:

```bash
: "${HF_TOKEN_FILE:?set HF_TOKEN_FILE to a readable token file outside the checkout}"
test -r "$HF_TOKEN_FILE"
set +x
HF_TOKEN="$(<"$HF_TOKEN_FILE")"
export HF_TOKEN
test -n "$HF_TOKEN"
```

Do not print the token, put its value in a command argument, copy it into the
checkout, or save it in a manifest or log. Framework's direct Hugging Face URL
downloader reads `HF_TOKEN`; a token available only under a custom `HF_HOME`
does not authenticate that downloader.

For Slurm with Pyxis, export the token in the submission shell as above, then
add both of these name-only options to the existing `srun` command:

```bash
--export=ALL,HF_TOKEN
--container-env=HF_TOKEN
```

Do not use `--container-env=HF_TOKEN=<value>`. Before loading the checkpoint or
hosted assets, verify inside the container without displaying the value:

```bash
test -n "${HF_TOKEN:-}" || { echo "HF_TOKEN is not available inside the container" >&2; exit 1; }
```

Unset the submission-shell value after the synchronous run or after the
scheduler has accepted the job: `unset HF_TOKEN`.

## 1. Choose the checkpoint path

The default example uses the current schema-5, four-step checkpoint. Keep its
complete configuration and weights together:

```bash
CHECKPOINT_DIR="$PWD/checkpoints/Cosmos3-Nano-Sim-Bimanual"

hf download nvidia/Cosmos3-Nano-Sim-Bimanual \
  --local-dir "$CHECKPOINT_DIR"
```

Authenticate to HF with model read access if required. Inspect the downloaded
metadata before loading weights; require schema 5 and the exact four-step
schedule:

```bash
python - "$CHECKPOINT_DIR/checkpoint.json" <<'PY'
import json
import sys
from pathlib import Path

metadata = json.loads(Path(sys.argv[1]).read_text())["cosmos3_nano_sim_bimanual"]
assert metadata["conditioning"]["schema_version"] == 5
assert metadata["conditioning"]["input_contract"]["action_dim"] == 59
assert metadata["fixed_step_sampler_config"]["t_list"] == [1.0, 0.9375, 0.8333333333333334, 0.625]
print("Bimanual checkpoint metadata passed")
PY
```

| Checkpoint                            | Input and prompting                                         | Saved sampling schedule |
| ------------------------------------- | ----------------------------------------------------------- | ----------------------- |
| Current released schema-5 model       | Normalized unified 59D sidecar, per-row domains, plain text | Four-step SDE           |
| Your canonical Stage 3 student export | Matched unified action metadata and training contract       | Four-step SDE           |

An HF export already selects its weight variant. Do not add a DCP EMA-selection
flag to the released-HF command.

## 2. Prepare the data contract and one input row

Each input row supplies:

- One non-empty plain-text prompt with `format_prompt_as_json=false`.
- `model_mode: forward_dynamics` and `action_condition: true`.
- `action_space: normalized_unified_v1` and `T` finite 59D action rows.
- Scalar or per-row domain routing supported by the checkpoint.
- `action_chunk_size: T`, `num_frames: T+1` and the action-aligned `fps`.

The published examples default to 900 actions and 901 frames at 30 FPS. This
is an example length, not a model limit; choose another aligned `T` for your
own trajectory.

The focused mixed AgiBot examples route 300 `agibotworld` rows, 300
`camera_pose` rows, then 300 `agibotworld` rows. The checkpoint maps these
domains to IDs 15, 2 and 15. Prepared values are already normalized and masked;
inference only pads five zero channels to reach 64. Do not apply training
normalization again. Read the
[action/data contract](../post-training/references/data-contract.md#cosmos-framework)
and [unified schema](../post-training/references/unified-v1-action-schema.md)
before preparing your own sidecar.

### Hugging Face-hosted examples

Download the published manifests and frustum configuration from the model
repository's default branch:

```bash
HF_EXAMPLE_DIR="${HF_EXAMPLE_DIR:-$PWD/hf-examples/Cosmos3-Nano-Sim-Bimanual}"
hf download nvidia/Cosmos3-Nano-Sim-Bimanual \
  examples/inference/sim_bimanual_unifiedv1_hf.jsonl \
  examples/inference/sim_bimanual_mixed_action_hf.jsonl \
  examples/inference/sim_bimanual_agibot_long60_hf.jsonl \
  examples/inference/bimanual_frustum.json \
  --local-dir "$HF_EXAMPLE_DIR"
```

Keep `HF_TOKEN` in the runtime environment when the gated model requires it.
The published manifests point directly to their Hugging Face images, action
sidecars and optional frustum files. Framework materializes those URLs with
`HF_TOKEN` through its standard download hook. Do not rewrite the example URLs
to the checkpoint revision.

Use AgiBot 001 for the worked example. Extract its complete row without
changing the hosted prompt, action routing or input URLs:

```bash
python - "$HF_EXAMPLE_DIR/examples/inference/sim_bimanual_unifiedv1_hf.jsonl" \
  "$HF_EXAMPLE_DIR/agibot_example.jsonl" <<'PY'
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

Additional YAM inputs are available in the
[Hugging Face examples](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Bimanual/tree/main/examples/inference).
Use the same command with a different `INPUT_MANIFEST`; its complete rows
select the matching image, prompt, actions and domains without source edits.

| Manifest                              | Contents                                                                                      |
| ------------------------------------- | --------------------------------------------------------------------------------------------- |
| `sim_bimanual_unifiedv1_hf.jsonl`     | Unified-action examples, including valid per-row domain routing                               |
| `sim_bimanual_mixed_action_hf.jsonl`  | AgiBot robot/camera/robot routing examples; overlap with the combined manifest is intentional |
| `sim_bimanual_agibot_long60_hf.jsonl` | Focused 60-second AgiBot-scene rollout; also present in the combined manifest                 |

`unifiedv1` names the normalized 59D action schema, not a scalar-routing-only
subset. Do not remove a row merely because its action sidecar uses per-step
robot/camera/robot domain routing. The focused mixed-action manifest exists to
make that behavior easy to select and may repeat rows from the combined
manifest.

Each manifest supplies one plain-text prompt per row with
`format_prompt_as_json=false`; do not add duration or resolution text.

Preserve `T` actions for `T+1` frames and the temporal-compression alignment:
the action count is divisible by four. Check RGB/action timestamps, domains,
units, gripper calibration and pose convention, not only tensor width.
An inference JSONL is not the completed training bundle.

## 3. Review prompting and prompt upsampling

Review the sample's task description against the initial RGB and supplied
actions. For user-authored prompts, describe the objects, starting scene,
arm assignments and task. Use the [prompt guide](../shared/prompt-guide.md).

The matched schema-5 export selects plain-text prompting automatically.
Prompt upsampling is not part of this recipe. If you use an authorized captioner,
review its output against the actual input and retain the original text.
Keep the selected initial-scene or future-described policy fixed for evaluation.

## 4. Choose important hyperparameter knobs

| Setting                            | Worked value                               | What to keep consistent                                                                |
| ---------------------------------- | ------------------------------------------ | -------------------------------------------------------------------------------------- |
| Checkpoint revision                | Current default HF revision                | Keep its schema-5 inputs and four-step schedule                                        |
| `--num-steps`                      | `4`                                        | Match the saved schedule; guidance/shift do not replace it                             |
| SDE ladder                         | `[1.0, 0.9375, 0.8333333333333334, 0.625]` | Read it from the selected export rather than reconstructing it                         |
| `action_chunk_size` / `num_frames` | From the input manifest                    | Supply one action per transition and respect compression by four                       |
| `--resolution` / `--image-size`    | `480` / `480`                              | The tier does not upscale arbitrary inputs; validate the selected row's saved geometry |
| `--seed`                           | `2026`                                     | Fix it for comparisons and record the effective value                                  |
| `--parallelism-preset`             | `latency` with one process                 | Recheck memory and outputs if placement changes                                        |
| Compile/cache                      | Compile off, KV cache 64, attention sink 8 | Keep matched when comparing outputs or timing                                          |

## 5. Generate video

From the same activated Framework environment:

```bash
INPUT_MANIFEST="${INPUT_MANIFEST:-$HF_EXAMPLE_DIR/agibot_example.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/outputs/sim_bimanual_hf_examples}"
export COSMOS_TRAINING=0
torchrun --standalone --nproc-per-node=1 \
  -m cosmos_framework.scripts.inference \
  -i "$INPUT_MANIFEST" \
  -o "$OUTPUT_DIR" \
  --checkpoint-path "$CHECKPOINT_DIR" \
  --model-mode=FORWARD_DYNAMICS --resolution=480 --image-size=480 \
  --num-steps=4 --seed=2026 \
  --autoregressive=True --kv-cache-inference-size=64 --attention-sink-size=8 \
  --parallelism-preset=latency --max-num-seqs=1 \
  --no-use-torch-compile --no-diffusion-cache --no-guardrails \
  --video-save-quality=4
```

This is the Framework model-validation command; it explicitly disables
guardrails. Use the deployment's guardrail policy for other applications.
Choose a new output directory when changing inputs or settings so existing
results are not mistaken for a new run.

The hosted manifests already contain their matching plain-text captions,
action routing, `action_chunk_size`, `num_frames` and `fps`. Preserve those row
fields and the four-step, KV-64, sink-8
command settings. Select another manifest by changing `INPUT_MANIFEST` and
choosing a new output directory. For your own trajectory, set both length
fields in the manifest; the command does not override them. To run a published
mixed-action example, use its complete row or the focused mixed-action manifest.
For frustum retrieval, copy one manifest row,
add `kv_cache_frustum_path` pointing to
`examples/inference/bimanual_frustum.json`, and use
`--no-use-torch-compile --kv-cache-inference-size=0 --attention-sink-size=0`
for an unbounded physical KV archive.

### Run the 60-second AgiBot example

Use the published long-trajectory manifest from the same download:

```bash
INPUT_MANIFEST="$HF_EXAMPLE_DIR/examples/inference/sim_bimanual_agibot_long60_hf.jsonl"
OUTPUT_DIR="$PWD/outputs/sim_bimanual_agibot_long60"
```

Rerun the generation command above with these paths. The manifest supplies
the complete initial image, prompt, action trajectory, frame count and FPS;
do not replace its length with the short example's settings. Check the output
against the manifest and inspect early, middle and late portions for drift,
flicker, identity changes and agreement with the actions.

### Use your own Stage 3 checkpoint

First [export the Stage 3 student](../post-training/SKILL.md#export-and-check-inference)
from its saved resolved config, selecting regular or EMA weights explicitly.
Set `CHECKPOINT_DIR` to that complete export and choose a new `OUTPUT_DIR`.

Use an input matching the export's action metadata and training semantics.
For the canonical four-step student, rerun the command above with the new
checkpoint/output paths. Keep its saved four-step SDE ladder rather than
assuming it is identical to the released model's ladder.
The released-example checker is specific to that release's metadata and is
not a general validator for every user export.

## 6. Verify outputs and hand off

1. Check the sample's `sample_outputs.json` for success and its generated MP4.
   Every requested row needs a playable output; keep failed or skipped rows
   visible in completeness accounting.
2. Set `OUTPUT_VIDEO` to the generated RGB video reported in that file and
   fully inspect its decoded structure:

```bash
ffprobe -v error -select_streams v:0 -count_frames \
  -show_entries stream=width,height,avg_frame_rate,nb_read_frames:format=duration \
  -of json "$OUTPUT_VIDEO"
```

3. Match the decoded frame count and FPS to that row's `num_frames` and `fps`.
   Validate the saved dimensions against the selected input's declared content geometry.
4. For a mixed-action row, inspect every domain transition at the timestamps
   defined by its action rows and FPS. For every row, inspect the full sequence for frozen
   frames, discontinuities and disagreement with the actions.
5. Preserve the effective prompt, input identity, checkpoint revision/weight
   selection, seed, command and outputs. Use the
   [evaluation guide](references/evaluation.md) for held-out scoring.

The [Framework inference contract](references/inference-contract.md#cosmos-framework)
links the native input checks and current schema rules. Use
[Framework troubleshooting](../../../../docs/faq.md) for environment/runtime failures.

## References

- [Action and data contract](../post-training/references/data-contract.md#cosmos-framework)
- [Unified action schema](../post-training/references/unified-v1-action-schema.md)
- [Prompt guide](../shared/prompt-guide.md)
- [Framework inference contract](references/inference-contract.md#cosmos-framework)
- [Held-out evaluation protocol](references/evaluation.md)
