# Cosmos3-Nano-Sim-Bimanual

Generate an open-loop video rollout from an initial RGB observation, a task
description, and a time-aligned bimanual action trajectory. The model predicts
the visual result of supplied actions; it does not choose robot actions.

Use the current Cosmos Framework `main` branch with the default revision of
[`nvidia/Cosmos3-Nano-Sim-Bimanual`](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Bimanual).
The released checkpoint uses schema 5, normalized unified 59D actions,
plain-text prompts, per-row action domains, and a fixed four-step SDE schedule.

For the complete input contract and troubleshooting guidance, use the
[Bimanual inference Skill](../../../../cosmos3-nano-sim-bimanual/inference/SKILL.md).

## Supported backends

- **Cosmos Framework** for native PyTorch inference with the commands below.
- **vLLM-Omni** for an OpenAI-compatible video server. The
  [runnable notebook](run_bimanual_with_vllm_omni.ipynb) uses the same current
  Hugging Face checkpoint and published examples.
- **[FlashDreams](https://flashdreams.org/main/models/cosmos3_nano_sim_bimanual/)**: [integrate with FlashDreams](https://github.com/NVIDIA/flashdreams/integrations_v2/cosmos3_nano_sim_bimanual) for accelerated inference.

## Set up Cosmos Framework

From a dedicated checkout of the public repository:

```bash
git clone https://github.com/NVIDIA/cosmos-framework.git
cd cosmos-framework
git lfs install --local
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
uvx --from 'uv==0.11.29' uv lock --check
uvx --from 'uv==0.11.29' uv sync --locked --group cu130 --extra guardrail
source .venv/bin/activate
export COSMOS_INTERNAL=0 COSMOS_TRAINING=0
```

Select the CUDA dependency group that matches the installed driver. Keep the
FFmpeg library search path required by the host installation.

## Download the model and examples

Authenticate to Hugging Face when the gated model requires it. Keep the token
in `HF_TOKEN`; do not put its value in a command, manifest, or log.

```bash
CHECKPOINT_DIR="$PWD/checkpoints/Cosmos3-Nano-Sim-Bimanual"
HF_EXAMPLE_DIR="$PWD/hf-examples/Cosmos3-Nano-Sim-Bimanual"

hf download nvidia/Cosmos3-Nano-Sim-Bimanual \
  --local-dir "$CHECKPOINT_DIR"
hf download nvidia/Cosmos3-Nano-Sim-Bimanual \
  examples/inference/sim_bimanual_unifiedv1_hf.jsonl \
  examples/inference/sim_bimanual_mixed_action_hf.jsonl \
  examples/inference/sim_bimanual_agibot_long60_hf.jsonl \
  examples/inference/bimanual_frustum.json \
  --local-dir "$HF_EXAMPLE_DIR"
```

The published manifests use `/resolve/main/` URLs for their image and action
assets. Framework downloads those inputs through its standard URL hook and
uses `HF_TOKEN` when access is gated. `bimanual_frustum.json` is an optional
frustum configuration, not an inference manifest.

The published examples default to 900 actions for 901 frames at 30 FPS; users
can choose another rollout length. The action count must equal `num_frames - 1`,
be divisible by the checkpoint's temporal
compression factor, and match `action_chunk_size`. Action sidecars are already
normalized; do not normalize them a second time.

The walkthrough uses AgiBot. Additional YAM inputs are available in
the [Hugging Face examples](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Bimanual/tree/main/examples/inference).
Use their complete manifest rows with the same command and checkpoint;
Framework source edits are not needed.

## Run inference

Extract the AgiBot 001 row for the worked example:

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

Choose the input manifest and a new output directory:

```bash
INPUT_MANIFEST="${INPUT_MANIFEST:-$HF_EXAMPLE_DIR/agibot_example.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/outputs/sim_bimanual_unifiedv1}"

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

This example command disables the safety guardrails. Apply the deployment's
guardrail policy in other applications. Use one active sequence at a time for
the long-horizon examples and preserve the manifest's prompt, action routing,
and geometry fields.

The command reads `action_chunk_size`, `num_frames` and `fps` from each row.
Set these fields for your own trajectory instead of overriding every row
with a fixed command-line length. To run the published 60-second AgiBot
trajectory, select it and rerun the same command with a fresh output directory:

```bash
INPUT_MANIFEST="$HF_EXAMPLE_DIR/examples/inference/sim_bimanual_agibot_long60_hf.jsonl"
OUTPUT_DIR="$PWD/outputs/sim_bimanual_agibot_long60"
```

## Check the result

Require `sample_outputs.json` to report success for every requested row. Fully
decode each MP4 and match the frame count and FPS to that row's `num_frames`
and `fps`:

```bash
ffprobe -v error -select_streams v:0 -count_frames \
  -show_entries stream=width,height,avg_frame_rate,nb_read_frames:format=duration \
  -of json "$OUTPUT_VIDEO"
```

Inspect the full rollout for frozen frames, discontinuities, and disagreement
with the supplied actions. For mixed-action samples, inspect each domain
transition at the times defined by its action rows and FPS.
For the long example, inspect early, middle and late portions for drift,
flicker, identity changes and action response.

## vLLM-Omni

The [Bimanual vLLM-Omni notebook](run_bimanual_with_vllm_omni.ipynb) shows how
to serve the checkpoint through vLLM-Omni's synchronous Videos API, download
the published inputs from Hugging Face, generate unified-action and
mixed-domain rollouts, and validate each output's frame count, frame rate,
dimensions, and full decode.

The notebook uses the `vllm/vllm-omni:cosmos3` Docker image shared by all
Simulation notebooks and keeps the checkpoint's four-step schedule. It reads
rollout length and FPS from each selected manifest row and includes a
selection for the hosted 60-second example. Its two default short examples
are not fixed model limits.

## Adapt the model

To train on another supported data source, follow the
[Bimanual post-training Skill](../../../../cosmos3-nano-sim-bimanual/post-training/SKILL.md).
Framework expects a completed local training bundle containing its manifest,
samples, local resources, and tokenizer files.

## Related

- [Bimanual inference guide](../../../../../../docs/inference_bimanual.md)
- [Bimanual data preflight](../../../../../../docs/bimanual_data_preflight.md)
- [Hugging Face model and examples](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Bimanual)
- [vLLM-Omni](https://github.com/vllm-project/vllm-omni)
