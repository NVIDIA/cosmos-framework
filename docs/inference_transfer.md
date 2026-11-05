# Simulation Transfer inference

This recipe generates video from a caption and one precomputed **edge**, **blur**,
**depth**, or **segmentation** control using
[`nvidia/Cosmos3-Nano-Sim-Transfer`](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Transfer).
The causal model generates video autoregressively while retaining control and
generated-latent history.

Compiled inference includes [numerical-preservation safeguards](transfer_numerical_preservation.md)
without an additional option. The recipe below keeps its existing eager settings.

Follow [the framework setup instructions](setup.md) first. Use one NVIDIA GPU
with sufficient memory and a CUDA dependency group matching your driver. The
control-video decoder requires PyAV 18. Run commands below from the framework
repository root with its virtual environment active.

## Checkpoint and cookbook inputs

Authenticate to Hugging Face with read access to the model repository and download
the exported checkpoint:

```bash
hf download nvidia/Cosmos3-Nano-Sim-Transfer \
  --local-dir checkpoints/Cosmos3-Nano-Sim-Transfer
```

Pass this local directory to `--checkpoint-path`. Its exported configuration
defines the model architecture and sampling schedule; no checkpoint export is
needed. Required tokenizer and VAE artifacts may be downloaded separately by
their registered loaders.

The four input specifications, captions, and control videos live in the
[Cosmos Transfer cookbook](../.agents/skills/cookbooks/cosmos3/simulation/transfer).
Point `COSMOS_COOKBOOK_ROOT` at the cookbook root, `.agents/skills` in this
checkout, and materialize its Git LFS assets:

```bash
export COSMOS_COOKBOOK_ROOT=/absolute/path/to/cosmos-framework/.agents/skills
export SIM_TRANSFER_ROOT="$COSMOS_COOKBOOK_ROOT/cookbooks/cosmos3/simulation/transfer"
git -C "$COSMOS_COOKBOOK_ROOT" lfs install --local
git -C "$COSMOS_COOKBOOK_ROOT" lfs pull
```

The JSON input paths resolve relative to each specification, including its
references to the shared Generator Transfer assets. Keep that cookbook directory
structure intact. Each specification explicitly supplies the model-specific sample settings,
so this recipe does not require changing the framework's generic defaults.

## Run inference

```bash
export LD_LIBRARY_PATH=
export COSMOS_TRAINING=0
python -m cosmos_framework.scripts.inference \
  -i "$SIM_TRANSFER_ROOT/specs/sim_transfer_depth.json" \
  -o outputs/sim-transfer-depth \
  --checkpoint-path checkpoints/Cosmos3-Nano-Sim-Transfer \
  --seed 42 --no-use-torch-compile --condition-stream-encode \
  --kv-cache-inference-size 30 --attention-sink-size 3 \
  --dp-shard-size 1 --cp-size 1 --cfgp-size 1 --no-guardrails --debug
```

Use `sim_transfer_edge.json`, `sim_transfer_blur.json`, or
`sim_transfer_seg.json` for the other controls, with a fresh output directory for
each run. These samples request **121 frames at 30 FPS**. Depth, edge, and
segmentation use an **832 × 480** canvas; blur uses **736 × 544**. The `480`
resolution tier selects an aspect-ratio-specific canvas, rather than requiring
every output to be 480 pixels high.

Keep `num_frames`, `max_frames`, and `num_video_frames_per_chunk` equal when
adapting the specifications. Supply enough control frames and a pre-upsampled
caption, with `prompt_upsampling: false`. The native recipe uses one control per
sample, autoregressive generation, `guidance: 1`, `control_guidance: 1`, and the
checkpoint's fixed four-step SDE schedule. `num_steps: 4` and `shift: 5` in these
specifications are compatibility fields; the fixed sampler does not apply the
RF shift again.

The command selects incremental control encoding, a 30-latent-frame history,
three attention-sink frames, and eager execution on one GPU. The dedicated
Transfer path does not invoke guardrails; the command makes that setting
explicit. See the
[native cookbook recipe](../.agents/skills/cookbooks/cosmos3/simulation/transfer/transfer.md#cosmos-framework)
for input requirements, supported canvases, and control-quality checks.

## Validate the output

For each run, verify that `sample_outputs.json` reports success, `vision.mp4`
decodes for all 121 requested frames at 30 FPS, and the generated video follows
the spatial control through the final chunk. Also inspect temporal consistency
and visible artifacts across the full clip.
