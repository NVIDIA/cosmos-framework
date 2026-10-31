# Cosmos3 FP8 Quantization Examples

Convert the Cosmos3 checkpoints to static-scale **FP8** with NVIDIA TensorRT Model
Optimizer (ModelOpt). FP8 reduces weight storage and can accelerate
inference on NVIDIA GPUs with FP8 tensor cores; total memory also includes
activations and unquantized components. Quality depends on calibration and the task.
The output is a Diffusers-layout checkpoint (FP8 weights + per-tensor `weight_scale` /
`input_scale` sidecars + `hf_quant_config.json`) that loads on vLLM-Omni.

The recipe lives in `cosmos_framework.quantization`: checkpoint loading,
calibration, FP8 export and metadata assembly are owned by cosmos-framework.
These notebooks are walkthroughs of that API.

## What's in this folder

| Path | Role |
| --- | --- |
| [quantize_nano_super.ipynb](./quantize_nano_super.ipynb) | Nano and Super video-model quantization. |
| [quantize_super_text2image.ipynb](./quantize_super_text2image.ipynb) | Base and distilled Text-to-Image quantization. |
| [quantize_super_image2video.ipynb](./quantize_super_image2video.ipynb) | Base and distilled Image-to-Video quantization. |

The whole pipeline for one checkpoint is a single call:

```python
from cosmos_framework.quantization import quantize_fp8_checkpoint, SHAPE_VIDEO, SAMPLER_VIDEO_BASE

quantize_fp8_checkpoint(
    model_name_or_path="nvidia/Cosmos3-Nano",  # Hugging Face repo ID or local checkpoint
    output_dir="/path/to/nano-fp8",       # the FP8 drop-in written here
    profile="t2v", sampler=SAMPLER_VIDEO_BASE, shape=SHAPE_VIDEO,
    num_samples=8, calibration_behavior="framework",
)
```

`calibration_behavior="framework"` is the default for new checkpoints and is
explicit in all six notebook calls. It uses the framework sampler and model
behavior. `calibration_behavior="legacy"` is an opt-in compatibility recipe;
it is not the normal calibration path in these notebooks.

Base and distilled checkpoints use the same quantization API with their
respective sampler presets and weights. Keep the preset paired with the
checkpoint selected in each notebook section.

## Mixed activation precision

These are static checkpoint FP8 weights and calibrated scales, exported with
`calibration_behavior="framework"`; this is separate from quantizing BF16 weights
at server load time. The exporter records the activation policy in
`transformer/config.json` under `quantization_config.runtime.diffusion_step_policy`.
The default policy requests A16 activations for the first/last three denoising
iterations and the generator reasoner, and native FP8 in the middle. Overlapping
first/last windows use A16, so a four-step student uses A16 throughout. Weights
remain FP8. Standalone vLLM reasoner inference uses ModelOpt FP8 independently of
this diffusion policy. Backend support for the exported policy must be checked
with the chosen runtime image.

## Setup (once)

Use a Linux machine with an NVIDIA GPU and Hugging Face model access
(`uvx hf@latest auth login` or `HF_TOKEN`). Follow the
[Framework environment setup](../README.md#cosmos-framework), including system
libraries. Use the local checkout, not a separate PyPI or Git installation:

```bash
cd /path/to/cosmos-framework
export FRAMEWORK_ROOT="$PWD"
export COSMOS3_QUANTIZE_VENV="${COSMOS3_QUANTIZE_VENV:-$FRAMEWORK_ROOT/.venv}"
# cu128-train on CUDA 12.8; COSMOS3_TORCH_BACKEND=cu128 is also honored by notebooks.
export COSMOS3_UV_GROUP="${COSMOS3_UV_GROUP:-cu130-train}"
export UV_PROJECT_ENVIRONMENT="$COSMOS3_QUANTIZE_VENV"
export GIT_LFS_SKIP_SMUDGE=1
uv sync --all-extras --group="$COSMOS3_UV_GROUP"
uv pip install --python "$COSMOS3_QUANTIZE_VENV/bin/python" \
  "nvidia-modelopt[torch]==0.44.0" datasets ipykernel jupyter-client nbconvert
"$COSMOS3_QUANTIZE_VENV/bin/python" -m ipykernel install --user \
  --name cosmos3-quantize --display-name "Cosmos3 Quantize (Python 3.13)"
export RUN_ARTIFACTS="$(mktemp -d /tmp/cosmos3-quantize.XXXXXX)"
```

The notebooks work from the Framework root or their own directory. They resolve
the checkout separately from `.agents/skills`, including after switching kernels.
Open a notebook and run top to bottom, switching to **Cosmos3 Quantize (Python
3.13)** after step 3. For headless execution, the commands below keep executed
notebooks in `RUN_ARTIFACTS`; `OUTPUT_ROOT` selects the checkpoint destination.
Set `HF_HOME` / `HF_DATASETS_CACHE` for cache storage and the `C3_*_DIR` variables
shown in each notebook to use local source checkpoints.

By default **DEMO=1** uses one calibration prompt and a smaller shape; it still
loads the full checkpoint and runs every sampler step. `DEMO=0` uses the larger
shape and eight prompts. Neither mode guarantees published-quality reproduction.

## Quantize Cosmos3-Nano and Cosmos3-Super

Convert the Nano (16B) and Super (64B) checkpoints. Same recipe, different checkpoint —
Nano first, then Super.

### Run

```bash
cd "$FRAMEWORK_ROOT/.agents/skills/cookbooks/cosmos3/quantization"
DEMO=1 "$COSMOS3_QUANTIZE_VENV/bin/jupyter" nbconvert --to notebook --execute --output-dir "$RUN_ARTIFACTS" \
  --ExecutePreprocessor.kernel_name=cosmos3-quantize \
  --ExecutePreprocessor.timeout=7200 \
  quantize_nano_super.ipynb
```

### Notebook walkthrough

[`quantize_nano_super.ipynb`](./quantize_nano_super.ipynb) walks
through prerequisites, environment setup, and then quantizes Nano and Super in separate
sections, printing a short summary of each FP8 checkpoint it writes.

## Quantize Cosmos3-Super Text-to-Image (base and distilled)

Quantize the full-quality Text-to-Image model, then its 4-step distilled student. The
two use different checkpoint weights and sampler presets (50-step UniPC vs.
4-step FlowMatchEuler, guidance off). The notebook calibrates each separately.

### Run

```bash
cd "$FRAMEWORK_ROOT/.agents/skills/cookbooks/cosmos3/quantization"
DEMO=1 "$COSMOS3_QUANTIZE_VENV/bin/jupyter" nbconvert --to notebook --execute --output-dir "$RUN_ARTIFACTS" \
  --ExecutePreprocessor.kernel_name=cosmos3-quantize \
  --ExecutePreprocessor.timeout=7200 \
  quantize_super_text2image.ipynb
```

### Notebook walkthrough

[`quantize_super_text2image.ipynb`](./quantize_super_text2image.ipynb)
quantizes the base model in one section, then the distilled model in a second section that
explains the sampler difference.

## Quantize Cosmos3-Super Image-to-Video (base and distilled)

Same as Text-to-Image, but the image-to-video regime: calibration conditions on a clean
first frame (a real image, VAE-encoded) held fixed while the rest of the clip denoises.
Provide the conditioning images through `I2V_COND_DIR` (a local folder); if unset, the
notebook falls back to a public calibration image dataset.

### Run

```bash
cd "$FRAMEWORK_ROOT/.agents/skills/cookbooks/cosmos3/quantization"
DEMO=1 I2V_COND_DIR=/path/to/cond_images \
  "$COSMOS3_QUANTIZE_VENV/bin/jupyter" nbconvert --to notebook --execute --output-dir "$RUN_ARTIFACTS" \
  --ExecutePreprocessor.kernel_name=cosmos3-quantize \
  --ExecutePreprocessor.timeout=7200 \
  quantize_super_image2video.ipynb
```

### Notebook walkthrough

[`quantize_super_image2video.ipynb`](./quantize_super_image2video.ipynb)
sets up conditioning images, then quantizes the base and distilled models in separate
sections.

## Demo vs. full runs

| | `DEMO=1` (default) | `DEMO=0` (full calibration) |
| --- | --- | --- |
| Calibration prompts | 1 | 8 |
| Shape | small (e.g. 480×720, 29 frames) | production (720×1280, up to 189 frames) |
| Purpose | quick end-to-end check | calibrate with more prompts and a larger shape |

The FP8 **weights** are identical either way — they are computed directly from the bf16
weights (`weight_scale = max|W| / 448`) and do not depend on calibration. DEMO only affects
the calibrated activation **scales** (fewer samples, smaller shape). Neither mode
guarantees reproduction of published scales or generation quality.

## What gets produced

Each run writes a drop-in checkpoint at `OUTPUT_ROOT/<name>-fp8/`:

```
<name>-fp8/
├── transformer/                 # FP8 weights + weight_scale/input_scale + quantization_config
│   ├── diffusion_pytorch_model-*.safetensors
│   ├── diffusion_pytorch_model.safetensors.index.json
│   ├── modelopt_state.pth       # Diffusers ModelOpt state
│   └── config.json              # includes quantization_config.runtime.diffusion_step_policy
├── modelopt_state.pth           # reasoner ModelOpt state for Transformers
├── quantization_metadata.json   # producing environment, source and calibration recipe
├── hf_quant_config.json         # FP8 quantization_config at the root
├── model.safetensors.index.json # regenerated whole-model index
└── vae/ scheduler/ ...          # symlinked back to the source checkpoint
```

Point your serving stack at this directory.

## Inference

The FP8 output is a **drop-in checkpoint** — the same layout as the bf16 source, plus an
`hf_quant_config.json` and a `quantization_config` in `transformer/config.json`. A
compatible serving backend detects the ModelOpt FP8 format from checkpoint metadata. Serve it from a **local
directory** — either the output produced above, or the published `nvidia/Cosmos3-Nano-FP8` / `nvidia/Cosmos3-Super-FP8` checkpoint
downloaded with `snapshot_download` (pass `--served-model-name` so clients see the usual model id):

```bash
# A) the local FP8 output produced above
vllm serve /path/to/OUTPUT_ROOT/nano-fp8 --served-model-name nvidia/Cosmos3-Nano ...

# B) download the published FP8 repo to a local dir, then serve that dir
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('nvidia/Cosmos3-Nano-FP8', local_dir='cosmos3-nano-fp8')"
vllm serve cosmos3-nano-fp8 --served-model-name nvidia/Cosmos3-Nano ...
```

> Passing the FP8 checkpoint by `--revision` alone is **not** honored by the vLLM-Omni Cosmos3 pipeline (it
> loads the default revision), so the FP8 weights must be materialized to a local directory
> first.

**Runnable inference examples live in the other Cosmos3 cookbooks, not here.** Each
vLLM / vLLM-Omni notebook has a **Quantized Checkpoints** section that downloads the FP8
weights and serves them from a local directory:

| Cookbook | Notebook(s) |
| --- | --- |
| Generator · Audiovisual | [`cosmos3-generate-media` · vLLM-Omni](../../../cosmos3-generate-media/references/vllm-omni.md#quantized-checkpoints) |
| Generator · Action | [`run_fd_with_vllm_omni.ipynb`](../generator/action/run_fd_with_vllm_omni.ipynb), [`run_id_with_vllm_omni.ipynb`](../generator/action/run_id_with_vllm_omni.ipynb) |
| Generator · Transfer | [`run_video_transfer_with_vllm_omni.ipynb`](../generator/transfer/run_video_transfer_with_vllm_omni.ipynb) |
| Reasoner | [`run_with_vllm.ipynb`](../reasoner/run_with_vllm.ipynb) |

See the shared [Cosmos3 cookbooks environment setup](../README.md) for server install and
launch details ([vLLM](../README.md#vllm), [vLLM-Omni](../README.md#vllm-omni)).

For the four specialized Super checkpoints, use the locally quantized outputs;
this guide does not assume published `fp8` revisions. The audiovisual notebook
maps all six output directories to their original served model IDs. Generated
outputs contain symlinks: mount the source checkpoint (and any backing HF cache)
at its original absolute path inside Docker as shown in the serving notebooks.
