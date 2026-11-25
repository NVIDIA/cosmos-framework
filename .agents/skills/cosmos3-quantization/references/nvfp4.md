# Cosmos3 NVFP4 quantization recipes

Export local NVFP4 checkpoints from the BF16 sources with ModelOpt. These recipes
require no published NVFP4 weights. Nano defaults to **NVFP4-H6-R8**; Super defaults
to **NVFP4-R8**. H6 means the final six generator blocks use FP8; R8 means the
reasoner uses fixed W8A8 FP8. Native NVFP4 inference requires Blackwell tensor cores.

Use the [FP8 cookbook setup](quantization-readme.md#setup-once) and its local Framework environment.
The calibration, source checkpoint access and full model memory requirements
still apply. Select a ModelOpt release with `NVFP4_DEFAULT_CFG` and the unified
packed-weight exporter; exporter APIs are version-sensitive.

Follow the [Nano and Super video export walkthrough](quantize_nvfp4_nano_super.md) for the two reference recipes.

## Export

```python
from pathlib import Path
from cosmos_framework.quantization import (
    SAMPLER_VIDEO_BASE, SHAPE_VIDEO, quantize_nvfp4_checkpoint,
)

quantize_nvfp4_checkpoint(
    model_name_or_path="nvidia/Cosmos3-Nano",  # BF16 repo or local directory
    output_dir=Path.home() / "scratch/cosmos3-nvfp4/nano-h6-r8",
    profile="t2v", sampler=SAMPLER_VIDEO_BASE, shape=SHAPE_VIDEO,
    num_samples=8, seed=0, calibration_behavior="framework",
    fp8_last_k_layers=6,
    quantize_language_model=True,
    mixed_precision_steps=3,
)
```

For Super, select `nvidia/Cosmos3-Super` and `fp8_last_k_layers=0`.
`fp8_last_k_layers=k` selects generator blocks `[num_layers-k, num_layers)`.
It accepts integers from zero through the model's decoder depth. All Q/K/V,
attention output and MLP linears within a selected generator block use FP8.
The other generator blocks use packed NVFP4 W4A4. Embeddings, norms, modality
towers and input/output projections remain BF16.

| Control                         | Environment variable      | Effect                                                                     |
| ------------------------------- | ------------------------- | -------------------------------------------------------------------------- |
| `fp8_last_k_layers`             | `FP8_LAST_K_LAYERS`       | Final `k` generator blocks use FP8; zero keeps every generator block NVFP4 |
| `quantize_language_model=True`  | `QUANTIZE_REASONER=1`     | Reasoner linears use W8A8 FP8 at every step                                |
| `quantize_language_model=False` | `QUANTIZE_REASONER=0`     | Reasoner stays BF16                                                        |
| `mixed_precision_steps=3`       | `MIXED_PRECISION_STEPS=3` | First/last three diffusion iterations use A16 generator activations        |
| `mixed_precision_steps=0`       | `MIXED_PRECISION_STEPS=0` | Native generator activations throughout                                    |

A16 iterations affect generator activations only: weights stay NVFP4 or FP8,
and reasoner precision stays fixed. Layer precision and diffusion iterations
are independent controls. H6-R8-native selects six FP8 generator blocks, an FP8
reasoner, and zero A16 iterations.

Set `HF_HOME`, `HF_DATASETS_CACHE`, and `OUTPUT_ROOT` to storage with sufficient
space before running the examples. Set `C3_NANO_DIR` or `C3_SUPER_DIR` to use a local BF16 source.

```bash
export HF_HOME="$HOME/scratch/cosmos3-nvfp4/cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export OUTPUT_ROOT="$HOME/scratch/cosmos3-nvfp4/run-001/checkpoints"
```

Run the Python blocks in the linked walkthrough in one interpreter, selecting the model sections you need.

`DEMO=1` uses one prompt at a small shape; `DEMO=0` uses eight prompts at the
production shape. Neither guarantees reference scales or quality. Calibration
uses native Framework generation, including the text prefix; it is not a
standalone multimodal reasoner calibration or an exact replay of prior benchmarks.

## Exported checkpoint

The output contains sharded transformer safetensors, a transformer weight index,
`transformer/config.json`, a fresh root weight index, `hf_quant_config.json`,
and `quantization_metadata.json`. Untouched components are linked to the source.
NVFP4 weights are packed `uint8` with block scales and secondary scales; FP8 weights
retain their logical matrix shape. Per-linear `quantized_layers` metadata records
the actual stored formats. `runtime.diffusion_step_policy` records A16 iterations
and keeps the reasoner in `native` mode (FP8 when R8 is selected).

This inference export does not write ModelOpt graph restoration state. Use the
existing FP8 recipes when a checkpoint for the Transformers ModelOpt graph loader
is required. These NVFP4 recipes target the Cosmos3 generator in vLLM-Omni.

## Local inference

Serve the generated local directory with a vLLM-Omni build supporting Cosmos3
ModelOpt mixed FP8/NVFP4 checkpoints and the diffusion-step precision policy.
Keep the original served model ID for the matching request examples.

```bash
vllm serve "$OUTPUT_ROOT/nano-nvfp4-h6-r8-steps3" \
  --served-model-name nvidia/Cosmos3-Nano \
  --omni --model-class-name Cosmos3OmniDiffusersPipeline \
  --allowed-local-media-path / --port 8000 --init-timeout 1800
```

Use the [generator inference requests](../../cosmos3-inference-generate-media/references/vllm-omni.md)
with the matching model. Mount the export, source and any backing cache at their
original absolute paths in a container. Do not download an assumed NVFP4 repository
or select NVFP4 using `--revision`.

The recipes require GPU export and serving validation with the selected ModelOpt
and vLLM-Omni versions. No quality, speed or memory measurements are established
by these recipes; historical benchmark values are not reproduction guarantees.
