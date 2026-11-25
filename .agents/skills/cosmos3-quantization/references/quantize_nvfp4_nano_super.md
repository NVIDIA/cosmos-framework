<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Export Nano and Super video checkpoints to NVFP4

Complete the [environment setup](quantization-readme.md#setup-once), then run the setup block and your chosen model section in the same Python session. See the [recipe controls](nvfp4.md#export) for precision overrides.

```python
import gc
import json
import os
import sys
from pathlib import Path

import torch

start = Path(os.environ.get("COSMOS3_REPO", Path.cwd())).expanduser().resolve()
FRAMEWORK_ROOT = next(p for p in (start, *start.parents) if (p / "cosmos_framework").is_dir())
sys.path.insert(0, str(FRAMEWORK_ROOT))
from cosmos_framework.quantization import (
    SAMPLER_VIDEO_BASE,
    SHAPE_VIDEO,
    SHAPE_VIDEO_DEMO,
    quantize_nvfp4_checkpoint,
)

OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", Path.home() / "scratch/cosmos3-nvfp4")).expanduser().resolve()
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
DEMO = os.environ.get("DEMO", "1") == "1"
NUM_SAMPLES = 1 if DEMO else 8
QUANTIZE_REASONER = os.environ.get("QUANTIZE_REASONER", "1") == "1"
MIXED_PRECISION_STEPS = int(os.environ.get("MIXED_PRECISION_STEPS", "3"))
print("Source checkout:", FRAMEWORK_ROOT)
print("Output root:", OUTPUT_ROOT)
```

## Nano

The default FP8 generator tail is 6 blocks. Choose a fresh output directory for each recipe.

```python
gc.collect()
torch.cuda.empty_cache()
fp8_tail = int(os.environ.get("FP8_LAST_K_LAYERS", "6"))
output_dir = OUTPUT_ROOT / f"nano-nvfp4-h{fp8_tail}-r{8 if QUANTIZE_REASONER else 16}-steps{MIXED_PRECISION_STEPS}"
if output_dir.exists():
    raise FileExistsError(f"Choose a fresh OUTPUT_ROOT to preserve {output_dir}")

quantize_nvfp4_checkpoint(
    model_name_or_path=os.environ.get("C3_NANO_DIR", "nvidia/Cosmos3-Nano"),
    output_dir=output_dir,
    profile="t2v",
    sampler=SAMPLER_VIDEO_BASE,
    shape=SHAPE_VIDEO_DEMO if DEMO else SHAPE_VIDEO,
    num_samples=NUM_SAMPLES,
    seed=0,
    calibration_behavior="framework",
    fp8_last_k_layers=fp8_tail,
    quantize_language_model=QUANTIZE_REASONER,
    mixed_precision_steps=MIXED_PRECISION_STEPS,
)
config = json.loads((output_dir / "transformer/config.json").read_text())["quantization_config"]
print(output_dir)
print(
    "Formats:",
    {fmt: sum(v["quant_algo"] == fmt for v in config["quantized_layers"].values()) for fmt in ("NVFP4", "FP8")},
)
print("Activation policy:", config["runtime"]["diffusion_step_policy"])
```

## Super

The default FP8 generator tail is 0 blocks. Choose a fresh output directory for each recipe.

```python
gc.collect()
torch.cuda.empty_cache()
fp8_tail = int(os.environ.get("FP8_LAST_K_LAYERS", "0"))
output_dir = OUTPUT_ROOT / f"super-nvfp4-h{fp8_tail}-r{8 if QUANTIZE_REASONER else 16}-steps{MIXED_PRECISION_STEPS}"
if output_dir.exists():
    raise FileExistsError(f"Choose a fresh OUTPUT_ROOT to preserve {output_dir}")

quantize_nvfp4_checkpoint(
    model_name_or_path=os.environ.get("C3_SUPER_DIR", "nvidia/Cosmos3-Super"),
    output_dir=output_dir,
    profile="t2v",
    sampler=SAMPLER_VIDEO_BASE,
    shape=SHAPE_VIDEO_DEMO if DEMO else SHAPE_VIDEO,
    num_samples=NUM_SAMPLES,
    seed=0,
    calibration_behavior="framework",
    fp8_last_k_layers=fp8_tail,
    quantize_language_model=QUANTIZE_REASONER,
    mixed_precision_steps=MIXED_PRECISION_STEPS,
)
config = json.loads((output_dir / "transformer/config.json").read_text())["quantization_config"]
print(output_dir)
print(
    "Formats:",
    {fmt: sum(v["quant_algo"] == fmt for v in config["quantized_layers"].values()) for fmt in ("NVFP4", "FP8")},
)
print("Activation policy:", config["runtime"]["diffusion_step_policy"])
```

For the output layout and serving command, see [Exported checkpoint](nvfp4.md#exported-checkpoint).
