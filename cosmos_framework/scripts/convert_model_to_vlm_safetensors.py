# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Convert a Cosmos3 OmniMoT checkpoint into a Qwen3-VL HF safetensors directory.

The result is a complete HF directory (config.json + safetensors + tokenizer)
shaped like a Qwen3VLForConditionalGeneration release, but with the language-model
tensors sourced from the Cosmos3 OmniMoT and the visual tower kept from the
Qwen3-VL-Instruct release (Cosmos3 has no visual counterpart).

Pass the resulting path as ``[model.backbone].safetensors_path`` in a VLM SFT
TOML to bootstrap training from Cosmos3 weights while keeping the public HF
``model_name`` for tokenizer/architecture discovery.

Example:
  python -m cosmos_framework.scripts.convert_model_to_vlm_safetensors \\
      --checkpoint-path Cosmos3-Nano \\
      -o examples/checkpoints/Cosmos3-Nano-VLM
"""

from cosmos_framework.inference.common.init import init_script

init_script(
    env={
        "COSMOS_DEVICE": "cpu",
    }
)

from typing import Annotated

import pydantic
import torch
import tyro
from torch.distributed.checkpoint.state_dict import get_model_state_dict
from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

from cosmos_framework.inference.args import OmniSetupOverrides
from cosmos_framework.inference.common.args import CheckpointOverrides, ResolvedPath
from cosmos_framework.inference.model import Cosmos3OmniModel


# Cosmos3 OmniMoT exposes its inner LM under ``net.language_model.*``; Qwen3VL
# expects ``lm_head.*`` (top-level) and ``model.language_model.*`` (text-decoder
# sub-tree of the VLM). The OmniMoT MoE-generation pathway (``*_moe_gen``) has
# no Qwen3VL counterpart and is dropped.
_OMNIMOT_LM_PREFIX = "net.language_model."


def _remap_to_qwen3vl(key: str) -> str | None:
    """Return the Qwen3VL VLM-shape key, or None if ``key`` should be dropped."""
    if not key.startswith(_OMNIMOT_LM_PREFIX):
        return None
    inner = key[len(_OMNIMOT_LM_PREFIX):]
    if "_moe_gen" in inner:
        return None
    if inner.startswith("lm_head."):
        return inner
    if inner.startswith("model."):
        return "model.language_model." + inner[len("model."):]
    return None


class Args(pydantic.BaseModel):
    checkpoint: CheckpointOverrides
    """Cosmos3 OmniMoT checkpoint (e.g. Cosmos3-Nano)."""
    output_path: Annotated[ResolvedPath, tyro.conf.arg(aliases=("-o",))]
    """Output Qwen3-VL HF safetensors directory."""
    vlm_model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    """HF Hub ID for the VLM whose visual tower + config + tokenizer to inherit."""


def convert_model_to_vlm_safetensors(args: Args) -> None:
    print(f"Loading Cosmos3 checkpoint via CheckpointOverrides...")
    cosmos3_config = args.checkpoint.build_checkpoint(checkpoints=OmniSetupOverrides.CHECKPOINTS)
    cosmos3_path = cosmos3_config.download_checkpoint()
    cosmos3_model = Cosmos3OmniModel.from_pretrained_dcp(cosmos3_path)
    cosmos3_state = get_model_state_dict(cosmos3_model.model)

    lm_state: dict[str, torch.Tensor] = {}
    for k, v in cosmos3_state.items():
        new_k = _remap_to_qwen3vl(k)
        if new_k is not None and isinstance(v, torch.Tensor):
            lm_state[new_k] = v
    print(f"  extracted {len(lm_state)} LM tensors from {len(cosmos3_state)} OmniMoT tensors")
    del cosmos3_state, cosmos3_model

    print(f"Loading {args.vlm_model_name} (visual tower + LM defaults, bf16, CPU)...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.vlm_model_name, dtype=torch.bfloat16
    )

    incompatible = model.load_state_dict(lm_state, strict=False)
    n_overlaid = len(lm_state) - len(incompatible.unexpected_keys)
    print(f"  overlaid {n_overlaid}/{len(lm_state)} LM tensors "
          f"(unexpected={len(incompatible.unexpected_keys)}, "
          f"missing-in-LM-state={len(incompatible.missing_keys)} — these are visual/etc kept from HF)")
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"Cosmos3 LM tensors not present in Qwen3VL: "
            f"{incompatible.unexpected_keys[:5]}{'...' if len(incompatible.unexpected_keys) > 5 else ''}"
        )

    print(f"Saving merged Qwen3-VL safetensors to {args.output_path}...")
    args.output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_path, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.vlm_model_name).save_pretrained(args.output_path)
    AutoProcessor.from_pretrained(args.vlm_model_name).save_pretrained(args.output_path)
    print(f"Done.")


def main() -> None:
    args = tyro.cli(Args, description=__doc__, config=(tyro.conf.OmitArgPrefixes,))
    convert_model_to_vlm_safetensors(args)


if __name__ == "__main__":
    main()
