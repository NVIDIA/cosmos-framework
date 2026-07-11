# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Extract the Cosmos3-Edge reasoner into a canonical VLM safetensors directory.

The public ``nvidia/Cosmos3-Edge`` release IS the Edge reasoner
(``NemotronSiglip2ForConditionCausalLM``), but its language-model tensors are
stored in "Diffusers-shard" key layout (``layers.N.self_attn.to_q`` etc.). The
repo's own ``Cosmos3EdgeForConditionCausalLM.from_pretrained`` remaps those to
the canonical reasoner keys (``model.language_model.layers.2N.mixer.q_proj`` …).

The VLM SFT training path (``HFModel``) builds the backbone with
``AutoModel.from_config`` + ``load_weights`` — which bypasses that
``from_pretrained`` remap — so it cannot consume ``nvidia/Cosmos3-Edge``'s raw
shards directly. This script loads the repo via ``AutoModel.from_pretrained``
(applying the remap) and re-saves a canonical reasoner snapshot, mirroring how
``convert_model_to_vlm_safetensors`` produces ``Cosmos3-Nano-VLM`` for the nano
reasoner recipe.

Pass the resulting path as ``VLM_SAFETENSORS_PATH`` (→ ``[model.backbone].safetensors_path``)
in the ``videophy2_sft_edge`` recipe; ``model_name`` stays ``nvidia/Cosmos3-Edge``
for config / tokenizer / architecture discovery.

Example:
  python -m cosmos_framework.scripts.convert_edge_reasoner_to_vlm_safetensors \\
      --checkpoint-path Cosmos3-Edge \\
      -o examples/checkpoints/Cosmos3-Edge-Reasoner-VLM
"""

from cosmos_framework.inference.common.init import init_script

init_script(
    env={
        "COSMOS_DEVICE": "cpu",
    }
)

import json
import shutil
from pathlib import Path
from typing import Annotated

import pydantic
import torch
import tyro
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from cosmos_framework.inference.args import OmniSetupOverrides
from cosmos_framework.inference.common.args import CheckpointOverrides, ResolvedPath

# Remote-code files bundled in nvidia/Cosmos3-Edge that the reasoner snapshot must
# carry so it is loadable standalone via trust_remote_code.
_REMOTE_CODE_FILES = (
    "modeling_nemotron_siglip2_h.py",
    "configuration_nemotron_siglip2_h.py",
    "processing.py",
    "chat_template.jinja",
)


class Args(pydantic.BaseModel):
    checkpoint: CheckpointOverrides
    """Cosmos3-Edge omni checkpoint (e.g. Cosmos3-Edge)."""
    output_path: Annotated[ResolvedPath, tyro.conf.arg(aliases=("-o",))]
    """Output canonical reasoner HF safetensors directory."""


def convert_edge_reasoner_to_vlm_safetensors(args: Args) -> None:
    print("Resolving Cosmos3-Edge checkpoint via CheckpointOverrides...")
    edge_config = args.checkpoint.build_checkpoint(checkpoints=OmniSetupOverrides.CHECKPOINTS)
    edge_path = Path(edge_config.download_checkpoint())

    print(f"Loading {edge_path} via AutoModel (Cosmos3EdgeForConditionCausalLM remaps "
          f"Diffusers shards -> canonical reasoner keys)...")
    model = AutoModel.from_pretrained(
        edge_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    n = sum(1 for _ in model.state_dict())
    print(f"  loaded {type(model).__name__} ({n} tensors, canonical layout)")

    args.output_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving canonical reasoner safetensors to {args.output_path}...")
    model.save_pretrained(args.output_path, safe_serialization=True)
    AutoTokenizer.from_pretrained(edge_path, trust_remote_code=True).save_pretrained(args.output_path)
    AutoProcessor.from_pretrained(edge_path, trust_remote_code=True).save_pretrained(args.output_path)

    for fn in _REMOTE_CODE_FILES:
        src = edge_path / fn
        if src.exists():
            shutil.copy(src, args.output_path / fn)

    # Repoint the saved config at the PLAIN reasoner class. The weights are now
    # canonical, so re-applying Cosmos3Edge's Diffusers->canonical remap (which
    # only fires in its from_pretrained) would double-map and corrupt a standalone
    # load. Training (HFModel: from_config + load_weights) never triggers it, but
    # this keeps the snapshot safe to load directly too.
    cfg_path = args.output_path / "config.json"
    cfg = json.loads(cfg_path.read_text())
    auto_map = cfg.get("auto_map", {})
    auto_map["AutoModel"] = "modeling_nemotron_siglip2_h.NemotronSiglip2ForConditionCausalLM"
    cfg["auto_map"] = auto_map
    cfg["architectures"] = ["NemotronSiglip2ForConditionCausalLM"]
    cfg_path.write_text(json.dumps(cfg, indent=2))

    print(f"Done. Pass {args.output_path} as VLM_SAFETENSORS_PATH to launch_sft_videophy2_edge.sh.")


def main() -> None:
    args = tyro.cli(Args, description=__doc__, config=(tyro.conf.OmitArgPrefixes,))
    convert_edge_reasoner_to_vlm_safetensors(args)


if __name__ == "__main__":
    main()
