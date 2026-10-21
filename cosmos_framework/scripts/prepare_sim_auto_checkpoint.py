# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Prepare a Transfer-Auto HF snapshot as weights-only Sim-Auto initialization.

This uses the native HF loader. Acceptance also requires strict readback in the
matched training runtime; conversion alone does not qualify a training run.
"""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-path", "-o", type=Path, required=True)
    args = parser.parse_args()
    source = args.checkpoint_path.resolve()
    output = args.output_path.resolve()
    lidar_vae = source / "lidar_vae/diffusion_pytorch_model.safetensors"
    for path in (source / "config.json", lidar_vae):
        if not path.is_file():
            parser.error(f"Required snapshot file is missing: {path}")
    if output.exists():
        parser.error(f"Use a new output directory: {output}")

    # Reuse the Framework HF loader; do not change attention or remap weights here.
    from cosmos_framework.inference.common.init import init_script

    init_script(env={"COSMOS_DEVICE": "cpu"})

    import torch
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemWriter
    from torch.distributed.checkpoint.metadata import StorageMeta
    from torch.distributed.checkpoint.state_dict import get_model_state_dict

    from cosmos_framework.inference.model import Cosmos3OmniConfig, Cosmos3OmniModel

    class PortableWriter(FileSystemWriter):
        """Do not pickle Python-version-specific pathlib classes in storage metadata."""

        def storage_meta(self) -> StorageMeta | None:
            metadata = super().storage_meta()
            if metadata is not None and metadata.checkpoint_id is not None:
                metadata.checkpoint_id = str(metadata.checkpoint_id)
            return metadata

    config = Cosmos3OmniConfig.from_pretrained(source)
    model_config = config.model["config"]
    model_config["vlm_config"]["pretrained_weights"]["enabled"] = False
    model_config["diffusion_expert_config"]["load_weights_from_pretrained"] = False
    model_config["lidar_tokenizer"].update(
        vae_path=str(lidar_vae),
        bucket_name="",
        object_store_credential_path_pretrained=None,
        device="cpu",
    )
    model = Cosmos3OmniModel.from_pretrained_dcp(source, config=config)
    state = get_model_state_dict(model.model)  # name -> parameter-shaped tensors
    for prefix in ("net.lidar2llm.", "net.llm2lidar.", "net.rig_view_embed."):
        if not any(name.startswith(prefix) for name in state):
            raise ValueError(f"Expected trained Transfer-Auto module is absent: {prefix}")
    if any(name.startswith("net_ema.") for name in state):
        raise ValueError("The exported HF loader must produce model weights, not an EMA history")
    size = sum(t.numel() * t.element_size() for t in state.values() if isinstance(t, torch.Tensor))
    output.mkdir(parents=True, exist_ok=False)
    dcp.save(
        state_dict=state,
        storage_writer=PortableWriter(output / "model", thread_count=max(1, math.ceil(size / (5 * 1024**3)))),
    )
    config.save_pretrained(output / "model")
    if (source / "checkpoint.json").is_file():
        shutil.copyfile(source / "checkpoint.json", output / "checkpoint.json")
    print(f"Saved weights-only DCP: {output}; retain the source snapshot for tokenizer assets.")


if __name__ == "__main__":
    main()
