# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Convert a posttrain CONSOLIDATED model checkpoint (model.pt) into an upstream-style DCP dir.

Why this exists: the upstream evaluation stack (e.g.
`cosmos3/evaluation/text_to_image/inference_unigenbench_distributed`) loads model
weights from a DCP directory of FLAT-keyed tensors (`net.<param>`, `net_ema.<param>` —
the format the upstream trainer writes and the wallace reference checkpoints use). posttrain's
consolidated format is one `torch.save` file holding NESTED state dicts
(`{"net": {...}, "net_ema": {...}}`). This tool bridges the two so any posttrain
checkpoint can go straight into the existing eval pipelines:

    python -m cosmos_framework.posttrain.utils.consolidated_to_dcp \
        <iter_dir>/model.pt <out_dir>/model

Notes:
- `torch.load(mmap=True)` + per-tensor streaming into DCP keeps peak RSS at O(one
  tensor), not O(checkpoint) — a real model.pt is ~85GB.
- A single-process `dcp.save` writes one shard; DCP reshards on load, so consumers at
  any world size (including the eval tool's per-GPU world=1) read it fine.
- Keys are flattened EXACTLY as the upstream trainer writes them: `<section>.<param_fqn>`.
"""

from __future__ import annotations

import argparse
import os


def convert(model_pt: str, out_dir: str) -> int:
    import torch
    import torch.distributed.checkpoint as dcp

    sd = torch.load(model_pt, map_location="cpu", weights_only=False, mmap=True)
    flat: dict[str, torch.Tensor] = {}
    for section, params in sd.items():
        if not isinstance(params, dict):
            raise ValueError(f"expected nested dict at {section!r}, got {type(params)}")
        for k, v in params.items():
            flat[f"{section}.{k}"] = v
    if not any(k.startswith("net.") for k in flat):
        raise ValueError(f"no 'net.*' keys after flattening {model_pt!r} — wrong file?")
    os.makedirs(out_dir, exist_ok=True)
    dcp.save(flat, checkpoint_id=out_dir)
    return len(flat)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("model_pt", help="path to a consolidated model.pt")
    p.add_argument("out_dir", help="DCP output dir (conventionally <...>/iter_X/model)")
    a = p.parse_args(argv)
    n = convert(a.model_pt, a.out_dir)
    print(f"wrote {n} tensors to DCP dir {a.out_dir}")


if __name__ == "__main__":
    main()
