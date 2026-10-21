# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Turn a posttrain checkpoint (the `action_head.*` wire scope only) back into a whole `Gr00tN1d7` checkpoint
that Isaac-GR00T and vLLM-Omni can load: rebuild the official model from the base checkpoint, overwrite the
action head with the trained one, `save_pretrained`, re-emit the processor, and verify the safetensors
headers carry the head (values are not re-read: the export is ~12 GB)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import torch

from cosmos_framework.posttrain.checkpoint.io import component_kind, load_component, load_component_dcp
from cosmos_framework.posttrain.checkpoint.store import LocalCheckpointStore


def read_head_state(iter_dir: str, template: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The trained `action_head.*` tensors from a posttrain `iter_NNNNNN` directory, in either checkpoint
    format. `template` (`Gr00tHeads.state_dict()`) supplies the keys and shapes DCP needs to load in place."""
    kind = component_kind(iter_dir, "model")
    if kind == "pt":
        return dict(load_component(iter_dir, "model")["net"])
    if kind == "dcp":
        inventory = LocalCheckpointStore().dcp_tensor_inventory(iter_dir, "model")
        missing = sorted(set(template) - set(inventory))
        extra = sorted(set(inventory) - set(template))
        mismatched = sorted(
            name for name in set(template) & set(inventory) if inventory[name] != tuple(template[name].shape)
        )
        if missing or extra or mismatched:
            raise RuntimeError(
                f"DCP head does not match the template: missing={missing[:5]}, extra={extra[:5]}, "
                f"shape_or_non_tensor={mismatched[:5]}"
            )
        state = {key: value.clone() for key, value in template.items()}
        if not load_component_dcp(iter_dir, "model", state):
            raise FileNotFoundError(f"{iter_dir}/model is not a readable DCP component")
        return state
    raise FileNotFoundError(f"no model component in {iter_dir}; point --posttrain-ckpt at an iter_NNNNNN directory")


def _safetensors_shapes(directory: str) -> dict[str, list[int]]:
    from safetensors import safe_open

    shapes: dict[str, list[int]] = {}
    for name in sorted(os.listdir(directory)):
        if name.endswith(".safetensors"):
            with safe_open(os.path.join(directory, name), framework="pt") as handle:
                for key in handle.keys():  # noqa: SIM118 - safetensors handles are not mappings
                    shapes[key] = list(handle.get_slice(key).get_shape())
    return shapes


def serving_config(base_config: dict, dtype: str) -> dict:
    """The exported ``config.json``: the BASE checkpoint's serving contract with only the storage dtype following
    ``dtype``.

    `save_pretrained` would write the TRAINING factory's config instead -- ``load_bf16=False``, ``tune_llm`` /
    ``tune_visual`` false, ``backbone_trainable_params_fp32`` -- and Isaac-GR00T honours those flags at load, so a
    checkpoint carrying them serves different numerics from the base it was trained from (measured 2026-09-09:
    `backbone_features` off by up to 2.9 against the same weights loaded under the base's config).
    """
    config = dict(base_config)
    storage = "bfloat16" if dtype == "bf16" else "float32"
    for key in ("dtype", "torch_dtype"):
        if key in config:
            config[key] = storage
    return config


def _base_file(base: str, name: str) -> str:
    """Path of ``name`` inside the base checkpoint: a local directory, or a Hub repo id through the HF cache."""
    from transformers.utils import cached_file

    path = cached_file(base, name)
    if path is None:
        raise FileNotFoundError(f"{base} has no {name}; the export needs the base checkpoint's own files")
    return path


#: The processor is part of the checkpoint contract; these are copied from the base byte for byte.
PROCESSOR_FILES = ("processor_config.json", "statistics.json", "embodiment_id.json")


def export_checkpoint(*, base: str, posttrain_ckpt: str, out: str, dtype: str = "fp32") -> None:
    """Write the complete checkpoint to `out`: the BASE checkpoint's files with the `action_head.*` tensors replaced.

    Every other tensor (`lm_head` included), the shard layout, the processor files and `config.json` (see
    `serving_config`) are the base's, byte for byte. That is the serving contract of an export -- "the base, with
    this head" -- and anything a model factory adds or drops changes what the same weights serve: Isaac's factory
    drops `lm_head`, the loader then re-initializes it, and `backbone_features` moved by up to 0.375 with the base's
    config and 2.9 with the factory's (measured 2026-09-09 on tensors otherwise byte-identical to the base).
    ``dtype`` applies to the head tensors: `bf16` is what `Gr00tPolicy` serves and is lossy for an fp32 master,
    `fp32` keeps the master.
    """
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    os.makedirs(out, exist_ok=True)
    with open(_base_file(base, "model.safetensors.index.json")) as handle:
        index = json.load(handle)
    weight_map: dict[str, str] = index["weight_map"]
    shard_names = sorted(set(weight_map.values()))
    shard_paths = {name: _base_file(base, name) for name in shard_names}
    # DCP loads in place: the template supplies the head's names and shapes, in fp32 so no master bit is lost
    # before `dtype` applies.
    template: dict[str, torch.Tensor] = {}
    for path in shard_paths.values():
        with safe_open(path, framework="pt") as handle:
            for key in handle.keys():  # noqa: SIM118 - safetensors handles are not mappings
                if key.startswith("action_head."):
                    template[key] = torch.empty(handle.get_slice(key).get_shape(), dtype=torch.float32)
    if not template:
        raise RuntimeError(f"{base} carries no action_head.* tensors; not a GR00T-N1.7 checkpoint")
    head = read_head_state(posttrain_ckpt, template)
    if set(head) != set(template):
        raise RuntimeError(
            f"the trained head does not match the base's: missing {sorted(set(template) - set(head))[:5]}, "
            f"extra {sorted(set(head) - set(template))[:5]}"
        )
    target = torch.bfloat16 if dtype == "bf16" else torch.float32
    total_size = 0
    for shard_name, path in shard_paths.items():
        tensors = load_file(path)
        for key in tensors:
            if key in head:
                if tuple(head[key].shape) != tuple(tensors[key].shape):
                    raise RuntimeError(
                        f"{key}: trained shape {tuple(head[key].shape)} vs base {tuple(tensors[key].shape)}"
                    )
                tensors[key] = head[key].to(target).contiguous()
        save_file(tensors, os.path.join(out, shard_name), metadata={"format": "pt"})
        total_size += sum(t.numel() * t.element_size() for t in tensors.values())
    index = {**index, "metadata": {**index.get("metadata", {}), "total_size": total_size}}
    with open(os.path.join(out, "model.safetensors.index.json"), "w") as handle:
        json.dump(index, handle, indent=2)
    with open(_base_file(base, "config.json")) as handle:
        base_config = json.load(handle)
    with open(os.path.join(out, "config.json"), "w") as handle:
        json.dump(serving_config(base_config, dtype), handle, indent=2)
    for name in PROCESSOR_FILES:
        shutil.copyfile(_base_file(base, name), os.path.join(out, name))

    shapes = _safetensors_shapes(out)
    wrong = [key for key, value in head.items() if shapes.get(key) != list(value.shape)]
    if wrong:
        raise RuntimeError(f"the exported safetensors do not carry the trained head: {wrong[:5]}")
    if set(shapes) != set(weight_map):
        raise RuntimeError(f"the export does not carry every base tensor: {sorted(set(weight_map) ^ set(shapes))[:5]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", required=True, help="the checkpoint training started from")
    parser.add_argument("--posttrain-ckpt", required=True, help="an iter_NNNNNN directory")
    parser.add_argument("--out", required=True, help="destination directory")
    parser.add_argument("--dtype", default="fp32", choices=("fp32", "bf16"))
    args = parser.parse_args(argv)
    export_checkpoint(base=args.base, posttrain_ckpt=args.posttrain_ckpt, out=args.out, dtype=args.dtype)
    print(f"exported {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
