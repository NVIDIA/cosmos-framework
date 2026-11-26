# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Immutable local Alpamayo/processor snapshots; architecture-specific target resolution."""

from __future__ import annotations

import json
from pathlib import Path

from cosmos_framework.posttrain.utils.identity import canonical_json_sha256
from cosmos_framework.posttrain.utils.object_store import file_sha256

BASE_REVISION = "dd4a24cacefc9a6477a6dfc7354de2443401409d"
PROCESSOR_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
BASE_CONFIG_SHA256 = "4a73d36eff2d76e5d222d3ffac3045236bc0c745cb8ee7ca438cdaf588cb4f03"
PROCESSOR_CONFIG_SHA256 = "5cd452860dc1e9c29dd71cc3cef7f39b338b7a40793f7a260655c2d3568f3661"
BASE_WEIGHTS = {
    "model-00001-of-00005.safetensors": "6aabd8d143cff0295a60b515dfcb5ba6a5b1b5acf7cea1d6c6254ed653d35965",
    "model-00002-of-00005.safetensors": "e346fcf2bf4ebd75853bd17b5744c4158cf621aba195aad7c7e2f1e484c1ae20",
    "model-00003-of-00005.safetensors": "7acad9950402f002825aa41048155ea1a8a2fb5f7a00501154d52828291171c9",
    "model-00004-of-00005.safetensors": "59f3ec7a6983ae2654b7ffe5d79f59d3d523b8a71a0ffc83362816886636ac0a",
    "model-00005-of-00005.safetensors": "5753b1fd57638b70db4882cc8061eedfa28c2748a37414eb4032e20b1bb2e4c1",
}

PROCESSOR_FILES = {
    "generation_config.json": "8469742d1fce0de951c8909b26a2c0c0d8490837ce476efb114da9e0cefc4d44",
    "model.safetensors.index.json": "520b2e05079402e9468a8701d03d1154d14b2599593afb6effa7fb60c1bff070",
    "chat_template.json": "5c72a170d2a4a1a3bc5adad2e689ae28138a9700e5b8c96c0266331e86c0acce",
    "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    "video_preprocessor_config.json": "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
    "tokenizer.json": "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7",
    "config.json": PROCESSOR_CONFIG_SHA256,
    "tokenizer_config.json": "c2da771801886ad9ae98181793ffd3dfb7f1af30f6f7c6a4e15d7dbba52e2399",
    "merges.txt": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
}


def snapshot_identity(directory, *, weights):
    root = Path(directory)
    if not root.is_dir() or not (root / "config.json").is_file():
        raise ValueError(f"stage a complete immutable Alpamayo/processor snapshot: {directory}")
    names = {p.name for p in root.iterdir() if p.is_file() and (p.suffix in {".json", ".jinja", ".txt", ".model"})}
    if weights:
        index = root / "model.safetensors.index.json"
        if not index.is_file():
            raise ValueError("Alpamayo requires its complete indexed five-shard base")
        shards = set(json.loads(index.read_text())["weight_map"].values())
        if shards != set(BASE_WEIGHTS):
            raise ValueError("Alpamayo base shard inventory disagrees with the pinned checkpoint")
        names.update(shards)
    files = {}
    for name in sorted(names):
        if Path(name).name != name or not (root / name).is_file():
            raise ValueError(f"missing or unsafe Alpamayo snapshot file: {name}")
        files[name] = file_sha256(str(root / name))
    expected_config = BASE_CONFIG_SHA256 if weights else PROCESSOR_CONFIG_SHA256
    if files["config.json"] != expected_config:
        raise ValueError("Alpamayo base/processor config differs from its pinned revision")
    if not weights and any(files.get(name) != digest for name, digest in PROCESSOR_FILES.items()):
        raise ValueError("Alpamayo processor files differ from the immutable processor revision")
    if weights and any(files[name] != digest for name, digest in BASE_WEIGHTS.items()):
        raise ValueError("Alpamayo checkpoint weights differ from the immutable base revision")
    return {
        "repository": "nvidia/Alpamayo-R1-10B" if weights else "Qwen/Qwen3-VL-8B-Instruct",
        "revision": BASE_REVISION if weights else PROCESSOR_REVISION,
        "files": files,
        "sha256": canonical_json_sha256(files, context="Alpamayo snapshot"),
    }


def backbone_targets(net):
    """Only language matrices that feed the native action expert's prefix K/V."""
    layers = net.vlm.model.language_model.layers
    if len(layers) != 36:
        raise ValueError("Alpamayo R1 QLoRA requires its pinned 36-layer Qwen3-VL-8B backbone")
    shapes = {
        "self_attn.q_proj": (4096, 4096),
        "self_attn.k_proj": (1024, 4096),
        "self_attn.v_proj": (1024, 4096),
        "self_attn.o_proj": (4096, 4096),
        "mlp.gate_proj": (12288, 4096),
        "mlp.up_proj": (12288, 4096),
        "mlp.down_proj": (4096, 12288),
    }
    # Final-layer Q/O/MLP and final norm produce an output never consumed by the
    # action objective. Adapting them would silently install unreachable trainables.
    return {
        f"vlm.model.language_model.layers.{index}.{name}": shape
        for index in range(36)
        for name, shape in shapes.items()
        if index < 35 or name in {"self_attn.k_proj", "self_attn.v_proj"}
    }


def auxiliary_parameters(net, *, train_action_head):
    if not train_action_head:
        return ()
    selected = tuple(
        name
        for name, _ in net.named_parameters()
        if name.startswith(("expert.", "action_in_proj.", "action_out_proj."))
    )
    if not all(
        any(name.startswith(prefix) for name in selected)
        for prefix in ("expert.", "action_in_proj.", "action_out_proj.")
    ):
        raise ValueError("Alpamayo native expert/action projection components are incomplete")
    return selected
