# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Bounded-memory merged adapter tensor source for the HF export transaction.

Keep whole target tensors through the existing merge math. The bound is one
source tensor, its conversion/merge workspace, and one serialization shard;
an oversized tensor occupies a shard by itself. No HF model or full model.pt is
materialized. Frozen fused experts retain their checkpoint layout on disk.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from collections.abc import Iterator, Mapping
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from cosmos_framework.posttrain.checkpoint.adapters import merged_export_metadata, snapshot_identity
from cosmos_framework.posttrain.checkpoint.conversion import hf
from cosmos_framework.posttrain.checkpoint.io import atomic_write_json
from cosmos_framework.posttrain.config.adapters import AdapterComponentConfig
from cosmos_framework.posttrain.model.adapter_families import adapter_family
from cosmos_framework.posttrain.model.adapters import PosttrainAdapterLinear


class AdapterTensorSource(Mapping[str, torch.Tensor]):
    """Captured portable tensors plus content-verified, held base shard inodes.

    ``probe`` identifies a versioned source descriptor, not a torch.save payload.
    Its base and adapter digests bind the actual bytes. Open/consume/check all
    sources during the materializer's single output-lock hold.
    """

    conversion = "materialize_adapter_shards_to_hf/1"

    def __init__(
        self,
        base: str,
        adapter: str,
        record: dict,
        state: dict,
        architecture: dict,
        output_dtype: str,
        *,
        shard_bytes: int = 256 * 2**20,
    ) -> None:
        if isinstance(shard_bytes, bool) or not isinstance(shard_bytes, int) or shard_bytes < 1:
            raise ValueError("adapter export shard_bytes must be a positive integer")
        self.base, self.adapter = base, adapter
        self.identity, self.state = json.loads(json.dumps(record["identity"])), state
        self.record = {**record, "identity": self.identity}
        self.output_dtype, self.shard_bytes = output_dtype, shard_bytes
        self.base_dtype = getattr(torch, self.identity["base_storage_dtype"])
        self.weight_map = hf._read_weight_map(os.path.join(base, hf.INDEX_FILE))
        for shard in set(self.weight_map.values()):
            if Path(shard).is_absolute() or ".." in Path(shard).parts:
                raise ValueError(f"invalid base weight shard path: {shard!r}")
        self.source_names = {}
        from cosmos_framework.posttrain.model.qwen_vl_hf_execution import canonical_weight_name

        remap = self.identity["resolved"].get("model_execution") == "qwen_vl_hf_ordered_v1"
        if remap:
            self.conversion = "materialize_adapter_shards_to_hf/2-ordered-qwen"
        self.handles = {}
        self.stats = {}
        self.excluded = {}
        self.shapes = {}
        self.sizes = {}
        for shard in sorted(set(self.weight_map.values())):
            header, _ = hf.safetensors_header(os.path.join(base, shard))
            for source_name, entry in header.items():
                name = canonical_weight_name(self.identity["architecture"], source_name) if remap else source_name
                if name in self.source_names:
                    raise ValueError(f"multiple base tensors resolve to the same owner: {name!r}")
                self.source_names[name] = source_name
                if name not in architecture:
                    if self.identity["architecture"] == "qwen3_5" and name.startswith("mtp."):
                        self.excluded[name] = {"shape": entry["shape"], "reason": "unused_source_mtp_head"}
                        continue
                    raise ValueError(f"base tensor {name!r} is absent from the bound export architecture")
                shape, expected = tuple(entry["shape"]), tuple(architecture[name])
                # Stock HF loads native fused expert tensors by transposing the
                # last axes. Export must keep the original checkpoint layout.
                if shape != expected and not (
                    ".mlp.experts." in name and len(shape) == 3 and (shape[0], shape[2], shape[1]) == expected
                ):
                    raise ValueError(f"base tensor {name!r} shape {shape} differs from architecture {expected}")
                if name in self.shapes:
                    raise ValueError(f"duplicate base tensor {name!r}")
                self.shapes[name] = shape
                self.sizes[name] = (
                    math.prod(shape) * (2 if output_dtype == "bfloat16" else 4)
                    if entry["dtype"].startswith("F") or entry["dtype"] == "BF16"
                    else entry["data_offsets"][1] - entry["data_offsets"][0]
                )
        from transformers import AutoConfig

        # HF configuration files may omit fields equal to architecture defaults.
        # Gemma/Gemma2 snapshots omit tie_word_embeddings=True and serialize
        # only the embedding owner. Resolve the same defaults as the stock
        # loader instead of interpreting an absent JSON field as untied.
        config = AutoConfig.from_pretrained(base, local_files_only=True)
        self.tied_embedding = (
            adapter_family(self.identity["architecture"]).embedding
            if config.tie_word_embeddings
            else None
        )
        missing = set(architecture) - self.shapes.keys()
        if self.tied_embedding and self.tied_embedding in self.shapes:
            missing.discard("lm_head.weight")
        if missing:
            raise ValueError(f"base is missing architecture tensors: {sorted(missing)[:10]}")
        self.targets = {row["name"] + ".weight": row for row in self.identity["resolved"]["targets"]}
        self.extras = {row["name"] for row in self.identity["resolved"]["extra_parameters"]}
        for name, row in self.targets.items():
            if self.shapes.get(name) != tuple(row["shape"]):
                raise ValueError(
                    f"base cannot resolve native adapter target {name}; packed/renamed HF conversions are unsupported"
                )
        self.keys_ordered = tuple(sorted(self.shapes.keys() - ({"lm_head.weight"} if self.tied_embedding else set())))
        descriptor = json.dumps(
            {
                "conversion": self.conversion,
                "identity": self.identity,
                "adapter_weights_sha256": record["weights_sha256"],
                "output_dtype": output_dtype,
                "shard_bytes": shard_bytes,
            },
            sort_keys=True,
            allow_nan=False,
        ).encode()
        self.probe = {
            "kind": "adapter_tensor_source",
            "path": os.path.abspath(adapter),
            "sha256": hashlib.sha256(descriptor).hexdigest(),
            "bytes": len(descriptor),
        }

    def validate_metadata(self, metadata: dict | None) -> None:
        # Reuse the same semantic record as the public adapter exporter, using
        # the captured record (never reread a possibly replaced adapter path).
        precision = (metadata or {}).get("deployment_precision", {})
        profile = precision.get("execution_profile") if isinstance(precision, dict) else None
        expected = merged_export_metadata(
            self.adapter,
            self.identity,
            artifact_record=self.record,
            output_dtype=self.output_dtype,
            execution_profile=profile,
        )
        if self.excluded:
            expected["excluded_base_tensors"] = self.excluded
        if metadata != expected:
            raise ValueError("export_metadata differs from the captured adapter tensor source identity")

    def open(self, held: ExitStack) -> dict:
        files = snapshot_identity(self.base, weights=False)["files"]
        index = Path(self.base) / hf.INDEX_FILE
        if index.is_file():
            files[hf.INDEX_FILE] = hashlib.sha256(index.read_bytes()).hexdigest()
        for shard in sorted(set(self.weight_map.values())):
            handle = held.enter_context(open(os.path.join(self.base, shard), "rb"))
            before = os.fstat(handle.fileno())
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if digest != self.identity["base"]["files"].get(shard):
                raise ValueError(f"wrong base identity for adapter merge: shard {shard}")
            self.handles[shard], self.stats[shard] = handle, before
            files[shard] = digest
        actual_base = {"files": files, "sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}
        if actual_base != self.identity["base"]:
            raise ValueError("wrong base identity for adapter merge; shard/config/index inventory differs")
        self.assert_unchanged()
        if self.tied_embedding and "lm_head.weight" in self.shapes:
            if not torch.equal(self._tensor(self.tied_embedding), self._tensor("lm_head.weight")):
                raise ValueError("base declares tied embeddings but embedding/head tensors disagree")
        return self.probe

    def assert_unchanged(self) -> None:
        for shard, handle in self.handles.items():
            before, after = self.stats[shard], os.fstat(handle.fileno())
            if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise ValueError(f"base shard changed during adapter export: {shard}")
        if snapshot_identity(self.base, weights=False) != self.identity["processor"]:
            raise ValueError("base config/processor/tokenizer/template changed during adapter export")
        # The index is not part of processor identity. Its tensor ownership must
        # still name the captured shard set even if replaced during a long run.
        index = Path(self.base) / hf.INDEX_FILE
        if index.is_file():
            digest = hashlib.sha256(index.read_bytes()).hexdigest()
            if digest != self.identity["base"]["files"].get(hf.INDEX_FILE):
                raise ValueError("base weight index changed during adapter export")
        elif hf.INDEX_FILE in self.identity["base"]["files"]:
            raise ValueError("base weight index disappeared during adapter export")

    def __len__(self) -> int:
        return len(self.keys_ordered)

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys_ordered)

    def __getitem__(self, name: str) -> torch.Tensor:
        if name not in self.keys_ordered:
            raise KeyError(name)
        return self._tensor(name)

    @torch.no_grad()
    def _tensor(self, name: str) -> torch.Tensor:
        if not self.handles:
            raise RuntimeError("adapter tensor source must be opened under the export lock")
        # A new mapping per tensor releases already touched shard pages. Holding
        # a safe_open for an entire 50GB shard would defeat the resident bound.
        path = f"/proc/self/fd/{self.handles[self.weight_map[self.source_names[name]]].fileno()}"
        with safe_open(path, framework="pt", device="cpu") as handle:
            original = handle.get_tensor(self.source_names[name])
            weight = original.to(self.base_dtype).float() if original.is_floating_point() else original.clone()
            # FP32 originals may alias the mapping. A clone gives the returned
            # tensor independent storage and lets that mapping close promptly.
            if weight.data_ptr() == original.data_ptr():
                weight = weight.clone()
        del original
        if name in self.extras:
            weight = self.state[name].float()
        row = self.targets.get(name)
        if row is not None:
            settings = AdapterComponentConfig(component=row["component"], targets=[row["name"]], **row["settings"])
            linear = torch.nn.Linear(row["shape"][1], row["shape"][0], bias=False, device="meta", dtype=torch.float32)
            layer = PosttrainAdapterLinear(linear, settings, initialize=False)
            weight = layer.merged_state_weight({**self.state, name: weight}, row["owner"])
        if self.output_dtype == "bfloat16" and weight.is_floating_point():
            weight = weight.bfloat16()
        return weight.contiguous()

    def build(self, out: str) -> None:
        os.makedirs(out, exist_ok=False)
        shutil.copyfile(os.path.join(self.base, "config.json"), os.path.join(out, "config.json"))
        # Plan from shapes/dtypes so an oversized tensor is never loaded beside
        # an already full serialization buffer. Nonfloating dtype is read from
        # the source header; it is preserved exactly.
        plans, current, size = [], [], 0
        for name in self:
            nbytes = self.sizes[name]
            if current and size + nbytes > self.shard_bytes:
                plans.append(current)
                current, size = [], 0
            current.append(name)
            size += nbytes
        if current:
            plans.append(current)
        weight_map, total = {}, 0
        for i, names in enumerate(plans, 1):
            shard = f"model-{i:05d}-of-{len(plans):05d}.safetensors"
            tensors = {name: self[name] for name in names}
            save_file(tensors, os.path.join(out, shard), metadata={"format": "pt"})
            total += sum(t.numel() * t.element_size() for t in tensors.values())
            weight_map.update({name: shard for name in names})
            del tensors
        atomic_write_json(
            os.path.join(out, hf.INDEX_FILE), {"metadata": {"total_size": total}, "weight_map": weight_map}
        )

    def verify(self, out: str, *, unified_chat_template: bool = False) -> dict:
        result = hf.verify_with_torch(
            out,
            self.base,
            self.adapter,
            compare_base=False,
            unified_chat_template=unified_chat_template,
            source_state=self,
        )
        self.assert_unchanged()
        return result
