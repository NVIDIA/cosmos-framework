# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Map-style PyTorch datasets that never resolve remote storage."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from torch.utils.data import Dataset

from cosmos_framework.data.generator.local_simulation.core import (
    FORMAT_VERSION,
    LocalByteRangeBackend,
    decode_config,
    local_path,
    sample_seed,
)
from cosmos_framework.utils.lazy_config import instantiate


def load_bundle_tokenizer(directory: str) -> Any:
    """Load the production Qwen tokenizer without HF or Cosmos cache access."""
    from transformers import Qwen2Tokenizer

    from cosmos_framework.data.generator.processors import LLMTokenizerProcessor

    return LLMTokenizerProcessor(Qwen2Tokenizer.from_pretrained(directory, local_files_only=True))


def tokenizer_config(root: Path) -> dict[str, Any]:
    return {
        "_target_": load_bundle_tokenizer,
        "directory": str(root / "resources" / "tokenizer"),
    }


def _model_constructors(model: str) -> dict[str, type[Any]]:
    """Keep the production constructor set in code, never in bundle metadata."""
    classes: list[type[Any]]
    if model == "transfer":
        classes = []
    elif model == "bimanual":
        from cosmos_framework.data.generator.action.canonical_lance_action_datasets import (
            CanonicalLanceAgiBotDataset,
            CanonicalLanceBehavior1KDataset,
            CanonicalLanceembodiment_aDataset,
            CanonicalLanceYAMDataset,
        )
        from cosmos_framework.data.generator.action.canonical_lance_dataset import CanonicalLanceActionDatasetSpec
        from cosmos_framework.data.generator.action.lance.lance_camera_dataset import LanceCameraDataset
        from cosmos_framework.data.generator.lance.lance_action_source import (
            ActionSamplesColumns,
            ActionSpansColumns,
            LanceActionDatasetSpec,
        )

        classes = [
            CanonicalLanceAgiBotDataset,
            CanonicalLanceBehavior1KDataset,
            CanonicalLanceembodiment_aDataset,
            CanonicalLanceYAMDataset,
            LanceCameraDataset,
            CanonicalLanceActionDatasetSpec,
            LanceActionDatasetSpec,
            ActionSamplesColumns,
            ActionSpansColumns,
        ]
    elif model == "auto":
        from cosmos_framework.data.generator.multiview.rig_sampler import ExplicitRigSampler
        from cosmos_framework.model.generator.tokenizers.lidar.range_projection import LidarRangeProjectionConfig

        classes = [ExplicitRigSampler, LidarRangeProjectionConfig]
    else:
        raise ValueError(f"Unsupported local simulation model: {model!r}")
    constructors = {f"{target.__module__}.{target.__qualname__}": target for target in classes}
    if model == "bimanual":
        # Offline bundles produced by the original exporter retain constructor
        # IDs under projects.cosmos3.cosmos3.datasets. Bind that closed set to
        # the corresponding Framework implementations; never import a class
        # named by bundle metadata.
        framework_prefix = "cosmos_framework.data.generator."
        legacy_prefix = "projects.cosmos3.cosmos3.datasets."
        constructors.update(
            {
                legacy_prefix + name.removeprefix(framework_prefix): target
                for name, target in list(constructors.items())
                if name.startswith(framework_prefix)
            }
        )
    if model == "auto":
        # Previously exported bundles retain these two literal constructor IDs.
        # They are aliases to trusted Framework classes, never dynamic imports.
        constructors.update(
            {
                "cosmos_framework.data.generator.multiview.rig_sampler.ExplicitRigSampler": classes[0],
                "cosmos_framework.model.generator.tokenizers.lidar.range_projection.LidarRangeProjectionConfig": classes[
                    1
                ],
            }
        )
    return constructors


def resolve_bundle_paths(value: Any, root: Path) -> Any:
    if isinstance(value, str) and value.startswith("bundle://"):
        return str(local_path(root, value.removeprefix("bundle://")))
    if isinstance(value, list):
        return [resolve_bundle_paths(item, root) for item in value]
    if isinstance(value, tuple):
        return tuple(resolve_bundle_paths(item, root) for item in value)
    if isinstance(value, dict):
        return {key: resolve_bundle_paths(item, root) for key, item in value.items()}
    return value


class SampleRejectedError(ValueError):
    """A production guard rejected this stochastic augmentation draw."""


class LocalSimulationDataset(Dataset[dict[str, Any]]):
    """Read one encoded bundle entry, then apply fresh production transforms.

    Call ``set_epoch`` before constructing workers for each epoch. Sample RNGs
    depend on seed, epoch, and index, independent of DataLoader worker count.
    Missing assets fail immediately; no source URI is opened by this class.
    """

    root: Path
    manifest: dict[str, Any]
    records: list[dict[str, Any]]
    seed: int
    epoch: int
    _pid: int | None
    _pipelines: dict[str, Any]

    def __init__(
        self,
        bundle_path: str | Path,
        seed: int = 42,
        *,
        allow_incomplete: bool = False,
        model: str | None = None,
        stratum: str | None = None,
        transform_overrides: dict[str, Any] | None = None,
        max_num_frames: int | None = None,
    ) -> None:
        self.root = Path(bundle_path).expanduser().resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest.get("format_version") != FORMAT_VERSION:
            raise ValueError("Unsupported local simulation bundle version.")
        if not allow_incomplete and self.manifest.get("complete") is not True:
            raise ValueError("Bundle export has not completed validation.")
        self.records = json.loads((self.root / "samples.json").read_text())
        if len(self.records) != self.manifest["sample_count"]:
            raise ValueError("Sample count does not match bundle manifest.")
        if model is not None and self.manifest["model"] != model:
            raise ValueError(f"Expected a {model} bundle, got {self.manifest['model']}.")
        if stratum is not None:
            self.records = [record for record in self.records if record["stratum"] == stratum]
            if not self.records:
                raise ValueError(f"Bundle has no samples for stratum {stratum}.")
        if seed < 0:
            raise ValueError("seed must be nonnegative.")
        self.seed = seed
        # Recipe-level data policy (e.g. a stage's caption dropout) applied over every stratum's transforms.
        overrides = (
            OmegaConf.to_container(transform_overrides, resolve=True)
            if OmegaConf.is_config(transform_overrides)
            else transform_overrides
        )
        self.transform_overrides = dict(overrides or {})
        if self.manifest["model"] == "auto":
            # The resume digest must identify the effective stage policy, not
            # only the original captured final-stage manifest on disk.
            self.manifest["local_auto_transform_overrides"] = copy.deepcopy(self.transform_overrides)
        if max_num_frames is not None and (max_num_frames < 5 or (max_num_frames - 1) % 4):
            raise ValueError("max_num_frames must be at least five and of the form 1+4n.")
        self.max_num_frames = max_num_frames
        self.epoch = 0
        self._pid = None
        self._pipelines = {}

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_pid"] = None
        state["_pipelines"] = {}
        return state

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be nonnegative.")
        self.epoch = epoch

    def _pipeline(self, stratum: str) -> Any:
        if self._pid != os.getpid():
            self._pipelines = {}
            self._pid = os.getpid()
        if stratum in self._pipelines:
            return self._pipelines[stratum]
        model = self.manifest["model"]
        captured = copy.deepcopy(self.manifest["transforms"][stratum])
        unknown = set(self.transform_overrides) - set(captured)
        if unknown:
            raise ValueError(f"Transform overrides name fields the bundle does not define: {sorted(unknown)}")
        captured.update(copy.deepcopy(self.transform_overrides))
        captured.pop("tokenizer_config", None)
        # The offline pipeline type is selected in code, not by this descriptor.
        captured.pop("_target_", None)
        if model == "auto":
            captured.pop("dataset_catalog", None)
            captured.pop("recipe_datacollection_name", None)
            captured.pop("augmentations_only", None)
        config = decode_config(resolve_bundle_paths(captured, self.root), constructors=_model_constructors(model))
        config["tokenizer_config"] = tokenizer_config(self.root)
        pipeline: Any
        if model == "transfer":
            from cosmos_framework.data.generator.indexed_transfer_webdataset_loader import IndexedTransferWebDataset

            # The shared pipeline needs only these data policy attributes. Its
            # normal constructor resolves an S3 index, which is export-only.
            pipeline = object.__new__(IndexedTransferWebDataset)
            for key, value in config.items():
                setattr(pipeline, key, value)
            pipeline._augmentors = {}
            pipeline._shared_augmentors = {}
        elif model == "bimanual":
            from cosmos_framework.data.generator.local_simulation.bimanual import LocalBimanualPipeline

            entry = config.pop("entry")
            entry["dataset"]["creds_path"] = str(self.root / "unused-offline-credentials")
            child = instantiate(entry["dataset"])
            config["global_asinh_stats_path"] = str(self.root / "resources" / "global_asinh.json")
            pipeline = LocalBimanualPipeline(child, config)
        elif model == "auto":
            from cosmos_framework.data.generator.multiview.local_pipeline import make_auto_augmentations

            pipeline = make_auto_augmentations({key: instantiate(value) for key, value in config.items()})
        else:
            raise ValueError(f"Unsupported local simulation model: {model!r}")
        self._pipelines[stratum] = pipeline
        return pipeline

    def materialize(self, record: dict[str, Any], draw_seed: int) -> dict[str, Any]:
        """Apply the model's production transforms to one captured local record."""
        from cosmos_framework.data.generator.lance.sample_rng import isolated_sample_rng

        pipeline = self._pipeline(record["stratum"])
        with isolated_sample_rng(draw_seed):
            if self.manifest["model"] == "transfer":
                import random

                from cosmos_framework.data.generator.indexed_transfer_webdataset_loader import (
                    _source_draw_seed,
                    transfer_sample_from_payloads,
                )

                available = list(record["assets"])
                modalities = [
                    name for name in pipeline.modality_weights if name in {"edge", "blur"} or name in available
                ]
                modality = random.Random(_source_draw_seed(draw_seed, "control")).choices(
                    modalities, weights=[pipeline.modality_weights[name] for name in modalities], k=1
                )[0]
                rgb = local_path(self.root, record["assets"]["rgb"]).read_bytes()
                control = (
                    local_path(self.root, record["assets"][modality]).read_bytes() if modality in available else None
                )
                raw = transfer_sample_from_payloads(
                    record["entry"], pipeline.caption_key, pipeline.dataset_name, modality, rgb, control
                )
                bucket = int(record["stratum"])
                if self.max_num_frames is not None:
                    bucket = min(bucket, 1 + (self.max_num_frames - 1) // 4)
                sample = pipeline._augment_sample(raw, bucket, draw_seed)
            elif self.manifest["model"] == "bimanual":
                from cosmos_framework.data.generator.action.lance.lance_action_dataset import LanceActionSampleRequest
                from cosmos_framework.data.generator.lance.action_window_index import ActionWindowLocation

                request_data = copy.deepcopy(record["request"])
                request_data["location"] = ActionWindowLocation(**request_data["location"])
                request_data["source_frame_indices"] = tuple(request_data["source_frame_indices"])
                request = replace(LanceActionSampleRequest(**request_data), aux_seed=draw_seed)
                # Apply the recipe tier before RGB decoding as well as before
                # transforms, caption metadata and sequence-plan construction.
                resolution = self.transform_overrides.get("resolution")
                if resolution is not None:
                    request = replace(request, resolution=resolution)
                backend = LocalByteRangeBackend(self.root, record["assets"])
                sample = pipeline.dataset.materialize_local_sample(request, record["source_rows"], backend)
                sample = pipeline.transform_resolved_sample(sample, request)
                sample["_action_family"] = record["stratum"]
                sample["action_sampler_family"] = record["stratum"]
            elif self.manifest["model"] == "auto":
                from cosmos_framework.data.generator.indexed_transfer_webdataset_loader import _IndexedTarSampleURL
                from cosmos_framework.data.generator.lance.augment_timing import iter_augmented_with_timing
                from cosmos_framework.data.generator.lance.sample_rng import LANCE_SAMPLE_RNG_SEED_KEY

                raw = {
                    key: local_path(self.root, asset["path"]).read_bytes() for key, asset in record["assets"].items()
                }
                raw = {
                    key.replace("/", "_"): json.loads(payload)
                    if record["assets"][key]["name"].endswith(".json")
                    else payload
                    for key, payload in raw.items()
                }
                raw.update(
                    __key__=record["identity"],
                    __url__=_IndexedTarSampleURL(record["source_uri"], record["identity"], "16,9"),
                    source_dataset_name="mads_large_v1",
                )
                raw[LANCE_SAMPLE_RNG_SEED_KEY] = draw_seed
                sample = next(iter_augmented_with_timing([raw], pipeline), None)
            else:
                raise ValueError(f"Unsupported local simulation model: {self.manifest['model']!r}")
        if sample is None:
            raise SampleRejectedError(f"Production transforms rejected local record {record['identity']}")
        sample["local_sample_id"] = record["identity"]
        sample["local_epoch"] = self.epoch
        return sample

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        sample, _ = self.materialize_index(self.records[index], index)
        return sample

    def materialize_index(self, record: dict[str, Any], index: int) -> tuple[dict[str, Any], int]:
        """Retry dropped windows, retaining the production conditional draw distribution."""
        seed = sample_seed(self.seed, self.epoch, index)
        for attempt in range(32):
            draw_seed = seed if attempt == 0 else sample_seed(seed, self.epoch, attempt)
            try:
                return self.materialize(record, draw_seed), draw_seed
            except SampleRejectedError:
                continue
        raise SampleRejectedError(f"Production guards rejected 32 draws of local record {record['identity']}")
