# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import contextlib
import json
import re
from pathlib import Path
from typing import Any

import attrs
import hydra
import omegaconf
import torch
import torch.distributed.checkpoint as dcp
import transformers
from torch.distributed.checkpoint.filesystem import FileSystemReader
from torch.distributed.checkpoint.hf_storage import (
    CUSTOM_METADATA_KEY,
    SAVED_OFFSETS_KEY,
    HuggingFaceStorageReader,
    _HFStorageInfo,
)
from torch.distributed.checkpoint.metadata import (
    STORAGE_TYPES,
    ChunkStorageMetadata,
    Metadata,
    StorageMeta,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import MetadataIndex
from torch.distributed.checkpoint.state_dict import get_model_state_dict
from typing_extensions import TYPE_CHECKING, assert_never

from cosmos_framework.configs.base.defaults.compile import CompileConfig
from cosmos_framework.configs.base.defaults.parallelism import ParallelismConfig
from cosmos_framework.inference.common.args import CheckpointType
from cosmos_framework.inference.common.checkpoints import register_checkpoints
from cosmos_framework.inference.common.config import structure_config, undo_config_dict_replacements, unstructure_config
from cosmos_framework.inference.common.public_model_config import (
    build_public_model_config,
    model_config_uses_public_aliases,
    restore_model_config_from_public_model_config,
)
from cosmos_framework.utils import misc
from cosmos_framework.utils.flags import SMOKE

if TYPE_CHECKING:
    from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel


# Resolve to the release-tree root so relative-path checkpoint config entries
# (e.g. `cosmos_framework/model/generator/vlm/qwen3_vl/configs/Qwen3-VL-8B-Instruct.json`) load
# correctly under contextlib.chdir(_ROOT_DIR) in __init__. In the original
# cosmos3 release the cosmos3 package lives at the tree root, so parents[1] is
# the release root. In the cosmos_training release the package lives at
# cosmos_framework/inference/, so the release root is one level higher (parents[2]).
try:
    import cosmos_framework.model.generator  # noqa: F401

    _ROOT_DIR = Path(__file__).parents[2].absolute()
except ImportError:
    _ROOT_DIR = Path(__file__).parents[1].absolute()


_DIFFUSERS_ROOT_INDEX = "model.safetensors.index.json"
_DIFFUSERS_MODEL_INDEX = "model_index.json"
_DIFFUSERS_DROP_WEIGHT_PATH_RES: tuple[re.Pattern[str], ...] = (re.compile(r"^(?!transformer/|vision_encoder/)"),)
_DIFFUSERS_DROP_KEY_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(?:feature_extractor|image_processor|scheduler|sound_tokenizer|text_encoder|tokenizer|vae)\."),
)
_DIFFUSERS_KEY_MAPPING_RES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^transformer\."), ""),
    (re.compile(r"^vision_encoder\."), ""),
    (re.compile(r"^model\.net\."), ""),
    (re.compile(r"^action_proj_in\."), "action2llm."),
    (re.compile(r"^action_proj_out\."), "llm2action."),
    (re.compile(r"^audio_proj_in\."), "sound2llm."),
    (re.compile(r"^audio_proj_out\."), "llm2sound."),
    (re.compile(r"^audio_modality_embed$"), "sound_modality_embed"),
    (re.compile(r"^proj_in\."), "vae2llm."),
    (re.compile(r"^proj_out\."), "llm2vae."),
    (re.compile(r"^time_embedder\.linear_1\."), "time_embedder.mlp.0."),
    (re.compile(r"^time_embedder\.linear_2\."), "time_embedder.mlp.2."),
    (re.compile(r"\.self_attn\.to_q\."), ".self_attn.q_proj."),
    (re.compile(r"\.self_attn\.to_k\."), ".self_attn.k_proj."),
    (re.compile(r"\.self_attn\.to_v\."), ".self_attn.v_proj."),
    (re.compile(r"\.self_attn\.to_out\."), ".self_attn.o_proj."),
    (re.compile(r"\.self_attn\.norm_q\."), ".self_attn.q_norm."),
    (re.compile(r"\.self_attn\.norm_k\."), ".self_attn.k_norm."),
    (re.compile(r"\.self_attn\.add_q_proj\."), ".self_attn.q_proj_moe_gen."),
    (re.compile(r"\.self_attn\.add_k_proj\."), ".self_attn.k_proj_moe_gen."),
    (re.compile(r"\.self_attn\.add_v_proj\."), ".self_attn.v_proj_moe_gen."),
    (re.compile(r"\.self_attn\.to_add_out\."), ".self_attn.o_proj_moe_gen."),
    (re.compile(r"\.self_attn\.norm_added_q\."), ".self_attn.q_norm_moe_gen."),
    (re.compile(r"\.self_attn\.norm_added_k\."), ".self_attn.k_norm_moe_gen."),
    (re.compile(r"^model\.lm_head\."), "language_model.lm_head."),
    (re.compile(r"^lm_head\."), "language_model.lm_head."),
    (re.compile(r"^model\.visual\."), "language_model.visual."),
    (re.compile(r"^visual\."), "language_model.visual."),
    (
        re.compile(r"^(blocks\.|deepstack_merger_list\.|merger\.|patch_embed\.|pos_embed\.)(.*)$"),
        r"language_model.visual.\1\2",
    ),
    (
        re.compile(
            r"^language_model\.(?!model\.|lm_head\.|visual\.)(embed_tokens\.|layers\.|norm(?:_moe_gen)?\.)(.*)$"
        ),
        r"language_model.model.\1\2",
    ),
    (
        re.compile(r"^model\.(embed_tokens\.|layers\.|norm(?:_moe_gen)?\.)(.*)$"),
        r"language_model.model.\1\2",
    ),
    (
        re.compile(r"^(embed_tokens\.|layers\.|norm(?:_moe_gen)?\.)(.*)$"),
        r"language_model.model.\1\2",
    ),
)
_DIFFUSERS_NET_KEY_PREFIXES: tuple[str, ...] = (
    "action2llm.",
    "action_pos_embed.",
    "language_model.",
    "latent_pos_embed.",
    "llm2action.",
    "llm2sound.",
    "llm2vae.",
    "sound2llm.",
    "time_embedder.",
    "vae2llm.",
)
_DIFFUSERS_NET_KEYS: frozenset[str] = frozenset(
    {
        "action_modality_embed",
        "latent_pos_embed",
        "sound_modality_embed",
    }
)


def _should_drop_diffusers_weight_path(path: str) -> bool:
    path = path.replace("\\", "/")
    return bool(path) and any(pattern.search(path) is not None for pattern in _DIFFUSERS_DROP_WEIGHT_PATH_RES)


def _should_drop_diffusers_key(name: str) -> bool:
    return any(pattern.search(name) is not None for pattern in _DIFFUSERS_DROP_KEY_RES)


def _is_diffusers_model_weight_path(path: str) -> bool:
    return bool(path) and not _should_drop_diffusers_weight_path(path)


def _apply_diffusers_key_mapping(name: str) -> str:
    for pattern, replacement in _DIFFUSERS_KEY_MAPPING_RES:
        name = pattern.sub(replacement, name)
    return name


def _is_loadable_diffusers_net_key(name: str) -> bool:
    return name in _DIFFUSERS_NET_KEYS or name.startswith(_DIFFUSERS_NET_KEY_PREFIXES)


def _diffusers_to_net_key(name: str, weight_path: str = "") -> str | None:
    """Rename a diffusers checkpoint key to its OmniMoTModel.net subtree key.

    Returns None for non-Cosmos model components in a full diffusers pipeline.
    """
    if _should_drop_diffusers_weight_path(weight_path) or _should_drop_diffusers_key(name):
        return None

    net_key = _apply_diffusers_key_mapping(name)
    if _should_drop_diffusers_key(net_key) or not _is_loadable_diffusers_net_key(net_key):
        return None
    return net_key


def _read_safetensors_index(index_path: Path) -> dict[str, str]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"{index_path} does not contain a safetensors weight_map.")

    result: dict[str, str] = {}
    for key, value in weight_map.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise TypeError(f"{index_path} weight_map must contain string keys and values.")
        result[key] = value
    return result


def _diffusers_weight_map(checkpoint_path: Path) -> dict[str, str]:
    index_path = checkpoint_path / _DIFFUSERS_ROOT_INDEX
    if not index_path.exists():
        raise FileNotFoundError(f"Diffusers safetensors index not found: {index_path}")
    return _read_safetensors_index(index_path)


def _diffusers_files_to_keys(weight_map: dict[str, str]) -> dict[str, list[str]]:
    files_to_keys: dict[str, list[str]] = {}
    for diff_key, rel_path in weight_map.items():
        if _should_drop_diffusers_weight_path(rel_path):
            continue
        files_to_keys.setdefault(rel_path, []).append(diff_key)
    return files_to_keys


def _is_diffusers_checkpoint(checkpoint_path: Path) -> bool:
    index_path = checkpoint_path / _DIFFUSERS_ROOT_INDEX
    if not index_path.exists():
        return False
    if (checkpoint_path / _DIFFUSERS_MODEL_INDEX).exists():
        return True
    return any(_is_diffusers_model_weight_path(path) for path in _read_safetensors_index(index_path).values())


def _normalize_diffusers_target_key(name: str) -> str:
    return name.removeprefix("model.net.").replace("_orig_mod.", "").replace("_checkpoint_wrapped_module.", "")


class _DiffusersHuggingFaceStorageReader(HuggingFaceStorageReader):
    """Hugging Face safetensors reader that follows diffusers' root weight map."""

    def __init__(self, checkpoint_path: Path) -> None:
        super().__init__(str(checkpoint_path))
        self.checkpoint_path = checkpoint_path
        self.files_to_keys = _diffusers_files_to_keys(_diffusers_weight_map(checkpoint_path))

    def read_metadata(self) -> Metadata:
        from safetensors import safe_open
        from safetensors.torch import _getdtype

        state_dict_metadata: dict[str, STORAGE_TYPES] = {}
        storage_data: dict[MetadataIndex, _HFStorageInfo] = {}

        for rel_path, diff_keys in sorted(self.files_to_keys.items()):
            shard_path = self.checkpoint_path / rel_path
            if not shard_path.exists():
                raise FileNotFoundError(f"Diffusers checkpoint shard not found: {shard_path}")

            with safe_open(str(shard_path), framework="pt") as f:
                shard_keys = set(f.keys())
                missing_keys = sorted(set(diff_keys) - shard_keys)
                if missing_keys:
                    raise KeyError(
                        f"Diffusers checkpoint shard {shard_path} is missing {len(missing_keys)} "
                        f"indexed tensor(s). First up to 10: {missing_keys[:10]}"
                    )

                extra_metadata = f.metadata()
                dcp_sharding_info: dict[str, Any] | None = None
                if extra_metadata and extra_metadata.get(CUSTOM_METADATA_KEY):
                    dcp_sharding_info = json.loads(extra_metadata[CUSTOM_METADATA_KEY])

                for diff_key in sorted(diff_keys):
                    tensor_slice = f.get_slice(diff_key)
                    shape = tensor_slice.get_shape()
                    dtype = _getdtype(tensor_slice.get_dtype())
                    offset = dcp_sharding_info[diff_key][SAVED_OFFSETS_KEY] if dcp_sharding_info else [0] * len(shape)
                    chunk = ChunkStorageMetadata(
                        offsets=torch.Size(offset),
                        sizes=torch.Size(shape),
                    )

                    if diff_key not in state_dict_metadata:
                        state_dict_metadata[diff_key] = TensorStorageMetadata(
                            properties=TensorProperties(dtype=dtype),
                            size=torch.Size(saved + start for saved, start in zip(shape, offset)),
                            chunks=[chunk],
                        )
                    else:
                        existing_metadata = state_dict_metadata[diff_key]
                        assert isinstance(existing_metadata, TensorStorageMetadata)
                        existing_metadata.chunks.append(chunk)
                        size = list(existing_metadata.size)
                        for i, dim_size in enumerate(size):
                            size[i] = max(dim_size, shape[i] + offset[i])
                        existing_metadata.size = torch.Size(size)

                    metadata_index = MetadataIndex(fqn=diff_key, offset=offset)
                    storage_data[metadata_index] = _HFStorageInfo(
                        relative_path=str(shard_path),
                        shape=torch.Size(shape),
                        dtype=dtype,
                    )

        metadata = Metadata(
            state_dict_metadata=state_dict_metadata,
            storage_data=storage_data,
        )
        storage_meta = metadata.storage_meta
        if storage_meta is None:
            storage_meta = StorageMeta()
            metadata.storage_meta = storage_meta
        storage_meta.load_id = self.load_id
        return metadata


class _DiffusersLoadPlanner(dcp.DefaultLoadPlanner):
    """Remap diffusers source keys onto the OmniMoTModel.net state dict for DCP load."""

    def __init__(self, checkpoint_path: Path) -> None:
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.weight_map = _diffusers_weight_map(checkpoint_path)
        self.files_to_keys = _diffusers_files_to_keys(self.weight_map)
        self.has_vision_weights = any(rel_path.startswith("vision_encoder/") for rel_path in self.files_to_keys)

    def set_up_planner(
        self,
        state_dict: dict[str, Any],
        metadata: Metadata | None = None,
        is_coordinator: bool = False,
    ) -> None:
        target_state_dict = self._normalize_target_state_dict(state_dict)
        remapped_state_dict, loaded_keys = self._build_remapped_state_dict(target_state_dict)

        missing_keys = set(target_state_dict) - loaded_keys
        if not self.has_vision_weights:
            missing_keys = {key for key in missing_keys if not key.startswith("language_model.visual.")}
        # Task-specialized checkpoints (e.g. Text2Image, Image2Video) omit the
        # optional generative-modality projection heads (action, sound). They
        # are unused for those tasks, so tolerate their absence the same way
        # vision weights are tolerated when the checkpoint provides none of them.
        for modality_prefixes in (
            ("action2llm.", "llm2action.", "action_modality_embed"),
            ("sound2llm.", "llm2sound.", "sound_modality_embed"),
        ):
            if not any(key.startswith(modality_prefixes) for key in loaded_keys):
                missing_keys = {key for key in missing_keys if not key.startswith(modality_prefixes)}
        if missing_keys:
            sample = sorted(missing_keys)[:10]
            raise ValueError(
                f"Diffusers checkpoint at {self.checkpoint_path} did not provide {len(missing_keys)} "
                f"required model tensor(s). First up to 10: {sample}"
            )

        super().set_up_planner(
            state_dict=remapped_state_dict,
            metadata=metadata,
            is_coordinator=is_coordinator,
        )

    @staticmethod
    def _normalize_target_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
        target_state_dict: dict[str, Any] = {}
        for name, tensor in state_dict.items():
            net_key = _normalize_diffusers_target_key(name)
            if net_key in target_state_dict:
                raise KeyError(f"Multiple target model keys normalize to {net_key!r}.")
            target_state_dict[net_key] = tensor
        return target_state_dict

    def _build_remapped_state_dict(self, target_state_dict: dict[str, Any]) -> tuple[dict[str, Any], set[str]]:
        remapped_state_dict: dict[str, Any] = {}
        loaded_keys: set[str] = set()
        for diff_key, rel_path in sorted(self.weight_map.items()):
            net_key = _diffusers_to_net_key(diff_key, rel_path)
            if net_key is None:
                if _is_diffusers_model_weight_path(rel_path):
                    raise KeyError(f"Diffusers model key {diff_key!r} from {rel_path!r} has no Cosmos3 mapping.")
                continue
            target_tensor = target_state_dict.get(net_key)
            if target_tensor is None:
                continue
            if net_key in loaded_keys:
                raise KeyError(f"Multiple diffusers keys map to target model key {net_key!r}.")
            remapped_state_dict[diff_key] = target_tensor
            loaded_keys.add(net_key)
        return remapped_state_dict, loaded_keys


class Cosmos3OmniConfig(transformers.PretrainedConfig):
    model_type = "cosmos3_omni"

    def __init__(self, model: dict | None = None, **kwargs):
        self._use_public_model_config = False
        if model is not None and model_config_uses_public_aliases(model):
            model = restore_model_config_from_public_model_config(model)
            self._use_public_model_config = True
        if model is not None:
            model = undo_config_dict_replacements(model)
        self.model = model or {}

        super().__init__(**kwargs)

        self.auto_map = {
            "AutoConfig": "cosmos3.model.Cosmos3OmniConfig",
            "AutoModel": "cosmos3.model.Cosmos3OmniModel",
        }

    def to_dict(self) -> dict[str, Any]:
        output = super().to_dict()
        output.pop("_use_public_model_config", None)
        if self._use_public_model_config:
            output["model"] = build_public_model_config(self.model)
        return output

    @property
    def parallelism(self) -> dict:
        return self.model.get("config", {}).get("parallelism", {})

    @parallelism.setter
    def parallelism(self, value: dict | None):
        if value is None:
            return
        self.model.setdefault("config", {})["parallelism"] = unstructure_config(ParallelismConfig(**value))

    @property
    def compile(self) -> dict:
        return self.model.get("config", {}).get("compile", {})

    @compile.setter
    def compile(self, value: dict | None):
        if value is None:
            return
        self.model.setdefault("config", {})["compile"] = unstructure_config(CompileConfig(**value))


class Cosmos3OmniModel(transformers.PreTrainedModel):
    config_class = Cosmos3OmniConfig  # type: ignore

    def __init__(self, config: Cosmos3OmniConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        self.before_load_model()
        model_dict: "OmniMoTModel" = structure_config(config.model, omegaconf.DictConfig)

        # Disable training-only features
        model_dict.config.ema.enabled = False
        model_dict.config.activation_checkpointing.mode = "none"
        if SMOKE:
            # Minimize model size for smoke test
            vlm_dict = model_dict.config.vlm_config.model_instance
            assert vlm_dict is not None
            with omegaconf.open_dict(vlm_dict.config):
                vlm_dict.config.text_config_overrides = {"num_hidden_layers": 2, "num_window_layers": 2}

        # The model loads some files by relative path 'cosmos3/...'
        with contextlib.chdir(_ROOT_DIR):
            self.model: "OmniMoTModel" = hydra.utils.instantiate(model_dict)
        self.after_load_model(self.model)

    @classmethod
    def from_pretrained_dcp(
        cls,
        checkpoint_path: Path,
        config: Cosmos3OmniConfig | None = None,
        parallelism_config: ParallelismConfig | None = None,
        compile_config: CompileConfig | None = None,
    ):
        if config is None:
            config = Cosmos3OmniConfig.from_pretrained(checkpoint_path)
        if parallelism_config is None:
            parallelism_config = ParallelismConfig()
        if compile_config is None:
            compile_config = CompileConfig()
        config.parallelism = attrs.asdict(parallelism_config)
        config.compile = attrs.asdict(compile_config)
        model = cls(config)
        checkpoint_type = CheckpointType.from_path(checkpoint_path)
        match checkpoint_type:
            case CheckpointType.DCP:
                state_dict = get_model_state_dict(model.model)
                storage_reader = FileSystemReader(str(checkpoint_path))
            case CheckpointType.HF:
                if _is_diffusers_checkpoint(checkpoint_path):
                    state_dict = get_model_state_dict(model.model.net)
                    dcp.load(
                        state_dict=state_dict,
                        storage_reader=_DiffusersHuggingFaceStorageReader(checkpoint_path),
                        planner=_DiffusersLoadPlanner(checkpoint_path),
                    )
                    return model
                state_dict = get_model_state_dict(model)
                storage_reader = HuggingFaceStorageReader(str(checkpoint_path))
            case _:
                assert_never(checkpoint_type)
        dcp.load(state_dict=state_dict, storage_reader=storage_reader)
        return model

    @classmethod
    def before_load_model(cls):
        # Disable duck shapes, which triggers recompile.
        misc.set_torch_compile_options(use_duck_shape=False)

        register_checkpoints()

    @classmethod
    def after_load_model(cls, model: "OmniMoTModel"):
        pass
