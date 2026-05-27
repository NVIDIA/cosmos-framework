# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import contextlib
from pathlib import Path
from typing import Any

import attrs
import hydra
import omegaconf
import torch.distributed.checkpoint as dcp
import transformers
from torch.distributed.checkpoint.filesystem import FileSystemReader
from torch.distributed.checkpoint.hf_storage import HuggingFaceStorageReader
from torch.distributed.checkpoint.state_dict import get_model_state_dict
from typing_extensions import TYPE_CHECKING, assert_never

from cosmos_framework.inference.common.args import CheckpointType
from cosmos_framework.inference.common.checkpoints import register_checkpoints
from cosmos_framework.inference.common.config import structure_config, undo_config_dict_replacements, unstructure_config
from cosmos_framework.utils.flags import SMOKE
from cosmos_framework.utils.lazy_config.lazy_call import LazyCall
from cosmos_framework.utils import log, misc
from cosmos_framework.configs.base.defaults.model_config import ParallelismConfig

# Some HF checkpoints store the vision tower under a shorter path than the
# model's nested attribute layout (e.g. ``model.visual.blocks.…`` instead of
# the model-side ``model.net.language_model.visual.blocks.…``).  ``dcp.load``
# does an exact key-name lookup against the checkpoint metadata, so without
# a rename the loader raises ``Missing key in checkpoint state_dict: ...``.
# We do a dict-level rename of the *requested* state_dict keys before
# ``dcp.load``, preserving tensor identity so the load still mutates the
# real model parameters in place.
#
# The match is on the *substring* ``net.language_model.visual.`` (not a
# whole-key prefix), so any prefix/suffix around it is preserved verbatim —
# e.g. ``model.net.language_model.visual.blocks.0.attn.proj.weight`` becomes
# ``model.visual.blocks.0.attn.proj.weight`` and a hypothetical
# ``ema.shadow.net.language_model.visual.layernorm.weight`` becomes
# ``ema.shadow.visual.layernorm.weight``.
_VISUAL_SUBSTR_MODEL = "net.language_model.visual."
_VISUAL_SUBSTR_CKPT = "visual."


def _maybe_remap_visual_prefix(
    state_dict: dict[str, Any],
) -> dict[str, Any]:
    """Rename the substring ``net.language_model.visual.`` to ``visual.``
    inside every key of ``state_dict`` since the checkpoint stores the vision
    tower under the short form.  The substring is matched anywhere in the
    key (not only at the start) so any prefix and suffix around the
    matched substring are copied through to the renamed key unchanged.
    No-op when the checkpoint already uses the long form or has no visual
    keys at all.

    Tensor identity is preserved by the rename — the returned dict points
    at the *same* tensor objects, just under different keys.  ``dcp.load``
    mutates those tensors in place, so loading into the renamed dict writes
    into the live model parameters with no extra copy.

    Replacement uses ``str.replace(..., count=1)`` so a pathological key
    that contained the long substring twice would only have its first
    occurrence rewritten; canonical transformer layouts hit the substring
    at most once per key.
    """
    remapped: dict[str, Any] = {}
    renamed = 0
    for k, v in state_dict.items():
        if _VISUAL_SUBSTR_MODEL in k:
            new_k = k.replace(_VISUAL_SUBSTR_MODEL, _VISUAL_SUBSTR_CKPT, 1)
            renamed += 1
        else:
            new_k = k
        remapped[new_k] = v
    log.info(f"_maybe_remap_visual_prefix: renamed {renamed} model state_dict keys.")
    return remapped


if TYPE_CHECKING:
    from cosmos_framework.model.vfm.omni_mot_model import OmniMoTModel


# Resolve to the release-tree root so relative-path checkpoint config entries
# (e.g. `cosmos_framework/model/vfm/vlm/qwen3_vl/configs/Qwen3-VL-8B-Instruct.json`) load
# correctly under contextlib.chdir(_ROOT_DIR) in __init__. In the original
# cosmos3 release the cosmos3 package lives at the tree root, so parents[1] is
# the release root. In the cosmos_training release the package lives at
# cosmos_framework/inference/, so the release root is one level higher (parents[2]).
try:
    import cosmos_framework.model.vfm  # noqa: F401

    _ROOT_DIR = Path(__file__).parents[2].absolute()
except ImportError:
    _ROOT_DIR = Path(__file__).parents[1].absolute()


class Cosmos3OmniConfig(transformers.PretrainedConfig):
    model_type = "cosmos3_omni"

    def __init__(self, model: dict | None = None, **kwargs):
        if model is not None:
            model = undo_config_dict_replacements(model)
        self.model = model or {}

        super().__init__(**kwargs)

        self.auto_map = {
            "AutoConfig": "cosmos3.model.Cosmos3OmniConfig",
            "AutoModel": "cosmos3.model.Cosmos3OmniModel",
        }

    @property
    def parallelism(self) -> dict:
        return self.model.get("config", {}).get("parallelism", {})

    @parallelism.setter
    def parallelism(self, value: dict | None):
        if value is None:
            return
        self.model.setdefault("config", {})["parallelism"] = unstructure_config(LazyCall(ParallelismConfig)(**value))


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
    ):
        if config is None:
            config = Cosmos3OmniConfig.from_pretrained(checkpoint_path)
        if parallelism_config is None:
            parallelism_config = ParallelismConfig()
        config.parallelism = attrs.asdict(parallelism_config)
        model = cls(config)
        checkpoint_type = CheckpointType.from_path(checkpoint_path)
        match checkpoint_type:
            case CheckpointType.DCP:
                state_dict = get_model_state_dict(model.model)
                storage_reader = FileSystemReader(str(checkpoint_path))
            case CheckpointType.HF:
                is_diffusers = next(checkpoint_path.glob("diffusion_pytorch_model.*"), None) is not None
                if is_diffusers:
                    state_dict = get_model_state_dict(model.model)
                else:
                    state_dict = get_model_state_dict(model)
                storage_reader = HuggingFaceStorageReader(str(checkpoint_path))
            case _:
                assert_never(checkpoint_type)
        state_dict = _maybe_remap_visual_prefix(state_dict)
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
