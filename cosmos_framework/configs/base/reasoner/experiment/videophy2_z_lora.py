# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""VideoPhy-2 LoRA SFT recipes — LoRA counterparts of ``videophy2_sft_*``.

Each recipe is a ``deepcopy`` of the matching full-fine-tune experiment plus the
LoRA delta below; the dataflow, dataset, freeze config, and parallelism come
along unchanged so the only variable between a LoRA run and its full-fine-tune
baseline is the adapter.

The delta, and why each piece is needed:

* ``model.config.policy.lora_enabled`` — turns on the injection in
  ``VLMModel._init_vlm`` (meta device, pre-FSDP).
* ``optimizer.lr`` 1e-6 -> 1e-4 — the full fine-tunes ship 1e-6; a rank-16
  adapter starting from ``lora_B=0`` needs roughly two orders of magnitude more
  to move at all in a short run.
* ``optimizer.keys_to_select=["lora_"]`` — belt-and-braces with the
  ``requires_grad`` enforcement in ``VLMModel.__init__``; keeps the optimizer
  state at adapter size rather than allocating for the frozen backbone.
* ``checkpoint.hf_export.enabled=False`` + a save_iter past ``max_iter`` — these
  are convergence smoke runs; exporting a 32B HF snapshot per save would
  dominate the wall clock and fill the disk.
* short cosine schedule matched to ``max_iter`` — the base recipes' 50-step
  cycle would leave the LR mid-decay at iteration 100.

Launch via ``examples/launch_sft_videophy2_lora_{nano,super,edge}.sh``.

Why the ``_z_`` in the filename
-------------------------------
``make_config`` calls ``import_all_modules_from_package(..., reload=True)``, and
``pkgutil.iter_modules`` walks this package in ALPHABETICAL order. Reloading a
module rebinds its module-level functions to fresh objects. So a module that
deepcopies a recipe from a sibling reloaded LATER ends up holding the sibling's
pre-reload function objects, and ``pickle`` — which the dataloader workers use —
rejects them with "it's not the same object as
``...videophy2_sft_nano.build_videophy2_local_dataset``".

This module must therefore sort AFTER every ``videophy2_sft_*`` module it clones.
``videophy2_sft_super`` gets away with the same pattern only because "super"
happens to sort after "nano". The assertion below turns that implicit ordering
constraint into a loud failure at config-load time rather than a confusing
pickling error minutes into a run.
"""

from __future__ import annotations

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.reasoner.experiment.videophy2_sft_nano import videophy2_sft_nano
from cosmos_framework.configs.base.reasoner.experiment.videophy2_sft_super import videophy2_sft_super
from cosmos_framework.configs.base.reasoner.experiment.videophy2_sft_edge import videophy2_sft_edge

cs = ConfigStore.instance()


def _assert_reload_order() -> None:
    """Fail loudly if this module no longer sorts after the recipes it clones.

    See the module docstring: a stale cross-module function reference surfaces as
    a ``_pickle.PicklingError`` from a dataloader worker, which is a long way
    from its cause. Comparing the object we captured against the one currently
    bound on the source module catches it here instead.
    """
    import pkgutil
    import os

    here = os.path.basename(__file__).removesuffix(".py")
    siblings = [m.name for m in pkgutil.iter_modules([os.path.dirname(__file__)])]
    cloned = [m for m in siblings if m.startswith("videophy2_sft_")]
    late = [m for m in cloned if m > here]
    assert not late, (
        f"{here} clones {cloned} but sorts BEFORE {late}, which are reloaded after it by "
        "import_all_modules_from_package(reload=True). The cloned recipes would carry "
        "stale function objects and fail to pickle in the dataloader workers. "
        f"Rename this module so it sorts after {late}."
    )


# Qwen3-VL LLM attention projections. Matched by EXACT child-module name, so the
# vision tower (``qkv`` / ``proj`` / ``linear_fc1`` / ``linear_fc2``) is not hit.
_QWEN3_VL_TARGETS = "q_proj,k_proj,v_proj,o_proj"


def _lora_variant(base, *, lora_target_modules: str, exclude_path_regex: str = ""):
    """Clone a full-fine-tune recipe and switch LoRA on — nothing else.

    Every training hyperparameter — lr, max_iter, scheduler (warmup / cycle /
    f_min), weight_decay, betas, validation cadence, grad_accum, dataset — is
    inherited UNCHANGED from ``base``. A LoRA recipe is therefore an
    apples-to-apples counterpart of its full-fine-tune sibling: the only deltas
    are the LoRA adapter itself and training only those adapters
    (``keys_to_select=["lora_"]``). This is what lets the LoRA and full-FT curves
    be compared directly under identical settings.

    The launch TOML stays authoritative and can override any inherited value; the
    shipped ``videophy2_lora_nano.toml`` keeps every training field identical to
    ``videophy2_sft_nano.toml``. (The edge/super LoRA TOMLs deliberately raise lr
    and max_iter for a longer sweep — that lives in the TOML, not here.)
    """
    item = copy.deepcopy(base)

    item.model.config.policy.lora_enabled = True
    item.model.config.policy.lora_rank = 16
    item.model.config.policy.lora_alpha = 32
    item.model.config.policy.lora_target_modules = lora_target_modules
    item.model.config.policy.lora_exclude_path_regex = exclude_path_regex

    item.optimizer.keys_to_select = ["lora_"]

    # Adapters, not a full HF snapshot — don't export (esp. the 32B tier).
    # This is a callback toggle; it does not touch the optimization.
    item.checkpoint.hf_export.enabled = False

    item.job.wandb_mode = "online"
    item.job.group = "vlm_videophy2_lora"
    return item


videophy2_lora_nano = _lora_variant(videophy2_sft_nano, lora_target_modules=_QWEN3_VL_TARGETS)
videophy2_lora_super = _lora_variant(videophy2_sft_super, lora_target_modules=_QWEN3_VL_TARGETS)

# Edge (``cosmos3_edge``) uses the same projection names as Qwen3-VL in its LLM,
# but its SigLIP2 vision tower reuses three of them — verified against the LIVE
# module tree (modeling_cosmos3_edge.py / vision_siglip2.py), NOT the checkpoint:
#   LLM attention:  self_attn.{q_proj,k_proj,v_proj,o_proj}
#   SigLIP2 ViT:    self_attn.{q_proj,k_proj,v_proj,out_proj}
# So name matching alone would also adapt the vision tower, which this recipe
# freezes. The exclusion regex is what keeps the adapters in the LLM; without it
# the zero-adapter assertion still passes and the run silently trains the ViT.
#
# (The snapshot's safetensors index spells the LLM projections to_q/to_k/to_v/
# to_out — those are pre-remap checkpoint keys and match nothing in the model.)
_COSMOS3_EDGE_EXCLUDE = r"^model\.visual\."

videophy2_lora_edge = _lora_variant(
    videophy2_sft_edge,
    lora_target_modules=_QWEN3_VL_TARGETS,
    exclude_path_regex=_COSMOS3_EDGE_EXCLUDE,
)


for _item in [videophy2_lora_nano, videophy2_lora_super, videophy2_lora_edge]:
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]
    if "job" not in _item:
        _item["job"] = dict(name=experiment_name + "_${now:%Y-%m-%d}_${now:%H-%M-%S}")
    else:
        _item["job"]["name"] = experiment_name + "_${now:%Y-%m-%d}_${now:%H-%M-%S}"

    cs.store(group="experiment", package="_global_", name=experiment_name, node=_item)
