# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""L1/GPU: the GR00T training parity gate run in-process, the optimizer partition on the real head, and the
checkpoint/export round trip. Needs the gr00t venv, one GPU and a warm HF cache for the DROID checkpoint.

`GR00T_DIT_SDPA_MODE=math` is set at import, before any CUDA work: without the deterministic DiT attention
backend the official trainer cannot reproduce its own gradients (see `operator GR00T training documentation`)."""

from __future__ import annotations

import os

import pytest
import torch

os.environ.setdefault("GR00T_DIT_SDPA_MODE", "math")

pytest.importorskip("gr00t", reason="Isaac-GR00T is only installed in the gr00t venv")

from cosmos_framework.utils.helper_test import RunIf  # noqa: E402
from cosmos_framework.posttrain.configs.gr00t_n1d7_droid import make_model_config, make_optimizer_config  # noqa: E402
from cosmos_framework.posttrain.tests.gr00t_train_parity_cli import TOLERANCES  # noqa: E402
from cosmos_framework.posttrain.tests.helpers.parity_diff import evaluate  # noqa: E402

CHECKPOINT = os.environ.get("GR00T_CKPT", "nvidia/GR00T-N1.7-DROID")


def _out_root(tmp_path) -> str:
    # Artifacts (a ~12 GB export among them) go under the scratch tree when one is configured, not /tmp.
    root = os.path.join(os.environ.get("GR00T_SCRATCH", str(tmp_path)), "stage2a", "l1")
    os.makedirs(os.path.join(root, "cfg"), exist_ok=True)
    return root


def _require_checkpoint() -> None:
    if "HF_HOME" not in os.environ:
        pytest.skip("HF_HOME is unset; point it at a cache holding the DROID checkpoint")


@pytest.fixture(scope="module")
def config():
    return make_model_config()


@pytest.mark.L1
@pytest.mark.GPU
@RunIf(min_gpus=1)
def test_training_matches_the_official_trainer_bitwise(tmp_path) -> None:
    """All six layers at the canonical full action horizon, where T1 validates the mask derivation
    against the processor's own mask (below it the recipe's mask is injected into the reference too)."""
    _require_checkpoint()
    from cosmos_framework.posttrain.tests.gr00t_train_parity_cli import run_train_parity

    report = run_train_parity(
        checkpoint=CHECKPOINT,
        batch=2,
        steps=2,  # the divergence this catches appears at the first optimizer step
        seed=1234,
        out_root=_out_root(tmp_path),
        training_prefix=40,
    )
    over_tolerance, not_bitwise = evaluate(report["diffs"], TOLERANCES)
    assert not report["notes"], report["notes"]
    assert not over_tolerance, [d["name"] for d in over_tolerance[:5]]
    assert not not_bitwise, [(d["name"], d["max_abs"]) for d in not_bitwise[:5]]
    assert len(report["diffs"]) > 1000, f"only {len(report['diffs'])} rows compared"  # no vacuous pass


@pytest.mark.L1
@pytest.mark.GPU
@RunIf(min_gpus=1)
def test_optimizer_groups_match_the_official_partition_on_the_real_model(tmp_path, config) -> None:
    _require_checkpoint()
    from transformers import Trainer

    from cosmos_framework.posttrain.config.build import build_optimizer
    from cosmos_framework.posttrain.model.gr00t.policy import Gr00tN1d7TrainPolicy

    policy = Gr00tN1d7TrainPolicy(config, device="cuda:0", save_cfg_dir=_out_root(tmp_path) + "/cfg")
    optimizer = build_optimizer(policy, make_optimizer_config())

    decay_names = set(Trainer.get_decay_parameter_names(None, policy.gr00t))
    by_id = {id(p): n for n, p in policy.net.named_parameters()}
    groups = [{by_id[id(p)] for p in g["params"]} for g in optimizer.param_groups]
    assert groups == [
        {n for n in by_id.values() if n in decay_names},
        {n for n in by_id.values() if n not in decay_names},
    ]
    assert groups[0] and groups[1]


@pytest.mark.L1
@pytest.mark.GPU
@RunIf(min_gpus=1)
def test_checkpoint_round_trip_and_export_are_bitwise(tmp_path, config) -> None:
    """`LLMTrainer.save` (DCP) -> exporter -> Isaac's strict `from_pretrained` must return OUR head."""
    _require_checkpoint()
    from transformers import AutoModel

    from cosmos_framework.posttrain.config.build import build_optimizer
    from cosmos_framework.posttrain.configs.gr00t_n1d7_droid import make_objective
    from cosmos_framework.posttrain.model.gr00t.policy import Gr00tN1d7TrainPolicy
    from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer
    from cosmos_framework.posttrain.scripts.export_gr00t_checkpoint import export_checkpoint

    root = _out_root(tmp_path)
    policy = Gr00tN1d7TrainPolicy(config, device="cuda:0", save_cfg_dir=root + "/cfg")
    trainer = LLMTrainer(
        policy,
        make_objective(config, policy.gr00t.collator),
        build_optimizer(policy, make_optimizer_config()),
        max_grad_norm=1.0,
        checkpoint_format="dcp",
    )
    # Move the weights so the round trip is not trivially satisfied by the untouched checkpoint.
    with torch.no_grad():
        next(iter(policy.net.parameters())).add_(0.01)
    saved = {k: v.detach().cpu().clone() for k, v in policy.net.state_dict().items()}

    iter_dir = os.path.join(root, "ckpt", "iter_000000")
    trainer.save(iter_dir)
    out_dir = os.path.join(root, "export")
    export_checkpoint(base=CHECKPOINT, posttrain_ckpt=iter_dir, out=out_dir)

    import gr00t.model.gr00t_n1d7.setup  # noqa: F401  (registers the auto-classes)

    reloaded = AutoModel.from_pretrained(out_dir)
    reloaded_state = {f"action_head.{k}": v for k, v in reloaded.action_head.state_dict().items()}
    assert all(torch.equal(reloaded_state[key].float().cpu(), value.float()) for key, value in saved.items())


@pytest.mark.L1
@pytest.mark.GPU
@RunIf(min_gpus=1)
def test_fsdp_precision_preset_trains_in_bf16(tmp_path) -> None:
    """The RLinf-style all-bf16 preset (`--precision-mode fsdp`): parameters stored in bf16, the compute dtype
    delivered by the FSDP cast rather than autocast, loss finite. There is no official reference for this recipe,
    so it is smoke-tested, not gated."""
    _require_checkpoint()
    import math

    from cosmos_framework.posttrain.tests.gr00t_train_parity_cli import run_our_side

    result = run_our_side(
        checkpoint=CHECKPOINT,
        batch=1,
        steps=2,
        seed=1234,
        out_root=_out_root(tmp_path),
        training_prefix=40,
        precision_mode="fsdp",
    )
    assert result["param_dtypes"] == ["torch.bfloat16"], result["param_dtypes"]  # the trainable head
    assert len(result["losses"]) == 2 and all(math.isfinite(loss) for loss in result["losses"]), result["losses"]
