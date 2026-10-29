# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""SFT configuration validation, recipe preflight, and install-free launch generation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from cosmos_framework.posttrain.config.rl import OptimizerConfig
from cosmos_framework.posttrain.config.sft import SFTConfig, SFTDatasetConfig, SFTValidationConfig

pytestmark = [pytest.mark.L0, pytest.mark.CPU]
_ROOT = Path(__file__).resolve().parents[4]
_POSTTRAIN = _ROOT / "cosmos_framework/posttrain"
# The served generation prefix of the path of record (tools present: no <think>), recorded live; review R138-01.
_SERVED_PREFIX_FIXTURE = _POSTTRAIN / "tests" / "fixtures" / "served_prefix_with_tools.json"


@pytest.mark.parametrize("kwargs", [{"micro_batch_size": 0}, {"max_length": 1}, {"grad_accum": 0}, {"max_steps": -1}])
def test_invalid_batching(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        SFTConfig(**kwargs)


def test_data_selection_and_validation_require_explicit_sources() -> None:
    with pytest.raises(ValueError, match="not both"):
        SFTDatasetConfig(path="data.jsonl", factory="example.dataset")
    with pytest.raises(ValueError, match="requires"):
        SFTDatasetConfig().require_source()
    with pytest.raises(ValueError, match="requires"):
        SFTConfig(validation=SFTValidationConfig(enabled=True))
    with pytest.raises(ValueError, match="reserved"):
        SFTConfig(export_every=10)


@pytest.mark.parametrize("remote", [False, True], ids=["local", "s3"])
def test_recipe_preflight_and_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, remote: bool) -> None:
    from cosmos_framework.utils.config import load_config
    from cosmos_framework.posttrain.control.bind import bind_check
    from cosmos_framework.posttrain.control.recipe import sft_manifest
    from cosmos_framework.posttrain.entrypoints.sft import role_specs, validate_config
    from cosmos_framework.posttrain.utils import object_store

    data_root = "s3://test-bucket/sft" if remote else str(tmp_path)
    monkeypatch.setenv("SFT_DATA_ROOT", data_root)
    monkeypatch.setenv("SFT_S3_CREDENTIALS", str(tmp_path / "gcs.json"))
    monkeypatch.setenv("SFT_MODEL_PATH", str(tmp_path / "model"))
    monkeypatch.setattr(object_store, "build_s3_client", lambda *args: pytest.fail("preflight must not access GCS"))
    for split in ("train", "validation"):
        # preflight checks availability; row semantics belong to source setup
        (tmp_path / f"{split}.jsonl").write_text("{}\n")
    config = load_config("cosmos_framework/posttrain/configs/sft_qwen3p8_27b_obj1k.py", [])
    rl = validate_config(config)
    assert config.job.wandb_mode == "disabled"
    assert config.job.project == "cosmos3_reasoner"
    assert rl.data.path == f"{data_root}/train.jsonl"
    assert rl.validation.data.path == f"{data_root}/validation.jsonl"
    assert rl.data.credentials == rl.validation.data.credentials == str(tmp_path / "gcs.json")
    assert config.model_hf.name_or_path == str(tmp_path / "model")
    bind_check(sft_manifest(role_specs(config))).raise_for_status()
    config.grid.backend = "ray"
    with pytest.raises(ValueError, match="local/torchrun"):
        validate_config(config)


def test_recipe_generates_install_free_torchrun_script(tmp_path: Path) -> None:
    profile_path = _POSTTRAIN / "configs/sft_qwen3p8_27b_obj1k_gcp.json"
    profile = json.loads(profile_path.read_text())
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(_POSTTRAIN / "utils/launch_multinode.py"),
            "--launch-config",
            str(profile_path),
            "--repo-root",
            str(_ROOT),
            "--log-dir",
            str(tmp_path),
            "--dry-run",
        ],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    script = result.stdout
    assert "launch_multinode_torchrun.sh" in script
    assert f"--container-image {profile['container_image']}" in script
    assert f"VENV_ACTIVATE={profile['venv']}" in script
    assert "pip install" not in script and "uv sync" not in script
    subprocess.run(["bash", "-n"], input=script, text=True, check=True, timeout=10)


def test_packer_refuses_old_transformers_before_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    from cosmos_framework.posttrain.data.packer.sft import build_sft_packer

    monkeypatch.setattr("cosmos_framework.posttrain.data.packer.sft.transformers_version", "4.57.6")
    with pytest.raises(RuntimeError, match="--no-group transformers4 --group sft"):
        build_sft_packer("not-a-real-model", max_length=32)


@pytest.fixture
def apex_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``importlib.util.find_spec("apex")`` succeed so fusedadam acceptance does not depend on the test runtime."""
    real = importlib.util.find_spec

    def find_spec(name: str, *args: Any, **kwargs: Any) -> Any:
        return types.SimpleNamespace(name="apex") if name == "apex" else real(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)


def _native_sft_config(tmp_path: Path, model_type: str, builder: str | None = None) -> Any:
    from cosmos_framework.posttrain.config.config_sft import make_config
    from cosmos_framework.posttrain.config.model import (
        NativePolicyBuildSpec,
        NativePolicyParallelismConfig,
        Qwen3VLModelConfig,
    )

    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(json.dumps({"model_type": model_type}))
    (tmp_path / "train.jsonl").write_text("{}\n")
    config = make_config()
    config.job.wandb_mode = "disabled"
    config.model_backend = "i4"
    config.model = NativePolicyBuildSpec(
        builder=builder or "cosmos_framework.posttrain.model.qwen3_vl.build_qwen3_vl_native_policy",
        model_id=str(model_dir),
        model_config=Qwen3VLModelConfig(dtype="bfloat16"),
        parallelism=NativePolicyParallelismConfig(dp_replicate=1, dp_shard=-1),
        requires_collated_policy_forward=True,
    )
    config.rl.master_dtype = config.rl.param_dtype = "bfloat16"
    config.rl.optimizer = OptimizerConfig(name="fusedadam", lr=1e-5)
    config.rl.data = SFTDatasetConfig(path=str(tmp_path / "train.jsonl"))
    return config


@pytest.mark.parametrize("model_type", ["qwen3_vl_moe", "qwen3_vl"])
def test_validate_config_accepts_native_qwen3_vl_spec(tmp_path: Path, model_type: str, apex_present: None) -> None:
    """The i4-native Qwen3-VL seat passes preflight without model_hf; the packer source is the spec's model_id."""
    from cosmos_framework.posttrain.entrypoints.sft import validate_config

    config = _native_sft_config(tmp_path, model_type)
    assert config.model_hf.name_or_path == ""
    rl = validate_config(config)
    assert rl.data.path == str(tmp_path / "train.jsonl")
    from cosmos_framework.posttrain.entrypoints.sft import sft_policy_source

    assert sft_policy_source(config) == (str(tmp_path / "model"), False)


def test_validate_config_refuses_native_specs_outside_the_qwen3_vl_seat(tmp_path: Path, apex_present: None) -> None:
    from cosmos_framework.posttrain.config.config_sft import make_config
    from cosmos_framework.posttrain.entrypoints.sft import validate_config

    with pytest.raises(ValueError, match="qwen3_vl"):
        validate_config(_native_sft_config(tmp_path, "llama"))
    other = _native_sft_config(
        tmp_path / "other", "qwen3_vl_moe", builder="cosmos_framework.posttrain.model.other.build"
    )
    with pytest.raises(ValueError, match="native builder"):
        validate_config(other)
    bare = make_config()
    bare.model_backend = "i4"
    with pytest.raises(ValueError, match="NativePolicyBuildSpec"):
        validate_config(bare)


def test_sft_support_is_declared_by_the_model_family_not_by_an_entrypoint_allowlist(
    tmp_path: Path, apex_present: None
) -> None:
    """FIXTURE: on de636dcaa5 ``entrypoints/sft.py`` defined ``NATIVE_SFT_BUILDER`` and ``NATIVE_SFT_MODEL_TYPES`` and
    refused any other builder with "supports the native builder ... only". The declaration lives beside the builder:
    ``model/qwen3_vl.py`` registers its SFT source in ``model/sft_source.py`` and owns the ``qwen3_vl`` /
    ``qwen3_vl_moe`` check; the entrypoint defines no ``NATIVE_SFT_*`` name and only consumes the registry. An
    importable builder that registered nothing is refused by name ("declares no SFT source"); the Qwen3-VL builder
    resolves to the spec's ``model_id`` as before."""
    import ast

    from cosmos_framework.posttrain.entrypoints import sft as entry
    from cosmos_framework.posttrain.entrypoints.sft import sft_policy_source

    tree = ast.parse(Path(entry.__file__).read_text())
    assigned = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert not {name for name in assigned if name.startswith("NATIVE_SFT_")}, assigned
    # Imported after the behavioural assertion so that a tree with the allowlist fails on it, not on this ImportError
    # (review round 1 of fix round 1, finding F3).
    from cosmos_framework.posttrain.model.sft_source import registered_native_sft_builders

    qwen = "cosmos_framework.posttrain.model.qwen3_vl.build_qwen3_vl_native_policy"
    assert sft_policy_source(_native_sft_config(tmp_path, "qwen3_vl_moe")) == (str(tmp_path / "model"), False)
    assert qwen in registered_native_sft_builders()
    undeclared = _native_sft_config(
        tmp_path / "undeclared", "qwen3_vl_moe", builder="cosmos_framework.posttrain.model.registry.build_model"
    )
    with pytest.raises(ValueError, match=r"native builder .*registry\.build_model.* declares no SFT source") as refused:
        sft_policy_source(undeclared)
    assert qwen in str(refused.value), "the refusal lists the registered builders"


def test_validate_config_refuses_adamw_on_the_bf16_native_seat(tmp_path: Path, apex_present: None) -> None:
    """bf16 storage is forced for qwen3_vl_moe; torch adamw keeps moments and updates in that dtype and most
    lr-1e-5 updates round to zero, so the seat requires fusedadam (apex fp32 master weights)."""
    from cosmos_framework.posttrain.entrypoints.sft import validate_config

    config = _native_sft_config(tmp_path, "qwen3_vl_moe")
    config.rl.optimizer = OptimizerConfig(name="adamw", lr=1e-5)
    with pytest.raises(ValueError, match="fusedadam"):
        validate_config(config)
    config.rl.optimizer = OptimizerConfig(name="fusedadam", lr=1e-5)
    validate_config(config)
    config.model.model_config.dtype = "float32"
    config.rl.optimizer = OptimizerConfig(name="adamw", lr=1e-5)
    validate_config(config)  # fp32 storage: adamw's moments are fp32 already


def test_native_recipe_carries_the_canary_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, apex_present: None
) -> None:
    """configs/sft_qwen3_vl_moe_native.py: sdpa, fusedadam, bf16 storage, and research document section 5's fields."""
    from cosmos_framework.utils.config import load_config
    from cosmos_framework.posttrain.entrypoints.sft import validate_config

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen3_vl_moe"}))
    for split in ("train", "validation"):
        (tmp_path / f"{split}.jsonl").write_text("{}\n")
    monkeypatch.setenv("SFT_MODEL_PATH", str(model_dir))
    monkeypatch.setenv("SFT_DATA_ROOT", str(tmp_path))
    config = load_config("cosmos_framework/posttrain/configs/sft_qwen3_vl_moe_native.py", [])
    assert config.model_backend == "i4"
    assert config.model.model_config.attn_impl == "sdpa"
    assert config.model.model_config.dtype == "bfloat16"
    rl = validate_config(config)
    assert rl.optimizer.name == "fusedadam" and rl.optimizer.warmup_steps == 1000
    assert rl.max_length == 16384
    assert rl.processor_kwargs["images_kwargs"]["size"] == {"shortest_edge": 65536, "longest_edge": 4194304}
    assert rl.checkpoint_format == "consolidated"
    assert rl.micro_batch_size == 1 and rl.grad_accum == 1
    assert rl.validation.enabled and rl.validation.before_training and rl.validation.every_steps == 50


def test_validate_config_refuses_adamw_below_fp32_master_on_the_hf_seat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rounding mechanism is seat-agnostic: the dense HF recipe at a bfloat16 master (set, or unset and falling
    back to the model dtype) with adamw is refused; the recipe's own fp32 master is accepted."""
    from cosmos_framework.utils.config import load_config
    from cosmos_framework.posttrain.entrypoints.sft import validate_config

    for split in ("train", "validation"):
        (tmp_path / f"{split}.jsonl").write_text("{}\n")
    monkeypatch.setenv("SFT_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("SFT_MODEL_PATH", str(tmp_path / "model"))
    recipe = "cosmos_framework/posttrain/configs/sft_qwen3p8_27b_multimodal.py"
    validate_config(load_config(recipe, []))  # master float32, autocast: accepted
    with pytest.raises(ValueError, match="float32"):
        validate_config(load_config(recipe, ["--", "rl.master_dtype=bfloat16"]))
    with pytest.raises(ValueError, match="float32"):
        validate_config(
            load_config(recipe, ["--", "rl.master_dtype=", "rl.param_dtype="])
        )  # falls back to model_hf.dtype


def test_validate_config_refuses_fusedadam_without_apex(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """apex is imported only when the trainer builds FusedAdam; preflight must name the missing runtime instead."""
    from cosmos_framework.posttrain.entrypoints.sft import validate_config

    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a, **k: None if name == "apex" else real(name, *a, **k)
    )
    with pytest.raises(ValueError, match="apex"):
        validate_config(_native_sft_config(tmp_path, "qwen3_vl_moe"))


def _native_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str | None, opts: list[str] | None = None
) -> Any:
    """Load the native recipe as the entrypoint does; ``opts`` are passed as ``["--", *opts]`` (``make_setup``)."""
    from cosmos_framework.utils.config import load_config

    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen3_vl_moe"}))
    for split in ("train", "validation"):
        (tmp_path / f"{split}.jsonl").write_text("{}\n")
    monkeypatch.setenv("SFT_MODEL_PATH", str(model_dir))
    monkeypatch.setenv("SFT_DATA_ROOT", str(tmp_path))
    if variant is None:
        monkeypatch.delenv("SFT_RECIPE_VARIANT", raising=False)
    else:
        monkeypatch.setenv("SFT_RECIPE_VARIANT", variant)
    return load_config("cosmos_framework/posttrain/configs/sft_qwen3_vl_moe_native.py", ["--", *opts] if opts else [])


def test_native_recipe_records_the_backbone_decisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, apex_present: None
) -> None:
    """Review R125-D-01: every objective-level knob is an explicit, validated field with the Cosmos post-training
    recipe's value as this recipe's default (the ``<think>`` opener excepted: the Nemo-Chat rule masks it, while on the
    serving path of record -- tools present -- vLLM 0.24 prefills no opener and the student must emit it, reviews
    R134-01 / R138-01; the decay grouping is the seat's own, review R134-03), and every one reaches the resolved SFT
    config. The parent had none of these fields."""
    from cosmos_framework.posttrain.entrypoints.sft import validate_config

    rl = validate_config(_native_recipe(tmp_path, monkeypatch, None))
    assert rl.optimizer.betas == [0.9, 0.95]
    assert rl.optimizer.weight_decay == 0.1 and rl.optimizer.decay_groups == "qwen_moe"
    assert rl.optimizer.lr_multipliers == {"model.visual": 0.1}
    assert rl.optimizer.warmup_steps == 1000
    assert rl.lr_schedule.kind == "cosine" and rl.lr_schedule.f_start == 0.01 and rl.lr_schedule.f_min == 0.1
    assert rl.lr_schedule.cycle_steps == 0  # = max_steps
    assert rl.ce_exponent == 0.5
    assert rl.mask_think_opener is False
    assert rl.moe_lbl_coeff == 8e-3 and rl.moe_lbl_method == "global"
    assert rl.validation.every_steps == 50
    assert rl.processor_kwargs["images_kwargs"]["size"]["longest_edge"] == 4194304 == 2048 * 2048
    assert rl.deterministic is False
    # The command line reaches them like any other field (one lever per arm).
    from cosmos_framework.utils.config import load_config

    overridden = load_config(
        "cosmos_framework/posttrain/configs/sft_qwen3_vl_moe_native.py",
        ["--", "rl.deterministic=true", "rl.ce_exponent=0.0", "rl.moe_lbl_coeff=0.0", "rl.lr_schedule.kind=constant"],
    )
    rl_det = validate_config(overridden)
    assert rl_det.deterministic is True and rl_det.ce_exponent == 0.0 and rl_det.moe_lbl_coeff == 0.0
    assert rl_det.lr_schedule.kind == "constant"


def test_native_recipe_a1_and_a1a_variants_reproduce_the_arms_of_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, apex_present: None
) -> None:
    """``SFT_RECIPE_VARIANT=a1`` / ``a1a`` = the values the A1 / A1a arms trained with (their ``config_resolved.yaml``):
    constant lr 1e-5 without warmup, betas (0.9, 0.999), weight decay 0.01 in one group, no multipliers, token-mean CE,
    no load-balancing term, the opener supervised, the 4,147,200-px budget; A1 validated every 50 steps and A1a every
    step, the one field the two resolved configs differ in (review R134-09: 22bfce78f40's single ``a1`` variant carried
    A1a's cadence). ``prefilled_opener`` = the defaults with the opener masked (for a serve that prefills it; not the
    path of record, review R138-01; it sets the typed field ``rl.serve_contract=prefills_opener``, review R142-04 and
    review threads b196e5e99f / 12f2ce563b); the earlier name ``nemo_chat_opener`` and an unknown variant are refused."""
    from cosmos_framework.posttrain.entrypoints.sft import validate_config

    for variant, cadence in (("a1", 50), ("a1a", 1)):
        rl = validate_config(_native_recipe(tmp_path, monkeypatch, variant))
        assert rl.optimizer.lr == 1e-5 and rl.optimizer.warmup_steps == 0 and rl.lr_schedule.kind == "constant"
        assert rl.optimizer.betas == [0.9, 0.999]
        assert (
            rl.optimizer.weight_decay == 0.01 and rl.optimizer.decay_groups == "" and rl.optimizer.lr_multipliers == {}
        )
        assert rl.ce_exponent == 0.0 and rl.moe_lbl_coeff == 0.0 and rl.mask_think_opener is False
        assert rl.validation.every_steps == cadence, variant
        assert rl.processor_kwargs["images_kwargs"]["size"]["longest_edge"] == 4147200
        assert rl.optimizer.grad_clip_norm == 1.0 and rl.optimizer.name == "fusedadam"
    opener = validate_config(_native_recipe(tmp_path, monkeypatch, "prefilled_opener"))
    assert opener.mask_think_opener is True and opener.moe_lbl_coeff == 8e-3 and opener.validation.every_steps == 50
    assert opener.serve_contract == "prefills_opener"
    for unknown in ("bogus", "nemo_chat_opener"):
        with pytest.raises(ValueError, match="SFT_RECIPE_VARIANT"):
            _native_recipe(tmp_path, monkeypatch, unknown)


def test_native_recipe_table_states_the_decay_references_and_the_no_prefill_with_tools() -> None:
    """Reviews R134-03 / R138-01 (decision-table truth, checked on the file as written, not on ``__doc__``): knob 4's
    cell names both references the grouping diverges from -- the backbone's ``_build_params_with_metadata`` decays all
    882 tensors; transformers' ``get_decay_parameter_names`` decays the embedding, the head and the router gates -- and
    states what the exemption protects and costs, instead of a "HF/Qwen convention" neither reference has; knob 7's
    cell names the MEASURED serving path of record -- Codex sends the tools array on every call, with tools present
    vLLM 0.24 prefills NO ``<think>`` (review R138-01: effort present or absent 157 = 157 input tokens), so the student
    must emit the opener itself and its loss is the only training signal for that token, ``True`` is a labelled RISK
    (UNMEASURED with tools present; C-04, the lab's live probe record of the served generation prompt on the A1a
    deployment, 2026-10-02, outside the repo, section 3's measured trigger quoted: an output that carries no
    ``</think>`` is
    returned as reasoning in full, tool-call turns unaffected; review R142-01), the prefill exists only for tool-less
    requests, the ``prefilled_opener`` variant is scoped as not the path of record and guarded (it sets the typed field
    ``rl.serve_contract=prefills_opener``; ``recipe_variant`` in provenance; review R142-04) -- instead of the superseded
    "PREFILLED on every turn" of review R134-01 (cef891c3e46) and the unmeasured consequence 4821e1d9943 stated for
    every no-think turn; this FIXTURE fails on both parents. The terms the rows use are defined in the Terms block above
    the table."""
    text = (Path(__file__).resolve().parents[1] / "configs" / "sft_qwen3_vl_moe_native.py").read_text()
    rows = {line.split("|")[1].strip(): line for line in text.splitlines() if line.startswith("| ")}
    decay = rows["weight decay"]
    for needed in (
        "``cosmos3/utils/optimizer.py::_build_params_with_metadata`` lines 326-331",
        "so all 882 tensors decay",
        "``Trainer.get_decay_parameter_names``",
        "DECAYS ``embed_tokens``, ``lm_head`` and every ``mlp.gate.weight``",
        "What that protects",
        "What it costs",
        "``decay_groups=''`` arm at wd 0.1",
    ):
        assert needed in decay, needed
    assert "HF/Qwen convention" not in text
    opener = rows["``<think>`` opener"]
    for needed in (
        "no prefill of the opener",
        "Codex sends the tools array on every call",
        "with tools present vLLM 0.24 appends ``<|im_start|>assistant\\n`` and NO ``<think>``",
        "the student must emit ``<think>`` itself",
        "NECESSARY, not merely harmless",
        "UNMEASURED with tools present",
        "no student was trained with ``True``",
        "the first token after ``assistant\\n`` with tools present was never sampled",
        "an output that carries no ``</think>`` as reasoning in full",
        "A no-think TOOL-CALL turn is unaffected",
        "stays supervised under ``True``",
        "the only training signal for a token the student must emit on the path of record",
        "``rl.serve_contract=prefills_opener``",
        "``recipe_variant`` (review R142-04",
        "round 138 P4 / P7 and review round 142 Q3 / Q4",
        "The ``<think>`` prefill exists only for requests without tools",
        "review R138-01",
        "supersedes review R134-01",
        "``<think>\\n\\n</think>\\n\\n``",
        "MUST stay supervised",
        "``RECIPE_VARIANTS['prefilled_opener']``",
        "not the path of record",
    ):
        assert needed in opener, needed
    for superseded in (
        "PREFILLED on every turn",
        "so the student never generates it",
        "prefills it on every turn",
        "harmless; on a serve without the effort field",
        "break every",  # the unlabelled consequence of 4821e1d9943 (review R142-01)
        "rest on the base's prior alone",
        "SFT_SERVE_PREFILLS_OPENER",  # the environment-variable acknowledgement (review threads b196e5e99f / 12f2ce563b)
    ):
        assert superseded not in text, superseded
    variant_doc = text.split("def _variant_prefilled_opener")[1].split('"""')[1]
    assert "NOT the path of record" in variant_doc and "review R138-01" in variant_doc
    assert "UNMEASURED with tools present" in variant_doc and "review R142-01" in variant_doc
    assert "``rl.serve_contract=prefills_opener``" in variant_doc and "review R142-04" in variant_doc
    for term in ('"the serving path of record"', '"the prefill"', '"a no-think turn"', '"pin" vs "fixture"'):
        assert term in text.split("| knob |")[0], term  # defined before the table
    assert (
        rows["validation cadence"].count("a1a") == 1
        and "every 50 steps (A1) / every step (A1a)" in rows["validation cadence"]
    )


def test_default_recipe_supervises_the_opener_the_student_emits_after_the_served_prefix_with_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, apex_present: None
) -> None:
    """Review R138-01 (knob 7's mechanism). PIN for the behaviour and FIXTURE for the record: the packer's labels did not
    change in the commit that adds this test (with the fixture file copied beside it, it passes on the parent cef891c3e46),
    while ``tests/fixtures/served_prefix_with_tools.json`` did not exist there and the recipe text stated the opposite
    mechanism. The fixture is the served prefix of the path of record, measured live on the A1a deployment (vLLM 0.24) with
    a tools array present: ``/tokenize`` ids that end in ``<|im_start|>assistant\n`` with NO ``<think>`` (141 ids; the
    effort field adds 0 input tokens: 157 = 157 with one synthetic tool, round 138 P4 / P7, and 7,862 = 7,862 with the 8
    Codex tools and the wire instructions, review round 142 Q3 / Q4, the ``eight_tools_with_wire`` block added by fix
    round T4 after review R142-02), beside the tool-less prefix that does end in ``<think>``. A no-think turn
    rendered the exporter's way (``<think>\n\n</think>\n\nok``) through the packer under the default recipe
    (``mask_think_opener=False``) starts with exactly that served prompt text and continues with the student's own
    ``<think>``, ``\n\n``, ``</think>``, ``\n\n``, all four supervised; ``mask_think_opener=True`` (the ``prefilled_opener``
    variant) unsupervises exactly that first token. Text-level here (the offline toy tokenizer with the base's own template;
    its ids are not the base's); id-level in ``test_real_base_sidecars_render_the_served_prefix_ids_with_tools``."""
    from cosmos_framework.posttrain.data.packer.sft import QwenSFTDataPacker
    from cosmos_framework.posttrain.entrypoints.sft import validate_config
    from cosmos_framework.posttrain.schema import SFTTarget
    from cosmos_framework.posttrain.tests.helpers import sft_packer as helpers

    fixture = json.loads(_SERVED_PREFIX_FIXTURE.read_text())
    served = fixture["served_prefix_with_tools"]
    assert served["ends_with_think"] is False
    assert served["ids"][-3:] == [151644, 77091, 198] == served["generation_prefix_ids"]
    assert 151667 not in served["ids"] and served["count"] == len(served["ids"]) == 141
    assert served["responses_path_effort_adds_input_tokens"] == 0
    pair = served["responses_path_input_tokens"]
    assert pair["effort_high"] == pair["no_reasoning_field"] == 157  # one synthetic tool
    eight = pair["eight_tools_with_wire"]
    assert eight["effort_high"] == eight["no_reasoning_field"] == 7862  # the 8 Codex tools
    tool_less = fixture["served_prefix_without_tools_enable_thinking"]
    assert tool_less["ids"][-4:] == [151644, 77091, 198, 151667] and tool_less["enable_thinking_adds_input_tokens"] == 1
    assert served["text"].endswith("<|im_end|>\n<|im_start|>assistant\n")
    rl = validate_config(_native_recipe(tmp_path, monkeypatch, None))
    assert rl.mask_think_opener is False, "the default recipe supervises the opener"
    prompt, tools = fixture["messages"], [fixture["tool_as_vllm_renders_it"]]
    turn = {"role": "assistant", "content": fixture["no_think_turn"]["assistant_content"]}
    assert turn["content"] == "<think>\n\n</think>\n\nok"

    tokenizer = helpers.local_tokenizer(merge_newlines=True)
    processor = helpers.local_processor(tokenizer, chat_template=helpers.official_qwen3_vl_chat_template())
    think, two, think_end = (tokenizer.convert_tokens_to_ids(token) for token in ("<think>", "ĊĊ", "</think>"))
    packer = QwenSFTDataPacker(tokenizer, max_length=4096, processor=processor, mask_think_opener=rl.mask_think_opener)
    served_text = processor.apply_chat_template(prompt, tools=tools, tokenize=False, add_generation_prompt=True)
    assert served_text == served["text"], "the base's template renders the prompt the server tokenized, without <think>"
    sample = packer.sft_process_sample(SFTTarget(prompt + [turn], tools=tools))
    prefix = tokenizer.encode(served_text, add_special_tokens=False)
    n = len(prefix)
    assert sample.input_ids[:n] == prefix and set(sample.labels[:n]) == {-100}, "the row starts with the served prompt"
    assert sample.input_ids[n : n + 4] == [think, two, think_end, two], "then the student's own empty block"
    assert sample.labels[n : n + 4] == [think, two, think_end, two], "opener and closer supervised under the default"
    assert tokenizer.decode(sample.input_ids[n + 4 : n + 6]) == "ok"
    masked = QwenSFTDataPacker(tokenizer, max_length=4096, processor=processor, mask_think_opener=True)
    variant = masked.sft_process_sample(SFTTarget(prompt + [turn], tools=tools))
    assert variant.input_ids == sample.input_ids and variant.labels[n] == -100, "the variant unsupervises it"
    assert variant.labels[n + 1 : n + 4] == [two, think_end, two]
    # The template's own enable_thinking branch would append <think> to the prompt; the server does not take it with tools.
    with_flag = processor.apply_chat_template(
        prompt, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    assert with_flag == served_text + "<think>"


def test_real_base_sidecars_render_the_served_prefix_ids_with_tools() -> None:
    """Opt-in (``COSMOS_SFT_QWEN3_VL_SIDECARS`` = the base checkpoint directory; the round's cpu job sets it): the id-level
    half of the fixture above, a PIN for the behaviour and a FIXTURE for the record. The packer built from the base's own
    sidecars renders the fixture's prompt (user + the tool as vLLM renders it) to exactly the 141 ids the live ``/tokenize``
    returned with tools and ``enable_thinking`` (ending 151644, 77091, 198; no 151667); the template's own
    ``enable_thinking`` branch would append 151667 (the server does not take it with tools); and the no-think row continues
    the served ids with 151667, 271, 151668, 271, 562 (``<think>``, ``\n\n``, ``</think>``, ``\n\n``, ``ok``), all five
    supervised under the default, then the supervised ``<|im_end|>``."""
    from cosmos_framework.posttrain.data.packer.sft import build_sft_packer
    from cosmos_framework.posttrain.schema import SFTTarget

    sidecars = os.environ.get("COSMOS_SFT_QWEN3_VL_SIDECARS")
    if not sidecars:
        pytest.skip("set COSMOS_SFT_QWEN3_VL_SIDECARS=<the base checkpoint dir> to compare the recorded /tokenize ids")
    fixture = json.loads(_SERVED_PREFIX_FIXTURE.read_text())
    for name, key in (("chat_template.json", "chat_template_sha256"), ("tokenizer.json", "tokenizer_json_sha256")):
        digest = hashlib.sha256((Path(sidecars) / name).read_bytes()).hexdigest()
        assert digest == fixture["grammar"][key], f"{name} is not the grammar the prefix was recorded on"
    served = fixture["served_prefix_with_tools"]
    prompt, tools = fixture["messages"], [fixture["tool_as_vllm_renders_it"]]
    turn = {"role": "assistant", "content": fixture["no_think_turn"]["assistant_content"]}
    packer = build_sft_packer(
        sidecars,
        max_length=4096,
        processor_kwargs={"images_kwargs": {"size": {"shortest_edge": 65536, "longest_edge": 65536}}},
    )
    served_text = packer.processor.apply_chat_template(prompt, tools=tools, tokenize=False, add_generation_prompt=True)
    assert served_text == served["text"]
    assert packer.tokenizer.encode(served_text, add_special_tokens=False) == served["ids"]
    with_flag = packer.processor.apply_chat_template(
        prompt, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    assert packer.tokenizer.encode(with_flag, add_special_tokens=False) == served["ids"] + [151667]
    sample = packer.sft_process_sample(SFTTarget(prompt + [turn], tools=tools))
    n = len(served["ids"])
    assert sample.input_ids[:n] == served["ids"] and set(sample.labels[:n]) == {-100}
    expected = fixture["no_think_turn"]["expected_continuation_ids"]
    assert expected == [151667, 271, 151668, 271, 562]
    assert sample.input_ids[n : n + 5] == expected and sample.labels[n : n + 5] == expected
    assert sample.input_ids[n + 5] == 151645 and sample.labels[n + 5] == 151645


def test_prefilled_opener_variant_sets_the_typed_serve_contract_and_is_recorded_in_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, apex_present: None
) -> None:
    """FIXTURE: on d8c7213d0a the variant was refused without the environment variable ``SFT_SERVE_PREFILLS_OPENER=1``
    and accepted with it, and neither the resolved config nor the provenance recorded that acknowledgement.
    ``SFT_RECIPE_VARIANT=prefilled_opener`` resolves ``mask_think_opener=True`` with
    ``rl.serve_contract="prefills_opener"`` and reads no environment variable; the field is in the resolved config
    (``config_resolved.yaml``); the resolved variant name is what ``utils/provenance.py`` records as ``recipe_variant``
    (review R142-04). The retired variable, set to ``1`` on the ``default`` variant, changes nothing."""
    from cosmos_framework.posttrain.entrypoints.sft import validate_config
    from cosmos_framework.posttrain.utils import provenance
    from cosmos_framework.posttrain.utils.entry import dump_resolved

    monkeypatch.delenv("SFT_SERVE_PREFILLS_OPENER", raising=False)
    config = _native_recipe(tmp_path, monkeypatch, "prefilled_opener")
    rl = validate_config(config)
    assert rl.mask_think_opener is True and rl.serve_contract == "prefills_opener"
    assert provenance.recipe_variant() == "prefilled_opener"
    record = provenance.collect_provenance(repo_root=tmp_path, posttrain_root=tmp_path, budget_s=5.0)
    assert record["recipe_variant"] == "prefilled_opener"
    monkeypatch.setenv("RANK", "0")
    dump_resolved(config, str(tmp_path / "out"))
    resolved = (tmp_path / "out" / "config_resolved.yaml").read_text()
    assert "serve_contract: prefills_opener" in resolved and "mask_think_opener: true" in resolved
    monkeypatch.setenv("SFT_SERVE_PREFILLS_OPENER", "1")  # the retired variable: read by nothing
    default = validate_config(_native_recipe(tmp_path, monkeypatch, None))
    assert default.mask_think_opener is False and default.serve_contract == "tools_no_prefill"
    assert provenance.recipe_variant() == "default"
    default_record = provenance.collect_provenance(repo_root=tmp_path, posttrain_root=tmp_path, budget_s=5.0)
    assert default_record["recipe_variant"] == "default"


def test_sft_config_refuses_an_opener_mask_outside_the_prefills_opener_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, apex_present: None
) -> None:
    """Review round 142b, section 11, re-shaped by review threads b196e5e99f / 12f2ce563b (FIXTURE: on d8c7213d0a the
    override ``rl.mask_think_opener=True`` was ACCEPTED whenever ``SFT_SERVE_PREFILLS_OPENER=1`` stood in the shell, and
    the resolved config of the accepted run was byte-identical to a refused one's). The refusal lives in
    ``SFTConfig.__post_init__``, so a ``--`` override on the ``default`` variant, a YAML and the recipe variant are judged
    by the same typed field: ``mask_think_opener=True`` under the default contract ``tools_no_prefill`` is refused naming
    review R138-01 and ``rl.serve_contract=prefills_opener`` whatever the environment says, and it is refused where the
    config is BUILT (``load_config`` constructs ``SFTConfig``; on the parent the override built and the entrypoint's
    guard judged it later); with the contract it is accepted; an unknown contract is refused; the default variant is
    unaffected."""
    from cosmos_framework.posttrain.entrypoints.sft import validate_config

    monkeypatch.setenv("SFT_SERVE_PREFILLS_OPENER", "1")  # the retired variable must rescue nothing
    with pytest.raises(
        ValueError, match=r"rl\.mask_think_opener=True is refused.*review R138-01.*rl\.serve_contract=prefills_opener"
    ) as refused:
        # Refused where the config is BUILT (``load_config`` constructs ``SFTConfig``), before any preflight runs.
        _native_recipe(tmp_path, monkeypatch, None, opts=["rl.mask_think_opener=True"])
    assert "prefills NO opener with tools present" in str(refused.value)
    assert "SFT_SERVE_PREFILLS_OPENER" not in str(refused.value)
    accepted = validate_config(
        _native_recipe(
            tmp_path, monkeypatch, None, opts=["rl.mask_think_opener=True", "rl.serve_contract=prefills_opener"]
        )
    )
    assert accepted.mask_think_opener is True and accepted.serve_contract == "prefills_opener"
    with pytest.raises(
        ValueError, match="rl.mask_think_opener=True is refused under rl.serve_contract='tools_no_prefill'"
    ):
        SFTConfig(mask_think_opener=True)
    assert SFTConfig(mask_think_opener=True, serve_contract="prefills_opener").mask_think_opener is True
    assert SFTConfig(serve_contract="prefills_opener").mask_think_opener is False
    with pytest.raises(ValueError, match="serve_contract='bogus' must be one of"):
        SFTConfig(serve_contract="bogus")
    monkeypatch.delenv("SFT_SERVE_PREFILLS_OPENER", raising=False)
    assert validate_config(_native_recipe(tmp_path, monkeypatch, None)).mask_think_opener is False


def test_optimizer_config_refuses_negative_warmup() -> None:
    """Review R125-D-06: ``warmup_steps=-1`` was accepted and silently meant no warmup."""
    with pytest.raises(ValueError, match="warmup_steps"):
        OptimizerConfig(warmup_steps=-1)
    with pytest.raises(ValueError, match="warmup_steps"):
        OptimizerConfig(warmup_steps=True)  # type: ignore[arg-type]
    assert OptimizerConfig(warmup_steps=0).warmup_steps == 0
    with pytest.raises(ValueError, match="decay_groups"):
        OptimizerConfig(decay_groups="bogus")
    with pytest.raises(ValueError, match="lr_multipliers"):
        OptimizerConfig(lr_multipliers={"model.visual": 0.0})
    with pytest.raises(ValueError, match="lr_multipliers"):
        OptimizerConfig(lr_multipliers={"": 0.5})


def test_sft_config_validates_the_objective_knobs() -> None:
    from cosmos_framework.posttrain.config.sft import SFTLRScheduleConfig

    assert (
        SFTConfig().ce_exponent == 0.0 and SFTConfig().moe_lbl_coeff == 0.0 and SFTConfig().mask_think_opener is False
    )
    assert SFTConfig().lr_schedule.kind == "constant" and SFTConfig().serve_contract == "tools_no_prefill"
    for kwargs in ({"ce_exponent": 1.5}, {"ce_exponent": -0.1}, {"moe_lbl_coeff": 1.0}, {"moe_lbl_method": "sample"}):
        with pytest.raises(ValueError):
            SFTConfig(**kwargs)
    with pytest.raises(TypeError):
        SFTConfig(mask_think_opener=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="kind"):
        SFTLRScheduleConfig(kind="linear")
    with pytest.raises(ValueError, match="f_min"):
        SFTLRScheduleConfig(f_min=0.0)


# The 882 parameter names of Qwen3-VL-MoE-30B-A3B (the Cosmos 3.5 Super base; the base's model.safetensors.index.json
# lists exactly these: 48 text layers, 27 vision blocks, 3 deepstack mergers), with the shape rank each tensor has.
def _qwen3_vl_moe_parameter_names() -> list[tuple[str, int]]:
    names: list[tuple[str, int]] = [
        ("lm_head.weight", 2),
        ("model.language_model.embed_tokens.weight", 2),
        ("model.language_model.norm.weight", 1),
    ]
    for layer in range(48):
        prefix = f"model.language_model.layers.{layer}"
        names += [
            (f"{prefix}.input_layernorm.weight", 1),
            (f"{prefix}.post_attention_layernorm.weight", 1),
            (f"{prefix}.self_attn.q_proj.weight", 2),
            (f"{prefix}.self_attn.k_proj.weight", 2),
            (f"{prefix}.self_attn.v_proj.weight", 2),
            (f"{prefix}.self_attn.o_proj.weight", 2),
            (f"{prefix}.self_attn.q_norm.weight", 1),
            (f"{prefix}.self_attn.k_norm.weight", 1),
            (f"{prefix}.mlp.gate.weight", 2),
            (f"{prefix}.mlp.experts.gate_up_proj", 3),
            (f"{prefix}.mlp.experts.down_proj", 3),
        ]
    for block in range(27):
        prefix = f"model.visual.blocks.{block}"
        for leaf in ("attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2"):
            names += [(f"{prefix}.{leaf}.weight", 2), (f"{prefix}.{leaf}.bias", 1)]
        for leaf in ("norm1", "norm2"):
            names += [(f"{prefix}.{leaf}.weight", 1), (f"{prefix}.{leaf}.bias", 1)]
    for prefix in ("model.visual.merger", *[f"model.visual.deepstack_merger_list.{i}" for i in range(3)]):
        names += [(f"{prefix}.linear_fc1.weight", 2), (f"{prefix}.linear_fc1.bias", 1)]
        names += [(f"{prefix}.linear_fc2.weight", 2), (f"{prefix}.linear_fc2.bias", 1)]
        names += [(f"{prefix}.norm.weight", 1), (f"{prefix}.norm.bias", 1)]
    names += [("model.visual.patch_embed.proj.weight", 2), ("model.visual.patch_embed.proj.bias", 1)]
    names += [("model.visual.pos_embed.weight", 2)]
    return names


def _module_with_parameters(names: list[tuple[str, int]], *, wrap_layers: bool = False) -> Any:
    """A tiny ``nn.Module`` tree carrying exactly these dotted parameter names (shape rank preserved, sizes 2)."""
    import torch
    from torch import nn

    root = nn.Module()
    for name, rank in names:
        parts = name.split(".")
        if wrap_layers and (".layers." in name or ".blocks." in name):
            # The activation-checkpoint wrapper's segment after the layer/block index, as `named_parameters`
            # yields it on the wrapped model (`...layers.0._checkpoint_wrapped_module.mlp.gate.weight`).
            parts = parts[:4] + ["_checkpoint_wrapped_module"] + parts[4:]
        module = root
        for part in parts[:-1]:
            if not hasattr(module, part):
                module.add_module(part, nn.Module())
            module = getattr(module, part)
        module.register_parameter(parts[-1], nn.Parameter(torch.zeros([2] * rank)))
    return root


@pytest.mark.parametrize("wrap_layers", [False, True], ids=["plain_names", "ac_wrapped_names"])
def test_qwen_moe_decay_groups_and_lr_multipliers_on_the_real_parameter_names(wrap_layers: bool) -> None:
    """Review R125-B-01 / D-01: on the 882 tensors of the base (175 biases, 193 text norm scales + 58 vision norm scales,
    1 embedding, 1 head, 48 router gates, 192 attention, 96 expert and 118 vision matrices) ``decay_groups='qwen_moe'``
    with ``lr_multipliers={'model.visual': 0.1}`` yields four optimizer groups: decayed 1.0x (288 = attention + experts),
    undecayed 1.0x (243 = text norms + embedding + head + routers), decayed 0.1x (118 vision matrices), undecayed 0.1x
    (233 = vision biases + vision norms). The parent built ONE group of 882 with decay on all."""
    from types import SimpleNamespace

    from cosmos_framework.posttrain.config.build import build_optimizer

    names = _qwen3_vl_moe_parameter_names()
    assert len(names) == 882
    net = _module_with_parameters(names, wrap_layers=wrap_layers)
    assert sum(1 for _ in net.parameters()) == 882
    opt = build_optimizer(
        SimpleNamespace(net=net),
        OptimizerConfig(
            name="adamw", lr=1e-5, weight_decay=0.1, decay_groups="qwen_moe", lr_multipliers={"model.visual": 0.1}
        ),
    )
    groups = {g["name"]: g for g in opt.param_groups}
    assert [g["name"] for g in opt.param_groups] == ["decay_lr1x", "no_decay_lr1x", "decay_lr0.1x", "no_decay_lr0.1x"]
    assert {name: len(g["params"]) for name, g in groups.items()} == {
        "decay_lr1x": 288,
        "no_decay_lr1x": 243,
        "decay_lr0.1x": 118,
        "no_decay_lr0.1x": 233,
    }
    assert groups["decay_lr1x"]["weight_decay"] == 0.1 and groups["no_decay_lr1x"]["weight_decay"] == 0.0
    assert groups["decay_lr0.1x"]["lr"] == pytest.approx(1e-6) and groups["decay_lr1x"]["lr"] == 1e-5
    assert opt.param_groups[0]["lr"] == 1e-5  # train/learning_rate reads group 0: the base lr
    by_id = {id(p): n for n, p in net.named_parameters()}
    undecayed = {by_id[id(p)] for key in ("no_decay_lr1x", "no_decay_lr0.1x") for p in groups[key]["params"]}
    assert all(
        n.endswith(".bias")
        or "norm" in n
        or "embed_tokens" in n
        or n.startswith("lm_head")
        or n.endswith(".mlp.gate.weight")
        for n in undecayed
    )
    assert sum(1 for n in undecayed if n.endswith(".bias")) == 175
    assert sum(1 for n in undecayed if n.endswith(".mlp.gate.weight")) == 48
    # Without multipliers the two classes alone; the one-group default is untouched.
    two = build_optimizer(
        SimpleNamespace(net=_module_with_parameters(names)),
        OptimizerConfig(name="adamw", weight_decay=0.1, decay_groups="qwen_moe"),
    )
    assert [(g["name"], len(g["params"])) for g in two.param_groups] == [("decay_lr1x", 406), ("no_decay_lr1x", 476)]
    one = build_optimizer(
        SimpleNamespace(net=_module_with_parameters(names)), OptimizerConfig(name="adamw", weight_decay=0.01)
    )
    assert len(one.param_groups) == 1 and len(one.param_groups[0]["params"]) == 882
    with pytest.raises(ValueError, match="qwen_moe"):
        build_optimizer(
            SimpleNamespace(net=_module_with_parameters(names)),
            OptimizerConfig(name="adamw", decay_groups="hf", lr_multipliers={"model.visual": 0.1}),
        )


def test_lr_multipliers_sharing_a_factor_build_each_group_once_in_first_seen_order() -> None:
    """Review round 142 P2: two ``lr_multipliers`` patterns sharing a factor must yield ONE pair of groups for it. The
    parent iterated ``[1.0, *values]`` with the repeat, appended the 0.1x groups twice and raised "optimizer decay/lr
    grouping lost parameters" whenever the shared factor matched parameters. ``{'model.visual.blocks': 0.1,
    'model.visual': 0.1}`` covers exactly the tower, so the canary record's four groups [288 / 243 / 118 / 233] are
    unchanged; a pattern with its own factor seen first (``lm_head`` at 0.5x, one undecayed tensor) places its group
    before the 0.1x pair: group order follows first sight, not value."""
    from types import SimpleNamespace

    from cosmos_framework.posttrain.config.build import build_optimizer

    names = _qwen3_vl_moe_parameter_names()
    shared = build_optimizer(
        SimpleNamespace(net=_module_with_parameters(names)),
        OptimizerConfig(
            name="adamw",
            lr=1e-5,
            weight_decay=0.1,
            decay_groups="qwen_moe",
            lr_multipliers={"model.visual.blocks": 0.1, "model.visual": 0.1},
        ),
    )
    assert [(g["name"], len(g["params"])) for g in shared.param_groups] == [
        ("decay_lr1x", 288),
        ("no_decay_lr1x", 243),
        ("decay_lr0.1x", 118),
        ("no_decay_lr0.1x", 233),
    ]
    three = build_optimizer(
        SimpleNamespace(net=_module_with_parameters(names)),
        OptimizerConfig(
            name="adamw",
            lr=1e-5,
            weight_decay=0.1,
            decay_groups="qwen_moe",
            lr_multipliers={"lm_head": 0.5, "model.visual.blocks": 0.1, "model.visual": 0.1},
        ),
    )
    assert [(g["name"], len(g["params"]), g["lr_multiplier"]) for g in three.param_groups] == [
        ("decay_lr1x", 288, 1.0),
        ("no_decay_lr1x", 242, 1.0),
        ("no_decay_lr0.5x", 1, 0.5),
        ("decay_lr0.1x", 118, 0.1),
        ("no_decay_lr0.1x", 233, 0.1),
    ]
    assert three.param_groups[2]["lr"] == pytest.approx(5e-6) and three.param_groups[2]["weight_decay"] == 0.0
    assert sum(len(g["params"]) for g in three.param_groups) == 882


def test_decay_grouping_references_on_the_real_parameter_names() -> None:
    """PIN, not a fixture (review R134-03): it passes on the parent commit too; it records the two references knob 4 now
    names. On the 882 names of the base: the Cosmos post-training recipe's rule
    (``cosmos3/utils/optimizer.py::_build_params_with_metadata`` with ``disable_weight_decay_for_1d_params=False`` and no
    skip pattern, the recipe entry's state) decays all 882; transformers' Trainer rule (``get_parameter_names`` with the
    forbidden patterns of 5.13's ``get_decay_parameter_names``) decays ``lm_head.weight``, ``embed_tokens.weight`` and every
    ``mlp.gate.weight`` while exempting biases and norm scales; ``qwen_moe`` exempts exactly those 50 among the >= 2-D
    tensors (48 gates, the embedding, the head), the divergence the recipe's table states."""
    pytest.importorskip("transformers", minversion="5.0")
    from types import SimpleNamespace

    from torch import nn
    from transformers.trainer_pt_utils import get_parameter_names

    from cosmos_framework.utils.generator.optimizer import _build_params_with_metadata
    from cosmos_framework.posttrain.config.build import qwen_moe_decays

    names = _qwen3_vl_moe_parameter_names()
    net = _module_with_parameters(names)
    tagged = _build_params_with_metadata(
        SimpleNamespace(net=net),  # the backbone optimizes the `net` subtree of its model holder
        keys_to_select=[],
        lr_multipliers={"model.visual": 0.1},
        base_lr=1e-5,
        base_weight_decay=0.1,
        disable_weight_decay_for_1d_params=False,
    )
    assert len(tagged) == 882 and all(meta.weight_decay == 0.1 for _, meta in tagged), (
        "the backbone decays every tensor"
    )
    # transformers 5.13 Trainer.get_decay_parameter_names: forbidden layer types + these name patterns.
    forbidden = [r"bias", r"layernorm", r"rmsnorm", r"(?:^|\.)norm(?:$|\.)", r"_norm(?:$|\.)"]
    hf_decayed = set(get_parameter_names(net, [nn.LayerNorm], forbidden))
    gates = {name for name, _ in names if name.endswith(".mlp.gate.weight")}
    assert len(gates) == 48
    assert {"lm_head.weight", "model.language_model.embed_tokens.weight"} | gates <= hf_decayed, "HF decays all three"
    assert (
        "model.language_model.norm.weight" not in hf_decayed and "model.visual.blocks.0.attn.qkv.bias" not in hf_decayed
    )
    exempt_nd = {
        name
        for name, parameter in net.named_parameters()
        if parameter.ndim >= 2 and not qwen_moe_decays(name, parameter)
    }
    assert exempt_nd == {"lm_head.weight", "model.language_model.embed_tokens.weight"} | gates and len(exempt_nd) == 50


def test_warmup_cosine_schedule_matches_the_backbone_scheduler_and_reaches_the_optimizer() -> None:
    """The SFT ``lr_schedule.kind='cosine'`` is ``imaginaire/functional/lr_scheduler.py::LambdaWarmUpCosineScheduler``
    (the Cosmos post-training recipe's ``lambdacosine``) step for step over A1's horizon, and it drives the optimizer:
    the first update runs at lr * f_start, the warmup ends at lr, the cycle ends at lr * f_min."""
    import torch

    from cosmos_framework.utils.functional.lr_scheduler import LambdaWarmUpCosineScheduler
    from cosmos_framework.posttrain.config.build import build_sft_lr_scheduler, warmup_cosine_factor
    from cosmos_framework.posttrain.config.sft import SFTLRScheduleConfig

    reference = LambdaWarmUpCosineScheduler(
        warm_up_steps=[1000], f_min=[0.1], f_max=[1.0], f_start=[0.01], cycle_lengths=[2847]
    )
    for step in range(0, 2847, 7):
        assert warmup_cosine_factor(
            step, warmup_steps=1000, cycle_steps=2847, f_start=0.01, f_min=0.1
        ) == pytest.approx(float(reference.schedule(step)), rel=1e-12, abs=1e-12)
    param = torch.nn.Parameter(torch.zeros(2))
    opt = torch.optim.AdamW([param], lr=1e-5)
    rl = SFTConfig(
        data=SFTDatasetConfig(path="x.jsonl"),
        max_steps=2847,
        optimizer=OptimizerConfig(lr=1e-5, warmup_steps=1000),
        lr_schedule=SFTLRScheduleConfig(kind="cosine", f_start=0.01, f_min=0.1),
    )
    scheduler = build_sft_lr_scheduler(opt, rl.optimizer, rl.lr_schedule, rl.max_steps)
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-7)  # the first update: lr * f_start
    opt.step()
    for _ in range(1000):
        scheduler.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-5)
    for _ in range(1847):
        scheduler.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-6)
    constant = SFTConfig(data=SFTDatasetConfig(path="x.jsonl"), max_steps=4, optimizer=OptimizerConfig(lr=1e-5))
    assert (
        build_sft_lr_scheduler(
            torch.optim.AdamW([param], lr=1e-5), constant.optimizer, constant.lr_schedule, constant.max_steps
        )
        is None
    )  # A1: no scheduler
    short = SFTConfig(
        data=SFTDatasetConfig(path="x.jsonl"),
        max_steps=2,
        optimizer=OptimizerConfig(lr=1e-5, warmup_steps=1000),
        lr_schedule=SFTLRScheduleConfig(kind="cosine"),
    )
    canary_opt = torch.optim.AdamW([param], lr=1e-5)
    canary = build_sft_lr_scheduler(
        canary_opt, short.optimizer, short.lr_schedule, short.max_steps
    )  # a 2-step canary of the 1,000-step warmup stays on the ramp
    assert canary_opt.param_groups[0]["lr"] == pytest.approx(1e-7)
    canary_opt.step()
    canary.step()
    assert canary_opt.param_groups[0]["lr"] == pytest.approx(1e-5 * (0.01 + 0.99 / 1000))


def test_optimizer_preflight_and_schedule_factory_live_below_the_entrypoint(tmp_path: Path, apex_present: None) -> None:
    """FIXTURE: on b1373b88db ``refuse_rounding_or_unavailable_optimizer``, ``build_sft_lr_scheduler`` and
    ``add_clip_metrics`` were defined in ``entrypoints/sft.py``. The optimizer/storage refusal lives beside
    ``build_optimizer`` and the SFT schedule factory beside the scheduler builders in ``config/build.py``; the clip
    record is the trainer's; the entrypoint defines none of the three and keeps role construction and the loop. The
    refusal keeps its behaviour on both seats through the new home."""
    import ast

    from cosmos_framework.posttrain.entrypoints import sft as entry

    defined = {
        node.name for node in ast.walk(ast.parse(Path(entry.__file__).read_text())) if isinstance(node, ast.FunctionDef)
    }
    assert not defined & {"refuse_rounding_or_unavailable_optimizer", "build_sft_lr_scheduler", "add_clip_metrics"}, (
        defined
    )
    # Imported after the behavioural assertion so that a tree with the three in the entrypoint fails on it, not on this
    # ImportError (review round 1 of fix round 1, finding F3).
    from cosmos_framework.posttrain.config.build import build_sft_lr_scheduler, refuse_rounding_or_unavailable_optimizer

    native = _native_sft_config(tmp_path, "qwen3_vl_moe")
    with pytest.raises(ValueError, match="fusedadam"):
        refuse_rounding_or_unavailable_optimizer(native, OptimizerConfig(name="adamw", lr=1e-5))
    refuse_rounding_or_unavailable_optimizer(native, OptimizerConfig(name="fusedadam", lr=1e-5))
    assert callable(build_sft_lr_scheduler)


def test_bf16_storage_freezes_large_scales_below_half_an_ulp() -> None:
    """PIN, not a fixture (review R134-06): it passes on the parent commit too, because it records pure bf16 arithmetic
    the framework did not change. Review R125-B-02, the mechanism on tiny tensors: apex FusedAdam keeps an fp32 master and refreshes the bf16 parameter
    as ``bf16(master)`` every step, so the stored value moves only when the master has drifted by more than half the
    parameter's bf16 ulp. A scale of 3.078125 (ulp 0.015625, the final RMSNorm's magnitude) is unchanged after 700 steps
    of 1e-5 (drift 0.007 < 0.0078) and flips after 800; a weight of 0.0036 (ulp 3.05e-5) moves at the first step. A1a's
    final norm: 0 of 2,048 elements changed in 1,401 steps -- its master drift stayed under half an ulp."""
    import torch

    def refresh(value: float, steps: int, update: float) -> tuple[float, float]:
        master = torch.tensor([value], dtype=torch.float32)
        stored = master.to(torch.bfloat16)
        for _ in range(steps):
            master = master - update
            stored = master.to(torch.bfloat16)  # the kernel's per-step refresh of the bf16 parameter from the master
        return float(master.item()), float(stored.item())

    ulp = float(torch.tensor([3.078125], dtype=torch.bfloat16).item()) - float(
        torch.nextafter(
            torch.tensor([3.078125], dtype=torch.bfloat16), torch.tensor([0.0], dtype=torch.bfloat16)
        ).item()
    )
    assert ulp == pytest.approx(0.015625)
    master, stored = refresh(3.078125, 700, 1e-5)
    assert (
        master == pytest.approx(3.078125 - 0.007, abs=5e-5) and stored == 3.078125
    )  # fp32 master: ~1e-5 accumulation error
    master, stored = refresh(3.078125, 800, 1e-5)
    assert stored == 3.0625  # the master crossed half an ulp: the stored value flips one ulp down
    master, stored = refresh(0.0036, 1, 1e-5)
    assert stored != float(torch.tensor([0.0036], dtype=torch.bfloat16).item())


def test_deterministic_knob_reaches_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    """PIN, not a fixture (review R134-06): it passes on the parent commit too, because the setup path it records
    predates the fix. Review R125-D-12: ``rl.deterministic=True`` makes setup call ``utils.determinism.enable_determinism``
    before the model is built; ``False`` (the recipe default) does not."""
    from types import SimpleNamespace

    from cosmos_framework.posttrain.entrypoints import sft as entry

    calls: list[str] = []
    monkeypatch.setattr("cosmos_framework.posttrain.utils.determinism.enable_determinism", lambda: calls.append("on"))
    monkeypatch.setattr(entry, "validate_config", lambda config: config.rl)
    monkeypatch.setattr("cosmos_framework.utils.config.load_config", lambda path, opts: SimpleNamespace(rl=state["rl"]))
    monkeypatch.setattr(
        "cosmos_framework.posttrain.control.checkpointing.CheckpointLifecycle.from_config", lambda config, rl: None
    )
    monkeypatch.setattr(
        "cosmos_framework.posttrain.roles.env.sft.build_sft_env",
        lambda *a, **k: SimpleNamespace(source_fingerprint="x", dataset=[]),
    )
    monkeypatch.setattr(entry, "build_trainer", lambda *a, **k: "trainer")
    state: dict[str, Any] = {}
    for deterministic in (False, True):
        state["rl"] = SimpleNamespace(
            deterministic=deterministic, seed=0, data=None, validation=SimpleNamespace(enabled=False)
        )
        bodies = entry.make_setup("recipe.py", [])()
        assert bodies["policy"] == "trainer"
    assert calls == ["on"]
