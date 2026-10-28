# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""HF verification with real local configs/processors and tiny synthetic tensor shards."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from cosmos_framework.posttrain.checkpoint.conversion import hf

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _fixture(tmp_path: Path, *, moe: bool = False) -> tuple[Path, Path, Path]:
    import torch
    from safetensors.torch import save_file

    pytest.importorskip("transformers", minversion="5.8.1")
    from transformers import Qwen3_5Config, Qwen3VLMoeConfig

    from cosmos_framework.posttrain.tests.helpers.sft_packer import local_processor, local_tokenizer

    base, out, checkpoint = tmp_path / "base", tmp_path / "export", tmp_path / "model.pt"
    config = Qwen3VLMoeConfig() if moe else Qwen3_5Config()
    config.architectures = ["Qwen3VLMoeForConditionalGeneration" if moe else "Qwen3_5ForConditionalGeneration"]
    config.save_pretrained(base)
    processor = local_processor(local_tokenizer())
    processor.save_pretrained(base)
    processor.image_processor.save_pretrained(base)
    state = {
        "model.language_model.norm.weight": torch.tensor([1.0, 1.125, 0.875, 1.25]),  # [4]
        "lm_head.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),  # [3,4]
    }
    if moe:
        state["model.language_model.layers.0.mlp.experts.gate_up_proj"] = torch.arange(24, dtype=torch.float32).reshape(
            2, 3, 4
        )  # [experts,in,out]
    shard = "model-00001-of-00001.safetensors"
    save_file(state, base / shard)
    (base / hf.INDEX_FILE).write_text(json.dumps({"weight_map": dict.fromkeys(state, shard)}))
    torch.save({"net": state}, checkpoint)
    shutil.copytree(base, out)
    return base, out, checkpoint


@pytest.mark.parametrize("moe", [False, True], ids=["qwen3_5_dense", "qwen3_vl_moe"])
def test_verify_hf_dense_and_moe(tmp_path: Path, moe: bool) -> None:
    base, out, checkpoint = _fixture(tmp_path, moe=moe)
    report = hf.verify_with_torch(str(out), str(base), str(checkpoint))
    assert report["expert_shape_checks"] == int(moe)
    # Every tensor is compared and digested, not one probe.
    assert report["bitwise_probe"] == "all_tensors"
    assert report["bitwise_equal_tensors"] == 2 + int(moe)
    assert set(report["tensor_sha256"]) == {"model.language_model.norm.weight", "lm_head.weight"} | (
        {"model.language_model.layers.0.mlp.experts.gate_up_proj"} if moe else set()
    )
    assert all(len(d) == 64 for d in report["tensor_sha256"].values())
    assert report["tensors_compared_to_base"] == 2 + int(moe) and report["tensors_equal_to_base"] == 2 + int(moe)
    assert report["tensors_moved_vs_base"] == 0
    assert report["processor_identity"] == "equal"


@pytest.mark.parametrize("corruption", ["weights", "keys", "config", "processor"])
def test_verify_dense_rejects_corruption(tmp_path: Path, corruption: str) -> None:
    from safetensors.torch import load_file, save_file

    base, out, checkpoint = _fixture(tmp_path)
    if corruption in {"weights", "keys"}:
        index = json.loads((out / hf.INDEX_FILE).read_text())
        shard = out / next(iter(index["weight_map"].values()))
        state = load_file(shard)
        if corruption == "weights":
            state["model.language_model.norm.weight"][0] += 1
            message = "differs bitwise"
        else:
            state.pop("lm_head.weight")
            index["weight_map"].pop("lm_head.weight")
            (out / hf.INDEX_FILE).write_text(json.dumps(index))
            message = "weight_map != model.pt keys"
        save_file(state, shard)
    elif corruption == "config":
        config = json.loads((out / "config.json").read_text())
        config["architectures"] = ["WrongArchitecture"]
        (out / "config.json").write_text(json.dumps(config))
        message = "config.json identity"
    else:
        template = out / "chat_template.jinja"
        template.write_text(template.read_text() + "different template")
        message = "chat_template.*differs"
    with pytest.raises(hf.ConversionError, match=message):
        hf.verify_with_torch(str(out), str(base), str(checkpoint))


def test_verify_moe_still_requires_experts_from_base(tmp_path: Path) -> None:
    import torch
    from safetensors.torch import load_file, save_file

    base, out, checkpoint = _fixture(tmp_path, moe=True)
    index = json.loads((out / hf.INDEX_FILE).read_text())
    shard = out / next(iter(index["weight_map"].values()))
    state = {name: tensor for name, tensor in load_file(shard).items() if ".mlp.experts." not in name}
    save_file(state, shard)
    index["weight_map"] = {name: value for name, value in index["weight_map"].items() if name in state}
    (out / hf.INDEX_FILE).write_text(json.dumps(index))
    # Both checkpoint and output lack experts, so exact output/source key equality
    # alone cannot catch the missing MoE state. The original base still has experts.
    torch.save({"net": state}, checkpoint)
    with pytest.raises(hf.ConversionError, match="no fused MoE expert tensors"):
        hf.verify_with_torch(str(out), str(base), str(checkpoint))


def test_verify_moe_still_checks_expert_layout(tmp_path: Path) -> None:
    from safetensors.torch import load_file, save_file

    base, out, checkpoint = _fixture(tmp_path, moe=True)
    index = json.loads((out / hf.INDEX_FILE).read_text())
    key = next(name for name in index["weight_map"] if ".mlp.experts." in name)
    shard = out / index["weight_map"][key]
    state = load_file(shard)
    state[key] = state[key].transpose(1, 2).contiguous()  # [experts,out,in], incorrect on-disk layout
    save_file(state, shard)
    with pytest.raises(hf.ConversionError, match="expert .* shape .* != checkpoint shape"):
        hf.verify_with_torch(str(out), str(base), str(checkpoint))


@pytest.mark.parametrize("min_tensors", [2, 3])
def test_dense_staged_materialization_verifies_and_honors_count_floor(tmp_path: Path, min_tensors: int) -> None:
    from cosmos_framework.posttrain.checkpoint.tools import materialize_hf as mat

    base, out, checkpoint = _fixture(tmp_path)
    staging = Path(str(out) + ".tmp")
    out.rename(staging)
    identity = mat.make_identity(mat.probe_local_source(str(checkpoint.parent)), mat.base_identity(str(base)))
    (staging / mat.SOURCE_STAMP).write_text(json.dumps({"identity": identity}))
    kwargs = {"base": str(base), "out": str(out), "ckpt_dir": str(checkpoint.parent), "min_tensors": min_tensors}
    if min_tensors > 2:
        with pytest.raises(hf.ConversionError, match="fewer than the 3 floor"):
            mat.run_request(**kwargs)
        assert not out.exists()
    else:
        # Reuse real local shards rather than rebuilding a model; leave tensor verification enabled.
        assert mat.run_request(**kwargs) == "built"
        marker = json.loads((out / mat.MATERIALIZED_MARKER).read_text())
        assert marker["reused_staging"]
        assert marker["verify"]["torch"] and marker["verify"]["expert_shape_checks"] == 0
        assert marker["verify"]["processor_identity"] == "equal"


def test_verify_rejects_a_changed_tensor_the_old_probe_never_read(tmp_path: Path) -> None:
    """The parent probed ``model.language_model.norm.weight`` alone, so a shard whose ``lm_head.weight``
    differs from ``model.pt`` passed; now every tensor is compared and a moved tensor is counted against the base."""
    import torch
    from safetensors.torch import load_file, save_file

    base, out, checkpoint = _fixture(tmp_path)
    index = json.loads((out / hf.INDEX_FILE).read_text())
    shard = out / next(iter(index["weight_map"].values()))
    state = load_file(shard)
    state["lm_head.weight"][0, 0] += 1  # not the old probe tensor
    save_file(state, shard)
    with pytest.raises(hf.ConversionError, match="1 of 2 tensors differ bitwise.*lm_head.weight"):
        hf.verify_with_torch(str(out), str(base), str(checkpoint))
    # A checkpoint that moved lm_head (saved consistently into the shard) verifies, and the base comparison says so.
    torch.save({"net": state}, checkpoint)
    report = hf.verify_with_torch(str(out), str(base), str(checkpoint))
    assert report["tensors_equal_to_base_names"] == ["model.language_model.norm.weight"]
    assert report["tensors_moved_vs_base"] == 1
    assert report["tensor_sha256"]["lm_head.weight"] == hf.tensor_sha256(state["lm_head.weight"])
    assert hf.tensor_sha256(torch.tensor([1.0, 2.0], dtype=torch.bfloat16)) != hf.tensor_sha256(
        torch.tensor([1.0, 2.5], dtype=torch.bfloat16)
    )


def test_materializer_writes_the_base_config_and_keeps_the_transformers_one(tmp_path: Path) -> None:
    """Transformers 5 re-serializes the Qwen3-VL-MoE config (``num_local_experts`` for
    ``num_experts``, ``rope_parameters`` for ``rope_scaling`` + ``rope_theta``, ``text_config.tie_word_embeddings: true``
    beside the top-level ``false``), unreadable as the same model by the service's transformers 4.57. The export's
    ``config.json`` becomes the base's bytes; the transformers-written one is kept once as ``config.json.trainer.bak``
    (the export's own file before the base overlay, excluded from publication); the overlay is idempotent."""
    pytest.importorskip("transformers", minversion="5.8.1")
    from transformers import AutoConfig

    fixture = Path(__file__).resolve().parent / "fixtures" / "qwen3_vl_moe_base_config.json"
    base, out = tmp_path / "base", tmp_path / "export"
    base.mkdir()
    out.mkdir()
    shutil.copy2(fixture, base / "config.json")
    AutoConfig.from_pretrained(str(base)).save_pretrained(str(out))  # what save_pretrained writes under transformers 5
    rewritten = json.loads((out / "config.json").read_text())
    assert "num_local_experts" in rewritten["text_config"] or "num_experts" not in rewritten["text_config"]
    assert rewritten["text_config"].get("tie_word_embeddings") is True and rewritten["tie_word_embeddings"] is False
    result = hf.write_config_from_base(str(out), str(base))
    assert result["config_source"] == "base" and result["transformers_config_kept_as"] == "config.json.trainer.bak"
    assert (out / "config.json").read_bytes() == fixture.read_bytes()
    kept = json.loads((out / "config.json.trainer.bak").read_text())
    assert kept == rewritten
    restored = json.loads((out / "config.json").read_text())
    assert restored["text_config"]["num_experts"] == 128 and "num_local_experts" not in restored["text_config"]
    assert "tie_word_embeddings" not in restored["text_config"] and restored["tie_word_embeddings"] is False
    assert (
        restored["text_config"]["rope_scaling"]["mrope_section"] == [24, 20, 20]
        and restored["text_config"]["rope_theta"] == 5000000
    )
    again = hf.write_config_from_base(str(out), str(base))
    assert again["kept_now"] is False and json.loads((out / "config.json.trainer.bak").read_text()) == rewritten
    from cosmos_framework.posttrain.checkpoint.hf_export import export_files

    assert "config.json" in export_files(str(out)) and "config.json.trainer.bak" not in export_files(str(out))
    bad = tmp_path / "other"
    bad.mkdir()
    other = json.loads(fixture.read_text())
    other["text_config"]["hidden_size"] = 1024
    (bad / "config.json").write_text(json.dumps(other))
    with pytest.raises(hf.ConversionError, match="identity mismatch"):
        hf.write_config_from_base(str(out), str(bad))


def test_service_processor_config_follows_the_base_sidecars(tmp_path: Path) -> None:
    """A service ``processor_config.json`` is composed from the base's own image and video sidecars (the
    fast image processor class the trainer used, the base's video defaults), never from a run's per-call kwargs."""
    base, _out, _checkpoint = _fixture(tmp_path)
    config = hf.service_processor_config(str(base))
    image = json.loads((base / "preprocessor_config.json").read_text())
    assert config["processor_class"] == "Qwen3VLProcessor"
    # The class name is the base's own, verbatim (transformers 5.x spells the fast class `Qwen2VLImageProcessor`,
    # 4.57 spells it `Qwen2VLImageProcessorFast`; a 4.57 service reads the former as the slow PIL class).
    assert config["image_processor"]["image_processor_type"] == image["image_processor_type"]
    assert "processor_class" not in config["image_processor"]
    if (base / "video_preprocessor_config.json").is_file():
        video = json.loads((base / "video_preprocessor_config.json").read_text())
        assert config["video_processor"]["video_processor_type"] == video["video_processor_type"]
        assert "processor_class" not in config["video_processor"]
    assert "num_frames" not in json.dumps(config)
    # The Cosmos 3.5 Super base's sidecars (written by transformers 4.57.0.dev0): the service file must say Fast.
    real = tmp_path / "real_base"
    real.mkdir()
    (real / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "size": {"longest_edge": 16777216, "shortest_edge": 65536},
                "patch_size": 16,
                "temporal_patch_size": 2,
                "merge_size": 2,
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
                "processor_class": "Qwen3VLProcessor",
                "image_processor_type": "Qwen2VLImageProcessorFast",
            }
        )
    )
    (real / "video_preprocessor_config.json").write_text(
        json.dumps(
            {
                "size": {"longest_edge": 25165824, "shortest_edge": 4096},
                "patch_size": 16,
                "temporal_patch_size": 2,
                "merge_size": 2,
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
                "processor_class": "Qwen3VLProcessor",
                "video_processor_type": "Qwen3VLVideoProcessor",
            }
        )
    )
    service = hf.service_processor_config(str(real))
    assert service["image_processor"]["image_processor_type"] == "Qwen2VLImageProcessorFast"
    assert service["video_processor"]["video_processor_type"] == "Qwen3VLVideoProcessor"
    assert service["image_processor"]["size"] == {"longest_edge": 16777216, "shortest_edge": 65536}
    assert "fps" not in service["video_processor"] and "max_frames" not in service["video_processor"]


@pytest.mark.parametrize("scenario", ["path_replaced_build_correct", "shards_follow_the_path", "wrong_shards"])
def test_default_verification_consumes_the_held_payload_when_the_path_is_replaced(
    tmp_path: Path, scenario: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The default verification must compare the staged shards against the payload the
    materializer hashed under its lock and the overlay consumed, not against a reopened path. The reviewer's scenario with the
    SUPPORTED merge tool and the REAL torch verifier: merge -> A; during the overlay a forced merge replaces the path with B
    (os.remove, tmp, os.replace); the staged shards are A (the overlay consumed the held payload; here the real overlay minus the
    HF model: the pre-made export of A is copied into the staging directory). [path_replaced_build_correct]: the build publishes,
    verified bitwise against A, marker A, the path holds B; on 65fe976387c verification reopened the path and REFUSED the correct
    build ("verify: 2 of 2 tensors differ bitwise"). [shards_follow_the_path]: an overlay that had read the path would have written
    B; verification against the payload A refuses that wrong build; on 65fe976387c it PASSED (B compared with B at the path).
    [wrong_shards]: no replacement, one staged tensor corrupted: verification refuses on both (the check keeps its teeth)."""
    import hashlib
    import os

    import torch
    from safetensors.torch import load_file, save_file

    from cosmos_framework.posttrain.checkpoint.tools import materialize_hf as mat
    from cosmos_framework.posttrain.checkpoint.tools import merge as merge_mod

    base, export, checkpoint = _fixture(tmp_path)  # export: what the overlay produces for A (a copy of the base's tree)
    state_a = {k: v.clone() for k, v in merge_mod.load_expert(str(checkpoint)).items()}  # values: [*shape]
    state_b = {k: v + 1 for k, v in state_a.items()}  # values: [*shape]

    def expert(name: str, state: dict[str, torch.Tensor]) -> str:
        path = tmp_path / f"{name}.pt"
        torch.save({"net": state}, path)
        return str(path)

    experts_a = [expert("a1", state_a), expert("a2", state_a)]
    experts_b = [expert("b1", state_b), expert("b2", state_b)]
    ckpt_dir = tmp_path / "iter_000300"  # the merge tool binds the request to an iter_* directory
    model_pt = ckpt_dir / "model.pt"
    assert merge_mod.run_request(experts_a, "avg", 1.0, None, str(model_pt), None) == "built"
    sha_a = hashlib.sha256(model_pt.read_bytes()).hexdigest()
    shard = next(iter(json.loads((export / hf.INDEX_FILE).read_text())["weight_map"].values()))

    def overlay(ckpt_dir_: str, model_id: str, staging_dir: str, payload: object = None) -> str:
        shutil.copytree(export, staging_dir)  # the overlay's product for A
        if scenario == "wrong_shards":
            staged = load_file(os.path.join(staging_dir, shard))
            staged["lm_head.weight"][0, 0] += 1
            save_file(staged, os.path.join(staging_dir, shard))
            return staging_dir
        # the supported replacement during the overlay: a forced merge puts B at the path
        path = os.path.join(ckpt_dir_, "model.pt")
        assert merge_mod.run_request(experts_b, "avg", 1.0, None, path, None, force=True) == "built"
        if (
            scenario == "shards_follow_the_path"
        ):  # what an overlay that read the PATH would have written: the wrong build
            save_file({k: v.clone() for k, v in merge_mod.load_expert(path).items()}, os.path.join(staging_dir, shard))
        return staging_dir

    monkeypatch.setattr(hf, "overlay_consolidated_onto_hf", overlay)
    out = tmp_path / "teacher"
    code = mat.main(["--ckpt-dir", str(ckpt_dir), "--base", str(base), "--out", str(out), "--min-tensors", "2"])
    err = capsys.readouterr().err
    if scenario == "path_replaced_build_correct":
        assert code == 0, (
            "a correct build verifies against the held payload (on 65fe976387c: refused, 'verify: 2 of 2 tensors differ "
            f"bitwise', because verification reopened the path, by then B): {err}"
        )
        marker = json.loads((out / mat.MATERIALIZED_MARKER).read_text())
        assert marker["identity"]["source_sha256"] == sha_a and marker["source"]["sha256"] == sha_a
        assert marker["verify"]["torch"] and marker["verify"]["bitwise_equal_tensors"] == 2
        published = load_file(out / shard)
        assert all(torch.equal(published[k], state_a[k]) for k in state_a), "the published tensors are A's"
        assert hashlib.sha256(model_pt.read_bytes()).hexdigest() != sha_a, "the path holds B"
        at_path = merge_mod.load_expert(str(model_pt))
        assert all(torch.equal(at_path[k], state_b[k]) for k in state_b)
    else:
        assert code == 1, (
            "a wrong build is refused (on 65fe976387c the shards-follow-the-path case PASSED: B compared with B at the "
            f"path): {err}"
        )
        assert "FATAL: the staged build failed at the verification (torch) stage" in err and "differ bitwise" in err, (
            err
        )
        assert not out.exists() and not Path(str(out) + ".tmp").exists()
