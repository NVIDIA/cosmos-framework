# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Local-first Edge vision tower / processor plumbing and curated vision errors."""

import copy
import json
import types
from pathlib import Path

import pytest

from cosmos_framework.inference.args import (
    ModelMode,
    OmniSampleOverrides,
    _reasoner_vision_capable,
)
from cosmos_framework.inference.inference import (
    _bundled_processor_serves_node,
    _checkpoint_has_processor_files,
    _point_tokenizer_node_at_dir,
)
from cosmos_framework.inference.model import _raise_on_missing_vision_keys
from cosmos_framework.model.generator.mot.unified_mot import _load_bundled_vision_tower

_VISION_SPEC = {
    "vision_config": {"hidden_size": 16},
    "projector_config": {
        "spatial_merge_size": 2,
        "input_hidden_size": 16,
        "merger_intermediate_size": 32,
        "out_hidden_size": 8,
    },
}
_TOKEN_IDS = {"image_token_id": 1, "video_token_id": 2, "vision_start_token_id": 3}


def _write_bundle(root: Path, *, standalone: dict | None, top_level: dict | None) -> None:
    ve_dir = root / "vision_encoder"
    ve_dir.mkdir(parents=True)
    (ve_dir / "model.safetensors").write_bytes(b"stub")
    if standalone is not None:
        (ve_dir / "config.json").write_text(json.dumps(standalone), encoding="utf-8")
    if top_level is not None:
        (root / "config.json").write_text(json.dumps(top_level), encoding="utf-8")


def test_load_bundled_vision_tower_absent(tmp_path: Path):
    assert _load_bundled_vision_tower(None) is None
    assert _load_bundled_vision_tower(str(tmp_path)) is None  # no vision_encoder/


def test_load_bundled_vision_tower_exported_bundle(tmp_path: Path):
    # export_model --vit layout: self-describing standalone config (spec + token ids).
    _write_bundle(tmp_path, standalone={**_VISION_SPEC, **_TOKEN_IDS}, top_level={"model": {}})

    bundled = _load_bundled_vision_tower(str(tmp_path))
    assert bundled is not None
    weights_file, ve_cfg, top_cfg = bundled
    assert weights_file == str(tmp_path / "vision_encoder" / "model.safetensors")
    assert ve_cfg["projector_config"]["out_hidden_size"] == 8
    # Self-describing: token ids come from the bundle, never the root config.json.
    assert top_cfg is ve_cfg
    assert top_cfg["image_token_id"] == 1


def test_load_bundled_vision_tower_hub_snapshot(tmp_path: Path):
    # Raw hub snapshot layout: no standalone file; spec + ids folded into the top-level config.
    _write_bundle(tmp_path, standalone=None, top_level={**_VISION_SPEC, **_TOKEN_IDS})

    bundled = _load_bundled_vision_tower(str(tmp_path))
    assert bundled is not None
    _, ve_cfg, top_cfg = bundled
    assert ve_cfg["vision_config"] == _VISION_SPEC["vision_config"]
    assert top_cfg["video_token_id"] == 2


def test_load_bundled_vision_tower_standalone_without_token_ids(tmp_path: Path):
    # Older standalone layout: spec-only vision_encoder/config.json; ids live top-level.
    _write_bundle(tmp_path, standalone=_VISION_SPEC, top_level={**_VISION_SPEC, **_TOKEN_IDS})

    bundled = _load_bundled_vision_tower(str(tmp_path))
    assert bundled is not None
    _, ve_cfg, top_cfg = bundled
    assert "image_token_id" not in ve_cfg
    assert top_cfg["vision_start_token_id"] == 3


def _vlm_model_config(*, target: str = "", model_name: str = "", include_visual=None, model_instance="auto"):
    if model_instance == "auto":
        model_instance = {"_target_": target, "config": {"include_visual": include_visual}}
    return types.SimpleNamespace(vlm_config=types.SimpleNamespace(model_name=model_name, model_instance=model_instance))


def test_reasoner_vision_capable_edge_lazy_tower():
    # Edge: include_visual=None BY DESIGN; SigLIP2 tower loads lazily -> capable.
    config = _vlm_model_config(
        target="cosmos3._src.vfm.models.mot.unified_mot.Nemotron3DenseVLTextForCausalLM",
        model_name="nvidia/Cosmos3-Edge-Reasoner",
        include_visual=None,
    )
    assert _reasoner_vision_capable(config)


def test_reasoner_vision_capable_qwen_with_visual():
    config = _vlm_model_config(
        target="cosmos3._src.vfm.models.mot.unified_mot.Qwen3VLTextForCausalLM",
        model_name="nvidia/Cosmos3-Nano-Reasoner",
        include_visual=True,
    )
    assert _reasoner_vision_capable(config)


def test_reasoner_vision_capable_qwen_without_visual():
    config = _vlm_model_config(
        target="cosmos3._src.vfm.models.mot.unified_mot.Qwen3VLTextForCausalLM",
        model_name="nvidia/Cosmos3-Nano-Reasoner",
        include_visual=False,
    )
    assert not _reasoner_vision_capable(config)


def test_reasoner_vision_capable_qwen_llm_without_model_instance():
    config = _vlm_model_config(model_name="Qwen/Qwen3-0.6B", model_instance=None)
    assert not _reasoner_vision_capable(config)


def test_reasoner_vision_capable_unknown_family_never_blocks():
    config = _vlm_model_config(target="some.other.Backbone", model_name="acme/Custom-1B")
    assert _reasoner_vision_capable(config)


def test_build_reasoner_data_fails_fast_for_visionless_qwen(tmp_path: Path):
    img = tmp_path / "cat.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0")  # minimal non-empty file; not actually decoded here
    overrides = OmniSampleOverrides(name="cat", prompt="describe", vision_path=str(img))
    sample_meta = types.SimpleNamespace(model_mode=ModelMode.REASONER)

    visionless = _vlm_model_config(
        target="cosmos3._src.vfm.models.mot.unified_mot.Qwen3VLTextForCausalLM",
        model_name="nvidia/Cosmos3-Nano-Reasoner",
        include_visual=False,
    )
    with pytest.raises(ValueError, match="visual tower"):
        overrides._build_reasoner_data(model_config=visionless, sample_meta=sample_meta)

    # Text-only reasoner prompts remain valid on vision-less checkpoints.
    text_only = OmniSampleOverrides(name="txt", prompt="hello")
    text_only._build_reasoner_data(model_config=visionless, sample_meta=sample_meta)

    # Edge reasoner vision samples must not be blocked (lazy tower).
    edge = _vlm_model_config(
        target="cosmos3._src.vfm.models.mot.unified_mot.Nemotron3DenseVLTextForCausalLM",
        model_name="nvidia/Cosmos3-Edge-Reasoner",
        include_visual=None,
    )
    overrides._build_reasoner_data(model_config=edge, sample_meta=sample_meta)


def _write_root_index(root: Path, keys: list[str]) -> None:
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {key: "model-00001-of-00001.safetensors" for key in keys}}),
        encoding="utf-8",
    )


def test_raise_on_missing_vision_keys_no_vit_export(tmp_path: Path):
    _write_root_index(tmp_path, ["model.net.vae2llm.weight"])
    state_dict = {
        "model.net.vae2llm.weight": None,
        "model.net.language_model.visual.blocks.0.attn.qkv.weight": None,
    }
    with pytest.raises(ValueError, match="--no-vit"):
        _raise_on_missing_vision_keys(tmp_path, state_dict)


def test_raise_on_missing_vision_keys_tolerates_complete_and_visionless(tmp_path: Path):
    visual_key = "model.net.language_model.visual.blocks.0.attn.qkv.weight"

    # Checkpoint provides vision keys -> fine.
    _write_root_index(tmp_path, ["model.net.vae2llm.weight", visual_key])
    _raise_on_missing_vision_keys(tmp_path, {visual_key: None})

    # Model built without a visual tower (include_visual=false) -> fine.
    _raise_on_missing_vision_keys(tmp_path, {"model.net.vae2llm.weight": None})

    # No root index (e.g. DCP dir) -> defer to the loader.
    _raise_on_missing_vision_keys(tmp_path / "missing", {visual_key: None})


def test_checkpoint_has_processor_files(tmp_path: Path):
    assert not _checkpoint_has_processor_files(tmp_path)
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    assert _checkpoint_has_processor_files(tmp_path)

    other = tmp_path / "other"
    other.mkdir()
    (other / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    assert _checkpoint_has_processor_files(other)


def test_vlm_model_size_covers_edge():
    assert OmniSampleOverrides._VLM_MODEL_SIZE["nvidia/Cosmos3-Edge-Reasoner"] == "2B"
    # Edge training shift (Cosmos3-Edge.yaml): 256->3, 480->5, 720->10.
    assert OmniSampleOverrides._RESOLUTION_SHIFT_DEFAULTS[("2B", "480")] == 5.0


# ---------------------------------------------------------------------------
# Checkpoint-local processor override: node-shape dispatch + dispatchability gate
# ---------------------------------------------------------------------------

_LAZY_TARGET = "cosmos_framework.data.generator.processors.build_processor_lazy"
_LEGACY_TARGET = "cosmos_framework.configs.base.defaults.reasoner.create_qwen2_tokenizer_with_download"


def _edge_lazy_node() -> dict:
    # Edge SFT export shape (edge_model_config.py / Cosmos3-Edge.yaml).
    return {"_target_": _LAZY_TARGET, "repository": "nvidia/Cosmos3-Edge", "revision": "main"}


def _legacy_qwen2_node(config_variant: str = "hf") -> dict:
    # Nano/Super SFT export shape (nano_model_config.py / super_model_config.py).
    return {
        "_target_": _LEGACY_TARGET,
        "pretrained_model_name": "Qwen/Qwen3-VL-8B-Instruct",
        "config_variant": config_variant,
    }


def _write_json(root: Path, name: str, payload: dict | None = None) -> None:
    (root / name).write_text(json.dumps(payload or {}), encoding="utf-8")


def _write_edge_native_bundle(root: Path) -> None:
    # Renewed (5bb63b97-style) layout as bundled by export_model: the export
    # root config.json says cosmos3_omni, so processor_config.json alone must
    # carry the native-snapshot marker.
    _write_json(root, "config.json", {"model_type": "cosmos3_omni"})
    _write_json(root, "processor_config.json", {"processor_class": "Cosmos3EdgeProcessor"})
    _write_json(root, "tokenizer_config.json")
    _write_json(root, "preprocessor_config.json")


def _write_pre_renewal_edge_bundle(root: Path) -> None:
    # 28a0b8e-style layout: no processor_config.json; preprocessor_config.json
    # auto_maps to a processing.py that is never bundled -> AutoProcessor crash.
    _write_json(root, "config.json", {"model_type": "cosmos3_omni"})
    _write_json(root, "tokenizer_config.json")
    _write_json(root, "preprocessor_config.json", {"auto_map": {"AutoProcessor": "processing.NemotronNanoV3Processor"}})


def _write_qwen_autoprocessor_bundle(root: Path) -> None:
    _write_json(root, "preprocessor_config.json")
    _write_json(root, "tokenizer_config.json")
    _write_json(root, "tokenizer.json")


def _write_qwen2_slow_tokenizer_bundle(root: Path) -> None:
    # Minimal but real file set for Qwen2Tokenizer.from_pretrained (vocab_files_names
    # + tokenizer_config.json).
    _write_json(root, "vocab.json", {"a": 0, "b": 1, "ab": 2})
    (root / "merges.txt").write_text("#version: 0.2\na b\n", encoding="utf-8")
    _write_json(root, "tokenizer_config.json", {"model_max_length": 131072})


def test_edge_lazy_node_rewritten_for_native_bundle(tmp_path: Path):
    _write_edge_native_bundle(tmp_path)
    node = _edge_lazy_node()

    assert _checkpoint_has_processor_files(tmp_path)
    assert _bundled_processor_serves_node(node, tmp_path)
    assert _point_tokenizer_node_at_dir(node, tmp_path)
    assert node == {"_target_": _LAZY_TARGET, "tokenizer_type": str(tmp_path)}


def test_edge_lazy_node_gate_rejects_pre_renewal_bundle(tmp_path: Path):
    _write_pre_renewal_edge_bundle(tmp_path)
    node = _edge_lazy_node()
    before = copy.deepcopy(node)

    # Passes the coarse file check but is NOT dispatchable for the Edge node
    # (build_processor's dir mode needs the native-snapshot markers).
    assert _checkpoint_has_processor_files(tmp_path)
    assert not _bundled_processor_serves_node(node, tmp_path)
    assert node == before  # the gate never mutates


def test_legacy_qwen2_node_rewritten_to_local_dir(tmp_path: Path):
    _write_qwen2_slow_tokenizer_bundle(tmp_path)
    node = _legacy_qwen2_node()

    assert _bundled_processor_serves_node(node, tmp_path)
    assert _point_tokenizer_node_at_dir(node, tmp_path)
    # Only pretrained_model_name is rewritten; no tokenizer_type is injected and
    # config_variant="hf" is kept so download_tokenizer_files passes the local
    # dir through verbatim.
    assert node == {
        "_target_": _LEGACY_TARGET,
        "pretrained_model_name": str(tmp_path),
        "config_variant": "hf",
    }

    # The rewritten kwargs must match create_qwen2_tokenizer_with_download's
    # exact (pretrained_model_name, config_variant) signature and load the
    # bundled files from the local dir (the original defect was a TypeError:
    # unexpected keyword argument 'tokenizer_type').
    from cosmos_framework.configs.base.defaults.reasoner import create_qwen2_tokenizer_with_download
    from cosmos_framework.data.generator.processors import LLMTokenizerProcessor

    kwargs = {key: value for key, value in node.items() if key != "_target_"}
    processor = create_qwen2_tokenizer_with_download(**kwargs)
    assert isinstance(processor, LLMTokenizerProcessor)
    assert processor.tokenizer.get_vocab()["ab"] == 2


def test_legacy_qwen2_node_non_hf_variant_untouched(tmp_path: Path):
    _write_qwen2_slow_tokenizer_bundle(tmp_path)
    node = _legacy_qwen2_node(config_variant="gcp")
    before = copy.deepcopy(node)

    # Non-"hf" variants route pretrained_model_name into the object-store
    # download branch, which cannot take a local dir -> old (hub) behavior.
    assert not _bundled_processor_serves_node(node, tmp_path)
    assert not _point_tokenizer_node_at_dir(node, tmp_path)
    assert node == before


def test_legacy_qwen2_gate_requires_slow_tokenizer_files(tmp_path: Path):
    # tokenizer_config.json alone passes the coarse file check but
    # Qwen2Tokenizer.from_pretrained also needs vocab.json + merges.txt.
    _write_json(tmp_path, "tokenizer_config.json")
    assert _checkpoint_has_processor_files(tmp_path)
    assert not _bundled_processor_serves_node(_legacy_qwen2_node(), tmp_path)


def test_llm_backbone_lazy_node_untouched(tmp_path: Path):
    # Qwen3-0.6B LLM backbone: dir-mode build_processor would mis-dispatch to
    # Qwen3VLProcessor instead of LLMTokenizerProcessor -> never use the bundle.
    _write_qwen_autoprocessor_bundle(tmp_path)
    node = {"_target_": _LAZY_TARGET, "tokenizer_type": "Qwen/Qwen3-0.6B", "config_variant": "hf"}
    assert not _bundled_processor_serves_node(node, tmp_path)


def test_qwen3vl_tokenizer_type_lazy_node_gate(tmp_path: Path):
    node = {"_target_": _LAZY_TARGET, "tokenizer_type": "Qwen/Qwen3-VL-8B-Instruct", "config_variant": "hf"}

    # Missing preprocessor_config.json -> AutoProcessor cannot build the image
    # processor -> gate fails.
    _write_json(tmp_path, "tokenizer_config.json")
    _write_json(tmp_path, "tokenizer.json")
    assert not _bundled_processor_serves_node(node, tmp_path)

    _write_json(tmp_path, "preprocessor_config.json")
    assert _bundled_processor_serves_node(node, tmp_path)
    assert _point_tokenizer_node_at_dir(node, tmp_path)
    assert node == {"_target_": _LAZY_TARGET, "config_variant": "hf", "tokenizer_type": str(tmp_path)}


def test_repository_lazy_node_qwen_family_gate(tmp_path: Path):
    node = {"_target_": _LAZY_TARGET, "repository": "nvidia/Cosmos3-Nano", "revision": "main"}
    _write_qwen_autoprocessor_bundle(tmp_path)
    assert _bundled_processor_serves_node(node, tmp_path)

    # An edge-native bundle would re-route dir-mode dispatch to the Nemotron
    # bridge -- wrong for a Qwen-family node.
    _write_json(tmp_path, "processor_config.json", {"processor_class": "Cosmos3EdgeProcessor"})
    assert not _bundled_processor_serves_node(node, tmp_path)


def test_unknown_tokenizer_node_untouched(tmp_path: Path):
    _write_edge_native_bundle(tmp_path)
    node = {"_target_": "acme.processors.build_custom_processor", "repository": "acme/Custom", "revision": "main"}
    before = copy.deepcopy(node)

    assert not _bundled_processor_serves_node(node, tmp_path)
    assert not _point_tokenizer_node_at_dir(node, tmp_path)
    assert node == before

    # Fail-safe on malformed nodes too.
    assert not _bundled_processor_serves_node(None, tmp_path)
    assert not _point_tokenizer_node_at_dir(None, tmp_path)
