# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Portable adapter processor proofs using real tiny fast tokenizers and native packers."""

from __future__ import annotations

import copy
import json
import shutil
from types import SimpleNamespace

import pytest
from tokenizers import AddedToken, Tokenizer, models
from transformers import PreTrainedTokenizerFast

from cosmos_framework.posttrain.data.packer.sft import QwenSFTDataPacker
from cosmos_framework.posttrain.utils import adapter_source
from cosmos_framework.posttrain.utils.adapter_identity import adapter_processor_identity


class _Media:
    def __init__(self, size=256):
        self.size = size

    def to_dict(self):
        return {"size": self.size, "resample": 3}


def _source(tmp_path):
    root = tmp_path / "original"
    root.mkdir()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({"<unk>": 0, "<eos>": 1, "A": 2}, unk_token="<unk>")),
        unk_token="<unk>",
        eos_token="<eos>",
        pad_token="<eos>",
        chat_template="{% for message in messages %}{{ message.content }}{% endfor %}",
        model_max_length=256,
    )
    tokenizer.save_pretrained(root)
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_vl",
                "text_config": {"num_hidden_layers": 36, "hidden_size": 4096, "intermediate_size": 12288},
            }
        )
    )
    (root / "preprocessor_config.json").write_text('{"fixture": "image"}')
    (root / "video_preprocessor_config.json").write_text('{"fixture": "video"}')
    (root / "model.safetensors").write_bytes(b"small source-identity fixture, not model weights")
    identity = adapter_source._identity(root)
    (root / adapter_source.SOURCE_MANIFEST).write_text(json.dumps(identity))
    return root, identity


def _packer(root):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(root, local_files_only=True)
    tokenizer.init_kwargs["tokenizer_file"] = str(root / "tokenizer.json")
    processor = SimpleNamespace(
        chat_template=tokenizer.chat_template, image_processor=_Media(), video_processor=_Media()
    )
    return QwenSFTDataPacker(tokenizer, processor=processor, max_length=256)


def test_actual_tokenizer_relocation_keeps_portable_identity_and_raw_resume_paths(tmp_path):
    root, source = _source(tmp_path)
    relocated = tmp_path / "relocated"
    shutil.copytree(root, relocated)
    first, second = _packer(root), _packer(relocated)
    raw_first, raw_second = first.validation_semantics_identity(), second.validation_semantics_identity()
    assert raw_first != raw_second
    initial_kwargs = copy.deepcopy(first.tokenizer.init_kwargs)
    portable = adapter_processor_identity(first, source_path=root, source_identity=source)
    assert portable == adapter_processor_identity(second, source_path=relocated, source_identity=source)
    assert first.validation_semantics_identity() == raw_first
    assert first.tokenizer.init_kwargs == initial_kwargs
    assert portable["tokenizer"]["settings"]["name_or_path"]["revision"] == source["revision"]
    assert str(root) not in json.dumps(portable) and str(relocated) not in json.dumps(portable)


@pytest.mark.parametrize("change", ["padding", "template", "media", "max_length", "added_token", "semantic_string"])
def test_portable_identity_preserves_all_semantic_settings(tmp_path, change):
    root, source = _source(tmp_path)
    packer = _packer(root)
    initial = adapter_processor_identity(packer, source_path=root, source_identity=source)
    if change == "padding":
        packer.tokenizer.padding_side = "left"
    elif change == "template":
        packer.processor.chat_template += " changed"
    elif change == "media":
        packer.processor.video_processor.size = 512
    elif change == "max_length":
        packer.max_length = 128
    elif change == "added_token":
        packer.tokenizer.init_kwargs["added_tokens_decoder"][99] = AddedToken(
            "<changed>", normalized=False, special=True
        )
    else:
        # A path-shaped semantic string is retained, never blanket stripped.
        packer.tokenizer.init_kwargs["custom_semantic_value"] = str(root)
    assert adapter_processor_identity(packer, source_path=root, source_identity=source) != initial


@pytest.mark.parametrize(
    "change", ["external", "unmanifested", "symlink_escape", "wrong_artifact", "unknown", "nested"]
)
def test_artifact_references_fail_closed(tmp_path, change):
    root, source = _source(tmp_path)
    packer = _packer(root)
    outside = tmp_path / "tokenizer.json"
    outside.write_bytes((root / "tokenizer.json").read_bytes())
    if change == "external":
        packer.tokenizer.init_kwargs["tokenizer_file"] = str(outside)
    elif change == "unmanifested":
        packer.tokenizer.init_kwargs["tokenizer_file"] = str(root / "absent.json")
    elif change == "symlink_escape":
        alias = root / "escape"
        alias.symlink_to(outside)
        packer.tokenizer.init_kwargs["tokenizer_file"] = str(alias)
    elif change == "wrong_artifact":
        packer.tokenizer.init_kwargs["tokenizer_file"] = str(root / "config.json")
    elif change == "unknown":
        packer.tokenizer.init_kwargs["novel_file"] = str(outside)
    else:
        packer.tokenizer.init_kwargs["novel_options"] = {"tokenizer_file": str(outside)}
    with pytest.raises(ValueError, match="artifact"):
        adapter_processor_identity(packer, source_path=root, source_identity=source)


def test_source_directory_identity_and_changed_content_cannot_be_blessed(tmp_path):
    root, source = _source(tmp_path)
    packer = _packer(root)
    wrong = {**source, "revision": "0" * 40}
    with pytest.raises(ValueError, match="identity differs"):
        adapter_processor_identity(packer, source_path=root, source_identity=wrong)
    packer.tokenizer.name_or_path = str(tmp_path)
    with pytest.raises(ValueError, match="source directory"):
        adapter_processor_identity(packer, source_path=root, source_identity=source)
    packer.tokenizer.name_or_path = str(root)
    (root / "tokenizer.json").write_text('{"tampered": true}')
    with pytest.raises(ValueError, match="contents differ"):
        adapter_processor_identity(packer, source_path=root, source_identity=source)


def test_bare_manifest_file_and_absolute_file_have_identical_identity(tmp_path):
    root, source = _source(tmp_path)
    packer = _packer(root)
    expected = adapter_processor_identity(packer, source_path=root, source_identity=source)
    packer.tokenizer.init_kwargs["tokenizer_file"] = "tokenizer.json"
    assert adapter_processor_identity(packer, source_path=root, source_identity=source) == expected


def test_hydra_manifest_preserves_nested_content_identity(tmp_path):
    from omegaconf import OmegaConf

    root, source = _source(tmp_path)
    packer = _packer(root)
    configured = OmegaConf.create(source)
    assert adapter_processor_identity(
        packer, source_path=root, source_identity=configured
    ) == adapter_processor_identity(packer, source_path=root, source_identity=source)
    configured.files["tokenizer.json"].sha256 = "0" * 64
    with pytest.raises(ValueError, match="identity differs"):
        adapter_processor_identity(packer, source_path=root, source_identity=configured)
