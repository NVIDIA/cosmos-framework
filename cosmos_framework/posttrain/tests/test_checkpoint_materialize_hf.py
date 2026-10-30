# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU tests for the identity-bound HF materialization (``checkpoint.tools.materialize_hf``) and the torch-free directory
checks of ``checkpoint.conversion.hf`` it runs (sidecar sync and verification, index verification).

Everything runs on synthetic files: minimal valid safetensors shards, a fake HF base with the eight sidecars the real
base carries, and an in-memory object store for the archived-artifact probe. ``main()`` is driven end to end with the
overlay replaced by a fake that writes such a directory; ``overlay_consolidated_onto_hf`` itself needs a real model
and is not exercised here. Torch-free, except the merge-race fixture, which runs the real merge tool and the
real consolidated-checkpoint loader (torch) in a subprocess and skips where torch is absent; the pinned ``safetensors`` (a
declared dependency) must be importable: the materializer's preflight asks that reader for every base
shard and refuses, fail closed, where it is absent.
"""

from __future__ import annotations

import argparse
import builtins
import contextlib
import errno
import hashlib
import json
import os
import re
import shutil
import socket
import struct
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from cosmos_framework.posttrain.checkpoint.artifact import CHECKPOINT_ARTIFACT_MANIFEST, build_artifact_manifest
from cosmos_framework.posttrain.checkpoint.conversion import hf as hf_overlay
from cosmos_framework.posttrain.checkpoint.conversion.hf import ConversionError
from cosmos_framework.posttrain.checkpoint.io import atomic_write_json
from cosmos_framework.posttrain.checkpoint.tools import materialize_hf as mat
from cosmos_framework.posttrain.tests.helpers.object_store_fake import FakeS3Client
from cosmos_framework.posttrain.utils import object_store

pytestmark = [pytest.mark.L0, pytest.mark.CPU]

SHA_ZERO10 = "01d448afd928065458cf670b60f5a594d735af0172c8d67f22a81680132681ca"  # sha256 of ten zero bytes
SHARD1, SHARD2 = "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"
WMAP = {"a.weight": SHARD1, "b.mlp.experts.gate_up_proj": SHARD2, "c.bias": SHARD2}
SHARD2_KEYS = {"b.mlp.experts.gate_up_proj", "c.bias"}
# The fake processor template carries the real template's reasoning line (the one line --unified-chat-template changes).
FAKE_PROCESSOR_TEMPLATE = "BASE-PROCESSOR-TEMPLATE\n" + hf_overlay.PROCESSOR_REASONING_LINE + "\nEND"
FAKE_CANONICAL_TEMPLATE = "BASE-PROCESSOR-TEMPLATE\n" + hf_overlay.CANONICAL_REASONING_LINE + "\nEND"
# The eight sidecars the real base carries (distinct contents), and no chat_template.jinja / processor_config.json.
BASE_FILES = {
    "tokenizer.json": '{"base": "tokenizer"}',
    "tokenizer_config.json": '{"chat_template": "BASE-TOKENIZER-TEMPLATE"}',
    "chat_template.json": json.dumps({"chat_template": FAKE_PROCESSOR_TEMPLATE}),
    "merges.txt": "a b\n",
    "vocab.json": '{"a": 0}',
    "preprocessor_config.json": '{"image_processor_type": "X", "processor_class": "P"}',
    "video_preprocessor_config.json": '{"video_processor_type": "Y"}',
    "generation_config.json": '{"top_p": 0.8}',
}
# The structural identity that config.json <- base requires through validate_model_identity (model_type, architectures
# and the text block's vocab_size / hidden_size / num_hidden_layers), in the real base's shape (a text_config block).
BASE_CONFIG = json.dumps(
    {
        "architectures": ["FakeForConditionalGeneration"],
        "model_type": "fake",
        "text_config": {"hidden_size": 4, "num_hidden_layers": 1, "vocab_size": 8},
    }
)
URI = "s3://bucket/proj/run/20260909000000/checkpoint_artifacts/iter_000300"
PREFIX = "proj/run/20260909000000/checkpoint_artifacts/iter_000300"


def _fake_safetensors(path: str, names: list[str], nbytes: int = 4) -> None:
    """Write a minimal VALID safetensors file (8-byte header length + JSON header + raw data) holding ``names``."""
    header = {
        name: {"dtype": "BF16", "shape": [nbytes // 2], "data_offsets": [i * nbytes, (i + 1) * nbytes]}
        for i, name in enumerate(names)
    }
    header["__metadata__"] = {"format": "pt"}
    encoded = json.dumps(header).encode()
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)) + encoded + b"\0" * (nbytes * len(names)))


def _fake_shards(directory: str, weight_map: dict[str, str]) -> None:
    for shard in sorted(set(weight_map.values())):
        _fake_safetensors(os.path.join(directory, shard), sorted(k for k, v in weight_map.items() if v == shard))


def _fake_base(base: Path, sidecars: dict[str, str], config: str = BASE_CONFIG) -> None:
    """A base dir with config.json (the structural identity by default) + index + complete shards + the given sidecars
    (what base_identity / main require)."""
    base.mkdir()
    (base / "config.json").write_text(config)
    (base / hf_overlay.INDEX_FILE).write_text(json.dumps({"metadata": {}, "weight_map": WMAP}))
    _fake_shards(str(base), WMAP)
    for name, content in sidecars.items():
        (base / name).write_text(content)


def _truncate(path: str, by: int = 1) -> None:
    with open(path, "r+b") as handle:
        handle.truncate(os.path.getsize(path) - by)


def _ns(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = dict(
        ckpt_dir=None,
        artifact_uri=None,
        out="/x/out",
        base="/x/base",
        min_tensors=800,
        expect_tensors=0,
        stage_dir=None,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def _manifest_payload(uri: str, sha256: str, size: int = 10) -> bytes:
    manifest = build_artifact_manifest(
        uri, "consolidated", ("model",), {"model": {"kind": "file", "path": "model.pt", "size": size, "sha256": sha256}}
    )
    return json.dumps(manifest).encode()


@pytest.fixture
def base(tmp_path: Path) -> Path:
    directory = tmp_path / "base"
    _fake_base(directory, BASE_FILES)
    return directory


@pytest.fixture
def hf_dir(tmp_path: Path) -> Path:
    """A staging dir as transformers 5.x save_pretrained leaves it: config + index + complete shards, re-serialized
    tokenizer files (other bytes, no chat_template key), stray chat_template.jinja (the PROCESSOR template) and
    processor_config.json, and no preprocessor / video_preprocessor / merges / vocab."""
    out = tmp_path / "hf"
    out.mkdir()
    (out / "config.json").write_text("{}")
    (out / hf_overlay.INDEX_FILE).write_text(json.dumps({"metadata": {}, "weight_map": WMAP}))
    _fake_shards(str(out), WMAP)
    for name, content in {
        "tokenizer.json": '{"transformers": "5.13"}',
        "tokenizer_config.json": '{"backend": "tokenizers"}',
        "chat_template.jinja": "BASE-PROCESSOR-TEMPLATE",
        "processor_config.json": '{"image_processor": {}}',
        "generation_config.json": '{"top_p": 0.8, "transformers_version": "5.13.0"}',
    }.items():
        (out / name).write_text(content)
    return out


def test_validate_args_every_branch() -> None:
    assert mat.validate_args(_ns(ckpt_dir="/x/iter_000300")) is None
    assert mat.validate_args(_ns(artifact_uri=URI)) is None
    assert "--out" in str(mat.validate_args(_ns(ckpt_dir="/x", out="")))
    assert "--base" in str(mat.validate_args(_ns(ckpt_dir="/x", base="")))
    assert "exactly one" in str(mat.validate_args(_ns()))
    assert "exactly one" in str(mat.validate_args(_ns(ckpt_dir="/x", artifact_uri=URI)))
    assert "s3://" in str(mat.validate_args(_ns(artifact_uri="https://bucket/x/iter_000300")))
    assert "iter_" in str(mat.validate_args(_ns(artifact_uri="s3://bucket/x/model.pt")))
    assert "--min-tensors" in str(mat.validate_args(_ns(ckpt_dir="/x", min_tensors=0)))
    assert "--expect-tensors" in str(mat.validate_args(_ns(ckpt_dir="/x", expect_tensors=-1)))
    assert "base model dir" in str(mat.validate_args(_ns(ckpt_dir="/x", out="/x/base/")))
    with pytest.raises(SystemExit):
        mat.main(["--ckpt-dir", "/x", "--artifact-uri", URI, "--base", "/b", "--out", "/o"])


def test_source_digest_known_answer(tmp_path: Path) -> None:
    model_pt = tmp_path / "model.pt"
    model_pt.write_bytes(b"\0" * 10)
    assert mat.source_digest(str(model_pt)) == {"sha256": SHA_ZERO10, "bytes": 10}


def test_base_identity_is_content_defined_and_sensitive(tmp_path: Path, base: Path) -> None:
    base_id = mat.base_identity(str(base))
    assert base_id["path"] == os.path.realpath(base) and base_id["n_shards"] == 2 and base_id["n_tensors"] == 3
    copy = tmp_path / "base_copy"
    shutil.copytree(base, copy)
    assert mat.base_identity(str(copy))["digest"] == base_id["digest"], (
        "a byte-identical base elsewhere is the same base"
    )
    (copy / "generation_config.json").write_text('{"top_p": 0.9}')
    assert mat.base_identity(str(copy))["digest"] != base_id["digest"], "one sidecar byte changes the identity"
    shutil.copy2(base / "generation_config.json", copy / "generation_config.json")
    assert mat.base_identity(str(copy))["digest"] == base_id["digest"]
    (copy / "config.json").write_text('{"model_type": "other"}')
    assert mat.base_identity(str(copy))["digest"] != base_id["digest"]
    shutil.copy2(base / "config.json", copy / "config.json")
    with open(copy / SHARD1, "ab") as handle:
        handle.write(b"\0")
    assert mat.base_identity(str(copy))["digest"] != base_id["digest"], "the shard inventory is part of the identity"
    (copy / SHARD1).unlink()
    with pytest.raises(FileNotFoundError, match=re.escape(str(copy / SHARD1))):
        mat.base_identity(str(copy))
    empty = tmp_path / "base_empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match=re.escape(f"base model incomplete at {empty}")):
        mat.base_identity(str(empty))
    (base / "additional_chat_templates").mkdir()
    (base / "additional_chat_templates" / "t.jinja").write_text("T")
    assert mat.base_identity(str(base))["digest"] != base_id["digest"], "additional_chat_templates/ is identity"
    shutil.rmtree(base / "additional_chat_templates")
    assert mat.base_identity(str(base))["digest"] == base_id["digest"]
    (base / hf_overlay.INDEX_FILE).write_text("{not json")
    with pytest.raises(ValueError, match="unreadable"):
        mat.base_identity(str(base))


def test_identity_fields_and_digest(base: Path) -> None:
    base_id = mat.base_identity(str(base))
    source_a = {"sha256": SHA_ZERO10, "bytes": 10}
    source_b = {"sha256": hashlib.sha256(b"\1" * 10).hexdigest(), "bytes": 10}
    ident_a, ident_b = mat.make_identity(source_a, base_id), mat.make_identity(source_b, base_id)
    assert ident_a == {
        "conversion": mat.CONVERSION_VERSION,
        "source_sha256": SHA_ZERO10,
        "source_bytes": 10,
        "base_digest": base_id["digest"],
    }
    assert mat.CONVERSION_VERSION == "materialize_model_pt_to_hf/5", "the bumped contract refuses older markers"
    assert ident_b != ident_a and ident_b["source_bytes"] == ident_a["source_bytes"]
    assert mat.identity_digest(ident_a) == mat.identity_digest(dict(reversed(list(ident_a.items()))))
    assert mat.identity_digest(ident_a) != mat.identity_digest(ident_b)
    # --unified-chat-template is identity, present only when on (every marker written before the option keeps its identity)
    unified = mat.make_identity(source_a, base_id, unified_chat_template=True)
    assert unified == {**ident_a, "unified_chat_template": True} and "unified-chat-template" in mat._short(unified)
    assert "unified_chat_template" not in ident_a and "unified-chat-template" not in mat._short(ident_a)
    assert mat.identity_digest(unified) != mat.identity_digest(ident_a)


def test_marker_status_match_mismatch_damaged_absent_and_legacy(tmp_path: Path, base: Path, hf_dir: Path) -> None:
    out = str(hf_dir)
    marker_path = os.path.join(out, mat.MATERIALIZED_MARKER)
    shard2 = os.path.join(out, SHARD2)
    base_id = mat.base_identity(str(base))
    want = {"conversion": mat.CONVERSION_VERSION, "base_digest": base_id["digest"], "base_path": str(base)}
    probe_a = {"kind": "local", "path": f"{tmp_path}/a/model.pt", "sha256": SHA_ZERO10, "bytes": 10}
    probe_b = {
        "kind": "local",
        "path": f"{tmp_path}/b/model.pt",
        "sha256": hashlib.sha256(b"\1" * 10).hexdigest(),
        "bytes": 10,
    }
    ident_a = mat.make_identity(probe_a, base_id)
    source_a = {"kind": "local", "path": probe_a["path"], "sha256": SHA_ZERO10, "bytes": 10, "artifact_uri": None}

    assert mat.marker_status(f"{tmp_path}/absent", want, probe_a) == (
        "absent",
        f"{tmp_path}/absent/{mat.MATERIALIZED_MARKER} missing",
    )
    atomic_write_json(marker_path, {"selftest": True})  # a legacy marker without an identity record
    status, why = mat.marker_status(out, want, probe_a)
    assert status == "mismatch" and "no identity record" in why and marker_path in why, (status, why)

    inventory = mat.shard_inventory(out)
    assert inventory == [
        {"name": SHARD1, "bytes": os.path.getsize(os.path.join(out, SHARD1)), "n_tensors": 1},
        {"name": SHARD2, "bytes": os.path.getsize(shard2), "n_tensors": 2},
    ]
    good = {"identity": ident_a, "source": source_a, "base": base_id, "shards": inventory}
    atomic_write_json(marker_path, good)
    status, why = mat.marker_status(out, want, probe_a)
    assert status == "match" and "2 shards on disk as recorded" in why, (status, why)
    # same-size different source -> mismatch (a files-exist check would have skipped here)
    status, why = mat.marker_status(out, want, probe_b)
    assert status == "mismatch" and "source sha256" in why and probe_b["path"] in why, (status, why)
    # other base / other conversion (an output built under the previous contract) -> mismatch
    status, why = mat.marker_status(out, {**want, "base_digest": "0" * 64}, probe_a)
    assert status == "mismatch" and "base digest" in why, (status, why)
    atomic_write_json(marker_path, {**good, "identity": {**ident_a, "conversion": "materialize_model_pt_to_hf/2"}})
    status, why = mat.marker_status(out, want, probe_a)
    assert status == "mismatch" and "conversion 'materialize_model_pt_to_hf/2'" in why, (status, why)
    # an output built under /3 (config.json re-serialized by transformers, no per-tensor digests) is refused the same way
    atomic_write_json(marker_path, {**good, "identity": {**ident_a, "conversion": "materialize_model_pt_to_hf/3"}})
    status, why = mat.marker_status(out, want, probe_a)
    assert status == "mismatch" and "conversion 'materialize_model_pt_to_hf/3'" in why, (status, why)
    atomic_write_json(marker_path, good)
    # an archived source is proven by the same sha256 + bytes the manifest carries
    probe_s3 = {"kind": "artifact", "artifact_uri": URI, "sha256": SHA_ZERO10, "bytes": 10, "manifest_sha256": "0" * 64}
    assert mat.marker_status(out, want, probe_s3)[0] == "match"
    status, why = mat.marker_status(out, want, {**probe_s3, "bytes": 11})
    assert status == "mismatch" and URI in why, (status, why)
    # no inventory / unreadable marker -> mismatch; damaged inventory -> damaged (rebuilt), naming the shard
    atomic_write_json(marker_path, {**good, "shards": []})
    assert mat.marker_status(out, want, probe_a) == ("mismatch", f"{marker_path} has no output shard inventory")
    Path(marker_path).write_text("{not json")
    assert mat.marker_status(out, want, probe_a)[0] == "mismatch"
    atomic_write_json(marker_path, good)
    size2 = os.path.getsize(shard2)
    _truncate(shard2)
    assert mat.marker_status(out, want, probe_a) == (
        "damaged",
        f"{shard2} is {size2 - 1} bytes, the marker recorded {size2}",
    )


def test_inventory_mismatch_checks_size_tensor_count_and_header(hf_dir: Path) -> None:
    out = str(hf_dir)
    inventory = mat.shard_inventory(out)
    assert mat.inventory_mismatch(out, inventory) is None
    assert mat.inventory_mismatch(out, [{**inventory[1], "n_tensors": 1}]) == (
        f"{out}/{SHARD2} holds 2 tensors, the marker recorded 1"
    )
    assert mat.inventory_mismatch(out, [{**inventory[1], "bytes": 1}]) == (
        f"{out}/{SHARD2} is {inventory[1]['bytes']} bytes, the marker recorded 1"
    )
    (hf_dir / "junk.safetensors").write_bytes(b"\xff" * 20)
    bad = mat.inventory_mismatch(out, [{"name": "junk.safetensors", "bytes": 20, "n_tensors": 1}])
    assert bad is not None and bad.startswith(f"{out}/junk.safetensors: unreadable safetensors header"), bad
    assert mat.inventory_mismatch(out, [{"name": "gone.safetensors", "bytes": 1, "n_tensors": 1}]) == (
        f"{out}/gone.safetensors recorded in the marker is missing"
    )


def test_safetensors_header_and_shard_status(hf_dir: Path) -> None:
    shard2 = str(hf_dir / SHARD2)
    header, n = hf_overlay.safetensors_header(shard2)
    assert set(header) == SHARD2_KEYS and header["c.bias"]["data_offsets"] == [4, 8] and n > 0
    assert "__metadata__" not in header and os.path.getsize(shard2) == 8 + n + 8
    assert hf_overlay.shard_status(shard2, SHARD2_KEYS) is None
    assert "tensors differ from the index" in str(hf_overlay.shard_status(shard2, {"b.mlp.experts.gate_up_proj"}))
    assert hf_overlay.shard_status(f"{hf_dir}/nope.safetensors") == f"{hf_dir}/nope.safetensors does not exist"
    junk = str(hf_dir / "junk.safetensors")
    Path(junk).write_bytes(b"\xff" * 20)
    assert "unreadable safetensors header" in str(hf_overlay.shard_status(junk))
    Path(junk).write_bytes(struct.pack("<Q", 2**40))
    assert "implausible" in str(hf_overlay.shard_status(junk))
    Path(junk).write_bytes(b"\0" * 4)
    assert "shorter than the 8-byte header" in str(hf_overlay.shard_status(junk))
    _fake_safetensors(junk, [])
    assert hf_overlay.shard_status(junk) == f"{junk}: header lists no tensors"
    _truncate(shard2)
    status = str(hf_overlay.shard_status(shard2, SHARD2_KEYS))
    assert status.startswith(shard2) and "truncated or partially written" in status


def test_tmp_complete_staging_reuse_decisions(tmp_path: Path, base: Path, hf_dir: Path) -> None:
    out = str(hf_dir)
    stamp_path = os.path.join(out, mat.SOURCE_STAMP)
    shard2 = os.path.join(out, SHARD2)
    base_id = mat.base_identity(str(base))
    ident_a = mat.make_identity({"sha256": SHA_ZERO10, "bytes": 10}, base_id)
    ident_b = mat.make_identity({"sha256": hashlib.sha256(b"\1" * 10).hexdigest(), "bytes": 10}, base_id)

    ok, why = mat.tmp_complete(out, ident_a)
    assert not ok and why == f"{stamp_path} missing: the staging dir carries no source/base/conversion identity"
    Path(stamp_path).write_text(json.dumps({"model_pt": "/old/model.pt", "bytes": 10}))  # legacy size-only stamp
    ok, why = mat.tmp_complete(out, ident_a)
    assert not ok and "no identity record" in why, why
    atomic_write_json(stamp_path, {"identity": ident_b})
    ok, why = mat.tmp_complete(out, ident_a)
    assert not ok and "identity differs on ['source_sha256']" in why, why
    atomic_write_json(stamp_path, {"identity": {**ident_a, "base_digest": "0" * 64}})
    assert "['base_digest']" in mat.tmp_complete(out, ident_a)[1]
    atomic_write_json(stamp_path, {"identity": {**ident_a, "conversion": "materialize_model_pt_to_hf/2"}})
    assert "['conversion']" in mat.tmp_complete(out, ident_a)[1]
    Path(stamp_path).write_text("{not json")
    assert "unreadable" in mat.tmp_complete(out, ident_a)[1]
    atomic_write_json(stamp_path, {"identity": ident_a, "base": base_id})
    ok, why = mat.tmp_complete(out, ident_a)
    assert ok and "2 complete shards (3 tensors), identity matches" in why, why
    assert mat.tmp_complete(out)[0], "the structure-only check (no identity) still passes"
    _truncate(shard2)
    ok, why = mat.tmp_complete(out, ident_a)
    assert not ok and why.startswith(shard2) and "truncated" in why, why
    _fake_safetensors(shard2, ["wrong.name", "c.bias"])
    ok, why = mat.tmp_complete(out, ident_a)
    assert not ok and "tensors differ from the index" in why and "missing=['b.mlp.experts.gate_up_proj']" in why, why
    assert mat.tmp_complete(f"{tmp_path}/absent", ident_a) == (False, f"{tmp_path}/absent does not exist")
    (tmp_path / "empty").mkdir()
    assert mat.tmp_complete(f"{tmp_path}/empty", ident_a) == (
        False,
        f"{tmp_path}/empty/{hf_overlay.INDEX_FILE} missing",
    )
    (tmp_path / "empty" / hf_overlay.INDEX_FILE).write_text("{}")
    assert mat.tmp_complete(f"{tmp_path}/empty", ident_a) == (False, f"{tmp_path}/empty/config.json missing")
    (tmp_path / "empty" / "config.json").write_text("{}")
    assert "unreadable" in mat.tmp_complete(f"{tmp_path}/empty", ident_a)[1]


def test_sync_and_verify_sidecars_including_additional_chat_templates(base: Path, hf_dir: Path) -> None:
    out, base_dir = str(hf_dir), str(base)

    def refused(contains: str) -> None:
        with pytest.raises(ConversionError, match=re.escape(contains)):
            hf_overlay.verify_sidecars(out, base_dir)

    # before the sync: the re-serialized tokenizer.json differs (first in SIDECAR_FILES order)
    refused(f"{out}/tokenizer.json differs from base {base_dir}/tokenizer.json")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (hf_dir / name).unlink()
    refused(f"{out}/tokenizer.json missing (base has {base_dir}/tokenizer.json)")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy2(base / name, hf_dir / name)
    # sync: every base sidecar copied (overwriting), the two strays removed, bytes identical, then verify passes
    sync = hf_overlay.sync_sidecars(base_dir, out)
    assert sync["copied"] == [n for n in hf_overlay.SIDECAR_FILES if n in BASE_FILES]
    assert sync["removed"] == ["chat_template.jinja", "processor_config.json"]
    for name, content in BASE_FILES.items():
        assert (hf_dir / name).read_text() == content, name
    assert not (hf_dir / "chat_template.jinja").exists() and not (hf_dir / "processor_config.json").exists()
    assert not list(hf_dir.glob("*.copying")), "the copy goes through a temporary and leaves none behind"
    assert hf_overlay.verify_sidecars(out, base_dir) == {"sidecars_identical": 8}
    assert hf_overlay.sync_sidecars(base_dir, out) == {"copied": sync["copied"], "removed": []}, "idempotent"
    # a stray that reappears, or a modified copy, is caught and repaired by the next sync
    (hf_dir / "chat_template.jinja").write_text("STRAY")
    refused(f"{out}/chat_template.jinja exists but the base has no")
    assert hf_overlay.sync_sidecars(base_dir, out)["removed"] == ["chat_template.jinja"]
    (hf_dir / "tokenizer_config.json").write_text("{}")
    refused(f"{out}/tokenizer_config.json differs from base")
    hf_overlay.sync_sidecars(base_dir, out)
    assert hf_overlay.verify_sidecars(out, base_dir) == {"sidecars_identical": 8}
    # additional_chat_templates/ is synced and verified as a directory
    (base / "additional_chat_templates").mkdir()
    (base / "additional_chat_templates" / "t.jinja").write_text("T")
    refused(f"{out}/additional_chat_templates missing")
    assert hf_overlay.sync_sidecars(base_dir, out)["copied"][-1] == "additional_chat_templates/"
    assert hf_overlay.verify_sidecars(out, base_dir) == {"sidecars_identical": 9}
    (hf_dir / "additional_chat_templates" / "t.jinja").write_text("changed")
    refused(f"{out}/additional_chat_templates differs from base {base_dir}/additional_chat_templates on ['t.jinja']")
    (hf_dir / "additional_chat_templates" / "extra.jinja").write_text("x")
    refused(f"{out}/additional_chat_templates lists different files than base")
    hf_overlay.sync_sidecars(base_dir, out)
    assert hf_overlay.verify_sidecars(out, base_dir) == {"sidecars_identical": 9}
    shutil.rmtree(base / "additional_chat_templates")
    refused(f"{out}/additional_chat_templates exists but the base has no")
    assert hf_overlay.sync_sidecars(base_dir, out)["removed"] == ["additional_chat_templates/"]
    assert hf_overlay.verify_sidecars(out, base_dir) == {"sidecars_identical": 8}
    # config.json, shards, index are not touched by the sync
    assert (hf_dir / "config.json").read_text() == "{}" and mat.tmp_complete(out)[0]


def test_verify_index_pass_and_every_failure_path(base: Path, hf_dir: Path) -> None:
    out = str(hf_dir)
    hf_overlay.sync_sidecars(str(base), out)  # REQUIRED_FILES present
    shard1, shard2 = os.path.join(out, SHARD1), os.path.join(out, SHARD2)
    index_path = os.path.join(out, hf_overlay.INDEX_FILE)

    def refused(contains: str, **kw: object) -> None:
        with pytest.raises(ConversionError, match=re.escape(contains)):
            hf_overlay.verify_index(out, **kw)  # type: ignore[arg-type]

    report = hf_overlay.verify_index(out, expect_keys=set(WMAP), expect_tensors=3, min_tensors=3)
    assert report == {"n_tensors": 3, "n_shards": 2, "bytes": os.path.getsize(shard1) + os.path.getsize(shard2)}
    refused(
        f"{index_path} weight_map != model.pt keys; missing=['zzz'] extra=[]",
        expect_keys=set(WMAP) | {"zzz"},
        min_tensors=3,
    )
    refused("extra=['b.mlp.experts.gate_up_proj', 'c.bias']", expect_keys={"a.weight"}, min_tensors=3)
    refused("weight_map has 3 tensors, expected exactly 4", expect_tensors=4, min_tensors=3)
    refused("lists 3 tensors, fewer than the 800 floor")
    os.remove(shard2)
    refused(f"shard listed in {index_path}: {shard2} does not exist", min_tensors=3)
    _fake_safetensors(shard2, ["b.mlp.experts.gate_up_proj"])
    refused(f"{shard2}: tensors differ from the index; missing=['c.bias']", min_tensors=3)
    _fake_safetensors(shard2, sorted(SHARD2_KEYS))
    (hf_dir / "tokenizer.json").unlink()
    refused(f"{out}/tokenizer.json missing", min_tensors=3)
    shutil.copy2(base / "tokenizer.json", hf_dir / "tokenizer.json")
    assert hf_overlay.verify_index(out, min_tensors=3)["n_tensors"] == 3
    (hf_dir / hf_overlay.INDEX_FILE).write_text(json.dumps({"metadata": {}, "weight_map": {}}))
    refused(f"weight_map missing/empty in {index_path}", min_tensors=3)


def test_parse_artifact_uri_and_archived_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    assert mat.parse_artifact_uri(URI) == ("bucket", PREFIX)
    for bad in ("https://bucket/x/iter_000300", "s3://bucket", "s3:///x/iter_000300", "s3://bucket/x/model.pt", ""):
        with pytest.raises(ValueError):
            mat.parse_artifact_uri(bad)

    client = FakeS3Client()
    payload = _manifest_payload(URI, SHA_ZERO10)
    client.plant(f"{PREFIX}/{CHECKPOINT_ARTIFACT_MANIFEST}", payload)
    seen: list[str] = []

    def fake_client(credentials: str) -> FakeS3Client:
        seen.append(credentials)
        return client

    monkeypatch.setattr(object_store, "build_s3_client", fake_client)
    probe = mat.probe_archived_source(URI, credentials="creds.json")
    assert probe == {
        "kind": "artifact",
        "artifact_uri": URI,
        "bucket": "bucket",
        "prefix": PREFIX,
        "sha256": SHA_ZERO10,
        "bytes": 10,
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
    }
    assert seen == ["creds.json"] and not any(key.endswith("model.pt") for key in client.heads), "no payload is touched"
    mat.probe_archived_source(URI)
    assert seen[-1] == object_store.DEFAULT_CREDENTIALS, "the framework default applies without --credentials"
    # a manifest that does not describe a hashed consolidated model.pt cannot prove a source
    other_uri = URI.replace("iter_000300", "iter_000400")
    other_prefix = PREFIX.replace("iter_000300", "iter_000400")
    unhashed = build_artifact_manifest(
        other_uri, "consolidated", ("model",), {"model": {"kind": "file", "path": "model.pt", "size": 10}}
    )
    client.plant(f"{other_prefix}/{CHECKPOINT_ARTIFACT_MANIFEST}", json.dumps(unhashed).encode())
    with pytest.raises(ValueError, match="no hashed consolidated 'model' component"):
        mat.probe_archived_source(other_uri)
    dcp = build_artifact_manifest(other_uri, "dcp", ("model",), {"model": {"kind": "dcp", "path": "model"}})
    client.plant(f"{other_prefix}/{CHECKPOINT_ARTIFACT_MANIFEST}", json.dumps(dcp).encode())
    with pytest.raises(ValueError, match="no hashed consolidated 'model' component"):
        mat.probe_archived_source(other_uri)
    client.plant(
        f"{other_prefix}/{CHECKPOINT_ARTIFACT_MANIFEST}", payload
    )  # names iter_000300, lives under iter_000400
    with pytest.raises(RuntimeError, match="iteration mismatch"):
        mat.probe_archived_source(other_uri)
    client.plant(f"{other_prefix}/{CHECKPOINT_ARTIFACT_MANIFEST}", b"{not json")
    with pytest.raises(RuntimeError):
        mat.probe_archived_source(other_uri)

    class _Unreachable:
        def get_object(self, **_: object) -> dict:
            raise RuntimeError("NoSuchKey")

    monkeypatch.setattr(object_store, "build_s3_client", lambda credentials: _Unreachable())
    with pytest.raises(FileNotFoundError, match="missing or unreadable"):
        mat.probe_archived_source(URI)


def _install_fake_overlay(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace the real overlay with one that writes config + index + shards + transformers-5 strays."""
    calls: list[str] = []

    def fake_overlay(ckpt_dir: str, model_id: str, staging_dir: str, payload: object = None) -> str:
        calls.append(ckpt_dir)
        os.makedirs(staging_dir)
        shutil.copy2(os.path.join(model_id, "config.json"), os.path.join(staging_dir, "config.json"))
        shutil.copy2(os.path.join(model_id, hf_overlay.INDEX_FILE), os.path.join(staging_dir, hf_overlay.INDEX_FILE))
        _fake_shards(staging_dir, WMAP)
        Path(staging_dir, "tokenizer.json").write_text('{"transformers": "5.13"}')
        Path(staging_dir, "chat_template.jinja").write_text("STRAY")
        return staging_dir

    monkeypatch.setattr(hf_overlay, "overlay_consolidated_onto_hf", fake_overlay)
    return calls


def _marker(out: str) -> dict:
    return json.loads(Path(out, mat.MATERIALIZED_MARKER).read_text())


def test_explicit_export_dtype_is_identity_bound_and_verified_before_publication(tmp_path, base, monkeypatch):
    _install_fake_overlay(monkeypatch)  # Produces BF16 shards deliberately.
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.pt").write_bytes(b"\0" * 10)
    output = str(tmp_path / "default")
    request = dict(base=str(base), ckpt_dir=str(checkpoint), min_tensors=3, verify=False)
    assert mat.run_request(out=output, **request) == "built"
    before = Path(output, mat.MATERIALIZED_MARKER).read_bytes()
    with pytest.raises(mat.MaterializeRefusedError, match="output_dtype"):
        mat.run_request(out=output, output_dtype="float32", **request)
    assert Path(output, mat.MATERIALIZED_MARKER).read_bytes() == before
    invalid = str(tmp_path / "wrong-precision")
    with pytest.raises(ConversionError, match="dtype BF16; expected float32"):
        mat.run_request(out=invalid, output_dtype="float32", **request)
    assert not Path(invalid).exists() and not Path(invalid + ".tmp").exists()
    with pytest.raises(ValueError, match="supports float32 only"):
        mat.run_request(out=invalid, output_dtype="bfloat16", **request)


def test_export_metadata_survives_transaction_and_binds_reuse(tmp_path, base, monkeypatch):
    calls = _install_fake_overlay(monkeypatch)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.pt").write_bytes(b"\0" * 10)
    output = str(tmp_path / "export")
    request = dict(base=str(base), ckpt_dir=str(checkpoint), min_tensors=3, verify=False)
    metadata = {"format": "test-export-v1", "media": {"num_frames": 4}}
    assert mat.run_request(out=output, export_metadata=metadata, **request) == "built"
    marker = Path(output, mat.MATERIALIZED_MARKER)
    before = marker.read_bytes()
    assert json.loads(before)["identity"]["export_metadata"] == metadata
    assert mat.run_request(out=output, export_metadata=metadata, **request) == "skipped"
    assert len(calls) == 1
    for changed in (None, {"format": "test-export-v1", "media": {"num_frames": 8}}):
        with pytest.raises(mat.MaterializeRefusedError, match="export_metadata"):
            mat.run_request(out=output, export_metadata=changed, **request)
        assert marker.read_bytes() == before
    # A legacy marker cannot silently satisfy a metadata-bound request.
    legacy = json.loads(before)
    del legacy["identity"]["export_metadata"]
    marker.write_text(json.dumps(legacy))
    with pytest.raises(mat.MaterializeRefusedError, match="missing export_metadata"):
        mat.run_request(out=output, export_metadata=metadata, **request)


@pytest.mark.parametrize("metadata", [[], {}, {"nonfinite": float("nan")}, {"not_json": object()}])
def test_invalid_export_metadata_fails_before_publication(tmp_path, metadata):
    output = tmp_path / "export"
    with pytest.raises(ValueError, match="export_metadata"):
        mat.run_request(base="unused", ckpt_dir="unused", out=str(output), export_metadata=metadata)
    assert not output.exists()


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def test_main_end_to_end_with_fake_overlay(
    tmp_path: Path, base: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _install_fake_overlay(monkeypatch)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "model.pt").write_bytes(b"\0" * 10)
    sha_b = hashlib.sha256(b"\1" * 10).hexdigest()
    out = str(tmp_path / "out")
    args = [
        "--ckpt-dir",
        str(ckpt),
        "--base",
        str(base),
        "--out",
        out,
        "--no-verify",
        "--min-tensors",
        "3",
        "--expect-tensors",
        "3",
    ]

    def run(extra: list[str] | None = None) -> tuple[int, str, str]:
        code = mat.main(args + (extra or []))
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    # build
    code, stdout, _ = run()
    assert code == 0 and calls == [str(ckpt)] and "MATERIALIZE DONE" in stdout and "idempotent" not in stdout
    marker = _marker(out)
    assert marker["protocol"] == mat.MARKER_PROTOCOL
    assert marker["identity"] == {
        "conversion": mat.CONVERSION_VERSION,
        "source_sha256": SHA_ZERO10,
        "source_bytes": 10,
        "base_digest": mat.base_identity(str(base))["digest"],
    }
    assert marker["source"]["kind"] == "local" and marker["source"]["path"] == str(ckpt / "model.pt")
    assert marker["source"]["artifact_uri"] is None and marker["reused_staging"] is False
    assert [s["name"] for s in marker["shards"]] == [SHARD1, SHARD2] and marker["shards"][1]["n_tensors"] == 2
    assert marker["options"] == {
        "expect_tensors": 3,
        "min_tensors": 3,
        "no_verify": True,
        "unified_chat_template": False,
    }
    assert marker["chat_template"] == {
        "unified": False,
        "processor_template_sha256": _sha(FAKE_PROCESSOR_TEMPLATE),
        "base_tokenizer_template_sha256": _sha("BASE-TOKENIZER-TEMPLATE"),
        "tokenizer_template_sha256": _sha("BASE-TOKENIZER-TEMPLATE"),
    }
    assert "chat_template" not in marker["sidecars"]
    assert Path(out, "tokenizer_config.json").read_bytes() == (base / "tokenizer_config.json").read_bytes(), (
        "without the option the tokenizer config is the base's bytes"
    )
    assert marker["verify"]["torch"] is False and marker["verify"]["n_tensors"] == 3
    assert marker["sidecars"]["removed"] == ["chat_template.jinja"], "the transformers-5 stray was removed"
    assert marker["sidecars"]["config"] == {
        "config_source": "base",
        "transformers_config_kept_as": None,
        "kept_now": False,
    }
    assert Path(out, "config.json").read_bytes() == (base / "config.json").read_bytes(), "config.json is the base's"
    assert os.path.isfile(os.path.join(out, mat.SOURCE_STAMP)) and not os.path.exists(out + ".tmp")
    assert hf_overlay.verify_sidecars(out, str(base)) == {"sidecars_identical": len(BASE_FILES)}
    # identical request -> identity proven -> skip without calling the overlay
    code, stdout, _ = run()
    assert code == 0 and "idempotent skip" in stdout and calls == [str(ckpt)]
    # same-size DIFFERENT checkpoint at the same --out: --no-verify must not skip the identity check; nothing is
    # rebuilt or deleted without --force
    (ckpt / "model.pt").write_bytes(b"\1" * 10)
    code, _, stderr = run()
    assert code == 1 and "different or unproven materialization" in stderr and "source sha256" in stderr
    assert calls == [str(ckpt)] and _marker(out)["identity"]["source_sha256"] == SHA_ZERO10
    code, stdout, _ = run(["--force"])
    assert code == 0 and len(calls) == 2 and "MATERIALIZE DONE" in stdout
    assert _marker(out)["identity"]["source_sha256"] == sha_b
    assert run()[0] == 0 and len(calls) == 2
    # other base (one sidecar byte differs) at the same --out -> refused
    base2 = tmp_path / "base2"
    _fake_base(base2, {**BASE_FILES, "generation_config.json": '{"top_p": 0.9}'})
    code = mat.main([*args[:2], "--base", str(base2), *args[4:]])
    assert code == 1 and "base digest" in capsys.readouterr().err and len(calls) == 2
    # damaged output (recorded shard truncated) with a valid marker -> rebuilt automatically
    shard2 = os.path.join(out, SHARD2)
    _truncate(shard2)
    code, stdout, _ = run()
    assert code == 0 and len(calls) == 3 and "MATERIALIZE DONE" in stdout
    assert hf_overlay.shard_status(shard2, SHARD2_KEYS) is None
    # a stale <out>.tmp (stamped for another source) is NOT reused; a correctly stamped one is
    out3 = str(tmp_path / "out3")
    hf_overlay.overlay_consolidated_onto_hf(str(ckpt), str(base), out3 + ".tmp")
    atomic_write_json(
        f"{out3}.tmp/{mat.SOURCE_STAMP}", {"identity": {**_marker(out)["identity"], "source_sha256": SHA_ZERO10}}
    )
    assert mat.main([*args[:4], "--out", out3, *args[6:]]) == 0
    assert len(calls) == 5 and _marker(out3)["reused_staging"] is False
    assert _marker(out3)["identity"]["source_sha256"] == sha_b and not os.path.exists(out3 + ".tmp")
    out4 = str(tmp_path / "out4")
    hf_overlay.overlay_consolidated_onto_hf(str(ckpt), str(base), out4 + ".tmp")
    atomic_write_json(f"{out4}.tmp/{mat.SOURCE_STAMP}", {"identity": _marker(out)["identity"]})
    assert mat.main([*args[:4], "--out", out4, *args[6:]]) == 0
    assert len(calls) == 6 and _marker(out4)["reused_staging"] is True, "the overlay was skipped for a matching stamp"
    assert hf_overlay.verify_sidecars(out4, str(base))["sidecars_identical"] == len(BASE_FILES)
    # the identity guard behind config.json <- base at CLI level: a base config without the structural identity is refused
    (base / "config.json").write_text('{"model_type": "fake"}')
    out5 = str(tmp_path / "out5")
    assert mat.main([*args[:4], "--out", out5, *args[6:]]) == 1
    assert "model identity mismatch" in capsys.readouterr().err and not os.path.exists(out5)
    (base / "config.json").write_text(BASE_CONFIG)
    # a missing source or an incomplete base fail closed with exit 1
    assert mat.main([*args[:1], str(tmp_path / "nowhere"), *args[2:]]) == 1
    assert "does not exist" in capsys.readouterr().err
    (base / "preprocessor_config.json").unlink()
    assert run()[0] == 1


def test_archived_source_stage_dir_ownership(
    tmp_path: Path, base: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _install_fake_overlay(monkeypatch)
    payload = b"\0" * 10
    client = FakeS3Client()
    client.plant(f"{PREFIX}/{CHECKPOINT_ARTIFACT_MANIFEST}", _manifest_payload(URI, SHA_ZERO10))
    monkeypatch.setattr(object_store, "build_s3_client", lambda credentials: client)
    fetches: list[tuple[str, str, str | None]] = []
    served = {"bytes": payload}

    def fake_fetch(artifact_uri: str, dst_dir: str, *, credentials: str | None) -> str:
        fetches.append((artifact_uri, dst_dir, credentials))
        artifact_dir = os.path.join(dst_dir, "iter_000300")
        os.makedirs(artifact_dir, exist_ok=True)
        Path(artifact_dir, "model.pt").write_bytes(served["bytes"])
        Path(artifact_dir, CHECKPOINT_ARTIFACT_MANIFEST).write_bytes(_manifest_payload(URI, SHA_ZERO10))
        Path(artifact_dir, ".staged.json").write_text("{}")
        return artifact_dir

    monkeypatch.setattr(mat, "_fetch_archived_artifact", fake_fetch)
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "sentinel.txt").write_text("caller-owned")
    (stage / "caller_subdir").mkdir()
    (stage / "caller_subdir" / "keep.bin").write_bytes(b"k")
    out = str(tmp_path / "teacher")
    args = ["--artifact-uri", URI, "--base", str(base), "--out", out, "--no-verify", "--min-tensors", "3"]

    assert mat.main([*args, "--stage-dir", str(stage), "--credentials", "creds.json"]) == 0
    assert fetches == [(URI, str(stage), "creds.json")] and calls == [f"{stage}/iter_000300"]
    marker = _marker(out)
    assert marker["source"]["kind"] == "artifact" and marker["source"]["artifact_uri"] == URI
    assert marker["identity"]["source_sha256"] == SHA_ZERO10 and marker["identity"]["source_bytes"] == 10
    assert marker["source"]["manifest_sha256"] == hashlib.sha256(_manifest_payload(URI, SHA_ZERO10)).hexdigest()
    assert sorted(os.listdir(stage)) == ["caller_subdir", "sentinel.txt"], "only what the fetch created was removed"
    assert (stage / "sentinel.txt").read_text() == "caller-owned" and (stage / "caller_subdir" / "keep.bin").is_file()
    # the skip decision reads the manifest only: no fetch, no overlay
    assert mat.main([*args, "--stage-dir", str(stage)]) == 0
    assert "idempotent skip" in capsys.readouterr().out and len(fetches) == 1 and len(calls) == 1
    # --keep-staged leaves the fetched copy. Files that already exist at the fetch paths were not created by the next
    # invocation, so it removes none of them: ownership is per invocation, never inferred from the layout.
    (stage / "iter_000300").mkdir()
    (stage / "iter_000300" / "foreign.txt").write_text("x")
    assert mat.main([*args, "--stage-dir", str(stage), "--force", "--keep-staged"]) == 0
    kept = sorted(os.listdir(stage / "iter_000300"))
    assert len(fetches) == 2 and kept == [CHECKPOINT_ARTIFACT_MANIFEST, ".staged.json", "foreign.txt", "model.pt"]
    assert mat.main([*args, "--stage-dir", str(stage), "--force"]) == 0
    assert len(fetches) == 3 and sorted(os.listdir(stage / "iter_000300")) == kept, (
        "pre-existing files are never removed"
    )
    # a directory that already existed is never removed either, only the files this invocation put into it
    for name in kept:
        if name != "foreign.txt":
            (stage / "iter_000300" / name).unlink()
    assert mat.main([*args, "--stage-dir", str(stage), "--force"]) == 0
    assert len(fetches) == 4 and sorted(os.listdir(stage / "iter_000300")) == ["foreign.txt"], "pre-existing dir stays"
    # the default <out>.stage is created by this invocation and therefore removed entirely after success
    out2 = str(tmp_path / "teacher2")
    assert mat.main([*args[:4], "--out", out2, *args[6:]]) == 0
    assert fetches[-1][1] == out2 + ".stage" and not os.path.exists(out2 + ".stage") and _marker(out2)
    # a fetched payload whose digest is not the manifest's is refused and the files it created are removed
    served["bytes"] = b"\1" * 10
    out3 = str(tmp_path / "teacher3")
    assert mat.main([*args[:4], "--out", out3, *args[6:], "--stage-dir", str(stage)]) == 1
    assert "unproven payload" in capsys.readouterr().err
    assert not os.path.exists(out3) and not os.path.exists(out3 + ".tmp")
    assert sorted(os.listdir(stage)) == ["caller_subdir", "iter_000300", "sentinel.txt"]
    assert os.listdir(stage / "iter_000300") == ["foreign.txt"], "the bad payload is gone, the caller's file stays"


def test_main_reports_a_fetch_refusal_as_fatal_and_leaves_no_output(
    tmp_path: Path, base: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The archive fetch refuses with ``RuntimeError`` (foreign staged copy, unprovable payload): exit 1, no output dir."""
    calls = _install_fake_overlay(monkeypatch)
    client = FakeS3Client()
    client.plant(f"{PREFIX}/{CHECKPOINT_ARTIFACT_MANIFEST}", _manifest_payload(URI, SHA_ZERO10))
    monkeypatch.setattr(object_store, "build_s3_client", lambda credentials: client)

    def refuse(artifact_uri: str, dst_dir: str, *, credentials: str | None) -> str:
        raise RuntimeError(
            f"refusing to reuse or replace {dst_dir}/iter_000300: it holds files this tool did not stage"
        )

    monkeypatch.setattr(mat, "_fetch_archived_artifact", refuse)
    out = str(tmp_path / "teacher")

    code = mat.main(["--artifact-uri", URI, "--base", str(base), "--out", out, "--no-verify", "--min-tensors", "3"])

    stderr = capsys.readouterr().err
    assert code == 1 and "FATAL" in stderr and "did not stage" in stderr
    assert calls == [] and not os.path.exists(out) and not os.path.exists(out + ".tmp")


def test_unified_chat_template_rewrites_the_tokenizer_template_only_and_is_verified_by_rule(
    base: Path, hf_dir: Path
) -> None:
    """``--unified-chat-template`` on a fake base with the two template texts: without the option the synced
    ``tokenizer_config.json`` is the base's bytes; with it
    the file's ``chat_template`` equals ``chat_template.json``'s text, every other key is unchanged and every other line
    byte-equal (the base's file is a ``json.dumps(indent=2)`` rendering like the base of record's), the record carries
    both shas, and ``verify_sidecars`` checks the RULE under the option (one key may differ, its value is the processor
    template) and byte identity without it; another changed key, an added key, a base without ``chat_template.json`` and
    an un-unified file under the option are refused."""
    out, base_dir = str(hf_dir), str(base)
    rich = {
        "added_tokens_decoder": {"0": {"content": "<|endoftext|>", "special": True}},
        "chat_template": "BASE-TOKENIZER-TEMPLATE",
        "model_max_length": 131072,
        "unk_token": None,
    }
    (base / "tokenizer_config.json").write_text(json.dumps(rich, indent=2))
    hf_overlay.sync_sidecars(base_dir, out)
    assert (hf_dir / "tokenizer_config.json").read_bytes() == (base / "tokenizer_config.json").read_bytes()
    assert hf_overlay.verify_sidecars(out, base_dir) == {"sidecars_identical": 8}
    with pytest.raises(ConversionError, match="is not the base's processor template"):
        hf_overlay.verify_sidecars(out, base_dir, unified_chat_template=True)  # claimed, but the file is the base's
    record = hf_overlay.write_unified_chat_template(out, base_dir)
    produced = json.loads((hf_dir / "tokenizer_config.json").read_text())
    processor_text = json.loads((base / "chat_template.json").read_text())["chat_template"]
    assert processor_text == FAKE_PROCESSOR_TEMPLATE and hf_overlay.PROCESSOR_REASONING_LINE in processor_text
    assert (
        produced["chat_template"]
        == FAKE_CANONICAL_TEMPLATE
        == hf_overlay.canonicalize_processor_template(processor_text)
    )
    assert hf_overlay.CANONICAL_REASONING_LINE in produced["chat_template"], "the one reasoning-branch change (rule R1)"
    assert (
        hf_overlay.PROCESSOR_REASONING_LINE not in produced["chat_template"]
        and produced["chat_template"] != processor_text
    )
    # The export's chat_template.json carries the same canonical text, its formatting reproduced
    assert (base / "chat_template.json").read_text() == BASE_FILES["chat_template.json"], "the base is never written"
    assert json.loads((hf_dir / "chat_template.json").read_text()) == {"chat_template": FAKE_CANONICAL_TEMPLATE}
    assert (hf_dir / "chat_template.json").read_bytes() == json.dumps(
        {"chat_template": FAKE_CANONICAL_TEMPLATE}
    ).encode()
    assert (hf_dir / "chat_template.json").read_bytes() != (base / "chat_template.json").read_bytes()
    assert {k: v for k, v in produced.items() if k != "chat_template"} == {
        k: v for k, v in rich.items() if k != "chat_template"
    }
    base_lines = (base / "tokenizer_config.json").read_text().splitlines()
    out_lines = (hf_dir / "tokenizer_config.json").read_text().splitlines()
    assert len(base_lines) == len(out_lines) == 11  # 1 + 6 (the nested added_tokens_decoder) + 3 scalar keys + 1
    differing = [i for i, (a, b) in enumerate(zip(base_lines, out_lines, strict=True)) if a != b]
    assert differing == [base_lines.index('  "chat_template": "BASE-TOKENIZER-TEMPLATE",')]
    assert out_lines[differing[0]] == '  "chat_template": ' + json.dumps(FAKE_CANONICAL_TEMPLATE) + ","
    diff = hf_overlay.canonical_template_diff(FAKE_PROCESSOR_TEMPLATE)
    assert [line for line in diff if line.startswith("-") and not line.startswith("---")] == [
        "-" + hf_overlay.PROCESSOR_REASONING_LINE
    ]
    assert [line for line in diff if line.startswith("+") and not line.startswith("+++")] == [
        "+" + hf_overlay.CANONICAL_REASONING_LINE
    ]
    assert record == {
        "unified": True,
        "template_mode": "processor-canonical",
        "files": ["tokenizer_config.json", "chat_template.json"],
        "processor_template_sha256": _sha(FAKE_PROCESSOR_TEMPLATE),
        "base_tokenizer_template_sha256": _sha("BASE-TOKENIZER-TEMPLATE"),
        "tokenizer_template_sha256": _sha(FAKE_CANONICAL_TEMPLATE),
        "shipped": {
            "tokenizer_config.json": _sha(FAKE_CANONICAL_TEMPLATE),
            "chat_template.json": _sha(FAKE_CANONICAL_TEMPLATE),
        },
        "canonical_diff": diff,
        "processor_template_chars": len(FAKE_PROCESSOR_TEMPLATE),
        "shipped_template_chars": len(FAKE_CANONICAL_TEMPLATE),
        "base_tokenizer_template_chars": 23,
        "templates_already_equal": False,
        "formatting_reproduced": {"tokenizer_config.json": True, "chat_template.json": True},
    }
    assert hf_overlay.verify_sidecars(out, base_dir, unified_chat_template=True) == {
        "sidecars_identical": 6,
        "sidecars_unified_chat_template": ["tokenizer_config.json", "chat_template.json"],
    }
    assert hf_overlay.verify_unified_template_files(out, base_dir) == ["tokenizer_config.json", "chat_template.json"]
    with pytest.raises(ConversionError, match="tokenizer_config.json differs from base"):
        hf_overlay.verify_sidecars(out, base_dir)  # without the option the unified file is a difference
    assert hf_overlay.chat_template_record(out, base_dir, unified_chat_template=True) == {
        "unified": True,
        "processor_template_sha256": _sha(FAKE_PROCESSOR_TEMPLATE),
        "base_tokenizer_template_sha256": _sha("BASE-TOKENIZER-TEMPLATE"),
        "tokenizer_template_sha256": _sha(FAKE_CANONICAL_TEMPLATE),
        "template_mode": "processor-canonical",
        "files": ["tokenizer_config.json", "chat_template.json"],
        "shipped": {
            "tokenizer_config.json": _sha(FAKE_CANONICAL_TEMPLATE),
            "chat_template.json": _sha(FAKE_CANONICAL_TEMPLATE),
        },
        "canonical_diff": diff,
        "serve_assumption": hf_overlay.UNIFIED_SERVE_ASSUMPTION,
        "preferred_sidecars": [],
    }
    assert "enable_thinking" in hf_overlay.UNIFIED_SERVE_ASSUMPTION and "R151-05" in hf_overlay.UNIFIED_SERVE_ASSUMPTION
    assert hf_overlay.write_unified_chat_template(out, base_dir) == record, "idempotent"
    assert not list(hf_dir.glob("*.writing"))
    # the rule on the processor file: the base's verbatim text, or another key, in the export's chat_template.json is refused
    (hf_dir / "chat_template.json").write_bytes((base / "chat_template.json").read_bytes())
    with pytest.raises(ConversionError, match="chat_template.json chat_template is not the base's processor template"):
        hf_overlay.verify_sidecars(out, base_dir, unified_chat_template=True)
    (hf_dir / "chat_template.json").write_text(json.dumps({"chat_template": FAKE_CANONICAL_TEMPLATE, "extra": 1}))
    with pytest.raises(ConversionError, match=re.escape("chat_template.json keys differ from base")):
        hf_overlay.verify_unified_template_files(out, base_dir)
    assert hf_overlay.write_unified_chat_template(out, base_dir) == record, "the next write repairs it"
    hf_overlay.verify_unified_template_files(out, base_dir)
    # the rule: another key changed, or a key added, is refused under the option
    produced["model_max_length"] = 1
    (hf_dir / "tokenizer_config.json").write_text(json.dumps(produced, indent=2))
    with pytest.raises(ConversionError, match=re.escape("on ['model_max_length']; --unified-chat-template may change")):
        hf_overlay.verify_sidecars(out, base_dir, unified_chat_template=True)
    produced["model_max_length"] = 131072
    produced["extra"] = 1
    (hf_dir / "tokenizer_config.json").write_text(json.dumps(produced, indent=2))
    with pytest.raises(ConversionError, match=re.escape("keys differ from base")) as refused:
        hf_overlay.verify_sidecars(out, base_dir, unified_chat_template=True)
    assert "extra=['extra']" in str(refused.value)
    # a base without chat_template.json (a text-only base) cannot be unified: refused, nothing written
    (base / "chat_template.json").unlink()
    before = (hf_dir / "tokenizer_config.json").read_bytes(), (hf_dir / "chat_template.json").read_bytes()
    with pytest.raises(ConversionError, match="chat_template.json missing or unreadable"):
        hf_overlay.write_unified_chat_template(out, base_dir)
    assert ((hf_dir / "tokenizer_config.json").read_bytes(), (hf_dir / "chat_template.json").read_bytes()) == before
    assert not list(hf_dir.glob("*.writing"))
    with pytest.raises(ConversionError, match="chat_template.json missing or unreadable"):
        hf_overlay.chat_template_record(out, base_dir, unified_chat_template=True)
    assert (
        hf_overlay.chat_template_record(out, base_dir, unified_chat_template=False)["processor_template_sha256"] is None
    )
    # base files no json.dumps formatting reproduces: every key still equal, formatting_reproduced False per file
    (base / "chat_template.json").write_text('{ "chat_template" : ' + json.dumps(FAKE_PROCESSOR_TEMPLATE) + " }")
    (base / "tokenizer_config.json").write_text('{ "chat_template" :"BASE-TOKENIZER-TEMPLATE" ,"model_max_length":5 }')
    record = hf_overlay.write_unified_chat_template(out, base_dir)
    assert record["formatting_reproduced"] == {"tokenizer_config.json": False, "chat_template.json": False}
    assert record["templates_already_equal"] is False
    assert json.loads((hf_dir / "tokenizer_config.json").read_text()) == {
        "chat_template": FAKE_CANONICAL_TEMPLATE,
        "model_max_length": 5,
    }
    assert json.loads((hf_dir / "chat_template.json").read_text()) == {"chat_template": FAKE_CANONICAL_TEMPLATE}
    hf_overlay.verify_unified_tokenizer_config(out, base_dir)
    hf_overlay.verify_unified_template_files(out, base_dir)
    # a processor template without the exact reasoning line cannot be canonicalized: refused, nothing written
    (base / "chat_template.json").write_text('{"chat_template": "NO-REASONING-BRANCH"}')
    before = (hf_dir / "tokenizer_config.json").read_bytes(), (hf_dir / "chat_template.json").read_bytes()
    with pytest.raises(ConversionError, match="does not carry the reasoning line"):
        hf_overlay.write_unified_chat_template(out, base_dir)
    assert ((hf_dir / "tokenizer_config.json").read_bytes(), (hf_dir / "chat_template.json").read_bytes()) == before
    assert not list(hf_dir.glob("*.writing"))
    with pytest.raises(ConversionError, match="does not carry the reasoning line"):
        hf_overlay.verify_sidecars(out, base_dir, unified_chat_template=True)
    twice = FAKE_PROCESSOR_TEMPLATE + "\n" + hf_overlay.PROCESSOR_REASONING_LINE
    with pytest.raises(ConversionError, match="exactly once \\(2 found\\)"):
        hf_overlay.canonicalize_processor_template(twice)


def test_main_unified_chat_template_option_identity_and_skip(
    tmp_path: Path, base: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI option end to end on the fake base: the export's ``tokenizer_config.json`` and ``chat_template.json`` both
    carry the canonical processor template, the marker records the option under ``options``, ``identity`` and
    ``chat_template`` (both shas); the identical request skips; a request WITHOUT the option on that output is refused
    (and the reverse on a default output) until ``--force``; a staging dir stamped without the option is not reused by a
    request with it; a base without ``chat_template.json`` fails closed."""
    calls = _install_fake_overlay(monkeypatch)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "model.pt").write_bytes(b"\0" * 10)
    out = str(tmp_path / "out")
    args = ["--ckpt-dir", str(ckpt), "--base", str(base), "--out", out, "--no-verify", "--min-tensors", "3"]

    assert mat.main([*args, "--unified-chat-template"]) == 0
    assert "MATERIALIZE DONE" in capsys.readouterr().out and len(calls) == 1
    marker = _marker(out)
    assert marker["identity"] == {
        "conversion": mat.CONVERSION_VERSION,
        "source_sha256": SHA_ZERO10,
        "source_bytes": 10,
        "base_digest": mat.base_identity(str(base))["digest"],
        "unified_chat_template": True,
    }
    assert marker["options"]["unified_chat_template"] is True
    assert marker["chat_template"] == {
        "unified": True,
        "processor_template_sha256": _sha(FAKE_PROCESSOR_TEMPLATE),
        "base_tokenizer_template_sha256": _sha("BASE-TOKENIZER-TEMPLATE"),
        "tokenizer_template_sha256": _sha(FAKE_CANONICAL_TEMPLATE),
        "template_mode": "processor-canonical",
        "files": ["tokenizer_config.json", "chat_template.json"],
        "shipped": {
            "tokenizer_config.json": _sha(FAKE_CANONICAL_TEMPLATE),
            "chat_template.json": _sha(FAKE_CANONICAL_TEMPLATE),
        },
        "canonical_diff": hf_overlay.canonical_template_diff(FAKE_PROCESSOR_TEMPLATE),
        "serve_assumption": hf_overlay.UNIFIED_SERVE_ASSUMPTION,
        "preferred_sidecars": [],
    }
    assert marker["sidecars"]["chat_template"]["unified"] is True
    assert marker["sidecars"]["chat_template"]["template_mode"] == "processor-canonical"
    assert marker["sidecars"]["chat_template"]["formatting_reproduced"] == {
        "tokenizer_config.json": True,
        "chat_template.json": True,
    }
    assert marker["verify"]["sidecars_identical"] == len(BASE_FILES) - 2
    assert marker["verify"]["sidecars_unified_chat_template"] == ["tokenizer_config.json", "chat_template.json"]
    assert json.loads(Path(out, "tokenizer_config.json").read_text())["chat_template"] == FAKE_CANONICAL_TEMPLATE
    assert json.loads(Path(out, "chat_template.json").read_text()) == {"chat_template": FAKE_CANONICAL_TEMPLATE}
    assert Path(out, "chat_template.json").read_bytes() != (base / "chat_template.json").read_bytes()
    assert hf_overlay.verify_sidecars(out, str(base), unified_chat_template=True)["sidecars_identical"] == 6
    # the identical request skips (the identity carries the option)
    assert mat.main([*args, "--unified-chat-template"]) == 0
    assert "idempotent skip" in capsys.readouterr().out and len(calls) == 1
    # a request WITHOUT the option on this output is refused; nothing rebuilt or deleted
    assert mat.main(args) == 1
    assert "built with --unified-chat-template, this request is without it" in capsys.readouterr().err
    assert len(calls) == 1 and _marker(out)["identity"]["unified_chat_template"] is True
    # --force without the option rebuilds the default output: the base's bytes, no key in the identity
    assert mat.main([*args, "--force"]) == 0 and len(calls) == 2
    marker = _marker(out)
    assert "unified_chat_template" not in marker["identity"] and marker["options"]["unified_chat_template"] is False
    assert Path(out, "tokenizer_config.json").read_bytes() == (base / "tokenizer_config.json").read_bytes()
    assert Path(out, "chat_template.json").read_bytes() == (base / "chat_template.json").read_bytes()
    assert marker["chat_template"]["tokenizer_template_sha256"] == _sha("BASE-TOKENIZER-TEMPLATE")
    assert "chat_template" not in marker["sidecars"] and "sidecars_unified_chat_template" not in marker["verify"]
    # and a request WITH the option on a default output is refused the same way
    assert mat.main([*args, "--unified-chat-template"]) == 1
    assert "built without --unified-chat-template, this request is with it" in capsys.readouterr().err
    assert len(calls) == 2
    # a staging dir stamped without the option is not reused by a request with it: one identity for stamp and marker
    out2 = str(tmp_path / "out2")
    hf_overlay.overlay_consolidated_onto_hf(str(ckpt), str(base), out2 + ".tmp")
    atomic_write_json(f"{out2}.tmp/{mat.SOURCE_STAMP}", {"identity": marker["identity"]})
    assert mat.main([*args[:4], "--out", out2, *args[6:], "--unified-chat-template"]) == 0
    assert len(calls) == 4 and _marker(out2)["reused_staging"] is False
    assert _marker(out2)["identity"]["unified_chat_template"] is True and not os.path.exists(out2 + ".tmp")
    # a base without chat_template.json (a text-only base) fails closed under the option
    base3 = tmp_path / "base3"
    _fake_base(base3, {k: v for k, v in BASE_FILES.items() if k != "chat_template.json"})
    out3 = str(tmp_path / "out3")
    assert mat.main([*args[:2], "--base", str(base3), "--out", out3, *args[6:], "--unified-chat-template"]) == 1
    assert "chat_template.json missing or unreadable" in capsys.readouterr().err and not os.path.exists(out3)
    assert mat.main([*args[:2], "--base", str(base3), "--out", out3, *args[6:]]) == 0, (
        "the default needs no processor template"
    )
    assert _marker(out3)["chat_template"]["processor_template_sha256"] is None


def test_unified_chat_template_refuses_a_base_whose_preferred_template_sidecar_diverges(
    tmp_path: Path, base: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An accepted base with a divergent ``chat_template.jinja`` used to
    publish both canonical JSON templates and the marker under ``--unified-chat-template --no-verify`` while the tokenizer
    loader renders the Jinja (it prefers that file over ``tokenizer_config.json`` on 4.57.6 and 5.8.1). Now: refused before
    anything is published, with and without ``--no-verify``, naming the file (the overlay never runs, no ``<out>`` and no
    ``<out>.tmp``); the default (non-unified) build still accepts the same base; without the sidecar the unified build
    publishes with no preferred sidecar in the export; a ``chat_template.jinja`` EQUAL to the canonical text is allowed,
    synced verbatim and recorded; an export whose Jinja diverges afterwards fails the rule; ``additional_chat_templates/``
    and a ``chat_template`` key in ``processor_config.json`` are refused the same way (the key allowed when equal). On
    the parent the divergent base publishes (exit 0, marker written)."""
    calls = _install_fake_overlay(monkeypatch)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "model.pt").write_bytes(b"\0" * 10)
    out = str(tmp_path / "out")
    common = ["--ckpt-dir", str(ckpt), "--base", str(base), "--min-tensors", "3"]
    (base / "chat_template.jinja").write_text("DIVERGENT-JINJA")
    for extra in ([], ["--no-verify"]):
        assert mat.main([*common, "--out", out, "--unified-chat-template", *extra]) == 1
        err = capsys.readouterr().err
        assert (
            "REFUSED before anything is published" in err and "chat_template.jinja" in err and "tokenizer loader" in err
        )
        assert calls == [] and not os.path.exists(out) and not os.path.exists(out + ".tmp")
    conflicts, allowed = hf_overlay.unified_template_conflicts(str(base))
    assert [c["file"] for c in conflicts] == ["chat_template.jinja"] and conflicts[0]["equals_canonical"] is False
    assert conflicts[0]["sha256"] == _sha("DIVERGENT-JINJA") and allowed == []
    # the default build accepts the same base (today's behaviour: the sidecar is copied verbatim and verified identical)
    assert mat.main([*common, "--out", out + "_default", "--no-verify"]) == 0 and len(calls) == 1
    assert Path(out + "_default", "chat_template.jinja").read_text() == "DIVERGENT-JINJA"
    # without the sidecar the unified build publishes, with no preferred sidecar in the export
    (base / "chat_template.jinja").unlink()
    assert mat.main([*common, "--out", out, "--unified-chat-template", "--no-verify"]) == 0 and len(calls) == 2
    marker = _marker(out)
    assert marker["chat_template"]["preferred_sidecars"] == [] and not os.path.exists(
        os.path.join(out, "chat_template.jinja")
    )
    assert json.loads(Path(out, "tokenizer_config.json").read_text())["chat_template"] == FAKE_CANONICAL_TEMPLATE
    assert json.loads(Path(out, "chat_template.json").read_text())["chat_template"] == FAKE_CANONICAL_TEMPLATE
    # a Jinja sidecar EQUAL to the canonical text: allowed, synced verbatim, recorded, and the rule accepts the export
    (base / "chat_template.jinja").write_text(FAKE_CANONICAL_TEMPLATE)
    out2 = str(tmp_path / "out2")
    assert mat.main([*common, "--out", out2, "--unified-chat-template", "--no-verify"]) == 0 and len(calls) == 3
    assert _marker(out2)["chat_template"]["preferred_sidecars"] == [
        {
            "file": "chat_template.jinja",
            "sha256": _sha(FAKE_CANONICAL_TEMPLATE),
            "preferred_by": "the tokenizer loader, over tokenizer_config.json's chat_template (4.57.6 and 5.8.1)",
            "equals_canonical": True,
        }
    ]
    assert Path(out2, "chat_template.jinja").read_text() == FAKE_CANONICAL_TEMPLATE
    assert hf_overlay.verify_sidecars(out2, str(base), unified_chat_template=True) == {
        "sidecars_identical": 7,
        "sidecars_unified_chat_template": ["tokenizer_config.json", "chat_template.json"],
    }
    Path(out2, "chat_template.jinja").write_text("DIVERGENT-JINJA")
    with pytest.raises(ConversionError, match="chat_template.jinja differs from base"):
        hf_overlay.verify_sidecars(out2, str(base), unified_chat_template=True)  # byte identity catches it first
    (base / "chat_template.jinja").write_text("DIVERGENT-JINJA")  # base and export agree on a divergent Jinja: the rule
    with pytest.raises(ConversionError, match="chat_template.jinja whose text is not the canonical processor template"):
        hf_overlay.verify_sidecars(out2, str(base), unified_chat_template=True)
    (base / "chat_template.jinja").unlink()
    # named template files and a processor_config.json chat_template key
    (base / "additional_chat_templates").mkdir()
    (base / "additional_chat_templates" / "tool_use.jinja").write_text("X")
    assert mat.main([*common, "--out", str(tmp_path / "out3"), "--unified-chat-template", "--no-verify"]) == 1
    assert "additional_chat_templates/tool_use.jinja" in capsys.readouterr().err and len(calls) == 3
    shutil.rmtree(base / "additional_chat_templates")
    (base / "processor_config.json").write_text(json.dumps({"chat_template": "OTHER"}))
    assert mat.main([*common, "--out", str(tmp_path / "out4"), "--unified-chat-template", "--no-verify"]) == 1
    assert "processor_config.json:chat_template" in capsys.readouterr().err and len(calls) == 3
    (base / "processor_config.json").write_text(json.dumps({"chat_template": FAKE_CANONICAL_TEMPLATE}))
    out5 = str(tmp_path / "out5")
    assert mat.main([*common, "--out", out5, "--unified-chat-template", "--no-verify"]) == 0 and len(calls) == 4
    assert [p["file"] for p in _marker(out5)["chat_template"]["preferred_sidecars"]] == [
        "processor_config.json:chat_template"
    ]
    assert hf_overlay.PREFERRED_TEMPLATE_SIDECARS == (
        "chat_template.jinja",
        "additional_chat_templates/",
        "processor_config.json:chat_template",
    )


def _tree_bytes(root: str) -> dict[str, bytes]:
    """Every file under ``root`` with its bytes, by relative path (the byte-identity assertion of an untouched output)."""
    return {
        os.path.relpath(os.path.join(directory, name), root): Path(directory, name).read_bytes()
        for directory, _, names in os.walk(root)
        for name in names
    }


def _shape_template_directory(base: Path, shape: str) -> None:
    """``<base>/additional_chat_templates/`` in one of the shapes the directory rule covers; ``absent`` leaves the base alone."""
    directory = base / "additional_chat_templates"
    if shape == "absent":
        return
    directory.mkdir()
    if shape == "nested_only":
        (directory / "nested").mkdir()
        (directory / "nested" / "default.jinja").write_text("NESTED")
    elif shape == "one_file":
        (directory / "tool_use.jinja").write_text("X")
    else:
        assert shape == "empty", shape


@pytest.mark.parametrize("shape", ["absent", "empty", "nested_only", "one_file"])
def test_unified_template_directory_rule_is_one_helper_for_preflight_and_verifier(
    shape: str, tmp_path: Path, base: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """On the parent the preflight recorded only direct regular
    files under ``additional_chat_templates/``, so an EMPTY directory or one holding only sub-directories passed it while
    ``verify_sidecars`` refuses the directory itself. Now ONE helper (``unified_template_directory_conflict``) decides for
    both: the directory is a conflict whenever it is a directory, whatever it holds, so for every shape the preflight
    refuses iff the verifier refuses, with one text naming the directory and its direct entries; the per-file records
    keep their sha evidence; without the directory both accept, and the default (non-unified) build still mirrors an
    empty directory. Both loaders ignore an empty or nested-only directory (``is_dir()`` then ``glob("*.jinja")``,
    transformers 4.57.6 and 5.8.1); the rule refuses it anyway, fail closed. On the parent ``empty`` and
    ``nested_only`` fail: no conflict is recorded while the verifier refuses, and the CLI publishes nothing only because
    the verifier fails after the overlay."""
    calls = _install_fake_overlay(monkeypatch)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "model.pt").write_bytes(b"\0" * 10)
    common = ["--ckpt-dir", str(ckpt), "--base", str(base), "--min-tensors", "3", "--no-verify"]
    out = str(tmp_path / "out")
    assert mat.main([*common, "--out", out, "--unified-chat-template"]) == 0 and len(calls) == 1
    capsys.readouterr()
    _shape_template_directory(base, shape)
    base_dir = str(base)
    record = hf_overlay.unified_template_directory_conflict(base_dir)
    conflicts, allowed = hf_overlay.unified_template_conflicts(base_dir)
    report, verifier_message = None, None
    try:
        report = hf_overlay.verify_sidecars(out, base_dir, unified_chat_template=True)
    except ConversionError as error:
        verifier_message = str(error)
    assert bool(conflicts) is (verifier_message is not None), "the preflight refuses iff the verifier refuses"
    assert allowed == []
    if shape == "absent":
        assert record is None and conflicts == []
        assert report == {
            "sidecars_identical": len(BASE_FILES) - 2,
            "sidecars_unified_chat_template": ["tokenizer_config.json", "chat_template.json"],
        }
        return
    expected_entries = {
        "empty": [],
        "nested_only": [{"name": "nested", "kind": "directory", "sha256": None}],
        "one_file": [{"name": "tool_use.jinja", "kind": "file", "sha256": _sha("X")}],
    }[shape]
    assert record == {
        "file": "additional_chat_templates/",
        "sha256": None,
        "preferred_by": hf_overlay.TEMPLATE_DIRECTORY_PREFERRED_BY,
        "equals_canonical": False,
        "entries": expected_entries,
    }
    assert conflicts[0] == record, "the directory itself is the first conflict"
    assert [c["file"] for c in conflicts[1:]] == (
        ["additional_chat_templates/tool_use.jinja"] if shape == "one_file" else []
    )
    if shape == "one_file":
        assert conflicts[1]["sha256"] == _sha("X") and conflicts[1]["equals_canonical"] is False
        assert conflicts[1]["preferred_by"] == hf_overlay.NAMED_TEMPLATE_PREFERRED_BY
    description = hf_overlay.describe_template_conflict(record)
    listed = {
        "empty": "direct entries: none",
        "nested_only": "direct entries: nested/",
        "one_file": "direct entries: tool_use.jinja",
    }
    assert (
        description.startswith("additional_chat_templates/ (a directory, refused whatever it holds; ")
        and listed[shape] in description
    )
    assert verifier_message is not None and description in verifier_message
    assert "not allowed under --unified-chat-template" in verifier_message
    with pytest.raises(ConversionError, match="REFUSED before anything is published") as refusal:
        hf_overlay.refuse_unified_template_conflicts(base_dir)
    assert description in str(refusal.value), "one text names the directory in the preflight and in the verifier"
    # the CLI refuses the same base before anything is published: no overlay call, no <out>, no <out>.tmp
    fresh = str(tmp_path / "fresh")
    assert mat.main([*common, "--out", fresh, "--unified-chat-template"]) == 1
    err = capsys.readouterr().err
    assert "REFUSED before anything is published" in err and description in err
    assert len(calls) == 1 and not os.path.exists(fresh) and not os.path.exists(fresh + ".tmp")
    if shape == "empty":
        # the harmless case the loaders ignore: the default build still accepts it and mirrors the empty directory
        assert mat.main([*common, "--out", out + "_default"]) == 0 and len(calls) == 2
        assert os.path.isdir(os.path.join(out + "_default", "additional_chat_templates"))
        assert hf_overlay.verify_sidecars(out + "_default", base_dir) == {"sidecars_identical": len(BASE_FILES) + 1}


@pytest.mark.parametrize("case", ["empty_directory", "nested_only_directory", "base_without_structural_identity"])
def test_force_rebuild_keeps_the_existing_output_when_a_torch_free_preflight_refuses(
    case: str, tmp_path: Path, base: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Probe: on the parent ``--force --unified-chat-template --no-verify`` against a base
    with an EMPTY ``additional_chat_templates/`` removed the existing output, ran the overlay, failed the final sidecar
    verifier and left only ``<out>.tmp``. Now every torch-free refusal that reads the base alone (``preflight``) runs
    before the removal, before ``<out>.tmp`` and before the overlay: the request is refused, the existing output is
    byte-identical afterwards, no ``<out>.tmp`` exists and the overlay was not called, with and without ``--force``. The
    third case is the other base-only refusal that used to come after the overlay (``config.json <- base`` needs the
    base's structural identity)."""
    calls = _install_fake_overlay(monkeypatch)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "model.pt").write_bytes(b"\0" * 10)
    out = str(tmp_path / "out")
    args = ["--ckpt-dir", str(ckpt), "--base", str(base), "--out", out, "--min-tensors", "3", "--no-verify"]
    args += ["--unified-chat-template"]
    assert mat.main(args) == 0 and len(calls) == 1 and os.path.isfile(os.path.join(out, mat.MATERIALIZED_MARKER))
    capsys.readouterr()
    before = _tree_bytes(out)
    assert len(before) >= 10
    if case == "base_without_structural_identity":
        (base / "config.json").write_text('{"model_type": "fake"}')
        expected = "model identity mismatch"
    else:
        _shape_template_directory(base, "empty" if case == "empty_directory" else "nested_only")
        expected = "REFUSED before anything is published"
    assert mat.main([*args, "--force"]) == 1
    err = capsys.readouterr().err
    assert expected in err
    if case != "base_without_structural_identity":
        assert "additional_chat_templates/ (a directory, refused whatever it holds" in err
    assert len(calls) == 1, "the overlay never ran"
    assert _tree_bytes(out) == before, "the existing output is byte-identical"
    assert not os.path.exists(out + ".tmp"), "no staging dir was created"
    # without --force the same request is refused the same way and the output is equally untouched
    assert mat.main(args) == 1 and expected in capsys.readouterr().err
    assert _tree_bytes(out) == before and not os.path.exists(out + ".tmp") and len(calls) == 1


def _default_export(tmp_path: Path, base: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], list[str], str]:
    """A DEFAULT (non-unified) export built by the CLI with the fake overlay: ``(calls, args, out)``; one overlay call made."""
    calls = _install_fake_overlay(monkeypatch)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "model.pt").write_bytes(b"\0" * 10)
    out = str(tmp_path / "out")
    args = ["--ckpt-dir", str(ckpt), "--base", str(base), "--out", out, "--min-tensors", "3", "--no-verify"]
    assert mat.main(args) == 0 and len(calls) == 1 and os.path.isfile(os.path.join(out, mat.MATERIALIZED_MARKER))
    return calls, args, out


@pytest.mark.parametrize(
    "case",
    [
        "tokenizer_config_malformed",
        "tokenizer_config_non_object",
        "chat_template_non_object",
        "generation_config_malformed",
    ],
)
def test_force_rebuild_keeps_the_existing_output_when_a_base_json_sidecar_does_not_parse(
    case: str, tmp_path: Path, base: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """On the parent the DEFAULT path's preflight did not parse the
    base's ``tokenizer_config.json``, so a malformed (``{``) or non-object (``[]``) file passed it; under ``--force --no-verify``
    the existing export was removed, the overlay ran, and the unconditional ``chat_template_record`` refused on
    ``read_tokenizer_config(base)``: exit 1, the published directory gone, only ``<out>.tmp`` left. Now the preflight parses
    every JSON sidecar the base carries (``tokenizer_config.json`` and ``chat_template.json`` as objects; the overlay's
    transformers loads and the template record read them after the overlay) before anything is removed: refused with a
    named reason, the existing output byte-identical, no ``<out>.tmp``, no overlay call; a fresh ``--out`` is refused the
    same way with nothing created. On the parent the two tokenizer_config cases lose the output and the other two publish
    (the default path never read those files torch-free)."""
    calls, args, out = _default_export(tmp_path, base, monkeypatch)
    capsys.readouterr()
    before = _tree_bytes(out)
    name, text, reason = {
        "tokenizer_config_malformed": ("tokenizer_config.json", "{", "missing or unreadable"),
        "tokenizer_config_non_object": ("tokenizer_config.json", "[]", "is not a JSON object"),
        "chat_template_non_object": ("chat_template.json", "[]", "is not a JSON object"),
        "generation_config_malformed": ("generation_config.json", "{", "is not valid JSON"),
    }[case]
    (base / name).write_text(text)
    assert mat.main([*args, "--force"]) == 1
    err = capsys.readouterr().err
    assert "REFUSED before anything is published" in err and name in err and reason in err
    assert len(calls) == 1, "the overlay never ran"
    assert _tree_bytes(out) == before, "the existing output is byte-identical"
    assert not os.path.exists(out + ".tmp"), "no staging dir was created"
    fresh = str(tmp_path / "fresh")
    assert mat.main([*args[:4], "--out", fresh, *args[6:]]) == 1 and reason in capsys.readouterr().err
    assert len(calls) == 1 and not os.path.exists(fresh) and not os.path.exists(fresh + ".tmp")


@pytest.mark.parametrize("case", ["nested_directory", "dangling_symlink", "jinja_named_subdirectory"])
def test_default_preflight_refuses_a_named_template_directory_with_entries_that_are_not_regular_files(
    case: str, tmp_path: Path, base: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The DEFAULT path copies ``additional_chat_templates/`` verbatim after the
    overlay (``copytree``) and verifies it with ``filecmp.cmpfiles``, which reports a sub-directory as a mismatch, while a
    dangling symlink fails the copy with ``shutil.Error`` (an OSError ``main`` used to let escape as a traceback). Now the
    preflight decides it: a directory with an entry that is not a regular file is refused before anything is removed,
    naming the entry and its kind; a directory of regular files only is still accepted and mirrored. A sub-directory NAMED
    ``*.jinja`` is the loader case: ``Path.glob("*.jinja")`` returns it, so the loaders do not ignore it; the
    record lists it as a ``directory`` and the helper's docstring says so. On the parent the nested cases lose the existing
    output after the overlay and the symlink case raises ``shutil.Error`` out of ``main``."""
    calls, args, out = _default_export(tmp_path, base, monkeypatch)
    capsys.readouterr()
    before = _tree_bytes(out)
    directory = base / "additional_chat_templates"
    directory.mkdir()
    (directory / "default.jinja").write_text("D")  # a regular file beside the irregular entry
    if case == "nested_directory":
        (directory / "nested").mkdir()
        entry, kind = "nested", "directory"
    elif case == "dangling_symlink":
        os.symlink(str(tmp_path / "nowhere.jinja"), str(directory / "broken.jinja"))
        entry, kind = "broken.jinja", "other"
    else:
        (directory / "tool_use.jinja").mkdir()
        entry, kind = "tool_use.jinja", "directory"
    record = hf_overlay.unified_template_directory_conflict(str(base))
    assert record is not None and {"name": entry, "kind": kind, "sha256": None} in record["entries"]
    assert mat.main([*args, "--force"]) == 1
    err = capsys.readouterr().err
    assert "REFUSED before anything is published" in err and "additional_chat_templates/" in err
    assert f"{entry} [{kind}]" in err and "not regular files" in err
    assert len(calls) == 1 and _tree_bytes(out) == before and not os.path.exists(out + ".tmp")
    fresh = str(tmp_path / "fresh")
    assert mat.main([*args[:4], "--out", fresh, *args[6:]]) == 1 and "not regular files" in capsys.readouterr().err
    assert len(calls) == 1 and not os.path.exists(fresh) and not os.path.exists(fresh + ".tmp")
    if case == "jinja_named_subdirectory":
        doc = hf_overlay.unified_template_directory_conflict.__doc__ or ""
        assert "*.jinja" in doc and "IsADirectoryError" in doc, "the loader clause for a sub-directory named *.jinja"
    # regular files only: the default build accepts and mirrors the directory (today's behaviour kept)
    if case == "dangling_symlink":
        (directory / entry).unlink()
    else:
        (directory / entry).rmdir()
    regular = str(tmp_path / "regular")
    assert mat.main([*args[:4], "--out", regular, *args[6:]]) == 0 and len(calls) == 2
    assert hf_overlay.verify_sidecars(regular, str(base)) == {"sidecars_identical": len(BASE_FILES) + 1}


def test_preflight_refuses_an_unreadable_base_with_a_named_reason_instead_of_a_traceback(
    tmp_path: Path, base: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An OSError other than FileNotFoundError raised while the preflight reads the base (an
    unreadable ``additional_chat_templates/`` listed by ``base_identity``) is refused with a named reason (exit 1, the class
    and the path in the message) instead of escaping ``main`` as a traceback; the existing output stays byte-identical. The
    PermissionError is injected for the base's directory path (``os.listdir`` wrapped, so the case is deterministic as root
    too) and, when the test does not run as root, also produced for real with mode 000. On the parent the PermissionError
    escapes ``main``."""
    calls, args, out = _default_export(tmp_path, base, monkeypatch)
    capsys.readouterr()
    before = _tree_bytes(out)
    directory = base / "additional_chat_templates"
    directory.mkdir()
    (directory / "default.jinja").write_text("D")
    real_listdir = os.listdir

    def listdir(path: object = ".") -> list[str]:
        try:
            text = os.fspath(path)  # type: ignore[arg-type]
        except TypeError:
            return real_listdir(path)  # type: ignore[arg-type]
        if os.path.abspath(text) == str(directory):
            raise PermissionError(13, "Permission denied", text)
        return real_listdir(path)  # type: ignore[arg-type]

    with monkeypatch.context() as patch:
        patch.setattr(os, "listdir", listdir)
        assert mat.main([*args, "--force"]) == 1
        err = capsys.readouterr().err
    assert "REFUSED before anything is published" in err and "not readable" in err and "PermissionError" in err
    assert str(directory) in err and len(calls) == 1
    assert _tree_bytes(out) == before and not os.path.exists(out + ".tmp")
    if os.geteuid() != 0:  # the real thing, when the test user cannot bypass permissions
        directory.chmod(0o000)
        try:
            assert mat.main([*args, "--force"]) == 1
            err = capsys.readouterr().err
        finally:
            directory.chmod(0o755)
        assert "REFUSED before anything is published" in err and "PermissionError" in err and len(calls) == 1
        assert _tree_bytes(out) == before and not os.path.exists(out + ".tmp")


@contextlib.contextmanager
def _unreadable_file(path: Path) -> Iterator[None]:
    """``open()`` of exactly ``path`` raises PermissionError while the context is active (deterministic as root, where mode
    000 bars nothing: the test bed's reviewed pattern); every other open is the real one."""
    real_open = builtins.open

    def guarded_open(file: object, *args: object, **kwargs: object) -> object:
        try:
            text = os.fspath(file)  # type: ignore[arg-type]
        except TypeError:
            return real_open(file, *args, **kwargs)  # type: ignore[arg-type]
        if isinstance(text, str) and os.path.abspath(text) == str(path):
            raise PermissionError(13, "Permission denied", text)
        return real_open(file, *args, **kwargs)  # type: ignore[arg-type]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builtins, "open", guarded_open)
        yield


@pytest.mark.parametrize("shape", ["truncated", "header_garbage", "header_keys_differ", "mode_000"])
def test_preflight_refuses_a_base_shard_the_safetensors_header_check_rejects_before_anything_is_removed(
    shape: str, tmp_path: Path, base: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The overlay's ``from_pretrained`` opens every base shard the index lists AFTER an
    existing output is removed, but on the parent the preflight parsed no shard (``base_identity`` records sizes only): a
    shard truncated by one byte, one with a garbage header, one whose header lists other tensors than the index maps to it
    and an unreadable one all passed it, so ``--force --no-verify`` removed the export and the overlay failed on the shard
    (``safetensors.safe_open``: ``incomplete metadata`` / ``header too large``). Now the preflight runs
    ``shard_status`` (the header check ``verify_index`` applies to the output) on every base shard: refused in the one
    family naming the shard and the reason, the existing output byte-identical, no ``<out>.tmp``, no overlay call; a fresh
    ``--out`` is refused with nothing created. The unreadable shard is injected (``open`` of that path raises
    PermissionError, deterministic as root, where mode 000 bars nothing) and, when the test does not run as root, also
    produced for real with mode 000. On the parent every shape passes the preflight and the fake overlay rebuilds the
    export (exit 0)."""
    calls, args, out = _default_export(tmp_path, base, monkeypatch)
    capsys.readouterr()
    before = _tree_bytes(out)
    shard = base / SHARD1
    if shape == "truncated":
        _truncate(str(shard))
        reason = "truncated or partially written"
    elif shape == "header_garbage":
        with open(shard, "r+b") as handle:
            handle.write(b"\xff" * 8)
        reason = "unreadable safetensors header (header length 18446744073709551615 is implausible)"
    elif shape == "header_keys_differ":
        _fake_safetensors(str(shard), ["a.weight", "z.extra"])  # the index maps a.weight alone to this shard
        reason = "tensors differ from the index; missing=[] extra=['z.extra']"
    else:
        reason = "unreadable safetensors header ([Errno 13] Permission denied"

    def guard() -> contextlib.AbstractContextManager[None]:
        return _unreadable_file(shard) if shape == "mode_000" else contextlib.nullcontext()

    with guard():
        assert mat.main([*args, "--force"]) == 1
        err = capsys.readouterr().err
    assert "REFUSED before anything is published" in err and f"base shard {shard}: " in err and reason in err
    assert "Traceback" not in err and len(calls) == 1, "the overlay never ran"
    assert _tree_bytes(out) == before, "the existing output is byte-identical"
    assert not os.path.exists(out + ".tmp"), "no staging dir was created"
    fresh = str(tmp_path / "fresh")
    with guard():
        assert mat.main([*args[:4], "--out", fresh, *args[6:]]) == 1 and reason in capsys.readouterr().err
    assert len(calls) == 1 and not os.path.exists(fresh) and not os.path.exists(fresh + ".tmp")
    if shape == "mode_000" and os.geteuid() != 0:  # the real thing, when the test user cannot bypass permissions
        shard.chmod(0o000)
        try:
            assert mat.main([*args, "--force"]) == 1
            err = capsys.readouterr().err
        finally:
            shard.chmod(0o644)
        assert "REFUSED before anything is published" in err and f"base shard {shard}: " in err and reason in err
        assert len(calls) == 1 and _tree_bytes(out) == before and not os.path.exists(out + ".tmp")


@pytest.mark.parametrize(
    "case",
    [
        "tokenizer_json_array",
        "preprocessor_config_array",
        "video_preprocessor_config_array",
        "processor_config_array",
        "generation_config_array",
        "chat_template_empty_object",
        "chat_template_empty_string",
    ],
)
def test_preflight_refuses_a_json_sidecar_the_loaders_reject_although_it_parses(
    case: str, tmp_path: Path, base: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Well-formed JSON the loaders reject passed the earlier preflight, which required an
    object for ``tokenizer_config.json`` and ``chat_template.json`` only: ``[]`` for ``tokenizer.json``,
    ``preprocessor_config.json``, ``video_preprocessor_config.json`` and ``processor_config.json`` (``AutoProcessor``:
    ``'list' object has no attribute 'get'``) and for ``generation_config.json`` (``GenerationConfig``: ``list indices must
    be integers``), and ``{}`` for ``chat_template.json`` (the processor loader's legacy path indexes ``["chat_template"]``:
    a KeyError, which ``main`` does not catch), read on the container's transformers 5.13.0 against the base of record's
    sidecars; the overlay calls these loaders after an existing output is removed. Now every JSON sidecar must
    be a JSON object and ``chat_template.json``, when present, must carry a non-empty ``chat_template`` string: refused in
    the one family naming the file and the rule, the existing output byte-identical, no ``<out>.tmp``, no overlay call; a
    fresh ``--out`` is refused with nothing created; the loader-accepted ``{}`` of the other sidecars stays accepted. On
    the parent every case passes the preflight and the fake overlay replaces the export (exit 0)."""
    calls, args, out = _default_export(tmp_path, base, monkeypatch)
    capsys.readouterr()
    before = _tree_bytes(out)
    array = "is not a JSON object (a JSON array)"
    name, text, reason = {
        "tokenizer_json_array": ("tokenizer.json", "[]", array),
        "preprocessor_config_array": ("preprocessor_config.json", "[]", array),
        "video_preprocessor_config_array": ("video_preprocessor_config.json", "[]", array),
        "processor_config_array": (
            "processor_config.json",
            "[]",
            array,
        ),  # absent from the fake base: present and broken
        "generation_config_array": ("generation_config.json", "[]", array),
        "chat_template_empty_object": (
            "chat_template.json",
            "{}",
            "carries no non-empty 'chat_template' string (found absent or null)",
        ),
        "chat_template_empty_string": (
            "chat_template.json",
            '{"chat_template": ""}',
            "carries no non-empty 'chat_template' string (found an empty JSON string)",
        ),
    }[case]
    (base / name).write_text(text)
    assert mat.main([*args, "--force"]) == 1
    err = capsys.readouterr().err
    assert "REFUSED before anything is published" in err and f"{base / name} {reason}" in err and "Traceback" not in err
    assert len(calls) == 1, "the overlay never ran"
    assert _tree_bytes(out) == before, "the existing output is byte-identical"
    assert not os.path.exists(out + ".tmp"), "no staging dir was created"
    fresh = str(tmp_path / "fresh")
    assert mat.main([*args[:4], "--out", fresh, *args[6:]]) == 1 and reason in capsys.readouterr().err
    assert len(calls) == 1 and not os.path.exists(fresh) and not os.path.exists(fresh + ".tmp")
    if case == "generation_config_array":  # the rule is "a JSON object", not "non-empty": {} loads and builds
        (base / name).write_text("{}")
        assert mat.main([*args[:4], "--out", fresh, *args[6:]]) == 0 and len(calls) == 2


@pytest.mark.parametrize(
    "case", ["weight_map_array", "weight_map_non_string_shard", "weight_map_empty_object", "index_array"]
)
def test_preflight_refuses_an_index_whose_weight_map_is_not_tensor_to_shard_names_without_a_traceback(
    case: str, tmp_path: Path, base: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """On the parent a well-formed index whose ``weight_map`` is a list escaped
    ``base_identity`` as ``AttributeError: 'list' object has no attribute 'values'`` and one mapping a tensor to a
    non-string as ``TypeError: join() argument must be str``, both out of ``main`` as tracebacks (before the removal, so
    the output was intact); an EMPTY ``weight_map`` passed the preflight and lost the existing output to the final
    ``verify_index`` after the overlay; an index that is a JSON array was refused outside the family. Now
    ``read_index_weight_map`` validates the shape for ``base_identity`` (``ValueError`` naming the file and what was
    found), the preflight refuses it in the one family before anything is removed, and ``tmp_complete`` reports the same
    shape inside ``<out>.tmp`` as not reusable with that reason."""
    calls, args, out = _default_export(tmp_path, base, monkeypatch)
    capsys.readouterr()
    before = _tree_bytes(out)
    text, reason = {
        "weight_map_array": (
            '{"weight_map": []}',
            "weight_map must be a non-empty JSON object mapping tensor names to shard names, found a JSON array",
        ),
        "weight_map_non_string_shard": (
            '{"weight_map": {"a.weight": 1}}',
            "weight_map must map tensor names to shard names (non-empty strings); ['a.weight'] map to [1]",
        ),
        "weight_map_empty_object": (
            '{"metadata": {}, "weight_map": {}}',
            "weight_map must be a non-empty JSON object mapping tensor names to shard names, found an empty JSON object",
        ),
        "index_array": ("[]", "the index is a JSON array, not a JSON object"),
    }[case]
    index = base / hf_overlay.INDEX_FILE
    index.write_text(text)
    with pytest.raises(ValueError, match=re.escape(f"{index} unreadable: {reason}")):
        mat.base_identity(str(base))
    with pytest.raises(ConversionError, match="REFUSED before anything is published") as refusal:
        mat.preflight(str(base), unified_chat_template=False)
    assert reason in str(refusal.value)
    assert mat.main([*args, "--force"]) == 1
    err = capsys.readouterr().err
    assert "REFUSED before anything is published" in err and reason in err and "Traceback" not in err
    assert len(calls) == 1 and _tree_bytes(out) == before and not os.path.exists(out + ".tmp")
    fresh = str(tmp_path / "fresh")
    assert mat.main([*args[:4], "--out", fresh, *args[6:]]) == 1 and reason in capsys.readouterr().err
    assert len(calls) == 1 and not os.path.exists(fresh) and not os.path.exists(fresh + ".tmp")
    staged = tmp_path / "staged.tmp"  # the same shape inside a staging dir: not reusable, with the reason, no traceback
    shutil.copytree(out, staged)
    (staged / hf_overlay.INDEX_FILE).write_text(text)
    assert mat.tmp_complete(str(staged)) == (False, f"{staged / hf_overlay.INDEX_FILE} unreadable: {reason}")


@pytest.mark.parametrize(
    "case",
    [
        "required_file_missing",
        "shard_missing",
        "index_malformed",
        "config_malformed",
        "config_array",
        "unified_template_without_string",
    ],
)
def test_every_preflight_refusal_is_in_the_one_message_family_and_names_what_it_found(
    case: str, tmp_path: Path, base: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``_refuse`` promised "the one message family of every preflight refusal" while on
    the parent the required-file and missing-shard refusals (FileNotFoundError), the index's ValueError and the unified
    reader's text left ``preflight`` without the prefix, and a malformed ``config.json`` was refused as "carries no
    structural identity" (it is unparseable, not identity-less). Now every refusal of ``preflight`` carries the prefix
    exactly once and says what it found (a text that already carries the prefix, the unified conflict text, is not prefixed
    twice); the existing output stays byte-identical, no ``<out>.tmp``, no overlay call, and a fresh ``--out`` is refused
    the same way."""
    calls, args, out = _default_export(tmp_path, base, monkeypatch)
    capsys.readouterr()
    before = _tree_bytes(out)
    extra: list[str] = []
    if case == "required_file_missing":
        (base / "tokenizer.json").unlink()
        found = f"the base {base} lacks tokenizer.json (no regular file at {base / 'tokenizer.json'})"
    elif case == "shard_missing":
        (base / SHARD2).unlink()
        found = f"base model incomplete at {base}: shard {base / SHARD2} listed in {base / hf_overlay.INDEX_FILE} is missing"
    elif case == "index_malformed":
        (base / hf_overlay.INDEX_FILE).write_text("{")
        found = f"{base / hf_overlay.INDEX_FILE} unreadable: Expecting property name enclosed in double quotes"
    elif case == "config_malformed":
        (base / "config.json").write_text("{")
        found = f"{base / 'config.json'} is not valid JSON (Expecting property name enclosed in double quotes"
    elif case == "config_array":
        (base / "config.json").write_text("[]")
        found = f"{base / 'config.json'} is not a JSON object (a JSON array)"
    else:
        (base / "tokenizer_config.json").write_text('{"model_max_length": 1}')  # an object: the default path accepts it
        found = f"{base / 'tokenizer_config.json'} carries no 'chat_template' string to unify"
        extra = ["--unified-chat-template"]
    assert mat.main([*args, "--force", *extra]) == 1
    err = capsys.readouterr().err
    assert err.count("REFUSED before anything is published") == 1 and found in err and "Traceback" not in err, err
    assert "carries no structural identity" not in err
    assert len(calls) == 1 and _tree_bytes(out) == before and not os.path.exists(out + ".tmp")
    fresh = str(tmp_path / "fresh")
    assert mat.main([*args[:4], "--out", fresh, *args[6:], *extra]) == 1 and found in capsys.readouterr().err
    assert len(calls) == 1 and not os.path.exists(fresh) and not os.path.exists(fresh + ".tmp")
    if (
        case == "unified_template_without_string"
    ):  # the unified conflict text already carries the prefix: once, not twice
        (base / "tokenizer_config.json").write_text(BASE_FILES["tokenizer_config.json"])
        (base / "chat_template.jinja").write_text("DIVERGENT")
        assert mat.main([*args[:4], "--out", fresh, *args[6:], *extra]) == 1
        err = capsys.readouterr().err
        assert err.count("REFUSED before anything is published") == 1 and "chat_template.jinja" in err
        assert len(calls) == 1 and not os.path.exists(fresh) and not os.path.exists(fresh + ".tmp")


@pytest.mark.parametrize(
    "case", ["config_json", "tokenizer_json", "generation_config_json", "tokenizer_config_json", "index"]
)
def test_preflight_refuses_a_utf8_byte_order_mark_in_front_of_valid_json_as_the_loaders_do(
    case: str, tmp_path: Path, base: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The earlier preflight decoded config.json and eight JSON sidecars from BYTES
    (``json.loads(bytes)`` detects a UTF-8 byte-order mark and strips it) while every reader after the removal decodes TEXT
    and rejects the mark (``model_identity``, ``_read_json_file``, transformers' loaders: ``Unexpected UTF-8 BOM``; AutoConfig /
    AutoProcessor / AutoTokenizer / GenerationConfig all fail on the base of record's files with a leading mark): on
    the parent a BOM-prefixed config.json passed the preflight, ``--force`` removed the export, the overlay ran and
    ``write_config_from_base`` refused (out gone, out.tmp left); a BOM-prefixed tokenizer.json or generation_config.json passed
    and the fake overlay replaced the export; tokenizer_config.json and the index were refused with json's own text, not naming
    the mark. Now ONE decoder serves the whole preflight (``conversion.hf.decode_json_bytes``: strict UTF-8 text, the mark refused
    by name): refused in the one family naming the file and the mark, the existing output byte-identical, no ``<out>.tmp``, no
    overlay call; a fresh ``--out`` is refused with nothing created; the same bytes without the mark build again."""
    calls, args, out = _default_export(tmp_path, base, monkeypatch)
    capsys.readouterr()
    before = _tree_bytes(out)
    name = {
        "config_json": "config.json",
        "tokenizer_json": "tokenizer.json",
        "generation_config_json": "generation_config.json",
        "tokenizer_config_json": "tokenizer_config.json",
        "index": hf_overlay.INDEX_FILE,
    }[case]
    path = base / name
    valid = path.read_bytes()
    path.write_bytes(b"\xef\xbb\xbf" + valid)
    assert json.loads(path.read_bytes()) == json.loads(valid), "json.loads(bytes) strips the mark: the decoder T6h used"
    with pytest.raises(ValueError, match="BOM"), open(path, encoding="utf-8") as handle:
        json.load(handle)  # what every text reader after the removal does with it
    assert mat.main([*args, "--force"]) == 1
    err = capsys.readouterr().err
    assert err.count("REFUSED before anything is published") == 1 and str(path) in err and "byte-order mark" in err, err
    assert "Traceback" not in err and len(calls) == 1, "the overlay never ran"
    assert _tree_bytes(out) == before, "the existing output is byte-identical"
    assert not os.path.exists(out + ".tmp"), "no staging dir was created"
    fresh = str(tmp_path / "fresh")
    assert mat.main([*args[:4], "--out", fresh, *args[6:]]) == 1 and "byte-order mark" in capsys.readouterr().err
    assert len(calls) == 1 and not os.path.exists(fresh) and not os.path.exists(fresh + ".tmp")
    with pytest.raises(hf_overlay.ByteOrderMarkError, match="byte-order mark"):
        hf_overlay.decode_json_bytes(path.read_bytes())
    assert hf_overlay.decode_json_bytes(valid) == json.loads(valid)
    path.write_bytes(valid)  # the same bytes without the mark build again
    assert mat.main([*args[:4], "--out", fresh, *args[6:]]) == 0 and len(calls) == 2


@pytest.mark.parametrize(
    "case",
    [
        "no_data_offsets",
        "entry_not_object",
        "header_list",
        "offsets_not_integers",
        "offsets_reversed",
        "metadata_not_object",
        "header_bom",
    ],
)
def test_shard_status_refuses_a_header_without_the_safetensors_shape_and_the_preflight_names_it(
    case: str, tmp_path: Path, base: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``safetensors_header`` / ``shard_status`` accepted any JSON as a header: on the parent
    a tensor entry without ``data_offsets`` escaped the preflight as KeyError, an entry that is ``5`` as TypeError, a list header
    as TypeError (``header.pop`` on a list), ``data_offsets`` = ``["x", "y"]`` was refused outside the family (ValueError from
    ``int()``), reversed offsets were refused with the size text, and a non-object ``__metadata__`` or a byte-order mark in front
    of the header JSON passed. Now ``safetensors_header`` decodes the header through the one decoder and validates its shape
    (``safetensors_header_shape_error``: a JSON object of tensor entries with a dtype string, a shape list and ``data_offsets`` of
    two non-negative integers begin <= end, an optional ``__metadata__`` object), so ``shard_status`` returns the reason: the
    preflight refuses in the one family naming the shard and the reason, ``tmp_complete`` reports a staged copy not reusable
    with it, and ``verify_index`` names it on the output."""
    calls, args, out = _default_export(tmp_path, base, monkeypatch)
    capsys.readouterr()
    before = _tree_bytes(out)
    valid_entry = {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}
    offsets_rule = "carries no data_offsets of two non-negative integers begin <= end"
    header, reason = {
        "no_data_offsets": (
            {"a.weight": {"dtype": "BF16", "shape": [2]}, "__metadata__": {"format": "pt"}},
            f"tensor 'a.weight' {offsets_rule} (found None)",
        ),
        "entry_not_object": (
            {"a.weight": 5, "__metadata__": {"format": "pt"}},
            "tensor 'a.weight' is a JSON number, not a JSON object",
        ),
        "header_list": ([1, 2, 3], "the header is a JSON array, not a JSON object"),
        "offsets_not_integers": (
            {"a.weight": {"dtype": "BF16", "shape": [2], "data_offsets": ["x", "y"]}, "__metadata__": {}},
            f"tensor 'a.weight' {offsets_rule} (found ['x', 'y'])",
        ),
        "offsets_reversed": (
            {"a.weight": {"dtype": "BF16", "shape": [2], "data_offsets": [4, 0]}, "__metadata__": {}},
            f"tensor 'a.weight' {offsets_rule} (found [4, 0])",
        ),
        "metadata_not_object": (
            {"a.weight": valid_entry, "__metadata__": 5},
            "__metadata__ is a JSON number, not a JSON object",
        ),
        "header_bom": (
            {"a.weight": valid_entry, "__metadata__": {"format": "pt"}},
            "starts with a UTF-8 byte-order mark (bytes EF BB BF), which the UTF-8 text readers of the loaders and of this module reject",
        ),
    }[case]
    shard = base / SHARD1
    encoded = (b"\xef\xbb\xbf" if case == "header_bom" else b"") + json.dumps(header).encode()
    shard.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\0" * 4)
    assert mat.main([*args, "--force"]) == 1
    err = capsys.readouterr().err
    assert err.count("REFUSED before anything is published") == 1, err
    assert f"base shard {shard}: unreadable safetensors header ({reason})" in err and "Traceback" not in err, err
    assert len(calls) == 1 and _tree_bytes(out) == before and not os.path.exists(out + ".tmp")
    fresh = str(tmp_path / "fresh")
    assert mat.main([*args[:4], "--out", fresh, *args[6:]]) == 1 and reason in capsys.readouterr().err
    assert len(calls) == 1 and not os.path.exists(fresh) and not os.path.exists(fresh + ".tmp")
    # the helper's contract, then the same shard inside a staging dir: the reuse decision and the output verifier name it
    assert hf_overlay.shard_status(str(shard), {"a.weight"}) == f"{shard}: unreadable safetensors header ({reason})"
    with pytest.raises(ValueError, match=re.escape(reason)):
        hf_overlay.safetensors_header(str(shard))
    staged = tmp_path / "staged.tmp"
    shutil.copytree(out, staged)
    shutil.copy2(shard, staged / SHARD1)
    assert mat.tmp_complete(str(staged)) == (False, f"{staged / SHARD1}: unreadable safetensors header ({reason})")
    with pytest.raises(
        ConversionError, match=re.escape(f"verify: shard listed in {staged / hf_overlay.INDEX_FILE}: {staged / SHARD1}")
    ):
        hf_overlay.verify_index(str(staged), min_tensors=3)
    _fake_safetensors(str(shard), ["a.weight"])  # a valid shard again: the base builds
    assert mat.main([*args[:4], "--out", fresh, *args[6:]]) == 0 and len(calls) == 2


# ---------------------------------------------------------------- staged construction, the pinned reader, the Jinja decode
def _shard_bytes(header: dict, data: bytes) -> bytes:
    """A safetensors file from a hand-written header object and raw data bytes (8-byte length + JSON + data)."""
    encoded = json.dumps(header).encode()
    return struct.pack("<Q", len(encoded)) + encoded + data


# Four probes (a BF16 [3] tensor over four bytes, an unknown dtype, a non-string
# __metadata__ value, an offset gap) plus an offset overlap. Each passes shard_status (the shape, the size and
# the index's keys) and is refused by the pinned reader. (header, data bytes, extra tensor the index maps to SHARD1 or
# None, a fragment of the reader's text.)
READER_REJECTED_SHARDS = {
    "bf16_shape3_four_bytes": (
        {"a.weight": {"dtype": "BF16", "shape": [3], "data_offsets": [0, 4]}, "__metadata__": {"format": "pt"}},
        b"\0" * 4,
        None,
        "invalid shape, data type, or offset",
    ),
    "unknown_dtype": (
        {"a.weight": {"dtype": "FOO", "shape": [2], "data_offsets": [0, 4]}, "__metadata__": {"format": "pt"}},
        b"\0" * 4,
        None,
        "unknown variant",
    ),
    "metadata_value_not_string": (
        {"a.weight": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}, "__metadata__": {"format": 5}},
        b"\0" * 4,
        None,
        "expected a string",
    ),
    "offset_gap": (
        {
            "a.weight": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]},
            "z.extra": {"dtype": "BF16", "shape": [1], "data_offsets": [4, 6]},
        },
        b"\0" * 6,
        "z.extra",
        "invalid offset for tensor",
    ),
    "offset_overlap": (
        {
            "a.weight": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]},
            "z.extra": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]},
        },
        b"\0" * 4,
        "z.extra",
        "invalid offset for tensor",
    ),
}


@pytest.mark.parametrize("case", list(READER_REJECTED_SHARDS))
def test_preflight_refuses_a_base_shard_the_pinned_safetensors_reader_rejects_and_keeps_the_export(
    case: str, tmp_path: Path, base: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``shard_status`` checks the header's SHAPE, the file
    size and the index's keys, and no more: on the parent a base shard whose header declares a BF16 ``[3]`` tensor over
    ``data_offsets [0, 4]`` and four data bytes, an unknown dtype string, a non-string ``__metadata__`` value and a gap (or an
    overlap) between two tensors' offsets all passed the preflight; ``--force --no-verify`` then removed the export and the
    pinned safetensors 0.8.0 parser refused the shard in the overlay. Now the preflight asks that reader itself for every base
    shard the index lists (``conversion.hf.safetensors_reader_error``: ``safe_open(path, framework="np")`` on the header, torch-
    free): refused in the one family naming the shard and the reader's own message, the existing export byte-identical, no
    staging residue (``<out>.tmp``, ``<out>.previous``), the overlay not called; a fresh ``--out`` is refused with nothing
    created; the valid shard builds again. On the parent every case rebuilds the export (exit 0: the fake overlay reads no
    shard)."""
    calls, args, out = _default_export(tmp_path, base, monkeypatch)
    capsys.readouterr()
    before = _tree_bytes(out)
    header, data, extra_tensor, fragment = READER_REJECTED_SHARDS[case]
    shard = base / SHARD1
    keys = {"a.weight"}
    if extra_tensor:
        keys.add(extra_tensor)
        index = {"metadata": {}, "weight_map": {**WMAP, extra_tensor: SHARD1}}
        (base / hf_overlay.INDEX_FILE).write_text(json.dumps(index))
    shard.write_bytes(_shard_bytes(header, data))
    assert hf_overlay.shard_status(str(shard), keys) is None, "the shape check passes: the reader is the only refusal"
    assert mat.main([*args, "--force"]) == 1
    err = capsys.readouterr().err
    assert err.count("REFUSED before anything is published") == 1, err
    assert f"base shard {shard}: refused by the pinned safetensors reader (SafetensorError: " in err, err
    assert fragment in err and "Traceback" not in err, err
    assert len(calls) == 1 and _tree_bytes(out) == before, (
        "the overlay never ran; the existing output is byte-identical"
    )
    assert not os.path.exists(out + ".tmp") and not os.path.exists(out + ".previous"), "no staging residue"
    fresh = str(tmp_path / "fresh")
    assert mat.main([*args[:4], "--out", fresh, *args[6:]]) == 1 and fragment in capsys.readouterr().err
    assert len(calls) == 1 and not os.path.exists(fresh) and not os.path.exists(fresh + ".tmp")
    reason = hf_overlay.safetensors_reader_error(str(shard))
    assert reason is not None and reason.startswith("SafetensorError: ") and fragment in reason, reason
    # the valid shard (and the original index) again: the reader accepts it and the base builds
    (base / hf_overlay.INDEX_FILE).write_text(json.dumps({"metadata": {}, "weight_map": WMAP}))
    _fake_safetensors(str(shard), ["a.weight"])
    assert hf_overlay.safetensors_reader_error(str(shard)) is None
    assert mat.main([*args[:4], "--out", fresh, *args[6:]]) == 0 and len(calls) == 2


@pytest.mark.parametrize("failure", ["overlay", "sidecar_sync", "verify_index"])
def test_a_failure_after_the_preflight_keeps_the_existing_export_and_removes_the_staging_directory(
    failure: str, tmp_path: Path, base: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The class fix: the construction is STAGED. On the parent ``run_request`` removed
    an existing ``--out`` BEFORE the overlay, so any failure the torch-free preflight could not foresee (the overlay on a base
    only its real readers fault, the sidecar sync, the output verification) cost the export under ``--force --no-verify``.
    Now the whole build happens in ``<out>.tmp`` beside the existing output and the swap (``publish_staging``) replaces it
    only after the staged build, its verification and the marker have succeeded: here the overlay is made to raise after a
    partial write, the sidecar sync to raise ``OSError`` after a partial write, and the output verifier to refuse the staged
    copy for real (``--expect-tensors 999``); each time exit 1, the error names the stage, the staging directory is gone and
    the existing export is byte-identical with its marker intact. With the cause removed the same request rebuilds through
    the swap and leaves neither ``<out>.tmp`` nor ``<out>.previous``. On the parent every case ends with the export gone."""
    calls, args, out = _default_export(tmp_path, base, monkeypatch)
    capsys.readouterr()
    before = _tree_bytes(out)
    (tmp_path / "ckpt" / "model.pt").write_bytes(b"\1" * 10)  # another source: the rebuild is real, never a skip
    fake_overlay = hf_overlay.overlay_consolidated_onto_hf  # the fake _default_export installed
    extra: list[str] = []
    if failure == "overlay":

        def failing_overlay(ckpt_dir: str, model_id: str, staging_dir: str, payload: object = None) -> str:
            fake_overlay(ckpt_dir, model_id, staging_dir)
            Path(staging_dir, SHARD1).write_bytes(b"partial")  # a partial write, then the failure
            raise ConversionError("simulated overlay failure after a partial write")

        monkeypatch.setattr(hf_overlay, "overlay_consolidated_onto_hf", failing_overlay)
        stage, detail = "overlay", "ConversionError: simulated overlay failure after a partial write"
    elif failure == "sidecar_sync":

        def failing_sync(base_dir: str, out_dir: str) -> dict[str, list[str]]:
            Path(out_dir, "tokenizer.json").write_text("partial")
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(mat, "sync_sidecars", failing_sync)
        stage, detail = "sidecar sync", "OSError: [Errno 28] No space left on device"
    else:
        extra = ["--expect-tensors", "999"]  # the output verifier refuses the staged copy for real
        stage, detail = "verification (index)", "expected exactly 999"
    assert mat.main([*args, "--force", *extra]) == 1
    err = capsys.readouterr().err
    assert _tree_bytes(out) == before, (
        "the existing output is byte-identical (on the parent: gone, removed before the overlay)"
    )
    assert not os.path.exists(out + ".tmp") and not os.path.exists(out + ".previous"), "no staging residue"
    assert _marker(out)["identity"]["source_sha256"] == SHA_ZERO10
    assert f"FATAL: the staged build failed at the {stage} stage: " in err and detail in err, err
    assert (
        f"the staging directory {out}.tmp was removed; the existing output {out} was not touched by this request" in err
    ), err
    assert "Traceback" not in err and "REFUSED before anything is published" not in err, err
    # the cause removed: the same request rebuilds through the swap
    if failure == "overlay":
        monkeypatch.setattr(hf_overlay, "overlay_consolidated_onto_hf", fake_overlay)
    elif failure == "sidecar_sync":
        monkeypatch.setattr(mat, "sync_sidecars", hf_overlay.sync_sidecars)
    assert mat.main([*args, "--force"]) == 0 and "MATERIALIZE DONE" in capsys.readouterr().out
    assert _marker(out)["identity"]["source_sha256"] == hashlib.sha256(b"\1" * 10).hexdigest()
    assert hf_overlay.verify_sidecars(out, str(base)) == {"sidecars_identical": len(BASE_FILES)}
    assert not os.path.exists(out + ".tmp") and not os.path.exists(out + ".previous")


@pytest.mark.parametrize("case", ["chat_template_jinja", "named_template", "default_named_template"])
def test_preflight_decodes_the_jinja_templates_the_loaders_read_as_strict_utf8_and_keeps_the_export(
    case: str, tmp_path: Path, base: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The preflight's JSON loop skips ``chat_template.jinja``
    and the default-mode identity only hashes it, so on the parent a base ``chat_template.jinja`` (or a named template under
    ``additional_chat_templates/``) holding byte FF passed; ``--force --no-verify`` removed the export and the tokenizer loader
    raised UnicodeDecodeError reading the file. The loaders read these files as UTF-8 TEXT: tokenization_utils_base.py opens
    ``chat_template.jinja`` with ``encoding="utf-8"`` (4.57.6:2188, 5.8.1:1795) and every ``additional_chat_templates/<name>.jinja``
    it discovered (discovery 4.57.6:2047, 5.8.1:1680; the open 4.57.6:2195, 5.8.1:1802), processing_utils.py the same two
    (4.57.6:1070 and 1082, 5.8.1:1081 and 1093) plus the legacy ``chat_template.json`` (4.57.6:1086, 5.8.1:1097, parsed as JSON
    by the preflight already); the container's 5.13.0 the same (tokenization_utils_base.py 1672-1684 discovery, 1799 and 1806
    the opens; processing_utils.py 1261-1274, 1371, 1383, 1387; ``CHAT_TEMPLATE_FILE`` / ``CHAT_TEMPLATE_DIR`` in utils/hub.py:66-67).
    Now the preflight decodes every file ``conversion.hf.loader_template_files`` lists, strictly as
    UTF-8 (``conversion.hf.utf8_text_error``): refused in the one family naming the file and the codec's byte and position,
    the existing export byte-identical, no staging residue, the overlay not called; a fresh ``--out`` refused with nothing
    created. What the loaders accept the preflight accepts: non-ASCII UTF-8 and a byte-order mark (U+FEFF text to them) build
    and are synced verbatim; a file without the ``.jinja`` suffix is not a template either loader discovers. On the parent
    every case rebuilds the export (exit 0)."""
    calls, args, out = _default_export(tmp_path, base, monkeypatch)
    capsys.readouterr()
    before = _tree_bytes(out)
    if case == "chat_template_jinja":
        path = base / "chat_template.jinja"
    else:
        (base / "additional_chat_templates").mkdir()
        name = "named.jinja" if case == "named_template" else "default.jinja"
        path = base / "additional_chat_templates" / name
    valid = b"{{ messages[0].content }}"
    path.write_bytes(valid + b"\xff")
    reason = f"'utf-8' codec can't decode byte 0xff in position {len(valid)}: invalid start byte"
    assert mat.main([*args, "--force"]) == 1, "refused (on the parent: exit 0, the export rebuilt over the removal)"
    err = capsys.readouterr().err
    assert _tree_bytes(out) == before and len(calls) == 1, (
        "the existing output is byte-identical; the overlay never ran"
    )
    assert not os.path.exists(out + ".tmp") and not os.path.exists(out + ".previous"), "no staging residue"
    assert err.count("REFUSED before anything is published") == 1, err
    assert f"{path} is not UTF-8 text ({reason})" in err and "Traceback" not in err, err
    fresh = str(tmp_path / "fresh")
    assert mat.main([*args[:4], "--out", fresh, *args[6:]]) == 1 and reason in capsys.readouterr().err
    assert len(calls) == 1 and not os.path.exists(fresh) and not os.path.exists(fresh + ".tmp")
    # the helpers' contracts: the files the loaders read, and the codec's text
    assert hf_overlay.loader_template_files(str(base)) == [str(path)]
    assert hf_overlay.utf8_text_error(str(path)) == reason
    for accepted in (valid + " ünïcödé ✓ 日本語".encode(), b"\xef\xbb\xbf" + valid):
        path.write_bytes(accepted)
        assert hf_overlay.utf8_text_error(str(path)) is None
    assert mat.main([*args[:4], "--out", fresh, *args[6:]]) == 0 and len(calls) == 2
    assert Path(fresh, path.relative_to(base)).read_bytes() == b"\xef\xbb\xbf" + valid, "synced verbatim"
    if case != "chat_template_jinja":
        (base / "additional_chat_templates" / "README").write_bytes(b"\xff")  # not discovered by either loader
        assert hf_overlay.loader_template_files(str(base)) == [str(path)]


def test_the_swap_publishes_by_rename_and_a_leftover_previous_output_is_refused_without_force(
    tmp_path: Path, base: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The swap's own contract and the policy for its name. ``<out>.previous`` holds the old output between the
    two renames; a leftover under that name (an interrupted swap, or a directory someone parked there) is reported and left on
    the idempotent skip, refused before the build without ``--force`` (nothing built, nothing removed), and discarded under
    ``--force``. ``publish_staging`` itself: ``out`` -> ``<out>.previous`` -> removed, ``tmp`` -> ``out``, no residue; without a
    previous output a plain rename. On the parent the damaged export is rebuilt over the leftover without a word (exit 0)."""
    calls, args, out = _default_export(tmp_path, base, monkeypatch)
    capsys.readouterr()
    previous = out + ".previous"
    os.mkdir(previous)
    Path(previous, "parked").write_text("x")
    assert mat.main(args) == 0 and "idempotent skip" in capsys.readouterr().out, "the proven output still skips"
    assert os.path.isfile(os.path.join(previous, "parked")) and len(calls) == 1
    _truncate(os.path.join(out, SHARD2))  # damaged: a rebuild is due without --force
    before = _tree_bytes(out)
    assert mat.main(args) == 1
    err = capsys.readouterr().err
    assert f"FATAL: REFUSED: {previous} exists" in err and "--force" in err and "Traceback" not in err, err
    assert _tree_bytes(out) == before and os.path.isfile(os.path.join(previous, "parked")) and len(calls) == 1
    assert not os.path.exists(out + ".tmp")
    assert mat.main([*args, "--force"]) == 0 and len(calls) == 2
    assert hf_overlay.shard_status(os.path.join(out, SHARD2), SHARD2_KEYS) is None
    assert not os.path.exists(previous) and not os.path.exists(out + ".tmp")
    assert os.path.isfile(os.path.join(out, mat.MATERIALIZED_MARKER))
    # publish_staging on its own
    staged = tmp_path / "staged.tmp"
    shutil.copytree(out, staged)
    Path(staged, "witness").write_text("new")
    assert mat.publish_staging(str(staged), out) == previous
    assert Path(out, "witness").read_text() == "new" and not os.path.exists(staged) and not os.path.exists(previous)
    fresh = str(tmp_path / "fresh")
    shutil.copytree(out, fresh + ".tmp")
    assert mat.publish_staging(fresh + ".tmp", fresh) is None
    assert os.path.isdir(fresh) and not os.path.exists(fresh + ".tmp") and not os.path.exists(fresh + ".previous")


def test_preflight_refuses_when_the_pinned_safetensors_reader_cannot_be_imported(
    tmp_path: Path, base: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Fail closed: the shard check of the preflight IS the pinned reader; an interpreter without ``safetensors``
    (a declared dependency, ``pyproject.toml``) cannot prove a shard and cannot run the overlay either, so the request is
    refused in the one family before anything is written, the existing export untouched. On the parent the rebuild runs
    (exit 0)."""
    calls, args, out = _default_export(tmp_path, base, monkeypatch)
    capsys.readouterr()
    before = _tree_bytes(out)
    monkeypatch.setitem(sys.modules, "safetensors", None)  # `from safetensors import safe_open` raises ImportError
    assert mat.main([*args, "--force"]) == 1
    err = capsys.readouterr().err
    assert err.count("REFUSED before anything is published") == 1, err
    assert (
        f"base shard {base / SHARD1}: refused by the pinned safetensors reader (the pinned safetensors reader cannot be imported (ModuleNotFoundError: "
        in err
    ), err
    assert len(calls) == 1 and _tree_bytes(out) == before and not os.path.exists(out + ".tmp")


# ---------------------------------------------------------------- the crash state under --force; one materializer per output
@pytest.mark.parametrize(
    "retry",
    ["same_request", "changed_source_transient_failure", "changed_source_success", "foreign_tmp_without_export"],
)
def test_force_after_a_death_between_the_swap_renames_restores_the_old_export_before_building(
    retry: str, tmp_path: Path, base: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The reproduction verbatim:
    build a marked export; a ``--force`` rebuild interrupted after ``out -> out.previous`` and BEFORE ``out.tmp -> out`` (a death:
    ``publish_staging`` replaced by one that does the first rename and raises ``KeyboardInterrupt``, which no cleanup catches);
    retry without ``--force``: refused, ``--force`` suggested; retry with ``--force``. On the parent the retry removed the
    byte-identical marked ``out.previous`` and the verified marked ``out.tmp`` before rebuilding, so a transient overlay failure
    left ``out``, ``out.previous`` and ``out.tmp`` ALL absent. Now ``--force`` FIRST renames ``out.previous`` back to the absent
    ``out`` and builds beside it; a complete marked ``out.tmp`` is never removed while it is the only complete tree, and
    ``--force`` REBUILDS FROM THE SOURCE: once ``out`` is complete again the kept copy is discarded whatever
    its marker says, and a kept copy is published only by a request without ``--force``. ``same_request`` (the reproduction's own
    retry): the overlay runs again beside the restored export, the kept copy discarded, nothing left over (on the parent the kept
    copy was published with the overlay skipped); ``changed_source_transient_failure`` (the retry asks for another checkpoint, the
    overlay raises once): ``out`` back, byte-identical and marked, ``out.tmp`` gone, ``out.previous`` gone (restored); then the
    overlay succeeding -> published, nothing left over; ``changed_source_success``: published, nothing left over, the overlay ran
    beside the restored export;
    ``foreign_tmp_without_export`` (the complete marked ``out.tmp`` is the only tree and the request is for another checkpoint):
    refused by name, nothing touched, then a request of its identity WITHOUT ``--force`` publishes it. On the parent: the
    same-request retry rebuilds after discarding both trees, the failing rebuild leaves no ``out`` (``{} == A``), the successful
    rebuild ran beside nothing, the foreign copy is discarded (exit 0); on the parent the same-request retry publishes the kept
    copy without an overlay call."""
    calls, args, out = _default_export(tmp_path, base, monkeypatch)  # A = ten zero bytes, SHA_ZERO10
    capsys.readouterr()
    export_a = _tree_bytes(out)
    ckpt = tmp_path / "ckpt"
    sha_c = hashlib.sha256(b"\2" * 10).hexdigest()
    seen_at_overlay: list[
        bool
    ] = []  # whether <out> stood while the overlay ran: the build happens beside the restored export
    fake_overlay = hf_overlay.overlay_consolidated_onto_hf

    def observing_overlay(ckpt_dir: str, model_id: str, staging_dir: str, payload: object = None) -> str:
        seen_at_overlay.append(os.path.isdir(out))
        return fake_overlay(ckpt_dir, model_id, staging_dir)

    monkeypatch.setattr(hf_overlay, "overlay_consolidated_onto_hf", observing_overlay)
    # the death between the two renames, through the CLI: a --force rebuild of the same request
    real_publish = mat.publish_staging

    def dying_publish(tmp_out: str, out_dir: str) -> str | None:
        os.rename(out_dir, out_dir + ".previous")
        raise KeyboardInterrupt("simulated death between the swap's two renames")

    monkeypatch.setattr(mat, "publish_staging", dying_publish)
    with pytest.raises(KeyboardInterrupt):
        mat.main([*args, "--force"])
    monkeypatch.setattr(mat, "publish_staging", real_publish)
    capsys.readouterr()
    assert len(calls) == 2 and seen_at_overlay == [True] and not os.path.exists(out)
    assert _tree_bytes(out + ".previous") == export_a, "the crash state: the old export under <out>.previous"
    want = {"conversion": mat.CONVERSION_VERSION, "base_digest": mat.base_identity(str(base))["digest"]}
    assert mat.marker_status(out + ".tmp", want, {"sha256": SHA_ZERO10, "bytes": 10})[0] == "match", "<out>.tmp: marked"
    staged = _tree_bytes(out + ".tmp")
    refusal = ""
    if retry == "foreign_tmp_without_export":
        shutil.rmtree(out + ".previous")  # the complete marked copy under <out>.tmp is the only tree
    else:
        # the retry without --force: refused by name, --force suggested, nothing touched
        assert mat.main(args) == 1
        refusal = capsys.readouterr().err
        assert (
            f"FATAL: REFUSED: {out}.previous exists" in refusal and "--force" in refusal and "Traceback" not in refusal
        )
        assert (
            not os.path.exists(out)
            and _tree_bytes(out + ".previous") == export_a
            and _tree_bytes(out + ".tmp") == staged
        )
    if retry == "same_request":
        # --force rebuilds from the source; the kept copy is discarded once <out> is complete again
        assert mat.main([*args, "--force"]) == 0 and len(calls) == 3, (
            "the overlay ran again (on the parent: the kept copy published)"
        )
        assert seen_at_overlay == [True, True], "the rebuild ran beside the restored export"
        marker = _marker(out)
        assert marker["identity"]["source_sha256"] == SHA_ZERO10 and marker["reused_staging"] is False
        assert not os.path.exists(out + ".previous") and not os.path.exists(out + ".tmp"), "nothing left over"
        assert mat.complete_marked(out) == (True, marker["identity_digest"][:12])
        assert "--force renames it back to" in refusal and "builds beside it" in refusal, refusal
        return
    (ckpt / "model.pt").write_bytes(b"\2" * 10)  # the retry asks for another checkpoint, C
    if retry == "foreign_tmp_without_export":
        assert mat.main([*args, "--force"]) == 1
        err = capsys.readouterr().err
        assert f"FATAL: REFUSED: {out}.tmp is a complete marked export (identity " in err, err
        assert "--force does not discard the only complete tree" in err and "Traceback" not in err and len(calls) == 2
        assert _tree_bytes(out + ".tmp") == staged and not os.path.exists(out), "nothing touched"
        (ckpt / "model.pt").write_bytes(b"\0" * 10)  # a request of its identity WITHOUT --force publishes it
        assert mat.main(args) == 0 and len(calls) == 2
        assert _marker(out)["identity"]["source_sha256"] == SHA_ZERO10 and _marker(out)["reused_staging"] is True
        assert not os.path.exists(out + ".tmp") and not os.path.exists(out + ".previous")
        return
    if retry == "changed_source_transient_failure":

        def failing_once(ckpt_dir: str, model_id: str, staging_dir: str, payload: object = None) -> str:
            observing_overlay(ckpt_dir, model_id, staging_dir)
            raise ConversionError("simulated transient overlay failure")

        monkeypatch.setattr(hf_overlay, "overlay_consolidated_onto_hf", failing_once)
        assert mat.main([*args, "--force"]) == 1
        err = capsys.readouterr().err
        assert _tree_bytes(out) == export_a, "<out> back, byte-identical and marked (on the parent: absent)"
        assert not os.path.exists(out + ".tmp") and not os.path.exists(out + ".previous"), (
            "<out>.tmp gone, <out>.previous gone"
        )
        assert seen_at_overlay == [True, True], "the rebuild ran beside the restored export"
        assert "FATAL: the staged build failed at the overlay stage" in err and "transient overlay failure" in err, err
        assert f"the existing output {out} was not touched by this request" in err and "Traceback" not in err, err
        monkeypatch.setattr(hf_overlay, "overlay_consolidated_onto_hf", observing_overlay)
    assert mat.main([*args, "--force"]) == 0, "the overlay succeeding: published"
    assert seen_at_overlay[-1] is True, "the rebuild ran beside the restored export (on the parent: beside nothing)"
    assert _marker(out)["identity"]["source_sha256"] == sha_c and _marker(out)["reused_staging"] is False
    assert not os.path.exists(out + ".previous") and not os.path.exists(out + ".tmp"), "nothing left over"
    assert "--force renames it back to" in refusal and "builds beside it" in refusal, refusal


_LOCK_DRIVER = r"""
import os, sys, time
from pathlib import Path
from cosmos_framework.posttrain.checkpoint.conversion import hf
from cosmos_framework.posttrain.checkpoint.tools import materialize_hf as mat
from cosmos_framework.posttrain.tests import test_checkpoint_materialize_hf as fixtures
mode, ckpt, base, out, gate = sys.argv[1:6]


def fake_overlay(ckpt_dir, model_id, staging_dir, payload=None):
    import shutil
    os.makedirs(staging_dir)
    shutil.copy2(os.path.join(model_id, "config.json"), os.path.join(staging_dir, "config.json"))
    shutil.copy2(os.path.join(model_id, hf.INDEX_FILE), os.path.join(staging_dir, hf.INDEX_FILE))
    fixtures._fake_shards(staging_dir, fixtures.WMAP)
    if mode == "wait":  # hold the lock inside the overlay until the test says go
        Path(gate, "in_overlay").write_text(str(os.getpid()))
        deadline = time.time() + 120
        while not os.path.exists(os.path.join(gate, "go")):
            if time.time() > deadline:
                raise RuntimeError("gate timeout")
            time.sleep(0.05)
    return staging_dir


hf.overlay_consolidated_onto_hf = fake_overlay
sys.exit(mat.main(["--ckpt-dir", ckpt, "--base", base, "--out", out, "--min-tensors", "3", "--no-verify", *sys.argv[6:]]))
"""


def test_two_materializers_on_one_output_exclude_each_other_through_the_per_output_lock(
    tmp_path: Path, base: Path
) -> None:
    """Two real subprocesses on one ``--out``: the first is held inside the overlay (a fake overlay
    that waits for a file) with ``<out>.lock`` taken and its staging directory half written; the second, with ``--force``, is refused
    by name (the holder's pid, host and start time from the lock record), touches nothing (the first's staging directory is
    byte-identical, no ``<out>``); released, the first publishes. The lock file stays beside the output and blocks nobody once its
    holder is gone: a third request skips idempotently. On the parent the second removes the first's staging directory as "not
    reusable" and publishes its own build (exit 0; timing 1)."""
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "model.pt").write_bytes(b"\0" * 10)
    out = str(tmp_path / "out")
    gate = tmp_path / "gate"
    gate.mkdir()
    roots = [p for p in sys.path if p] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(dict.fromkeys(roots)), "PYTHONDONTWRITEBYTECODE": "1"}

    def command(mode: str, *extra: str) -> list[str]:
        return [sys.executable, "-c", _LOCK_DRIVER, mode, str(ckpt), str(base), out, str(gate), *extra]

    first = subprocess.Popen(command("wait"), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.time() + 120
        while not (gate / "in_overlay").exists():
            assert first.poll() is None, first.communicate()
            assert time.time() < deadline, "the first materializer never reached the overlay"
            time.sleep(0.05)
        assert os.path.isdir(out + ".tmp") and not os.path.exists(out), "the first is inside the overlay"
        staging = _tree_bytes(out + ".tmp")
        second = subprocess.run(
            command("now", "--force", "--lock-wait-seconds", "1"), env=env, capture_output=True, text=True, timeout=120
        )
        assert second.returncode == 1, f"the second must be refused (on the parent it publishes): {second.stderr}"
        assert os.path.isfile(out + ".lock"), "the lock file stands beside the output"
        expected = (
            f"FATAL: REFUSED: {out} is being materialized by another process ({out}.lock held by pid {first.pid} on host "
            f"{socket.gethostname()} since "
        )
        assert expected in second.stderr and "nothing was touched" in second.stderr, second.stderr
        assert "Traceback" not in second.stderr and f"for source {ckpt}" in second.stderr
        assert _tree_bytes(out + ".tmp") == staging and not os.path.exists(out), "the loser touched nothing"
    finally:
        (gate / "go").write_text("go")
        first_out, first_err = first.communicate(timeout=120)
    assert first.returncode == 0, first_err
    assert "MATERIALIZE DONE" in first_out and os.path.isfile(os.path.join(out, mat.MATERIALIZED_MARKER))
    assert not os.path.exists(out + ".tmp") and hf_overlay.verify_index(out, min_tensors=3)["n_tensors"] == 3
    assert os.path.isfile(out + ".lock"), "the lock file stays beside the output"
    third = subprocess.run(command("now"), env=env, capture_output=True, text=True, timeout=120)
    assert third.returncode == 0 and "idempotent skip" in third.stdout, third.stderr


def test_the_per_output_lock_refuses_by_name_in_process_and_fails_closed_without_flock(
    tmp_path: Path, base: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The lock's own contract: a second ``output_lock`` on the same output (another file
    description, the same process here) is refused naming the holder's record; the CLI under ``--force`` while the lock is held is
    refused by name with the existing export byte-identical and no staging directory; released, the same request builds; where
    flock is refused by the file system (``ENOLCK``: Lustre without the ``flock`` mount option) the request is refused by name,
    fail closed. On the parent ``output_lock`` does not exist and the CLI rebuilds under a held lock."""
    calls, args, out = _default_export(tmp_path, base, monkeypatch)
    capsys.readouterr()
    before = _tree_bytes(out)
    (tmp_path / "ckpt" / "model.pt").write_bytes(b"\1" * 10)  # a rebuild is due
    with mat.output_lock(out, source="held-by-the-test") as lock_path:
        record = json.loads(Path(lock_path).read_text())
        assert lock_path == out + ".lock" and record["pid"] == os.getpid() and record["source"] == "held-by-the-test"
        holder = f"{out}.lock held by pid {os.getpid()} on host {socket.gethostname()} since {record['started']}"
        with pytest.raises(mat.MaterializeRefusedError, match=re.escape(holder)), mat.output_lock(out, wait_seconds=0):
            pass
        assert mat.main([*args, "--force", "--lock-wait-seconds", "0"]) == 1
        err = capsys.readouterr().err
        assert f"FATAL: REFUSED: {out} is being materialized by another process ({holder}" in err, err
        assert "nothing was touched" in err and "Traceback" not in err, err
        assert _tree_bytes(out) == before and not os.path.exists(out + ".tmp") and len(calls) == 1
    assert mat.main([*args, "--force"]) == 0 and len(calls) == 2, "released: the same request builds"
    assert _marker(out)["identity"]["source_sha256"] == hashlib.sha256(b"\1" * 10).hexdigest()

    def no_flock(fd: int, operation: int) -> None:
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(mat.fcntl, "flock", no_flock)
    (tmp_path / "ckpt" / "model.pt").write_bytes(b"\2" * 10)
    before = _tree_bytes(out)
    assert mat.main([*args, "--force"]) == 1
    err = capsys.readouterr().err
    assert f"FATAL: REFUSED: cannot take the per-output lock {out}.lock (OSError: [Errno {errno.ENOLCK}]" in err, err
    assert "flock" in err and "nothing was touched" in err and "Traceback" not in err, err
    assert _tree_bytes(out) == before and not os.path.exists(out + ".tmp") and len(calls) == 2


# ---------------------------------------------------------------- --force rebuilds from the source
@pytest.mark.parametrize("case", ["flipped_byte_no_verify", "base_bytes_changed_same_size"])
def test_force_rebuilds_from_the_source_instead_of_publishing_a_death_left_copy(
    case: str, tmp_path: Path, base: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Both reported cases through the CLI. On the parent a death-left ``<out>.tmp`` whose
    marker matched the request was published under ``--force`` without a fresh overlay, trusting its shard bytes on size and tensor
    count and a base digest that excludes the base shards' bytes: a kept copy with one flipped data byte was published under
    ``--force --no-verify`` (B1), and after a base shard's bytes changed with the size kept the stale copy was published (B2). Now
    ``--force`` rebuilds from the source once ``<out>`` is complete again: the crash state (the death between the swap's renames, as
    in the restore fixture), then the kept copy's last data byte flipped (B1) or the base shard's last data byte flipped with the
    base digest unchanged (B2); ``--force`` restores ``<out>.previous``, discards the kept copy, runs the overlay (which copies the
    base's shards here, as the real overlay carries the base's untouched tensors) and publishes a shard equal to a fresh overlay's
    bytes: ``reused_staging`` False, one more overlay call, no residue. On the parent: the kept copy published with the flipped
    byte / the stale bytes, no overlay call."""
    calls, args, out = _default_export(tmp_path, base, monkeypatch)
    capsys.readouterr()
    fake_overlay = hf_overlay.overlay_consolidated_onto_hf

    def base_copying_overlay(ckpt_dir: str, model_id: str, staging_dir: str, payload: object = None) -> str:
        fake_overlay(ckpt_dir, model_id, staging_dir)  # the fake build, its shards then replaced by the base's bytes
        for shard in sorted(set(WMAP.values())):
            shutil.copy2(os.path.join(model_id, shard), os.path.join(staging_dir, shard))
        return staging_dir

    monkeypatch.setattr(hf_overlay, "overlay_consolidated_onto_hf", base_copying_overlay)
    real_publish = mat.publish_staging

    def dying_publish(tmp_out: str, out_dir: str) -> str | None:
        os.rename(out_dir, out_dir + ".previous")
        raise KeyboardInterrupt("simulated death between the swap's two renames")

    monkeypatch.setattr(mat, "publish_staging", dying_publish)
    with pytest.raises(KeyboardInterrupt):
        mat.main([*args, "--force"])
    monkeypatch.setattr(mat, "publish_staging", real_publish)
    capsys.readouterr()
    assert len(calls) == 2 and not os.path.exists(out) and mat.complete_marked(out + ".tmp")[0]
    kept_shard, base_shard = Path(out + ".tmp", SHARD1), base / SHARD1
    assert kept_shard.read_bytes() == base_shard.read_bytes()
    digest_before = mat.base_identity(str(base))["digest"]
    stale = kept_shard.read_bytes()
    target = kept_shard if case == "flipped_byte_no_verify" else base_shard
    data = bytearray(target.read_bytes())
    data[-1] ^= 0xFF  # the last data byte: the header, the size and the tensor count unchanged
    target.write_bytes(bytes(data))
    expected = base_shard.read_bytes()  # what a fresh overlay publishes: the base's bytes
    if case == "base_bytes_changed_same_size":
        assert mat.base_identity(str(base))["digest"] == digest_before, "the base digest excludes the shards' bytes"
        assert expected != stale
    else:
        assert kept_shard.read_bytes() != expected and "--no-verify" in args
    assert mat.main([*args, "--force"]) == 0 and len(calls) == 3, (
        "--force rebuilt (on the parent: the kept copy published)"
    )
    marker = _marker(out)
    assert marker["reused_staging"] is False and marker["identity"]["base_digest"] == digest_before
    assert Path(out, SHARD1).read_bytes() == expected, "the published shard equals a fresh overlay's bytes"
    assert not os.path.exists(out + ".tmp") and not os.path.exists(out + ".previous")


def test_a_death_left_copy_is_published_only_by_a_request_without_force_after_verification(
    tmp_path: Path, base: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The pinned counterpart: the kept-copy publication without a fresh overlay remains for a request
    WITHOUT ``--force`` (the idempotent recovery). A first build that dies right before its single rename leaves ``<out>.tmp``
    complete and marked and no ``<out>``: ``--force`` is refused by name (the copy is the only complete tree, which ``--force`` does
    not discard; a request without ``--force`` is the way out) and touches nothing; without ``--force`` the copy is published after
    the verification a fresh build gets (index, sidecars; the torch probe unless ``--no-verify``, which trusts its bytes on size and
    tensor count): ``reused_staging`` True, the overlay not called, the published shards the kept copy's bytes, no residue. On
    the parent ``--force`` published the kept copy (its marker matched the request)."""
    calls = _install_fake_overlay(monkeypatch)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "model.pt").write_bytes(b"\0" * 10)
    out = str(tmp_path / "out")
    args = ["--ckpt-dir", str(ckpt), "--base", str(base), "--out", out, "--min-tensors", "3", "--no-verify"]
    real_publish = mat.publish_staging

    def dying_publish(tmp_out: str, out_dir: str) -> str | None:
        raise KeyboardInterrupt("simulated death right before the rename")

    monkeypatch.setattr(mat, "publish_staging", dying_publish)
    with pytest.raises(KeyboardInterrupt):
        mat.main(args)
    monkeypatch.setattr(mat, "publish_staging", real_publish)
    capsys.readouterr()
    assert len(calls) == 1 and not os.path.exists(out) and mat.complete_marked(out + ".tmp")[0]
    kept = _tree_bytes(out + ".tmp")
    assert mat.main([*args, "--force"]) == 1, (
        "the only complete tree is not discarded by --force (on the parent: published)"
    )
    err = capsys.readouterr().err
    assert f"FATAL: REFUSED: {out}.tmp is a complete marked export (identity " in err, err
    assert "--force does not discard the only complete tree" in err and "Traceback" not in err, err
    assert "a request without --force publishes it after verification" in err, err
    assert _tree_bytes(out + ".tmp") == kept and not os.path.exists(out) and len(calls) == 1, "nothing touched"
    assert mat.main(args) == 0 and len(calls) == 1, (
        "without --force: published after verification, the overlay not called"
    )
    marker = _marker(out)
    assert marker["reused_staging"] is True and marker["identity"]["source_sha256"] == SHA_ZERO10
    assert marker["options"]["no_verify"] is True
    shards = {k: v for k, v in _tree_bytes(out).items() if k.endswith(".safetensors")}
    assert shards == {k: v for k, v in kept.items() if k.endswith(".safetensors")}, "the kept copy's bytes, published"
    assert not os.path.exists(out + ".tmp") and not os.path.exists(out + ".previous")


# ---------------------------------------------------------------- the inherited sole marked copy; the decision under the lock
@pytest.mark.parametrize(
    "fault",
    [
        "index_verification_oserror",
        "stricter_tensor_count",
        "torch_verification_error",
        "config_missing_then_overlay_failure",
    ],
)
def test_a_failed_recovery_check_leaves_the_inherited_sole_marked_copy_as_found(
    fault: str, tmp_path: Path, base: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The reproduction verbatim: a first publication dies
    after writing ``.materialized`` and before ``tmp -> out`` (``publish_staging`` replaced by one that raises ``KeyboardInterrupt``
    before any rename), so ``<out>.tmp`` is the only completed copy; the recommended retry without ``--force`` reuses it, and on
    the parent a failed recovery check (an OSError injected into the index verification; a stricter tensor count, no injection; a
    transient torch-verification error) reached ``_discard_staging`` and left ``out``, ``out.tmp`` and ``out.previous`` ALL absent,
    while a missing staged ``config.json`` made the remedy discard the marked shards before rebuilding, so a transient overlay
    failure lost them too. Now the inherited sole complete marked copy is never deleted: the failed check refuses by name and the
    tree is left exactly as found (tree hash equal; no ``out``, no ``out.previous``; the overlay not run), and the retry with the
    cause gone publishes it with the shards untouched; a missing ``config.json`` is repaired from the base through a temporary file
    and a rename, the overlay never called (its injected failure cannot bite). On the parent: everything absent after the failed
    check (``{} == found``), and the config case ends with exit 1 and the marked shards gone."""
    calls = _install_fake_overlay(monkeypatch)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "model.pt").write_bytes(b"\0" * 10)
    out = str(tmp_path / "out")
    args = ["--ckpt-dir", str(ckpt), "--base", str(base), "--out", out, "--min-tensors", "3"]
    if fault == "torch_verification_error":
        monkeypatch.setattr(
            mat, "verify_with_torch", lambda *a, **k: {"torch_probe": "stand-in"}
        )  # torch-free stand-in
    else:
        args.append("--no-verify")
    real_publish = mat.publish_staging

    def dying_publish(tmp_out: str, out_dir: str) -> str | None:
        raise KeyboardInterrupt("simulated death right before the publication rename")

    monkeypatch.setattr(mat, "publish_staging", dying_publish)
    with pytest.raises(KeyboardInterrupt):
        mat.main(args)
    monkeypatch.setattr(mat, "publish_staging", real_publish)
    capsys.readouterr()
    tmp = out + ".tmp"
    assert len(calls) == 1 and not os.path.exists(out) and mat.complete_marked(tmp)[0], "the sole completed copy"
    if fault == "config_missing_then_overlay_failure":
        os.remove(os.path.join(tmp, "config.json"))
    found = _tree_bytes(tmp)
    shards_found = {k: v for k, v in found.items() if k.endswith(".safetensors")}
    extra: list[str] = []
    armed = {"fault": True}
    if fault == "index_verification_oserror":
        real_verify_index = mat.verify_index

        def faulty_verify_index(*a: object, **k: object) -> dict[str, int]:
            if armed["fault"]:
                armed["fault"] = False
                raise OSError(5, "Input/output error")
            return real_verify_index(*a, **k)  # type: ignore[arg-type]

        monkeypatch.setattr(mat, "verify_index", faulty_verify_index)
        stage, detail = "verification (index)", "OSError: [Errno 5] Input/output error"
    elif fault == "stricter_tensor_count":
        extra = ["--expect-tensors", "999"]
        stage, detail = "verification (index)", "expected exactly 999"
    elif fault == "torch_verification_error":

        def faulty_torch(*a: object, **k: object) -> dict[str, str]:
            if armed["fault"]:
                armed["fault"] = False
                raise RuntimeError("transient torch verification error")
            return {"torch_probe": "stand-in"}

        monkeypatch.setattr(mat, "verify_with_torch", faulty_torch)
        stage, detail = "verification (torch)", "RuntimeError: transient torch verification error"
    else:

        def failing_overlay(ckpt_dir: str, model_id: str, staging_dir: str, payload: object = None) -> str:
            calls.append(ckpt_dir)
            raise ConversionError("simulated transient overlay failure")

        monkeypatch.setattr(hf_overlay, "overlay_consolidated_onto_hf", failing_overlay)
        stage = detail = ""
    if fault != "config_missing_then_overlay_failure":
        assert mat.main([*args, *extra]) == 1, "the failed recovery check refuses"
        err = capsys.readouterr().err
        assert _tree_bytes(tmp) == found, "the inherited copy is left exactly as found (on the parent: deleted)"
        assert not os.path.exists(out) and not os.path.exists(out + ".previous")
        assert f"FATAL: the recovery of the inherited staging directory {tmp} failed at the {stage} stage: " in err, err
        assert detail in err, err
        assert (
            "the directory is kept: its shards and its marker are untouched" in err
            and "retry without --force" in err
            and "Traceback" not in err
        ), err
        assert len(calls) == 1, "the overlay was not run while the copy is the sole complete tree"
    # the cause gone (the config case: no fault to remove, the repair is the remedy): the unforced retry publishes the copy
    assert mat.main(args) == 0 and len(calls) == 1, "published without an overlay call (on the parent: rebuilt or lost)"
    marker = _marker(out)
    assert marker["reused_staging"] is True and marker["identity"]["source_sha256"] == SHA_ZERO10
    published = _tree_bytes(out)
    assert {k: v for k, v in published.items() if k.endswith(".safetensors")} == shards_found, "the shards untouched"
    assert published["config.json"] == (base / "config.json").read_bytes()
    assert not os.path.exists(tmp) and not os.path.exists(out + ".previous")


_RACE_DRIVER = r"""
import contextlib, os, sys, time
from pathlib import Path
from cosmos_framework.posttrain.checkpoint.conversion import hf
from cosmos_framework.posttrain.checkpoint.tools import materialize_hf as mat
from cosmos_framework.posttrain.tests import test_checkpoint_materialize_hf as fixtures
mode, ckpt, base, out, gate = sys.argv[1:6]


def fake_overlay(ckpt_dir, model_id, staging_dir, payload=None):
    import shutil
    os.makedirs(staging_dir)
    shutil.copy2(os.path.join(model_id, "config.json"), os.path.join(staging_dir, "config.json"))
    shutil.copy2(os.path.join(model_id, hf.INDEX_FILE), os.path.join(staging_dir, hf.INDEX_FILE))
    fixtures._fake_shards(staging_dir, fixtures.WMAP)
    return staging_dir


hf.overlay_consolidated_onto_hf = fake_overlay
real_lock = mat.output_lock


@contextlib.contextmanager
def pausing_lock(out_dir, *args, **kwargs):
    if mode == "pause_before_lock":  # A pauses right before acquiring the lock, until the test says go
        Path(gate, "before_lock").write_text(str(os.getpid()))
        deadline = time.time() + 120
        while not os.path.exists(os.path.join(gate, "go")):
            if time.time() > deadline:
                raise RuntimeError("gate timeout")
            time.sleep(0.05)
    with real_lock(out_dir, *args, **kwargs) as lock_path:
        yield lock_path


mat.output_lock = pausing_lock
sys.exit(mat.main(["--ckpt-dir", ckpt, "--base", base, "--out", out, "--min-tensors", "3", "--no-verify", *sys.argv[6:]]))
"""


def test_the_marker_decision_is_taken_under_the_output_lock_so_a_paused_materializer_cannot_overwrite_a_newer_export(
    tmp_path: Path, base: Path
) -> None:
    """The race with two real subprocesses: A (source
    A) pauses right before acquiring the output lock; B (source B) publishes and releases; A resumes. On the parent A had evaluated
    ``marker_status`` BEFORE the lock ("absent"), so it acquired the free lock and overwrote B with that stale decision without
    ``--force`` (B's marked identity replaced, its previous copy removed). Now the marker / skip / mismatch decision is taken under
    the lock: A sees B's marker and is refused by name ("holds a different or unproven materialization", the source sha256 named),
    B's export byte-identical, no residue."""
    ckpt_a, ckpt_b = tmp_path / "ckpt_a", tmp_path / "ckpt_b"
    for ckpt, byte in ((ckpt_a, b"\0"), (ckpt_b, b"\1")):
        ckpt.mkdir()
        (ckpt / "model.pt").write_bytes(byte * 10)
    out = str(tmp_path / "out")
    gate = tmp_path / "gate"
    gate.mkdir()
    roots = [p for p in sys.path if p] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(dict.fromkeys(roots)), "PYTHONDONTWRITEBYTECODE": "1"}

    def command(mode: str, ckpt: Path, *extra: str) -> list[str]:
        return [sys.executable, "-c", _RACE_DRIVER, mode, str(ckpt), str(base), out, str(gate), *extra]

    first = subprocess.Popen(
        command("pause_before_lock", ckpt_a), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        deadline = time.time() + 120
        while not (gate / "before_lock").exists():
            assert first.poll() is None, first.communicate()
            assert time.time() < deadline, "A never reached the lock"
            time.sleep(0.05)
        assert not os.path.exists(out) and not os.path.exists(out + ".tmp"), "A has touched nothing yet"
        second = subprocess.run(command("now", ckpt_b), env=env, capture_output=True, text=True, timeout=120)
        assert second.returncode == 0 and "MATERIALIZE DONE" in second.stdout, second.stderr
        export_b = _tree_bytes(out)
        assert _marker(out)["identity"]["source_sha256"] == hashlib.sha256(b"\1" * 10).hexdigest()
    finally:
        (gate / "go").write_text("go")
        first_out, first_err = first.communicate(timeout=120)
    assert first.returncode == 1, f"A must be refused on B's marker (on the parent A overwrote B): {first_err}"
    assert f"FATAL: REFUSED: {out} holds a different or unproven materialization" in first_err, first_err
    assert "source sha256" in first_err and "Traceback" not in first_err, first_err
    assert _tree_bytes(out) == export_b, "B's export is intact"
    assert not os.path.exists(out + ".tmp") and not os.path.exists(out + ".previous")


# ---------------------------------------------------------------- the truthful recovery refusal; the bounded lock wait
@pytest.mark.parametrize("out_state", ["out_absent", "out_unmarked"])
def test_a_repair_then_failed_recovery_check_keeps_the_shards_and_the_marker_and_says_what_it_changed(
    out_state: str, tmp_path: Path, base: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The recovery of an inherited sole complete marked ``<out>.tmp`` repairs and re-syncs BEFORE it
    checks, so on the parent its refusal said "left as found" while config.json had been rewritten from the base, a differing
    sidecar re-synced and a stray one removed, and said "no output exists at <out>" while an unmarked directory stood there. Now the
    refusal says what is true: the shards and the marker are untouched, the sidecars may have been repaired from the base before the
    check, and the state at ``<out>`` is named. Here: the first publication dies before its rename; the staged config.json removed,
    generation_config.json made to differ from the base's, a stray processor_config.json added (and, for ``out_unmarked``, an
    unmarked directory put at ``<out>``); the unforced retry with ``--expect-tensors 999`` fails its index check after the repair:
    the shards, the index, the stamp and the marker byte-identical, config.json = the base's, generation_config.json = the base's,
    the stray gone, the refusal naming the repair and the state at ``<out>``; the retry without the stricter count publishes the
    repaired copy (the unmarked directory replaced by the swap). On the parent the invariant holds but the words do not ("left as
    found", "no output exists" beside the standing directory)."""
    calls = _install_fake_overlay(monkeypatch)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "model.pt").write_bytes(b"\0" * 10)
    out = str(tmp_path / "out")
    args = ["--ckpt-dir", str(ckpt), "--base", str(base), "--out", out, "--min-tensors", "3", "--no-verify"]
    real_publish = mat.publish_staging

    def dying_publish(tmp_out: str, out_dir: str) -> str | None:
        raise KeyboardInterrupt("simulated death right before the publication rename")

    monkeypatch.setattr(mat, "publish_staging", dying_publish)
    with pytest.raises(KeyboardInterrupt):
        mat.main(args)
    monkeypatch.setattr(mat, "publish_staging", real_publish)
    capsys.readouterr()
    tmp = out + ".tmp"
    os.remove(os.path.join(tmp, "config.json"))  # repaired from the base before the check
    Path(tmp, "generation_config.json").write_text(
        '{"top_p": 0.1}'
    )  # differs from the base: re-synced before the check
    Path(tmp, "processor_config.json").write_text('{"stray": true}')  # the base lacks it: removed before the check
    if out_state == "out_unmarked":
        os.mkdir(out)
        Path(out, "junk").write_text("x")
    found = _tree_bytes(tmp)
    kept_names = (mat.MATERIALIZED_MARKER, mat.SOURCE_STAMP, hf_overlay.INDEX_FILE)
    invariant = {k: v for k, v in found.items() if k.endswith(".safetensors") or k in kept_names}
    assert mat.complete_marked(tmp)[0] and len(invariant) == 5
    assert mat.main([*args, "--expect-tensors", "999"]) == 1
    err = capsys.readouterr().err
    after = _tree_bytes(tmp)
    assert {k: v for k, v in after.items() if k in invariant} == invariant, "shards, index, stamp and marker untouched"
    assert after["config.json"] == (base / "config.json").read_bytes(), (
        "config.json repaired from the base before the check"
    )
    assert after["generation_config.json"] == (base / "generation_config.json").read_bytes(), (
        "re-synced before the check"
    )
    assert "processor_config.json" not in after, "the stray sidecar removed before the check"
    assert len(calls) == 1, "the overlay never ran"
    assert (
        "the directory is kept: its shards and its marker are untouched, its sidecars may have been repaired from the base before "
        "the check (config.json rewritten from the base, a differing one kept as config.json.trainer.bak, a sidecar the base lacks "
        "removed, a .copying or .writing temporary left by an interrupted copy or unified-template rewrite, and under "
        "--unified-chat-template the two template files back in the base's form when that rewrite did not reach its rename)"
    ) in err, err
    assert "left as found" not in err and "expected exactly 999" in err and "Traceback" not in err, err
    if out_state == "out_absent":
        assert f"no output exists at {out}" in err and not os.path.exists(out), err
    else:
        assert f"{out} stands but is not a complete marked export (" in err and os.path.isfile(
            os.path.join(out, "junk")
        ), err
    assert "nothing was published" in err
    assert mat.main(args) == 0 and len(calls) == 1, "the retry publishes the repaired copy without an overlay call"
    published = _tree_bytes(out)
    shards = {k: v for k, v in invariant.items() if k.endswith(".safetensors")}
    assert {k: v for k, v in published.items() if k.endswith(".safetensors")} == shards
    assert published["config.json"] == (base / "config.json").read_bytes() and "junk" not in published
    assert not os.path.exists(tmp) and not os.path.exists(out + ".previous")


_WAIT_DRIVER = r"""
import contextlib, os, sys, time
from pathlib import Path
from cosmos_framework.posttrain.checkpoint.conversion import hf
from cosmos_framework.posttrain.checkpoint.tools import materialize_hf as mat
from cosmos_framework.posttrain.tests import test_checkpoint_materialize_hf as fixtures
mode, ckpt, base, out, gate = sys.argv[1:6]


def fake_overlay(ckpt_dir, model_id, staging_dir, payload=None):
    import shutil
    os.makedirs(staging_dir)
    shutil.copy2(os.path.join(model_id, "config.json"), os.path.join(staging_dir, "config.json"))
    shutil.copy2(os.path.join(model_id, hf.INDEX_FILE), os.path.join(staging_dir, hf.INDEX_FILE))
    fixtures._fake_shards(staging_dir, fixtures.WMAP)
    return staging_dir


hf.overlay_consolidated_onto_hf = fake_overlay
real_lock = mat.output_lock


@contextlib.contextmanager
def holding_lock(out_dir, *args, **kwargs):
    with real_lock(out_dir, *args, **kwargs) as lock_path:
        if mode == "hold_until_go":  # hold the acquired lock until the test says go
            Path(gate, "holding").write_text(str(os.getpid()))
            deadline = time.time() + 120
            while not os.path.exists(os.path.join(gate, "go")):
                if time.time() > deadline:
                    raise RuntimeError("gate timeout")
                time.sleep(0.05)
        yield lock_path


mat.output_lock = holding_lock
sys.exit(mat.main(["--ckpt-dir", ckpt, "--base", base, "--out", out, "--min-tensors", "3", "--no-verify", *sys.argv[6:]]))
"""


@pytest.mark.parametrize("second", ["identical_skips_with_0", "differing_refused_on_the_marker"])
def test_an_identical_request_waits_for_the_lock_holder_and_skips_with_0(
    second: str, tmp_path: Path, base: Path
) -> None:
    """The idempotent skip's re-check holds the output lock, so on the parent the
    second of two simultaneous identical requests of an already-published source exited 1 ("being materialized by another
    process") where the parent returned 0 for both: an undocumented change of the idempotent contract. Now a held lock is waited
    for, polling, up to 120 s (``--lock-wait-seconds``): two real subprocesses on a published export, the holder an identical request
    that keeps the acquired lock until the test says go; the second request is still running after 1.5 s (it waits), and once the
    holder has finished it acquires and decides under the lock: an identical request skips with 0, a differing one is refused on the
    marker ("holds a different or unproven materialization"), never with the "being materialized" text; the export byte-identical.
    On the parent the second exits 1 at once."""
    ckpt_a, ckpt_b = tmp_path / "ckpt_a", tmp_path / "ckpt_b"
    for ckpt, byte in ((ckpt_a, b"\0"), (ckpt_b, b"\1")):
        ckpt.mkdir()
        (ckpt / "model.pt").write_bytes(byte * 10)
    out = str(tmp_path / "out")
    gate = tmp_path / "gate"
    gate.mkdir()
    roots = [p for p in sys.path if p] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(dict.fromkeys(roots)), "PYTHONDONTWRITEBYTECODE": "1"}

    def command(mode: str, ckpt: Path, *extra: str) -> list[str]:
        return [sys.executable, "-c", _WAIT_DRIVER, mode, str(ckpt), str(base), out, str(gate), *extra]

    first = subprocess.run(command("now", ckpt_a), env=env, capture_output=True, text=True, timeout=120)
    assert first.returncode == 0 and "MATERIALIZE DONE" in first.stdout, first.stderr
    export_a = _tree_bytes(out)
    holder = subprocess.Popen(
        command("hold_until_go", ckpt_a), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    waiter = None
    try:
        deadline = time.time() + 120
        while not (gate / "holding").exists():
            assert holder.poll() is None, holder.communicate()
            assert time.time() < deadline, "the holder never acquired the lock"
            time.sleep(0.05)
        waiter_ckpt = ckpt_a if second == "identical_skips_with_0" else ckpt_b
        waiter = subprocess.Popen(
            command("now", waiter_ckpt), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        time.sleep(1.5)
        assert waiter.poll() is None, (
            f"the second request waits for the lock (on the parent it exits 1 at once): {waiter.communicate()}"
        )
    finally:
        (gate / "go").write_text("go")
        holder_out, holder_err = holder.communicate(timeout=120)
    assert holder.returncode == 0 and "idempotent skip" in holder_out, holder_err
    assert waiter is not None
    waiter_out, waiter_err = waiter.communicate(timeout=180)
    assert "being materialized by another process" not in waiter_err, waiter_err
    if second == "identical_skips_with_0":
        assert waiter.returncode == 0 and "idempotent skip" in waiter_out, waiter_err
    else:
        assert (
            waiter.returncode == 1 and f"REFUSED: {out} holds a different or unproven materialization" in waiter_err
        ), waiter_err
    assert _tree_bytes(out) == export_a, "the published export is intact"
    assert not os.path.exists(out + ".tmp") and not os.path.exists(out + ".previous")


def test_the_lock_wait_is_bounded_and_the_refusal_names_the_bound(
    tmp_path: Path, base: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The bound: with the lock held by this process, an identical request with
    ``--lock-wait-seconds 0.6`` waits about that long and is then refused by name with the bound stated, touching nothing;
    ``output_lock(..., wait_seconds=0)`` refuses at once; a negative bound is an argument error; released, the same request skips
    with 0. On the parent the option does not exist (an argument error)."""
    calls, args, out = _default_export(tmp_path, base, monkeypatch)
    capsys.readouterr()
    before = _tree_bytes(out)
    with mat.output_lock(out, source="held-by-the-test"):
        started = time.monotonic()
        assert mat.main([*args, "--lock-wait-seconds", "0.6"]) == 1
        elapsed = time.monotonic() - started
        err = capsys.readouterr().err
        assert 0.5 <= elapsed < 30, elapsed
        assert (
            f"FATAL: REFUSED: {out} is being materialized by another process ({out}.lock held by pid {os.getpid()}"
            in err
        ), err
        assert (
            "and the lock was not released within 0.6 s (--lock-wait-seconds)" in err and "nothing was touched" in err
        ), err
        assert _tree_bytes(out) == before and len(calls) == 1 and "Traceback" not in err
        with (
            pytest.raises(mat.MaterializeRefusedError, match="not released within 0 s"),
            mat.output_lock(out, wait_seconds=0),
        ):
            pass
        assert mat.main([*args, "--lock-wait-seconds", "-0.0"]) == 1, "a held lock with a -0.0 bound: refused at once"
        err = capsys.readouterr().err
        assert "not released within 0 s (--lock-wait-seconds)" in err and "-0 s" not in err, err  # No "-0 s"
    assert mat.main(args) == 0 and "idempotent skip" in capsys.readouterr().out, (
        "released: the same request skips with 0"
    )
    with pytest.raises(SystemExit):
        mat.main([*args, "--lock-wait-seconds", "-1"])
    assert "--lock-wait-seconds must be >= 0 and finite" in capsys.readouterr().err


# ---------------------------------------------------------------- the .writing temporary of the unified-template writer
def test_a_failed_unified_rewrite_during_recovery_leaves_the_writing_temporary_and_the_refusal_names_it(
    tmp_path: Path, base: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Under ``--unified-chat-template`` the recovery of an inherited sole complete marked copy
    re-syncs both template files to the base's non-unified form at the sidecar-sync stage and rewrites them with the unified text
    at the unified-template stage through a ``.writing`` temporary (``hf._write_unified_file``: write ``<name>.writing``, then
    ``os.replace``); a failure between the write and the rename leaves ``tokenizer_config.json.writing`` beside both files in the
    base's form, the shards and the marker untouched. On the parent the refusal listed ".copying temporary" as the only
    temporary. Here: a unified build whose first publication dies before its rename; the unforced unified retry with the writer
    made to fail after writing the temporary of ``tokenizer_config.json``: exit 1, the refusal names the stage and the
    ``.writing`` temporary of the unified-template writer, and the tree is as it says (shards, index, stamp and marker
    byte-identical; ``tokenizer_config.json.writing`` present; both template files = the base's bytes; no ``out``); the plain
    unified retry consumes the temporary (the writer overwrites and renames it) and publishes the unified files with no residue."""
    calls = _install_fake_overlay(monkeypatch)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "model.pt").write_bytes(b"\0" * 10)
    out = str(tmp_path / "out")
    args = [
        "--ckpt-dir",
        str(ckpt),
        "--base",
        str(base),
        "--out",
        out,
        "--min-tensors",
        "3",
        "--no-verify",
        "--unified-chat-template",
    ]
    real_publish = mat.publish_staging

    def dying_publish(tmp_out: str, out_dir: str) -> str | None:
        raise KeyboardInterrupt("simulated death right before the publication rename")

    monkeypatch.setattr(mat, "publish_staging", dying_publish)
    with pytest.raises(KeyboardInterrupt):
        mat.main(args)
    monkeypatch.setattr(mat, "publish_staging", real_publish)
    capsys.readouterr()
    tmp = out + ".tmp"
    assert len(calls) == 1 and not os.path.exists(out) and mat.complete_marked(tmp)[0]
    assert _marker(tmp)["identity"]["unified_chat_template"] is True
    found = _tree_bytes(tmp)
    kept_names = (mat.MATERIALIZED_MARKER, mat.SOURCE_STAMP, hf_overlay.INDEX_FILE)
    invariant = {k: v for k, v in found.items() if k.endswith(".safetensors") or k in kept_names}
    assert json.loads(found["tokenizer_config.json"])["chat_template"] == FAKE_CANONICAL_TEMPLATE, (
        "the unified text, as built"
    )
    real_write = hf_overlay._write_unified_file

    def failing_write(out_dir: str, base_dir: str, name: str, unified_text: str) -> tuple[str, bool]:
        Path(out_dir, name + ".writing").write_bytes(b'{"partial":')  # the temporary written, the rename never reached
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(hf_overlay, "_write_unified_file", failing_write)
    assert mat.main(args) == 1
    err = capsys.readouterr().err
    after = _tree_bytes(tmp)
    assert {k: v for k, v in after.items() if k in invariant} == invariant, "shards, index, stamp and marker untouched"
    assert after["tokenizer_config.json.writing"] == b'{"partial":', "the .writing temporary left by the failed writer"
    assert after["tokenizer_config.json"] == (base / "tokenizer_config.json").read_bytes(), (
        "re-synced to the base's form"
    )
    assert after["chat_template.json"] == (base / "chat_template.json").read_bytes(), "re-synced to the base's form"
    assert not os.path.exists(out) and not os.path.exists(out + ".previous") and len(calls) == 1
    assert (
        f"FATAL: the recovery of the inherited staging directory {tmp} failed at the unified chat template stage: "
        in err
    ), err
    assert (
        "OSError: [Errno 5] Input/output error" in err
        and "the directory is kept: its shards and its marker are untouched" in err
    )
    assert (
        "a .copying or .writing temporary left by an interrupted copy or unified-template rewrite, and under "
        "--unified-chat-template the two template files back in the base's form when that rewrite did not reach its rename"
    ) in err, err
    assert f"no output exists at {out}" in err and "Traceback" not in err, err
    monkeypatch.setattr(hf_overlay, "_write_unified_file", real_write)
    assert mat.main(args) == 0 and len(calls) == 1, "the plain unified retry publishes the copy, no overlay call"
    published = _tree_bytes(out)
    assert not any(k.endswith(".writing") for k in published), "the writer consumed the temporary"
    assert json.loads(published["tokenizer_config.json"])["chat_template"] == FAKE_CANONICAL_TEMPLATE
    assert json.loads(published["chat_template.json"])["chat_template"] == FAKE_CANONICAL_TEMPLATE
    assert {k: v for k, v in published.items() if k.endswith(".safetensors")} == {
        k: v for k, v in invariant.items() if k.endswith(".safetensors")
    }
    assert not os.path.exists(tmp) and not os.path.exists(out + ".previous")


@pytest.mark.parametrize("value", ["inf", "nan"])
def test_lock_wait_seconds_refuses_non_finite_values(
    value: str, tmp_path: Path, base: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Re-derived from the code: ``--lock-wait-seconds`` is ``type=float``, so argparse
    accepted ``inf`` (``time.monotonic() + inf`` never passes: an unbounded wait) and ``nan``
    (``max(0.0, nan)`` is ``0.0``: a silent refuse-at-once). Now the CLI refuses a non-finite or negative value as an argument error
    naming it, ``run_request`` raises ``ValueError`` naming it before touching anything, and ``output_lock`` does the same; a finite
    bound still works. On the parent ``inf`` waited and ``nan`` acted as 0."""
    calls, args, out = _default_export(tmp_path, base, monkeypatch)
    capsys.readouterr()
    before = _tree_bytes(out)
    with pytest.raises(SystemExit):
        mat.main([*args, "--lock-wait-seconds", value])
    err = capsys.readouterr().err
    assert (
        f"--lock-wait-seconds must be >= 0 and finite (0 = refuse a held lock at once), got {float(value)!r}" in err
    ), err
    with pytest.raises(
        ValueError, match=re.escape(f"lock_wait_seconds must be a finite number >= 0, got {float(value)!r}")
    ):
        mat.run_request(base=str(base), out=out, ckpt_dir=str(tmp_path / "ckpt"), lock_wait_seconds=float(value))
    with pytest.raises(ValueError, match=re.escape(f"wait_seconds must be a finite number >= 0, got {float(value)!r}")):
        with mat.output_lock(out, wait_seconds=float(value)):
            pass
    assert _tree_bytes(out) == before and len(calls) == 1 and not os.path.exists(out + ".tmp")
    assert mat.main([*args, "--lock-wait-seconds", "5"]) == 0 and "idempotent skip" in capsys.readouterr().out
    assert mat.main([*args, "--lock-wait-seconds", "-0.0"]) == 0, "-0.0 is a finite 0: accepted"


# ---------------------------------------------------------------- the source identity is read under the lock
_SOURCE_RACE_DRIVER = r"""
import contextlib, hashlib, json, os, struct, sys, time
from pathlib import Path
from cosmos_framework.posttrain.checkpoint.conversion import hf
from cosmos_framework.posttrain.checkpoint.tools import materialize_hf as mat
from cosmos_framework.posttrain.tests import test_checkpoint_materialize_hf as fixtures
mode, ckpt, base, out, gate = sys.argv[1:6]


def fake_overlay(ckpt_dir, model_id, staging_dir, payload=None):
    import shutil
    os.makedirs(staging_dir)
    shutil.copy2(os.path.join(model_id, "config.json"), os.path.join(staging_dir, "config.json"))
    shutil.copy2(os.path.join(model_id, hf.INDEX_FILE), os.path.join(staging_dir, hf.INDEX_FILE))
    if payload is not None:  # the materializer hands the overlay the file object it hashed under the lock
        payload.seek(0)
        source = payload.read()  # the tensors carry the bytes the overlay READ
        payload.seek(0)
    else:
        source = Path(ckpt_dir, "model.pt").read_bytes()
    for shard in sorted(set(fixtures.WMAP.values())):
        names = sorted(k for k, v in fixtures.WMAP.items() if v == shard)
        header = {n: {"dtype": "BF16", "shape": [2], "data_offsets": [i * 4, (i + 1) * 4]} for i, n in enumerate(names)}
        header["__metadata__"] = {"format": "pt"}
        encoded = json.dumps(header).encode()
        Path(staging_dir, shard).write_bytes(struct.pack("<Q", len(encoded)) + encoded + source[:4] * len(names))
    return staging_dir


hf.overlay_consolidated_onto_hf = fake_overlay
if mode == "hold_lock_only":  # a holder that only keeps the lock until the test says go
    with mat.output_lock(out, source="holder"):
        Path(gate, "holding").write_text(str(os.getpid()))
        deadline = time.time() + 120
        while not os.path.exists(os.path.join(gate, "go")):
            if time.time() > deadline:
                raise RuntimeError("gate timeout")
            time.sleep(0.05)
    sys.exit(0)
sys.exit(mat.main(["--ckpt-dir", ckpt, "--base", base, "--out", out, "--min-tensors", "3", "--no-verify", *sys.argv[6:]]))
"""


def _shard_data(path: str) -> bytes:
    """The data region of a one-tensor fake shard (after the 8-byte length and the header)."""
    raw = Path(path).read_bytes()
    return raw[8 + struct.unpack("<Q", raw[:8])[0] :]


@pytest.mark.parametrize("during_the_wait", ["source_replaced", "source_unchanged"])
def test_the_marker_records_the_source_the_overlay_read_when_the_source_changes_during_the_lock_wait(
    during_the_wait: str, tmp_path: Path, base: Path
) -> None:
    """Reproduction as real subprocesses: on the parent ``probe_local_source``
    hashed ``model.pt`` BEFORE the output lock, which may wait up to 120 s; a ``merge --force`` (tmp + os.replace over model.pt) during
    the wait is not excluded by that lock, so after acquisition the overlay read the new checkpoint B while the source record and the
    published marker carried A's hash. Here a holder keeps the lock; the waiter starts and waits; model.pt is replaced with B the way the
    merge tool writes it (a temporary, then os.replace); the holder is released: the waiter builds, the marker's source sha256 is B's
    and the published tensors carry B's bytes (the driver's fake overlay writes the bytes it read into the shards). The unchanged case
    is pinned beside it (A throughout). On the parent the replaced case publishes B's tensors under A's sha."""
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "model.pt").write_bytes(b"\0" * 10)
    source_a, source_b = hashlib.sha256(b"\0" * 10).hexdigest(), hashlib.sha256(b"\1" * 10).hexdigest()
    out = str(tmp_path / "out")
    gate = tmp_path / "gate"
    gate.mkdir()
    roots = [p for p in sys.path if p] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(dict.fromkeys(roots)), "PYTHONDONTWRITEBYTECODE": "1"}

    def command(mode: str, *extra: str) -> list[str]:
        return [sys.executable, "-c", _SOURCE_RACE_DRIVER, mode, str(ckpt), str(base), out, str(gate), *extra]

    holder = subprocess.Popen(
        command("hold_lock_only"), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    waiter = None
    try:
        deadline = time.time() + 120
        while not (gate / "holding").exists():
            assert holder.poll() is None, holder.communicate()
            assert time.time() < deadline, "the holder never acquired the lock"
            time.sleep(0.05)
        waiter = subprocess.Popen(command("now"), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        time.sleep(1.5)  # the waiter has hashed A before the lock and waits
        assert waiter.poll() is None, f"the waiter must be waiting for the lock: {waiter.communicate()}"
        if during_the_wait == "source_replaced":  # the merge tool's write: a temporary, then os.replace over model.pt
            Path(ckpt, "model.pt.tmp").write_bytes(b"\1" * 10)
            os.replace(ckpt / "model.pt.tmp", ckpt / "model.pt")
    finally:
        (gate / "go").write_text("go")
        holder_out, holder_err = holder.communicate(timeout=120)
    assert holder.returncode == 0, holder_err
    assert waiter is not None
    waiter_out, waiter_err = waiter.communicate(timeout=180)
    assert waiter.returncode == 0 and "MATERIALIZE DONE" in waiter_out, waiter_err
    expected_sha, expected_data = (
        (source_b, b"\1" * 4) if during_the_wait == "source_replaced" else (source_a, b"\0" * 4)
    )
    marker = _marker(out)
    assert _shard_data(os.path.join(out, SHARD1)) == expected_data, (
        "the published tensors carry the bytes the overlay read"
    )
    assert marker["identity"]["source_sha256"] == expected_sha, (
        "the marker names the source the overlay read (on the parent: A's pre-wait sha over B's tensors)"
    )
    assert (
        marker["source"]["sha256"] == expected_sha
        and json.loads(Path(out, mat.SOURCE_STAMP).read_text())["identity"]["source_sha256"] == expected_sha
    )
    assert not os.path.exists(out + ".tmp") and not os.path.exists(out + ".previous")


@pytest.mark.parametrize("rewrite", ["different_bytes", "same_bytes"])
def test_a_source_replaced_during_the_overlay_does_not_change_what_was_built_or_recorded(
    rewrite: str, tmp_path: Path, base: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This supersedes the earlier post-overlay re-hash: the overlay maps the file object the
    materializer opened and hashed under the lock, so a replacement of the PATH during the overlay (a fake overlay that writes over
    model.pt by tmp + os.replace after the build read it) changes nothing of what was built or recorded: the build publishes, the
    marker and the source record name the opened inode's identity (A) and the path now holds the replacement. Under the earlier re-hash
    the different-bytes case failed the build and the same-bytes case was "accepted" on the digest of the final file,
    which does not say what the overlay consumed."""
    calls = _install_fake_overlay(monkeypatch)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "model.pt").write_bytes(b"\0" * 10)
    out = str(tmp_path / "out")
    args = ["--ckpt-dir", str(ckpt), "--base", str(base), "--out", out, "--min-tensors", "3", "--no-verify"]
    fake_overlay = hf_overlay.overlay_consolidated_onto_hf
    replacement = b"\1" * 10 if rewrite == "different_bytes" else b"\0" * 10

    def replacing_overlay(ckpt_dir: str, model_id: str, staging_dir: str, payload: object = None) -> str:
        fake_overlay(ckpt_dir, model_id, staging_dir, payload)
        time.sleep(0.01)  # a distinct mtime_ns even on a coarse clock
        Path(ckpt_dir, "model.pt.tmp").write_bytes(replacement)
        os.replace(Path(ckpt_dir, "model.pt.tmp"), Path(ckpt_dir, "model.pt"))
        return staging_dir

    monkeypatch.setattr(hf_overlay, "overlay_consolidated_onto_hf", replacing_overlay)
    assert mat.main(args) == 0 and len(calls) == 1, (
        "the build publishes (on the parent the different-bytes case failed)"
    )
    marker = _marker(out)
    assert marker["identity"]["source_sha256"] == SHA_ZERO10 and marker["source"]["sha256"] == SHA_ZERO10
    assert (ckpt / "model.pt").read_bytes() == replacement and not os.path.exists(out + ".tmp")


_MERGE_RACE_DRIVER = r"""
import json, os, struct, sys
from pathlib import Path
from cosmos_framework.posttrain.checkpoint.conversion import hf
from cosmos_framework.posttrain.checkpoint.tools import materialize_hf as mat
from cosmos_framework.posttrain.checkpoint.tools import merge as merge_mod
from cosmos_framework.posttrain.tests import test_checkpoint_materialize_hf as fixtures
mode, ckpt, base, out, experts_a, experts_b = sys.argv[1:7]


def fake_overlay(ckpt_dir, model_id, staging_dir, payload=None):
    # The real overlay minus the HF model: the REAL consolidated-checkpoint loader reads model.pt (through the payload the
    # materializer hands over; by path before), and the bytes of the tensor it loaded go into the shards.
    import shutil
    import torch
    model_pt = os.path.join(ckpt_dir, "model.pt")
    if mode == "a_b_a":  # a forced merge of the SUPPORTED merge tool replaces model.pt with B after the identity of record
        assert merge_mod.main(["--mode", "avg", "--experts", experts_b, "--out", model_pt, "--force"]) == 0
    if payload is not None:
        state = hf.load_consolidated_state_dict(model_pt, payload)
    else:
        state = hf.load_consolidated_state_dict(model_pt)
    data = state["a.weight"].contiguous().view(torch.uint8).numpy().tobytes()  # the bytes the overlay consumed
    if mode == "a_b_a":  # a second forced merge restores A at the path before the materializer's post-overlay check
        assert merge_mod.main(["--mode", "avg", "--experts", experts_a, "--out", model_pt, "--force"]) == 0
    os.makedirs(staging_dir)
    shutil.copy2(os.path.join(model_id, "config.json"), os.path.join(staging_dir, "config.json"))
    shutil.copy2(os.path.join(model_id, hf.INDEX_FILE), os.path.join(staging_dir, hf.INDEX_FILE))
    for shard in sorted(set(fixtures.WMAP.values())):
        names = sorted(k for k, v in fixtures.WMAP.items() if v == shard)
        header = {n: {"dtype": "BF16", "shape": [2], "data_offsets": [i * 4, (i + 1) * 4]} for i, n in enumerate(names)}
        header["__metadata__"] = {"format": "pt"}
        encoded = json.dumps(header).encode()
        Path(staging_dir, shard).write_bytes(struct.pack("<Q", len(encoded)) + encoded + data[:4] * len(names))
    return staging_dir


hf.overlay_consolidated_onto_hf = fake_overlay
sys.exit(mat.main(["--ckpt-dir", ckpt, "--base", base, "--out", out, "--min-tensors", "3", "--no-verify"]))
"""


@pytest.mark.parametrize("during_the_overlay", ["a_b_a", "unchanged"])
def test_the_marker_names_the_bytes_the_overlay_consumed_when_forced_merges_race_the_overlay(
    during_the_overlay: str, tmp_path: Path, base: Path
) -> None:
    """The residual case with the SUPPORTED merge tool, as a real subprocess: the
    identity of record is A; during the overlay a forced merge replaces model.pt with B, the loader reads, a second forced merge
    restores A; the final digest is A again. On the parent the loader read the PATH (B) and the post-overlay re-hash accepted the
    final digest as "same bytes": the marker recorded A over B's tensors. Now the overlay maps the file object opened under
    the lock, so it consumed A: the marker names A and the published tensors carry A's bytes, whatever the path held meanwhile. The
    unchanged case (no merge during the overlay) is pinned beside it. Needs torch (the merge tool, the loader); skips without it."""
    torch = pytest.importorskip("torch")
    from cosmos_framework.posttrain.checkpoint.tools import merge as merge_mod

    def expert(name: str, value: float) -> str:
        path = tmp_path / f"{name}.pt"
        torch.save({"net": {"a.weight": torch.full((2,), value, dtype=torch.bfloat16)}}, path)  # [2]
        return str(path)

    experts_a = ",".join([expert("a1", 1.0), expert("a2", 1.0)])
    experts_b = ",".join([expert("b1", 2.0), expert("b2", 2.0)])
    bytes_a, bytes_b = b"\x80\x3f" * 2, b"\x00\x40" * 2  # bf16 1.0 and 2.0, little endian, two elements
    ckpt = tmp_path / "iter_000300"  # the merge tool binds the request to an iter_* directory
    model_pt = ckpt / "model.pt"
    assert merge_mod.main(["--mode", "avg", "--experts", experts_a, "--out", str(model_pt)]) == 0
    sha_a = hashlib.sha256(model_pt.read_bytes()).hexdigest()
    out = str(tmp_path / "out")
    roots = [p for p in sys.path if p] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(dict.fromkeys(roots)), "PYTHONDONTWRITEBYTECODE": "1"}
    run = subprocess.run(
        [sys.executable, "-c", _MERGE_RACE_DRIVER, during_the_overlay, str(ckpt), str(base), out, experts_a, experts_b],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert run.returncode == 0 and "MATERIALIZE DONE" in run.stdout, run.stderr
    assert hashlib.sha256(model_pt.read_bytes()).hexdigest() == sha_a, (
        "the merge tool reproduces A byte for byte (the precondition of this scenario)"
    )
    marker = _marker(out)
    assert marker["identity"]["source_sha256"] == sha_a and marker["source"]["sha256"] == sha_a
    assert _shard_data(os.path.join(out, SHARD1)) == bytes_a, (
        "the published tensors carry the bytes of the source the marker names (on the parent: B's bytes under A's sha)"
    )
    assert _shard_data(os.path.join(out, SHARD1)) != bytes_b and not os.path.exists(out + ".tmp")


@pytest.mark.parametrize("kind", ["local", "archived"])
def test_the_default_verification_receives_the_held_payload(
    kind: str, tmp_path: Path, base: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On the default verification path the materializer hands ``verify_with_torch`` the
    SAME open file object it hashed under the lock and the overlay consumed (keyword ``payload``: open, the inode of the source it
    names), for a local checkpoint and for the fetched copy of an archived artifact; ``--no-verify`` is untouched. On the parent
    ``verify_with_torch`` received the path only (``payload_given`` False)."""
    calls = _install_fake_overlay(monkeypatch)
    seen: dict[str, object] = {}

    def recording_verify(out: str, base_dir: str, model_pt: str, *args: object, **kwargs: object) -> dict[str, str]:
        payload = kwargs.get("payload")
        seen["payload_given"] = payload is not None
        if payload is not None:
            seen["same_inode"] = os.fstat(payload.fileno()).st_ino == os.stat(model_pt).st_ino  # type: ignore[attr-defined]
            seen["open"] = not payload.closed  # type: ignore[attr-defined]
        return {"torch_probe": "stand-in"}

    monkeypatch.setattr(mat, "verify_with_torch", recording_verify)
    out = str(tmp_path / "out")
    if kind == "local":
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        (ckpt / "model.pt").write_bytes(b"\0" * 10)
        args = ["--ckpt-dir", str(ckpt), "--base", str(base), "--out", out, "--min-tensors", "3"]
    else:
        client = FakeS3Client()
        client.plant(f"{PREFIX}/{CHECKPOINT_ARTIFACT_MANIFEST}", _manifest_payload(URI, SHA_ZERO10))
        monkeypatch.setattr(object_store, "build_s3_client", lambda credentials: client)

        def fake_fetch(artifact_uri: str, dst_dir: str, *, credentials: str | None) -> str:
            artifact_dir = os.path.join(dst_dir, "iter_000300")
            os.makedirs(artifact_dir, exist_ok=True)
            Path(artifact_dir, "model.pt").write_bytes(b"\0" * 10)
            Path(artifact_dir, CHECKPOINT_ARTIFACT_MANIFEST).write_bytes(_manifest_payload(URI, SHA_ZERO10))
            Path(artifact_dir, ".staged.json").write_text("{}")
            return artifact_dir

        monkeypatch.setattr(mat, "_fetch_archived_artifact", fake_fetch)
        args = ["--artifact-uri", URI, "--base", str(base), "--out", out, "--min-tensors", "3"]
    assert mat.main(args) == 0 and len(calls) == 1
    assert seen == {"payload_given": True, "same_inode": True, "open": True}, (
        f"verify_with_torch gets the held payload (on the parent: the path only): {seen}"
    )
    marker = _marker(out)
    assert marker["verify"]["torch"] is True and marker["identity"]["source_sha256"] == SHA_ZERO10
