# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU tests for the consolidated-expert merge (``checkpoint.tools.merge``).

Known answers on a synthetic two-shard safetensors base plus three experts whose deltas are multiples of 3 (exact in
bf16, so the mean over three experts is exact as well): the arithmetic, the bitwise reload, the native artifact
manifest, the request-identity idempotency (skip / refuse / force) and every recovery path from an unfinished or
damaged artifact directory.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from cosmos_framework.posttrain.checkpoint.artifact import (
    CHECKPOINT_ARTIFACT_MANIFEST,
    read_local_artifact_manifest,
    validate_artifact_manifest,
)
from cosmos_framework.posttrain.checkpoint.tools import merge as merge_mod

pytestmark = [pytest.mark.L0, pytest.mark.CPU]

BF16 = torch.bfloat16
SHAPES = {"model.a.weight": (64, 32), "model.b.weight": (16,), "lm_head.weight": (40, 32), "model.c.weight": (8, 8)}
ITERATION = 300
ITER_NAME = "iter_000300"
MANIFEST_KEYS = {"protocol", "iteration", "iteration_name", "checkpoint_format", "checkpoint_components", "components"}


@dataclass
class Synthetic:
    """The synthetic base directory, the three expert files and the exact expected merges."""

    root: Path
    base: dict[str, torch.Tensor]  # [*SHAPES[key]] per value, bf16
    deltas: list[dict[str, torch.Tensor]]  # one dict per expert, [*SHAPES[key]] per value, bf16
    experts: list[str]

    @property
    def base_dir(self) -> str:
        return str(self.root)

    def out(self, name: str) -> str:
        return str(self.root / name / ITER_NAME / "model.pt")

    def want_avg(self) -> dict[str, torch.Tensor]:  # values: [*SHAPES[key]]
        return {
            k: (self.base[k].float() + sum(d[k].float() for d in self.deltas) / 3).to(BF16) for k in SHAPES
        }  # [*SHAPES[k]]

    def want_tv(self) -> dict[str, torch.Tensor]:  # values: [*SHAPES[key]]
        return {
            k: (self.base[k].float() + sum(d[k].float() for d in self.deltas)).to(BF16) for k in SHAPES
        }  # [*SHAPES[k]]


@pytest.fixture
def synthetic(tmp_path: Path) -> Synthetic:
    from safetensors.torch import save_file

    torch.manual_seed(0)
    base = {k: torch.randint(-8, 9, s).to(BF16) for k, s in SHAPES.items()}  # values: [*SHAPES[k]]
    shards = {
        "model-00001-of-00002.safetensors": ["model.a.weight", "model.b.weight"],
        "model-00002-of-00002.safetensors": ["lm_head.weight", "model.c.weight"],
    }
    weight_map: dict[str, str] = {}
    for shard, names in shards.items():
        save_file({k: base[k] for k in names}, str(tmp_path / shard))
        weight_map.update({k: shard for k in names})
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    deltas, experts = [], []
    for t in range(3):
        delta = {k: (torch.randint(-2, 3, s) * 3).to(BF16) for k, s in SHAPES.items()}  # values: [*SHAPES[k]]
        deltas.append(delta)
        path = tmp_path / f"expert{t}.pt"
        torch.save(
            {"net": {k: (base[k].float() + delta[k].float()).to(BF16) for k in SHAPES}}, path
        )  # values: [*SHAPES[k]]
        experts.append(str(path))
    return Synthetic(tmp_path, base, deltas, experts)


def _assert_equal(state: dict[str, torch.Tensor], want: dict[str, torch.Tensor]) -> None:  # values: [*SHAPES[key]]
    assert set(state) == set(want)
    for k, v in want.items():
        assert torch.equal(state[k], v), f"{k} differs from the expected merge"


def _record(out: str) -> dict:
    return json.loads((Path(out).parent / merge_mod.MERGE_COMPLETE).read_text())


def _assert_complete(out: str, want: dict[str, torch.Tensor]) -> None:  # want values: [*SHAPES[key]]
    directory = Path(out).parent
    assert (directory / CHECKPOINT_ARTIFACT_MANIFEST).is_file() and (directory / merge_mod.MERGE_COMPLETE).is_file()
    _assert_equal(merge_mod.load_expert(out), want)
    assert _record(out)["artifacts"]["payload"]["sha256"] == merge_mod.payload_digest(out)["sha256"]
    assert read_local_artifact_manifest(str(directory))["components"]["model"]["size"] == os.path.getsize(out)


def _refused(fn: Callable[[], object]) -> str:
    with pytest.raises(merge_mod.MergeRefusedError, match="REFUSED") as info:
        fn()
    return str(info.value)


def test_avg_is_the_elementwise_mean_and_records_the_request(synthetic: Synthetic) -> None:
    out, diag = synthetic.out("avg"), str(synthetic.root / "avg_diag.json")
    assert merge_mod.run_request(synthetic.experts, "avg", 1.0, synthetic.base_dir, out, diag, ITERATION) == "built"
    _assert_complete(out, synthetic.want_avg())
    assert os.path.isfile(diag)
    record = _record(out)
    assert record["protocol"] == merge_mod.MERGE_COMPLETE_PROTOCOL and record["tool"] == merge_mod.TOOL_VERSION
    assert record["identity_digest"] == merge_mod.identity_digest(record["identity"])
    assert record["identity"]["experts"][1]["sha256"] == merge_mod.payload_digest(synthetic.experts[1])["sha256"]
    assert record["merge"] == {
        "mode": "avg",
        "lam": 1.0,
        "iteration": ITERATION,
        "experts": synthetic.experts,
        "base": synthetic.base_dir,
        "experts_sha256": [e["sha256"] for e in record["identity"]["experts"]],
    }
    assert record["artifacts"]["diag"] == {"path": os.path.realpath(diag), "bytes": os.path.getsize(diag)}
    assert json.loads(Path(diag).read_text())["identity_digest"] == record["identity_digest"]


def test_emitted_manifest_is_a_valid_native_artifact_manifest(synthetic: Synthetic) -> None:
    out = synthetic.out("avg")
    merge_mod.run_request(synthetic.experts, "avg", 1.0, None, out, None)
    directory = str(Path(out).parent)
    raw = json.loads((Path(directory) / CHECKPOINT_ARTIFACT_MANIFEST).read_text())
    assert set(raw) == MANIFEST_KEYS
    assert validate_artifact_manifest(raw, artifact_location=directory) == raw
    manifest = read_local_artifact_manifest(directory)
    assert manifest["iteration"] == ITERATION and manifest["iteration_name"] == ITER_NAME
    assert manifest["checkpoint_format"] == "consolidated" and manifest["checkpoint_components"] == ["model"]
    assert manifest["components"] == {
        "model": {
            "kind": "file",
            "path": "model.pt",
            "size": os.path.getsize(out),
            "sha256": merge_mod.payload_digest(out)["sha256"],
        }
    }
    # The merge request lives in the completion record, not in the manifest: any extra manifest key is rejected.
    with pytest.raises(RuntimeError, match="unsupported"):
        validate_artifact_manifest({**raw, "merge": {}}, artifact_location=directory)
    record = _record(out)
    assert record["artifacts"]["manifest"] == {
        "name": CHECKPOINT_ARTIFACT_MANIFEST,
        "bytes": os.path.getsize(os.path.join(directory, CHECKPOINT_ARTIFACT_MANIFEST)),
    }
    assert record["artifacts"]["diag"] is None


def test_tv_lambda_one_is_base_plus_sum_of_deltas(synthetic: Synthetic) -> None:
    out = synthetic.out("tv")
    merge_mod.merge(synthetic.experts, "tv", 1.0, synthetic.base_dir, out, None)
    _assert_complete(out, synthetic.want_tv())


def test_tv_lambda_one_third_matches_avg(synthetic: Synthetic) -> None:
    avg_out, tv_out = synthetic.out("avg"), synthetic.out("tv3")
    merge_mod.merge(synthetic.experts, "avg", 1.0, synthetic.base_dir, avg_out, None)
    merge_mod.merge(synthetic.experts, "tv", 1.0 / 3, synthetic.base_dir, tv_out, None)
    avg, tv3 = merge_mod.load_expert(avg_out), merge_mod.load_expert(tv_out)  # values: [*SHAPES[key]]
    for k in SHAPES:
        # lam=1/3 is not exact in fp32: exact zeros come out as ~6e-8 residuals (representable in bf16), so compare
        # with a tolerance far below the smallest real weight delta.
        assert torch.allclose(tv3[k].float(), avg[k].float(), atol=1e-6, rtol=0), (
            f"tv(1/3) != avg on {k}"
        )  # [*SHAPES[k]]


def test_key_set_mismatch_is_refused(synthetic: Synthetic) -> None:
    bad = str(synthetic.root / "bad.pt")
    torch.save(
        {"net": {k: v for k, v in merge_mod.load_expert(synthetic.experts[0]).items() if k != "model.c.weight"}}, bad
    )
    out = synthetic.out("bad_out")
    with pytest.raises(ValueError, match="key set differs"):
        merge_mod.merge([synthetic.experts[0], bad], "avg", 1.0, None, out, None)
    assert not os.path.exists(out)


def test_invalid_requests_are_rejected_before_anything_is_written(synthetic: Synthetic) -> None:
    out = synthetic.out("invalid")
    with pytest.raises(ValueError, match="needs --base"):
        merge_mod.run_request(synthetic.experts, "tv", 1.0, None, out, None)
    with pytest.raises(ValueError, match="at least two"):
        merge_mod.run_request(synthetic.experts[:1], "avg", 1.0, None, out, None)
    with pytest.raises(ValueError, match="iter_"):
        merge_mod.run_request(synthetic.experts, "avg", 1.0, None, str(synthetic.root / "flat" / "model.pt"), None)
    with pytest.raises(ValueError, match="does not match the output directory"):
        merge_mod.run_request(synthetic.experts, "avg", 1.0, None, out, None, ITERATION + 1)
    with pytest.raises(FileNotFoundError):
        merge_mod.run_request([*synthetic.experts, str(synthetic.root / "missing.pt")], "avg", 1.0, None, out, None)
    assert not os.path.exists(Path(out).parent) and not os.path.exists(synthetic.root / "flat")


def test_same_request_skips_and_leaves_the_payload_untouched(synthetic: Synthetic) -> None:
    out, diag = synthetic.out("avg"), str(synthetic.root / "avg_diag.json")
    assert merge_mod.run_request(synthetic.experts, "avg", 1.0, synthetic.base_dir, out, diag, ITERATION) == "built"
    before = os.stat(out)
    assert merge_mod.run_request(synthetic.experts, "avg", 1.0, synthetic.base_dir, out, diag, ITERATION) == "skipped"
    assert merge_mod.run_request(synthetic.experts, "avg", 1.0, synthetic.base_dir, out, diag) == "skipped"
    assert merge_mod.run_request(synthetic.experts, "avg", 1.0, synthetic.base_dir, out, None) == "skipped"
    after = os.stat(out)
    assert (after.st_mtime_ns, after.st_size) == (before.st_mtime_ns, before.st_size)


def test_different_request_is_refused_and_the_payload_stays(synthetic: Synthetic) -> None:
    out = synthetic.out("avg")
    experts, base = synthetic.experts, synthetic.base_dir
    assert merge_mod.run_request(experts, "avg", 1.0, base, out, None) == "built"
    sha = merge_mod.payload_digest(out)["sha256"]
    record = _record(out)
    message = _refused(lambda: merge_mod.run_request(experts, "tv", 1.0, base, out, None))
    assert "mode:" in message, message
    _refused(lambda: merge_mod.run_request(experts, "avg", 0.5, base, out, None))
    _refused(lambda: merge_mod.run_request(experts[::-1], "avg", 1.0, base, out, None))
    other_base = synthetic.root / "base2"
    other_base.mkdir()
    for name in os.listdir(synthetic.root):
        if name.endswith(".safetensors") or name.endswith(".index.json"):
            shutil.copy(synthetic.root / name, other_base / name)
    _refused(lambda: merge_mod.run_request(experts, "avg", 1.0, str(other_base), out, None))
    _refused(lambda: merge_mod.run_request(experts, "avg", 1.0, None, out, None))
    assert merge_mod.payload_digest(out)["sha256"] == sha, "a refused request modified the payload"
    assert _record(out) == record, "a refused request modified the completion record"


def test_changed_expert_content_is_refused_until_forced(synthetic: Synthetic) -> None:
    out, diag = synthetic.out("avg"), str(synthetic.root / "avg_diag.json")
    experts, base = synthetic.experts, synthetic.base_dir
    assert merge_mod.run_request(experts, "avg", 1.0, base, out, diag) == "built"
    avg = {k: v.clone() for k, v in merge_mod.load_expert(out).items()}  # values: [*SHAPES[k]]
    sha = merge_mod.payload_digest(out)["sha256"]
    # same path, same byte size, different content: the mean over three experts shifts by exactly +1
    size = os.path.getsize(experts[2])
    changed = {k: v.clone() for k, v in merge_mod.load_expert(experts[2]).items()}  # values: [*SHAPES[k]]
    changed["model.b.weight"] = (changed["model.b.weight"].float() + 3).to(BF16)  # [16]
    torch.save({"net": changed}, experts[2])
    assert os.path.getsize(experts[2]) == size, "the content-change probe must keep the byte size"
    message = _refused(lambda: merge_mod.run_request(experts, "avg", 1.0, base, out, None))
    assert "expert[2]" in message, message
    assert merge_mod.payload_digest(out)["sha256"] == sha, "refusal must not touch the stale payload"
    assert merge_mod.run_request(experts, "avg", 1.0, base, out, diag, force=True) == "built"
    rebuilt = merge_mod.load_expert(out)  # values: [*SHAPES[key]]
    assert torch.equal(rebuilt["model.b.weight"].float(), avg["model.b.weight"].float() + 1), (  # [16]
        "forced rebuild used old data"
    )
    for k in SHAPES:
        if k != "model.b.weight":
            assert torch.equal(rebuilt[k], avg[k]), f"forced rebuild changed {k}"
    assert _record(out)["identity"]["experts"][2]["sha256"] == merge_mod.payload_digest(experts[2])["sha256"]
    assert merge_mod.run_request(experts, "avg", 1.0, base, out, diag) == "skipped"


def test_recovery_rebuilds_from_every_unfinished_or_damaged_state(synthetic: Synthetic) -> None:
    out = synthetic.out("tv")
    experts, base, want = synthetic.experts, synthetic.base_dir, synthetic.want_tv()
    directory = Path(out).parent
    manifest, record = directory / CHECKPOINT_ARTIFACT_MANIFEST, directory / merge_mod.MERGE_COMPLETE
    assert merge_mod.run_request(experts, "tv", 1.0, base, out, None) == "built"
    # payload written, manifest + record missing (crash right after the payload rename)
    manifest.unlink()
    record.unlink()
    assert merge_mod.run_request(experts, "tv", 1.0, base, out, None) == "built"
    _assert_complete(out, want)
    # payload + manifest written, record missing (crash between manifest and record)
    record.unlink()
    assert merge_mod.run_request(experts, "tv", 1.0, base, out, None) == "built"
    _assert_complete(out, want)
    # record intact, payload truncated
    with open(out, "r+b") as handle:
        handle.truncate(os.path.getsize(out) // 2)
    assert merge_mod.run_request(experts, "tv", 1.0, base, out, None) == "built"
    _assert_complete(out, want)
    # record intact, same-size corrupted payload: only the sha256 catches it
    with open(out, "r+b") as handle:
        handle.seek(os.path.getsize(out) // 2)
        handle.write(bytes(8))
    assert merge_mod.run_request(experts, "tv", 1.0, base, out, None) == "built"
    _assert_complete(out, want)
    # garbage payload, no record
    record.unlink()
    Path(out).write_bytes(b"not a checkpoint")
    assert merge_mod.run_request(experts, "tv", 1.0, base, out, None) == "built"
    _assert_complete(out, want)
    # a requested --diag that is absent can only be produced by re-merging; afterwards both spellings skip
    diag = str(synthetic.root / "tv_diag.json")
    assert merge_mod.run_request(experts, "tv", 1.0, base, out, diag) == "built"
    assert os.path.isfile(diag)
    assert merge_mod.run_request(experts, "tv", 1.0, base, out, diag) == "skipped"
    assert merge_mod.run_request(experts, "tv", 1.0, base, out, None) == "skipped"
    assert not list(directory.glob("*.tmp")), "no temporary residue may survive a completed merge"


def test_main_exit_codes_built_skipped_refused_forced(synthetic: Synthetic, capsys: pytest.CaptureFixture[str]) -> None:
    out = synthetic.out("cli")
    common = ["--experts", ",".join(synthetic.experts), "--base", synthetic.base_dir, "--out", out]
    assert merge_mod.main(["--mode", "avg", *common]) == 0
    _assert_complete(out, synthetic.want_avg())
    assert merge_mod.main(["--mode", "avg", "--iteration", str(ITERATION), *common]) == 0
    assert merge_mod.main(["--mode", "tv", "--lam", "1.0", *common]) == 1
    assert "REFUSED" in capsys.readouterr().err
    _assert_complete(out, synthetic.want_avg())
    assert merge_mod.main(["--mode", "tv", "--lam", "1.0", "--force", *common]) == 0
    _assert_complete(out, synthetic.want_tv())
    with pytest.raises(SystemExit):
        merge_mod.main(["--mode", "sum", *common])
