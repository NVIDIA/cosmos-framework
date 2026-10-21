# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU tests for the tensor-level helpers of the HF overlay (``checkpoint.conversion.hf``).

The fused-expert transpose rule, the consolidated ``model.pt`` loader and the shard rewrite are exercised on tiny
tensors; the tiny safetensors round trip proves that the torch-free header parser agrees with a file the real
``safetensors`` writer produced, which is exactly what ``verify_with_torch`` relies on. The full overlay
(``overlay_consolidated_onto_hf``) needs a real Hugging Face model and is deliberately not exercised here.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from cosmos_framework.posttrain.checkpoint.conversion import hf as hf_overlay

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_maybe_transpose_expert_known_answers() -> None:
    tensor = torch.arange(24).reshape(2, 3, 4)  # [2,3,4]
    assert hf_overlay.maybe_transpose_expert(tensor, (2, 3, 4)) is tensor, "an equal shape returns the same object"
    fixed = hf_overlay.maybe_transpose_expert(tensor, (2, 4, 3))  # [2,4,3]
    assert fixed is not None and tuple(fixed.shape) == (2, 4, 3) and fixed.is_contiguous()
    for expert in range(2):
        assert torch.equal(fixed[expert], tensor[expert].T), "each expert's matrix is transposed"  # [4,3]
    assert hf_overlay.maybe_transpose_expert(tensor, (2, 5, 4)) is None, "any other mismatch is left to accounting"
    assert hf_overlay.maybe_transpose_expert(tensor, (4, 3, 2)) is None, "a full axis reversal is not the expert rule"
    assert hf_overlay.maybe_transpose_expert(torch.zeros(3, 4), (4, 3)) is None, (
        "only 3-d fused experts qualify"
    )  # [3,4]


def test_load_consolidated_state_dict_validates_top_level(tmp_path: Path) -> None:
    net = {"model.a.weight": torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)}  # [2,3]
    good = tmp_path / "good.pt"
    torch.save({"net": net}, good)
    loaded = hf_overlay.load_consolidated_state_dict(str(good))  # [2,3]
    assert set(loaded) == {"model.a.weight"} and torch.equal(loaded["model.a.weight"], net["model.a.weight"])
    ema = tmp_path / "ema.pt"
    torch.save({"net": net, "net_ema": net}, ema)
    assert set(hf_overlay.load_consolidated_state_dict(str(ema))) == {"model.a.weight"}, "net_ema is tolerated"
    with pytest.raises(hf_overlay.ConversionError, match="does not exist"):
        hf_overlay.load_consolidated_state_dict(str(tmp_path / "missing.pt"))
    no_net = tmp_path / "no_net.pt"
    torch.save({"weights": net}, no_net)
    with pytest.raises(hf_overlay.ConversionError, match="expected a dict containing 'net'"):
        hf_overlay.load_consolidated_state_dict(str(no_net))
    extra = tmp_path / "extra.pt"
    torch.save({"net": net, "optim": {}}, extra)
    with pytest.raises(hf_overlay.ConversionError, match=r"unrecognized top-level keys \['optim'\]"):
        hf_overlay.load_consolidated_state_dict(str(extra))
    empty = tmp_path / "empty.pt"
    torch.save({"net": {}}, empty)
    with pytest.raises(hf_overlay.ConversionError, match="non-empty state dict"):
        hf_overlay.load_consolidated_state_dict(str(empty))


def test_tiny_safetensors_round_trip_matches_torch_free_parser(tmp_path: Path) -> None:
    pytest.importorskip("safetensors")
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    tensor = torch.arange(24, dtype=torch.bfloat16).reshape(2, 3, 4)  # [2,3,4]
    path = str(tmp_path / "s.safetensors")
    save_file({"x.mlp.experts.w": tensor.contiguous()}, path, metadata={"format": "pt"})  # [2,3,4]
    with safe_open(path, framework="pt") as handle:
        assert list(handle.get_slice("x.mlp.experts.w").get_shape()) == [2, 3, 4]
        assert torch.equal(handle.get_tensor("x.mlp.experts.w"), tensor)  # [2,3,4]
    assert torch.equal(load_file(path)["x.mlp.experts.w"], tensor)  # [2,3,4]
    header, n = hf_overlay.safetensors_header(path)
    assert header["x.mlp.experts.w"]["shape"] == [2, 3, 4] and header["x.mlp.experts.w"]["dtype"] == "BF16"
    assert os.path.getsize(path) == 8 + n + tensor.numel() * 2, "8-byte length + header + bf16 data"
    assert hf_overlay.shard_status(path, {"x.mlp.experts.w"}) is None
    assert "tensors differ from the index" in str(hf_overlay.shard_status(path, {"other"}))


def test_rewrite_expert_shards_replaces_only_named_tensors(tmp_path: Path) -> None:
    pytest.importorskip("safetensors")
    import json

    from safetensors.torch import load_file, save_file

    expert_hf = torch.arange(24, dtype=torch.bfloat16).reshape(2, 4, 3)  # [2,4,3]; the transposed in-memory layout
    expert_ckpt = expert_hf.transpose(1, 2).contiguous()  # [2,3,4]; the checkpoint layout vLLM expects on disk
    other = torch.ones(5, dtype=torch.bfloat16)  # [5]
    shard_a, shard_b = "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"
    save_file({"l.mlp.experts.gate_up_proj": expert_hf, "l.norm.weight": other}, str(tmp_path / shard_a))
    save_file({"lm_head.weight": other * 2}, str(tmp_path / shard_b))  # [5]
    weight_map = {"l.mlp.experts.gate_up_proj": shard_a, "l.norm.weight": shard_a, "lm_head.weight": shard_b}
    (tmp_path / hf_overlay.INDEX_FILE).write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    before_b = os.stat(tmp_path / shard_b)

    assert hf_overlay.rewrite_expert_shards(str(tmp_path), {"l.mlp.experts.gate_up_proj": expert_ckpt}) == 1
    rewritten = load_file(str(tmp_path / shard_a))  # expert: [2,3,4], norm: [5]
    assert torch.equal(rewritten["l.mlp.experts.gate_up_proj"], expert_ckpt), "the named tensor carries the new layout"
    assert torch.equal(rewritten["l.norm.weight"], other), "the other tensor in the same shard is untouched"
    after_b = os.stat(tmp_path / shard_b)
    assert (after_b.st_mtime_ns, after_b.st_size) == (before_b.st_mtime_ns, before_b.st_size), "unrelated shards stay"
    assert hf_overlay.shard_status(str(tmp_path / shard_a), {"l.mlp.experts.gate_up_proj", "l.norm.weight"}) is None
    assert not list(tmp_path.glob("*.tmp")), "the rewrite leaves no temporary behind"
    with pytest.raises(hf_overlay.ConversionError, match="rewrite incomplete"):
        hf_overlay.rewrite_expert_shards(str(tmp_path), {"absent.mlp.experts.down_proj": expert_ckpt})
    (tmp_path / hf_overlay.INDEX_FILE).unlink()
    assert hf_overlay.rewrite_expert_shards(str(tmp_path), {"lm_head.weight": other * 3}) == 1, (
        "index-less scan works"
    )  # [5]
    assert torch.equal(load_file(str(tmp_path / shard_b))["lm_head.weight"], other * 3)  # [5]


@pytest.mark.parametrize("reference", ["proc_fd_mmap", "buffer_fallback"])
def test_load_consolidated_state_dict_reads_the_open_file_not_the_path(
    reference: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fix round T6r (review thread 524355e746): with ``payload`` the loader returns the tensors of the file object it was handed,
    memory-mapped through ``/proc/self/fd`` (or read whole through the object where that reference does not exist), even after the
    path was replaced the way the merge tool writes (a temporary, then os.replace); loading by path still reads the path."""
    path = tmp_path / "model.pt"

    def save(value: float) -> None:
        tmp = tmp_path / "model.pt.tmp"
        torch.save({"net": {"w": torch.full((2,), value, dtype=torch.bfloat16)}}, tmp)  # [2]
        os.replace(tmp, path)

    save(1.0)
    if reference == "buffer_fallback":
        monkeypatch.setattr(hf_overlay, "open_file_reference", lambda payload: None)
    with open(path, "rb") as payload:
        if reference == "proc_fd_mmap":
            assert hf_overlay.open_file_reference(payload) == f"/proc/self/fd/{payload.fileno()}"
        save(2.0)  # the merge tool's write replaces the path; the open file object still names the first inode
        loaded = hf_overlay.load_consolidated_state_dict(str(path), payload)["w"]  # [2]
        by_path = hf_overlay.load_consolidated_state_dict(str(path))["w"]  # [2]
    assert torch.equal(loaded, torch.full((2,), 1.0, dtype=torch.bfloat16)), "the open file's bytes, not the path's"
    assert torch.equal(by_path, torch.full((2,), 2.0, dtype=torch.bfloat16)), "loading by path reads the path"
    assert float(loaded.sum()) == 2.0, "the loaded tensors stay readable after the file object is closed"
