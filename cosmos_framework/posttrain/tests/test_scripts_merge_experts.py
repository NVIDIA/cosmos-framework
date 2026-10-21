# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""End-to-end coverage for the copied task-arithmetic merge script."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")


def _write_indexed_checkpoint(directory: Path, tensors: dict[str, torch.Tensor]) -> None:  # tensors values: [*]
    from safetensors.torch import save_file

    directory.mkdir()
    shard_names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    keys = sorted(tensors)
    weight_map = {keys[0]: shard_names[0], keys[1]: shard_names[1]}
    for key, shard_name in weight_map.items():
        save_file({key: tensors[key]}, str(directory / shard_name), metadata={"format": "pt"})
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map}),
        encoding="utf-8",
    )


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("lambda_args", "scales"),
    [
        (["--lam", "0.5"], (0.5, 0.5)),
        (["--lams", "0.5,0.25"], (0.5, 0.25)),
    ],
    ids=("shared-lambda", "per-expert-lambdas"),
)
def test_merge_experts_preserves_original_cli_and_formula(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lambda_args: list[str],
    scales: tuple[float, float],
) -> None:
    from safetensors.torch import load_file

    from cosmos_framework.posttrain.scripts import merge_experts

    base = tmp_path / "base"
    expert_a = tmp_path / "expert-a"
    expert_b = tmp_path / "expert-b"
    output = tmp_path / "merged"

    base_tensors = {
        "model.layer.weight": torch.tensor([1.0, 2.0]),  # [D]
        "model.norm.weight": torch.tensor([2.0, 4.0]),  # [D]
    }
    expert_a_tensors = {
        "model.layer.weight": torch.tensor([3.0, 4.0]),  # [D]
        "model.norm.weight": torch.tensor([4.0, 6.0]),  # [D]
    }
    expert_b_tensors = {
        "model.layer.weight": torch.tensor([5.0, 6.0]),  # [D]
        "model.norm.weight": torch.tensor([6.0, 8.0]),  # [D]
    }
    _write_indexed_checkpoint(base, base_tensors)
    _write_indexed_checkpoint(expert_a, expert_a_tensors)
    _write_indexed_checkpoint(expert_b, expert_b_tensors)
    (base / "config.json").write_text('{"model_type": "tiny"}\n', encoding="utf-8")
    (base / "tokenizer.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(merge_experts, "BASE", str(base))
    merge_experts.main(
        [
            "--experts",
            f"{expert_a},{expert_b}",
            *lambda_args,
            "--out",
            str(output),
        ]
    )

    output_index = json.loads((output / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert output_index["weight_map"] == {
        "model.layer.weight": "model-00001-of-00002.safetensors",
        "model.norm.weight": "model-00002-of-00002.safetensors",
    }
    for key, base_tensor in base_tensors.items():  # base_tensor: [D]
        expert_a_tensor = expert_a_tensors[key]  # [D]
        expert_b_tensor = expert_b_tensors[key]  # [D]
        expected = (  # [D]
            base_tensor + scales[0] * (expert_a_tensor - base_tensor) + scales[1] * (expert_b_tensor - base_tensor)
        )
        output_tensor = load_file(str(output / output_index["weight_map"][key]))[key]  # [D]
        expected_bfloat16 = expected.to(torch.bfloat16)  # [D]
        assert output_tensor.dtype == torch.bfloat16
        assert torch.equal(output_tensor, expected_bfloat16)
    assert (output / "config.json").read_text(encoding="utf-8") == '{"model_type": "tiny"}\n'
    assert (output / "tokenizer.json").read_text(encoding="utf-8") == "{}\n"
