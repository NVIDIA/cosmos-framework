# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import cosmos_framework.scripts.extract_reasoner_features as extraction_cli
from cosmos_framework.data.generator.local_datasets.sft_dataset import get_sft_dataset
from cosmos_framework.model.generator.reasoner_feature_cache import ReasonerFeatureCacheIdentity
from cosmos_framework.model.generator.reasoner_features import ReasonerFeatureRequest
from cosmos_framework.scripts.extract_reasoner_features import (
    DistributedContext,
    ExtractionStats,
    _atomic_write_json,
    _commit_rank_result,
    _find_sft_dataset_config,
    _initialize_distributed,
    _iter_rank_requests,
    _parse_args,
    _prepare_reasoner_model_config,
    _request_owner,
    _validate_extraction_config,
    _wait_for_rank_completion,
)


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_find_sft_dataset_config_requires_exactly_one_lazy_node() -> None:
    node = {"_target_": get_sft_dataset, "jsonl_paths": ["data.jsonl"]}
    assert _find_sft_dataset_config({"outer": {"dataset": node}}) is node

    with pytest.raises(ValueError, match="found 0"):
        _find_sft_dataset_config({"outer": {"dataset": {}}})
    with pytest.raises(ValueError, match="found 2"):
        _find_sft_dataset_config({"first": node, "second": dict(node)})


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_request_owner_is_stable_and_content_sensitive() -> None:
    request = ReasonerFeatureRequest(
        sample_key="sample",
        token_ids=torch.tensor([1, 2]),
        position_ids=torch.tensor([0, 1]),
        causal_offsets=torch.tensor([0, 2]),
        fingerprint="fingerprint-a",
    )
    assert _request_owner(request, 8) == _request_owner(request, 8)
    other = ReasonerFeatureRequest(
        sample_key=request.sample_key,
        token_ids=request.token_ids,
        position_ids=request.position_ids,
        causal_offsets=request.causal_offsets,
        fingerprint="fingerprint-b",
    )
    assert any(_request_owner(request, size) != _request_owner(other, size) for size in range(2, 17))


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_rank_requests_frame_tokens_and_deduplicate_standard_cfg_null(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = SimpleNamespace(
        cfg_dropout_rate=0.1,
        cfg_dropout_keep_metadata=False,
        metadata=[{"uuid": "video"}],
        _tokenize_caption=lambda caption: ([7] if caption else [8], caption),
    )
    documents = [
        SimpleNamespace(sample_key="video_w0", caption="", text_token_ids=(8,)),
        SimpleNamespace(sample_key="video_w0", caption="caption", text_token_ids=(7,)),
    ]
    monkeypatch.setattr(
        extraction_cli, "_special_tokens", lambda _dataset: {"eos_token_id": 9, "start_of_generation": 10}
    )
    monkeypatch.setattr(extraction_cli, "iter_sft_reasoner_documents", lambda _dataset: iter(documents))

    requests = list(
        _iter_rank_requests(
            dataset,  # type: ignore[arg-type]
            identity=ReasonerFeatureCacheIdentity("reasoner", "tokenizer", "framing"),
            rank=0,
            world_size=1,
            use_float_positions=True,
            max_documents_per_rank=None,
        )
    )

    assert [request.sample_key for request in requests] == ["__null__", "video_w0"]
    assert requests[0].token_ids.tolist() == [8, 9, 10]
    assert requests[1].token_ids.tolist() == [7, 9, 10]
    assert all(request.position_ids.dtype == torch.float32 for request in requests)


def _config() -> SimpleNamespace:
    model_instance = {
        "_target_": "cosmos_framework.model.generator.mot.unified_mot.Qwen3VLTextForCausalLM",
        "config": {
            "_target_": "cosmos_framework.configs.base.defaults.reasoner.create_vlm_config",
        },
    }
    model = SimpleNamespace(
        joint_attn_implementation="two_way",
        parallelism=SimpleNamespace(context_parallel_shard_degree=1),
        video_temporal_causal=False,
        causal_training_strategy="none",
        diffusion_expert_config=SimpleNamespace(vision_temporal_position_mode="latent_index"),
        vlm_config=SimpleNamespace(model_instance=model_instance),
    )
    return SimpleNamespace(model=SimpleNamespace(config=model))


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_prepare_reasoner_model_config_disables_generator_and_visual() -> None:
    prepared = _prepare_reasoner_model_config(_config())
    assert prepared["config"]["include_gen_pathway"] is False
    assert prepared["config"]["include_und_pathway"] is True
    assert prepared["config"]["include_visual"] is False


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_validate_extraction_config_fails_closed_on_unsupported_layouts() -> None:
    config = _config()
    _validate_extraction_config(config)

    config.model.config.parallelism.context_parallel_shard_degree = 2
    with pytest.raises(ValueError, match="context_parallel"):
        _validate_extraction_config(config)


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_parse_args_validates_bounded_writer_controls() -> None:
    base = [
        "--sft-toml",
        "recipe.toml",
        "--checkpoint",
        "checkpoint",
        "--output",
        "cache",
        "--reasoner-fingerprint",
        "r",
        "--tokenizer-fingerprint",
        "t",
        "--framing-fingerprint",
        "f",
    ]
    args = _parse_args([*base, "--max-documents-per-rank", "2", "--", "optimizer.lr=1e-5"])
    assert args.max_documents_per_rank == 2
    assert args.max_shard_bytes == 512 * 1024**2
    assert args.overrides == ["optimizer.lr=1e-5"]

    with pytest.raises(SystemExit):
        _parse_args([*base, "--max-shard-bytes", "0"])
    with pytest.raises(SystemExit):
        _parse_args([*base, "--coordination-timeout-seconds", "0"])


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_torchrun_context_uses_global_rank_without_initializing_collectives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RANK", "5")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_RANK", "2")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "job")
    monkeypatch.setenv("TORCHELASTIC_RESTART_COUNT", "3")
    selected: list[int] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(torch.cuda, "set_device", selected.append)

    context = _initialize_distributed("explicit-run")

    assert (context.rank, context.world_size, context.local_rank) == (5, 8, 2)
    assert context.device == torch.device("cuda", 2)
    assert context.run_id == "explicit-run:3"
    assert selected == [2]

    monkeypatch.delenv("RANK")
    with pytest.raises(ValueError, match="Incomplete torchrun topology"):
        _initialize_distributed("explicit-run")

    monkeypatch.setenv("RANK", "5")
    with pytest.raises(ValueError, match="Multi-node extraction requires"):
        _initialize_distributed()


class _FakeWriter:
    def __init__(self, rank_dir: Path) -> None:
        self.rank_dir = rank_dir
        self.finalized = False

    def finalize(self) -> Path:
        stats = json.loads((self.rank_dir / "extraction.stats.json").read_text(encoding="utf-8"))
        assert stats["publishable"] is True
        marker = self.rank_dir / "rank.complete.json"
        marker.write_text("{}", encoding="utf-8")
        self.finalized = True
        return marker


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_smoke_result_never_writes_completion_but_full_result_commits_stats_first(tmp_path: Path) -> None:
    context = DistributedContext(0, 1, 0, torch.device("cpu"), "run:0")
    smoke_writer = _FakeWriter(tmp_path / "smoke" / "rank-00000")
    _commit_rank_result(  # type: ignore[arg-type]
        smoke_writer,
        ExtractionStats(rank=0, written_records=2),
        context=context,
        publishable=False,
    )
    assert not smoke_writer.finalized
    smoke_payload = json.loads((smoke_writer.rank_dir / "extraction.stats.json").read_text(encoding="utf-8"))
    assert smoke_payload["publishable"] is False
    assert not (smoke_writer.rank_dir / "rank.complete.json").exists()

    full_writer = _FakeWriter(tmp_path / "full" / "rank-00000")
    _commit_rank_result(  # type: ignore[arg-type]
        full_writer,
        ExtractionStats(rank=0, written_records=3),
        context=context,
        publishable=True,
    )
    assert full_writer.finalized
    assert (full_writer.rank_dir / "rank.complete.json").is_file()


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_rank_zero_coordinates_through_current_run_files_and_surfaces_failures(tmp_path: Path) -> None:
    context = DistributedContext(0, 2, 0, torch.device("cpu"), "run:0")
    for rank in range(2):
        rank_dir = tmp_path / f"rank-{rank:05d}"
        _atomic_write_json(
            rank_dir / "extraction.stats.json",
            {
                "run_id": context.run_id,
                "world_size": 2,
                "publishable": True,
                "stats": {"rank": rank},
            },
        )
        _atomic_write_json(rank_dir / "rank.complete.json", {})

    assert _wait_for_rank_completion(tmp_path, context=context, timeout_seconds=0.01) == [
        {"rank": 0},
        {"rank": 1},
    ]

    _atomic_write_json(
        tmp_path / "rank-00001" / "extraction.failure.json",
        {"run_id": context.run_id, "traceback": "rank one failed"},
    )
    with pytest.raises(RuntimeError, match="rank one failed"):
        _wait_for_rank_completion(tmp_path, context=context, timeout_seconds=0.01)
