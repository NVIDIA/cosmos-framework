# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Input failures must not terminate a keep-going manifest iterator."""

import json
from contextlib import contextmanager, nullcontext
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest


def _pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    counts: list[tuple[int, int]],
    *,
    max_num_seqs: int = 1,
    keep_going: bool = True,
) -> tuple[Any, list[Any], list[str]]:
    from cosmos_framework.inference import inference
    from cosmos_framework.inference.args import ModelMode, OmniSampleArgs, OmniSetupArgs
    from cosmos_framework.inference.bimanual import BimanualActionContract
    from cosmos_framework.inference.common import inference as common_inference
    from cosmos_framework.inference.common.args import SampleOutputs

    monkeypatch.setattr(common_inference, "sync_distributed_errors", nullcontext)
    monkeypatch.setattr(inference, "sync_distributed_errors", nullcontext)
    monkeypatch.setattr(inference, "get_rank", lambda: 0)
    monkeypatch.setattr(inference, "get_world_size", lambda: 1)
    monkeypatch.setattr(inference.torch.distributed, "is_initialized", lambda: False)
    contract = BimanualActionContract(action_dim=59, model_action_dim=64, domain_ids={"agibotworld": 15})
    samples = []
    for index, (declared, actual) in enumerate(counts):
        path = tmp_path / f"actions_{index}.json"
        path.write_text(
            json.dumps(
                {
                    "action_space": "normalized_unified_v1",
                    "domain_names": "agibotworld",
                    "action": [[0.0] * 59 for _ in range(actual)],
                }
            )
        )
        samples.append(
            OmniSampleArgs.model_construct(
                name=f"sample_{index}",
                output_dir=tmp_path / f"sample_{index}",
                model_mode=ModelMode.FORWARD_DYNAMICS,
                num_outputs=1,
                seed=2026,
                multiview=None,
                action_path=path,
                action_chunk_size=declared,
                num_frames=declared + 1,
                domain_name="agibotworld",
                tensors_file=None,
                pickle_file=None,
            )
        )

    def get_sample_data(*, sample_args: Any, model: Any) -> dict[str, Any]:
        payload = json.loads(sample_args.action_path.read_text())
        contract.validate(payload, action_count=sample_args.action_chunk_size, domain_name=sample_args.domain_name)
        return {"names": [sample_args.name]}

    monkeypatch.setattr(inference, "get_sample_data", get_sample_data)
    setup = OmniSetupArgs.model_construct(
        max_model_len=None,
        max_num_seqs=max_num_seqs,
        keep_going=keep_going,
        debug=False,
        profile=False,
        benchmark=False,
        warmup=0,
        num_iterations=1,
    )
    pipe = inference.OmniInference(setup_args=setup, model=SimpleNamespace(), guardrails=None, _timer=None)
    generated: list[str] = []

    def generate_batch(sample_args_list: list[Any], data_batch: dict[str, Any], **kwargs: Any) -> list[SampleOutputs]:
        outputs = []
        for sample in sample_args_list:
            if not pipe.should_process_sample(sample):
                continue
            assert sample.name in data_batch["names"]
            generated.append(sample.name)
            output = SampleOutputs(args=sample.model_dump(mode="json"))
            sample.output_dir.mkdir(parents=True, exist_ok=True)
            (sample.output_dir / "sample_outputs.json").write_text(output.model_dump_json())
            outputs.append(output)
        return outputs

    monkeypatch.setattr(pipe, "generate_batch", generate_batch)
    return pipe, samples, generated


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("bad_index", [0, 1, 2])
@pytest.mark.parametrize("max_num_seqs", [1, 2, 3])
def test_bad_sidecar_isolated_and_remaining_rows_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_index: int, max_num_seqs: int
) -> None:
    counts = [(4, 4), (8, 8), (12, 12)]
    declared, _ = counts[bad_index]
    counts[bad_index] = (declared, declared + 4)
    pipe, samples, generated = _pipeline(tmp_path, monkeypatch, counts, max_num_seqs=max_num_seqs)

    outputs = pipe.generate(samples)

    assert generated == [sample.name for index, sample in enumerate(samples) if index != bad_index]
    assert len(outputs) == 3
    assert {output.name: output.status for output in outputs} == {
        sample.name: "error" if index == bad_index else "success" for index, sample in enumerate(samples)
    }
    receipt = json.loads((samples[bad_index].output_dir / "sample_outputs.json").read_text())
    assert receipt["status"] == "error"
    assert receipt["args"]["action_chunk_size"] == declared
    assert f"Expected exactly {declared} action rows" in receipt["message"]
    assert "contract.validate" in receipt["stack_trace"]
    assert receipt["outputs"] == []


@pytest.mark.L0
@pytest.mark.CPU
def test_reported_900_1800_mismatch_does_not_block_later_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipe, samples, generated = _pipeline(tmp_path, monkeypatch, [(900, 900), (900, 1800), (1800, 1800)])

    outputs = pipe.generate(samples)

    assert generated == ["sample_0", "sample_2"]
    assert [output.status for output in outputs] == ["success", "error", "success"]


@pytest.mark.L0
@pytest.mark.CPU
def test_all_invalid_samples_have_receipts_and_do_not_generate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipe, samples, generated = _pipeline(tmp_path, monkeypatch, [(4, 8), (8, 12)])

    outputs = pipe.generate(samples)

    assert generated == []
    assert [output.status for output in outputs] == ["error", "error"]
    assert all((sample.output_dir / "sample_outputs.json").is_file() for sample in samples)
    assert pipe.generate([]) == []


@pytest.mark.L0
@pytest.mark.CPU
def test_fail_fast_remains_the_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipe, samples, generated = _pipeline(tmp_path, monkeypatch, [(4, 4), (4, 8), (4, 4)], keep_going=False)

    with pytest.raises(ValueError, match="Expected exactly 4 action rows"):
        pipe.generate(samples)

    assert generated == ["sample_0"]


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("error", [FileNotFoundError("missing sidecar"), ValueError("invalid action")])
def test_input_read_errors_continue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    from cosmos_framework.inference import inference

    pipe, samples, generated = _pipeline(tmp_path, monkeypatch, [(4, 4), (4, 4)])
    original = inference.get_sample_data

    def prepare(*, sample_args: Any, model: Any) -> dict[str, Any]:
        if sample_args.name == "sample_0":
            raise error
        return original(sample_args=sample_args, model=model)

    monkeypatch.setattr(inference, "get_sample_data", prepare)

    assert [output.status for output in pipe.generate(samples)] == ["error", "success"]
    assert generated == ["sample_1"]


@pytest.mark.L0
@pytest.mark.CPU
def test_unexpected_runtime_error_is_not_swallowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from cosmos_framework.inference import inference

    pipe, samples, generated = _pipeline(tmp_path, monkeypatch, [(4, 4), (4, 4)])
    monkeypatch.setattr(inference, "get_sample_data", Mock(side_effect=RuntimeError("CUDA failure")))

    with pytest.raises(RuntimeError, match="CUDA failure"):
        pipe.generate(samples)

    assert generated == []


@pytest.mark.L0
@pytest.mark.CPU
def test_other_rank_rejection_drops_the_same_sample_locally(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from cosmos_framework.inference import inference

    pipe, samples, generated = _pipeline(tmp_path, monkeypatch, [(4, 4), (4, 4)], max_num_seqs=2)
    failure = inference._SamplePreparationFailure(0, "ValueError: peer validation failure", "peer traceback", True)

    def gather(outcomes: list[Any], local: Any) -> None:
        outcomes[:] = [local, ([1], [failure])]

    monkeypatch.setattr(inference.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(inference.torch.distributed, "all_gather_object", gather)
    # Two cooperating ranks in one computation replica.
    pipe.model.parallel_dims = SimpleNamespace(cp_size=2, cfgp_size=1)
    monkeypatch.setattr(inference, "get_world_size", lambda: 2)

    assert {output.name: output.status for output in pipe.generate(samples)} == {
        "sample_0": "error",
        "sample_1": "success",
    }
    assert generated == ["sample_1"]
    receipt = json.loads((samples[0].output_dir / "sample_outputs.json").read_text())
    assert receipt["stack_trace"] == "peer traceback"


@pytest.mark.L0
@pytest.mark.CPU
def test_empty_replica_pads_from_valid_sample_not_rejected_first_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cosmos_framework.inference import inference
    from cosmos_framework.inference.common.args import SampleOutputs

    pipe, samples, _ = _pipeline(tmp_path, monkeypatch, [(4, 8), (4, 4)])

    def gather(outcomes: list[Any], local: Any) -> None:
        outcomes[:] = [local, ([1], [])]

    monkeypatch.setattr(inference.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(inference.torch.distributed, "all_gather_object", gather)
    monkeypatch.setattr(inference.torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(inference, "get_world_size", lambda: 2)
    zeros = inference.torch.zeros
    monkeypatch.setattr(
        inference.torch, "tensor", lambda data, **kwargs: zeros(1, dtype=inference.torch.long) + data[0]
    )

    batches = list(pipe.create_batches(samples))

    assert isinstance(batches[0], SampleOutputs) and batches[0].status == "error"
    (padding,), data = batches[1]
    assert padding.output_dir is None and padding.name == "padding"
    assert padding.num_steps == 1 and padding.guidance == 1.0
    assert data == {"names": ["sample_1"]}


def _distributed_preparation_worker(rank: int, root: str, replica_size: int) -> None:
    """Exercise real Gloo collectives, replacing only CUDA allocation/generation."""
    import torch
    import torch.distributed as dist

    from cosmos_framework.inference import inference
    from cosmos_framework.inference.common import inference as common_inference

    directory = Path(root)
    dist.init_process_group(
        "gloo", init_method=f"file://{directory / 'store'}", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        with pytest.MonkeyPatch.context() as monkeypatch:
            rank_root = directory / f"rank_{rank}"
            rank_root.mkdir()
            initialized = dist.is_initialized
            pipe, samples, generated = _pipeline(rank_root, monkeypatch, [(4, 8), (4, 4), (8, 8)])
            for sample in samples:
                sample.output_dir = directory / "outputs" / sample.name
            pipe.model.parallel_dims = SimpleNamespace(cp_size=replica_size, cfgp_size=1)
            monkeypatch.setattr(inference, "get_rank", lambda: rank)
            monkeypatch.setattr(inference, "get_world_size", lambda: 2)
            monkeypatch.setattr(dist, "is_initialized", initialized)
            tensor = torch.tensor

            def cpu_tensor(*args: Any, **kwargs: Any) -> Any:
                if kwargs.get("device") == "cuda":
                    kwargs["device"] = "cpu"
                return tensor(*args, **kwargs)

            monkeypatch.setattr(torch, "tensor", cpu_tensor)

            @contextmanager
            def cpu_sync_errors():
                flag = torch.zeros(1, dtype=torch.int32)
                error = None
                try:
                    yield
                except Exception as caught:
                    flag += 1
                    error = caught
                dist.all_reduce(flag)
                if flag.item():
                    raise error or RuntimeError("Another rank failed")

            monkeypatch.setattr(inference, "sync_distributed_errors", cpu_sync_errors)
            monkeypatch.setattr(common_inference, "sync_distributed_errors", cpu_sync_errors)
            generate_batch = pipe.generate_batch
            calls = []

            def collective_generation(*args: Any, **kwargs: Any) -> Any:
                calls.append(1)
                dist.all_reduce(torch.ones(1))
                return generate_batch(*args, **kwargs)

            monkeypatch.setattr(pipe, "generate_batch", collective_generation)
            outputs = pipe.generate(samples)
            (directory / f"rank_{rank}.json").write_text(
                json.dumps(
                    {"generated": generated, "statuses": {o.name: o.status for o in outputs}, "calls": len(calls)}
                )
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("replica_size", [1, 2])
def test_real_distributed_ranks_continue_and_keep_collectives_aligned(tmp_path: Path, replica_size: int) -> None:
    import torch.distributed as dist
    import torch.multiprocessing as mp

    if not dist.is_gloo_available():
        pytest.skip("Gloo is required for CPU distributed validation")
    context = mp.spawn(_distributed_preparation_worker, args=(str(tmp_path), replica_size), nprocs=2, join=False)
    try:
        for _ in range(24):
            if context.join(timeout=5):
                break
        else:
            pytest.fail("Preparation error left distributed inference ranks blocked")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    reports = [json.loads((tmp_path / f"rank_{rank}.json").read_text()) for rank in range(2)]
    assert [report["calls"] for report in reports] == [2, 2]
    generated = [name for report in reports for name in report["generated"]]
    assert sorted(generated) == ["sample_1", "sample_2"]
    receipt = json.loads((tmp_path / "outputs/sample_0/sample_outputs.json").read_text())
    assert receipt["status"] == "error" and "Expected exactly 4 action rows" in receipt["message"]
