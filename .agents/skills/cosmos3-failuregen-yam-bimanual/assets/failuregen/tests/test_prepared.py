# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Frozen EEF inputs remain unchanged through loading and generation."""

import copy
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import av
import numpy as np
import pytest
from failuregen import plan
from failuregen.adapters.cosmos import CosmosForwardDynamics
from failuregen.config import Profile
from failuregen.prepared import load_inputs, sha256
from failuregen.runner import run_manifest
from failuregen.yam_training.contract import CAPTION, EEF_CONTRACT, normalize
from failuregen.yam_training.failure_test import transplant
from failuregen.yam_training.failure_types import profile_dict
from PIL import Image


@pytest.fixture
def prepared(tmp_path):
    output = tmp_path / "prepared"
    output.mkdir()
    stats = {"contract": EEF_CONTRACT, "q01": [-1.0] * 20, "q99": [2.0] * 20}
    stats_path = output / "action_stats.json"
    stats_path.write_text(json.dumps(stats))
    contract = output / "yam_training_contract.json"
    contract.write_text(json.dumps({"contract": EEF_CONTRACT, "stats_sha256": sha256(stats_path)}))
    config = output / "config.yaml"
    config.write_text("model:\n  config:\n    num_embodiment_domains: 32\n")
    profile = Profile.from_dict(
        profile_dict(
            output,
            tmp_path / "dataset",
            output,
            {"annotations": "failuregen_annotations.json"},
            [{"phase": "lift_extract", "score_arm": "left", "label": "pickup_slip"}],
        )
    )
    host = np.zeros((16, 20), dtype=np.float32)
    for offset in (0, 10):
        host[:, offset + 3] = 1
        host[:, offset + 7] = 1
        host[:, offset + 9] = 0.25
    donor = host.copy()
    donor[:, 0] = 0.1
    donor[:, 9] = 0.8
    variants = {"control": host, "failure_donor": transplant(host, donor, arms=("left",))}
    image = output / "obs.png"
    Image.new("RGB", (960, 960), (20, 40, 60)).save(image)
    runs = []
    for name, raw in variants.items():
        path = output / f"{name}.json"
        path.write_text(json.dumps(normalize(raw, stats).tolist()))
        runs.append(
            dict(
                name=name,
                recipe=name,
                mode="control" if name == "control" else "perturbation",
                collection="YAM",
                host_episode=4,
                donor=None,
                start_seconds=0.0,
                inject_at_s=0.8,
                arm="left",
                target_phase="lift_extract",
                caption=CAPTION,
                provenance={"condition": name},
                perturbation=None,
                prepared=dict(
                    vision_path="obs.png",
                    vision_sha256=sha256(image),
                    action_path=path.name,
                    action_sha256=sha256(path),
                    chunks=1,
                    normalization="train_quantile",
                    stats_sha256=sha256(stats_path),
                    training_contract_sha256=sha256(contract),
                    config_sha256=sha256(config),
                ),
            )
        )
    (output / "manifest.json").write_text(json.dumps({"runs": runs}))
    return profile, output, normalize(host, stats).tolist(), None


def test_frozen_pair_is_portable_and_preserves_host_prefix(prepared, tmp_path):
    profile, output, original, _ = prepared
    moved = tmp_path / "moved"
    shutil.copytree(output, moved)
    for key in ("normalizer_stats", "training_contract_file", "config_file"):
        profile.model[key] = str(moved / Path(profile.model[key]).name)
    runs, donors = plan.load(moved / "manifest.json")
    assert donors == {} and len(runs) == 2
    baseline_image, baseline = load_inputs(profile, runs[0])
    changed_image, changed = load_inputs(profile, runs[1])
    assert baseline_image == changed_image == moved / "obs.png"
    assert baseline == original
    np.testing.assert_array_equal(np.array(changed)[:4], np.array(baseline)[:4])
    np.testing.assert_array_equal(np.array(changed)[:, 10:], np.array(baseline)[:, 10:])
    assert not np.array_equal(changed, baseline)
    assert runs[0]["caption"] == runs[1]["caption"] == CAPTION


@pytest.mark.parametrize("corruption", ["hash", "width", "nan"])
def test_prepared_inputs_reject_corruption(prepared, corruption):
    profile, output, _, _ = prepared
    run = plan.load(output / "manifest.json")[0][0]
    path = Path(run["prepared"]["action_path"])
    values = json.loads(path.read_text())
    if corruption == "width":
        values = [row[:14] for row in values]
    else:
        values[0][0] = float("nan") if corruption == "nan" else 100
    path.write_text(json.dumps(values))
    if corruption != "hash":
        run["prepared"]["action_sha256"] = sha256(path)
    with pytest.raises(ValueError, match="changed after planning|Expected finite"):
        load_inputs(profile, run)


def test_prepared_inputs_reject_mismatched_checkpoint_statistics(prepared):
    profile, output, _, _ = prepared
    run = plan.load(output / "manifest.json")[0][0]
    path = output / "action_stats.json"
    stats = json.loads(path.read_text())
    stats["q99"][0] = 3.0
    path.write_text(json.dumps(stats))
    with pytest.raises(ValueError, match="saved training config/normalizer"):
        load_inputs(profile, run)


def test_runner_passes_exact_inputs_without_renormalizing(prepared):
    profile, output, original, _ = prepared
    runs, donors = plan.load(output / "manifest.json")

    class Model:
        checkpoint = "/example/Cosmos3-Nano"

        def __init__(self):
            self.calls = []

        def encode_actions(self, states):
            pytest.fail("Prepared inputs must never go through the joint encoder")

        def rollout_chunk(self, image, actions, prompt, out_dir, seed):
            self.calls.append((Image.open(image).tobytes(), copy.deepcopy(actions), prompt, seed))
            out_dir.mkdir(parents=True)
            clip = out_dir / "vision.mp4"
            with av.open(str(clip), "w") as container:
                stream = container.add_stream("libx264", rate=5)
                stream.width = stream.height = 64
                stream.pix_fmt = "yuv420p"
                for i in range(17):
                    frame = av.VideoFrame.from_image(Image.new("RGB", (64, 64), (i * 10, 20, 40)))
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            return clip

    model = Model()
    root = output / "runs"
    result = run_manifest(profile, model, runs, donors, root, seed=17)
    assert result["rendered"] == 2 and result["failed"] == []
    assert model.calls[0][1] == original
    assert model.calls[0][0] == model.calls[1][0]
    assert model.calls[0][2:] == model.calls[1][2:] == (CAPTION, 17)
    for run in runs:
        directory = root / run["name"]
        record = json.loads((directory / "record.json").read_text())
        assert record["seed"] == 17 and record["checkpoint"] == model.checkpoint
        assert record["failure_time_s"] == 0.8
        assert (directory / "actions_precomputed.json").is_file()


def test_adapter_skips_normalization_only_when_explicitly_precomputed(prepared, monkeypatch):
    profile, _, _, _ = prepared
    seen = []
    setup = SimpleNamespace(
        get_inference_cls=lambda: SimpleNamespace(
            create=lambda _: SimpleNamespace(model_config=SimpleNamespace(num_embodiment_domains=32))
        ),
        get_sample_overrides_cls=lambda: object(),
    )
    args = ModuleType("cosmos_framework.inference.args")

    def overrides(**kwargs):
        seen.append(kwargs)
        return SimpleNamespace(build_setup=lambda: setup)

    args.OmniSetupOverrides = overrides
    monkeypatch.setitem(sys.modules, "cosmos_framework.inference.args", args)
    monkeypatch.setattr(CosmosForwardDynamics, "_load_stats", lambda *a: pytest.fail("Unexpected normalization"))
    adapter = CosmosForwardDynamics(profile, checkpoint="/unchanged/Cosmos3-Nano")
    assert adapter._stats is None
    assert seen[0]["checkpoint_path"] == "/unchanged/Cosmos3-Nano"
    assert seen[0]["use_torch_compile"] is False and seen[0]["use_cuda_graphs"] is False
    with pytest.raises(ValueError, match="cannot encode joint states"):
        adapter.encode_actions(np.zeros((17, 14)))
    profile.model.pop("action_encoding")
    profile.model.pop("normalizer_stats")
    with pytest.raises(ValueError, match="normalizer_stats is required"):
        CosmosForwardDynamics(profile, checkpoint="/unchanged/Cosmos3-Nano")
