# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Prepare a clean-host/donor-failure EEF test for the standard failuregen runner."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml

from failuregen.config import Profile

from .contract import CHUNK, EEF_CONTRACT, FPS, file_sha256, load_prepared, normalize
from .prepare_eef import poses
from .prepare_replay import prepare as prepare_replay


def transplant(
    host: np.ndarray,
    donor: np.ndarray,
    *,
    inject_frame: int = 4,
    blend_frames: int = 2,
    arms: tuple[str, ...] = ("left", "right"),
) -> np.ndarray:
    """Blend donor relative tool transforms and absolute grippers into host actions.

    Translation stays in each arm's local previous-tool frame. Rotation blends
    are projected onto SO(3); interpolating six rotation numbers alone is invalid.
    Donor step zero begins at the injection. The host prefix is unchanged.
    """
    host, donor = np.asarray(host, dtype=np.float64), np.asarray(donor, dtype=np.float64)
    if host.ndim != 2 or host.shape[1] != 20 or len(host) < CHUNK or len(host) % CHUNK or donor.shape != host.shape:
        raise ValueError("Expected two equal raw EEF windows of shape (16 * chunks, 20)")
    if not 2 <= inject_frame <= len(host) - 3 or not 1 <= blend_frames <= len(host) - inject_frame:
        raise ValueError("Injection must leave a prefix and a scorable suffix; invalid blend length")
    if not arms or len(set(arms)) != len(arms) or set(arms) - {"left", "right"}:
        raise ValueError("Select left and/or right arm exactly once")
    h_pose, d_pose = poses(host), poses(donor)
    changed = host.copy()
    count = len(host) - inject_frame
    weights = np.minimum(np.arange(1, count + 1) / blend_frames, 1.0)
    for arm in arms:
        index, offset = (0, 0) if arm == "left" else (1, 10)
        for j, weight in enumerate(weights):
            i = inject_frame + j
            for column in (*range(offset, offset + 3), offset + 9):
                changed[i, column] = (1 - weight) * host[i, column] + weight * donor[j, column]
            matrix = (1 - weight) * h_pose[i, index, :3, :3] + weight * d_pose[j, index, :3, :3]
            u, _, vt = np.linalg.svd(matrix)
            rotation = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
            changed[i, offset + 3 : offset + 6] = rotation[:, 0]
            changed[i, offset + 6 : offset + 9] = rotation[:, 1]
    poses(changed)  # Proper transforms and finite values, before normalization.
    if np.allclose(changed, host, atol=1e-8, rtol=0):
        raise ValueError("Donor transplant has no effect")
    return changed.astype(np.float32)


def grid_index(seconds: float) -> int:
    if not math.isfinite(seconds) or seconds < 0 or not math.isclose(round(seconds * FPS) / FPS, seconds):
        raise ValueError("Times must be nonnegative and on the 5 FPS grid")
    return round(seconds * FPS)


def load_context(prepared: Path, dataset: Path, run_dir: Path) -> tuple[dict, dict, dict]:
    """Validate the frozen training inputs before selecting or preparing windows."""
    manifest, stats = load_prepared(prepared)
    contract_file, config_file = run_dir / "yam_training_contract.json", run_dir / "config.yaml"
    training = json.loads(contract_file.read_text())
    if (
        manifest["contract"] != EEF_CONTRACT
        or training["contract"] != EEF_CONTRACT
        or training["stats_sha256"] != manifest["stats_sha256"]
        or training["prepared_manifest_sha256"] != file_sha256(prepared / "manifest.json")
    ):
        raise ValueError("Prepared EEF dataset/normalizer differs from the training run")
    config = yaml.safe_load(config_file.read_text())
    if config["model"]["config"]["num_embodiment_domains"] != 32:
        raise ValueError("Expected the fine-tuned 32-domain model config")
    annotation_file = dataset / manifest["annotations"]
    if file_sha256(annotation_file) != manifest["annotations_sha256"]:
        raise ValueError("Dataset annotations changed since training preparation")
    annotations = json.loads(annotation_file.read_text())["episodes"]
    return manifest, stats, annotations


def prepare(
    prepared: Path,
    dataset: Path,
    run_dir: Path,
    output: Path,
    *,
    host_episode: int = 4,
    host_start: float = 58.8,
    donor_episode: int = 3,
    donor_start: float = 60.0,
    inject_frame: int = 4,
    blend_frames: int = 2,
    ffmpeg: str = "ffmpeg",
) -> dict:
    prepared, dataset, run_dir, output = [p.resolve() for p in (prepared, dataset, run_dir, output)]
    manifest, stats, annotations = load_context(prepared, dataset, run_dir)
    contract_file, config_file = run_dir / "yam_training_contract.json", run_dir / "config.yaml"

    def window(episode, start):
        index = grid_index(start)
        ep = next(e for e in manifest["episodes"] if e["episode"] == episode)
        if index not in ep["valid_starts"]:
            raise ValueError("Selected window touches unknown footage or exceeds the episode")
        path = prepared / ep["states"]
        if file_sha256(path) != ep["states_sha256"]:
            raise ValueError("Prepared EEF states changed")
        with np.load(path, allow_pickle=False) as data:
            actions = data["actions_raw"][index : index + CHUNK].copy()
        return ep, actions

    host, host_actions = window(host_episode, host_start)
    donor, donor_actions = window(donor_episode, donor_start)
    if host_episode == donor_episode or host["outcome"] != "success_clean" or donor["split"] != "train":
        raise ValueError("Use a clean host and a distinct training-set failure donor")
    inject_time = host_start + inject_frame / FPS
    host_ann = annotations[f"episode_{host_episode:06d}"]
    donor_ann = annotations[f"episode_{donor_episode:06d}"]
    if not any(s["phase"] == "handover" and s["start"] <= inject_time < s["end"] for s in host_ann["phase_segments"]):
        raise ValueError("Host injection must be within an annotated handover")
    events = [s for s in donor_ann["segments"] if s["label"] == "handover_drop"]
    event_index, event = next(
        (
            (i, s)
            for i, s in enumerate(events)
            if donor_start <= s["start"] < donor_start + (CHUNK - inject_frame) / FPS
        ),
        (None, None),
    )
    if event is None:
        raise ValueError("The transplanted donor suffix must contain a handover drop")
    if event["start"] < donor_start + blend_frames / FPS:
        raise ValueError("Finish blending before the annotated donor drop")
    variants = {
        "control": host_actions,
        "left_arm_donor": transplant(
            host_actions, donor_actions, inject_frame=inject_frame, blend_frames=blend_frames, arms=("left",)
        ),
        "both_arms_donor": transplant(
            host_actions, donor_actions, inject_frame=inject_frame, blend_frames=blend_frames
        ),
    }
    output.mkdir(parents=True, exist_ok=False)
    # Same CPU preparation and camera geometry as the successful checkpoint replays.
    prepare_replay(prepared, dataset, output / "host", host_episode, host_start, ffmpeg=ffmpeg)
    prepare_replay(prepared, dataset, output / "donor", donor_episode, donor_start, ffmpeg=ffmpeg)
    profile = {
        "robot": {
            "state_dim": 14,
            "arms": {
                "left": {"joints": [0, 6], "jaw": 6, "free": 0.022},
                "right": {"joints": [7, 13], "jaw": 13, "free": 0.996},
            },
            "hold_band": [0.15, 0.5],
            "phase_arm": {"handover": "left"},
        },
        "cameras": {
            "front": EEF_CONTRACT["camera_order"][0],
            "wrist": {"left": EEF_CONTRACT["camera_order"][1], "right": EEF_CONTRACT["camera_order"][2]},
            "front_fraction": 0.5,
        },
        "dataset": {
            "roots": {"YAM": str(dataset)},
            "annotations": manifest["annotations"],
            "source_fps": 30,
            "target_fps": FPS,
            "chunk_length": CHUNK,
            "clean_outcome": "success_clean",
            "failure_labels": ["handover_drop"],
        },
        "model": {
            "adapter": "failuregen.adapters.cosmos:CosmosForwardDynamics",
            "domain_name": "xdof_yam",
            "action_encoding": "precomputed",
            "action_dim": 20,
            "normalization": "train_quantile",
            "normalizer_stats": str(prepared / "action_stats.json"),
            "training_contract_file": str(contract_file),
            "config_file": str(config_file),
            "expected_num_embodiment_domains": 32,
            "view_point": "concat_view",
            "image_size": 720,
            "num_steps": 30,
            "guidance": 1.0,
            "shift": 10.0,
            "parallelism_preset": "throughput",
            "use_torch_compile": False,
            "use_cuda_graphs": False,
        },
    }
    Profile.from_dict(profile)
    (output / "profile.yaml").write_text(yaml.safe_dump(profile, sort_keys=False))
    donor_record = dict(
        source="YAM",
        episode_index=donor_episode,
        event_index=event_index,
        t_start=donor_start,
        failure_time_s=event["start"],
        arm="left",
        label="handover_drop",
        split=donor["split"],
        states_sha256=donor["states_sha256"],
    )
    provenance = dict(
        contract=EEF_CONTRACT,
        host_split=host["split"],
        donor=donor_record,
        host_states_sha256=host["states_sha256"],
        annotations_sha256=manifest["annotations_sha256"],
        donor_event_time_in_generated_clip=inject_frame / FPS + event["start"] - donor_start,
        injection_frame=inject_frame,
        blend_frames=blend_frames,
        rotation_blend="polar projection onto SO(3)",
        grippers="blend absolute recorded values, preserve per-arm polarity",
        manual_failure_confirmation_required=True,
    )
    runs = []
    for name, raw_actions in variants.items():
        action_path = output / f"actions_{name}.json"
        action_path.write_text(json.dumps(normalize(raw_actions, stats).tolist(), allow_nan=False) + "\n")
        (output / f"actions_{name}_raw.json").write_text(json.dumps(raw_actions.tolist(), allow_nan=False) + "\n")
        control = name == "control"
        runs.append(
            dict(
                name=name,
                recipe="control" if control else name,
                mode="control" if control else "perturbation",
                collection="YAM",
                host_episode=host_episode,
                donor=None if control else f"{donor_episode}:{event_index}",
                start_seconds=host_start,
                inject_at_s=inject_time,
                arm="left",
                target_phase="handover",
                caption=host["caption"],
                provenance=provenance,
                perturbation=None
                if control
                else dict(
                    kind="eef_donor_transplant",
                    arms=["left"] if name == "left_arm_donor" else ["left", "right"],
                    rows=[inject_frame, CHUNK],
                    donor_rows=[0, CHUNK - inject_frame],
                ),
                prepared=dict(
                    vision_path="host/obs.png",
                    vision_sha256=file_sha256(output / "host/obs.png"),
                    action_path=action_path.name,
                    action_sha256=file_sha256(action_path),
                    normalization="train_quantile",
                    stats_sha256=manifest["stats_sha256"],
                    training_contract_sha256=file_sha256(contract_file),
                    config_sha256=file_sha256(config_file),
                ),
            )
        )
    result = dict(n_runs=len(runs), runs=runs)
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "manifest_donors.json").write_text(json.dumps({"events": [donor_record]}, indent=2) + "\n")
    print(
        f"Prepared control + two donor transplants: host ep{host_episode}@{host_start}s, "
        f"donor ep{donor_episode}@{donor_start}s; injection at +{inject_frame / FPS}s"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "dataset", "run-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--host-episode", type=int, default=4)
    parser.add_argument("--host-start", type=float, default=58.8)
    parser.add_argument("--donor-episode", type=int, default=3)
    parser.add_argument("--donor-start", type=float, default=60.0)
    parser.add_argument("--inject-frame", type=int, default=4)
    parser.add_argument("--blend-frames", type=int, default=2)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    prepare(**vars(args))


if __name__ == "__main__":
    main()
