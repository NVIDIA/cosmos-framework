# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Adapter for a Cosmos action-conditioned world model.

Build inference from an adapted checkpoint, write observation/action/text inputs,
and generate the visual continuation. The notebook uses precomputed 20-D EEF
actions, normalized once with the checkpoint's saved statistics. The adapter
passes these inputs directly to Cosmos in forward-dynamics mode.

Joint-state encoding remains available for older callers; the EEF notebook does
not use it.

Run under ``torchrun``; the framework expects a distributed context even on one node.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def encode_chunk(states: np.ndarray, stats: dict, jaw_columns: list[int], chunk_length: int) -> np.ndarray:
    """One action chunk from ``(chunk_length + 1, state_dim)`` absolute states.

    Module level, not a method, because the exporter needs this same encoding to prove a
    reconstructed trajectory against what the model was fed, and constructing an adapter for that
    would mean loading a checkpoint. Two copies of the encoding would be free to drift apart, and
    the drift would show up as a verification that passes while meaning nothing.
    """
    import torch

    from cosmos_framework.data.generator.action.action_normalization import normalize_action

    action = (states[1:] - states[:-1]).copy()
    action[:, jaw_columns] = states[:chunk_length][:, jaw_columns]
    norm = normalize_action(torch.from_numpy(action).float(), "quantile", stats)
    return norm.detach().cpu().numpy().astype(np.float32)


class CosmosForwardDynamics:
    def __init__(self, profile, checkpoint: str | Path | None = None, output_dir: str | Path | None = None):
        from cosmos_framework.inference.args import OmniSetupOverrides

        self.profile = profile
        cfg = dict(profile.model)
        self.checkpoint = str(checkpoint or cfg.get("checkpoint") or "")
        if not self.checkpoint:
            raise ValueError("no checkpoint: pass one to the adapter or set model.checkpoint in the profile")
        self.domain = cfg.get("domain_name")
        self.view_point = cfg.get("view_point", "concat_view")
        self.image_size = int(cfg.get("image_size", 480))
        self.num_steps = int(cfg.get("num_steps", 50))
        self.guidance = float(cfg.get("guidance", 1.0))
        self.shift = cfg.get("shift")
        self.chunk_length = profile.dataset.chunk_length
        self.fps = profile.dataset.target_fps

        self.action_encoding = cfg.get("action_encoding", "joint_delta")
        if self.action_encoding not in ("joint_delta", "precomputed"):
            raise ValueError(f"unknown action_encoding: {self.action_encoding!r}")
        self._stats = None
        if self.action_encoding == "joint_delta":
            stats_path = cfg.get("normalizer_stats")
            if not stats_path:
                raise ValueError("model.normalizer_stats is required: actions must reach the model normalized")
            self._stats = self._load_stats(Path(stats_path))

        setup_options = {key: cfg[key] for key in ("use_torch_compile", "use_cuda_graphs", "config_file") if key in cfg}
        self._setup = OmniSetupOverrides(
            checkpoint_path=self.checkpoint,
            output_dir=Path(output_dir or "."),
            parallelism_preset=cfg.get("parallelism_preset", "latency"),
            guardrails=False,
            **setup_options,
        ).build_setup()
        self._pipe = self._setup.get_inference_cls().create(self._setup)
        expected_domains = cfg.get("expected_num_embodiment_domains")
        if expected_domains is not None and self._pipe.model_config.num_embodiment_domains != expected_domains:
            raise ValueError("Loaded checkpoint domain count differs from the prepared training contract")
        self._sample_cls = self._setup.get_sample_overrides_cls()

    # -- action encoding ----------------------------------------------------
    @staticmethod
    def _load_stats(path: Path) -> dict:
        import torch

        from cosmos_framework.data.generator.action.action_normalization import load_action_stats

        return {k: torch.from_numpy(v).float() for k, v in load_action_stats(str(path)).items()}

    def encode_actions(self, states: np.ndarray) -> list[list[float]]:
        """Joint deltas plus absolute grippers, quantile normalized.

        The gripper columns pass through as absolute values rather than differences, which is
        precisely why a slip is expressible: the aperture is a state the model reads, not a
        change it has to integrate.
        """
        if self.action_encoding == "precomputed":
            raise ValueError("precomputed actions require a prepared manifest; cannot encode joint states")
        return encode_chunk(states, self._stats, self.profile.robot.jaw_columns, self.chunk_length).tolist()

    # -- rollout ------------------------------------------------------------
    def rollout_chunk(self, obs_image: Path, actions: list[list[float]], prompt: str, out_dir: Path, seed: int) -> Path:
        from cosmos_framework.inference.common.init import is_rank0

        inputs = out_dir / "inputs"
        if is_rank0():
            inputs.mkdir(parents=True, exist_ok=True)
            obs = inputs / "obs.png"
            if Path(obs_image) != obs:
                from PIL import Image

                Image.open(obs_image).convert("RGB").save(obs, format="PNG")
            action_path = inputs / "action.json"
            action_path.write_text(json.dumps(actions))
            (out_dir / "sample.json").write_text(
                json.dumps(
                    {
                        "model_mode": "forward_dynamics",
                        "name": out_dir.name,
                        "domain_name": self.domain,
                        "vision_path": str(obs),
                        "action_path": str(action_path),
                        "prompt": prompt,
                        "action_chunk_size": self.chunk_length,
                        "fps": self.fps,
                        "image_size": self.image_size,
                        "view_point": self.view_point,
                        "num_steps": self.num_steps,
                        "guidance": self.guidance,
                        **({"shift": float(self.shift)} if self.shift is not None else {}),
                        "seed": seed,
                    },
                    indent=2,
                )
            )
        self._barrier()

        overrides = self._sample_cls.from_files([out_dir / "sample.json"], overrides=self._setup.sample_overrides)[0]
        overrides.output_dir = out_dir
        overrides.download(inputs)
        self._pipe.generate([overrides.build_sample(model_config=self._pipe.model_config)])
        self._barrier()

        clip = out_dir / "vision.mp4"
        if is_rank0() and not clip.is_file():
            raise FileNotFoundError(f"model produced no clip at {clip}")
        return clip

    def infer_actions(self, video: Path, prompt: str, out_dir: Path) -> np.ndarray | None:
        from cosmos_framework.inference.common.init import is_rank0

        if is_rank0():
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "sample.json").write_text(
                json.dumps(
                    {
                        "model_mode": "inverse_dynamics",
                        "name": out_dir.name,
                        "domain_name": self.domain,
                        "vision_path": str(video),
                        "prompt": prompt,
                        "action_chunk_size": self.chunk_length,
                        "fps": self.fps,
                        "image_size": self.image_size,
                        "view_point": self.view_point,
                        "num_steps": self.num_steps,
                        "seed": 0,
                    },
                    indent=2,
                )
            )
        self._barrier()
        overrides = self._sample_cls.from_files([out_dir / "sample.json"], overrides=self._setup.sample_overrides)[0]
        overrides.output_dir = out_dir
        overrides.download(out_dir / "inputs")
        self._pipe.generate([overrides.build_sample(model_config=self._pipe.model_config)])
        self._barrier()
        if not is_rank0():
            return None
        got = json.loads((out_dir / "sample_outputs.json").read_text())
        return np.asarray(got["outputs"][0]["content"]["action"], dtype=np.float32)

    @staticmethod
    def _barrier() -> None:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
