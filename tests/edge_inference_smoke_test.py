# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""8-GPU multi-modality inference smoke test for Cosmos3-Edge.

Mirrors ``nano_inference_smoke_test.py`` but targets the smaller Cosmos3-Edge
checkpoint, adapted to its capabilities: Edge has ``sound_gen=false`` (so the
``t2vs`` text2video+sound sample is replaced by a plain ``t2v`` sample) and a
native 480p resolution, but keeps ``action_gen=true``.

A single ``throughput`` ``cosmos_framework.scripts.inference`` call runs three
input samples of different modalities (the ``-i`` flag takes a list of files)
and validates each output:

  * ``inputs/omni/t2v.json`` (text2video, no explicit resolution/num_frames/fps)
    -> a ``vision.mp4`` validated with the stronger whole-clip check (decodes the
    full clip; asserts frame count + real pixel variation, so a collapsed /
    near-constant video fails). Because the input pins none of those fields, this
    sample also exercises the Cosmos3-Edge generation defaults resolved in
    ``args.py`` -- ``resolution=480``, ``num_frames=121`` (the Edge-specific
    default; other models default to 189), ``fps=24`` -- which the test asserts
    from the serialized ``args``.
  * ``inputs/omni/action_forward_dynamics_camera.json`` (forward_dynamics) -> a
    ``vision.mp4`` that decodes to at least one valid video frame (``action_path``
    is an input, not an output).
  * ``inputs/omni/action_policy_robot.json`` (policy) -> BOTH a ``vision.mp4`` and
    a finite, non-empty predicted ``action`` array in ``sample_outputs.json``.

Every sample produces a video; the policy sample additionally produces an action.
Unlike the Nano suite there is no sound sample (Edge has no sound generation) and
no transfer/multi-control run.

Smoke-level only (output validity + the Edge default triple, not numeric
goldens). The checkpoint + its tokenizers download from the HF Hub on first run
and are reused afterward.

Invocation (inside the inference container, from the repo root, on an 8-GPU
node)::

    pytest -s tests/edge_inference_smoke_test.py --num-gpus=8 --levels=2 -o addopts=

Without ``--num-gpus``/``--levels`` (e.g. the no-GPU pre-commit CI) the test is
not collected.
"""

import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from cosmos_framework.inference.fixtures.args import MAX_GPUS

REPO_ROOT = Path(__file__).resolve().parents[1]

_INPUTS = [
    "inputs/omni/t2v.json",
    "inputs/omni/action_policy_robot.json",
    "inputs/omni/action_forward_dynamics_camera.json",
]

# Cosmos3-Edge generation defaults for a plain text2video sample that pins none
# of these fields (see ``_NUM_FRAMES_DEFAULTS`` + the model config resolution and
# the per-modality fps default in ``cosmos_framework/inference/args.py``).
_EDGE_T2V_RESOLUTION = "480"
_EDGE_T2V_NUM_FRAMES = 121
_EDGE_T2V_FPS = 24


def _free_port() -> int:
    """Return a currently-free TCP port for torchrun's rendezvous (avoids
    EADDRINUSE from a hardcoded port / lingering process)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _run(cmd: list[str], log_file: Path) -> str:
    """Run ``cmd`` from the repo root, tee combined output (live to stdout under
    ``pytest -s`` + into ``log_file``). Inherits the caller's env (HF cache, ...)
    plus ``PYTHONPATH=.``. Fails with the log tail on a non-zero exit."""
    env = os.environ.copy()
    env["PYTHONPATH"] = f".:{env.get('PYTHONPATH', '')}"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    captured: list[str] = []
    with log_file.open("w") as fp:
        proc = subprocess.Popen(
            cmd, env=env, cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            fp.write(line)
            captured.append(line)
        returncode = proc.wait()
    text = "".join(captured)
    if returncode != 0:
        pytest.fail(f"inference failed with exit code {returncode}:\n  {' '.join(cmd)}\nLog tail:\n{text[-3000:]}")
    return text


def _assert_valid_video(mp4_path: Path) -> None:
    """Assert ``mp4_path`` decodes to at least one valid, non-degenerate video frame."""
    import av

    assert mp4_path.is_file() and mp4_path.stat().st_size > 1024, f"video missing/too small: {mp4_path}"
    with av.open(str(mp4_path)) as container:
        vstreams = container.streams.video
        assert vstreams, f"no video stream in {mp4_path}"
        width = height = frames = 0
        for frame in container.decode(vstreams[0]):
            width, height, frames = frame.width, frame.height, frames + 1
            break
    assert frames >= 1 and width > 0 and height > 0, f"no decodable video frame in {mp4_path}"


def _assert_video_has_content(mp4_path: Path, *, min_frames: int = 16) -> None:
    """Assert ``mp4_path`` decodes to enough non-degenerate frames.

    Stronger than ``_assert_valid_video`` (which only inspects the first frame):
    decodes the whole clip and checks the frame count plus real pixel variation,
    so a run that produced a well-formed container but collapsed to a constant /
    blank video fails instead of passing.
    """
    import av
    import numpy as np

    with av.open(str(mp4_path)) as container:
        vstreams = container.streams.video
        assert vstreams, f"no video stream in {mp4_path}"
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(vstreams[0])]
    assert len(frames) >= min_frames, f"{mp4_path}: expected >= {min_frames} frames, got {len(frames)}"
    arr = np.stack(frames).astype(np.float64)
    assert np.all(np.isfinite(arr)), f"{mp4_path}: decoded video has non-finite pixels"
    # Both spatial and temporal flatness collapse global std toward 0; a real
    # generated clip sits well above this floor (typically tens on a 0-255 scale).
    assert arr.std() > 3.0, f"{mp4_path}: degenerate/near-constant video (pixel std={arr.std():.3f})"


def _assert_valid_action(content: dict, where: str) -> None:
    """Assert a policy sample's predicted ``action`` is a non-empty, all-finite array."""
    import numpy as np

    assert isinstance(content, dict) and content.get("action") is not None, (
        f"no 'action' in policy output ({where}); content keys={list(content) if isinstance(content, dict) else content}"
    )
    arr = np.asarray(content["action"], dtype=np.float64)
    assert arr.size > 0, f"empty action output ({where})"
    assert np.all(np.isfinite(arr)), f"action output has NaN/Inf ({where})"


@pytest.fixture(scope="module", autouse=True)
def _require_8_gpus() -> None:
    """Skip the module unless we can launch an 8-GPU run here."""
    if shutil.which("torchrun") is None:
        pytest.skip("torchrun not on PATH -- must run inside the inference container")
    try:
        import torch
    except Exception as exc:  # pragma: no cover -- surfaces during dev only
        pytest.skip(f"torch unavailable ({exc!r})")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 8:
        pytest.skip(f"requires 8 visible CUDA devices, found {torch.cuda.device_count()}")


# Defined only when the active MAX_GPUS is 8 -- the conftest rejects ``gpus(N)``
# markers outside ``ALL_NUM_GPUS = (0, 1, MAX_GPUS)``.
if MAX_GPUS == 8:

    @pytest.mark.level(2)
    @pytest.mark.gpus(8)
    def test_edge_inference_omni(tmp_path: Path) -> None:
        """Throughput run over t2v + policy + forward_dynamics on Cosmos3-Edge."""
        out_dir = tmp_path / "out"
        cmd = [
            "torchrun",
            "--nproc_per_node=8",
            f"--master_port={_free_port()}",
            "-m",
            "cosmos_framework.scripts.inference",
            "--parallelism-preset=throughput",
            "-i",
            *_INPUTS,
            "-o",
            str(out_dir),
            "--checkpoint-path",
            "Cosmos3-Edge",
            "--seed=0",
        ]
        _run(cmd, tmp_path / "inference.log")

        results = sorted(out_dir.rglob("sample_outputs.json"))
        assert len(results) == len(_INPUTS), (
            f"expected {len(_INPUTS)} sample_outputs.json (one per input), found {[str(p) for p in results]}"
        )

        # Dispatch validation by what each sample produced (robust to model_mode
        # string formatting): a vision.mp4 -> valid video; an `action` content ->
        # valid action array. The plain t2v sample (no conditioning `vision_path`)
        # gets the stronger whole-clip check and additionally verifies the
        # Cosmos3-Edge generation defaults.
        n_video = n_action = 0
        edge_t2v_checked = False
        for so in results:
            data = json.loads(so.read_text())
            args = data.get("args", {})
            content = data["outputs"][0]["content"]
            sample_dir = so.parent
            video = sample_dir / "vision.mp4"
            # The t2v sample is the only one with no conditioning input (policy and
            # forward_dynamics both set `vision_path`); use it to exercise both the
            # stronger video check and the Edge generation defaults.
            is_t2v = not args.get("vision_path")
            if video.is_file():
                # t2v -> whole-clip check (frame count + real pixel variation, so a
                # collapsed/near-constant clip fails); conditioned modes keep the
                # first-frame check, matching the Nano suite.
                if is_t2v:
                    _assert_video_has_content(video)
                else:
                    _assert_valid_video(video)
                n_video += 1
            if isinstance(content, dict) and content.get("action") is not None:
                _assert_valid_action(content, str(so))
                n_action += 1
            if is_t2v:
                # t2v.json pins none of these, so the resolved args should carry the
                # Edge defaults.
                assert str(args.get("resolution")) == _EDGE_T2V_RESOLUTION, (
                    f"expected Edge t2v resolution {_EDGE_T2V_RESOLUTION}, got {args.get('resolution')} ({so})"
                )
                assert args.get("num_frames") == _EDGE_T2V_NUM_FRAMES, (
                    f"expected Edge t2v num_frames {_EDGE_T2V_NUM_FRAMES}, got {args.get('num_frames')} ({so})"
                )
                assert args.get("fps") == _EDGE_T2V_FPS, (
                    f"expected Edge t2v fps {_EDGE_T2V_FPS}, got {args.get('fps')} ({so})"
                )
                edge_t2v_checked = True

        # Every sample produces a valid video (t2v, forward_dynamics, policy); the
        # policy sample additionally yields an action, and the t2v sample pins the
        # Edge generation defaults.
        assert n_video == len(_INPUTS), f"expected every sample to produce a valid video, got {n_video}/{len(_INPUTS)}"
        assert n_action >= 1, f"expected the policy sample's action to be checked, got {n_action}"
        assert edge_t2v_checked, "expected the t2v sample's Edge defaults (resolution/num_frames/fps) to be checked"
