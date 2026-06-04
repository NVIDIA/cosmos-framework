# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""8-GPU multi-modality inference smoke test for Cosmos3-Nano.

Runs ONE ``cosmos_framework.scripts.inference`` call over three input samples of
different modalities (the ``-i`` flag takes a list of files) and validates each
sample's output:

  * ``inputs/omni/t2vs.json`` (text2video + sound) -> a ``vision.mp4`` whose
    muxed audio is real sound (finite, non-empty, non-silent, non-constant).
  * ``inputs/omni/action_forward_dynamics_camera.json`` (forward_dynamics) -> a
    ``vision.mp4`` that decodes to at least one valid video frame (``action_path``
    is an input, not an output).
  * ``inputs/omni/action_policy_robot.json`` (policy) -> BOTH a ``vision.mp4`` and
    a finite, non-empty predicted ``action`` array in ``sample_outputs.json``.

All three samples produce a video; the policy sample additionally produces an
action and the t2vs sample an audio track.

Smoke-level only (output validity, not numeric goldens). The checkpoint + its
tokenizers download from the HF Hub on first run and are reused afterward.

Invocation (inside the inference container, from the repo root, on an 8-GPU
node)::

    pytest -s tests/nano_inference_smoke_test.py --num-gpus=8 --levels=2 -o addopts=

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
    "inputs/omni/t2vs.json",
    "inputs/omni/action_policy_robot.json",
    "inputs/omni/action_forward_dynamics_camera.json",
]

# Audio sanity thresholds for the muxed sound track.
_RMS_SILENCE_FLOOR = 1e-4  # below this the track is effectively silence
_PEAK_SANITY_CEIL = 1.5    # decoded float audio should sit within ~[-1, 1]


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


def _decode_audio_track(mp4_path: Path):
    """Decode the muxed audio track of ``mp4_path`` to a (channels, samples) waveform.

    Returns ``(waveform_float64, sample_rate)``. Fails if there is no audio
    stream or it decodes to zero frames.
    """
    import av
    import numpy as np

    with av.open(str(mp4_path)) as container:
        audio_streams = container.streams.audio
        assert audio_streams, f"{mp4_path} has no audio stream"
        astream = audio_streams[0]
        sample_rate = int(astream.rate)
        chunks = [frame.to_ndarray() for frame in container.decode(astream)]
    assert chunks, f"audio stream in {mp4_path} decoded to zero frames"

    orig_dtype = chunks[0].dtype
    wav = np.concatenate(chunks, axis=1).astype(np.float64)
    if np.issubdtype(orig_dtype, np.integer):
        wav = wav / float(np.iinfo(orig_dtype).max)
    return wav, sample_rate


def _assert_sound_not_noise(mp4_path: Path) -> None:
    """Assert the muxed audio is real sound: finite, non-empty, non-silent, non-constant."""
    import numpy as np

    wav, sample_rate = _decode_audio_track(mp4_path)
    assert wav.size > 0, f"empty audio in {mp4_path}"
    assert sample_rate > 0, f"non-positive sample rate {sample_rate} in {mp4_path}"
    assert np.all(np.isfinite(wav)), f"audio in {mp4_path} contains NaN/Inf"

    peak = float(np.max(np.abs(wav)))
    rms = float(np.sqrt(np.mean(wav**2)))
    std = float(wav.std())
    assert peak <= _PEAK_SANITY_CEIL, f"audio peak {peak} outside expected normalized range"
    assert std > 1e-6, f"audio is constant/degenerate (std={std}) in {mp4_path}"
    assert rms > _RMS_SILENCE_FLOOR, f"audio is silent/near-silent (rms={rms}) in {mp4_path}"


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
    def test_nano_inference_omni(tmp_path: Path) -> None:
        """One Cosmos3-Nano inference call over t2vs + policy + forward_dynamics; check each output."""
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
            "Cosmos3-Nano",
            "--seed=0",
        ]
        _run(cmd, tmp_path / "inference.log")

        results = sorted(out_dir.rglob("sample_outputs.json"))
        assert len(results) == len(_INPUTS), (
            f"expected {len(_INPUTS)} sample_outputs.json (one per input), found {[str(p) for p in results]}"
        )

        # Dispatch validation by what each sample produced (robust to model_mode
        # string formatting): a vision.mp4 -> valid video (+ sound if enabled);
        # an `action` content -> valid action array.
        n_video = n_sound = n_action = 0
        for so in results:
            data = json.loads(so.read_text())
            args = data.get("args", {})
            content = data["outputs"][0]["content"]
            sample_dir = so.parent
            video = sample_dir / "vision.mp4"
            if video.is_file():
                _assert_valid_video(video)
                n_video += 1
                if args.get("enable_sound"):
                    _assert_sound_not_noise(video)
                    n_sound += 1
            if isinstance(content, dict) and content.get("action") is not None:
                _assert_valid_action(content, str(so))
                n_action += 1

        # Every sample produces a valid video (t2vs, forward_dynamics, policy);
        # the policy sample additionally yields an action, t2vs an audio track.
        assert n_video == len(_INPUTS), f"expected every sample to produce a valid video, got {n_video}/{len(_INPUTS)}"
        assert n_sound >= 1, f"expected the t2vs sample's audio to be checked, got {n_sound}"
        assert n_action >= 1, f"expected the policy sample's action to be checked, got {n_action}"
