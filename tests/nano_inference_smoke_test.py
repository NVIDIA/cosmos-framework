# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""8-GPU smoke test for Cosmos3-Nano text-to-video-with-sound (t2vs) inference.

Runs the canonical Cosmos3-Nano inference command from ``docs/inference.md`` on
the ``inputs/omni/t2vs.json`` sample (``model_mode=text2video`` +
``enable_sound=True``) on 8 GPUs, and asserts that the run completes, writes a
video, and the muxed audio track is real sound (finite, non-empty, not silence,
not a degenerate/constant signal) -- not numeric goldens (that is
``launch_regression_test.py``'s job).

The checkpoint (and its sound tokenizer) download from the Hugging Face Hub on
first run and are reused from the HF cache afterward.

Invocation (inside the inference container, from the repo root, on an 8-GPU
node)::

    pytest -s tests/nano_inference_smoke_test.py --num-gpus=8 --levels=2 -o addopts=

* ``--num-gpus=8 --levels=2`` matches the markers below; the conftest pins
  ``CUDA_VISIBLE_DEVICES`` accordingly.
* ``-o addopts=`` clears the repo ``.pytest.toml`` addopts that reference an
  optional plugin not installed in the container.

Without ``--num-gpus``/``--levels`` (e.g. the no-GPU pre-commit CI) the test is
not collected.
"""

import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from cosmos_framework.inference.fixtures.args import MAX_GPUS

REPO_ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    """Return a currently-free TCP port for torchrun's rendezvous.

    Avoids hardcoded ports that ``EADDRINUSE`` when a prior run's process
    lingers or a port is in TIME_WAIT. (Small TOCTOU window between close and
    torchrun's bind, acceptable for a single-node test.)
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

# Audio sanity thresholds for the muxed sound track.
_RMS_SILENCE_FLOOR = 1e-4  # below this the track is effectively silence
_PEAK_SANITY_CEIL = 1.5    # decoded float audio should sit within ~[-1, 1]


def _run(cmd: list[str], log_file: Path) -> str:
    """Run ``cmd`` from the repo root, tee combined output to ``log_file``.

    Inherits the caller's environment (notably the HF cache, so a
    previously-downloaded Cosmos3-Nano is reused). Fails the test with the log
    tail on a non-zero exit.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = f".:{env.get('PYTHONPATH', '')}"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    returncode, text = _stream(cmd, env, log_file)
    if returncode != 0:
        pytest.fail(
            f"inference failed with exit code {returncode}:\n"
            f"  {' '.join(cmd)}\n"
            f"Log tail:\n{text[-3000:]}"
        )
    return text


def _stream(cmd: list[str], env: dict, log_file: Path) -> tuple[int, str]:
    """Run ``cmd`` and tee its combined output: live to stdout (so CI shows
    progress under ``pytest -s``) and into ``log_file`` + a returned string.
    """
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
    return returncode, "".join(captured)


def _decode_audio_track(mp4_path: Path):
    """Decode the muxed audio track of ``mp4_path`` to a (channels, samples) waveform.

    Returns ``(waveform_float64, sample_rate)``. Fails the test if the file has
    no audio stream or it decodes to zero frames.
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
    def test_nano_inference_t2vs(tmp_path: Path) -> None:
        """Run the docs/inference.md Cosmos3-Nano t2vs command; check the video + its sound."""
        out_dir = tmp_path / "out"
        cmd = [
            "torchrun",
            "--nproc_per_node=8",
            f"--master_port={_free_port()}",
            "-m",
            "cosmos_framework.scripts.inference",
            "--parallelism-preset=throughput",
            "-i",
            "inputs/omni/t2vs.json",
            "-o",
            str(out_dir),
            "--checkpoint-path",
            "Cosmos3-Nano",
            "--seed=0",
        ]
        _run(cmd, tmp_path / "inference.log")

        videos = list(out_dir.rglob("vision.mp4"))
        assert len(videos) == 1, f"expected exactly one vision.mp4 under {out_dir}, found {videos}"
        video = videos[0]
        assert video.stat().st_size > 0, f"empty output video at {video}"
        assert list(out_dir.rglob("sample_outputs.json")), f"no sample_outputs.json under {out_dir}"

        _assert_sound_not_noise(video)
