# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""4-GPU inference smoke tests for the published Cosmos3 distilled models.

Runs the real public inference CLI against both pinned four-step checkpoints:

* ``Cosmos3-Super-Text2Image-4Step`` -> a non-degenerate JPEG.
* ``Cosmos3-Super-Image2Video-4Step`` -> a non-degenerate 121-frame MP4.

Both runs must select the checkpoint-defined FixedStep schedule. The test is
smoke-level output validation, not a numerical or visual-quality golden.

Invocation from the repository root on a node with at least four GPUs::

    TEST_MAX_GPUS=4 pytest -s tests/distilled_inference_smoke_test.py \
        --num-gpus=4 --levels=2 -o addopts=

Without ``--num-gpus`` and ``--levels`` the GPU cases are not selected.
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

_FIXED_STEP_LOG = "Using sampler: FixedStep (t_list=[1.0, 0.9375, 0.8333333333333334, 0.625, 0.0], sample_type=sde)"


def _free_port() -> int:
    """Return a free local port for torchrun rendezvous."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _run(cmd: list[str], log_file: Path) -> str:
    """Run a command from the repository root and tee its combined output."""
    env = os.environ.copy()
    env["PYTHONPATH"] = f".:{env.get('PYTHONPATH', '')}"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    captured: list[str] = []
    with log_file.open("w") as stream:
        proc = subprocess.Popen(
            cmd,
            env=env,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            stream.write(line)
            captured.append(line)
        returncode = proc.wait()
    text = "".join(captured)
    if returncode != 0:
        pytest.fail(f"inference failed with exit code {returncode}:\n  {' '.join(cmd)}\nLog tail:\n{text[-3000:]}")
    return text


def _run_distilled_inference(
    *,
    checkpoint: str,
    input_path: str,
    output_dir: Path,
    mode_args: tuple[str, ...],
    log_file: Path,
) -> str:
    """Launch one published distilled checkpoint on four GPUs."""
    cmd = [
        "torchrun",
        "--nproc_per_node=4",
        f"--master_port={_free_port()}",
        "-m",
        "cosmos_framework.scripts.inference",
        "--parallelism-preset=throughput",
        "-i",
        input_path,
        "-o",
        str(output_dir),
        "--checkpoint-path",
        checkpoint,
        "--dp-shard-size=4",
        "--dp-replicate-size=1",
        "--cp-size=1",
        "--cfgp-size=1",
        *mode_args,
        "--guidance=1.0",
        "--seed=1",
        "--no-guardrails",
    ]
    return _run(cmd, log_file)


def _single_sample_dir(output_dir: Path) -> Path:
    """Return the only generated sample directory."""
    results = sorted(output_dir.rglob("sample_outputs.json"))
    assert len(results) == 1, (
        f"expected one sample_outputs.json under {output_dir}, found {[str(path) for path in results]}"
    )
    data = json.loads(results[0].read_text())
    assert len(data.get("outputs", [])) == 1, f"unexpected sample output: {data}"
    return results[0].parent


def _assert_image_has_content(image_path: Path) -> None:
    """Require a decodable, finite, non-degenerate RGB image."""
    import numpy as np
    from PIL import Image

    assert image_path.is_file() and image_path.stat().st_size > 1024, f"image missing or too small: {image_path}"
    with Image.open(image_path) as image:
        image.load()
        rgb = np.asarray(image.convert("RGB"), dtype=np.float64)
    assert rgb.ndim == 3 and rgb.shape[2] == 3, f"unexpected image shape: {rgb.shape}"
    assert rgb.shape[0] > 0 and rgb.shape[1] > 0, f"empty image: {image_path}"
    assert np.all(np.isfinite(rgb)), f"non-finite pixels in {image_path}"
    assert rgb.std() > 3.0, f"degenerate image (pixel std={rgb.std():.3f}): {image_path}"


def _assert_video_has_content(video_path: Path, *, expected_frames: int) -> None:
    """Require a decodable video with spatial and temporal variation."""
    import av
    import numpy as np

    assert video_path.is_file() and video_path.stat().st_size > 1024, f"video missing or too small: {video_path}"
    with av.open(str(video_path)) as container:
        streams = container.streams.video
        assert streams, f"no video stream in {video_path}"
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(streams[0])]
    assert len(frames) == expected_frames, f"expected {expected_frames} frames in {video_path}, got {len(frames)}"
    sampled = np.stack((frames[0], frames[len(frames) // 2], frames[-1])).astype(np.float64)
    assert np.all(np.isfinite(sampled)), f"non-finite pixels in {video_path}"
    assert sampled.std() > 3.0, f"degenerate video (pixel std={sampled.std():.3f}): {video_path}"
    temporal_deltas = [float(np.mean(np.abs(sampled[index + 1] - sampled[index]))) for index in range(len(sampled) - 1)]
    assert max(temporal_deltas) > 0.5, f"near-static video (frame deltas={temporal_deltas}): {video_path}"


@pytest.fixture(scope="module", autouse=True)
def _require_4_gpus() -> None:
    """Skip the module unless four GPUs and torchrun are available."""
    if shutil.which("torchrun") is None:
        pytest.skip("torchrun not on PATH")
    try:
        import torch
    except Exception as exc:  # pragma: no cover - surfaces during development only
        pytest.skip(f"torch unavailable ({exc!r})")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
        pytest.skip(f"requires four visible GPUs, found {torch.cuda.device_count()}")


if MAX_GPUS == 4:

    @pytest.mark.level(2)
    @pytest.mark.gpus(4)
    def test_distilled_text2image_inference(tmp_path: Path) -> None:
        output_dir = tmp_path / "t2i"
        log = _run_distilled_inference(
            checkpoint="Cosmos3-Super-Text2Image-4Step",
            input_path="inputs/omni/t2i.json",
            output_dir=output_dir,
            mode_args=(
                "--resolution=768",
                "--aspect-ratio=1,1",
                "--fps=30",
                "--num-frames=1",
            ),
            log_file=tmp_path / "t2i.log",
        )

        assert _FIXED_STEP_LOG in log
        sample_dir = _single_sample_dir(output_dir)
        sample_args = json.loads((sample_dir / "sample_args.json").read_text())
        assert sample_args["model_mode"] == "text2image"
        assert sample_args["resolution"] == "768"
        assert sample_args["num_frames"] == 1
        _assert_image_has_content(sample_dir / "vision.jpg")

    @pytest.mark.level(2)
    @pytest.mark.gpus(4)
    def test_distilled_image2video_inference(tmp_path: Path) -> None:
        output_dir = tmp_path / "i2v"
        log = _run_distilled_inference(
            checkpoint="Cosmos3-Super-Image2Video-4Step",
            input_path="inputs/omni/i2v.json",
            output_dir=output_dir,
            mode_args=(
                "--resolution=480",
                "--aspect-ratio=4,3",
                "--fps=30",
                "--num-frames=121",
            ),
            log_file=tmp_path / "i2v.log",
        )

        assert _FIXED_STEP_LOG in log
        sample_dir = _single_sample_dir(output_dir)
        sample_args = json.loads((sample_dir / "sample_args.json").read_text())
        assert sample_args["model_mode"] == "image2video"
        assert sample_args["resolution"] == "480"
        assert sample_args["num_frames"] == 121
        _assert_video_has_content(sample_dir / "vision.mp4", expected_frames=121)
