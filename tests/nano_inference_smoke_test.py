# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""8-GPU multi-modality inference smoke test for Cosmos3-Nano.

Runs two ``cosmos_framework.scripts.inference`` calls and validates each output:

1. A ``throughput`` call over three input samples of different modalities (the
   ``-i`` flag takes a list of files):

  * ``inputs/omni/t2vs.json`` (text2video + sound) -> a ``vision.mp4`` whose
    muxed audio is real sound (finite, non-empty, non-silent, non-constant).
  * ``inputs/omni/action_forward_dynamics_camera.json`` (forward_dynamics) -> a
    ``vision.mp4`` that decodes to at least one valid video frame (``action_path``
    is an input, not an output).
  * ``inputs/omni/action_policy_robot.json`` (policy) -> BOTH a ``vision.mp4`` and
    a finite, non-empty predicted ``action`` array in ``sample_outputs.json``.

2. A separate ``latency`` call for a video2video transfer spec (``_TRANSFER_SPEC``,
   an edge control hint with ``control_guidance`` > 1.0, written to a temp file at
   run time rather than committed under ``inputs/``) -> a non-degenerate
   ``vision.mp4``. Exercises the transfer control-CFG path (the extra control-input
   forward driven by ``control_guidance``). Transfer needs the ``latency`` preset:
   under ``throughput`` (data-parallel over samples, FSDP-sharded) the extra
   control forward runs on only the transfer rank and deadlocks the cross-rank
   allgather, so it cannot share the call above — matching the cookbook's
   multi-GPU transfer recipe, which is also ``latency``.

All four samples produce a video; the policy sample additionally produces an
action, the t2vs sample an audio track, and the transfer sample exercises the
control-guidance branch.

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

# Transfer (video2video, edge control) input, written to a temp file at run time
# rather than committed under inputs/. Mirrors the cookbook
# ``cookbooks/cosmos3/generator/transfer/specs/edge.json`` behavior — the edge
# control hint with guidance=3.0 + control_guidance=1.5, which selects the
# control-CFG path — but downscaled (480p / 10 steps / single 29-frame chunk)
# for a fast smoke run. The control video is the exact same file the cookbook
# uses, pulled from the public NVIDIA/cosmos GitHub raw URL; the prompt is a
# compact caption of that clip (the dense cookbook caption is not needed to
# exercise the path).
_TRANSFER_CONTROL_URL = (
    "https://github.com/NVIDIA/cosmos/raw/main/"
    "cookbooks/cosmos3/generator/transfer/assets/edge/control_edge.mp4"
)
_TRANSFER_SPEC = {
    "name": "transfer_edge",
    "model_mode": "video2video",
    "resolution": "480",
    "aspect_ratio": "16,9",
    "num_frames": 29,
    "fps": 30,
    "shift": 10.0,
    "num_steps": 10,
    "seed": 2026,
    "num_video_frames_per_chunk": 29,
    "max_frames": 29,
    "num_conditional_frames": 1,
    "num_first_chunk_conditional_frames": 0,
    "share_vision_temporal_positions": True,
    "guidance": 3.0,
    "control_guidance": 1.5,
    "prompt": (
        "A woman with blonde hair in a low ponytail, wearing a black sleeveless top and black "
        "leggings, practices a dance routine in a brightly lit rehearsal studio with light wood "
        "floors, a large red-framed window, and a black curtain."
    ),
    "negative_prompt": "blurry, distorted, deformed, low quality, flickering, artifacts",
    "edge": {"control_path": _TRANSFER_CONTROL_URL, "preset_edge_threshold": "medium"},
}

# Multi-control transfer (video2video, edge + blur) input, written to a temp file
# at run time. Mirrors the cookbook
# ``cookbooks/cosmos3/generator/transfer/specs/multi_control.json`` — two control
# hints (edge + blur) computed on the fly from a single source video (``vision_path``)
# and blended by ``multi_control_two_way_attention`` (N independent maskless SDPA
# passes, one per control, summed by the per-hint ``weight``) — but downscaled
# (480p / 10 steps / single 29-frame chunk) for a fast smoke run. The source clip is
# the exact one the cookbook uses (a robot arm pouring into a glass), pinned to a
# public raw URL; the prompt is a compact caption of it. Unlike ``_TRANSFER_SPEC``
# (a single pre-computed ``control_path``), both controls here are derived on the
# fly, so this exercises the transfer control augmentor in addition to the weighted
# multi-control aggregation. ``guidance`` + ``control_guidance`` > 1.0 also keep the
# text-CFG and control-CFG branches active.
_MULTI_CONTROL_VISION_URL = (
    "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/"
    "2b17a2413bd86b2cf9b03823637108851e4ddf2d/inputs/vision/robot_pouring.mp4"
)
_MULTI_CONTROL_SPEC = {
    "name": "transfer_multi_control",
    "model_mode": "video2video",
    "resolution": "480",
    "aspect_ratio": "16,9",
    "num_frames": 29,
    "fps": 30,
    "shift": 10.0,
    "num_steps": 10,
    "seed": 2026,
    "num_video_frames_per_chunk": 29,
    "max_frames": 29,
    "num_conditional_frames": 1,
    "num_first_chunk_conditional_frames": 0,
    "share_vision_temporal_positions": True,
    "guidance": 3.0,
    "control_guidance": 1.5,
    "vision_path": _MULTI_CONTROL_VISION_URL,
    "prompt": (
        "A white robotic arm with black joints and cables carefully pours a clear liquid from a "
        "small light-green pitcher into a glass on a white tabletop, in a clean, brightly lit "
        "modern indoor setting."
    ),
    "negative_prompt": "blurry, distorted, deformed, low quality, flickering, artifacts",
    # Two hints, no control_path -> both derived on the fly from vision_path; the
    # per-hint weights drive the weighted multi-control attention aggregation.
    "edge": {"weight": 0.5, "preset_edge_threshold": "medium"},
    "blur": {"weight": 0.5, "preset_blur_strength": "medium"},
    "emphasize_control_in_prompt": False,
}

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


def _assert_video_has_content(mp4_path: Path, *, min_frames: int = 16) -> None:
    """Assert ``mp4_path`` decodes to enough non-degenerate frames.

    Stronger than ``_assert_valid_video`` (which only inspects the first frame):
    decodes the whole clip and checks the frame count plus real pixel variation,
    so a run that produced a well-formed container but collapsed to a constant /
    blank video (e.g. a broken control-CFG path) fails instead of passing.
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
    def test_nano_inference_omni(tmp_path: Path) -> None:
        """Throughput run over t2vs + policy + forward_dynamics, plus a separate latency transfer run."""
        # --- 1) Throughput run: t2vs + policy + forward_dynamics ----------------
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
        # the policy sample additionally yields an action and t2vs an audio track.
        assert n_video == len(_INPUTS), f"expected every sample to produce a valid video, got {n_video}/{len(_INPUTS)}"
        assert n_sound >= 1, f"expected the t2vs sample's audio to be checked, got {n_sound}"
        assert n_action >= 1, f"expected the policy sample's action to be checked, got {n_action}"

        # --- 2) Transfer run (separate, latency preset) -------------------------
        # Control-CFG (control_guidance > 1.0) runs an extra control-dropped forward
        # each step. Under the throughput preset (data-parallel over samples, FSDP-
        # sharded) that extra forward executes on only the transfer rank and
        # deadlocks the cross-rank allgather, so transfer cannot share the call
        # above; it needs the latency preset (context/CFG parallel -- every rank
        # runs the same sample together), matching the cookbook multi-GPU transfer
        # recipe. The spec is generated here (not committed under inputs/) and the
        # control video is pulled from the public NVIDIA/cosmos GitHub raw URL.
        # 4 ranks -> cfgp=2, cp=2 (the cookbook Cosmos3-Super transfer layout).
        transfer_spec = tmp_path / "transfer_edge.json"
        transfer_spec.write_text(json.dumps(_TRANSFER_SPEC))
        transfer_out = tmp_path / "out_transfer"
        transfer_cmd = [
            "torchrun",
            "--nproc_per_node=4",
            f"--master_port={_free_port()}",
            "-m",
            "cosmos_framework.scripts.inference",
            "--parallelism-preset=latency",
            "-i",
            str(transfer_spec),
            "-o",
            str(transfer_out),
            "--checkpoint-path",
            "Cosmos3-Nano",
            "--seed=0",
        ]
        _run(transfer_cmd, tmp_path / "inference_transfer.log")

        transfer_results = sorted(transfer_out.rglob("sample_outputs.json"))
        assert len(transfer_results) == 1, (
            f"expected 1 transfer sample_outputs.json, found {[str(p) for p in transfer_results]}"
        )
        so = transfer_results[0]
        args = json.loads(so.read_text()).get("args", {})
        # Transfer-specific input attributes: the edge control hint + the CFG knobs
        # that select the control-CFG path.
        edge = args.get("edge") or {}
        assert edge.get("control_path"), f"transfer sample missing edge control_path ({so}); args keys={list(args)}"
        assert args.get("control_guidance", 1.0) > 1.0, (
            f"expected control-CFG (control_guidance > 1.0), got {args.get('control_guidance')} ({so})"
        )
        assert (args.get("guidance") or 1.0) > 1.0, (
            f"expected text-CFG (guidance > 1.0), got {args.get('guidance')} ({so})"
        )
        # A valid, non-degenerate clip produced under control_guidance > 1.0 means the
        # control-CFG branch ran to completion: a broken postprocess would raise
        # mid-sampling, and a numerically broken one would collapse the output (caught
        # by _assert_video_has_content).
        transfer_video = so.parent / "vision.mp4"
        assert transfer_video.is_file(), f"transfer run produced no vision.mp4 ({so})"
        _assert_video_has_content(transfer_video)

    @pytest.mark.level(2)
    @pytest.mark.gpus(8)
    def test_nano_inference_multi_control_transfer(tmp_path: Path) -> None:
        """Multi-control transfer: edge + blur derived on the fly from ONE source
        video, blended by ``multi_control_two_way_attention``.

        Mirrors ``test_nano_inference_omni``'s single-control transfer run (same
        ``latency`` preset, 4 ranks -> cfgp=2, cp=2 -- the cookbook Cosmos3-Super
        transfer layout), but the generated spec sets TWO control hints (edge +
        blur) each with a per-hint ``weight`` and no ``control_path``, so both
        controls are computed on the fly from ``vision_path`` and aggregated by the
        weighted multi-control attention path (``multi_control_two_way_attention``:
        N maskless SDPA passes summed by weight). A non-degenerate clip confirms
        that path ran end to end -- a broken multi-control route would raise
        mid-sampling, and a numerically broken one would collapse the output
        (caught by ``_assert_video_has_content``). The on-the-fly derivation also
        exercises the transfer control augmentor (opencv), unlike the single-control
        run above which loads a pre-computed control_path."""
        spec_file = tmp_path / "transfer_multi_control.json"
        spec_file.write_text(json.dumps(_MULTI_CONTROL_SPEC))
        out_dir = tmp_path / "out_multi_control"
        cmd = [
            "torchrun",
            "--nproc_per_node=4",
            f"--master_port={_free_port()}",
            "-m",
            "cosmos_framework.scripts.inference",
            "--parallelism-preset=latency",
            "-i",
            str(spec_file),
            "-o",
            str(out_dir),
            "--checkpoint-path",
            "Cosmos3-Nano",
            "--seed=0",
        ]
        _run(cmd, tmp_path / "inference_multi_control.log")

        results = sorted(out_dir.rglob("sample_outputs.json"))
        assert len(results) == 1, (
            f"expected 1 multi-control sample_outputs.json, found {[str(p) for p in results]}"
        )
        so = results[0]
        args = json.loads(so.read_text()).get("args", {})
        # Multi-control-specific: BOTH edge and blur hints are active (2 controls ->
        # the weighted multi_control_two_way_attention path), each carries a weight,
        # and neither has a control_path (both derived on the fly from vision_path).
        edge = args.get("edge") or {}
        blur = args.get("blur") or {}
        assert edge and blur, f"expected both edge and blur hints active ({so}); edge={edge} blur={blur}"
        assert edge.get("weight") is not None and blur.get("weight") is not None, (
            f"expected a per-hint weight on both controls ({so}); edge={edge} blur={blur}"
        )
        assert not edge.get("control_path") and not blur.get("control_path"), (
            f"expected on-the-fly controls (no control_path) ({so}); edge={edge} blur={blur}"
        )
        assert args.get("vision_path"), f"multi-control run missing vision_path ({so})"
        assert args.get("control_guidance", 1.0) > 1.0, (
            f"expected control-CFG (control_guidance > 1.0), got {args.get('control_guidance')} ({so})"
        )
        assert (args.get("guidance") or 1.0) > 1.0, (
            f"expected text-CFG (guidance > 1.0), got {args.get('guidance')} ({so})"
        )
        video = so.parent / "vision.mp4"
        assert video.is_file(), f"multi-control run produced no vision.mp4 ({so})"
        _assert_video_has_content(video)
