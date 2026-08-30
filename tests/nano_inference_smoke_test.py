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

3. Two ``text2video`` calls against the ModelOpt static-FP8 Nano checkpoint, one
   per parallelism layout (FSDP-sharded and replicated) -> a non-degenerate
   ``vision.mp4`` each. Covers the FP8 checkpoint path end to end: detection,
   the meta-device linear swap, the TorchAO weight install, and — in the sharded
   layout — the FP8 all-gather. Skipped when the checkpoint is not reachable
   (see ``_download_fp8_checkpoint``).

4. One more ``text2video`` call against the same FP8 checkpoint with FP8
   mixed-precision diffusion steps enabled (``--mixed-precision-first-steps`` /
   ``--mixed-precision-last-steps``): the first/last N denoising steps run
   W8A16 (dequantized weight + dense GEMM) while middle steps keep the TorchAO
   W8A8 path. Asserts the exact per-step precision schedule from the
   ``MIXED_PRECISION_TRACE`` log line plus a non-degenerate ``vision.mp4``.

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
import re
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

# ModelOpt static-FP8 Cosmos3-Nano checkpoint. It is not published under its own
# repository yet, so it cannot be a ``--checkpoint-path`` registry name (see
# ``_CHECKPOINTS`` in ``cosmos_framework/inference/args.py``): it lives in a
# subdirectory of the access-controlled nvidia/Cosmos3-Experimental repo, pinned to
# the revision the FP8 loader was validated against. The test downloads that one
# subdirectory and passes the resulting local path to the CLI. Once the checkpoint
# is released under its own name, register it and drop ``_download_fp8_checkpoint``.
_FP8_REPOSITORY = "nvidia/Cosmos3-Experimental"
_FP8_REVISION = "f0cdb8ea37360e8510e2c0caf84c0f9f3e8751c8"
_FP8_SUBDIRECTORY = "cosmos3-nano-fp8-14072026"

# Emitted by ``swap_modelopt_fp8_linears_on_meta`` / ``install_torchao_float8_fsdp_support``
# (``cosmos_framework/utils/generator/quantization.py``). The swap count is parsed rather
# than string-matched: a checkpoint whose FP8 targets failed to resolve would swap zero
# linears, run the whole model in bf16, and otherwise produce a perfectly valid video.
_FP8_SWAP_LOG = re.compile(r"Swapped (\d+) linears to meta-device ModelOpt FP8 modules")
_FP8_FSDP_LOG = "Installed TorchAO static-FP8 FSDP support"

# Downscaled generation overrides shared by both FP8 layouts (480p / 29 frames /
# 10 steps, one chunk). The FP8 path under test is per-tensor weight quantization,
# which is independent of resolution and step count, so a short clip exercises it
# just as well as the checkpoint's 720p/189-frame defaults at a fraction of the
# runtime — the same trade-off ``_TRANSFER_SPEC`` above makes.
_FP8_GENERATION_ARGS = (
    "--resolution=480",
    "--aspect-ratio=16,9",
    "--fps=30",
    "--num-frames=29",
    "--num-steps=10",
    "--seed=0",
)

# One entry per parallelism layout. ``sharded`` is the layout that matters most
# here: FSDP2 all-gathers the FP8 weights through the TorchAO tensor-subclass
# hooks, which is the path that does not exist upstream, and it is the only
# layout a Super-class FP8 model fits in. ``replicated`` guards the
# single-device-weights path that worked before those hooks were added.
# Both layouts follow MAX_GPUS so the product of the parallel degrees equals
# WORLD_SIZE, which is what ParallelDims validates. `sharded` stays pure FSDP and
# `replicated` pure context-parallel -- the distinction each case exists to
# cover -- at whichever width the run has. cp_size is bounded by MAX_CP_SIZE=32,
# so 4 and 8 are both in range.
_FP8_LAYOUTS = {
    "sharded": (
        "--parallelism-preset=throughput",
        f"--dp-shard-size={MAX_GPUS}",
        "--dp-replicate-size=1",
        "--cp-size=1",
        "--cfgp-size=1",
    ),
    "replicated": (
        "--parallelism-preset=latency",
        "--dp-shard-size=1",
        "--dp-replicate-size=1",
        f"--cp-size={MAX_GPUS}",
        "--cfgp-size=1",
    ),
}

# Mixed-precision diffusion steps (FP8 W8A16 edge steps) schedule for the
# dedicated smoke case: with the 10-step ``_FP8_GENERATION_ARGS`` run this
# selects 2x W8A16 / 6x W8A8 / 2x W8A16 (``use_w8a16_step``). Small enough to
# stay cheap, large enough that first, middle, and last regions are all
# non-empty.
_MIXED_PRECISION_FIRST_STEPS = 2
_MIXED_PRECISION_LAST_STEPS = 2

# Emitted by ``MixedPrecisionRuntime`` (``cosmos_framework/utils/generator/
# mixed_precision.py``): the install summary at load time and the per-request
# per-step precision trace at request end. The trace is parsed and compared
# exactly — a run that silently ignored the flags would log an all-W8A8 trace
# (or none at all) and still produce a perfectly valid video.
_MIXED_PRECISION_INSTALL_LOG = "Mixed precision installed:"
_MIXED_PRECISION_TRACE_LOG = re.compile(r"MIXED_PRECISION_TRACE steps=([A-Za-z0-9,]+)")

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


def _download_fp8_checkpoint() -> Path:
    """Download the pinned ModelOpt FP8 Nano checkpoint and return its local root.

    Skips the test — rather than failing it — when the repository is unreachable
    for a credentials reason (no ``HF_TOKEN``, or a token without access to
    nvidia/Cosmos3-Experimental), so a fork PR without the runner secret does not
    go red. Any other failure (a deleted revision, a broken download) still fails
    loudly: a silently-skipping FP8 job would look green while testing nothing.

    ``REQUIRE_FP8=1`` promotes that skip to a hard failure. The CI job whose
    stated purpose includes FP8 coverage sets it, so a token rotation or a
    permissions change on the gated repo turns that job red instead of silently
    dropping every FP8 case while still reporting green.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

    def _skip_or_fail_no_access(reason: str) -> None:
        if os.environ.get("REQUIRE_FP8") == "1":
            pytest.fail(f"REQUIRE_FP8=1 but the FP8 checkpoint is unreachable: {reason}")
        pytest.skip(reason)

    try:
        repo_root = snapshot_download(
            repo_id=_FP8_REPOSITORY,
            revision=_FP8_REVISION,
            allow_patterns=[f"{_FP8_SUBDIRECTORY}/*"],
        )
    except (GatedRepoError, RepositoryNotFoundError) as error:
        _skip_or_fail_no_access(f"no access to {_FP8_REPOSITORY} (needs an HF_TOKEN with read access): {error!r}")
    except HfHubHTTPError as error:
        status_code = getattr(error.response, "status_code", None)
        if status_code in (401, 403):
            _skip_or_fail_no_access(f"no access to {_FP8_REPOSITORY} (HTTP {status_code}): {error!r}")
        raise

    checkpoint_path = Path(repo_root) / _FP8_SUBDIRECTORY
    assert (checkpoint_path / "hf_quant_config.json").is_file(), (
        f"{checkpoint_path} is not a ModelOpt FP8 checkpoint (no hf_quant_config.json); "
        f"revision {_FP8_REVISION} may have changed"
    )
    return checkpoint_path


@pytest.fixture(scope="module", autouse=True)
def _require_gpus() -> None:
    """Skip the module unless we can launch a ``MAX_GPUS``-wide run here."""
    if shutil.which("torchrun") is None:
        pytest.skip("torchrun not on PATH -- must run inside the inference container")
    try:
        import torch
    except Exception as exc:  # pragma: no cover -- surfaces during dev only
        pytest.skip(f"torch unavailable ({exc!r})")
    if not torch.cuda.is_available() or torch.cuda.device_count() < MAX_GPUS:
        pytest.skip(f"requires {MAX_GPUS} visible CUDA devices, found {torch.cuda.device_count()}")


# Markers use MAX_GPUS because the conftest rejects ``gpus(N)`` outside
# ``ALL_NUM_GPUS = (0, 1, MAX_GPUS)``. Every case here derives its parallelism
# from the active width, so both supported widths are listed.
if MAX_GPUS in (4, 8):

    @pytest.mark.level(2)
    @pytest.mark.gpus(MAX_GPUS)
    def test_nano_inference_omni(tmp_path: Path) -> None:
        """Throughput run over t2vs + policy + forward_dynamics, plus a separate latency transfer run."""
        # --- 1) Throughput run: t2vs + policy + forward_dynamics ----------------
        out_dir = tmp_path / "out"
        cmd = [
            "torchrun",
            f"--nproc_per_node={MAX_GPUS}",
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
    @pytest.mark.gpus(MAX_GPUS)
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

    @pytest.mark.level(2)
    @pytest.mark.gpus(MAX_GPUS)
    @pytest.mark.parametrize("layout", sorted(_FP8_LAYOUTS))
    def test_nano_fp8_inference(tmp_path: Path, layout: str) -> None:
        """text2video from the ModelOpt static-FP8 Nano checkpoint, once per layout.

        The FP8 checkpoint ships already-quantized E4M3 weights plus static
        per-tensor scales, so this run covers a path the bf16 cases above never
        touch: ``is_modelopt_fp8_checkpoint`` detection, the meta-device swap of the
        target linears to TorchAO FP8 modules (before FSDP wrap, so peak memory
        follows the FP8 shapes), the deferred weight install, and the FP8 forward.

        ``sharded`` additionally covers the FSDP2 path — the TorchAO static-FP8
        tensor upstream implements neither the all-gather hooks nor the shape ops
        FSDP2 needs, so without the local support shim the run dies on the first
        all-gather rather than producing a degraded video. Completing the run *is*
        the assertion there; ``_assert_video_has_content`` then catches the
        numerically-broken-but-still-running case (wrong scales -> collapsed clip).
        """
        checkpoint_path = _download_fp8_checkpoint()
        out_dir = tmp_path / f"out_fp8_{layout}"
        cmd = [
            "torchrun",
            f"--nproc_per_node={MAX_GPUS}",
            f"--master_port={_free_port()}",
            "-m",
            "cosmos_framework.scripts.inference",
            *_FP8_LAYOUTS[layout],
            "-i",
            "inputs/omni/t2v.json",
            "-o",
            str(out_dir),
            "--checkpoint-path",
            str(checkpoint_path),
            *_FP8_GENERATION_ARGS,
        ]
        log = _run(cmd, tmp_path / f"inference_fp8_{layout}.log")

        # The checkpoint was recognized as ModelOpt FP8 and its linears really were
        # swapped. Without the count check a checkpoint whose targets failed to
        # resolve would run entirely in bf16 and still pass every output assertion.
        swap_match = _FP8_SWAP_LOG.search(log)
        assert swap_match is not None, f"no ModelOpt FP8 linear swap in the {layout} run; FP8 path never engaged"
        assert int(swap_match.group(1)) > 0, f"ModelOpt FP8 swap matched 0 linears in the {layout} run"
        assert _FP8_FSDP_LOG in log, f"TorchAO static-FP8 FSDP support was not installed in the {layout} run"

        results = sorted(out_dir.rglob("sample_outputs.json"))
        assert len(results) == 1, f"expected 1 FP8 sample_outputs.json, found {[str(p) for p in results]}"
        so = results[0]
        args = json.loads(so.read_text()).get("args", {})
        assert args.get("model_mode") == "text2video", f"expected a text2video sample, got {args.get('model_mode')}"
        video = so.parent / "vision.mp4"
        assert video.is_file(), f"FP8 {layout} run produced no vision.mp4 ({so})"
        _assert_video_has_content(video)

    @pytest.mark.level(2)
    @pytest.mark.gpus(MAX_GPUS)
    def test_nano_fp8_mixed_precision_inference(tmp_path: Path) -> None:
        """text2video from the FP8 Nano checkpoint with mixed-precision diffusion steps.

        Same run as the ``sharded`` case of ``test_nano_fp8_inference`` plus the
        ``--mixed-precision-first-steps`` / ``--mixed-precision-last-steps`` flags,
        so the first/last 2 of the 10 denoising steps run W8A16 (dequantized E4M3
        weight + dense GEMM) while the middle 6 keep the TorchAO W8A8 path. The
        sharded layout is the one that constrains the feature: FSDP-sharded FP8
        weights support only the default ``mixed_precision_w8a16_cache='none'``
        (per-step on-the-fly dequant), which is exactly the mode exercised here.

        The pass criterion is the schedule itself, not just a valid video: the
        ``MIXED_PRECISION_TRACE`` line is parsed and compared exactly against the
        expected ``2x W8A16 / 6x W8A8 / 2x W8A16`` sequence, so a run where the
        flags never engaged (all-W8A8 trace, or no trace at all) fails even though
        its output video would look fine. ``_assert_video_has_content`` then
        catches the numerically-broken-but-still-running case.
        """
        checkpoint_path = _download_fp8_checkpoint()
        out_dir = tmp_path / "out_fp8_mixed_precision"
        cmd = [
            "torchrun",
            f"--nproc_per_node={MAX_GPUS}",
            f"--master_port={_free_port()}",
            "-m",
            "cosmos_framework.scripts.inference",
            *_FP8_LAYOUTS["sharded"],
            "-i",
            "inputs/omni/t2v.json",
            "-o",
            str(out_dir),
            "--checkpoint-path",
            str(checkpoint_path),
            *_FP8_GENERATION_ARGS,
            f"--mixed-precision-first-steps={_MIXED_PRECISION_FIRST_STEPS}",
            f"--mixed-precision-last-steps={_MIXED_PRECISION_LAST_STEPS}",
        ]
        log = _run(cmd, tmp_path / "inference_fp8_mixed_precision.log")

        # The FP8 path itself still engaged (same guard as test_nano_fp8_inference).
        swap_match = _FP8_SWAP_LOG.search(log)
        assert swap_match is not None and int(swap_match.group(1)) > 0, (
            "no ModelOpt FP8 linear swap in the mixed-precision run; FP8 path never engaged"
        )
        # ... and the mixed-precision runtime was installed on top of it.
        assert _MIXED_PRECISION_INSTALL_LOG in log, (
            "mixed precision was never installed despite --mixed-precision-first/last-steps"
        )

        # Exact per-step precision schedule. num_steps is read from
        # _FP8_GENERATION_ARGS so the expectation cannot drift from the run.
        (num_steps,) = [int(a.split("=")[1]) for a in _FP8_GENERATION_ARGS if a.startswith("--num-steps=")]
        expected = (
            ["W8A16"] * _MIXED_PRECISION_FIRST_STEPS
            + ["W8A8"] * (num_steps - _MIXED_PRECISION_FIRST_STEPS - _MIXED_PRECISION_LAST_STEPS)
            + ["W8A16"] * _MIXED_PRECISION_LAST_STEPS
        )
        traces = [m.split(",") for m in _MIXED_PRECISION_TRACE_LOG.findall(log)]
        assert expected in traces, (
            f"expected a MIXED_PRECISION_TRACE of {'/'.join(expected)}, got traces={traces}"
        )

        results = sorted(out_dir.rglob("sample_outputs.json"))
        assert len(results) == 1, f"expected 1 mixed-precision sample_outputs.json, found {[str(p) for p in results]}"
        so = results[0]
        args = json.loads(so.read_text()).get("args", {})
        assert args.get("model_mode") == "text2video", f"expected a text2video sample, got {args.get('model_mode')}"
        video = so.parent / "vision.mp4"
        assert video.is_file(), f"FP8 mixed-precision run produced no vision.mp4 ({so})"
        _assert_video_has_content(video)
