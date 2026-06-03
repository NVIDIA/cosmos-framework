# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""8-GPU smoke test for Cosmos3-Nano SFT training.

Runs the documented Vision SFT (Cosmos3-Nano) flow from ``docs/training.md``
end to end on 8 GPUs, capped to a single optimizer step via the
``vision_sft_nano_1iter`` recipe (``max_iter=1``, ``save_iter=1``):

  1. Step 1 -- download the bridge-v2 subset dataset + the Wan2.2 VAE.
  2. Step 2 -- ``convert_model_to_dcp`` the Cosmos3-Nano checkpoint to DCP.
  3. Step 3 -- run the paired launch shell ``launch_sft_vision_nano_1iter.sh``.

It asserts only that training completes and writes a checkpoint with a finite
loss (smoke -- no numeric goldens; that is ``launch_regression_test.py``'s job).

Inputs land in the documented, ``.gitignore``-d default locations
(``examples/data/``, ``examples/checkpoints/``) so they are cached across runs;
the training output goes under ``outputs/`` (also git-ignored). Steps 1-2 are
skipped when their artifacts already exist.

Invocation (inside the training container, from the repo root, on an 8-GPU
node)::

    pytest -s tests/nano_training_smoke_test.py --num-gpus=8 --levels=2 -o addopts=

Without ``--num-gpus``/``--levels`` (e.g. the no-GPU pre-commit CI) the test is
not collected.
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from cosmos_framework.inference.fixtures.args import MAX_GPUS

REPO_ROOT = Path(__file__).resolve().parents[1]

# Documented default locations (all git-ignored). Match the launcher defaults so
# Step 3 needs no path overrides.
_DATA_DIR = REPO_ROOT / "examples/data/bridge-v2-subset-synthetic-captions"
_DATASET_PATH = _DATA_DIR / "sft_dataset_bridge"
_DATASET_REVISION = "46468e12ac0dd36901e9e3240d4fc7620942b5d7"
_WAN_VAE = REPO_ROOT / "examples/checkpoints/wan22_vae/Wan2.2_VAE.pth"
_DCP_DIR = REPO_ROOT / "examples/checkpoints/Cosmos3-Nano"
_LAUNCHER = "tests/launch_sft_vision_nano_1iter.sh"

# Distinct from torchrun's default (29500) and the inference smoke port (29560).
_MASTER_PORT = 50112


def _run(cmd: list[str], log_file: Path, extra_env: dict | None = None) -> tuple[int, str]:
    """Run ``cmd`` from the repo root, tee combined output to ``log_file``.

    Returns ``(returncode, combined_output)``. Inherits the caller's env (HF
    cache, etc.) plus ``PYTHONPATH=.``.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = f".:{env.get('PYTHONPATH', '')}"
    if extra_env:
        env.update(extra_env)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    # Tee: stream the subprocess output live to stdout (so CI shows progress
    # under ``pytest -s``) while capturing it into the log file + a string.
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


def _ensure_inputs(log_dir: Path) -> None:
    """Step 1: download the dataset + Wan2.2 VAE if not already present."""
    if not (_DATASET_PATH / "train" / "video_dataset_file.jsonl").is_file():
        rc, out = _run(
            [
                "uvx", "hf@latest", "download", "--repo-type", "dataset",
                "nvidia/bridge-v2-subset-synthetic-captions",
                "--revision", _DATASET_REVISION,
                "--local-dir", str(_DATA_DIR), "--quiet",
            ],
            log_dir / "download_dataset.log",
        )
        assert rc == 0, f"dataset download failed (exit {rc}):\n{out[-2000:]}"
    assert (_DATASET_PATH / "train" / "video_dataset_file.jsonl").is_file(), (
        f"dataset missing {_DATASET_PATH}/train/video_dataset_file.jsonl after download"
    )

    if not _WAN_VAE.is_file():
        rc, out = _run(
            [
                "uvx", "hf@latest", "download", "Wan-AI/Wan2.2-TI2V-5B", "Wan2.2_VAE.pth",
                "--local-dir", str(_WAN_VAE.parent), "--quiet",
            ],
            log_dir / "download_wan_vae.log",
        )
        assert rc == 0, f"Wan VAE download failed (exit {rc}):\n{out[-2000:]}"
    assert _WAN_VAE.is_file(), f"Wan VAE missing at {_WAN_VAE} after download"


def _ensure_dcp(log_dir: Path) -> None:
    """Step 2: convert Cosmos3-Nano to DCP if not already present."""
    if _DCP_DIR.is_dir() and any(_DCP_DIR.iterdir()):
        return
    rc, out = _run(
        [
            "python", "-m", "cosmos_framework.scripts.convert_model_to_dcp",
            "--checkpoint-path", "Cosmos3-Nano",
            "-o", str(_DCP_DIR),
        ],
        log_dir / "convert_to_dcp.log",
    )
    assert rc == 0, f"convert_model_to_dcp failed (exit {rc}):\n{out[-3000:]}"
    assert _DCP_DIR.is_dir() and any(_DCP_DIR.iterdir()), f"DCP not written to {_DCP_DIR}"


def _finite_losses(text: str) -> list[float]:
    """Parse per-iteration ``Loss:`` values from the training log.

    Matches the ``iter_speed`` callback line, e.g.
    ``Iteration 1: Hit counter: 1/50 | Loss: 0.2302 | Time: ...``.
    """
    vals = []
    for m in re.finditer(r"Loss:\s*([-+0-9.eE]+)", text):
        try:
            v = float(m.group(1))
        except ValueError:
            continue
        if v == v and abs(v) != float("inf"):  # finite (NaN != NaN)
            vals.append(v)
    return vals


@pytest.fixture(scope="module", autouse=True)
def _require_8_gpus() -> None:
    """Skip the module unless we can launch an 8-GPU training run here."""
    if shutil.which("torchrun") is None:
        pytest.skip("torchrun not on PATH -- must run inside the training container")
    if shutil.which("uvx") is None:
        pytest.skip("uvx not on PATH -- required to download the dataset / Wan VAE")
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"torch unavailable ({exc!r})")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 8:
        pytest.skip(f"requires 8 visible CUDA devices, found {torch.cuda.device_count()}")


if MAX_GPUS == 8:

    @pytest.mark.level(2)
    @pytest.mark.gpus(8)
    def test_nano_sft_vision_1iter(tmp_path: Path) -> None:
        """Run the full Vision SFT (Cosmos3-Nano) 1-iter flow and check it trains a step."""
        _ensure_inputs(tmp_path)
        _ensure_dcp(tmp_path)

        # Route all run-specific output (launcher logs + the saved checkpoint via
        # the harness's IMAGINAIRE_OUTPUT_ROOT) under the pytest tmp dir, which
        # pytest auto-cleans. Nothing run-specific is left in the repo tree.
        rc, out = _run(
            ["bash", _LAUNCHER],
            tmp_path / "train.log",
            extra_env={
                "MASTER_PORT": str(_MASTER_PORT),
                "OUTPUT_ROOT": str(tmp_path / "launcher_out"),
                "NPROC_PER_NODE": "8",
            },
        )
        assert rc == 0, f"SFT launch failed (exit {rc}):\nLog tail:\n{out[-4000:]}"

        assert "Done with training" in out, f"training did not finish cleanly:\nLog tail:\n{out[-4000:]}"

        losses = _finite_losses(out)
        assert losses, f"no finite per-iteration 'Loss:' value found in training log:\n{out[-3000:]}"

        # save_iter=1 -> the trainer logs the DCP checkpoint path it wrote. Its
        # location is governed by IMAGINAIRE_OUTPUT_ROOT (the test harness points
        # this at a pytest tmp dir), so read it from the log rather than guessing.
        saved = re.findall(r"Saved checkpoint to (\S+)", out)
        assert saved, f"no 'Saved checkpoint to ...' line in training log (save_iter=1):\n{out[-3000:]}"
        ckpt = Path(saved[-1])
        assert ckpt.is_dir() and any(ckpt.iterdir()), f"saved checkpoint dir missing/empty: {ckpt}"
