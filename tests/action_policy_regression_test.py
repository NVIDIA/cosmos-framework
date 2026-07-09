# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Numerical-stability regression test for the ACTION-policy SFT launches (DROID + LIBERO).

The action-policy analogue of ``tests/launch_regression_test.py`` (vision/VLM):
re-runs the same ``torchrun`` command the launcher shells execute — capped to
10 iterations, ``--deterministic``, seed 42 — for each of the two action-policy
recipes and asserts that the rank-0 per-step ``Loss:`` series reproduces the
inline goldens at the bottom of this file (per GPU arch, with a tolerance).

Both recipes post-train the public **Cosmos3-Nano** base (registered experiments
``action_policy_{libero,droid}_nano``) — the same base + Wan2.2 VAE the vision
smoke/regression tests use — so a single 4-GPU node (``data_parallel_shard_degree=4``,
``replicate_degree=1``) runs them. The recipe knobs (optimizer, action-loss
weight, dataset transforms, iterable episode-shuffle) live in the experiment;
this test only caps iters + shapes the run for a deterministic single-node
capture.

Specs
-----
* ``action_policy_libero`` — ``examples/toml/sft_config/action_policy_libero_repro.toml``.
  Data auto-downloads: the ``libero_10`` suite of ``nvidia/LIBERO_LeRobot_v3``
  (small LeRobot dir), cached across runs. Collapses the recipe's HSDP replicate
  2 -> 1 to fit one 4-GPU node.
* ``action_policy_droid`` — ``examples/toml/sft_config/action_policy_droid_repro.toml``.
  The full DROID split is far too large to auto-download in CI, so this spec
  requires a pre-staged LOCAL copy via ``DROID_ROOT`` and SKIPS when unset.
  ``DROID_ROOT`` is the **versioned merged root** whose basename is a
  ``LEROBOT_ROOTS`` version key (e.g. ``…/droid_plus_lerobot_640x360_20260412``),
  NOT its ``success/`` subdir — the recipe's i4 lazy dataset appends ``success/``
  itself (``use_success_only``). The skip-check looks for
  ``<DROID_ROOT>/success/meta/info.json``.

Determinism notes
-----------------
The launch passes ``--deterministic`` + ``PYTHONHASHSEED=42`` and the recipes
seed the episode-shuffle stream (``episode_shuffle_seed=42``); the shard
assignment ``(rank, worker)`` and per-epoch permutation are seed-derived, so the
data order reproduces across runs for any fixed world size / ``num_workers``.
``compile_tokenizer`` is disabled for the capture (torch.compile makes the
all-rank grad-norm reduction non-bit-exact — same reason ``vision_sft_nano``
pins ``grad_norm=None`` on H100), so ``grad_norm`` goldens are left ``None`` and
only the loss series is asserted. On GB200 the sibling vision recipe reproduces
loss bit-exact across all 10 iters under ``--deterministic``; the action recipe
shares the MoT/attention stack, so the goldens use the tight default tolerance
(loosen via ``loss_tol_bands`` if a captured tail proves noisy).

Inputs land in the documented ``.gitignore``-d locations (``examples/data/``,
``examples/checkpoints/``), cached across runs and shared with the vision smoke
test; run output goes under the pytest tmp dir.

Invocation (inside the training container, from the repo root, on a 4-GPU
node)::

    # LIBERO only (auto-downloads its data):
    pytest -s tests/action_policy_regression_test.py --num-gpus=4 --levels=2 -o addopts=
    # include DROID (pre-stage the local LeRobot split):
    DROID_ROOT=/path/to/Cosmos3-DROID/success \
        pytest -s tests/action_policy_regression_test.py --num-gpus=4 --levels=2 -o addopts=

Without ``--num-gpus``/``--levels`` (e.g. the no-GPU pre-commit CI) the tests
are not collected.

Refreshing the goldens (after an intentional numerical change, on the target
arch)::

    COSMOS_ACTION_REGRESSION_UPDATE_GOLDENS=1 pytest -s tests/action_policy_regression_test.py \
        --num-gpus=4 --levels=2 -o addopts=

That prints the captured series for each spec; copy them into the matching
``_GOLDENS[<arch>]`` entry below.
"""

from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from cosmos_framework.inference.fixtures.args import MAX_GPUS

REPO_ROOT = Path(__file__).resolve().parents[1]

# Shared base artifacts — match the vision smoke test + the action launcher
# defaults so no path overrides are needed once present (all .gitignore-d,
# cached across runs).
_WAN_VAE = REPO_ROOT / "examples/checkpoints/wan22_vae/Wan2.2_VAE.pth"
_DCP_DIR = REPO_ROOT / "examples/checkpoints/Cosmos3-Nano"
# LIBERO-10 LeRobot suite (auto-downloaded). ``LIBERO_ROOT`` must point at the
# suite dir itself (contains meta/info.json).
_LIBERO_DIR = REPO_ROOT / "examples/data/LIBERO_LeRobot_v3"
_LIBERO_ROOT = _LIBERO_DIR / "libero_10"

# rank-0 per-iteration loss from the IterSpeed callback's on_training_step_end:
#   [RANK 0] Iteration 1: Hit counter: 1/50 | Loss: 0.2515 | Time: 120.42s
_VFM_LOSS_RE = re.compile(r"\[RANK\s+0\]\s+Iteration\s+\d+:\s+Hit counter:[^|]+\|\s+Loss:\s+([0-9.eE+-]+)")

_DEFAULT_RTOL = 1e-3
_DEFAULT_ATOL = 1e-3


def _free_port() -> int:
    """Return a currently-free TCP port for torchrun's rendezvous (avoids
    ``EADDRINUSE`` from a hardcoded ``master_port`` when a prior run lingers)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _detect_arch() -> str:
    """Map ``torch.cuda.get_device_name(0)`` to a goldens key."""
    import torch

    if not torch.cuda.is_available():
        return "unknown"
    name = torch.cuda.get_device_name(0).upper()
    if "GB200" in name:
        return "gb200"
    if "H100" in name or "H200" in name:
        return "h100"
    return "unknown"


def _run(cmd: list[str], log_file: Path, extra_env: dict | None = None) -> tuple[int, str]:
    """Run ``cmd`` from the repo root, tee combined output to ``log_file``.

    Streams live to stdout (so CI shows progress under ``pytest -s``) while
    capturing into the log + a string. Inherits the caller's env plus
    ``PYTHONPATH=.``.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = f".:{env.get('PYTHONPATH', '')}"
    if extra_env:
        env.update(extra_env)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    captured: list[str] = []
    with log_file.open("w") as fp:
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
            fp.write(line)
            captured.append(line)
        returncode = proc.wait()
    return returncode, "".join(captured)


# --- input staging (shared with tests/nano_training_smoke_test.py) -----------


def _ensure_wan_vae(log_dir: Path) -> None:
    """Download the Wan2.2 VAE if not already present."""
    if _WAN_VAE.is_file():
        return
    rc, out = _run(
        [
            "uvx",
            "hf@latest",
            "download",
            "Wan-AI/Wan2.2-TI2V-5B",
            "Wan2.2_VAE.pth",
            "--local-dir",
            str(_WAN_VAE.parent),
            "--quiet",
        ],
        log_dir / "download_wan_vae.log",
    )
    assert rc == 0, f"Wan VAE download failed (exit {rc}):\n{out[-2000:]}"
    assert _WAN_VAE.is_file(), f"Wan VAE missing at {_WAN_VAE} after download"


def _ensure_dcp(log_dir: Path) -> None:
    """Convert Cosmos3-Nano to DCP if not already present."""
    if _DCP_DIR.is_dir() and any(_DCP_DIR.iterdir()):
        return
    rc, out = _run(
        [
            "python",
            "-m",
            "cosmos_framework.scripts.convert_model_to_dcp",
            "--checkpoint-path",
            "Cosmos3-Nano",
            "-o",
            str(_DCP_DIR),
        ],
        log_dir / "convert_to_dcp.log",
    )
    assert rc == 0, f"convert_model_to_dcp failed (exit {rc}):\n{out[-3000:]}"
    assert _DCP_DIR.is_dir() and any(_DCP_DIR.iterdir()), f"DCP not written to {_DCP_DIR}"


def _ensure_libero(log_dir: Path) -> None:
    """Download the ``libero_10`` suite of ``nvidia/LIBERO_LeRobot_v3`` if absent."""
    if (_LIBERO_ROOT / "meta" / "info.json").is_file():
        return
    rc, out = _run(
        [
            "uvx",
            "hf@latest",
            "download",
            "--repo-type",
            "dataset",
            "nvidia/LIBERO_LeRobot_v3",
            "--include",
            "libero_10/**",
            "--local-dir",
            str(_LIBERO_DIR),
            "--quiet",
        ],
        log_dir / "download_libero.log",
    )
    assert rc == 0, f"LIBERO_LeRobot_v3 download failed (exit {rc}):\n{out[-2000:]}"
    assert (_LIBERO_ROOT / "meta" / "info.json").is_file(), (
        f"LIBERO suite missing {_LIBERO_ROOT}/meta/info.json after download"
    )


# --- launch specs ------------------------------------------------------------


# Overrides shared by both action specs: cap the run to a deterministic 10-iter
# single-node capture. Keeps ``model.config.compile.enabled`` at its recipe
# default (ON) to match launch_regression_test.py's nano spec — the loss is
# bit-exact under compile (only grad-norm is perturbed, which we don't assert).
# ``compile_tokenizer`` off just to skip its warmup. shard 4 x replicate 1 fits
# one 4-GPU node so the FSDP reduce-scatter/all-gather stays intra-node (NVLink)
# and reproduces bit-exact. A small packed batch keeps the 10 iters quick.
#
# NB (DROID on Blackwell): on gb200 the GradClip callback's compiled
# ``_fused_nan_to_num`` kernel can fail to launch through torch's static Triton
# launcher at this small shard=4 config ("CUDA driver error: invalid argument")
# — a launcher edge case, not a numeric issue (loss is identical eager). The
# H200/CI path this test targets does not hit it; the LIBERO spec is unaffected
# on either arch.
_COMMON_OVERRIDES: tuple[str, ...] = (
    "trainer.max_iter=10",
    "trainer.logging_iter=1",
    "trainer.seed=42",
    "job.wandb_mode=disabled",
    "upload_reproducible_setup=false",
    "checkpoint.save_iter=999999",  # no checkpoint writes during the capture
    "trainer.callbacks.compile_tokenizer.enabled=false",
    "model.config.parallelism.data_parallel_shard_degree=4",
    "model.config.parallelism.data_parallel_replicate_degree=1",
    "dataloader_train.max_samples_per_batch=8",
)


@dataclass(frozen=True)
class LaunchSpec:
    """A single action-policy launch flow under regression."""

    key: str
    sft_toml: str
    extra_hydra_args: tuple[str, ...]
    requires_droid_root: bool = False
    nproc_per_node: int = 4
    deterministic: bool = True
    loss_rtol: float = _DEFAULT_RTOL
    loss_atol: float = _DEFAULT_ATOL
    # Optional tiered tolerance: each ``(count, rtol, atol)`` applies to the next
    # ``count`` iters in order; counts must sum to 10. Empty => uniform default.
    loss_tol_bands: tuple[tuple[int, float, float], ...] = ()


_SPECS: dict[str, LaunchSpec] = {
    "action_policy_libero": LaunchSpec(
        key="action_policy_libero",
        sft_toml="examples/toml/sft_config/action_policy_libero_repro.toml",
        extra_hydra_args=_COMMON_OVERRIDES,
    ),
    "action_policy_droid": LaunchSpec(
        key="action_policy_droid",
        sft_toml="examples/toml/sft_config/action_policy_droid_repro.toml",
        extra_hydra_args=_COMMON_OVERRIDES,
        requires_droid_root=True,
    ),
}


def _run_torchrun(spec: LaunchSpec, run_dir: Path) -> str:
    """Invoke the same ``torchrun`` command the launcher shell runs; return the log text."""
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "torchrun",
        f"--nproc_per_node={spec.nproc_per_node}",
        f"--master_port={_free_port()}",
        "-m",
        "cosmos_framework.scripts.train",
        f"--sft-toml={spec.sft_toml}",
    ]
    if spec.deterministic:
        cmd.append("--deterministic")
    cmd += ["--", *spec.extra_hydra_args]

    rc, out = _run(
        cmd,
        run_dir / "training.log",
        extra_env={
            "PYTHONHASHSEED": "42",
            "IMAGINAIRE_OUTPUT_ROOT": str(run_dir / "output"),
            "WAN_VAE_PATH": str(_WAN_VAE),
            "BASE_CHECKPOINT_PATH": str(_DCP_DIR),
            "LIBERO_ROOT": str(_LIBERO_ROOT),
            # DROID_ROOT passes through from the caller's env (spec skips if unset).
        },
    )
    if rc != 0 and "Done with training" not in out:
        pytest.fail(
            f"{spec.key}: torchrun failed (exit {rc}) and log lacks 'Done with training'.\nLog tail:\n{out[-3000:]}"
        )
    return out


def _rank0_losses(text: str) -> list[float]:
    """Parse the rank-0 per-iteration ``Loss:`` series (one value per step), in order."""
    vals: list[float] = []
    for m in _VFM_LOSS_RE.finditer(text):
        v = float(m.group(1))
        vals.append(v)
    return vals


# --- fixtures ----------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _require_4_gpus() -> None:
    """Skip the module unless we can launch a 4-GPU training run here."""
    if shutil.which("torchrun") is None:
        pytest.skip("torchrun not on PATH — must run inside the training container")
    if shutil.which("uvx") is None:
        pytest.skip("uvx not on PATH — required to download the dataset / Wan VAE")
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"torch unavailable ({exc!r})")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
        pytest.skip(f"requires 4 visible CUDA devices, found {torch.cuda.device_count()}")


def _assert_spec_matches_goldens(spec_key: str, tmp_path: Path) -> None:
    """Re-run ``spec``'s torchrun command and check the loss series against goldens."""
    spec = _SPECS[spec_key]
    if spec.requires_droid_root:
        droid_root = os.environ.get("DROID_ROOT", "")
        # DROID_ROOT is the versioned merged root (basename is a LEROBOT_ROOTS key);
        # the i4 lazy dataset appends success/ (use_success_only), so meta/info.json
        # lives under <DROID_ROOT>/success, not at the root.
        if not droid_root or not (Path(droid_root) / "success" / "meta" / "info.json").is_file():
            pytest.skip(
                f"{spec.key}: set DROID_ROOT to a local versioned DROID LeRobot root "
                "(basename a LEROBOT_ROOTS key, e.g. droid_plus_lerobot_640x360_20260412, "
                "containing success/meta/info.json) to run this spec"
            )

    arch = _detect_arch()

    # Stage shared inputs (cached across runs); LIBERO data only for the libero spec.
    _ensure_wan_vae(tmp_path)
    _ensure_dcp(tmp_path)
    if not spec.requires_droid_root:
        _ensure_libero(tmp_path)

    out = _run_torchrun(spec, tmp_path)
    assert "Done with training" in out, f"{spec.key}: training did not finish:\n{out[-3000:]}"
    losses = _rank0_losses(out)
    run_detail = f"\n--- {spec.key} run log (last 3000 chars) ---\n{out[-3000:]}"
    assert len(losses) == 10, f"{spec.key}: expected 10 rank-0 losses, parsed {losses}{run_detail}"
    assert all(v == v and abs(v) != float("inf") for v in losses), (
        f"{spec.key}: non-finite loss in series {losses}{run_detail}"
    )

    # Refresh path: print captured values for manual copy into ``_GOLDENS``.
    if os.environ.get("COSMOS_ACTION_REGRESSION_UPDATE_GOLDENS") == "1":
        print(f"\n# --- goldens for arch={arch!r} key={spec.key!r} ---")
        print(f'"{spec.key}": {{"loss": {losses}}},')
        pytest.skip(
            f"captured fresh loss series for arch={arch!r} key={spec.key!r}; copy the printed "
            f"dict into _GOLDENS[{arch!r}] at the bottom of action_policy_regression_test.py, "
            "then rerun without COSMOS_ACTION_REGRESSION_UPDATE_GOLDENS to assert."
        )

    arch_goldens = _GOLDENS.get(arch)
    if not arch_goldens or spec.key not in arch_goldens:
        pytest.skip(
            f"no goldens for arch={arch!r} key={spec.key!r} yet — capture on this hardware with "
            f"COSMOS_ACTION_REGRESSION_UPDATE_GOLDENS=1 (parsed {len(losses)} finite losses this run)"
        )
    expected = arch_goldens[spec.key]["loss"]

    bands = spec.loss_tol_bands or ((10, spec.loss_rtol, spec.loss_atol),)
    assert sum(c for c, _, _ in bands) == 10, f"{spec.key}: loss_tol_bands must sum to 10"
    start = 0
    for count, rtol, atol in bands:
        end = start + count
        assert losses[start:end] == pytest.approx(expected[start:end], rel=rtol, abs=atol), (
            f"{spec.key} ({arch}): rank-0 loss[{start}:{end}] (rel={rtol}, abs={atol}) "
            f"does not match goldens\n  got     : {losses[start:end]}\n"
            f"  expected: {expected[start:end]}{run_detail}"
        )
        start = end


# --- tests -------------------------------------------------------------------


if MAX_GPUS == 4:

    @pytest.mark.level(2)
    @pytest.mark.gpus(4)
    @pytest.mark.parametrize("spec_key", list(_SPECS), ids=list(_SPECS))
    def test_action_policy_regression(spec_key: str, tmp_path: Path) -> None:
        """Deterministic 10-iter action-policy SFT; assert the rank-0 loss goldens."""
        _assert_spec_matches_goldens(spec_key, tmp_path)


# Goldens keyed by GPU arch then ``LaunchSpec.key``. ``loss`` is the rank-0
# per-step series over the 10 deterministic iters. Refresh on the target arch
# with ``COSMOS_ACTION_REGRESSION_UPDATE_GOLDENS=1`` (see the module docstring).
# The test skips (not fails) for any arch/spec without an entry, so goldens can
# land incrementally as they are captured on each arch.
#
# Captured with torch.compile ON, --deterministic, seed 42, single-node
# data_parallel_shard_degree=4 (intra-node NVLink FSDP reduction is bit-exact).
_GOLDENS: dict[str, dict[str, dict[str, list[float]]]] = {
    # H200 (Hopper) CI arch. LIBERO is the primary numerical golden; the DROID
    # spec needs its dataset (DROID_ROOT), so it skips unless one is provided.
    "h100": {
        "action_policy_libero": {
            "loss": [
                15.8107,
                15.2467,
                15.9856,
                16.5306,
                14.3738,
                16.1460,
                16.6093,
                14.8846,
                16.0632,
                16.6449,
            ],
        },
    },
}


if __name__ == "__main__":  # pragma: no cover — manual driver
    sys.exit(pytest.main([__file__, "-v", "-s", "-o", "addopts="]))
