# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Self-contained regression test for the SFT smoke launch flow.

Re-runs the same ``torchrun`` invocation that ``launch_sft_llava_ov.sh``
executes (limited to 10 iterations, ``--deterministic`` mode) and asserts that
the rank-0 ``loss`` and global ``clip_grad_norm`` reproduce the inline goldens
at the bottom of this file. The launch goes through
``cosmos_framework.scripts.train --sft-toml=examples/toml/sft_config/<recipe>.toml``
— the only training entrypoint after the structured-TOML refactor.

Per-GPU goldens
---------------

Goldens are keyed by detected GPU architecture (``torch.cuda.get_device_name``):

* ``gb200`` — original values captured 2026-05-18 against the legacy
  ``cosmos_framework.scripts.train`` pipeline. The inputs and VLM backbone
  used at the time are not part of the OSS layout. The entries stay inline
  as a documented historical reference; don't re-run the GB200 path locally.
* ``h100`` — captured on 8× H100 (4-GPU subset). The VLM backbone is
  ``Qwen/Qwen3-VL-8B-Instruct``. Input paths come from env vars matching the
  names in ``docs/training.md``::

      MODEL_PATH            VLM backbone (Qwen/Qwen3-VL-8B-Instruct local snapshot)

  Use ``tests/_stage_h100_inputs.sh`` to download/convert this and emit an
  ``env.sh`` that ``source``s ``MODEL_PATH`` before invoking pytest.

This file is intentionally the only deliverable — the goldens are embedded as a
Python constant and the ``torchrun`` command line is reproduced here, so the
upstream launch shell stays untouched and there is no separate JSON file to
commit.

Invocation (on a 4-GPU node, inside the training container, from the repo
root)::

    pytest -s tests/launch_regression_test.py --num-gpus=4 --levels=2 -o addopts=

* ``--num-gpus=4 --levels=2`` matches the markers on the test below and lets
  the conftest's per-test setup pin ``CUDA_VISIBLE_DEVICES=0,1,2,3`` for
  torchrun. (``4`` is in ``ALL_NUM_GPUS`` in
  ``cosmos_framework/inference/fixtures/args.py``.)
* ``-o addopts=`` clears the ``addopts`` line in the repo's ``.pytest.toml``
  which references ``--suppress-no-test-exit-code`` from the optional
  ``pytest-custom-exit-code`` plugin (not installed in the training image).

Determinism notes:
  * ``llava_ov_datapacker`` runs **without** ``--deterministic`` on H100 AND
    overrides ``model.config.train.deterministic=false``: the Qwen3-VL text
    path uses an attention backend whose Hopper FMHA backward kernel has no
    deterministic mode (raises ``NotImplementedError`` under PyTorch's
    deterministic context). ``VLMModel.__init__`` honors the config-level
    flag via ``init_flash_attn_meta`` independently of the launcher arg, so
    both must be off. It also streams ``lmms-lab/LLaVA-OneVision-Data`` from
    HuggingFace Hub, so only the first 2 iters reproduce in practice (later
    iters drift with shard arrival order + non-det kernels). Set
    ``COSMOS_REGRESSION_VLM_FULL=1`` to assert all 10 (expected to fail).

Refreshing the goldens (after an intentional numerical change)::

    COSMOS_REGRESSION_UPDATE_GOLDENS=1 pytest -s launch_regression_test.py ...

That prints the captured series for each spec; copy them into the matching
``_GOLDENS[<arch>]`` entry below.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from cosmos_framework.inference.fixtures.args import MAX_GPUS

THIS_DIR = Path(__file__).resolve().parent
# ``cosmos_framework.scripts.train`` and the ``--sft-toml=...`` paths are relative to
# the repo root; we always invoke torchrun from there.
REPO_ROOT = THIS_DIR.parent

# --- per-arch input paths ----------------------------------------------------
#
# GB200: the original input snapshot lived on an internal read-only filesystem
# that is not in the OSS layout, so the GB200 path is not runnable here. The
# GB200 goldens dict is kept as a historical reference; ``_resolve_paths``
# below skips the GB200 arch instead of re-running it.


def _h100_paths_from_env() -> dict[str, str]:
    """Resolve H100 input paths from env vars (set by tests/_stage_h100_inputs.sh).

    All four env vars are required because the SFT TOMLs interpolate
    ``DATASET_PATH`` / ``WAN_VAE_PATH`` / ``BASE_CHECKPOINT_PATH`` at load time
    and the VLM spec passes ``MODEL_PATH`` as a Hydra backbone override.
    """
    missing = [
        var
        for var in ("DATASET_PATH", "WAN_VAE_PATH", "BASE_CHECKPOINT_PATH", "MODEL_PATH")
        if not os.environ.get(var)
    ]
    if missing:
        pytest.skip(
            f"H100 regression needs env vars: {missing}. "
            "Run tests/_stage_h100_inputs.sh and `source $STAGE_DIR/env.sh` first."
        )
    return {"vlm_model_path": os.environ["MODEL_PATH"]}


def _detect_arch() -> str:
    """Map ``torch.cuda.get_device_name(0)`` to a goldens key."""
    import torch  # local import keeps module import side-effects light

    if not torch.cuda.is_available():
        return "unknown"
    name = torch.cuda.get_device_name(0).upper()
    if "GB200" in name:
        return "gb200"
    if "H100" in name:
        return "h100"
    return "unknown"


def _resolve_paths(arch: str) -> dict[str, str]:
    if arch == "h100":
        return _h100_paths_from_env()
    if arch == "gb200":
        pytest.skip("gb200 inputs not in OSS layout; goldens kept for historical reference only.")
    pytest.skip(f"no regression goldens for GPU arch {arch!r}; only h100 supported")


# Tolerances for ``pytest.approx``. The launch passes ``--deterministic`` and
# ``PYTHONHASHSEED=42``; the tolerance only absorbs minor noise from
# non-deterministic NCCL reductions.
_DEFAULT_RTOL = 1e-3
_DEFAULT_ATOL = 1e-3

# --- log parsers -------------------------------------------------------------
#
# VLM (``pre_exp012_llava_ov_datapacker``) logs the DP-reduced loss on rank 0::
#
#     train/loss_avg: 1.32225 (iteration 0)
#
# ``GradClip`` emits the global grad-norm via every rank, prefixed with
# ``[RANK X]``. Key is ``clip_grad_norm/global`` for VLM.
_VLM_LOSS_RE = re.compile(r"train/loss_avg:\s+([0-9.eE+-]+)\s+\(iteration\s+\d+\)")
# VFM logs per-rank loss via the IterSpeed callback's on_training_step_end:
#     [RANK 0] Iteration 1: Hit counter: 1/50 | Loss: 0.2515 | Time: 120.42s
_VFM_LOSS_RE = re.compile(
    r"\[RANK\s+0\]\s+Iteration\s+\d+:\s+Hit counter:[^|]+\|\s+Loss:\s+([0-9.eE+-]+)"
)
_GRAD_NORM_RE = re.compile(
    r"\[RANK\s+0\][^\n]*clip_grad_norm/(?:[^/]+/)?global:\s+([0-9.eE+-]+)\s+\(iteration\s+\d+\)"
)


@dataclass(frozen=True)
class LaunchSpec:
    """A single launch flow under regression — mirrors the launcher shell."""

    key: str  # goldens key + pytest parametrize id source
    sft_toml: str  # ``--sft-toml=...`` value, relative to REPO_ROOT
    master_port: int
    extra_hydra_args: tuple[str, ...]
    loss_re: re.Pattern[str]
    deterministic_iters: int  # how many leading iters are bit-exact deterministic
    extra_env: dict[str, str] = field(default_factory=dict)
    nproc_per_node: int = 4
    # Some specs can't run under ``--deterministic`` on H100: the Qwen3-VL text
    # attention's Hopper FMHA backward kernel has no deterministic mode and
    # raises NotImplementedError. For those specs we drop the flag and accept
    # the tighter goldens tolerance only on the iters that still reproduce in
    # practice (see ``deterministic_iters``).
    deterministic: bool = True


# 4-GPU specs run by ``test_launch_regression``; 8-GPU specs run by
# ``test_launch_regression_8gpu`` (the ``gpus`` marker carries only one value,
# so the test functions are split).
_SPEC_KEYS = (
    "llava_ov_datapacker",
    "vision_sft_nano",
)
_SPEC_KEYS_8GPU = ("vision_sft_super",)


def _build_specs(paths: dict[str, str]) -> dict[str, LaunchSpec]:
    """Build the per-arch ``LaunchSpec`` list using the resolved input paths."""
    return {
        "llava_ov_datapacker": LaunchSpec(
            # Replicates launch_sft_llava_ov.sh, capped to 10 iters.
            key="llava_ov_datapacker",
            sft_toml="examples/toml/sft_config/llava_ov_datapacker.toml",
            master_port=50012,
            extra_hydra_args=(
                # TAIL_OVERRIDES from launch_sft_llava_ov.sh — fields not modeled
                # by SFTExperimentConfig.
                f"model.config.policy.backbone.model_name={paths['vlm_model_path']}",
                "data_setting.max_tokens=16000",
                # 4-GPU subset for the test (TOML pins dp_shard=8 for the 8-GPU
                # launch shell).
                "model.config.policy.parallelism.data_parallel_shard_degree=4",
                # The Qwen3-VL text path's Hopper FMHA backward (under the
                # ``cosmos`` attention impl) raises NotImplementedError under
                # PyTorch deterministic mode on H100. ``VLMModel.__init__``
                # honors ``train.deterministic`` (default ``True`` in the VLM
                # TrainConfig) by calling ``init_flash_attn_meta(True)`` which
                # flips ``torch.use_deterministic_algorithms(True)``, so
                # dropping the launcher's ``--deterministic`` flag isn't
                # enough — the config-level default must also be off.
                "model.config.train.deterministic=false",
                # Regression-specific tweaks.
                "trainer.max_iter=10",
                "trainer.logging_iter=1",
                "job.wandb_mode=disabled",
                "ckpt_type=dummy",
                "checkpoint.load_from_object_store.enabled=false",
                "checkpoint.save_to_object_store.enabled=false",
                "upload_reproducible_setup=false",
            ),
            loss_re=_VLM_LOSS_RE,
            # Only iter-0 loss reproduces under non-deterministic mode: it's a
            # pure forward on a seed-fixed batch with seed-fixed init weights,
            # so it's bit-exact. Iter 1+ depends on iter-0's non-deterministic
            # backward (no deterministic Hopper FMHA kernel on H100) and drifts
            # immediately.
            deterministic_iters=1,
            # See the ``train.deterministic=false`` override above for the
            # Hopper-FMHA rationale; the launcher flag is dropped to match.
            deterministic=False,
        ),
        "vision_sft_nano": LaunchSpec(
            # Replicates launch_sft_vision_nano.sh, capped to 10 iters.
            # ``DATASET_PATH`` / ``WAN_VAE_PATH`` / ``BASE_CHECKPOINT_PATH`` flow
            # in via the TOML's ``${oc.env:...}`` interpolation; no Hydra plumbing
            # needed beyond the regression-cap overrides below.
            key="vision_sft_nano",
            sft_toml="examples/toml/sft_config/vision_sft_nano.toml",
            master_port=50022,
            extra_hydra_args=(
                "model.config.parallelism.data_parallel_shard_degree=4",
                "model.config.parallelism.use_torch_compile=true",
                "trainer.max_iter=10",
                "trainer.logging_iter=1",
                "job.wandb_mode=disabled",
                "upload_reproducible_setup=false",
                "checkpoint.save_iter=999999",
            ),
            loss_re=_VFM_LOSS_RE,
            deterministic_iters=10,
        ),
        "vision_sft_super": LaunchSpec(
            # Replicates launch_sft_vision_super.sh on 8 GPUs (dp_shard=4 × cp=2),
            # capped to 10 iters. ``use_torch_compile=false`` because the Super
            # backbone's compile path is not bit-exact across runs on H100.
            key="vision_sft_super",
            sft_toml="examples/toml/sft_config/vision_sft_super.toml",
            master_port=50023,
            nproc_per_node=8,
            extra_hydra_args=(
                "model.config.parallelism.data_parallel_shard_degree=4",
                "model.config.parallelism.context_parallel_shard_degree=2",
                "model.config.parallelism.use_torch_compile=false",
                "trainer.max_iter=10",
                "trainer.logging_iter=1",
                "job.wandb_mode=disabled",
                "upload_reproducible_setup=false",
                "checkpoint.save_iter=999999",
            ),
            loss_re=_VFM_LOSS_RE,
            deterministic_iters=10,
        ),
    }


# --- helpers -----------------------------------------------------------------


def _parse_series(log_text: str, loss_re: re.Pattern[str]) -> tuple[list[float], list[float]]:
    """Extract per-iteration rank-0 loss and global grad-norm series, in order."""
    losses = [float(m.group(1)) for m in loss_re.finditer(log_text)]
    grad_norms = [float(m.group(1)) for m in _GRAD_NORM_RE.finditer(log_text)]
    assert losses and grad_norms, (
        f"No loss/grad-norm pairs found in log (losses={len(losses)}, grads={len(grad_norms)})"
    )
    assert len(losses) == len(grad_norms), (
        f"loss vs grad-norm length mismatch ({len(losses)} vs {len(grad_norms)}): "
        "the log must contain one rank-0 entry of each per training step."
    )
    return losses, grad_norms


def _run_torchrun(spec: LaunchSpec, run_dir: Path) -> Path:
    """Invoke the same ``torchrun`` command that the launcher shell runs.

    Returns the path of the captured combined stdout+stderr log.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "training.log"

    cmd = [
        "torchrun",
        f"--nproc_per_node={spec.nproc_per_node}",
        f"--master_port={spec.master_port}",
        "-m",
        "cosmos_framework.scripts.train",
        f"--sft-toml={spec.sft_toml}",
    ]
    if spec.deterministic:
        cmd.append("--deterministic")
    cmd += ["--", *spec.extra_hydra_args]

    env = os.environ.copy()
    # HF env mirrors what the launcher shell sets up; ``HF_TOKEN`` must already
    # be exported in the caller's environment if the experiment hits gated Hub
    # endpoints (e.g. the LLaVA-OneVision-Data streaming dataset).
    env.setdefault("HF_HOME", "/tmp/hf_cache")
    Path(env["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    env["PYTHONHASHSEED"] = "42"  # must be set before interpreter starts
    env["PYTHONPATH"] = f".:{env.get('PYTHONPATH', '')}"
    env["IMAGINAIRE_OUTPUT_ROOT"] = str(run_dir / "output")
    env.update(spec.extra_env)

    with log_file.open("w") as fp:
        result = subprocess.run(
            cmd,
            env=env,
            cwd=str(REPO_ROOT),
            stdout=fp,
            stderr=subprocess.STDOUT,
        )
    if result.returncode != 0:
        # Tolerate harmless PyGIL teardown warnings if training did complete.
        text = log_file.read_text(errors="replace")
        if "Done with training" not in text:
            pytest.fail(
                f"{spec.key}: torchrun failed with exit code {result.returncode} "
                "and log does not contain 'Done with training'.\n"
                f"Log tail:\n{text[-2000:]}"
            )
    return log_file


# --- fixtures ----------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _require_4_gpus() -> None:
    """Skip the whole module unless we can launch 4-GPU training here."""
    if shutil.which("torchrun") is None:
        pytest.skip("torchrun not on PATH — must run inside the training container")
    try:
        import torch
    except Exception as exc:  # pragma: no cover — surfaces during dev only
        pytest.skip(f"torch unavailable ({exc!r})")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
        pytest.skip(f"requires 4 visible CUDA devices, found {torch.cuda.device_count()}")


# --- tests -------------------------------------------------------------------


def _assert_spec_matches_goldens(spec_key: str, tmp_path: Path) -> None:
    """Re-run ``spec``'s torchrun command and check loss / grad-norm against goldens."""
    arch = _detect_arch()
    paths = _resolve_paths(arch)
    spec = _build_specs(paths)[spec_key]

    log_path = _run_torchrun(spec, tmp_path)
    loss, grad_norm = _parse_series(log_path.read_text(errors="replace"), spec.loss_re)
    assert len(loss) == 10, f"expected 10 iterations, parsed {len(loss)} (loss={loss})"

    # Refresh path: print captured values for manual copy into ``_GOLDENS``.
    if os.environ.get("COSMOS_REGRESSION_UPDATE_GOLDENS") == "1":
        print(f"\n# --- goldens for arch={arch!r} key={spec.key!r} ---")
        print(f'"{spec.key}": {{')
        print(f'    "loss": {loss},')
        print(f'    "grad_norm": {grad_norm},')
        print("},")
        pytest.skip(
            f"captured fresh series for arch={arch!r} key={spec.key!r}; copy the printed "
            f"dict into _GOLDENS[{arch!r}] at the bottom of launch_regression_test.py, "
            "then rerun without COSMOS_REGRESSION_UPDATE_GOLDENS to assert."
        )

    arch_goldens = _GOLDENS.get(arch)
    assert arch_goldens is not None, (
        f"no goldens table for arch {arch!r}; capture with COSMOS_REGRESSION_UPDATE_GOLDENS=1"
    )
    expected = arch_goldens.get(spec.key)
    assert expected is not None, (
        f"no goldens for arch={arch!r} key={spec.key!r}; capture with COSMOS_REGRESSION_UPDATE_GOLDENS=1"
    )

    n = spec.deterministic_iters
    if spec.key == "llava_ov_datapacker" and os.environ.get("COSMOS_REGRESSION_VLM_FULL") == "1":
        n = 10

    assert loss[:n] == pytest.approx(
        expected["loss"][:n], rel=_DEFAULT_RTOL, abs=_DEFAULT_ATOL
    ), f"{spec.key} ({arch}): rank-0 loss[:{n}] does not match goldens"
    # ``grad_norm`` is optional: ``None`` skips the check when the FSDP
    # global-norm all-reduce isn't bit-exact on this arch.
    if expected["grad_norm"] is None:
        return
    assert grad_norm[:n] == pytest.approx(
        expected["grad_norm"][:n], rel=_DEFAULT_RTOL, abs=_DEFAULT_ATOL
    ), f"{spec.key} ({arch}): global grad-norm[:{n}] does not match goldens"


# Define only the test function matching MAX_GPUS — the conftest rejects
# ``gpus(N)`` markers outside the active ``ALL_NUM_GPUS = (0, 1, MAX_GPUS)``.
if MAX_GPUS == 4:

    @pytest.mark.level(2)
    @pytest.mark.gpus(4)
    @pytest.mark.parametrize("spec_key", _SPEC_KEYS, ids=lambda k: k.removeprefix("launch_"))
    def test_launch_regression(spec_key: str, tmp_path: Path) -> None:
        """Re-run ``spec``'s torchrun command and check loss / grad-norm against goldens."""
        _assert_spec_matches_goldens(spec_key, tmp_path)


if MAX_GPUS == 8:

    @pytest.mark.level(2)
    @pytest.mark.gpus(8)
    @pytest.mark.parametrize(
        "spec_key", _SPEC_KEYS_8GPU, ids=lambda k: k.removeprefix("launch_")
    )
    def test_launch_regression_8gpu(spec_key: str, tmp_path: Path) -> None:
        """8-GPU variant for ``vision_sft_super`` (dp_shard=4 × cp=2)."""
        _assert_spec_matches_goldens(spec_key, tmp_path)


# Goldens keyed by GPU arch then ``LaunchSpec.key``. Refresh with
# ``COSMOS_REGRESSION_UPDATE_GOLDENS=1``.
_GOLDENS: dict[str, dict[str, dict[str, list[float] | None]]] = {
    # Captured 2026-05-18 on a 4 × NVIDIA GB200 node with ``--deterministic``
    # and seed 42 against the legacy training pipeline. VLM backbone is not
    # part of the OSS layout.
    "gb200": {
        "llava_ov_datapacker": {
            "loss": [1.32208, 1.20886, 1.39254, 1.40460, 1.16652, 1.24852, 1.38463, 1.22766, 0.96263, 1.14468],
            "grad_norm": [
                38.62454, 23.61477, 30.53218, 36.46255, 25.06240,
                39.70305, 48.52226, 52.18334, 22.77521, 25.06970,
            ],
        },
    },
    # Captured 2026-05-27 on a 4 × NVIDIA H100 80GB HBM3 node with seed 42.
    # Inputs come from ``tests/_stage_h100_inputs.sh``; VLM model is
    # ``Qwen/Qwen3-VL-8B-Instruct``.
    "h100": {
        # Recaptured 2026-05-27 with deterministic mode off (both ``--deterministic``
        # and ``model.config.train.deterministic`` are False — the Hopper FMHA
        # backward refuses to run under PyTorch deterministic mode on H100, see
        # ``LaunchSpec.deterministic`` and the spec's hydra override). The full
        # 10-iter series is captured for reference, but only ``deterministic_iters=1``
        # loss is asserted; iter 1+ drifts because the backward isn't bit-exact,
        # and even iter-0 grad-norm drifts (so grad_norm is skipped via ``None``).
        "llava_ov_datapacker": {
            "loss": [0.88798, 1.01436, 1.06162, 1.04558, 1.00519, 0.91837, 1.10527, 1.03337, 0.9421, 0.69604],
            "grad_norm": None,
        },
        # Recaptured 2026-05-27 after the TOML-config rewrite shifted some
        # defaults. Runs under ``--deterministic`` so loss reproduces bit-exact
        # across all 10 iters, but grad_norm is non-det because
        # ``use_torch_compile=true`` makes the all-rank reduction not bit-exact
        # on H100.
        "vision_sft_nano": {
            "loss": [0.2337, 0.2233, 0.2075, 0.2374, 0.2228, 0.2778, 0.2907, 0.223, 0.2125, 0.2699],
            "grad_norm": None,
        },
        "vision_sft_super": {
            "loss": [0.2133, 0.2028, 0.1992, 0.2373, 0.2539, 0.2645, 0.2679, 0.2182, 0.1959, 0.2457],
            "grad_norm": [0.00403, 0.00255, 0.00412, 0.00485, 0.00305, 0.00331, 0.00375, 0.00371, 0.00313, 0.00276],
        },
    },
}


if __name__ == "__main__":  # pragma: no cover — manual driver
    sys.exit(pytest.main([__file__, "-v", "-s", "-o", "addopts="]))
