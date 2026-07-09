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
  * ``llava_ov`` runs **without** ``--deterministic`` on H100 AND
    overrides ``model.config.deterministic=false``: the Qwen3-VL text
    path uses an attention backend whose Hopper FMHA backward kernel has no
    deterministic mode (raises ``NotImplementedError`` under PyTorch's
    deterministic context). ``VLMModel.__init__`` honors the config-level
    flag via ``init_flash_attn_meta`` independently of the launcher arg, so
    both must be off. It streams ``lmms-lab/LLaVA-OneVision-Data`` from the
    HuggingFace Hub with ``dataloader_train.num_workers=0`` so the data order is
    fully deterministic (single process); the only run-to-run noise left is the
    FMHA backward kernel. iter-0 is bit-exact (forward only) but iters 1+ drift
    (the Hopper FMHA backward has no deterministic mode — confirmed: forcing
    ``deterministic=true`` raises ``NotImplementedError``, and a 2-run
    ``num_workers=0`` check still drifts ≤0.006 on iters 1-9). All 10 iters are
    asserted with a tiered tolerance (``loss_tol_bands``): iter-0 at
    1e-3, iters 1-2 at 1e-2, iters 3-9 at 2e-2.

Refreshing the goldens (after an intentional numerical change)::

    COSMOS_REGRESSION_UPDATE_GOLDENS=1 pytest -s launch_regression_test.py ...

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
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from cosmos_framework.inference.fixtures.args import MAX_GPUS

THIS_DIR = Path(__file__).resolve().parent
# ``cosmos_framework.scripts.train`` and the ``--sft-toml=...`` paths are relative to
# the repo root; we always invoke torchrun from there.
REPO_ROOT = THIS_DIR.parent


def _free_port() -> int:
    """Return a currently-free TCP port for torchrun's rendezvous, instead of a
    hardcoded ``master_port`` that ``EADDRINUSE``s when a prior run lingers."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

# --- per-arch input paths ----------------------------------------------------
#
# GB200: the original input snapshot lived on an internal read-only filesystem
# that is not in the OSS layout, so the GB200 path is not runnable here. The
# GB200 goldens dict is kept as a historical reference; ``_resolve_paths``
# below skips the GB200 arch instead of re-running it.


def _hf_download(args: list[str]) -> str:
    """``uvx hf download <args> --quiet`` -> the local path it prints (from the HF cache)."""
    result = subprocess.run(
        ["uvx", "hf@latest", "download", *args, "--quiet"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"hf download failed for {args} (exit {result.returncode}):\n{result.stdout}\n{result.stderr}")
    lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
    if not lines:
        pytest.fail(f"hf download for {args} printed no path:\n{result.stdout}\n{result.stderr}")
    return lines[-1]


def _convert_nano_dcp(dest: Path) -> None:
    """Convert the Cosmos3-Nano checkpoint to DCP at ``dest`` (Step 2 of docs/training.md)."""
    env = os.environ.copy()
    env["PYTHONPATH"] = f".:{env.get('PYTHONPATH', '')}"
    result = subprocess.run(
        [
            sys.executable, "-m", "cosmos_framework.scripts.convert_model_to_dcp",
            "-o", str(dest), "--checkpoint-path", "Cosmos3-Nano",
        ],
        cwd=str(REPO_ROOT),
        env=env,
    )
    if result.returncode != 0:
        pytest.fail(f"convert_model_to_dcp (Cosmos3-Nano) failed with exit code {result.returncode}")


def _detect_arch() -> str:
    """Map ``torch.cuda.get_device_name(0)`` to a goldens key."""
    import torch  # local import keeps module import side-effects light

    if not torch.cuda.is_available():
        return "unknown"
    name = torch.cuda.get_device_name(0).upper()
    if "GB200" in name:
        return "gb200"
    # H200 shares the Hopper kernels with H100 and is treated identically here:
    # both map to the ``h100`` goldens key (the GitHub GPU CI runs on 8×H200).
    if "H100" in name or "H200" in name:
        return "h100"
    return "unknown"


# Pinned revisions mirror tests/_stage_h100_inputs.sh so prepared inputs match
# the captured h100 goldens.
_BRIDGE_REVISION = "46468e12ac0dd36901e9e3240d4fc7620942b5d7"
_QWEN_VL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"


# Tolerances for ``pytest.approx``. The launch passes ``--deterministic`` and
# ``PYTHONHASHSEED=42``; the tolerance only absorbs minor noise from
# non-deterministic NCCL reductions.
_DEFAULT_RTOL = 1e-3
_DEFAULT_ATOL = 1e-3

# --- log parsers -------------------------------------------------------------
#
# VLM (``pre_exp012_llava_ov``) logs the DP-reduced loss on rank 0::
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
    # Per-spec goldens tolerance for ``pytest.approx``. Deterministic specs use
    # the tight default uniformly across all asserted iters.
    loss_rtol: float = _DEFAULT_RTOL
    loss_atol: float = _DEFAULT_ATOL
    # Optional tiered tolerance: each ``(count, rtol, atol)`` applies to the next
    # ``count`` iters in order, and the counts must sum to ``deterministic_iters``.
    # Lets the reasoner tighten its bit-exact iter-0 while loosening the
    # non-deterministic tail. When empty, all iters use ``loss_rtol/loss_atol``.
    loss_tol_bands: tuple[tuple[int, float, float], ...] = ()


# 4-GPU specs run by ``test_launch_regression``; 8-GPU specs run by
# ``test_launch_regression_8gpu`` (the ``gpus`` marker carries only one value,
# so the test functions are split).
_SPEC_KEYS = (
    "llava_ov",
    "vision_sft_nano",
)
_SPEC_KEYS_8GPU = ("vision_sft_super",)


def _build_specs(paths: dict[str, str]) -> dict[str, LaunchSpec]:
    """Build the per-arch ``LaunchSpec`` list using the resolved input paths."""
    # vision_sft_super needs a Cosmos3-Super DCP; the default staging script
    # only produces Cosmos3-Nano. If BASE_CHECKPOINT_PATH_SUPER is set,
    # redirect BASE_CHECKPOINT_PATH for this spec via extra_env.
    super_extra_env: dict[str, str] = {}
    if super_ckpt := os.environ.get("BASE_CHECKPOINT_PATH_SUPER"):
        super_extra_env["BASE_CHECKPOINT_PATH"] = super_ckpt

    return {
        "llava_ov": LaunchSpec(
            # Replicates launch_sft_llava_ov.sh, capped to 10 iters.
            key="llava_ov",
            sft_toml="examples/toml/sft_config/llava_ov.toml",
            extra_hydra_args=(
                # TAIL_OVERRIDES from launch_sft_llava_ov.sh — fields not modeled
                # by SFTExperimentConfig.
                f"model.config.policy.backbone.model_name={paths['vlm_model_path']}",
                "data_setting.max_tokens=16000",
                # 4-GPU subset for the test (TOML pins dp_shard=8 for the 8-GPU
                # launch shell).
                "model.config.parallelism.data_parallel_shard_degree=4",
                # The Hopper FMHA backward raises under PyTorch
                # deterministic mode, so both the config default and the
                # launcher's --deterministic flag must be off (see the
                # determinism notes in the module docstring).
                "model.config.deterministic=false",
                # num_workers=0: fully-ordered single-process streaming, so the
                # only run-to-run noise is the FMHA backward kernel, not data
                # order. prefetch_factor/persistent_workers must be unset for
                # num_workers=0 (torch DataLoader rejects them otherwise).
                "dataloader_train.num_workers=0",
                "dataloader_train.prefetch_factor=null",
                "dataloader_train.persistent_workers=false",
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
            deterministic_iters=10,
            deterministic=False,
            # Tiered tolerance: iter-0 is bit-exact (forward only) → 1e-3; iters
            # 1+ carry the FMHA-backward noise (≤0.006 across two num_workers=0
            # runs) → 1e-2 for the early iters 1-2, 2e-2 for the tail 3-9.
            loss_tol_bands=(
                (1, 1e-3, 1e-3),  # iter 0
                (2, 1e-2, 1e-2),  # iters 1-2
                (7, 2e-2, 2e-2),  # iters 3-9
            ),
        ),
        "vision_sft_nano": LaunchSpec(
            # Replicates launch_sft_vision_nano.sh, capped to 10 iters.
            # ``DATASET_PATH`` / ``WAN_VAE_PATH`` / ``BASE_CHECKPOINT_PATH`` flow
            # in via the TOML's ``${oc.env:...}`` interpolation; no Hydra plumbing
            # needed beyond the regression-cap overrides below.
            key="vision_sft_nano",
            sft_toml="examples/toml/sft_config/vision_sft_nano.toml",
            extra_hydra_args=(
                "model.config.parallelism.data_parallel_shard_degree=4",
                "model.config.compile.enabled=true",
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
            # capped to 10 iters. ``compile.enabled=false`` because the Super
            # backbone's compile path is not bit-exact across runs on H100.
            key="vision_sft_super",
            sft_toml="examples/toml/sft_config/vision_sft_super.toml",
            nproc_per_node=8,
            extra_hydra_args=(
                "model.config.parallelism.data_parallel_shard_degree=4",
                "model.config.parallelism.context_parallel_shard_degree=2",
                "model.config.compile.enabled=false",
                "trainer.max_iter=10",
                "trainer.logging_iter=1",
                "job.wandb_mode=disabled",
                "upload_reproducible_setup=false",
                "checkpoint.save_iter=999999",
            ),
            loss_re=_VFM_LOSS_RE,
            deterministic_iters=10,
            extra_env=super_extra_env,
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
        f"--master_port={_free_port()}",
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

    # Tee: stream the torchrun output live to stdout (so CI shows training
    # progress under ``pytest -s``) while capturing it into the log file.
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
        returncode = proc.wait()
    if returncode != 0:
        # Tolerate harmless PyGIL teardown warnings if training did complete.
        text = log_file.read_text(errors="replace")
        if "Done with training" not in text:
            pytest.fail(
                f"{spec.key}: torchrun failed with exit code {returncode} "
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


@pytest.fixture(scope="module")
def h100_inputs(tmp_path_factory: pytest.TempPathFactory):
    """Provide the regression input paths, preparing any not already set in env.

    Mirrors the download/convert steps of ``tests/_stage_h100_inputs.sh`` (it
    does NOT set up the environment -- ``uv sync`` and the ``transformers``
    pin still belong to that script / the caller). Honors pre-set env vars (so
    ``source env.sh`` still works); anything prepared here goes under a temp
    stage dir that is removed on teardown. The four vars are exported because
    the SFT TOMLs interpolate ``DATASET_PATH`` / ``WAN_VAE_PATH`` /
    ``BASE_CHECKPOINT_PATH`` at load time and the VLM spec passes ``MODEL_PATH``
    as a Hydra backbone override.
    """
    arch = _detect_arch()
    if arch not in ("h100", "gb200"):
        pytest.skip(f"no regression goldens for GPU arch {arch!r}; only h100/gb200 supported")
    if shutil.which("uvx") is None:
        pytest.skip("uvx not on PATH -- required to prepare regression inputs")

    stage = tmp_path_factory.mktemp("h100_stage")
    set_vars: list[str] = []

    def _ensure(var: str, value_fn) -> None:
        if not os.environ.get(var):
            os.environ[var] = str(value_fn())
            set_vars.append(var)

    _ensure(
        "DATASET_PATH",
        lambda: Path(
            _hf_download(
                ["--repo-type", "dataset", "nvidia/bridge-v2-subset-synthetic-captions",
                 "--revision", _BRIDGE_REVISION]
            )
        ) / "sft_dataset_bridge",
    )
    _ensure("WAN_VAE_PATH", lambda: _hf_download(["Wan-AI/Wan2.2-TI2V-5B", "Wan2.2_VAE.pth"]))
    _ensure("MODEL_PATH", lambda: _hf_download(["Qwen/Qwen3-VL-8B-Instruct", "--revision", _QWEN_VL_REVISION]))

    def _make_dcp() -> Path:
        dest = stage / "Cosmos3-Nano-DCP"
        _convert_nano_dcp(dest)
        return dest

    _ensure("BASE_CHECKPOINT_PATH", _make_dcp)

    try:
        yield {"vlm_model_path": os.environ.get("MODEL_PATH", "")}
    finally:
        for var in set_vars:
            os.environ.pop(var, None)
        shutil.rmtree(stage, ignore_errors=True)


@pytest.fixture(scope="module")
def qwen_vl_model_path() -> str:
    """Local Qwen3-VL-8B-Instruct snapshot for the convert-reasoner test.

    Honors a pre-set ``MODEL_PATH`` (so ``source env.sh`` / the ``h100_inputs``
    staging is reused); otherwise downloads the pinned revision from the Hub.
    Independent of ``h100_inputs`` so this test does not depend on the Nano→DCP
    conversion that ``h100_inputs`` also performs for the training specs.
    """
    if shutil.which("uvx") is None:
        pytest.skip("uvx not on PATH -- required to stage Qwen3-VL")
    return os.environ.get("MODEL_PATH") or _hf_download(
        ["Qwen/Qwen3-VL-8B-Instruct", "--revision", _QWEN_VL_REVISION]
    )


def _weight_map(directory: Path) -> dict[str, str]:
    """``key -> shard filename`` for a (sharded or single-file) safetensors dir."""
    index = directory / "model.safetensors.index.json"
    if index.exists():
        import json

        return json.loads(index.read_text())["weight_map"]
    return {}  # single-file: callers fall back to model.safetensors


def _load_st_tensor(directory: Path, key: str, weight_map: dict[str, str]):
    """Lazily read a single tensor by key (keeps peak memory to one tensor)."""
    from safetensors import safe_open

    rel = weight_map.get(key, "model.safetensors")
    with safe_open(str(directory / rel), framework="pt") as f:
        return f.get_tensor(key)


# --- tests -------------------------------------------------------------------


def _assert_spec_matches_goldens(spec_key: str, tmp_path: Path, paths: dict[str, str]) -> None:
    """Re-run ``spec``'s torchrun command and check loss / grad-norm against goldens."""
    arch = _detect_arch()
    spec = _build_specs(paths)[spec_key]

    log_path = _run_torchrun(spec, tmp_path)
    log_text = log_path.read_text(errors="replace")
    loss, grad_norm = _parse_series(log_text, spec.loss_re)
    # The run log also streamed live under ``pytest -s``; include its tail in any
    # failure message so the run detail is attached to the failure report too.
    run_detail = f"\n--- {spec.key} run log (last 4000 chars) ---\n{log_text[-4000:]}"
    assert len(loss) == 10, f"expected 10 iterations, parsed {len(loss)} (loss={loss}){run_detail}"

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

    # Build the per-iter tolerance segments: either the spec's tiered bands or a
    # single uniform band spanning all asserted iters.
    if spec.loss_tol_bands:
        assert sum(c for c, _, _ in spec.loss_tol_bands) == n, (
            f"{spec.key}: loss_tol_bands counts {[c for c, _, _ in spec.loss_tol_bands]} "
            f"must sum to deterministic_iters={n}"
        )
        bands = spec.loss_tol_bands
    else:
        bands = ((n, spec.loss_rtol, spec.loss_atol),)

    start = 0
    for count, rtol, atol in bands:
        end = start + count
        assert loss[start:end] == pytest.approx(
            expected["loss"][start:end], rel=rtol, abs=atol
        ), (
            f"{spec.key} ({arch}): rank-0 loss[{start}:{end}] (rel/abs={rtol}) "
            f"does not match goldens\n"
            f"  got     : {loss[start:end]}\n"
            f"  expected: {expected['loss'][start:end]}{run_detail}"
        )
        start = end
    # ``grad_norm`` is optional: ``None`` skips the check when the FSDP
    # global-norm all-reduce isn't bit-exact on this arch.
    if expected["grad_norm"] is None:
        return
    assert grad_norm[:n] == pytest.approx(
        expected["grad_norm"][:n], rel=spec.loss_rtol, abs=spec.loss_atol
    ), (
        f"{spec.key} ({arch}): global grad-norm[:{n}] does not match goldens\n"
        f"  got     : {grad_norm[:n]}\n"
        f"  expected: {expected['grad_norm'][:n]}{run_detail}"
    )


# Define only the test function matching MAX_GPUS — the conftest rejects
# ``gpus(N)`` markers outside the active ``ALL_NUM_GPUS = (0, 1, MAX_GPUS)``.
if MAX_GPUS == 4:

    @pytest.mark.level(2)
    @pytest.mark.gpus(4)
    @pytest.mark.parametrize("spec_key", _SPEC_KEYS, ids=lambda k: k.removeprefix("launch_"))
    def test_launch_regression(spec_key: str, tmp_path: Path, h100_inputs: dict[str, str]) -> None:
        """Re-run ``spec``'s torchrun command and check loss / grad-norm against goldens."""
        _assert_spec_matches_goldens(spec_key, tmp_path, h100_inputs)

    @pytest.mark.level(2)
    @pytest.mark.gpus(4)
    def test_convert_reasoner_converts_all_qwen_tensors(
        tmp_path: Path, qwen_vl_model_path: str, request: pytest.FixtureRequest
    ) -> None:
        """Regression guard for ``convert_model_to_vlm_safetensors`` (Reasoner /
        VideoPhy-2 SFT "Step 2": merge Cosmos3-Nano onto the Qwen3-VL shell).

        The converter always *saves* a full ``Qwen3VLForConditionalGeneration``,
        so its tensor set matches Qwen3-VL by construction — a bare key-coverage
        check is trivially true. The invariant that actually regressed once (the
        visual tower was silently kept as the stock Qwen3-VL weights) is that
        **every** Qwen3-VL tensor is sourced from Cosmos3-Nano. This asserts:

          1. the merged tensor set == the stock Qwen3-VL tensor set (all
             included, none extra);
          2. the whole visual tower (``model.visual.*``) matches the Cosmos3-Nano
             ``vision_encoder/`` source bit-for-bit (converted from Cosmos3, not
             left as the stock Qwen3-VL default);
          3. that source genuinely differs from stock Qwen3-VL for a non-trivial
             number of visual tensors — so check (2) has teeth against the
             vision-drop regression. (Cosmos3's tower is derived from Qwen3-VL, so
             a subset of tensors, e.g. some biases, legitimately coincide, which
             is why this is a "some differ", not "all differ", check); and
          4. the language tower was overlaid too (layer-0 projection *weights*
             differ from stock Qwen3-VL).

        CPU-only (the converter forces ``COSMOS_DEVICE=cpu``); it carries the
        ``gpus(4)`` marker only so it is collected by the same 4-GPU regression
        invocation as the launch specs.
        """
        import torch

        qwen_dir = Path(qwen_vl_model_path)
        out_dir = tmp_path / "Cosmos3-Nano-VLM"
        # The merged output is ~16 GB; pytest only rotates ``tmp_path`` (keeps the
        # last few runs), so delete it unconditionally after the test to avoid
        # filling the temp filesystem. Inputs (the Qwen3-VL / Cosmos3-Nano HF
        # caches) are shared and intentionally left in place.
        request.addfinalizer(lambda: shutil.rmtree(out_dir, ignore_errors=True))

        # Run the real conversion, reusing the staged Qwen3-VL snapshot as the
        # shell so the test does not download an 8B model a second time.
        env = os.environ.copy()
        env["PYTHONPATH"] = f".:{env.get('PYTHONPATH', '')}"
        result = subprocess.run(
            [
                sys.executable, "-m", "cosmos_framework.scripts.convert_model_to_vlm_safetensors",
                "--checkpoint-path", "Cosmos3-Nano",
                "-o", str(out_dir),
                "--vlm-model-name", str(qwen_dir),
            ],
            cwd=str(REPO_ROOT),
            env=env,
        )
        assert result.returncode == 0, (
            f"convert_model_to_vlm_safetensors failed with exit code {result.returncode}"
        )

        merged_wm = _weight_map(out_dir)
        qwen_wm = _weight_map(qwen_dir)
        assert merged_wm and qwen_wm, "both checkpoints must be sharded safetensors with an index"

        # (1) Coverage: identical tensor sets — every Qwen3-VL tensor included, none extra.
        missing = sorted(set(qwen_wm) - set(merged_wm))
        extra = sorted(set(merged_wm) - set(qwen_wm))
        assert not missing and not extra, (
            f"merged tensor set != stock Qwen3-VL: missing={missing[:10]} extra={extra[:10]}"
        )

        # (2)+(3) Visual tower: every tensor is converted from the Cosmos3-Nano
        # ``vision_encoder/`` source bit-for-bit, and a non-trivial subset differs
        # from stock Qwen3-VL (so (2) actually distinguishes a converted tower
        # from a stock one — the vision-drop regression).
        from cosmos_framework.inference.args import OmniSetupOverrides
        from cosmos_framework.inference.common.args import CheckpointOverrides
        from safetensors import safe_open

        nano_dir = Path(
            CheckpointOverrides(checkpoint_path="Cosmos3-Nano")
            .build_checkpoint(checkpoints=OmniSetupOverrides.CHECKPOINTS)
            .download_checkpoint()
        )
        vision_src = nano_dir / "vision_encoder" / "model.safetensors"
        assert vision_src.exists(), f"Cosmos3-Nano vision tower not found at {vision_src}"

        visual_keys = sorted(k for k in merged_wm if k.startswith("model.visual."))
        assert visual_keys, "no model.visual.* tensors in merged checkpoint"
        n_differ_from_stock = 0
        with safe_open(str(vision_src), framework="pt") as src:
            src_keys = set(src.keys())
            for k in visual_keys:
                sub = k[len("model.visual."):]
                assert sub in src_keys, f"{k} has no Cosmos3-Nano vision_encoder counterpart"
                got = _load_st_tensor(out_dir, k, merged_wm).float()
                want = src.get_tensor(sub).float()
                assert torch.equal(got, want), f"visual tensor {k} not sourced from Cosmos3-Nano"
                stock = _load_st_tensor(qwen_dir, k, qwen_wm).float()
                if got.shape != stock.shape or not torch.equal(got, stock):
                    n_differ_from_stock += 1
        # Cosmos3's tower is derived from Qwen3-VL, so some tensors coincide; but a
        # meaningful fraction must differ, else keeping the stock tower (the bug)
        # would be indistinguishable from converting.
        assert n_differ_from_stock > 0, (
            f"all {len(visual_keys)} visual tensors equal stock Qwen3-VL — "
            "conversion is indistinguishable from keeping the stock tower"
        )
        print(
            f"\nconvert-reasoner: {len(visual_keys)} visual tensors all sourced from "
            f"Cosmos3-Nano; {n_differ_from_stock} differ from stock Qwen3-VL."
        )

        # (4) Language tower overlaid too: layer-0 projection *weights* (a distinct
        # model, so never bit-identical) differ from stock Qwen3-VL. Restricted to
        # ``*proj.weight`` to avoid biases/norms that can coincide by init.
        lm_sample = [
            k for k in sorted(merged_wm)
            if ".layers.0." in k and not k.startswith("model.visual.") and k.endswith("proj.weight")
        ][:6]
        assert lm_sample, "no layer-0 language-model projection weights found in merged checkpoint"
        for k in lm_sample:
            got = _load_st_tensor(out_dir, k, merged_wm).float()
            stock = _load_st_tensor(qwen_dir, k, qwen_wm).float()
            assert got.shape != stock.shape or not torch.equal(got, stock), (
                f"LM tensor {k} still equals stock Qwen3-VL (not converted from Cosmos3-Nano)"
            )


if MAX_GPUS == 8:

    @pytest.mark.skip(reason="vision_sft_super spec disabled")
    @pytest.mark.level(2)
    @pytest.mark.gpus(8)
    @pytest.mark.parametrize(
        "spec_key", _SPEC_KEYS_8GPU, ids=lambda k: k.removeprefix("launch_")
    )
    def test_launch_regression_8gpu(spec_key: str, tmp_path: Path, h100_inputs: dict[str, str]) -> None:
        """8-GPU variant for ``vision_sft_super`` (dp_shard=4 × cp=2)."""
        _assert_spec_matches_goldens(spec_key, tmp_path, h100_inputs)


# Goldens keyed by GPU arch then ``LaunchSpec.key``. Refresh with
# ``COSMOS_REGRESSION_UPDATE_GOLDENS=1``.
_GOLDENS: dict[str, dict[str, dict[str, list[float] | None]]] = {
    # Captured 2026-05-18 on a 4 × NVIDIA GB200 node with ``--deterministic``
    # and seed 42 against the legacy training pipeline. VLM backbone is not
    # part of the OSS layout.
    "gb200": {
        "llava_ov": {
            "loss": [1.32208, 1.20886, 1.39254, 1.40460, 1.16652, 1.24852, 1.38463, 1.22766, 0.96263, 1.14468],
            "grad_norm": [
                38.62454, 23.61477, 30.53218, 36.46255, 25.06240,
                39.70305, 48.52226, 52.18334, 22.77521, 25.06970,
            ],
        },
        # Recaptured 2026-06-25 on a 4 × NVIDIA GB200 node with seed 42 against the
        # current TOML-config pipeline (inputs prepared in-test by ``h100_inputs``,
        # which now also serves gb200). The numerical shift from the 2026-06-09
        # capture reflects the rectified-flow sigma-sampling refactor
        # (``t = 1 - t_raw`` flip moved into the sampler via per-sample ``shifts``)
        # and is expected. Runs under ``--deterministic`` so loss reproduces bit-exact
        # across all 10 iters. grad_norm is deterministic here (compile.enabled=false
        # in nano_model_config under the new release branch), so values are pinned;
        # flip to None if a future change re-enables compile and reintroduces
        # non-determinism in the all-rank reduction.
        "vision_sft_nano": {
            "loss": [0.2243, 0.2133, 0.2437, 0.2255, 0.2616, 0.2552, 0.3313, 0.2247, 0.2036, 0.2621],
            "grad_norm": [0.42188, 0.30469, 0.30078, 0.26953, 0.30273, 0.41406, 0.42773, 0.38477, 0.27344, 0.27344],
        },
    },
    # Recaptured 2026-06-03 on a 4 × NVIDIA H100 80GB HBM3 node with seed 42 and
    # transformers==4.57.6. VLM model is ``Qwen/Qwen3-VL-8B-Instruct``; inputs are
    # prepared in-test by the ``h100_inputs`` fixture (or via
    # ``tests/_stage_h100_inputs.sh`` if its env vars are pre-set).
    "h100": {
        # num_workers=0, deterministic mode off (see the spec's hydra overrides
        # and the loss_tol_bands tiers). Centered on the midpoint of two H200 CI
        # runs (CI runs on H200) so the tiered bands keep maximum margin; iter-0
        # is bit-exact across H100/H200 runs. grad-norm is non-det, so None.
        "llava_ov": {
            "loss": [1.06924, 0.88399, 1.09293, 1.16314, 1.03592, 0.99041, 1.11041, 0.97001, 0.81246, 0.98548],
            "grad_norm": None,
        },
        # Recaptured 2026-06-03 after the TOML-config rewrite shifted some
        # defaults. Runs under ``--deterministic`` so loss reproduces bit-exact
        # across all 10 iters, but grad_norm is non-det because
        # ``compile.enabled=true`` makes the all-rank reduction not bit-exact
        # on H100.
        "vision_sft_nano": {
            "loss": [0.2242, 0.2141, 0.2429, 0.2259, 0.2608, 0.2555, 0.332, 0.2256, 0.2041, 0.2621],
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
