# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""4-GPU reasoner inference test for Cosmos3-Nano.

Two cases, each a separate ``cosmos_framework.scripts.inference`` torchrun
launched through ``tests/_reasoner_logits_probe.py`` (which pins deterministic
kernels and captures the first decoded token's logits on rank 0):

  * ``test_nano_reasoner_first_token_logits`` — text-only reasoner inference
    (``inputs/reasoner/reasoner.json``).
  * ``test_nano_reasoner_image_first_token_logits`` — image-conditioned reasoner
    inference (``inputs/reasoner/reasoner_image.json``).

Each asserts a non-empty ``reasoner_text`` was produced AND compares the
captured first-token logits against its own committed golden tensor: exact
argmax match + ``torch.allclose(rtol=1e-3, atol=1e-3)``. Determinism is pinned
in the probe (greedy decode, deterministic cuBLAS/cuDNN/flash-attn, fixed seed),
so a clean run reproduces the golden run-to-run on the same 4-GPU config.

Goldens (one per case)::

    tests/data/nano_reasoner_inference_smoke_test/first_token_logits_golden.pt
    tests/data/nano_reasoner_inference_smoke_test/first_token_logits_image_golden.pt

Golden bootstrap: on the first run a golden does not exist; the test writes the
captured tensor next to the golden path (``*_golden`` suffix dropped) and skips
with instructions to rename it to the golden name and commit. Subsequent runs
compare against the committed golden.

Invocation (inside the inference container, from the repo root, on a >=4-GPU
node)::

    TEST_MAX_GPUS=4 pytest -s tests/nano_reasoner_inference_smoke_test.py \
        --num-gpus=4 --levels=2 -o addopts=

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

# Goldens live under the repo's tests/data/<module> convention, one per case.
_GOLDEN_DIR = REPO_ROOT / "tests" / "data" / "nano_reasoner_inference_smoke_test"
_TEXT_GOLDEN = _GOLDEN_DIR / "first_token_logits_golden.pt"
_IMAGE_GOLDEN = _GOLDEN_DIR / "first_token_logits_image_golden.pt"

# Tight tolerance — the probe pins deterministic kernels + greedy decode.
_RTOL = 1e-3
_ATOL = 1e-3


def _free_port() -> int:
    """Return a currently-free TCP port for torchrun's rendezvous."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _run(cmd: list[str], log_file: Path, extra_env: dict[str, str] | None = None) -> str:
    """Run ``cmd`` from the repo root, tee combined output (live under ``-s`` +
    into ``log_file``). Inherits the caller's env plus ``PYTHONPATH=.`` and any
    ``extra_env``. Fails with the log tail on a non-zero exit."""
    env = os.environ.copy()
    env["PYTHONPATH"] = f".:{env.get('PYTHONPATH', '')}"
    if extra_env:
        env.update(extra_env)
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


def _reasoner_text(out_dir: Path) -> str:
    """Read the single sample's ``reasoner_text`` from ``sample_outputs.json``."""
    results = sorted(out_dir.rglob("sample_outputs.json"))
    assert len(results) == 1, f"expected one sample_outputs.json, found {[str(p) for p in results]}"
    content = json.loads(results[0].read_text())["outputs"][0]["content"]
    text = content.get("reasoner_text") if isinstance(content, dict) else None
    assert isinstance(text, str) and text.strip(), f"empty/missing reasoner_text in {results[0]}: {content!r}"
    return text


def _run_reasoner_probe(tmp_path: Path, input_json: str) -> Path:
    """Launch a 4-GPU reasoner inference for ``input_json`` through the logits
    probe; assert a non-empty ``reasoner_text``; return the dumped logits path."""
    out_dir = tmp_path / "out"
    dump = tmp_path / "first_token_logits.pt"
    cmd = [
        "torchrun",
        "--nproc_per_node=4",
        f"--master_port={_free_port()}",
        "tests/_reasoner_logits_probe.py",
        "--parallelism-preset=throughput",
        "-i",
        input_json,
        "-o",
        str(out_dir),
        "--checkpoint-path",
        "Cosmos3-Nano",
        "--seed=0",
    ]
    _run(cmd, tmp_path / "inference.log", extra_env={"REASONER_LOGITS_DUMP": str(dump)})
    _reasoner_text(out_dir)
    assert dump.is_file(), f"probe did not write first-token logits to {dump}"
    return dump


def _assert_matches_golden(dump: Path, golden_path: Path) -> None:
    """Compare captured logits to ``golden_path``: exact argmax + tight allclose.

    On the first run (no golden) stage the candidate next to the golden path
    (``*_golden`` suffix dropped) and skip with rename instructions.
    """
    import torch

    current = torch.load(dump)
    if not golden_path.is_file():
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        candidate = golden_path.with_name(golden_path.name.replace("_golden", ""))
        shutil.copyfile(dump, candidate)
        pytest.skip(f"golden created at {candidate}; rename to {golden_path.name} and commit, then re-run")

    ref = torch.load(golden_path)
    assert current.shape == ref.shape, f"logits shape {tuple(current.shape)} != golden {tuple(ref.shape)}"
    # Hard gate: the greedily-predicted first token must match exactly.
    assert int(current.argmax()) == int(ref.argmax()), (
        f"first-token argmax {int(current.argmax())} != golden {int(ref.argmax())}"
    )
    # Sensitive gate: full logits within tight tolerance.
    assert torch.allclose(current, ref, rtol=_RTOL, atol=_ATOL), (
        f"first-token logits differ from golden beyond rtol={_RTOL}, atol={_ATOL}; "
        f"max|Δ|={float((current - ref).abs().max()):.3e}"
    )


@pytest.fixture(scope="module", autouse=True)
def _require_4_gpus() -> None:
    """Skip the module unless we can launch a 4-GPU run here."""
    if shutil.which("torchrun") is None:
        pytest.skip("torchrun not on PATH -- must run inside the inference container")
    try:
        import torch
    except Exception as exc:  # pragma: no cover -- surfaces during dev only
        pytest.skip(f"torch unavailable ({exc!r})")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
        pytest.skip(f"requires 4 visible CUDA devices, found {torch.cuda.device_count()}")


# Defined only when the active MAX_GPUS is 4 -- the conftest rejects ``gpus(N)``
# markers outside ``ALL_NUM_GPUS = (0, 1, MAX_GPUS)``. Run with TEST_MAX_GPUS=4.
if MAX_GPUS == 4:

    @pytest.mark.level(2)
    @pytest.mark.gpus(4)
    def test_nano_reasoner_first_token_logits(tmp_path: Path) -> None:
        """Text-only reasoner inference; reasoner_text + golden first-token logits."""
        dump = _run_reasoner_probe(tmp_path, "inputs/reasoner/reasoner.json")
        _assert_matches_golden(dump, _TEXT_GOLDEN)

    @pytest.mark.level(2)
    @pytest.mark.gpus(4)
    def test_nano_reasoner_image_first_token_logits(tmp_path: Path) -> None:
        """Image-conditioned reasoner inference; reasoner_text + golden first-token logits."""
        dump = _run_reasoner_probe(tmp_path, "inputs/reasoner/reasoner_image.json")
        _assert_matches_golden(dump, _IMAGE_GOLDEN)
