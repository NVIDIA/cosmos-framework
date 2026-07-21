# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""4-GPU reasoner inference smoke test for Cosmos3-Edge.

The Edge counterpart of ``nano_reasoner_inference_smoke_test.py``, but at
smoke level: instead of comparing first-token logits against a committed golden
(the Nano suite's tight regression), it validates the generated ``reasoner_text``
directly -- non-empty, coherent (min length + lexical diversity), and on-topic
for the input image (references the robotic subject that dominates the frame).
This keeps the Edge case runnable in CI without a per-checkpoint golden bootstrap
while still checking the generated content, not just that something was emitted.

One case, image-conditioned reasoner inference (matching what the Nano CI job
actually runs): a single ``cosmos_framework.scripts.inference`` torchrun over
``inputs/reasoner/reasoner_image.json``. Edge's Nemotron reasoner is
vision-capable (it loads a SigLIP2 tower lazily -- see ``_reasoner_vision_capable``
in ``cosmos_framework/inference/args.py``), so the image prompt is encoded and
the run must emit non-empty generated text.

Smoke-level only (output validity, not numeric goldens). The checkpoint + its
reasoner backbone download from the HF Hub on first run and are reused afterward.

Invocation (inside the inference container, from the repo root, on a >=4-GPU
node)::

    TEST_MAX_GPUS=4 pytest -s tests/edge_reasoner_inference_smoke_test.py \
        --num-gpus=4 --levels=2 -o addopts=

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

# The image prompt (``inputs/reasoner/reasoner_image.json``) is
# "Describe what is happening in this image in one sentence." over ``robot_153.jpg``:
# an ego-view of two robotic hands/arms reaching over a table (red apple + lab
# equipment + a person in the background). Decoding is greedy (``do_sample: false``
# in ``defaults/reasoner/sample_args.json``), so a correct description reliably
# references that dominant robotic subject. The stems below are broad enough to
# survive phrasing ("robotic hands", "mechanical arms", "grippers", ...) yet fail
# on garbage, an empty/degenerate reply, or a description of some other scene.
_SUBJECT_PATTERN = re.compile(
    r"\b(robot\w*|mechanical|gripper\w*|prosthetic|arm\w*|hand\w*|finger\w*)\b",
    re.IGNORECASE,
)
# A real one-sentence description clears these easily; the floors only catch a
# collapsed/degenerate generation (empty-ish, or one token repeated).
_MIN_TEXT_CHARS = 15
_MIN_UNIQUE_WORDS = 3


def _free_port() -> int:
    """Return a currently-free TCP port for torchrun's rendezvous."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _run(cmd: list[str], log_file: Path) -> str:
    """Run ``cmd`` from the repo root, tee combined output (live under ``-s`` +
    into ``log_file``). Inherits the caller's env plus ``PYTHONPATH=.``. Fails
    with the log tail on a non-zero exit."""
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


def _assert_reasoner_text(out_dir: Path) -> None:
    """Assert the single sample produced a valid, on-topic ``reasoner_text``.

    Three gates: (1) a non-empty string was generated; (2) it is coherent, not a
    collapsed/degenerate reply (min length + lexical diversity); (3) its content
    is correct for the input image -- it references the robotic subject that
    dominates ``robot_153.jpg`` (see ``_SUBJECT_PATTERN``).
    """
    results = sorted(out_dir.rglob("sample_outputs.json"))
    assert len(results) == 1, f"expected one sample_outputs.json, found {[str(p) for p in results]}"
    content = json.loads(results[0].read_text())["outputs"][0]["content"]
    text = content.get("reasoner_text") if isinstance(content, dict) else None

    # (1) non-empty string.
    assert isinstance(text, str) and text.strip(), f"empty/missing reasoner_text in {results[0]}: {content!r}"

    # (2) coherent, non-degenerate text (not empty-ish, not a single repeated token).
    stripped = text.strip()
    words = re.findall(r"[a-zA-Z]+", stripped.lower())
    assert len(stripped) >= _MIN_TEXT_CHARS, (
        f"reasoner_text too short to be a real description ({len(stripped)} chars): {stripped!r}"
    )
    assert len(set(words)) >= _MIN_UNIQUE_WORDS, (
        f"reasoner_text is degenerate/repetitive ({len(set(words))} unique words): {stripped!r}"
    )

    # (3) content correctness: the reply describes the actual image -- it must
    # reference the robotic hands/arms that dominate the frame.
    assert _SUBJECT_PATTERN.search(stripped), (
        f"reasoner_text does not describe the image's robotic subject "
        f"(expected a term like robot/robotic/arm/hand/gripper): {stripped!r}"
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
    def test_edge_reasoner_image_reasoner_text(tmp_path: Path) -> None:
        """Image-conditioned reasoner inference on Cosmos3-Edge; assert non-empty reasoner_text."""
        out_dir = tmp_path / "out"
        cmd = [
            "torchrun",
            "--nproc_per_node=4",
            f"--master_port={_free_port()}",
            "-m",
            "cosmos_framework.scripts.inference",
            "--parallelism-preset=throughput",
            "-i",
            "inputs/reasoner/reasoner_image.json",
            "-o",
            str(out_dir),
            "--checkpoint-path",
            "Cosmos3-Edge",
            "--seed=0",
        ]
        _run(cmd, tmp_path / "inference.log")
        _assert_reasoner_text(out_dir)
