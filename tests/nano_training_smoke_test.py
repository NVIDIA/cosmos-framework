# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""8-GPU Cosmos3-Nano SFT pipeline smoke test (train -> export -> infer).

Runs the documented Vision SFT (Cosmos3-Nano) lifecycle from ``docs/training.md``
end to end on 8 GPUs and validates each artifact:

  1. Step 1 -- download the bridge-v2 subset dataset + the Wan2.2 VAE.
  2. Step 2 -- ``convert_model_to_dcp`` Cosmos3-Nano -> DCP; check DCP completeness.
  3. Step 3 -- train 5 steps (``vision_sft_nano_5iter``); check the rank-0 loss
     drops below its starting value (``min(loss) < loss[0]``; per-step diffusion
     loss is too noisy for a strict trend over only 5 steps).
  4. Export -- ``export_model`` the trained DCP -> HF safetensors; check export
     completeness (the ``checkpoint.json`` sentinel + config + safetensors).
  5. Inference -- a t2i generation from the exported model; check the image is
     valid.

Smoke-level checks only (artifact validity + a downward loss trend), not numeric
goldens -- that is ``launch_regression_test.py``'s job.

Inputs land in the documented ``.gitignore``-d locations (``examples/data/``,
``examples/checkpoints/``, cached across runs); run output goes under the pytest
tmp dir. Steps 1-2 are skipped when their artifacts already exist.

Invocation (inside the training container, from the repo root, on an 8-GPU
node)::

    pytest -s tests/nano_training_smoke_test.py --num-gpus=8 --levels=2 -o addopts=

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

# Documented default locations (all git-ignored). Match the launcher defaults so
# Step 3 needs no path overrides.
_DATA_DIR = REPO_ROOT / "examples/data/bridge-v2-subset-synthetic-captions"
_DATASET_PATH = _DATA_DIR / "sft_dataset_bridge"
_DATASET_REVISION = "46468e12ac0dd36901e9e3240d4fc7620942b5d7"
_WAN_VAE = REPO_ROOT / "examples/checkpoints/wan22_vae/Wan2.2_VAE.pth"
_DCP_DIR = REPO_ROOT / "examples/checkpoints/Cosmos3-Nano"
_LAUNCHER = "tests/launch_sft_vision_nano_5iter.sh"

# rank-0 per-iteration loss from the IterSpeed callback, e.g.
#   [RANK 0] Iteration 1: Hit counter: 1/50 | Loss: 0.2302 | Time: ...
_RANK0_LOSS_RE = re.compile(
    r"\[RANK\s+0\]\s+Iteration\s+\d+:\s+Hit counter:[^|]+\|\s+Loss:\s+([-+0-9.eE]+)"
)


def _free_port() -> int:
    """Return a currently-free TCP port for the launcher's torchrun rendezvous
    (avoids EADDRINUSE from a hardcoded port / lingering process)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _run(cmd: list[str], log_file: Path, extra_env: dict | None = None) -> tuple[int, str]:
    """Run ``cmd`` from the repo root, tee combined output to ``log_file``.

    Returns ``(returncode, combined_output)``. Streams live to stdout (so CI
    shows progress under ``pytest -s``) while capturing into the log + a string.
    Inherits the caller's env (HF cache, LD_LIBRARY_PATH, ...) plus ``PYTHONPATH=.``.
    """
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
    return returncode, "".join(captured)


def _hf_download(args: list[str], log_path: Path) -> tuple[int, str]:
    """``uvx hf@latest download <args>``, retrying once with ``--refresh``
    (``@latest`` doesn't refresh dependency index metadata)."""
    rc, out = _run(["uvx", "hf@latest", "download", *args], log_path)
    if rc != 0:
        rc, out = _run(["uvx", "--refresh", "hf@latest", "download", *args], log_path)
    return rc, out


def _ensure_inputs(log_dir: Path) -> None:
    """Step 1: download the dataset + Wan2.2 VAE if not already present."""
    if not (_DATASET_PATH / "train" / "video_dataset_file.jsonl").is_file():
        rc, out = _hf_download(
            [
                "--repo-type", "dataset",
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
        rc, out = _hf_download(
            [
                "Wan-AI/Wan2.2-TI2V-5B", "Wan2.2_VAE.pth",
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


def _rank0_losses(text: str) -> list[float]:
    """Parse the rank-0 per-iteration ``Loss:`` series (one value per step)."""
    vals = []
    for m in _RANK0_LOSS_RE.finditer(text):
        try:
            v = float(m.group(1))
        except ValueError:
            continue
        if v == v and abs(v) != float("inf"):  # finite (NaN != NaN)
            vals.append(v)
    return vals


def _safetensors_tensor_names(path: Path) -> set[str]:
    """Validate a .safetensors header (8-byte LE length + JSON) and return its tensor names."""
    assert path.is_file() and path.stat().st_size > 8, f"safetensors shard missing/empty: {path}"
    with path.open("rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        assert 0 < header_len < path.stat().st_size, f"bad safetensors header length in {path}: {header_len}"
        header = json.loads(f.read(header_len))  # raises if the header isn't valid JSON
    return {k for k in header if k != "__metadata__"}


def _assert_dcp_complete(dcp_root: Path) -> None:
    """Structural + index-consistency completeness of a torch DCP (no tensor load).

    For each ``.metadata`` under ``dcp_root``: the shard files beside it must all
    exist and be non-empty, and the set/count of ``*.distcp`` files on disk must
    match the storage files the ``.metadata`` index references (no missing/extra).
    Reading ``.metadata`` only parses the index, not the tensors.
    """
    assert dcp_root.is_dir(), f"DCP dir missing: {dcp_root}"
    metas = list(dcp_root.rglob(".metadata"))
    assert metas, f"no DCP .metadata under {dcp_root}"
    from torch.distributed.checkpoint import FileSystemReader

    for meta in metas:
        assert meta.stat().st_size > 0, f"empty DCP .metadata: {meta}"
        present = sorted(p.name for p in meta.parent.glob("*.distcp"))
        assert present, f"no .distcp shards beside {meta}"
        empty = [s for s in present if (meta.parent / s).stat().st_size == 0]
        assert not empty, f"empty .distcp shards beside {meta}: {empty}"

        # Index consistency: the .metadata declares which shard files exist.
        metadata = FileSystemReader(str(meta.parent)).read_metadata()
        referenced = {getattr(info, "relative_path", None) for info in metadata.storage_data.values()}
        referenced.discard(None)
        if referenced:  # skip only if this reader doesn't expose shard paths
            missing = sorted(set(referenced) - set(present))
            assert not missing, (
                f"DCP {meta.parent}: .metadata references {len(referenced)} shard file(s) but "
                f"these are missing on disk: {missing}"
            )
            assert len(present) == len(referenced), (
                f"DCP {meta.parent}: {len(present)} .distcp file(s) on disk != "
                f"{len(referenced)} referenced by .metadata ({present} vs {sorted(referenced)})"
            )

        # Tensor-manifest self-consistency: every tensor the .metadata declares
        # (state_dict_metadata) must be backed by storage (no omitted param).
        declared = set(metadata.state_dict_metadata.keys())
        stored = {getattr(idx, "fqn", None) for idx in metadata.storage_data.keys()}
        stored.discard(None)
        assert declared, f"DCP .metadata declares no tensors: {meta}"
        if stored:  # skip only if storage keys don't expose fqn
            unstored = sorted(declared - stored)
            assert not unstored, (
                f"DCP {meta.parent}: {len(unstored)} declared tensor(s) have no storage "
                f"(omitted): {unstored[:10]}"
            )


def _assert_export_complete(model_dir: Path) -> None:
    """Structural + index completeness of an exported HF safetensors checkpoint."""
    assert model_dir.is_dir(), f"export dir missing: {model_dir}"
    # export_model writes checkpoint.json LAST as the "model is complete" sentinel.
    for name in ("checkpoint.json", "config.json"):
        p = model_dir / name
        assert p.is_file() and p.stat().st_size > 0, f"export missing/empty {name} in {model_dir}"
        json.loads(p.read_text())  # valid JSON
    index = model_dir / "model.safetensors.index.json"
    on_disk = sorted(p.name for p in model_dir.glob("*.safetensors"))
    if index.is_file():
        weight_map = json.loads(index.read_text()).get("weight_map", {})
        declared = set(weight_map.keys())
        shards = sorted(set(weight_map.values()))
        assert declared and shards, f"empty weight_map in {index}"
        missing = sorted(set(shards) - set(on_disk))
        assert not missing, f"export {model_dir}: index references missing shards: {missing}"
        # File-count consistency: exactly the index's shards on disk (no extra/missing).
        assert len(on_disk) == len(shards), (
            f"export {model_dir}: {len(on_disk)} .safetensors on disk != {len(shards)} in index "
            f"weight_map ({on_disk} vs {shards})"
        )
        # Tensor-manifest self-consistency: the tensors actually stored across the
        # shards must equal the index's declared keys (no omitted/extra param).
        stored: set[str] = set()
        for shard in shards:
            stored |= _safetensors_tensor_names(model_dir / shard)
        assert declared == stored, (
            f"export {model_dir}: index declares {len(declared)} tensors but shards hold {len(stored)} "
            f"(missing from shards: {sorted(declared - stored)[:10]}; not in index: {sorted(stored - declared)[:10]})"
        )
    else:
        assert on_disk == ["model.safetensors"], (
            f"export {model_dir}: expected a single model.safetensors (no index), found {on_disk}"
        )
        names = _safetensors_tensor_names(model_dir / "model.safetensors")
        assert names, f"export {model_dir}: model.safetensors holds no tensors"


def _assert_diffusers_complete(model_dir: Path, reference_dir: Path) -> None:
    """Structural + index completeness of a Diffusers pipeline converted from the HF export,
    and a tensor-level comparison against the published ``nvidia/Cosmos3-Nano`` diffusers.

    The ``vision_sft_nano`` export has no sound tokenizer (``sound_gen=False``) and no
    standalone reasoner ViT (``include_visual`` unset), so the ``sound_tokenizer/`` and
    ``vision_encoder/`` components — and the sound-only ``audio_*`` transformer tensors —
    are absent; the golden comparison ignores exactly those. Every component that *is*
    present is validated as thoroughly as the HF export: required files, pipeline class,
    per-shard/per-tensor self-consistency of both the transformer index and the aggregated
    root weight index, and (against the golden) the transformer tensor set + config
    ``architectures`` / ``model_type``.
    """
    assert model_dir.is_dir(), f"diffusers dir missing: {model_dir}"
    required = (
        "config.json",
        "model_index.json",
        "modular_model_index.json",
        "model.safetensors.index.json",
        "scheduler/scheduler_config.json",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    )
    for rel in required:
        p = model_dir / rel
        assert p.is_file() and p.stat().st_size > 0, f"diffusers export missing/empty: {p}"

    model_index = json.loads((model_dir / "model_index.json").read_text())
    assert model_index.get("_class_name") == "Cosmos3OmniPipeline", (
        f"unexpected diffusers pipeline class: {model_index.get('_class_name')!r}"
    )
    for component in ("scheduler", "transformer", "vae"):
        assert component in model_index, f"model_index.json missing component {component!r}"

    # Root weight index: references existing shards and every declared tensor is
    # actually stored across them (no omitted/extra param).
    root_index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    root_weight_map: dict[str, str] = root_index["weight_map"]
    assert root_weight_map, "root Diffusers safetensors index has no tensors"
    assert root_index.get("metadata", {}).get("total_size", 0) > 0
    root_shards = sorted(set(root_weight_map.values()))
    missing_shards = [s for s in root_shards if not (model_dir / s).is_file()]
    assert not missing_shards, f"root index references missing shards: {missing_shards}"
    stored: set[str] = set()
    for shard in root_shards:
        stored |= _safetensors_tensor_names(model_dir / shard)
    assert set(root_weight_map) == stored, (
        f"root index declares {len(root_weight_map)} tensors but shards hold {len(stored)} "
        f"(missing from shards: {sorted(set(root_weight_map) - stored)[:10]}; "
        f"not in index: {sorted(stored - set(root_weight_map))[:10]})"
    )

    # Transformer index: its weight_map is the transformer-scoped slice of the root
    # index and points at exactly the transformer shards on disk.
    transformer_index = json.loads(
        (model_dir / "transformer/diffusion_pytorch_model.safetensors.index.json").read_text()
    )
    transformer_weight_map: dict[str, str] = transformer_index["weight_map"]
    expected_transformer = {
        name: filename.removeprefix("transformer/")
        for name, filename in root_weight_map.items()
        if filename.startswith("transformer/")
    }
    assert transformer_weight_map == expected_transformer, "transformer index inconsistent with root index"
    assert set(transformer_weight_map.values()) == {
        p.name for p in (model_dir / "transformer").glob("*.safetensors")
    }

    # VAE weights hold tensors; no raw DCP shards leak into the Diffusers export.
    assert _safetensors_tensor_names(model_dir / "vae/diffusion_pytorch_model.safetensors")
    assert not list(model_dir.rglob("*.distcp")), "Diffusers export unexpectedly contains DCP shards"

    # Golden comparison against nvidia/Cosmos3-Nano: the transformer tensor set must equal
    # the reference's, ignoring the sound-only ``audio_*`` tensors (this export has
    # sound_gen=False) and the reference's vision_encoder/ shards (include_visual unset
    # here). config architectures/model_type must match exactly.
    reference_weight_map = json.loads((reference_dir / "model.safetensors.index.json").read_text())["weight_map"]
    reference_transformer = {
        name
        for name, filename in reference_weight_map.items()
        if filename.startswith("transformer/") and not name.startswith("audio_")
    }
    out_transformer = {name for name, filename in root_weight_map.items() if filename.startswith("transformer/")}
    assert out_transformer == reference_transformer, (
        "transformer tensor set differs from nvidia/Cosmos3-Nano (ignoring sound/vision): "
        f"missing={sorted(reference_transformer - out_transformer)[:8]}, "
        f"extra={sorted(out_transformer - reference_transformer)[:8]}"
    )
    reference_config = json.loads((reference_dir / "config.json").read_text())
    out_config = json.loads((model_dir / "config.json").read_text())
    assert out_config.get("architectures") == reference_config.get("architectures"), (
        f"config architectures differ from golden: {out_config.get('architectures')} vs "
        f"{reference_config.get('architectures')}"
    )
    assert out_config.get("model_type") == reference_config.get("model_type"), (
        f"config model_type differs from golden: {out_config.get('model_type')} vs {reference_config.get('model_type')}"
    )


def _assert_valid_image(path: Path) -> None:
    """Assert ``path`` is a valid, non-degenerate image."""
    assert path.is_file() and path.stat().st_size > 1024, f"output image missing/too small: {path}"
    try:
        from PIL import Image
    except Exception:  # pragma: no cover -- PIL expected in the env
        assert path.read_bytes()[:3] == b"\xff\xd8\xff", f"not a JPEG: {path}"
        return
    with Image.open(path) as im:
        im.verify()  # detects truncation/corruption
    with Image.open(path) as im:
        width, height = im.size
    assert width > 0 and height > 0, f"degenerate image size {width}x{height}: {path}"


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
    def test_nano_sft_train_export_infer(tmp_path: Path) -> None:
        """Full Cosmos3-Nano SFT pipeline: convert -> train 5 -> export -> t2i infer."""
        # 1-2. Inputs + HF->DCP convert, then DCP completeness.
        _ensure_inputs(tmp_path)
        _ensure_dcp(tmp_path)
        _assert_dcp_complete(_DCP_DIR)

        # 3. Train 5 steps (run output -> pytest tmp via OUTPUT_ROOT + the harness's
        #    IMAGINAIRE_OUTPUT_ROOT). Free port avoids EADDRINUSE.
        rc, out = _run(
            ["bash", _LAUNCHER],
            tmp_path / "train.log",
            extra_env={
                "MASTER_PORT": str(_free_port()),
                "OUTPUT_ROOT": str(tmp_path / "launcher_out"),
                "NPROC_PER_NODE": "8",
            },
        )
        assert rc == 0, f"SFT launch failed (exit {rc}):\nLog tail:\n{out[-4000:]}"
        assert "Done with training" in out, f"training did not finish cleanly:\nLog tail:\n{out[-4000:]}"

        losses = _rank0_losses(out)
        assert len(losses) == 5, f"expected 5 rank-0 losses, parsed {losses}\nLog tail:\n{out[-2000:]}"
        # Per-step diffusion loss is noisy (a random timestep is sampled each step),
        # so a strict trend over just 5 steps flakes on a single noisy step. The
        # robust "training is learning" signal is that the loss dropped below its
        # starting value at some point.
        assert min(losses) < losses[0], (
            f"loss never dropped below the first step over 5 steps (training not degrading): {losses}"
        )

        # 4. Locate the trained DCP + config, export to HF safetensors, check completeness.
        saved = re.findall(r"Saved checkpoint to (\S+)", out)
        assert saved, f"no 'Saved checkpoint to ...' line in training log:\n{out[-2000:]}"
        ckpt = Path(saved[-1])
        assert ckpt.is_dir() and any(ckpt.iterdir()), f"trained checkpoint dir missing/empty: {ckpt}"
        run_dir = ckpt.parent.parent  # <RUN_DIR>/checkpoints/iter_X -> <RUN_DIR>
        config_yaml = run_dir / "config.yaml"
        assert config_yaml.is_file(), f"run config.yaml missing at {config_yaml}"

        export_dir = run_dir / "model"
        rc, out = _run(
            [
                "python", "-m", "cosmos_framework.scripts.export_model",
                "--checkpoint-path", str(ckpt),
                "--config-file", str(config_yaml),
                "-o", str(export_dir),
            ],
            tmp_path / "export.log",
        )
        assert rc == 0, f"export_model failed (exit {rc}):\nLog tail:\n{out[-4000:]}"
        _assert_export_complete(export_dir)

        # 4b. Convert the exported HF checkpoint to a Diffusers pipeline; check layout.
        diffusers_dir = run_dir / "diffusers"
        rc, out = _run(
            [
                "python", "-m", "cosmos_framework.scripts.convert_model_to_diffusers",
                "--checkpoint-path", str(export_dir),
                "-o", str(diffusers_dir),
            ],
            tmp_path / "convert.log",
        )
        assert rc == 0, f"convert_model_to_diffusers failed (exit {rc}):\nLog tail:\n{out[-4000:]}"
        # Compare against the published Cosmos3-Nano diffusers (index + config only),
        # ignoring the sound_tokenizer/ and vision_encoder/ components this export lacks.
        golden_dir = tmp_path / "Cosmos3-Nano-ref"
        rc, out = _hf_download(
            [
                "nvidia/Cosmos3-Nano", "model.safetensors.index.json", "config.json",
                "--local-dir", str(golden_dir), "--quiet",
            ],
            tmp_path / "download_golden.log",
        )
        assert rc == 0, f"Cosmos3-Nano reference download failed (exit {rc}):\n{out[-2000:]}"
        _assert_diffusers_complete(diffusers_dir, golden_dir)

        # 5. t2i inference from the exported model; check the image is valid.
        infer_out = tmp_path / "exported_out"
        rc, out = _run(
            [
                "torchrun", "--nproc_per_node=8", f"--master_port={_free_port()}",
                "-m", "cosmos_framework.scripts.inference",
                "--parallelism-preset=throughput",
                "-i", "inputs/omni/t2i.json",
                "-o", str(infer_out),
                "--checkpoint-path", str(export_dir),
                "--seed=0",
            ],
            tmp_path / "infer.log",
        )
        assert rc == 0, f"t2i inference from exported model failed (exit {rc}):\nLog tail:\n{out[-4000:]}"
        images = list(infer_out.rglob("vision.jpg"))
        assert len(images) == 1, f"expected one vision.jpg under {infer_out}, found {images}"
        _assert_valid_image(images[0])
