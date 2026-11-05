#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Convert the pinned Nano snapshot and verify portable DCP conversion receipts."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

MODEL_REPO = "nvidia/Cosmos3-Nano"
MODEL_REVISION = "e59a53c25979a090fa8706c9acc0c254a6e89b92"
CONVERTER_MODULE = "cosmos_framework.scripts.convert_model_to_dcp"
CONVERTER_SHA256 = "fd9fb5c09ee24605bf19915b17ee52e7b218defe23610122f083ecc8e9891339"
WAN_REPO = "Wan-AI/Wan2.2-TI2V-5B"
WAN_REVISION = "921dbaf3f1674a56f47e83fb80a34bac8a8f203e"
WAN_FILE = "Wan2.2_VAE.pth"
RECEIPT = "nano_conversion.json"
DERIVED_AUDIO = {
    "sound_tokenizer/avae_48k_noncausal_25hz_64ch.ckpt",
    "sound_tokenizer/avae_48k_noncausal_25hz_64ch.json",
}
CONVERTER_ENV = {
    "COSMOS_DEVICE": "cpu",
    "CUDA_VISIBLE_DEVICES": "",
    "COSMOS_INTERNAL": "0",
    "COSMOS_TRAINING": "0",
    "COSMOS_SMOKE": "0",
    "HF_HUB_OFFLINE": "1",
}


def fingerprint(path: Path, *, git_blob: bool = False) -> dict[str, Any]:
    """Hash bytes, including a Git blob hash when checking non-LFS Hub files."""
    size = path.stat().st_size
    sha256 = hashlib.sha256()
    blob = hashlib.sha1(f"blob {size}\0".encode()) if git_blob else None
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha256.update(chunk)
            if blob is not None:
                blob.update(chunk)
    record: dict[str, Any] = {"bytes": size, "sha256": sha256.hexdigest()}
    if blob is not None:
        record["git_blob_sha1"] = blob.hexdigest()
    return record


def inventory(root: Path, *, exclude_cache: bool = False) -> dict[str, dict[str, Any]]:
    """Use relative filenames so a copied checkpoint retains its identity."""
    return {
        path.relative_to(root).as_posix(): fingerprint(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and (not exclude_cache or path.relative_to(root).parts[0] != ".cache")
        and path.relative_to(root).as_posix() != RECEIPT
    }


def identity(framework_revision: str) -> dict[str, Any]:
    return {
        "schema": 1,
        "source": {"repo": MODEL_REPO, "revision": MODEL_REVISION},
        "converter": {
            "revision": framework_revision,
            "module": CONVERTER_MODULE,
            "sha256": CONVERTER_SHA256,
            "argv": ["-m", CONVERTER_MODULE, "--checkpoint-path", "<source>", "-o", "<output>"],
            "environment": CONVERTER_ENV,
        },
        "auxiliary": {
            "processor": {"repo": MODEL_REPO, "revision": MODEL_REVISION},
            "video_vae": {"repo": WAN_REPO, "revision": WAN_REVISION, "file": WAN_FILE},
        },
    }


def check_checkout(framework: Path) -> str:
    """Check converter bytes and record the clean checkout, without requiring a historical commit."""
    revision = subprocess.check_output(["git", "-C", str(framework), "rev-parse", "HEAD"], text=True).strip()
    subprocess.run(
        ["git", "-C", str(framework), "diff", "--exit-code", "HEAD", "--"], check=True, stdout=subprocess.DEVNULL
    )
    module_path = framework / "cosmos_framework/scripts/convert_model_to_dcp.py"
    if fingerprint(module_path)["sha256"] != CONVERTER_SHA256:
        raise ValueError(
            "Converter source differs from the supported SHA-256; review the converter before updating the hash"
        )
    return revision


def check_environment(framework: Path) -> dict[str, str]:
    """Fail before downloads if the interpreter cannot import the checked converter."""
    revision = check_checkout(framework)
    module_path = framework / "cosmos_framework/scripts/convert_model_to_dcp.py"
    probe = (
        "import importlib, pathlib, sys; "
        f"module = importlib.import_module({CONVERTER_MODULE!r}); "
        "\nif pathlib.Path(module.__file__).resolve() != pathlib.Path(sys.argv[1]).resolve():\n"
        "    raise RuntimeError('Converter imported from the wrong checkout')"
    )
    subprocess.run(
        [sys.executable, "-c", probe, str(module_path)],
        check=True,
        cwd=framework,
        env={**os.environ, **CONVERTER_ENV},
    )
    return {
        "framework_revision": revision,
        "python": sys.version.split()[0],
        **{
            name: importlib.metadata.version(name)
            for name in ("cosmos-framework", "torch", "transformers", "huggingface-hub", "tokenizers")
        },
    }


def download_source(source: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Download into a new directory and check bytes against immutable Hub metadata."""
    hub = importlib.import_module("huggingface_hub")
    api = hub.HfApi()
    info = api.model_info(MODEL_REPO, revision=MODEL_REVISION, files_metadata=True)
    if info.sha != MODEL_REVISION:
        raise ValueError("Hub did not resolve the requested Nano revision")
    files = [item for item in info.siblings if not item.rfilename.startswith(("assets/", "images/", ".gitattributes"))]
    hub.snapshot_download(
        MODEL_REPO,
        revision=MODEL_REVISION,
        local_dir=source,
        allow_patterns=[item.rfilename for item in files],
        max_workers=4,
    )
    records: dict[str, Any] = {}
    for item in files:
        record = fingerprint(source / item.rfilename, git_blob=item.lfs is None)
        expected = item.lfs.sha256 if item.lfs is not None else item.blob_id
        actual = record["sha256"] if item.lfs is not None else record["git_blob_sha1"]
        if item.size != record["bytes"] or not expected or expected != actual:
            raise ValueError(f"Source file differs from pinned Hub content: {item.rfilename}")
        records[item.rfilename] = {"bytes": record["bytes"], "sha256": record["sha256"]}
    if inventory(source, exclude_cache=True) != records:
        raise ValueError("Unexpected files in the fresh Nano snapshot")
    # The pinned architecture resolves processor metadata through main. Bind
    # that alias to the verified snapshot in this conversion's private cache;
    # the offline converter must not depend on the live Hub branch.
    cache = source / ".cache/conversion-hub"
    processor = Path(
        hub.snapshot_download(
            MODEL_REPO,
            revision=MODEL_REVISION,
            # Select only verified source metadata; broad globs also fetch excluded assets/ examples.
            allow_patterns=[name for name in records if Path(name).suffix in {".json", ".txt", ".jinja"}],
            cache_dir=cache,
            max_workers=4,
        )
    )
    if processor.name != MODEL_REVISION:
        raise ValueError("Processor cache does not contain the pinned Nano revision")
    processor_files = inventory(processor)
    if not processor_files or any(record != records.get(name) for name, record in processor_files.items()):
        raise ValueError("Processor cache differs from the verified Nano source")
    reference = processor.parent.parent / "refs/main"
    reference.parent.mkdir(parents=True, exist_ok=True)
    reference.write_text(MODEL_REVISION)
    vae = Path(hub.hf_hub_download(WAN_REPO, WAN_FILE, revision=WAN_REVISION, cache_dir=cache))
    return records, {"processor": processor_files, "video_vae": fingerprint(vae)}


def check_dcp_layout(files: dict[str, dict[str, Any]]) -> None:
    if (
        "model/.metadata" not in files
        or "model/config.json" not in files
        or not any(name.endswith(".distcp") for name in files)
    ):
        raise ValueError("Incomplete DCP model directory")


def check_auxiliary_files(files: object) -> None:
    """Require portable processor and video-VAE fingerprints in the receipt."""
    if not isinstance(files, dict):
        raise ValueError("Missing or invalid auxiliary file inventory")
    processor = files.get("processor")
    if not isinstance(processor, dict) or not processor:
        raise ValueError("Missing or invalid auxiliary processor inventory")
    if any(not isinstance(name, str) or not name for name in processor):
        raise ValueError("Invalid auxiliary processor filename")
    for record in [*processor.values(), files.get("video_vae")]:
        if not isinstance(record, dict):
            raise ValueError("Missing or invalid auxiliary file fingerprint")
        size, digest = record.get("bytes"), record.get("sha256")
        if (
            type(size) is not int
            or size < 0
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise ValueError("Invalid auxiliary file byte count or SHA-256 hash")


def verify(output: Path) -> None:
    """Accept a copied conversion only with matching provenance and payload bytes."""
    receipt_path = output / RECEIPT
    if not receipt_path.is_file():
        raise ValueError("Missing conversion receipt; make a fresh conversion instead of labeling old weights")
    receipt = json.loads(receipt_path.read_text())
    recorded_identity = receipt.get("identity")
    converter = recorded_identity.get("converter", {}) if isinstance(recorded_identity, dict) else {}
    revision = converter.get("revision") if isinstance(converter, dict) else None
    if (
        not isinstance(revision, str)
        or re.fullmatch(r"[0-9a-f]{40}", revision) is None
        or recorded_identity != identity(revision)
    ):
        raise ValueError("Conversion source revision, converter, options or auxiliary identity does not match")
    check_auxiliary_files(receipt.get("auxiliary_files"))
    files = inventory(output)
    check_dcp_layout(files)
    runtime = receipt.get("runtime")
    if (
        not receipt.get("source_files")
        or not isinstance(runtime, dict)
        or not runtime
        or files != receipt.get("output_files")
    ):
        raise ValueError("Conversion receipt or checkpoint content is incomplete or changed")
    # Older receipts recorded the checkout only in identity.converter.revision.
    if runtime.get("framework_revision", revision) != revision:
        raise ValueError("Conversion runtime revision does not match the recorded converter revision")
    print("Verified pinned Nano conversion and checkpoint file hashes; training compatibility remains unchecked.")


def convert(framework: Path, source: Path, output: Path) -> None:
    """Create provenance only while performing a fresh pinned conversion."""
    if source.exists() or output.exists():
        raise ValueError("Fresh conversion requires new source and output directories; use verify for an existing DCP")
    if source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("Source and output must be separate directories")
    runtime = check_environment(framework)
    source_files, auxiliary_files = download_source(source)
    child_env = {
        **os.environ,
        **CONVERTER_ENV,
        "HF_HUB_CACHE": str(source / ".cache/conversion-hub"),
        "HUGGINGFACE_HUB_CACHE": str(source / ".cache/conversion-hub"),
    }
    subprocess.run(
        [sys.executable, "-m", CONVERTER_MODULE, "--checkpoint-path", str(source), "-o", str(output)],
        check=True,
        cwd=framework,
        env=child_env,
    )
    after = inventory(source, exclude_cache=True)
    derived = {name: record for name, record in after.items() if name not in source_files}
    if set(derived) - DERIVED_AUDIO or {name: after.get(name) for name in source_files} != source_files:
        raise ValueError("Conversion changed pinned source files or added unexpected files; no receipt written")
    # Check bytes and HEAD again without rerunning the heavy import probe.
    if check_checkout(framework) != runtime["framework_revision"]:
        raise ValueError("Framework checkout changed during conversion; no receipt written")
    output_files = inventory(output)
    check_dcp_layout(output_files)
    receipt = {
        "identity": identity(runtime["framework_revision"]),
        "runtime": runtime,
        "source_files": source_files,
        "derived_source_files": derived,
        "auxiliary_files": auxiliary_files,
        "output_files": output_files,
    }
    (output / RECEIPT).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    verify(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    environment = subparsers.add_parser(
        "check-env", help="Check converter bytes, checkout identity, imports and installed distribution"
    )
    environment.add_argument("--framework-dir", type=Path, required=True)
    fresh = subparsers.add_parser("convert", help="Download, convert and write a content-bound receipt")
    fresh.add_argument("--framework-dir", type=Path, required=True)
    fresh.add_argument("--source-dir", type=Path, required=True)
    fresh.add_argument("--output-dir", type=Path, required=True)
    reuse = subparsers.add_parser("verify", help="Verify a receipt and every DCP file, without importing Framework")
    reuse.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.operation == "check-env":
            print(json.dumps(check_environment(args.framework_dir.resolve()), indent=2))
        elif args.operation == "convert":
            convert(args.framework_dir.resolve(), args.source_dir.resolve(), args.output_dir.resolve())
        else:
            verify(args.output_dir.resolve())
    except (ValueError, OSError, subprocess.CalledProcessError, importlib.metadata.PackageNotFoundError) as error:
        parser.exit(1, f"Nano conversion check failed: {error}\n")


if __name__ == "__main__":
    main()
