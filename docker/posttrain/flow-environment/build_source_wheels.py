# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Package byte-identical upstream modules with the Cosmos flow runtime contract.

This is deliberately a named derived distribution, not an edit to upstream SDK
metadata or a --no-deps installation. Its supported API is the Cosmos SFT model
integration; upstream deployment/training command lines are outside this closure.
Builds from immutable git archives and includes the upstream license and revision.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

SOURCES = {
    "gr00t": {
        "revision": "51d4c89f72fda44cbf77285c6a8114b52676b8a1",
        "directory": "gr00t",
        "distribution": "cosmos-framework-flow-gr00t",
        "version": "0.1.0+51d4c89",
        "repository": "https://github.com/NVIDIA/Isaac-GR00T",
        "dependencies": [
            "torch==2.10.0",
            "torchvision==0.25.0",
            "transformers==4.57.6",
            "diffusers==0.35.1",
            "albumentations==1.4.18",
            "opencv-python-headless==4.11.0.86",
            "dm-tree",
            "lmdb==1.7.5",
            "msgpack",
            "msgpack-numpy==0.4.8",
            "pandas==2.2.3",
            "peft==0.17.1",
            "termcolor",
            "tyro==0.9.17",
            "datasets",
            "einops",
            "jsonlines==4.0.0",
            "gymnasium==1.2.2",
            "numpy==1.26.4",
            "omegaconf",
            "scipy",
            "torchcodec==0.10.0",
            "wandb",
            "pyzmq",
            "safetensors==0.8.0",
            "imageio",
            "av==18.0.0",
        ],
    },
    "alpamayo": {
        "revision": "11a0e01c13a5622377c45ee37d653351453ec43b",
        "directory": "src/alpamayo_r1",
        "distribution": "cosmos-framework-flow-alpamayo",
        "version": "0.1.0+11a0e01",
        "repository": "https://github.com/NVlabs/alpamayo",
        "dependencies": [
            "torch==2.10.0",
            "torchvision==0.25.0",
            "transformers==4.57.6",
            "colorlog",
            "einops>=0.8.1",
            "hydra-core>=1.3.2",
            "physical-ai-av==0.2.3",
            "pillow==12.2.0",
            "scipy",
        ],
    },
}


def build(source: Path, kind: str, output: Path) -> dict:
    spec = SOURCES[kind]
    archive = subprocess.check_output(
        ["git", "-C", str(source), "archive", spec["revision"], spec["directory"], "LICENSE"]
    )
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cosmos-flow-source-") as tmp:
        root = Path(tmp)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(root, filter="data")
        package = root / spec["directory"]
        files = {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in package.rglob("*")
            if p.is_file()
        }
        record = {**spec, "source_files": files, "git_archive_sha256": hashlib.sha256(archive).hexdigest()}
        (package / "cosmos_source_provenance.json").write_text(json.dumps(record, indent=2) + "\n")
        metadata = "\n".join(
            [
                "[build-system]",
                'requires = ["hatchling"]',
                'build-backend = "hatchling.build"',
                "[project]",
                f"name = {json.dumps(spec['distribution'])}",
                f"version = {json.dumps(spec['version'])}",
                'requires-python = ">=3.12,<3.13"',
                'license = {file = "LICENSE"}',
                f"dependencies = {json.dumps(spec['dependencies'])}",
                "[tool.hatch.build.targets.wheel]",
                f"packages = [{json.dumps(spec['directory'])}]",
            ]
        )
        (root / "pyproject.toml").write_text(metadata + "\n")
        subprocess.run(
            [sys.executable, "-m", "build", "--wheel", "--outdir", str(output.resolve()), str(root)], check=True
        )
        wheel = next(output.glob(spec["distribution"].replace("-", "_") + "-" + spec["version"] + "-*.whl"))
        with zipfile.ZipFile(wheel) as z:
            for path, digest in files.items():
                name = path.removeprefix("src/")
                if hashlib.sha256(z.read(name)).hexdigest() != digest:
                    raise ValueError(f"upstream source bytes changed while packaging: {name}")
        record["wheel_sha256"] = hashlib.sha256(wheel.read_bytes()).hexdigest()
        (output / f"{kind}-source-build.json").write_text(json.dumps(record, indent=2) + "\n")
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gr00t", type=Path, required=True)
    parser.add_argument("--alpamayo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for kind in SOURCES:
        result = build(getattr(args, kind), kind, args.out)
        print(kind, result["revision"], result["wheel_sha256"])


if __name__ == "__main__":
    main()
