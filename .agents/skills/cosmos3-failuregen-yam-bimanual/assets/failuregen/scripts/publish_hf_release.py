#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Export regular FP32 DCP weights, package example inputs, and upload privately.

Run inside the existing Cosmos container on a Slurm compute node. No source
checkpoint files are modified. Serialized tensor chunks are copied byte-for-byte.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pathlib
import pickle
import shutil
import sys
import uuid
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
import yaml
from huggingface_hub import HfApi, snapshot_download
from torch.distributed.checkpoint import FileSystemReader

sys.modules.setdefault("pathlib._local", pathlib)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def compact_model(source, target):
    report_path = target.parent / "export.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report["source_metadata_sha256"] != digest(source / ".metadata"):
            raise ValueError("Source checkpoint changed")
        for name, expected in report["files"].items():
            if digest(target / name) != expected:
                raise ValueError(f"Existing export changed: {name}")
        return report
    if target.exists():
        raise ValueError("Incomplete export; use a fresh release directory")
    target.mkdir(parents=True)
    metadata = FileSystemReader(source).read_metadata()
    keep = {name for name in metadata.state_dict_metadata if name.startswith("net.")}
    if len(keep) != 809 or any(metadata.state_dict_metadata[k].properties.dtype != torch.float32 for k in keep):
        raise ValueError("Unexpected model keys or precision; inspect before exporting")
    exported = copy.deepcopy(metadata)
    exported.state_dict_metadata = {k: v for k, v in exported.state_dict_metadata.items() if k in keep}
    exported.planner_data = {k: v for k, v in exported.planner_data.items() if k in keep}
    exported.storage_data = {k: v for k, v in exported.storage_data.items() if k.fqn in keep}
    if exported.storage_meta is not None:
        exported.storage_meta.checkpoint_id = target
        exported.storage_meta.save_id = str(uuid.uuid4())
    by_file = {}
    for index, info in exported.storage_data.items():
        by_file.setdefault(info.relative_path, []).append((index, info))
    for name, entries in sorted(by_file.items()):
        path = target / name
        expected = hashlib.sha256()
        with (source / name).open("rb") as reader, path.open("wb") as writer:
            for _, info in sorted(entries, key=lambda x: x[1].offset):
                reader.seek(info.offset)
                info.offset = writer.tell()
                remaining = info.length
                while remaining:
                    chunk = reader.read(min(8 * 1024**2, remaining))
                    if not chunk:
                        raise EOFError(name)
                    writer.write(chunk)
                    expected.update(chunk)
                    remaining -= len(chunk)
        if digest(path) != expected.hexdigest():
            raise ValueError(f"Copied tensor bytes differ: {name}")
        print("EXPORTED", name, path.stat().st_size, flush=True)
    with (target / ".metadata").open("wb") as f:
        pickle.dump(exported, f)
    # Exercise PyTorch's actual loader and compare reconstructed tensors from
    # all source shards with the compacted checkpoint, without loading 60 GB.
    samples = sorted(keep, key=lambda k: metadata.state_dict_metadata[k].size.numel())[:8]
    for name in samples:
        shape = metadata.state_dict_metadata[name].size
        old = {name: torch.empty(shape, dtype=torch.float32)}
        new = {name: torch.empty(shape, dtype=torch.float32)}
        dcp.load(old, checkpoint_id=source, no_dist=True)
        dcp.load(new, checkpoint_id=target, no_dist=True)
        if not torch.equal(old[name], new[name]):
            raise ValueError(f"DCP round-trip mismatch: {name}")
    report = dict(
        source_metadata_sha256=digest(source / ".metadata"),
        retained_prefix="net.",
        removed_prefix="net_ema.",
        dtype="float32",
        tensor_count=len(keep),
        byte_identical_serialized_chunks=True,
        dcp_roundtrip_tensors=samples,
        files={p.name: digest(p) for p in target.iterdir() if p.is_file()},
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", default="nvidia/failure-gen-yam-bimanual")
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--cookbook", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.output.resolve()
    token = args.token_file.read_text().strip()
    api = HfApi(token=token)
    info = api.repo_info(args.repo, repo_type="model")
    if not info.private:
        raise ValueError("Refusing upload: target must already be private")
    print("PRIVATE_REPO", info.id, flush=True)
    work = root / "outputs/yam-eval-eef-gb200-v6"
    run = work / "training/cosmos3_yam/fd_id/yam_eval_eef_gb200_v6"
    checkpoint = run / "checkpoints/iter_000001243/model"
    target = out / "checkpoints/iter_000001243/model"
    report = compact_model(checkpoint, target)

    def copy_file(source, relative):
        dest = out / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if dest.stat().st_size != source.stat().st_size or digest(dest) != digest(source):
                raise ValueError(f"Existing file differs: {relative}")
        else:
            shutil.copy2(source, dest)

    for name in ("manifest.json", "action_stats.json"):
        copy_file(work / "prepared" / name, Path("example_data/prepared") / name)
    copy_file(work / "dataset/meta/info.json", Path("example_data/dataset/meta/info.json"))
    manifest = json.loads((work / "prepared/manifest.json").read_text())
    for ep in manifest["episodes"]:
        if ep["episode"] not in (40, 42, 43, 46, 61, 79):
            continue
        copy_file(work / "prepared" / ep["states"], Path("example_data/prepared") / ep["states"])
        for name in ep["videos"].values():
            copy_file(work / "dataset" / name, Path("example_data/dataset") / name)
    copy_file(run / "yam_training_contract.json", Path("yam_training_contract.json"))
    copy_file(root / "checkpoints/wan22_vae/Wan2.2_VAE.pth", Path("auxiliary/Wan2.2_VAE.pth"))
    tokenizer_repo = "Qwen/Qwen3-VL-8B-Instruct"
    tokenizer_info = api.repo_info(tokenizer_repo, repo_type="model")
    snapshot_download(
        tokenizer_repo,
        revision=tokenizer_info.sha,
        token=token,
        local_dir=out / "auxiliary/tokenizer",
        allow_patterns=[
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "added_tokens.json",
            "special_tokens_map.json",
            "chat_template.json",
            "chat_template.jinja",
            "preprocessor_config.json",
            "processor_config.json",
        ],
    )
    config = {"model": yaml.safe_load((run / "config.yaml").read_text())["model"]}
    model = config["model"]["config"]
    assert not model.get("exclude_reasoner_weights_from_checkpoint")
    model["tokenizer"]["vae_path"] = "auxiliary/Wan2.2_VAE.pth"
    model["vlm_config"]["tokenizer"]["pretrained_model_name"] = "auxiliary/tokenizer"
    model["vlm_config"]["tokenizer"]["config_variant"] = "hf"
    model["vlm_config"]["pretrained_weights"]["enabled"] = False
    model["vlm_config"]["pretrained_weights"]["backbone_path"] = ""
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    release = dict(
        format_version=1,
        checkpoint="checkpoints/iter_000001243",
        dataset="example_data/dataset",
        prepared="example_data/prepared",
        vae="auxiliary/Wan2.2_VAE.pth",
        tokenizer="auxiliary/tokenizer",
        tokenizer_repository=tokenizer_repo,
        tokenizer_revision=tokenizer_info.sha,
        example_episodes=[40, 42, 43, 46, 61, 79],
        generation_seed=0,
        original_prepared_manifest_preserved=True,
        model_export=report,
        framework="Compatible Cosmos framework and four GB200 GPUs required.",
    )
    (out / "release.json").write_text(json.dumps(release, indent=2) + "\n")
    (out / "README.md").write_text("""---
tags:
- robotics
- video-generation
- cosmos
---
# FailureGen YAM bimanual inference bundle

Private inference release adapted from Cosmos3-Nano, iteration 1243.
Contains regular FP32 weights only; EMA and optimizer/trainer state are omitted.
Tensor payloads are unchanged. This is not a training-resume checkpoint.

Includes the Wan 2.2 VAE, tokenizer files, and source videos/prepared actions for
three donor/host pairs: 42/79, 40/43, and 61/46. All examples use seed 0.
No synthetic output videos or annotation documents are included.
The frozen prepared manifest is preserved for checkpoint compatibility; only the
six example episodes' state archives and videos are included.

Use the FailureGen cookbook's release downloader to resolve asset paths locally.
A compatible Cosmos framework environment with four GB200 GPUs is required.
The source model and auxiliary assets retain their respective license terms.
""")
    # CPU preflight using only staged files; no source dataset or annotation reads.
    sys.path.insert(0, str(args.cookbook / "src"))
    from failuregen.release import materialize_run
    from failuregen.showcase import prepare

    validation = out.parent / (out.name + "-validation")
    runtime = materialize_run(out, validation)
    prepare(
        out / release["prepared"],
        out / release["dataset"],
        runtime,
        out / release["checkpoint"],
        args.cookbook / "configs/showcase.json",
        validation / "showcase",
    )
    print("THREE_EXAMPLE_CPU_PREFLIGHT_OK", flush=True)
    files = [p for p in out.rglob("*") if p.is_file() and ".cache" not in p.parts]
    hashes = {str(p.relative_to(out)): digest(p) for p in files}
    (out / "SHA256SUMS.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print("UPLOAD_START", len(files) + 1, sum(p.stat().st_size for p in files), flush=True)
    commit = api.upload_folder(
        repo_id=args.repo,
        repo_type="model",
        folder_path=out,
        ignore_patterns=["**/.cache/**"],
        commit_message="Add inference-only YAM FD checkpoint and runnable example inputs",
    )
    remote = api.repo_info(args.repo, repo_type="model", revision=commit.oid, files_metadata=True)
    assert remote.private
    siblings = {s.rfilename: s for s in remote.siblings}
    for name, digest_value in hashes.items():
        entry = siblings[name]
        assert entry.size == (out / name).stat().st_size, name
        if entry.lfs:
            remote_hash = entry.lfs.sha256 if hasattr(entry.lfs, "sha256") else entry.lfs["sha256"]
            assert remote_hash == digest_value, name
    result = dict(
        repo=args.repo,
        revision=commit.oid,
        url=str(commit.commit_url),
        private=remote.private,
        verified_files=len(hashes),
        bytes=sum(p.stat().st_size for p in files),
    )
    (out.parent / (out.name + "-upload.json")).write_text(json.dumps(result, indent=2) + "\n")
    print("UPLOAD_VERIFIED", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
