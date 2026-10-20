# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Operational GPU probe for the native video-QA recipe, not a training run.

May be staged and invoked as a script inside an existing Framework image.
It imports that image's implementation; it never replaces model/package code.
"""

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import tomllib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-dir", type=Path, required=True)
    parser.add_argument("--gpus", type=int, required=True)
    args = parser.parse_args()
    trial = args.trial_dir.resolve()
    output = trial / "preflight"
    output.mkdir(exist_ok=True)
    recipe = tomllib.loads((trial / "config.toml").read_text())
    evaluation = tomllib.loads((trial / "evaluation.toml").read_text())
    model = Path(recipe["model"]["backbone"]["model_name"])
    report = {"checks": {}}

    def check(name, operation):
        started = time.monotonic()
        try:
            detail = operation()
            result = {"passed": True, "detail": detail}
        except Exception as exc:
            result = {"passed": False, "error_type": type(exc).__name__, "message": str(exc)}
        result["seconds"] = round(time.monotonic() - started, 3)
        report["checks"][name] = result
        (output / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({name: result}), flush=True)

    def assets():
        if os.getuid() == 0 or not os.access(output, os.W_OK):
            raise ValueError("Run as the submitting user with writable trial output")
        if not model.is_dir():
            raise ValueError("The explicit local base model is not available")
        records = {}
        for split, annotation, media in (
            ("train", os.environ["COSMOS_VIDEO_TRAIN_ANNOTATION"], os.environ["COSMOS_VIDEO_TRAIN_MEDIA"]),
            ("eval", evaluation["dataset"]["annotation_path"], evaluation["dataset"]["media_dir"]),
        ):
            raw = Path(annotation).read_bytes()
            rows = json.loads(raw)
            if not rows or not all((Path(media) / row["video"]).is_file() for row in rows):
                raise ValueError(f"Empty annotations or missing media: {split}")
            records[split] = rows
            report[split] = {"records": len(rows), "sha256": hashlib.sha256(raw).hexdigest()}
        if {r["id"] for r in records["train"]} & {r["id"] for r in records["eval"]}:
            raise ValueError("Training and validation IDs overlap")
        if {r["video"] for r in records["train"]} & {r["video"] for r in records["eval"]}:
            raise ValueError("Training and validation video references overlap")
        if os.environ.get("COSMOS_VIDEO_TRAIN_LIMIT") or os.environ.get("COSMOS_VIDEO_VAL_LIMIT"):
            raise ValueError("Full-data preflight does not permit environment dataset limits")
        if evaluation.get("evaluation", {}).get("limit") not in (None, -1):
            raise ValueError("Full-data preflight does not permit an evaluation limit")
        return {key: report[key] for key in ("train", "eval")}

    def imports():
        versions = {}
        for name in ("torch", "torchvision", "torchcodec", "transformers", "transformer_engine.pytorch", "flash_attn"):
            module = importlib.import_module(name)
            versions[name] = str(getattr(module, "__version__", "not exposed"))
        importlib.import_module("cosmos_framework.automl.evaluate_trial")
        return versions

    def kernels():
        import torch
        from flash_attn import flash_attn_func

        if not torch.cuda.is_available() or torch.cuda.device_count() != args.gpus:
            raise ValueError("CUDA visibility does not match the allocated topology")
        a = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        q = torch.randn(1, 32, 2, 64, device="cuda", dtype=torch.bfloat16)
        if not torch.isfinite(a @ a.T).all() or not torch.isfinite(flash_attn_func(q, q, q, causal=True)).all():
            raise ValueError("Nonfinite GPU kernel result")
        torch.cuda.synchronize()
        return {"gpu": torch.cuda.get_device_name(), "visible_gpus": torch.cuda.device_count()}

    def checkpoint():
        import torch
        from transformers import AutoConfig, AutoProcessor
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionPatchEmbed

        from cosmos_framework.model.generator.qwen3_vl_compat import apply_qwen3_vl_patch_embed_compat

        config = AutoConfig.from_pretrained(model, local_files_only=True)
        processor = AutoProcessor.from_pretrained(model, local_files_only=True)
        if config.model_type != "qwen3_vl":
            raise ValueError("This video-QA preflight targets a Qwen3-VL reasoner")
        module = Qwen3VLVisionPatchEmbed(config.vision_config).to(device="cuda", dtype=torch.bfloat16)
        keys = list(module.state_dict())
        applied = apply_qwen3_vl_patch_embed_compat(module, model_type=config.model_type, mode="auto")
        elements = module.in_channels * module.temporal_patch_size * module.patch_size**2
        result = module(torch.randn(4, elements, device="cuda", dtype=torch.bfloat16))
        if keys != list(module.state_dict()) or not torch.isfinite(result).all():
            raise ValueError("Native patch embedding probe failed")
        return {"processor": type(processor).__name__, "native_linear_patch_applied": applied}

    def decoder():
        from cosmos_framework.inference.reasoner.framework_torchcodec_video import FrameworkTorchCodecVideoPreprocessor
        from cosmos_framework.utils.generator.torchcodec_video import TorchCodecVideoReader

        vision = evaluation["vision"]
        if vision["video_decoder"] != "torchcodec-cuda-on-demand":
            raise ValueError("This preflight validates the explicit TorchCodec CUDA evaluator")
        FrameworkTorchCodecVideoPreprocessor(
            num_frames=vision["num_frames"], cache_size=0, device=vision["decoder_device"]
        )
        shapes = []
        for annotation, media in (
            (os.environ["COSMOS_VIDEO_TRAIN_ANNOTATION"], os.environ["COSMOS_VIDEO_TRAIN_MEDIA"]),
            (evaluation["dataset"]["annotation_path"], evaluation["dataset"]["media_dir"]),
        ):
            video = json.loads(Path(annotation).read_text())[0]["video"]
            reader = TorchCodecVideoReader(str(Path(media) / video), device="cuda:0")
            frames = reader.get_frames_nhwc_uint8([0, len(reader) - 1])
            if not str(reader.last_output_device).startswith("cuda") or frames.shape[-1] != 3:
                raise ValueError("Selected decoder did not produce CUDA RGB frames")
            shapes.append(list(frames.shape))
        return shapes

    def dryrun():
        env = os.environ.copy()
        env["IMAGINAIRE_OUTPUT_ROOT"] = str(output / "dryrun")
        command = json.loads((trial / "_execution.json").read_text())["commands"][0]
        training_args = command[command.index("cosmos_framework.scripts.train") + 1 :]
        with (output / "native-dryrun.log").open("w") as log:
            subprocess.run(
                [sys.executable, "-m", "cosmos_framework.scripts.train", "--dryrun", *training_args],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=240,
                check=True,
            )
        if not list((output / "dryrun").rglob("config.yaml")):
            raise ValueError("Native dryrun did not produce a resolved configuration")
        return {"training_launched": False}

    for name, operation in (
        ("assets", assets),
        ("imports", imports),
        ("gpu_kernels", kernels),
        ("checkpoint", checkpoint),
        ("decoder", decoder),
        ("native_dryrun", dryrun),
    ):
        check(name, operation)
    return 0 if all(item["passed"] for item in report["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
