#!/usr/bin/env -S uv run --script
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# https://docs.astral.sh/uv/guides/scripts/#using-a-shebang-to-create-an-executable-file
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "accelerate>=1.10.1",
#   "qwen-vl-utils>=0.0.11",
#   "torchcodec>=0.6.0",
#   "torch>=2.7.1",
#   "transformers>=4.51.3",
#   "vllm>=0.10.1.1",
# ]
# [tool.uv.sources]
# torch = [
#   { index = "pytorch-cu128"},
# ]
# torchvision = [
#   { index = "pytorch-cu128"},
# ]
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# ///

"""Minimal example of inference with Cosmos-Reason1.

Examples:

Single file:
```shell
python -m cosmos_framework.scripts.inference_reasoner --model_path /path/to/model --media test_video.mp4 --type video --prompt "Describe this video." --results_dir /path/to/results
```

Folder of files:
```shell
python -m cosmos_framework.scripts.inference_reasoner --model_path /path/to/model --media /path/to/videos --type video --prompt "Describe this video." --results_dir /path/to/results
```
"""

import argparse
import json
import time
from pathlib import Path

from cosmos_framework.inference.reasoner.runtime import CosmosFrameworkRuntime
from cosmos_framework.utils.workflow_status import (
    Status,
    Verbosity,
    get_status_logger,
    log_workflow_status,
    monitor_status,
)

ROOT = Path(__file__).parents[1]
SEPARATOR = "-" * 20
COMPONENT_NAME = "Cosmos Framework Inference"

# Common video and image file extensions
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".m4v", ".mpg", ".mpeg", ".3gp"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff", ".tif", ".heic", ".heif"}


def str_to_bool(value):
    """Convert string to boolean for argparse."""
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif value.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got: {value}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Cosmos-Reason1 inference script")

    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument(
        "--torch_dtype", type=str, default="auto", help="PyTorch data type (auto, float16, float32, bfloat16)"
    )
    parser.add_argument(
        "--device_map", type=str, default="auto", help="Device mapping strategy (auto, cpu, cuda, etc.)"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Number of GPUs to use for tensor parallelism (requires device_map='auto')",
    )
    parser.add_argument(
        "--type", type=str, default="video", choices=["video", "image"], help="Input type (video or image)"
    )
    parser.add_argument(
        "--media", type=str, required=True, help="Path to video/image file or folder containing videos/images"
    )
    parser.add_argument("--fps", type=int, default=4, help="Frames per second for video processing")
    parser.add_argument(
        "--total_pixels", type=int, default=6422528, help="Total pixels for video processing (8192 * 28**2)"
    )
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for the model")
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="Maximum number of new tokens to generate")
    parser.add_argument("--results_dir", type=str, required=True, help="Results directory")
    parser.add_argument("--enable_lora", type=str_to_bool, default=False, help="Enable LoRA model merging")
    parser.add_argument(
        "--base_model_path", type=str, help="Base model path for LoRA merging (required if enable_lora is True)"
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Framework config.yaml for a DCP checkpoint (inferred for standard run layouts)",
    )
    parser.add_argument(
        "--export_dir",
        type=str,
        default=None,
        help="Optional directory for the automatic DCP-to-HF export",
    )
    parser.add_argument(
        "--vit_checkpoint_path",
        type=str,
        default=None,
        help="Optional local base checkpoint used to bundle the vision tower during export",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=None,
        help="Uniformly sample this many frames from each video",
    )

    return parser.parse_args()


def get_media_files(media_path: str, media_type: str) -> list[Path]:
    """Get list of media files from a path (file or folder).

    Args:
        media_path: Path to a file or folder
        media_type: Type of media ('video' or 'image')

    Returns:
        List of Path objects to media files
    """
    path = Path(media_path)

    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {media_path}")

    if path.is_file():
        return [path]

    if path.is_dir():
        extensions = VIDEO_EXTENSIONS if media_type == "video" else IMAGE_EXTENSIONS
        media_files = [f for f in path.iterdir() if f.is_file() and f.suffix.lower() in extensions]
        if not media_files:
            raise ValueError(
                f"No {media_type} files found in directory: {media_path}. Supported extensions: {sorted(extensions)}"
            )
        return sorted(media_files)

    raise ValueError(f"Path is neither a file nor a directory: {media_path}")


def process_single_media(
    runtime,
    media_file: Path,
    media_type: str,
    prompt: str,
    fps: int,
    total_pixels: int,
    num_frames: int | None,
    max_new_tokens: int,
    results_dir: Path,
    s_logger,
) -> dict:
    """Process a single media file and return the result dict.

    Args:
        model: Loaded model
        processor: Loaded processor
        media_file: Path to the media file
        media_type: Type of media ('video' or 'image')
        prompt: Text prompt
        fps: Frames per second for video
        total_pixels: Total pixels for video
        max_new_tokens: Maximum tokens to generate
        results_dir: Directory to save results
        s_logger: Status logger

    Returns:
        Dict with keys: file, prompt, response, tokens_generated, inference_time_s
    """
    media_str = str(media_file)
    s_logger.write(status_level=Status.RUNNING, message=f"Processing {media_type}: {media_file.name}")

    infer_start = time.time()
    responses, _ = runtime.generate_tasks(
        [
            {
                "id": media_file.stem,
                "question": prompt,
                "prompt": [{"role": "user", "content": prompt}],
                "media_paths": [media_str],
                "media_mode": media_type,
            }
        ],
        generation_config={"max_tokens": max_new_tokens, "temperature": 0.0},
        vision_config={"fps": fps, "num_frames": num_frames, "total_pixels": total_pixels},
    )
    inference_time = time.time() - infer_start
    result_text = responses[0]
    tokens_generated = len(result_text.split())

    result_record = {
        "file": media_file.name,
        "prompt": prompt,
        "response": result_text,
        "tokens_generated": tokens_generated,
        "inference_time_s": round(inference_time, 3),
    }

    txt_file = results_dir / f"{media_file.stem}_result.txt"
    txt_file.write_text(result_text, encoding="utf-8")

    json_file = results_dir / f"{media_file.stem}_result.json"
    json_file.write_text(json.dumps(result_record, indent=2, ensure_ascii=False), encoding="utf-8")

    s_logger.write(
        status_level=Status.SUCCESS, message=f"Completed {media_file.name} - Result saved to {json_file.name}"
    )

    return result_record


def main():
    args = parse_args()

    # Apply monitor_status decorator with results_dir from args
    @monitor_status(name="Cosmos Framework Inference", mode="inference", results_dir=args.results_dir)
    def run_inference():
        s_logger = get_status_logger()

        s_logger.write(status_level=Status.RUNNING, message=f"Loading model from: {args.model_path}")

        if args.num_gpus is not None and args.num_gpus > 0:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available. Cannot use --num_gpus.")

            num_available_gpus = torch.cuda.device_count()
            if args.num_gpus > num_available_gpus:
                raise ValueError(f"Requested {args.num_gpus} GPUs but only {num_available_gpus} available")

            if args.num_gpus != 1:
                raise ValueError(
                    "Framework inference uses one model replica per process. "
                    "Use torchrun for multi-GPU data parallel inference."
                )

        if args.enable_lora:
            raise ValueError(
                "Cosmos Framework actions require a full checkpoint; merge/export the adapter before inference."
            )
        runtime = CosmosFrameworkRuntime(
            args.model_path,
            config_file=args.config_file,
            export_dir=args.export_dir,
            vit_checkpoint_path=args.vit_checkpoint_path,
            dtype=args.torch_dtype,
            device_map=args.device_map,
        )
        model_path = runtime.model_path

        s_logger.write(status_level=Status.SUCCESS, message="Model and processor loaded successfully")

        pipeline_start = time.time()

        try:
            media_files = get_media_files(args.media, args.type)
            results_dir_path = Path(args.results_dir)

            inference_output_dir = results_dir_path / "inference_results"
            inference_output_dir.mkdir(parents=True, exist_ok=True)

            s_logger.write(
                status_level=Status.RUNNING, message=f"Found {len(media_files)} {args.type} file(s) to process"
            )

            log_workflow_status(
                data={
                    "inference_status": "started",
                    "model_path": args.model_path,
                    "media_path": args.media,
                    "total_files": len(media_files),
                    "media_type": args.type,
                    "results_dir": str(inference_output_dir),
                },
                component_name=COMPONENT_NAME,
            )

            successful_results = []
            failed_files = []

            for idx, media_file in enumerate(media_files, 1):
                try:
                    s_logger.write(
                        status_level=Status.RUNNING,
                        message=f"Processing file {idx}/{len(media_files)}: {media_file.name}",
                    )

                    result = process_single_media(
                        runtime=runtime,
                        media_file=media_file,
                        media_type=args.type,
                        prompt=args.prompt,
                        fps=args.fps,
                        total_pixels=args.total_pixels,
                        num_frames=args.num_frames,
                        max_new_tokens=args.max_new_tokens,
                        results_dir=inference_output_dir,
                        s_logger=s_logger,
                    )

                    successful_results.append(result)

                    s_logger.write(
                        status_level=Status.SUCCESS,
                        message=f"Inference result for {media_file.name}: {result['response'][:200]}",
                    )

                    print(SEPARATOR)
                    print(f"File: {media_file.name}")
                    print(SEPARATOR)
                    print(result["response"])
                    print(SEPARATOR)
                    print()

                except KeyboardInterrupt:
                    s_logger.write(
                        status_level=Status.FAILURE,
                        message=f"Inference was interrupted while processing {media_file.name}",
                        verbosity_level=Verbosity.WARNING,
                    )
                    raise
                except Exception as e:
                    error_msg = f"Failed to process {media_file.name}: {str(e)}"
                    s_logger.write(status_level=Status.FAILURE, message=error_msg, verbosity_level=Verbosity.ERROR)
                    failed_files.append({"file": media_file.name, "error": str(e)})
                    print(f"ERROR: {error_msg}")
                    continue

            total_time = time.time() - pipeline_start
            total_tokens = sum(r["tokens_generated"] for r in successful_results)

            summary = {
                "inference_status": "completed",
                "total_files": len(media_files),
                "successful": len(successful_results),
                "failed": len(failed_files),
                "total_tokens_generated": total_tokens,
                "total_time_s": round(total_time, 3),
                "model_path": args.model_path,
                "media_type": args.type,
                "prompt": args.prompt,
                "results_path": str(inference_output_dir),
                "results": successful_results,
            }
            if failed_files:
                summary["failed_files"] = failed_files

            summary_file = inference_output_dir / "inference_summary.json"
            summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

            kpi_data = {
                "inference_status": "completed",
                "total_files": len(media_files),
                "successful": len(successful_results),
                "failed": len(failed_files),
                "total_tokens_generated": total_tokens,
                "total_time_s": round(total_time, 3),
                "results_path": str(inference_output_dir),
            }
            for r in successful_results:
                kpi_data[f"result_{r['file']}"] = r["response"][:500]

            log_workflow_status(data=kpi_data, component_name=COMPONENT_NAME)

            s_logger.write(
                status_level=Status.SUCCESS if not failed_files else Status.RUNNING,
                message=(
                    f"Processing complete: {len(successful_results)} successful, "
                    f"{len(failed_files)} failed, {total_tokens} tokens in {total_time:.1f}s"
                ),
            )

            if failed_files:
                print("\n" + SEPARATOR)
                print("FAILED FILES:")
                print(SEPARATOR)
                for entry in failed_files:
                    print(f"{entry['file']}: {entry['error']}")
                print(SEPARATOR + "\n")

        except KeyboardInterrupt:
            s_logger.write(
                status_level=Status.FAILURE,
                message="Inference was interrupted by user (Ctrl+C)",
                verbosity_level=Verbosity.WARNING,
            )
            log_workflow_status(
                data={"inference_status": "interrupted", "error": "User interrupted"}, component_name=COMPONENT_NAME
            )
            raise

        except Exception as e:
            error_msg = f"Inference failed: {str(e)}"
            s_logger.write(status_level=Status.FAILURE, message=error_msg, verbosity_level=Verbosity.ERROR)
            log_workflow_status(data={"inference_status": "failed", "error": str(e)}, component_name=COMPONENT_NAME)
            raise

    # Execute the decorated inference function
    run_inference()


if __name__ == "__main__":
    main()
