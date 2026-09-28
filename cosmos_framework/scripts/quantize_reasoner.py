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

"""Quantization script for Cosmos Framework.

Example:

```shell
cosmos-reasoner-quantize --model_path nvidia/Cosmos-Reason1-7B --results_dir /results/quantized_model
```
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import shutil
import sys
import types
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

from cosmos_framework.checkpoint.lora import get_sequential_targets, load_model_and_processor
from cosmos_framework.utils.workflow_status import (
    Status,
    Verbosity,
    get_status_logger,
    log_workflow_status,
    monitor_status,
)

# Constants
COMPONENT_NAME = "Cosmos Framework Quantization"
SEPARATOR = "-" * 50
TOKENIZER_PROCESSOR_FILENAMES = (
    "added_tokens.json",
    "chat_template.jinja",
    "chat_template.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _load_quantization_dependencies():
    """Load the isolated compression environment only after CLI validation."""
    global requests, torch, HFDataset, load_dataset, snapshot_download
    global oneshot, QuantizationModifier, SmoothQuantModifier, dispatch_for_generation
    global Image, process_vision_info
    overlay = os.environ.get("COSMOS_QUANTIZE_DEPS", "/opt/quantize_deps")
    if os.path.isdir(overlay):
        sys.path.insert(0, overlay)
    import compressed_tensors.utils.match as match
    from compressed_tensors.config import CompressionFormat

    # Compatibility required by the pinned compressor and compressed-tensors.
    if not hasattr(match, "_match_name") and hasattr(match, "match_name"):
        match._match_name = match.match_name
    format_module = types.ModuleType("compressed_tensors.config.format")

    def quant_format(input_activations=None, weights=None):
        qtype = getattr(weights, "type", None) or getattr(input_activations, "type", None)
        return CompressionFormat.float_quantized if str(qtype).lower() == "float" else CompressionFormat.int_quantized

    format_module._get_quant_compression_format = quant_format
    sys.modules[format_module.__name__] = format_module
    import requests
    import torch
    from datasets import Dataset as HFDataset
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
    from llmcompressor.utils import dispatch_for_generation
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    from cosmos_framework.inference.reasoner.system_pyav_video_reader import register_system_pyav_video_reader

    register_system_pyav_video_reader()


def preprocess_and_tokenize(example, processor, max_sequence_length):
    """Apply chat template and tokenize inputs."""
    buffered = BytesIO()
    example["image"].save(buffered, format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue())
    encoded_image_text = encoded_image.decode("utf-8")
    base64_qwen = f"data:image;base64,{encoded_image_text}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": base64_qwen},
                {"type": "text", "text": "What does the image show?"},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    return processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=False,
        max_length=max_sequence_length,
        truncation=True,
    )


def data_collator(batch):
    """Oneshot data collator for multimodal inputs."""
    assert len(batch) == 1
    return {key: torch.tensor(value) for key, value in batch[0].items()}


def load_custom_dataset(annotation_path: str, media_dir: str = None) -> List[Dict[str, Any]]:
    """
    Load custom dataset from annotation JSON file.

    Args:
        annotation_path: Path to annotation JSON file
        media_dir: Optional directory containing media files (prepended to relative paths)

    Returns:
        List of dataset samples with 'image' field containing PIL Image objects
    """
    logger.info(f"Loading custom dataset from: {annotation_path}")

    with open(annotation_path, "r") as f:
        annotations = json.load(f)

    logger.info(f"Loaded {len(annotations)} annotations")

    samples = []
    for item in annotations:
        # Extract image or video path(s)
        images = item.get("image") or item.get("images")
        videos = item.get("video") or item.get("videos")

        # For videos, extract first frame as image for calibration
        if videos and not images:
            try:
                import cv2

                # Handle single video or list of videos
                if isinstance(videos, str):
                    video_path = videos
                elif isinstance(videos, list) and len(videos) > 0:
                    video_path = videos[0]
                else:
                    logger.warning(f"Invalid video format for sample: {item.get('id', 'unknown')}")
                    continue

                # Prepend media_dir if provided
                if media_dir:
                    video_path = os.path.join(media_dir, video_path)

                # Extract first frame from video
                if os.path.exists(video_path):
                    cap = cv2.VideoCapture(video_path)
                    ret, frame = cap.read()
                    cap.release()

                    if ret:
                        # Convert BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image = Image.fromarray(frame_rgb)
                        samples.append({"image": image})
                    else:
                        logger.warning(f"Failed to read first frame from video: {video_path}")
                else:
                    logger.warning(f"Video file not found: {video_path}")

            except ImportError:
                logger.warning(f"cv2 not available - skipping video sample: {item.get('id', 'unknown')}")
                continue
            except Exception as e:
                logger.warning(f"Failed to extract frame from video: {e}")
                continue

        elif images:
            # Handle single image or list of images
            if isinstance(images, str):
                image_path = images
            elif isinstance(images, list) and len(images) > 0:
                image_path = images[0]  # Use first image for calibration
            else:
                logger.warning(f"Invalid image format for sample: {item.get('id', 'unknown')}")
                continue

            # Prepend media_dir if provided
            if media_dir:
                image_path = os.path.join(media_dir, image_path)

            # Load image
            try:
                if os.path.exists(image_path):
                    image = Image.open(image_path).convert("RGB")
                    samples.append({"image": image})
                else:
                    logger.warning(f"Image file not found: {image_path}")
            except Exception as e:
                logger.warning(f"Failed to load image {image_path}: {e}")
                continue
        else:
            logger.warning(f"Skipping sample without image or video: {item.get('id', 'unknown')}")
            continue

    logger.info(f"Successfully loaded {len(samples)} samples from custom dataset")
    return samples


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
    parser = argparse.ArgumentParser(description="Cosmos Framework Quantization Script")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path or name of the model to quantize (e.g., nvidia/Cosmos-Reason1-7B)",
    )
    parser.add_argument(
        "--dataset_id",
        type=str,
        default="lmms-lab/flickr30k",
        help="HuggingFace dataset ID for calibration (e.g., lmms-lab/flickr30k). Leave empty to use custom dataset.",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="test[:512]",
        help="Dataset split for calibration (e.g., test[:512]). Only used with HuggingFace datasets.",
    )
    parser.add_argument(
        "--annotation_path",
        type=str,
        default=None,
        help="Path to custom annotation JSON file. Use this instead of dataset_id for local datasets.",
    )
    parser.add_argument(
        "--media_dir",
        type=str,
        default=None,
        help="Directory containing media files for custom dataset. Paths in annotations are relative to this directory.",
    )
    parser.add_argument("--num_calibration_samples", type=int, default=512, help="Number of calibration samples to use")
    parser.add_argument(
        "--max_sequence_length", type=int, default=2048, help="Maximum sequence length for tokenization"
    )
    parser.add_argument(
        "--results_dir", type=str, default="/results", help="Directory to save the quantized model and Cosmos logs"
    )
    parser.add_argument(
        "--quantization_scheme",
        type=str,
        default="FP8_DYNAMIC",
        choices=["FP8_DYNAMIC", "NVFP4", "FP8", "W8A8", "W8A16", "W4A16"],
        help="Quantization scheme to use",
    )
    parser.add_argument(
        "--kv_precision", type=str, default="bf16", choices=["bf16", "fp8"], help="Precision for KV cache quantization"
    )
    parser.add_argument(
        "--smoothing_strength", type=float, default=0.8, help="SmoothQuant smoothing strength (0.0 to 1.0)"
    )
    parser.add_argument(
        "--skip_test_generation", type=str_to_bool, default=False, help="Skip test generation after quantization"
    )
    parser.add_argument(
        "--enable_lora",
        type=str_to_bool,
        default=False,
        help="Enable LoRA model merging (required if model_path is a LoRA checkpoint)",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=None,
        help="Base model path for LoRA merging (required if enable_lora is True)",
    )

    return parser.parse_args()


def _postprocess_config(config_path: Path):
    """Remove unsupported dtype keys that can cause vLLM loading failures."""

    def _remove_keys(d, keys_to_remove):
        if isinstance(d, dict):
            return {k: _remove_keys(v, keys_to_remove) for k, v in d.items() if k not in keys_to_remove}
        elif isinstance(d, list):
            return [_remove_keys(i, keys_to_remove) for i in d]
        return d

    with open(config_path) as f:
        config = json.load(f)
    clean_config = _remove_keys(config, keys_to_remove=["zp_dtype", "scale_dtype"])
    with open(config_path, "w") as f:
        json.dump(clean_config, f, indent=2)
    logger.info(f"Postprocessed config: removed zp_dtype/scale_dtype keys from {config_path}")


def _ensure_preprocessor_pixel_config(model_dir: Path):
    """Save non-null processor pixel limits for vLLM loading."""
    preproc_path = model_dir / "preprocessor_config.json"
    if not preproc_path.exists():
        return

    with open(preproc_path) as f:
        config = json.load(f)

    defaults = {"min_pixels": 3136, "max_pixels": 12845056}
    updated = False
    for key, default_value in defaults.items():
        if config.get(key) is None:
            config[key] = default_value
            updated = True

    if updated:
        with open(preproc_path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(
            "Postprocessed preprocessor_config.json with pixel limits: "
            f"min_pixels={config['min_pixels']}, max_pixels={config['max_pixels']}"
        )


def _copy_tokenizer_processor_files(source_dir: Path, target_dir: Path):
    """Copy tokenizer/processor sidecars without overwriting model weights/config."""
    if not source_dir.exists() or not source_dir.is_dir():
        return

    copied = []
    for filename in TOKENIZER_PROCESSOR_FILENAMES:
        source_file = source_dir / filename
        if not source_file.exists() or not source_file.is_file():
            continue
        shutil.copy2(source_file, target_dir / filename)
        copied.append(filename)

    if copied:
        logger.info(
            "Copied tokenizer/processor files from %s: %s",
            source_dir,
            sorted(copied),
        )


def run_quantization(args):
    """
    Run the quantization pipeline.

    Args:
        args: Parsed command line arguments
    """

    # Get status logger for Cosmos integration
    s_logger = get_status_logger()

    try:
        s_logger.write(
            status_level=Status.RUNNING, message="Starting quantization pipeline...", verbosity_level=Verbosity.INFO
        )

        # Create save directory
        results_dir = Path(args.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Quantized model will be saved to: {results_dir}")

        # Log quantization start to Cosmos
        log_workflow_status(
            data={
                "quantization_status": "started",
                "model_path": args.model_path,
                "results_dir": str(results_dir),
                "quantization_scheme": args.quantization_scheme,
                "num_calibration_samples": args.num_calibration_samples,
                "lora_enabled": args.enable_lora,
                "base_model_path": args.base_model_path if args.enable_lora else None,
            },
            component_name=COMPONENT_NAME,
        )

        logger.info(SEPARATOR)
        logger.info("STARTING QUANTIZATION PIPELINE")
        logger.info(SEPARATOR)
        logger.info(f"Model: {args.model_path}")
        logger.info(f"Quantization Scheme: {args.quantization_scheme}")

        # Log dataset information
        if args.annotation_path and os.path.exists(args.annotation_path):
            logger.info(f"Dataset: Custom (annotation: {args.annotation_path})")
            if args.media_dir:
                logger.info(f"Media Directory: {args.media_dir}")
        else:
            logger.info(f"Dataset: HuggingFace ({args.dataset_id}, split: {args.dataset_split})")

        logger.info(f"Calibration Samples: {args.num_calibration_samples}")
        logger.info(f"Save Directory: {results_dir}")
        logger.info(SEPARATOR)

        # Check if unmerged LoRA checkpoint without --enable_lora
        if not args.enable_lora:
            adapter_config_path = os.path.join(args.model_path, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                error_msg = (
                    f"The checkpoint at {args.model_path} is in PEFT/LoRA format. "
                    "Please enable LoRA merging by setting --enable_lora=true and providing --base_model_path."
                )
                s_logger.write(status_level=Status.FAILURE, message=error_msg, verbosity_level=Verbosity.ERROR)
                raise ValueError(error_msg)

        # Load model (auto-detects CR1/Qwen2.5-VL and CR2/Qwen3-VL)
        s_logger.write(
            status_level=Status.RUNNING, message=f"Loading model: {args.model_path}", verbosity_level=Verbosity.INFO
        )
        model, processor, model_path = load_model_and_processor(
            model_path=args.model_path,
            enable_lora=args.enable_lora,
            base_model_path=args.base_model_path,
            # Avoid accelerate device-map hooks during quantization.  Those hooks
            # replace module.forward with functools.partial, which compressed-
            # tensors offload cannot patch because it expects bound methods.
            device_map=None,
        )

        model.config.num_attention_heads = model.config.text_config.num_attention_heads
        model.config.num_key_value_heads = model.config.text_config.num_key_value_heads
        model.config.head_dim = model.config.text_config.head_dim

        sequential_targets = get_sequential_targets(model)

        s_logger.write(
            status_level=Status.SUCCESS,
            message=f"Model loaded successfully: {type(model).__name__}",
            verbosity_level=Verbosity.INFO,
        )

        # Load calibration dataset (either HuggingFace or custom)
        use_custom_dataset = args.annotation_path and os.path.exists(args.annotation_path)

        if use_custom_dataset:
            # Load custom dataset from annotation file
            s_logger.write(
                status_level=Status.RUNNING,
                message=f"Loading custom calibration dataset: {args.annotation_path}",
                verbosity_level=Verbosity.INFO,
            )
            logger.info(f"Loading custom calibration dataset: {args.annotation_path}")

            custom_samples = load_custom_dataset(args.annotation_path, args.media_dir)

            # Convert to HuggingFace Dataset format
            ds = HFDataset.from_list(custom_samples)

            # Shuffle and limit samples
            ds = ds.shuffle(seed=42)
            if len(ds) > args.num_calibration_samples:
                ds = ds.select(range(args.num_calibration_samples))

            logger.info(f"Using {len(ds)} samples from custom dataset")
        else:
            # Load HuggingFace dataset
            s_logger.write(
                status_level=Status.RUNNING,
                message=f"Loading HuggingFace calibration dataset: {args.dataset_id}",
                verbosity_level=Verbosity.INFO,
            )
            logger.info(f"Loading HuggingFace calibration dataset: {args.dataset_id}")
            ds = load_dataset(args.dataset_id, split=args.dataset_split)
            ds = ds.shuffle(seed=42)

        s_logger.write(
            status_level=Status.RUNNING, message="Preprocessing calibration dataset...", verbosity_level=Verbosity.INFO
        )
        logger.info("Preprocessing dataset...")
        ds = ds.map(
            lambda x: preprocess_and_tokenize(x, processor, args.max_sequence_length),
            remove_columns=ds.column_names,
        )

        s_logger.write(
            status_level=Status.SUCCESS,
            message=f"Dataset preprocessed: {len(ds)} samples",
            verbosity_level=Verbosity.INFO,
        )

        # Recipe for quantization
        kv_scheme = (
            None
            if args.kv_precision == "bf16"
            else {"num_bits": 8, "type": "float", "strategy": "tensor", "dynamic": False}
        )
        recipe = [
            SmoothQuantModifier(
                smoothing_strength=args.smoothing_strength,
                mappings=[
                    [["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"], "re:.*input_layernorm"],
                    [["re:.*gate_proj", "re:.*up_proj"], "re:.*post_attention_layernorm"],
                ],
            ),
            QuantizationModifier(
                targets="Linear",
                scheme=args.quantization_scheme,
                ignore=[
                    "re:.*lm_head",
                    "re:visual.*",
                    "re:model.visual.*",
                ],
                kv_cache_scheme=kv_scheme,
            ),
        ]

        s_logger.write(
            status_level=Status.RUNNING,
            message=f"Starting {args.quantization_scheme} quantization process...",
            verbosity_level=Verbosity.INFO,
        )
        logger.info(f"Starting {args.quantization_scheme} quantization process...")
        logger.info("This may take a while depending on your GPU...")

        # Perform oneshot quantization
        oneshot(
            model=model,
            recipe=recipe,
            max_seq_length=args.max_sequence_length,
            num_calibration_samples=args.num_calibration_samples,
            dataset=ds,
            data_collator=data_collator,
            sequential_targets=sequential_targets,
        )

        s_logger.write(status_level=Status.SUCCESS, message="Quantization complete!", verbosity_level=Verbosity.INFO)
        logger.info("Quantization complete!")

        # Test the quantized model with a sample generation (unless skipped)
        if not args.skip_test_generation:
            logger.info("========== SAMPLE GENERATION ==============")
            s_logger.write(
                status_level=Status.RUNNING, message="Running test generation...", verbosity_level=Verbosity.INFO
            )

            dispatch_for_generation(model)
            # Test with a sample image
            test_url = "http://images.cocodataset.org/train2017/000000231895.jpg"
            test_image = Image.open(BytesIO(requests.get(test_url).content))
            test_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": test_url},
                        {"type": "text", "text": "Please describe the animal in this image\n"},
                    ],
                }
            ]
            prompt = processor.apply_chat_template(test_messages, add_generation_prompt=True)
            inputs = processor(
                text=[prompt],
                images=[test_image],
                padding=False,
                max_length=args.max_sequence_length,
                truncation=True,
                return_tensors="pt",
            ).to("cuda")

            logger.info("Generating response...")
            output = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
            generated_text = processor.decode(output[0], skip_special_tokens=True)

            logger.info(f"Generated: {generated_text}")
            logger.info("==========================================")

        # Save the quantized model
        s_logger.write(
            status_level=Status.RUNNING,
            message=f"Saving quantized model to: {results_dir}",
            verbosity_level=Verbosity.INFO,
        )
        logger.info(f"\nSaving quantized model to: {results_dir}")

        model.save_pretrained(results_dir, save_compressed=True)

        # Postprocess config: remove unsupported dtype keys that break vLLM loading
        config_path = results_dir / "config.json"
        if config_path.exists():
            _postprocess_config(config_path)

        # Copy processor files
        # use snapshot_download or copy files to make sure correct files are being stored with the checkpoint
        # processor files are incorrect after save_pretrained
        source_model_path = Path(model_path)
        if not source_model_path.exists():
            snapshot_download(
                repo_id=model_path,
                ignore_patterns=["config.json", "*.safetensors*"],
                local_dir=results_dir,
            )
        else:
            _copy_tokenizer_processor_files(source_model_path, results_dir)

        if args.enable_lora:
            # The merged LoRA model may only contain processor_config.json. The adapter
            # export and base model can carry the tokenizer/preprocessor files required
            # by evaluation and inference, so copy those sidecars explicitly.
            _copy_tokenizer_processor_files(Path(args.model_path), results_dir)
            if args.base_model_path:
                _copy_tokenizer_processor_files(Path(args.base_model_path), results_dir)

        _ensure_preprocessor_pixel_config(results_dir)

        s_logger.write(
            status_level=Status.SUCCESS,
            message=f"Quantized model saved successfully to: {results_dir}",
            verbosity_level=Verbosity.INFO,
        )

        # Prepare KPI data
        kpi_data = {
            "quantization_status": "completed",
            "model_path": args.model_path,
            "results_dir": str(results_dir),
            "quantization_scheme": args.quantization_scheme,
            "num_calibration_samples": args.num_calibration_samples,
            "smoothing_strength": args.smoothing_strength,
            "lora_enabled": args.enable_lora,
            "base_model_path": args.base_model_path if args.enable_lora else None,
        }

        # Log final results to Cosmos
        log_workflow_status(data=kpi_data, component_name=COMPONENT_NAME)

        # Print summary
        print("\n" + "=" * 60)
        print("QUANTIZATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Model: {args.model_path}")
        print(f"Quantization Scheme: {args.quantization_scheme}")
        print(f"Saved to: {results_dir}")
        print()
        print(f"Note: {args.quantization_scheme} quantization provides:")
        if args.quantization_scheme == "FP8_DYNAMIC":
            print("- 8-bit weight quantization reduces model size")
            print("- 8-bit activation quantization speeds up inference")
        print()
        print("To use the quantized model with vLLM:")
        print("  from vllm import LLM")
        print(f'  model = LLM("{results_dir}")')
        print("  # vLLM will automatically handle quantized inference")
        print("=" * 60)

        s_logger.write(
            status_level=Status.SUCCESS,
            message=f"Quantization completed successfully. Model saved to: {results_dir}",
            verbosity_level=Verbosity.INFO,
        )

    except KeyboardInterrupt:
        s_logger.write(
            status_level=Status.FAILURE,
            message="Quantization was interrupted by user (Ctrl+C)",
            verbosity_level=Verbosity.WARNING,
        )
        log_workflow_status(
            data={"quantization_status": "interrupted", "error": "User interrupted"}, component_name=COMPONENT_NAME
        )
        raise

    except Exception as e:
        error_msg = f"Quantization failed: {str(e)}"
        s_logger.write(status_level=Status.FAILURE, message=error_msg, verbosity_level=Verbosity.ERROR)
        log_workflow_status(data={"quantization_status": "failed", "error": str(e)}, component_name=COMPONENT_NAME)
        logger.error(error_msg)
        raise


def main():
    """Main entry point for the cosmos-reasoner-quantize command."""
    args = parse_args()
    _load_quantization_dependencies()

    # Parse first so --help and argument errors never create run artifacts.
    monitored = monitor_status(name="Cosmos Framework Quantization", mode="quantize")(run_quantization)
    monitored(args)


if __name__ == "__main__":
    main()
