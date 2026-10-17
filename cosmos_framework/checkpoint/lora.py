# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
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

"""
Shared LoRA utilities for Cosmos Framework.

This module provides common functionality for merging LoRA adapters with base models,
used across evaluation, inference, and other components.
"""

import json
import logging
import os
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _candidate_module_names(adapter_module_name: str) -> list[str]:
    candidates = [adapter_module_name]
    for prefix in ("base_model.model.", "base_model."):
        if adapter_module_name.startswith(prefix):
            candidates.append(adapter_module_name[len(prefix) :])
    for candidate in list(candidates):
        if candidate.startswith("model."):
            candidates.append(candidate[len("model.") :])
    return list(dict.fromkeys(candidates))


def _merge_lora_weights_directly(model, adapter_path: str, adapter_cfg: dict) -> int:
    """Merge standard LoRA A/B adapter tensors without importing PEFT."""
    import torch
    from safetensors.torch import safe_open

    adapter_file = Path(adapter_path) / "adapter_model.safetensors"
    if not adapter_file.exists():
        raise FileNotFoundError(f"LoRA adapter weights not found at {adapter_file}")

    grouped: dict[str, dict[str, torch.Tensor]] = {}
    with safe_open(str(adapter_file), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.endswith(".lora_A.weight"):
                grouped.setdefault(key[: -len(".lora_A.weight")], {})["A"] = handle.get_tensor(key)
            elif key.endswith(".lora_B.weight"):
                grouped.setdefault(key[: -len(".lora_B.weight")], {})["B"] = handle.get_tensor(key)

    if not grouped:
        raise ValueError(f"No LoRA A/B tensors found in {adapter_file}")

    r = float(adapter_cfg.get("r") or next(iter(grouped.values()))["A"].shape[0])
    alpha = float(adapter_cfg.get("lora_alpha") or r)
    scaling = alpha / (r**0.5 if adapter_cfg.get("use_rslora") else r)

    modules = dict(model.named_modules())
    merged_count = 0
    missing_modules = []

    for adapter_module_name, tensors in grouped.items():
        if "A" not in tensors or "B" not in tensors:
            raise ValueError(f"Incomplete LoRA tensors for {adapter_module_name}")

        target_module = None
        target_name = None
        for candidate in _candidate_module_names(adapter_module_name):
            maybe_module = modules.get(candidate)
            if maybe_module is not None and hasattr(maybe_module, "weight"):
                target_module = maybe_module
                target_name = candidate
                break

        if target_module is None:
            missing_modules.append(adapter_module_name)
            continue

        lora_a = tensors["A"].float()
        lora_b = tensors["B"].float()
        delta = (lora_b @ lora_a) * scaling

        if tuple(delta.shape) != tuple(target_module.weight.shape):
            raise ValueError(
                f"LoRA delta shape {tuple(delta.shape)} does not match "
                f"{target_name}.weight shape {tuple(target_module.weight.shape)}"
            )

        with torch.no_grad():
            target_module.weight.add_(delta.to(device=target_module.weight.device, dtype=target_module.weight.dtype))
        merged_count += 1

    if missing_modules:
        raise ValueError(
            "Could not find target modules for LoRA tensors: "
            + ", ".join(missing_modules[:5])
            + ("..." if len(missing_modules) > 5 else "")
        )

    return merged_count


def merge_lora_model(
    lora_path: str,
    base_model_path: Optional[str] = None,
    progress_callback=None,
    merged_model_path: Optional[str] = None,
) -> str:
    """
    Merge LoRA weights with base model.

    This function handles the complete LoRA merging process:
    1. Checks if a merged model already exists (caching)
    2. Loads the base model and LoRA adapter
    3. Merges the weights and saves the combined model
    4. Handles processor/tokenizer saving
    5. Provides proper error handling and memory cleanup

    Args:
        lora_path: Path to the LoRA model directory
        base_model_path: Path to the base model (optional, can be inferred from adapter config)
        progress_callback: Optional callable(message: str) invoked at each step to report progress

    Returns:
        Path to the merged model directory

    Raises:
        ValueError: If base_model_path cannot be determined
        ImportError: If required libraries (transformers, peft) are not available
    """
    # Strip trailing slashes from input path
    lora_path = lora_path.rstrip("/")
    logger.info(f"Merging LoRA model: {lora_path}")

    # Public callers use fingerprinted, atomic checkpoint preparation. The
    # low-level merge only writes a new temporary destination owned by it.
    from cosmos_framework.checkpoint.reasoner import _check_output_path, ensure_evaluation_checkpoint

    if not base_model_path:
        with (Path(lora_path) / "adapter_config.json").open() as handle:
            base_model_path = json.load(handle).get("base_model_name_or_path")
    if not base_model_path:
        raise ValueError("A local base_model_path is required to merge a LoRA adapter")
    if merged_model_path is None:
        return ensure_evaluation_checkpoint(lora_path, base_model_path=base_model_path, enable_lora=True)
    merged_path = str(Path(merged_model_path).resolve())
    _check_output_path(Path(merged_path), Path(lora_path).resolve(), Path(base_model_path).resolve())
    if Path(merged_path).exists():
        raise ValueError(f"Merge destination already exists: {merged_path}")

    try:
        # Import required libraries for LoRA merging
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor

        # Strip trailing slash from base_model_path if provided
        if base_model_path:
            base_model_path = base_model_path.rstrip("/")

        # Use provided base model path or infer from LoRA config
        if not base_model_path:
            # Try to read base model path from adapter config
            adapter_config_path = os.path.join(lora_path, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                with open(adapter_config_path, "r") as f:
                    adapter_config = json.load(f)
                    base_model_path = adapter_config.get("base_model_name_or_path")
                    if base_model_path:
                        base_model_path = base_model_path.rstrip("/")

            if not base_model_path:
                raise ValueError(
                    "Base model path not provided and could not be inferred from adapter config. "
                    "Please provide base_model_path parameter or ensure adapter_config.json exists "
                    f"in {lora_path} with 'base_model_name_or_path' field."
                )

        def _progress(msg):
            logger.info(msg)
            if progress_callback:
                try:
                    progress_callback(msg)
                except Exception:
                    pass

        _progress(f"LoRA merge: loading base model from {base_model_path}")

        # Load base model (use AutoModelForImageTextToText to auto-detect architecture)
        logger.info("Step 1/4: Loading base model...")
        try:
            from transformers import AutoModelForImageTextToText

            model = AutoModelForImageTextToText.from_pretrained(
                base_model_path,
                torch_dtype="auto",
                trust_remote_code=True,  # Required for Qwen models
            )
            logger.info(f"Base model loaded successfully. Type: {type(model)}")
        except Exception as e:
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            logger.error(f"Failed to load base model from {base_model_path}: {e}")
            raise RuntimeError(f"Base model loading failed: {e}") from e

        _progress("LoRA merge: loading adapter weights")
        logger.info("Step 2/4: Loading LoRA adapter...")
        logger.info(f"  Base model device: {next(model.parameters()).device if model.parameters() else 'N/A'}")
        peft_model = None
        adapter_cfg = {}
        try:
            # Fix PEFT config compatibility issue: null alpha_pattern/r_pattern causes errors
            adapter_load_path = lora_path
            adapter_config_path = os.path.join(adapter_load_path, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                with open(adapter_config_path, "r") as f:
                    adapter_cfg = json.load(f)
                needs_fix = False
                if adapter_cfg.get("alpha_pattern") is None:
                    adapter_cfg["alpha_pattern"] = {}
                    needs_fix = True
                if adapter_cfg.get("r_pattern") is None:
                    adapter_cfg["r_pattern"] = {}
                    needs_fix = True
                if needs_fix:
                    adapter_load_path = tempfile.mkdtemp(prefix="cosmos_lora_adapter_")
                    shutil.copytree(lora_path, adapter_load_path, dirs_exist_ok=True)
                    adapter_config_path = os.path.join(adapter_load_path, "adapter_config.json")
                    with open(adapter_config_path, "w") as f:
                        json.dump(adapter_cfg, f, indent=4)
                    logger.info(
                        "Fixed adapter_config.json in a writable temp copy: set null alpha_pattern/r_pattern to {}"
                    )

            peft_model = PeftModel.from_pretrained(model, adapter_load_path)
            logger.info(f"LoRA adapter loaded successfully. Type: {type(peft_model)}")
        except Exception as e:
            logger.warning(f"PEFT LoRA loading failed, trying direct weight merge: {e}")
            logger.warning(f"Traceback:\n{traceback.format_exc()}")
            try:
                merged_count = _merge_lora_weights_directly(model, adapter_load_path, adapter_cfg)
                logger.info(f"Direct LoRA merge applied to {merged_count} modules")
            except Exception as direct_error:
                logger.error(f"Failed to load LoRA adapter from {lora_path}: {direct_error}")
                logger.error(f"Traceback:\n{traceback.format_exc()}")
                raise RuntimeError(f"LoRA adapter loading failed: {direct_error}") from direct_error

        _progress("LoRA merge: merging weights")
        logger.info("Step 3/4: Merging LoRA weights with base model...")
        try:
            if peft_model is not None:
                merged_model = peft_model.merge_and_unload()
                logger.info(f"LoRA weights merged successfully. Type: {type(merged_model)}")
            else:
                merged_model = model
                logger.info("LoRA weights merged successfully with direct weight merge")
        except Exception as e:
            logger.error(f"Failed during merge_and_unload: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            raise RuntimeError(f"LoRA merge failed: {e}") from e

        _progress(f"LoRA merge: saving merged model to {merged_path}")
        logger.info(f"Step 4/4: Saving merged model to: {merged_path}")
        os.makedirs(merged_path, exist_ok=True)
        merged_model.save_pretrained(merged_path)

        # Also save the processor/tokenizer
        try:
            processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
            processor.save_pretrained(merged_path)
            logger.info("Saved processor to merged model directory")
        except Exception as e:
            logger.warning(f"Failed to save processor: {e}")

        # Clean up GPU memory
        del model, peft_model, merged_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"LoRA merging completed successfully: {merged_path}")
        return merged_path

    except ImportError as e:
        logger.error(f"Required libraries not available for LoRA merging: {e}")
        logger.error("Please ensure 'transformers' and 'peft' are installed")
        raise

    except Exception as e:
        logger.error(f"LoRA merging failed: {e}")
        raise RuntimeError(f"LoRA merging failed: {e}") from e


def should_enable_lora(config: dict, enable_lora_flag: Optional[bool] = None) -> bool:
    """
    Determine if LoRA should be enabled based on configuration and flags.

    Args:
        config: Configuration dictionary that may contain LoRA settings
        enable_lora_flag: Explicit LoRA enable flag (overrides config)

    Returns:
        True if LoRA should be enabled, False otherwise
    """
    # Explicit flag takes precedence
    if enable_lora_flag is not None:
        return enable_lora_flag

    # Check various config paths for LoRA settings
    model_config = config.get("model", {})
    return model_config.get("enable_lora", False)


def get_base_model_path(config: dict, explicit_path: Optional[str] = None) -> Optional[str]:
    """
    Get base model path from configuration or explicit parameter.

    Args:
        config: Configuration dictionary that may contain base model path
        explicit_path: Explicitly provided base model path

    Returns:
        Base model path if found, None otherwise
    """
    # Explicit path takes precedence
    if explicit_path:
        return explicit_path

    # Check config for base model path
    model_config = config.get("model", {})
    return model_config.get("base_model_path")


def load_model_and_processor(
    model_path: str,
    enable_lora: bool = False,
    base_model_path: Optional[str] = None,
    torch_dtype: str = "auto",
    device_map: str = "auto",
    trust_remote_code: bool = True,
    progress_callback=None,
    **extra_model_kwargs,
):
    """
    Load a model and processor with automatic architecture detection.

    Handles provenance-checked LoRA merging and works with both
    Qwen2.5-VL (CR1) and Qwen3-VL (CR2) models.

    Args:
        model_path: Path to model or LoRA checkpoint
        enable_lora: Whether to merge LoRA adapter with base model
        base_model_path: Base model path for LoRA merging
        torch_dtype: PyTorch dtype string (auto, float16, bfloat16, etc.)
        device_map: Device mapping strategy
        trust_remote_code: Whether to trust remote code
        progress_callback: Optional callable(message: str) invoked to report progress
        **extra_model_kwargs: Additional kwargs passed to from_pretrained

    Returns:
        Tuple of (model, processor, resolved_model_path)
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor

    resolved_path = model_path

    if enable_lora:
        if not base_model_path:
            raise ValueError("base_model_path is required when enable_lora is True")
        resolved_path = merge_lora_model(model_path, base_model_path, progress_callback=progress_callback)
        logger.info(f"LoRA merging enabled. Using merged model: {resolved_path}")
    else:
        adapter_config = Path(model_path) / "adapter_config.json"
        if adapter_config.exists():
            raise ValueError("Adapter checkpoints require enable_lora=True and a local base_model_path")

    model = AutoModelForImageTextToText.from_pretrained(
        resolved_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        **extra_model_kwargs,
    )
    processor = AutoProcessor.from_pretrained(resolved_path, trust_remote_code=trust_remote_code)
    logger.info(f"Loaded model: {type(model).__name__} from {resolved_path}")

    return model, processor, resolved_path


def get_sequential_targets(model) -> list:
    """
    Get the correct sequential_targets for llmcompressor based on model architecture.

    Returns:
        List of decoder layer class names for sequential quantization
    """
    model_type = getattr(model, "config", None)
    model_type = getattr(model_type, "model_type", "") if model_type else ""

    targets_map = {
        "qwen2_5_vl": ["Qwen2_5_VLDecoderLayer"],
        "qwen2_vl": ["Qwen2VLDecoderLayer"],
        "qwen3_vl": ["Qwen3VLTextDecoderLayer"],
    }

    targets = targets_map.get(model_type)
    if targets:
        logger.info(f"Detected model_type={model_type}, using sequential_targets={targets}")
        return targets

    logger.warning(
        f"Unknown model_type={model_type!r}, falling back to Qwen2_5_VLDecoderLayer. "
        "Override sequential_targets if quantization fails."
    )
    return ["Qwen2_5_VLDecoderLayer"]
