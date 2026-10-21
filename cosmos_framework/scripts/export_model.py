# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Convert DCP checkpoint to Hugging Face model."""

from cosmos_framework.inference.common.init import init_script

init_script(
    env={"COSMOS_TRAINING": "1"},
    default_env={"COSMOS_DEVICE": "cpu"},
)

import inspect
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated, Any, Callable, Literal

import attrs
import safetensors.torch
import torch.distributed.checkpoint as dcp
import tyro
from torch.distributed.checkpoint.filesystem import FileSystemReader
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

from cosmos_framework.checkpoint.dcp import CustomLoadPlanner
from cosmos_framework.checkpoint.s3_filesystem import S3StorageReader
from cosmos_framework.inference.bimanual import bimanual_action_metadata_for_model_config
from cosmos_framework.inference.common.args import (
    CheckpointOverrides,
    ParallelismOverrides,
    ResolvedPath,
    tyro_cli,
)
from cosmos_framework.inference.common.checkpoints import register_checkpoints
from cosmos_framework.inference.common.config import serialize_config_dict
from cosmos_framework.inference.common.distillation_export import (
    build_student_checkpoint_metadata,
    resolve_student_base_model,
    resolve_vision_checkpoint_path,
    sanitize_student_model_config,
    sanitize_student_public_model_config,
)
from cosmos_framework.inference.common.init import is_rank0
from cosmos_framework.inference.common.public_model_config import build_public_model_config
from cosmos_framework.inference.model import Cosmos3OmniConfig, Cosmos3OmniModel
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from cosmos_framework.scripts._export_model_helpers import (
    EDGE_VIT_BUNDLE_HF_INCLUDE,
    EXPORTED_VISUAL_PREFIXES,
    INTERNAL_VISUAL_PREFIX,
    LIDAR_VAE_BUNDLE_DIR,
    bind_bundled_lidar_vae,
    build_artifact_source,
    build_export_manifest,
    bundle_lidar_vae,
    bundle_processor_files,
    bundle_processor_from_tokenizer_node,
    bundle_vision_encoder,
    clean_stale_export_artifacts,
    constant_image_failure,
    is_edge_model,
    lidar_tokenizer_node,
    prepare_local_bundle_tokenizer,
    read_framework_commit,
    reasoner_vision_capable,
    remove_export_completion_marker,
    reset_export_parallelism,
    resolve_per_view_caption_fields,
    resolve_vision_bundle_dirs,
    rewrite_visual_fqns_for_vfm,
    sanitize_export_args,
    set_include_visual,
)
from cosmos_framework.utils import log
from cosmos_framework.utils.checkpoint_db import CheckpointConfig, CheckpointDirHf, sanitize_uri
from cosmos_framework.utils.lazy_config.registry import convert_target_to_string


def _config_value(config: Any, key: str) -> Any:
    if isinstance(config, dict):
        return config.get(key)
    return getattr(config, key, None)


def _dataset_config_value(dataset_config: Any, key: str) -> Any:
    value = _config_value(dataset_config, key)
    if value is not None:
        return value

    target = _config_value(dataset_config, "_target_") or _config_value(dataset_config, "_target")
    if target is None:
        return None
    try:
        parameter = inspect.signature(target).parameters.get(key)
    except (TypeError, ValueError):
        return None
    if parameter is None or parameter.default is inspect.Parameter.empty:
        return None
    return parameter.default


def _build_edge_policy_metadata(training_config: Any) -> dict[str, Any]:
    """Resolve policy manifest fields from the action experiment config."""
    try:
        dataset_config = training_config.dataloader_train.dataloaders.action_data.dataloader.dataset
    except AttributeError as exc:
        raise ValueError(
            "Cosmos3 Edge export requires an action dataset config at "
            "dataloader_train.dataloaders.action_data.dataloader.dataset."
        ) from exc

    dataset_entries = _config_value(dataset_config, "list_of_datasets")
    if not dataset_entries:
        raise ValueError("Cosmos3 Edge export requires at least one action dataset entry.")

    metadata_by_dataset: list[dict[str, Any]] = []
    for entry in dataset_entries:
        action_dataset_config = _config_value(entry, "dataset")
        if action_dataset_config is None:
            raise ValueError("Cosmos3 Edge action dataset entries must define a dataset config.")

        target = _config_value(action_dataset_config, "_target_") or _config_value(action_dataset_config, "_target")
        action_chunk_size = _dataset_config_value(action_dataset_config, "chunk_length")
        conditioning_fps = _dataset_config_value(action_dataset_config, "fps")
        domain_name = _config_value(action_dataset_config, "embodiment_type")
        if domain_name is None and target is not None:
            domain_name = getattr(target, "EMBODIMENT_TYPE", None)

        if not isinstance(action_chunk_size, int) or isinstance(action_chunk_size, bool) or action_chunk_size <= 0:
            raise ValueError(
                "Cosmos3 Edge action dataset config must define a positive integer `chunk_length`, "
                f"got {action_chunk_size!r}."
            )
        if (
            not isinstance(conditioning_fps, (int, float))
            or isinstance(conditioning_fps, bool)
            or conditioning_fps <= 0
        ):
            raise ValueError(
                f"Cosmos3 Edge action dataset config must define a positive numeric `fps`, got {conditioning_fps!r}."
            )
        if not isinstance(domain_name, str) or not domain_name.strip():
            target_name = getattr(target, "__name__", repr(target))
            raise ValueError(
                "Cosmos3 Edge action dataset config must define `embodiment_type` or expose "
                f"`EMBODIMENT_TYPE`; dataset target is {target_name}."
            )

        metadata_by_dataset.append(
            {
                "action_chunk_size": action_chunk_size,
                "conditioning_fps": float(conditioning_fps),
                "domain_name": domain_name.strip(),
            }
        )

    metadata = metadata_by_dataset[0]
    for field in metadata:
        if any(dataset_metadata[field] != metadata[field] for dataset_metadata in metadata_by_dataset[1:]):
            raise ValueError(
                "Cosmos3 Edge checkpoint.json can represent only one action policy metadata value per field; "
                f"the configured datasets disagree on `{field}`."
            )
    return metadata


def _coerce_to_base_model(model_dict: dict[str, Any]) -> None:
    """For distillation training configs, rewrite the target to the base
    student model so the exported checkpoint contains no teacher/scorer state."""
    target = model_dict.get("_target_", "")
    base_model, base_config = resolve_student_base_model(model_dict, default_base_model=OmniMoTModel)
    base_model_target = convert_target_to_string(base_model)
    if target != base_model_target:
        log.info(f"Overriding model target from {target} to {base_model.__name__} for export")

    sanitize_student_model_config(
        model_dict,
        base_model_target=base_model_target,
        base_config_type=convert_target_to_string(base_config),
        base_config_field_names={field.name for field in attrs.fields(base_config)},
    )


class Args(ParallelismOverrides):
    checkpoint: CheckpointOverrides = CheckpointOverrides.model_construct()
    output_dir: Annotated[ResolvedPath, tyro.conf.arg(aliases=("-o",))]
    """Output model directory."""
    config_only: bool = False
    """If True, only export config."""
    student_only_checkpoint_metadata: bool = False
    """If True, omit source checkpoint and credential paths from checkpoint metadata."""
    vit: bool = True
    """If True, export ViT weights."""
    vit_checkpoint_path: ResolvedPath | None = None
    """Optional local Hugging Face checkpoint directory containing ViT weights."""
    bundle_lidar_vae: bool = False
    """If True and the model has a LiDAR tokenizer, bundle its VAE under 'lidar_vae/' and point the
    exported config at it. Off by default: the export keeps the training-time VAE location."""
    per_view_captions: Literal["auto", "on", "off"] = "auto"
    """Per-view caption layout recorded in config.json: 'auto' derives it from the training dataloaders;
    'on'/'off' records it explicitly (needed when the config carries no dataloaders, e.g. a prepared
    student config)."""
    per_view_caption_task: str | None = None
    """Tokenizer task of the per-view captions (e.g. 'av_joint_camera_lidar_transfer'); derived when omitted."""
    verify: bool = False
    """If True, smoke-test the exported checkpoint with single-GPU inference (tiny reasoner + generation samples)."""


def _load_safetensor_weights(model_dir: Path, predicate: Callable[[str], bool]) -> dict[str, Any]:
    """Load weights from a safetensors file."""
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        shards = {v for k, v in weight_map.items() if predicate(k)}
        vision_weights = {}
        for shard in shards:
            tensors = safetensors.torch.load_file(model_dir / shard)
            vision_weights.update({k: v for k, v in tensors.items() if predicate(k)})
    else:
        tensors = safetensors.torch.load_file(model_dir / "model.safetensors")
        vision_weights = {k: v for k, v in tensors.items() if predicate(k)}
    return vision_weights


# Env vars stripped from the --verify subprocess: init_script defaults to
# COSMOS_DEVICE=cpu and pins COSMOS_TRAINING=1 for the export process itself, and any
# torchrun rendezvous vars would make single-process inference hang.
_VERIFY_ENV_EXCLUDE = (
    "COSMOS_DEVICE",
    "COSMOS_TRAINING",
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
)


def _verify_exported_checkpoint(output_dir: Path, *, run_reasoner_check: bool) -> None:
    """Smoke-test the exported checkpoint via single-GPU 'cosmos_framework.scripts.inference'.

    Runs a minimal generation sample (plus a reasoner-image sample when
    'run_reasoner_check') and asserts each succeeded with non-empty outputs.
    Skips when no CUDA device is available.
    """
    import torch

    if not torch.cuda.is_available():
        log.warning("Skipping --verify: no CUDA device is available on this node.")
        return

    verify_dir = Path(tempfile.mkdtemp(prefix="export_model_verify_"))
    inputs_dir = verify_dir / "inputs"
    outputs_dir = verify_dir / "outputs"
    inputs_dir.mkdir(parents=True)

    # (sample name, output files that must exist and be non-empty)
    checks: list[tuple[str, list[str]]] = []
    if run_reasoner_check:
        from PIL import Image

        # Local synthetic image (mirrors inputs/reasoner/reasoner_image.json
        # without a network fetch).
        image_path = inputs_dir / "verify_image.jpg"
        Image.new("RGB", (256, 256), color=(90, 90, 90)).save(image_path)
        (inputs_dir / "verify_reasoner_image.json").write_text(
            json.dumps(
                {
                    "model_mode": "reasoner",
                    "prompt": "Describe this image in one short sentence.",
                    "vision_path": str(image_path),
                    "seed": 0,
                }
            )
        )
        checks.append(("verify_reasoner_image", ["reasoner_text.txt"]))
    # Smallest generation the sample schema allows: one 256px square frame.
    (inputs_dir / "verify_t2i.json").write_text(
        json.dumps(
            {
                "model_mode": "text2image",
                "prompt": "A gray robotic arm on a white table.",
                "resolution": "256",
                "aspect_ratio": "1,1",
                "num_frames": 1,
                "seed": 0,
            }
        )
    )
    checks.append(("verify_t2i", ["vision.jpg"]))

    cmd = [
        sys.executable,
        "-m",
        "cosmos_framework.scripts.inference",
        "--parallelism-preset=latency",
        # Guardrails pull their own HF models (nvidia/Cosmos-Guardrail1) — an
        # orthogonal hub dependency that would break offline/cache-only verify.
        "--no-guardrails",
        "--checkpoint-path",
        str(output_dir),
        "-o",
        str(outputs_dir),
        "-i",
        *(str(inputs_dir / f"{name}.json") for name, _ in checks),
    ]
    env = {k: v for k, v in os.environ.items() if k not in _VERIFY_ENV_EXCLUDE and not k.startswith("TORCHELASTIC_")}
    log.info(f"Verifying exported checkpoint: {shlex.join(cmd)}")
    result = subprocess.run(cmd, env=env)

    failures: list[str] = []
    if result.returncode != 0:
        failures.append(f"inference subprocess exited with code {result.returncode}")
    for name, output_files in checks:
        sample_dir = outputs_dir / name
        sample_outputs_file = sample_dir / "sample_outputs.json"
        if sample_outputs_file.is_file():
            try:
                sample_outputs = json.loads(sample_outputs_file.read_text())
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                # A truncated/corrupt file is a verify failure, not a crash.
                sample_outputs = None
                failures.append(f"sample '{name}' wrote a corrupt sample_outputs.json ({e})")
            if sample_outputs is not None:
                status = sample_outputs.get("status") if isinstance(sample_outputs, dict) else None
                if status != "success":
                    failures.append(f"sample '{name}' finished with status '{status}'")
        else:
            failures.append(f"sample '{name}' wrote no sample_outputs.json")
        for output_file in output_files:
            output_path = sample_dir / output_file
            if not output_path.is_file() or output_path.stat().st_size == 0:
                failures.append(f"sample '{name}' output '{output_file}' is missing or empty")
            elif output_path.suffix == ".txt" and not output_path.read_text().strip():
                failures.append(f"sample '{name}' output '{output_file}' is blank")
            elif output_path.suffix == ".jpg":
                # Heuristic NaN check: NaN latents decode through clamp() into a
                # valid, non-empty, constant (all-black/uniform) JPEG with status
                # 'success' — catch that class via (near-)zero pixel variance.
                # See 'constant_image_failure'; not a general quality gate.
                import numpy as np
                from PIL import Image

                try:
                    with Image.open(output_path) as image:
                        pixels = np.asarray(image.convert("RGB"))
                except Exception as e:
                    failures.append(f"sample '{name}' output '{output_file}' is not a readable image ({e})")
                else:
                    image_failure = constant_image_failure(pixels)
                    if image_failure is not None:
                        failures.append(f"sample '{name}' output '{output_file}' {image_failure}")
    if failures:
        raise RuntimeError(
            f"--verify failed for '{output_dir}' (the export artifacts were still written): "
            + "; ".join(failures)
            + f". Inspect '{verify_dir}'."
        )
    log.success(f"Verified exported checkpoint '{output_dir}'")
    shutil.rmtree(verify_dir, ignore_errors=True)


def export_model(args: Args) -> None:
    register_checkpoints()
    checkpoint_args = args.checkpoint.build_checkpoint(checkpoints={})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.config_only and is_rank0():
        # 'checkpoint.json' is rewritten only after every artifact succeeds;
        # drop the previous one first so a failed re-export never looks complete.
        if remove_export_completion_marker(args.output_dir):
            log.info(f"Removed the completion marker of a previous export from '{args.output_dir}'")
        # Re-export into the same -o dir: drop artifacts a prior export may have
        # left (stale vision_encoder/ or lidar_vae/ bundle, stale processor files,
        # stale manifest) so the dir always reflects THIS export — inference
        # prefers those files local-first and would silently load stale weights.
        removed = clean_stale_export_artifacts(args.output_dir, include_lidar_vae=args.bundle_lidar_vae)
        if removed:
            log.info(f"Removed stale artifacts of a previous export from '{args.output_dir}': {', '.join(removed)}")

    # Load config
    log.info("Loading config...")
    model_dict = checkpoint_args.load_model_config_dict()
    if not model_dict["config"]["ema"]["enabled"]:
        checkpoint_args.use_ema_weights = False
    model_dict["config"]["ema"]["enabled"] = False
    # The export builds the model in one process; the training topology (FSDP /
    # CP / CFG degrees) is meaningless in the exported config.
    reset_export_parallelism(model_dict)

    # The training config is loaded at most once (Edge policy metadata and the
    # per-view caption layout both read its dataloaders).
    training_config_cache: list[Any] = []

    def training_config() -> Any:
        if not training_config_cache:
            training_config_cache.append(checkpoint_args.load_config())
        return training_config_cache[0]

    is_edge = is_edge_model(model_dict)
    # Cosmos3 Edge action policies need the diffusers converter's `policy` block
    # (action_chunk_size / conditioning_fps / domain_name) in checkpoint.json.
    # Only action Edge models carry an action dataloader; non-action Edge exports
    # (e.g. Edge SFT video recipes) skip this and stay unaffected.
    edge_policy_metadata = (
        _build_edge_policy_metadata(training_config()) if is_edge and model_dict["config"].get("action_gen") else None
    )
    if not args.vit:
        # Text/gen-only export: write include_visual=False into the exported model
        # config so inference skips visual-tower construction instead of dying on
        # missing 'model.net.language_model.visual.*' keys.
        if set_include_visual(model_dict, False):
            log.info("Exporting with include_visual=False (--no-vit)")
        else:
            log.warning("Could not set include_visual=False: model config has no 'create_vlm_config' node")

    # Download VLM checkpoint. Skipped under --config-only (no downloads are
    # needed to export a config).
    vlm_checkpoint_path: str | None = None
    vit_repo: str | None = None
    if args.vit and not args.config_only:
        configured_vlm_checkpoint = model_dict["config"]["vlm_config"]["pretrained_weights"]["backbone_path"]
        resolved_repositories: list[str] = []

        def download_vlm_checkpoint(configured_uri: str) -> str:
            sanitized_uri = sanitize_uri(configured_uri)
            checkpoint: CheckpointConfig | None = CheckpointConfig.maybe_from_uri(sanitized_uri)
            if checkpoint is None:
                raise ValueError(
                    f"VLM backbone checkpoint URI '{configured_uri}' is not in the checkpoint "
                    "registry, so the vision tower cannot be resolved automatically. Either "
                    "export without vision weights (--no-vit), or pass "
                    "--vit-checkpoint-path <local snapshot dir> pointing at a local "
                    "Hugging Face snapshot that contains them."
                )
            resolved_repositories.append(checkpoint.hf.repository)
            if is_edge and isinstance(checkpoint.hf, CheckpointDirHf) and not checkpoint.hf.include:
                # Narrow the Edge download to the ViT bundle (~1 GB) instead of
                # the full repo (tens of GB). Done at this call site, not on the
                # registry entry: that entry has no include filter because
                # training's backbone seeding needs the full repo. Qwen-family
                # exports keep the full snapshot (the ViT merge reads the root
                # safetensors).
                narrowed = checkpoint.hf.model_copy(update={"include": EDGE_VIT_BUNDLE_HF_INCLUDE})
                return narrowed.download()
            return checkpoint.hf.download()

        vlm_checkpoint_path = resolve_vision_checkpoint_path(
            local_path=str(args.vit_checkpoint_path) if args.vit_checkpoint_path is not None else None,
            configured_uri=configured_vlm_checkpoint,
            download_checkpoint=download_vlm_checkpoint,
        )
        vit_repo = resolved_repositories[0] if resolved_repositories else str(args.vit_checkpoint_path)

    # Load model
    log.info("Loading model...")
    _coerce_to_base_model(model_dict)
    # Local training bundles use a data-loader helper with no public alias.
    # Bundle its exact tokenizer and use the existing inference constructor.
    local_bundle_tokenizer = prepare_local_bundle_tokenizer(model_dict, args.output_dir, config_only=args.config_only)
    hf_config = Cosmos3OmniConfig(model=build_public_model_config(model_dict))
    hf_config.save_pretrained(args.output_dir)
    hf_model = Cosmos3OmniModel(hf_config)

    # Save model
    log.info("Saving model...")
    vision_tower_source: dict[str, Any] | None = None
    processor_source: dict[str, Any] | None = (
        {"repo": None, "revision": None, "bundled": True} if local_bundle_tokenizer and not args.config_only else None
    )
    if not args.config_only:
        # Load checkpoint
        if checkpoint_args.checkpoint_path.startswith("s3://"):
            storage_reader = S3StorageReader(
                credential_path=checkpoint_args.credential_path,
                path=checkpoint_args.checkpoint_path,
            )
        else:
            storage_reader = FileSystemReader(checkpoint_args.checkpoint_path)
        state_dict = get_model_state_dict(hf_model.model)
        dcp.load(
            state_dict=state_dict,
            storage_reader=storage_reader,
            planner=CustomLoadPlanner(
                load_ema_to_reg=checkpoint_args.use_ema_weights,
            ),
        )
        state_dict = get_model_state_dict(
            hf_model,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
            ),
        )
        if not is_rank0():
            return

        # Attach the vision tower
        if args.vit:
            assert vlm_checkpoint_path is not None
            assert vit_repo is not None
            if is_edge:
                # Edge bundle mode: the SigLIP2 tower is a self-contained
                # 'vision_encoder/' artifact that inference loads lazily (see
                # Nemotron3DenseVLTextForCausalLM._ensure_vision_tower); copy it
                # verbatim (incl. 'model.projector.*') instead of merging into the
                # root safetensors.
                if any(key.startswith(INTERNAL_VISUAL_PREFIX) for key in state_dict):
                    # DCP-first: never clobber a trained in-DCP tower.
                    log.warning(
                        "DCP checkpoint unexpectedly contains a vision tower "
                        f"('{INTERNAL_VISUAL_PREFIX}*'); keeping the in-DCP weights and "
                        "skipping the vision_encoder/ bundle copy."
                    )
                else:
                    vision_encoder_dir, snapshot_root = resolve_vision_bundle_dirs(Path(vlm_checkpoint_path))
                    bundle_vision_encoder(vision_encoder_dir, snapshot_root, args.output_dir)
                    vision_tower_source = build_artifact_source(
                        repo=vit_repo, resolved_path=vlm_checkpoint_path, bundled=True
                    )
                    log.info(f"Bundled vision_encoder/ from '{vlm_checkpoint_path}'")
                    if snapshot_root is not None:
                        copied = bundle_processor_files(snapshot_root, args.output_dir)
                        if copied:
                            processor_source = build_artifact_source(
                                repo=vit_repo, resolved_path=vlm_checkpoint_path, bundled=True
                            )
                            log.info(f"Bundled processor files: {', '.join(copied)}")
            else:
                # Qwen family: merge the ViT weights into the root safetensors.
                vit_state_dict = _load_safetensor_weights(
                    Path(vlm_checkpoint_path), lambda x: x.startswith(EXPORTED_VISUAL_PREFIXES)
                )
                assert vit_state_dict, "No vision weights found"
                # Accepts 'model.visual.*' or 'visual.*'; raises on a mix of both.
                state_dict.update(rewrite_visual_fqns_for_vfm(vit_state_dict))
                vision_tower_source = build_artifact_source(
                    repo=vit_repo, resolved_path=vlm_checkpoint_path, bundled=False
                )

        # Bundle processor/tokenizer files for every export ('--no-vit' too:
        # generation-only inference still tokenizes prompts through the VLM
        # processor). Skipped when the Edge --vit path above already bundled
        # them, or the local training tokenizer was bundled above. Hub bundling
        # is best-effort: failures log a warning and leave processor_source
        # None. Local tokenizers have no hub fallback and must bundle successfully.
        if processor_source is None:
            processor_source = bundle_processor_from_tokenizer_node(
                (model_dict["config"].get("vlm_config") or {}).get("tokenizer"), args.output_dir
            )

        # Save checkpoint
        hf_model.save_pretrained(
            args.output_dir,
            state_dict=state_dict,
        )

        # With --bundle-lidar-vae, bundle the LiDAR VAE so the export does not depend on the
        # training-time VAE location (an env-var path or object-store URI). The loader binds
        # 'lidar_vae/diffusion_pytorch_model.safetensors' when present.
        lidar_tokenizer = lidar_tokenizer_node(model_dict)
        if lidar_tokenizer is not None:
            if args.bundle_lidar_vae:
                lidar_vae_source = bundle_lidar_vae(lidar_tokenizer, args.output_dir)
                log.info(f"Bundled {LIDAR_VAE_BUNDLE_DIR}/ from '{lidar_vae_source}'")

    # Re-write 'config.json' to apply replacements.
    hf_config_file = args.output_dir / "config.json"
    hf_config_json = json.loads(hf_config_file.read_text())
    if local_bundle_tokenizer:
        # Inference binds this node to the checkpoint-bundled files. Never
        # publish the training-bundle location or the export machine's path.
        hf_config_json["model"]["config"]["vlm_config"]["tokenizer"]["pretrained_model_name"] = "."
    if args.student_only_checkpoint_metadata:
        sanitize_student_public_model_config(hf_config_json["model"])
    bimanual_metadata = bimanual_action_metadata_for_model_config(model_dict["config"])
    if bimanual_metadata is not None:
        hf_config_json.update(bimanual_metadata)
    # The training dataloaders do not travel with an exported checkpoint, and the
    # caption layout is only discoverable from them. Record it so inference does
    # not have to guess: without it a per-view checkpoint silently falls back to
    # one merged prompt (and the wrong system prompt).
    # A config source that cannot be loaded as a training config, or carries no
    # dataloaders (e.g. a prepared student config), records nothing in 'auto'
    # mode; '--per-view-captions on|off' records the layout explicitly.
    hf_config_json.update(
        resolve_per_view_caption_fields(args.per_view_captions, args.per_view_caption_task, training_config)
    )
    # Point the LiDAR tokenizer at the bundled VAE. After the student-only
    # sanitization above, which would otherwise rewrite it to the published
    # registry location.
    if args.bundle_lidar_vae and bind_bundled_lidar_vae(hf_config_json, args.output_dir):
        log.info(f"Exported config references the bundled {LIDAR_VAE_BUNDLE_DIR}/")
    hf_config_json["model_type"] = "cosmos3_omni"
    serialize_config_dict(hf_config_json, hf_config_file)

    # Write the provenance sidecar (kept separate from 'checkpoint.json', whose
    # schema is checkpoint_args.model_dump).
    export_args: dict[str, Any] = {
        "config_only": args.config_only,
        "student_only_checkpoint_metadata": args.student_only_checkpoint_metadata,
        "verify": args.verify,
        "vit": args.vit,
        "vit_checkpoint_path": str(args.vit_checkpoint_path) if args.vit_checkpoint_path is not None else None,
    }
    # Only non-default values of the newer options, so default manifests keep their previous shape.
    for key, default in (("bundle_lidar_vae", False), ("per_view_captions", "auto"), ("per_view_caption_task", None)):
        if getattr(args, key) != default:
            export_args[key] = getattr(args, key)
    if args.student_only_checkpoint_metadata:
        # 'checkpoint.json' is scrubbed of local paths in this mode; the
        # manifest must not leak them through 'export_args' either. The
        # repo@revision provenance in '*_source' stays untouched.
        export_args = sanitize_export_args(export_args)
    manifest = build_export_manifest(
        vision_tower_source=vision_tower_source,
        processor_source=processor_source,
        export_args=export_args,
        framework_commit=read_framework_commit(),
    )
    serialize_config_dict(manifest, args.output_dir / "export_manifest.json")

    # Write 'checkpoint.json' last to indicate that the model is complete.
    checkpoint_metadata = (
        build_student_checkpoint_metadata(use_ema_weights=checkpoint_args.use_ema_weights)
        if args.student_only_checkpoint_metadata
        else checkpoint_args.model_dump(mode="json")
    )
    if edge_policy_metadata is not None:
        checkpoint_metadata["policy"] = edge_policy_metadata
    serialize_config_dict(checkpoint_metadata, args.output_dir / "checkpoint.json")

    print(f"Saved model to {args.output_dir}")

    if args.verify:
        if args.config_only:
            log.warning("--verify skipped: --config-only exports nothing to verify")
            return
        # The reasoner-image check needs a checkpoint whose reasoner can encode
        # vision prompts: Edge always can (lazy tower); Qwen-family (Nano/Super)
        # only with include_visual truthy — the shipped SFT configs leave it
        # unset, and the sample-build fail-fast correctly rejects the vision
        # sample there, so gating on args.vit alone would fail healthy exports.
        run_reasoner_check = args.vit and reasoner_vision_capable(model_dict)
        if args.vit and not run_reasoner_check:
            log.info(
                "--verify: skipping the reasoner-image check — the exported reasoner LM has no visual "
                "tower (model config 'include_visual' is falsy); the generation check still runs."
            )
        # Runs only after every export artifact is written: verify is a check,
        # not a gate, and a failure exits non-zero without touching the export.
        _verify_exported_checkpoint(args.output_dir, run_reasoner_check=run_reasoner_check)


def main() -> None:
    args = tyro_cli(Args, description=__doc__, config=(tyro.conf.OmitArgPrefixes,))
    export_model(args)


if __name__ == "__main__":
    main()
