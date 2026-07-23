# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Convert transformers checkpoint to diffusers checkpoint."""

from cosmos_framework.inference.common.init import init_script

init_script(
    env={
        "COSMOS_DEVICE": "cpu",
    }
)

import copy
import json
import shutil
import struct
from pathlib import Path
from typing import Annotated, Any, Literal

import pydantic
import tyro

from cosmos_framework.inference.args import OmniSetupOverrides
from cosmos_framework.inference.common.args import CheckpointOverrides, ResolvedPath
from cosmos_framework.inference.common.checkpoints import register_checkpoints
from cosmos_framework.inference.common.config import deserialize_config_dict, serialize_config_dict
from cosmos_framework.inference.common.public_model_config import (
    build_public_model_config,
    load_model_config_from_hf_config,
)
from cosmos_framework.utils import log
from cosmos_framework.utils.checkpoint_db import CheckpointConfig, CheckpointDirHf


class Args(pydantic.BaseModel):
    checkpoint: CheckpointOverrides
    """Transformers checkpoint."""
    output_path: Annotated[ResolvedPath, tyro.conf.arg(aliases=("-o",))]
    """Output diffusers checkpoint directory."""

    config_only: bool = False
    """If True, only save config."""

    skip_vision_encoder: bool = False
    """Do not save the vision encoder sidecar in the Diffusers checkpoint."""

    distilled_scheduler: Literal["auto", "on", "off"] = "auto"
    """Scheduler export mode for distilled (few-step) checkpoints.

    * auto: export the distilled FlowMatchEuler scheduler when the checkpoint
      defines `fixed_step_sampler_config`, else UniPC.
    * on: require and always export the distilled scheduler.
    * off: always export the UniPC scheduler.
    """

    repo_id: str | None = None
    """HF repo id embedded in modular_model_index.json component specs.

    Defaults to the output directory name (matches diffusers' own default). Set
    this to the target Hub id (e.g. 'nvidia/Cosmos3-Super-Image2Video-4Step')
    when preparing a release so the modular pipeline resolves its components.
    """

    edge_include_reasoner: bool = True
    """Keep the pinned Edge metadata and vision sidecar in the final repository."""

    edge_reasoner_repo_id: str = "nvidia/Cosmos3-Edge"
    """Hugging Face repository containing the Cosmos3 Edge reasoner checkpoint."""

    edge_reasoner_revision: str = "be935d6931e4e176d7353abad41ca529d7b33b12"
    """Pinned revision of the Cosmos3 Edge reasoner checkpoint."""

    edge_reasoner_path: Path | None = None
    """Optional local Cosmos3 Edge reasoner snapshot, used instead of downloading it."""

    edge_action_chunk_size: pydantic.PositiveInt | None = None
    """Optional action chunk size override for a Cosmos3 Edge policy manifest."""


class EdgePolicyMetadata(pydantic.BaseModel):
    """Policy metadata resolved from the Stage 1 Edge export."""

    action_chunk_size: pydantic.PositiveInt
    conditioning_fps: pydantic.PositiveFloat
    domain_name: str = pydantic.Field(min_length=1)


class SafetensorsIndexMetadata(pydantic.BaseModel):
    total_size: int = 0


class SafetensorsIndex(pydantic.BaseModel):
    metadata: SafetensorsIndexMetadata = pydantic.Field(default_factory=SafetensorsIndexMetadata)
    weight_map: dict[str, str] = pydantic.Field(default_factory=dict)

    def update(self, safetensors_path: Path, rel_path: str, exclude_key_prefixes: tuple[str, ...] = ()) -> None:
        with safetensors_path.open("rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_size).decode("utf-8"))

        for name, info in header.items():
            if name == "__metadata__":
                continue
            if name.startswith(exclude_key_prefixes):
                continue
            self.metadata.total_size += info["data_offsets"][1] - info["data_offsets"][0]
            if name in self.weight_map:
                raise ValueError(f"Key {name} already in weight map")
            self.weight_map[name] = rel_path

    def update_dir(self, safetensors_dir: Path, rel_path: str):
        for safetensors_path in safetensors_dir.glob("*.safetensors"):
            self.update(safetensors_path, f"{rel_path}/{safetensors_path.name}")


def _write_diffusers_weight_index(output_path: Path) -> None:
    """Write the root index over Diffusers component shards."""
    index = SafetensorsIndex()
    index.update_dir(output_path / "transformer", "transformer")
    vision_encoder_path = output_path / "vision_encoder/model.safetensors"
    if vision_encoder_path.is_file():
        index.update(vision_encoder_path, "vision_encoder/model.safetensors")
    (output_path / "model.safetensors.index.json").write_text(index.model_dump_json(indent=2))


def _build_public_export_model_config(model_dict: dict[str, Any]) -> dict[str, Any]:
    """Remove unsupported internal-only settings before building the public config."""
    public_model_dict = copy.deepcopy(model_dict)
    quantization = public_model_dict.get("config", {}).pop("quantization", None)
    if quantization is not None:
        quantization_values = {key: value for key, value in quantization.items() if key not in {"_type", "_target_"}}
        disabled_quantization = {
            "exclude_regex": [],
            "include_regex": [],
            "method": None,
        }
        if quantization_values != disabled_quantization:
            raise ValueError(
                "Cannot export an enabled or non-default internal quantization config to the public Cosmos3 schema: "
                f"{quantization_values}"
            )
    return build_public_model_config(public_model_dict)


def _is_edge_model_config(model_dict: dict[str, Any]) -> bool:
    """Return whether a model config uses the Nemotron Cosmos3 Edge backbone."""
    config = model_dict.get("config", {})
    vlm_config = config.get("vlm_config", {})
    pretrained_weights = vlm_config.get("pretrained_weights", {})
    model_instance = vlm_config.get("model_instance", {})
    return bool(
        pretrained_weights.get("checkpoint_format") == "nemotron_3_dense_vl"
        or "Nemotron3" in str(model_instance.get("_target_", ""))
    )


def _load_edge_policy_metadata(checkpoint_path: Path) -> EdgePolicyMetadata:
    """Load policy metadata emitted by the config-driven Stage 1 export."""
    metadata_path = checkpoint_path / "checkpoint.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            "Cosmos3 Edge conversion requires the Stage 1 export checkpoint.json; "
            f"no metadata file was found at {metadata_path}."
        )

    checkpoint_metadata = deserialize_config_dict(metadata_path)
    policy_metadata = checkpoint_metadata.get("policy")
    if not isinstance(policy_metadata, dict):
        raise ValueError(
            "Cosmos3 Edge checkpoint.json is missing `policy` metadata. Re-export the DCP checkpoint with "
            "export_model.py so action_chunk_size, conditioning_fps, and domain_name come from the experiment config."
        )
    try:
        return EdgePolicyMetadata.model_validate(policy_metadata)
    except pydantic.ValidationError as exc:
        raise ValueError("Invalid Cosmos3 Edge policy metadata in checkpoint.json.") from exc


# Modular pipeline / blocks classes keyed by whether the export is distilled.
# The distilled variant samples on a fixed schedule with guidance baked into the
# weights; the base variant uses the standard Cosmos3 omni blocks.
MODULAR_PIPELINE_CLASSES = {
    False: ("Cosmos3OmniModularPipeline", "Cosmos3OmniBlocks"),
    True: ("Cosmos3DistilledModularPipeline", "Cosmos3DistilledBlocks"),
}

PIPELINE_BEHAVIOR_FIELDS = (
    "default_use_system_prompt",
    "enable_safety_checker",
    "use_native_flow_schedule",
    "native_flow_shift",
)


def _write_modular_model_index(output_path: Path, repo_id: str) -> None:
    """Write modular_model_index.json next to the task-based model_index.json.

    Component classes (tokenizer, scheduler, vae, transformer) are derived from
    the already-written model_index.json rather than hardcoded, so the modular
    index always matches whatever the pipeline save produced (e.g. UniPC vs.
    FlowMatchEuler scheduler, slow vs. fast tokenizer). The distilled vs. base
    pipeline/blocks classes are selected from the exported scheduler config.

    Note: the distilled modular pipeline classes are not yet in a released
    diffusers version (they ship via the modular Cosmos3 PR), so `_diffusers_version`
    is copied from the pipeline save rather than asserted, and the classes are only
    referenced by name here — loading the result requires a diffusers build that
    provides them.
    """
    model_index = json.loads((output_path / "model_index.json").read_text())

    # Distilled checkpoints register `fixed_step_sampler_config` on the scheduler.
    scheduler_config_path = output_path / "scheduler" / "scheduler_config.json"
    is_distilled = False
    distilled_sigmas = None
    if scheduler_config_path.is_file():
        scheduler_config = json.loads(scheduler_config_path.read_text())
        fixed_step_cfg = scheduler_config.get("fixed_step_sampler_config")
        is_distilled = fixed_step_cfg is not None
        if fixed_step_cfg and fixed_step_cfg.get("t_list"):
            distilled_sigmas = [float(s) for s in fixed_step_cfg["t_list"]]

    pipeline_class, blocks_class = MODULAR_PIPELINE_CLASSES[is_distilled]

    modular_index: dict[str, Any] = {
        "_blocks_class_name": blocks_class,
        "_class_name": pipeline_class,
        "_diffusers_version": model_index.get("_diffusers_version"),
        "is_distilled": is_distilled,
    }
    if is_distilled:
        # diffusers distilled modular pipeline reads its fixed sampling schedule from this pipeline config rather
        # than off the scheduler; `scheduler_config.json` keeps `fixed_step_sampler_config` for non-modular
        # consumers (e.g. vllm-omni).
        modular_index["distilled_sigmas"] = distilled_sigmas

    for name in PIPELINE_BEHAVIOR_FIELDS:
        if name in model_index:
            modular_index[name] = model_index[name]

    for name, value in model_index.items():
        # Keep only saved components: [library, class] pairs with non-null entries.
        # Skips meta keys (_class_name, ...) and unset components ([null, null]).
        if name.startswith("_"):
            continue
        if not (isinstance(value, list) and len(value) == 2):
            continue
        library, class_name = value
        if library is None or class_name is None:
            continue
        modular_index[name] = [
            library,
            class_name,
            {
                "pretrained_model_name_or_path": repo_id,
                "subfolder": name,
                "type_hint": [library, class_name],
                "variant": None,
            },
        ]

    (output_path / "modular_model_index.json").write_text(json.dumps(modular_index, indent=2) + "\n")
    log.info(f"Wrote modular_model_index.json ({pipeline_class}, is_distilled={is_distilled}).")


def convert_model_to_diffusers(args: Args) -> None:
    register_checkpoints()
    checkpoint_config = args.checkpoint.build_checkpoint(checkpoints=OmniSetupOverrides.CHECKPOINTS)
    checkpoint_path = checkpoint_config.download_checkpoint()
    model_dict = checkpoint_config.load_model_config_dict()
    is_edge_model = _is_edge_model_config(model_dict)

    from cosmos_framework.scripts import _convert_model_to_diffusers

    if is_edge_model and (Path(checkpoint_path) / ".metadata").is_file():
        raise ValueError(
            "Raw Cosmos3 Edge DCP cannot be converted directly. Run export_model.py first with the experiment "
            "config, then pass the resulting HF directory to convert_model_to_diffusers.py."
        )

    if is_edge_model:
        if args.config_only:
            raise ValueError("Cosmos3 Edge conversion does not support --config-only.")
        if args.skip_vision_encoder:
            raise ValueError("Cosmos3 Edge conversion cannot skip the reasoner vision assets.")
        if not args.edge_include_reasoner:
            raise ValueError("Cosmos3 Edge conversion requires the pinned reasoner metadata and vision assets.")
        # Fail fast with an actionable message if the installed Diffusers build lacks the
        # Edge transformer API (the exported public config isn't detected as Edge by the
        # low-level checkpoint sniff, so gate here where is_edge_model is already known).
        _convert_model_to_diffusers._validate_edge_transformer_support()
        # Action-policy Edge checkpoints carry a `policy` block (action_chunk_size /
        # conditioning_fps / domain_name) that the diffusers pipeline needs. Non-action
        # Edge (e.g. a video SFT) has no such metadata — convert it too, just without the
        # policy block. The core reasoner+vision conversion below is not action-specific.
        is_action_policy = bool(model_dict["config"]["action_gen"])
        edge_policy_metadata = _load_edge_policy_metadata(Path(checkpoint_path)) if is_action_policy else None

        args.output_path.mkdir(parents=True, exist_ok=True)
        _convert_model_to_diffusers.convert_model_to_diffusers(
            _convert_model_to_diffusers.Args(
                checkpoint_path=Path(checkpoint_path),
                output=str(args.output_path),
                save_pipeline=True,
                dtype="bf16",
                include_reasoner=True,
                reasoner_repo_id=args.edge_reasoner_repo_id,
                reasoner_revision=args.edge_reasoner_revision,
                reasoner_path=args.edge_reasoner_path,
            )
        )
        # The published Edge repository follows the existing Diffusers layout:
        # its root index points at component shards, rather than at an
        # intermediate native Transformers `model.safetensors` file.
        _convert_model_to_diffusers._write_diffusers_safetensors_index(args.output_path)
        (args.output_path / "model.safetensors").unlink(missing_ok=True)
        _write_modular_model_index(args.output_path, repo_id=args.repo_id or args.output_path.name)

        checkpoint_payload: dict[str, Any] = {
            "config_file": None,
            "experiment": None,
            "experiment_overrides": None,
        }
        if edge_policy_metadata is not None:
            checkpoint_payload["policy"] = {
                "action_chunk_size": (
                    args.edge_action_chunk_size
                    if args.edge_action_chunk_size is not None
                    else edge_policy_metadata.action_chunk_size
                ),
                "conditioning_fps": edge_policy_metadata.conditioning_fps,
                "domain_name": edge_policy_metadata.domain_name,
            }
        elif args.edge_action_chunk_size is not None:
            raise ValueError(
                "--edge-action-chunk-size was set, but this Edge checkpoint is not an action policy "
                "(config action_gen is false), so there is no policy manifest to override."
            )
        serialize_config_dict(checkpoint_payload, args.output_path / "checkpoint.json")
        print(f"Saved diffusers checkpoint to {args.output_path}")
        return

    model_dict = load_model_config_from_hf_config(deserialize_config_dict(checkpoint_path / "config.json"))
    args.output_path.mkdir(parents=True, exist_ok=True)

    supports_action = model_dict["config"]["action_gen"]
    supports_sound = model_dict["config"]["sound_gen"]
    if not supports_action:
        log.warning(
            "The checkpoint does not support action generation. For some checkpoints, like "
            "'Cosmos3-Super-Image2Video' it's fine, but make sure it's expected."
        )
    if not supports_sound:
        log.warning(
            "The checkpoint does not support sound generation. For some checkpoints, like "
            "'Cosmos3-Nano-Policy-DROID' it's fine, but make sure it's expected."
        )

    sound_tokenizer_path: Path | None = None
    sound_tokenizer_config_path: Path | None = None
    if supports_sound and not args.config_only:
        sound_tokenizer_dir, sound_tokenizer_name = model_dict["config"]["sound_tokenizer"]["avae_path"].rsplit("/", 1)
        sound_tokenizer_checkpoint = CheckpointConfig.maybe_from_uri(f"s3://bucket/{sound_tokenizer_dir}")
        assert sound_tokenizer_checkpoint is not None
        sound_tokenizer_local = Path(sound_tokenizer_checkpoint.hf.download())
        # HF-published checkpoints ship sound_tokenizer/ in the diffusers layout
        # (config.json + diffusion_pytorch_model.safetensors), which the converter
        # consumes directly; fall back to the legacy AVAE file pair named by the
        # model config's avae_path.
        sound_tokenizer_path = sound_tokenizer_local / "diffusion_pytorch_model.safetensors"
        sound_tokenizer_config_path = sound_tokenizer_local / "config.json"
        if not sound_tokenizer_path.is_file():
            sound_tokenizer_path = sound_tokenizer_local / sound_tokenizer_name
            sound_tokenizer_config_path = sound_tokenizer_path.with_suffix(".json")
        assert sound_tokenizer_path.is_file(), f"Sound tokenizer checkpoint not found: {sound_tokenizer_path}"
        assert sound_tokenizer_config_path.is_file(), f"Sound tokenizer config not found: {sound_tokenizer_config_path}"

    vlm_config = model_dict["config"]["vlm_config"]
    tokenizer_config = vlm_config["tokenizer"]
    vision_encoder_model = (
        tokenizer_config.get("pretrained_model_name")
        or tokenizer_config.get("tokenizer_type")
        or vlm_config["model_name"]
    )

    if not args.config_only:
        _args = _convert_model_to_diffusers.Args(
            checkpoint_path=str(checkpoint_path),
            output=str(args.output_path),
            save_pipeline=True,
            dtype="bf16",
            sound_tokenizer_path=str(sound_tokenizer_path) if sound_tokenizer_path else None,
            sound_tokenizer_config_path=str(sound_tokenizer_config_path) if sound_tokenizer_config_path else None,
            include_sound_tokenizer=supports_sound,
            vision_encoder_model=vision_encoder_model,
            skip_vision_encoder=args.skip_vision_encoder,
            distilled_scheduler=args.distilled_scheduler,
        )
        _convert_model_to_diffusers.convert_model_to_diffusers(_args)

    # Add vlm files
    vlm_repository = model_dict["config"]["vlm_config"]["model_name"]
    vlm_checkpoint = CheckpointDirHf(
        repository=vlm_repository,
        revision="main",
        include=("*.jinja", "*.json", "*.txt"),
    )
    vlm_checkpoint_path = vlm_checkpoint.download()
    for pattern in vlm_checkpoint.include:
        for p in Path(vlm_checkpoint_path).glob(pattern):
            if p.name == "model.safetensors.index.json":
                continue
            shutil.copy(p, args.output_path / p.name)

    # Add top-level config
    config_dict = deserialize_config_dict(args.output_path / "config.json")
    config_dict["architectures"] = ["Cosmos3ForConditionalGeneration"]
    config_dict["model_type"] = "cosmos3_omni"
    # vLLM's `_prepare_weights` breaks after the first pattern with any match, so
    # collapse to a single glob spanning both component subdirs. The unified
    # `model.safetensors.index.json` written below dedupes the consolidated shard.
    config_dict["allow_patterns_overrides"] = ["*/*.safetensors"]
    config_dict["model"] = _build_public_export_model_config(model_dict)
    serialize_config_dict(config_dict, args.output_path / "config.json")

    if not args.config_only:
        # Add top-level index
        _write_diffusers_weight_index(args.output_path)

        # Modular pipeline index (for diffusers ModularPipeline.from_pretrained).
        # Written alongside the task-based model_index.json produced by the save.
        _write_modular_model_index(args.output_path, repo_id=args.repo_id or args.output_path.name)

    checkpoint_metadata_path = checkpoint_path / "checkpoint.json"
    if not checkpoint_metadata_path.is_file():
        checkpoint_metadata_path = checkpoint_path.parent / "checkpoint.json"
    shutil.copy(checkpoint_metadata_path, args.output_path / "checkpoint.json")

    print(f"Saved diffusers checkpoint to {args.output_path}")


def main():
    args = tyro.cli(Args, description=__doc__, config=(tyro.conf.OmitArgPrefixes,))
    convert_model_to_diffusers(args)


if __name__ == "__main__":
    main()
