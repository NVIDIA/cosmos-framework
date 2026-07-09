# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""WebSocket inference server for RoboLab/openpi-style Action policy serving.

The server uses OpenPI's WebsocketPolicyServer and speaks its msgpack+NumPy protocol:

- on connection, it sends an empty metadata dict;
- each client message is an observation dict;
- each response is a dict with ``action`` and, when enabled, ``video``.

Example:

  PYTHONPATH=. python -m cosmos_framework.scripts.action_policy_server_robolab \
    --checkpoint-path nvidia/Cosmos3-Nano-Policy-DROID \
    --port 8000
"""

# Initialize the script runtime before any cosmos-framework imports, mirroring
# the LIBERO server.  The shared helpers below (single-rank distributed init,
# local IP discovery, frozen-config EMA disable) live in
# ``action_policy_server_utils`` so this module doesn't have to import from
# the sibling LIBERO server just to share runtime utilities.
from cosmos_framework.inference.common.init import init_script

init_script()

import socket
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pydantic
import torch
import torch.nn.functional as F
import tyro

from cosmos_framework.data.generator.action.domain_utils import get_domain_id
from cosmos_framework.data.generator.action.pose_utils import (
    build_abs_pose_from_components,
    convert_rotation,
    pose_abs_to_rel,
    pose_rel_to_abs,
)
from cosmos_framework.data.generator.action.transforms import ActionTransformPipeline
from cosmos_framework.data.generator.joint_dataloader import IterativeJointDataLoader
from cosmos_framework.inference.args import OmniSetupArgs, OmniSetupOverrides
from cosmos_framework.inference.common.args import ConfigFileType, ConfigOverrides, tyro_cli
from cosmos_framework.inference.common.config import deserialize_config, deserialize_config_dict, load_config
from cosmos_framework.inference.common.init import init_output_dir
from cosmos_framework.inference.inference import OmniInference
from cosmos_framework.scripts.action_policy_server_utils import (
    DEFAULT_FALLBACK_OUTPUT_DIR,
    disable_runtime_ema_for_frozen_config,
    get_local_ip,
    maybe_init_distributed,
)
from cosmos_framework.utils import log
from cosmos_framework.utils.checkpoint_db import CheckpointDirHf
from cosmos_framework.utils.lazy_config import instantiate

_DEFAULT_DROID_POLICY_CHECKPOINT = "nvidia/Cosmos3-Nano-Policy-DROID"
_DEFAULT_CONDITIONING_FPS = 15.0
_DEFAULT_ACTION_CHUNK_SIZE = 32
_DEFAULT_IMAGE_HEIGHT = 540
_DEFAULT_IMAGE_WIDTH = 640
_DEFAULT_ACTION_DIM = 8
_DEFAULT_ROBOLAB_OUTPUT_DIR = DEFAULT_FALLBACK_OUTPUT_DIR / "robolab"
_CONCAT_VIEW_DESCRIPTION = (
    "The top row is from the wrist-mounted camera. "
    "The bottom row contains two horizontally concatenated third-person perspective views of the scene from opposite "
    "sides, with the robot visible."
)
_DEFAULT_HF_REVISION = "main"
_ROBOLAB_POLICY_HF_REPOSITORIES = {
    "Cosmos3-Nano-Policy-DROID": "nvidia/Cosmos3-Nano-Policy-DROID",
    "nvidia/Cosmos3-Nano-Policy-DROID": "nvidia/Cosmos3-Nano-Policy-DROID",
}

ActionSpace = Literal["joint_pos", "midtrain"]


def _load_checkpoint_metadata(checkpoint_path: str) -> dict[str, Any] | None:
    if "://" in checkpoint_path:
        return None
    checkpoint_dir = Path(checkpoint_path).expanduser().absolute()
    if (checkpoint_dir / "model").is_dir():
        checkpoint_dir = checkpoint_dir / "model"
    metadata_path = checkpoint_dir / "checkpoint.json"
    if not metadata_path.exists():
        return None
    return deserialize_config_dict(metadata_path)


def _load_training_config_from_metadata(metadata: dict[str, Any]) -> Any | None:
    config_file = metadata.get("config_file")
    if not isinstance(config_file, str) or not config_file:
        return None
    config_overrides = ConfigOverrides(
        config_file=config_file,
        experiment=str(metadata.get("experiment", "")),
        experiment_overrides=list(metadata.get("experiment_overrides", [])),
    )
    config_args = config_overrides.build_config()
    if config_args.config_file_type == ConfigFileType.MODULE:
        return load_config(config_args.config_file, config_args.experiment, overrides=config_args.experiment_overrides)
    return deserialize_config(Path(config_args.config_file))


def _load_training_config(setup_args: OmniSetupArgs, checkpoint_path: str) -> Any | None:
    metadata = _load_checkpoint_metadata(checkpoint_path)
    if metadata is not None:
        try:
            config = _load_training_config_from_metadata(metadata)
        except Exception as exc:
            log.warning(f"[robolab-policy-server] could not load checkpoint metadata config for transforms: {exc}")
            config = None
        if config is not None:
            return config

    if setup_args.config_file_type == ConfigFileType.MODULE and not setup_args.experiment:
        return None

    try:
        return setup_args.load_config()
    except Exception as exc:
        log.warning(f"[robolab-policy-server] could not load training config for transforms: {exc}")
        return None


def _resolve_checkpoint_path(checkpoint_path: str, *, hf_revision: str) -> str:
    if Path(checkpoint_path).expanduser().exists():
        return checkpoint_path

    repository = _ROBOLAB_POLICY_HF_REPOSITORIES.get(checkpoint_path)
    if repository is None:
        return checkpoint_path

    log.info(
        f"[robolab-policy-server] downloading consolidated checkpoint from Hugging Face: "
        f"repository={repository!r} revision={hf_revision!r}"
    )
    return CheckpointDirHf(repository=repository, revision=hf_revision).download()


def _validate_checkpoint(checkpoint_path: str, *, allow_dcp_checkpoint: bool) -> None:
    if checkpoint_path in OmniSetupOverrides.CHECKPOINTS:
        return
    if "://" in checkpoint_path:
        if allow_dcp_checkpoint:
            return
        raise ValueError(
            "RoboLab OSS serving expects a consolidated local safetensors checkpoint directory, not a DCP path. "
            "Run cosmos_framework.scripts.export_model first and pass the exported model directory, or pass "
            "--allow-dcp-checkpoint to opt into direct DCP loading."
        )

    checkpoint_dir = Path(checkpoint_path).expanduser().absolute()
    if (checkpoint_dir / "model").is_dir():
        checkpoint_dir = checkpoint_dir / "model"
    if any(checkpoint_dir.glob("*.distcp")):
        if allow_dcp_checkpoint:
            return
        raise ValueError(
            "RoboLab OSS serving expects a consolidated safetensors checkpoint, but found a DCP checkpoint. "
            "Run cosmos_framework.scripts.export_model first and pass the exported model directory, or pass "
            "--allow-dcp-checkpoint to opt into direct DCP loading."
        )
    has_config = (checkpoint_dir / "config.json").exists()
    has_consolidated_safetensors = any(checkpoint_dir.glob("*.safetensors"))
    has_diffusers_safetensors_index = (checkpoint_dir / "model.safetensors.index.json").exists()
    if not checkpoint_dir.is_dir() or not has_config or not (
        has_consolidated_safetensors or has_diffusers_safetensors_index
    ):
        raise ValueError(f"Invalid safetensors checkpoint directory: {checkpoint_dir}")


def _ensure_rgb_uint8_image(value: Any, key: str) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"{key!r} must have shape [H,W,3], got {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _ensure_2d_float_array(value: Any, key: str, width: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise ValueError(f"{key!r} must have shape [T,D] or [D], got {array.shape}")
    if width is not None and array.shape[-1] != width:
        raise ValueError(f"{key!r} must have width {width}, got {array.shape[-1]}")
    return np.ascontiguousarray(array)


def _ensure_gripper_array(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[-1] != 1:
        raise ValueError(f"'observation/gripper_position' must have shape [T,1], [T], or scalar, got {array.shape}")
    return np.ascontiguousarray(array)


def _resize_rgb_uint8(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()  # [1,3,H,W]
    resized = F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)  # [1,3,H2,W2]
    return resized.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)  # [H2,W2,3]


def _compose_roboarena_views(obs: dict[str, Any]) -> np.ndarray | None:
    required_keys = (
        "observation/wrist_image_left",
        "observation/exterior_image_1_left",
        "observation/exterior_image_2_left",
    )
    if not all(key in obs for key in required_keys):
        return None
    wrist = _ensure_rgb_uint8_image(obs["observation/wrist_image_left"], "observation/wrist_image_left")
    left_raw = _ensure_rgb_uint8_image(obs["observation/exterior_image_1_left"], "observation/exterior_image_1_left")
    right_raw = _ensure_rgb_uint8_image(obs["observation/exterior_image_2_left"], "observation/exterior_image_2_left")
    half_h, half_w = wrist.shape[0] // 2, wrist.shape[1] // 2
    left = _resize_rgb_uint8(left_raw, (half_h, half_w))
    right = _resize_rgb_uint8(right_raw, (half_h, half_w))
    return np.concatenate([wrist, np.concatenate([left, right], axis=1)], axis=0)


def _extract_observation_image(obs: dict[str, Any]) -> np.ndarray:
    if "observation/image" in obs:
        return _ensure_rgb_uint8_image(obs["observation/image"], "observation/image")
    image = _compose_roboarena_views(obs)
    if image is not None:
        return image
    raise ValueError("Observation must contain 'observation/image' or RoBoArena wrist/exterior image keys")


def _build_data_batch_from_sample(sample: dict[str, Any]) -> dict[str, Any]:
    data_batch: dict[str, Any] = {}
    for key, value in sample.items():
        if key in IterativeJointDataLoader._MULTI_ITEM_KEYS:
            data_batch[key] = [[value]]
        elif isinstance(value, torch.Tensor):
            data_batch[key] = [value.unsqueeze(0)]  # value: [...], batch item: [1,...]
        else:
            data_batch[key] = [value]
    return data_batch


def _load_openpi_websocket_policy_server() -> type[Any]:
    try:
        from openpi_server.websocket_policy_server import WebsocketPolicyServer
    except ModuleNotFoundError:
        try:
            from openpi.serving.websocket_policy_server import WebsocketPolicyServer
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "RoboLab WebSocket serving uses OpenPI's WebsocketPolicyServer. Install it with "
                "`uv sync --all-extras --group=cu130-train --group=policy-server`, "
                "or install the full Physical-Intelligence/openpi package."
            ) from exc
    return WebsocketPolicyServer


class _DummyDataset(torch.utils.data.IterableDataset):
    def __iter__(self) -> Any:
        return iter(())


@dataclass(frozen=True)
class RobolabPolicyConfig:
    checkpoint_path: str
    domain_name: str
    decode_video: bool
    seed: int
    deterministic_seed: bool
    guidance: float
    num_steps: int
    shift: float
    conditioning_fps: float
    resolution: str | None
    action_chunk_size: int
    action_dim: int
    image_height: int = _DEFAULT_IMAGE_HEIGHT
    image_width: int = _DEFAULT_IMAGE_WIDTH
    action_space: ActionSpace = "joint_pos"
    use_state: bool = True
    history_length: int = 1


class RobolabServerArgs(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", use_attribute_docstrings=True)

    checkpoint_path: str = _DEFAULT_DROID_POLICY_CHECKPOINT
    """Consolidated local safetensors checkpoint directory, registered checkpoint name, or DCP path with --allow-dcp-checkpoint."""
    hf_revision: str = _DEFAULT_HF_REVISION
    """Hugging Face revision used when --checkpoint-path is a supported public RoboLab policy repository."""
    allow_dcp_checkpoint: bool = False
    """If set, allow direct DCP/S3 checkpoint loading instead of requiring a consolidated safetensors export."""
    experiment: str | None = None
    """Experiment name for DCP checkpoints using module configs, e.g. droid_lerobot_8b_policy."""
    experiment_overrides: list[str] = pydantic.Field(default_factory=list)
    """Hydra experiment overrides forwarded to OmniSetup for DCP checkpoint loading."""
    credential_path: str | None = None
    """Optional checkpoint object-store credential path for DCP/S3 loading."""

    port: int = 8000
    """WebSocket port to bind."""
    host: str = "0.0.0.0"
    """WebSocket host to bind."""
    domain_name: str = "droid_lerobot"
    """Action domain name passed to get_domain_id()."""
    decode_video: bool = False
    """If set, decode and return the predicted rollout video as a uint8 NumPy array."""

    output_dir: Path | None = None
    """Output directory for OmniInference. Defaults to /tmp/cosmos3_action_server/robolab."""
    sampler: Literal["unipc", "edm"] = "unipc"
    """Diffusion sampler used by OmniInference."""

    seed: int = 0
    """Base generation seed used to initialize the request RNG."""
    deterministic_seed: bool = False
    """Use the same seed for every request. If false, advance a NumPy RNG seeded by --seed."""
    guidance: float = 3.0
    """Guidance scale for denoising."""
    num_steps: int = 4
    """Number of denoising steps."""
    shift: float = 5.0
    """UniPC sampler shift."""

    resolution: str | None = "480"
    """Action transform resolution. The default matches the released DROID RoboLab policy."""
    conditioning_fps: float | None = _DEFAULT_CONDITIONING_FPS
    """Conditioning FPS. The default matches the released DROID RoboLab policy."""
    action_chunk_size: int | None = _DEFAULT_ACTION_CHUNK_SIZE
    """Number of action steps to predict. The default matches the released DROID RoboLab policy."""
    action_dim: int | None = _DEFAULT_ACTION_DIM
    """Raw action dimension. The default matches the released DROID RoboLab policy."""
    image_height: int = _DEFAULT_IMAGE_HEIGHT
    """Input observation image height. The default matches the released DROID RoboLab policy."""
    image_width: int = _DEFAULT_IMAGE_WIDTH
    """Input observation image width. The default matches the released DROID RoboLab policy."""
    action_space: ActionSpace = "joint_pos"
    """RoboLab action representation to serve."""
    use_state: bool = True
    """Whether the first action row contains the current state."""
    history_length: int = 1
    """State/history action rows to trim from the generated action output."""


class RobolabPolicyService:
    def __init__(self, args: RobolabServerArgs) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for OmniMoTModel inference in this repo.")
        resolved_checkpoint_path = _resolve_checkpoint_path(args.checkpoint_path, hf_revision=args.hf_revision)
        args = args.model_copy(update={"checkpoint_path": resolved_checkpoint_path})
        _validate_checkpoint(args.checkpoint_path, allow_dcp_checkpoint=args.allow_dcp_checkpoint)
        maybe_init_distributed()

        setup_args = self._build_setup_args(args)
        log.info(
            f"[robolab-policy-server] loading model: checkpoint_path={setup_args.checkpoint_path!r} "
            f"config_file={setup_args.config_file!r} experiment={setup_args.experiment!r}"
        )
        pipe = OmniInference.create(setup_args)
        self.pipe: OmniInference = pipe
        self.model = pipe.model
        self.model.eval()
        assert isinstance(pipe.setup_args, OmniSetupArgs)
        self.setup_args: OmniSetupArgs = pipe.setup_args

        training_config = _load_training_config(self.setup_args, args.checkpoint_path)
        self._transform, inferred = self._build_transform(training_config, args)
        self.cfg = RobolabPolicyConfig(
            checkpoint_path=self.setup_args.checkpoint_path,
            domain_name=args.domain_name,
            decode_video=bool(args.decode_video),
            seed=int(args.seed),
            deterministic_seed=bool(args.deterministic_seed),
            guidance=float(args.guidance),
            num_steps=int(args.num_steps),
            shift=float(args.shift),
            conditioning_fps=float(
                args.conditioning_fps or inferred.get("conditioning_fps") or _DEFAULT_CONDITIONING_FPS
            ),
            resolution=args.resolution or inferred.get("resolution"),
            action_chunk_size=int(
                args.action_chunk_size or inferred.get("action_chunk_size") or _DEFAULT_ACTION_CHUNK_SIZE
            ),
            action_dim=int(args.action_dim or (8 if args.action_space == "joint_pos" else 10)),
            image_height=int(args.image_height),
            image_width=int(args.image_width),
            action_space=args.action_space,
            use_state=bool(args.use_state),
            history_length=int(args.history_length),
        )
        if self.cfg.history_length < (1 if self.cfg.use_state else 0):
            raise ValueError("--history-length must be >= 1 when --use-state is true")
        if self.cfg.image_height <= 0 or self.cfg.image_width <= 0:
            raise ValueError("--image-height and --image-width must be positive")

        self._lock = threading.Lock()
        self._rng = np.random.default_rng(self.cfg.seed)
        log.info(
            f"[robolab-policy-server] ready domain={self.cfg.domain_name!r} resolution={self.cfg.resolution!r} "
            f"action_space={self.cfg.action_space} action_dim={self.cfg.action_dim} "
            f"chunk={self.cfg.action_chunk_size} history={self.cfg.history_length} use_state={self.cfg.use_state} "
            f"image={self.cfg.image_height}x{self.cfg.image_width} fps={self.cfg.conditioning_fps} "
            f"guidance={self.cfg.guidance} num_steps={self.cfg.num_steps} shift={self.cfg.shift} "
            f"seed={self.cfg.seed} deterministic_seed={self.cfg.deterministic_seed}"
        )

    def _build_setup_args(self, args: RobolabServerArgs) -> OmniSetupArgs:
        setup_overrides: dict[str, Any] = {
            "checkpoint_path": args.checkpoint_path,
            "output_dir": args.output_dir or _DEFAULT_ROBOLAB_OUTPUT_DIR,
            "sampler": args.sampler,
        }
        if args.experiment is not None:
            setup_overrides["experiment"] = args.experiment
        if args.experiment_overrides:
            setup_overrides["experiment_overrides"] = list(args.experiment_overrides)
        if args.credential_path is not None:
            setup_overrides["credential_path"] = args.credential_path
        overrides = OmniSetupOverrides.model_validate(setup_overrides)
        setup_args = overrides.build_setup()
        init_output_dir(setup_args.output_dir)
        return disable_runtime_ema_for_frozen_config(setup_args)

    def _build_transform(self, training_config: Any | None, args: RobolabServerArgs) -> tuple[Any, dict[str, Any]]:
        inferred: dict[str, Any] = {}
        model_max_action_dim = getattr(getattr(self.model, "config", None), "max_action_dim", None)
        max_action_dim = int(model_max_action_dim) if isinstance(model_max_action_dim, int) else 64

        try:
            dataset_config = (
                training_config.dataloader_train.dataloaders.action_data.dataloader.dataset
                if training_config is not None
                else None
            )
            dataset_entry = dataset_config.list_of_datasets[0] if dataset_config is not None else None
            action_dataset_config = dataset_entry.dataset if dataset_entry is not None else None
        except (AttributeError, IndexError, TypeError):
            dataset_config = None
            dataset_entry = None
            action_dataset_config = None

        if dataset_config is None or dataset_entry is None:
            log.warning(
                "[robolab-policy-server] no training action dataset config found; using default ActionTransformPipeline"
            )
            return ActionTransformPipeline(max_action_dim=max_action_dim, cfg_dropout_rate=0.0), inferred

        if action_dataset_config is not None:
            chunk_length = getattr(action_dataset_config, "chunk_length", None)
            if isinstance(chunk_length, int):
                inferred["action_chunk_size"] = chunk_length
            fps = getattr(action_dataset_config, "fps", None)
            if isinstance(fps, (int, float)):
                inferred["conditioning_fps"] = float(fps)

        if args.resolution is not None:
            dataset_entry.resolution = args.resolution
        else:
            inferred_resolution = dataset_entry.resolution
            if inferred_resolution is None:
                inferred_resolution = dataset_config.resolution
            if inferred_resolution is not None:
                inferred["resolution"] = str(inferred_resolution)

        if dataset_config.cfg_dropout_rate != 0.0:
            dataset_config.cfg_dropout_rate = 0.0

        dataset_entry.dataset = {"_target_": f"{__name__}._DummyDataset"}

        wrapped_dataset = instantiate(dataset_config)
        return wrapped_dataset.transform, inferred

    def _next_seed(self) -> int:
        if self.cfg.deterministic_seed:
            return self.cfg.seed
        return int(self._rng.integers(0, 2**31))

    def _build_sample(self, obs: dict[str, Any]) -> dict[str, Any]:
        prompt = obs.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError("'prompt' must be a string")

        image = _extract_observation_image(obs)
        image_h = self.cfg.image_height
        image_w = self.cfg.image_width
        if image.shape[:2] != (image_h, image_w):
            image = _resize_rgb_uint8(image, (image_h, image_w))
        t_frames = self.cfg.action_chunk_size + 1
        video = torch.zeros((3, t_frames, image_h, image_w), dtype=torch.uint8)  # [3,T,H,W]
        video[:, 0] = torch.from_numpy(image.copy()).permute(2, 0, 1)  # [3,H,W]

        use_state_rows = 1 if self.cfg.use_state else 0
        action = torch.zeros(
            (self.cfg.action_chunk_size + use_state_rows, self.cfg.action_dim),
            dtype=torch.float32,
        )  # [T,D]
        history_action: torch.Tensor | None = None
        num_history_rows = self.cfg.history_length - use_state_rows
        gripper_position = 1.0 - _ensure_gripper_array(obs["observation/gripper_position"])

        if self.cfg.action_space == "joint_pos":
            joint_position = _ensure_2d_float_array(obs["observation/joint_position"], "observation/joint_position", 7)
            if self.cfg.use_state:
                action[0] = torch.from_numpy(np.concatenate((joint_position[-1], gripper_position[-1])))  # [D]
            if num_history_rows > 0:
                if len(joint_position) < num_history_rows + 1:
                    raise ValueError("Not enough joint_position rows for requested history_length")
                history_np = np.concatenate(
                    (joint_position[-num_history_rows - 1 : -1], gripper_position[-num_history_rows - 1 : -1]),
                    axis=-1,
                )
                history_action = torch.from_numpy(history_np).float()  # [H,D]

        if self.cfg.action_space == "midtrain":
            eef_pos = _ensure_2d_float_array(obs["observation/eef_pos"], "observation/eef_pos", 3)
            eef_quat = _ensure_2d_float_array(obs["observation/eef_quat"], "observation/eef_quat", 4)
            if self.cfg.use_state:
                rot6d = convert_rotation(eef_quat[-1], "quat_xyzw", "rot6d")
                action[0] = torch.from_numpy(np.concatenate((eef_pos[-1], rot6d, gripper_position[-1])))  # [D]
            if num_history_rows > 0:
                if len(eef_pos) < num_history_rows + 1 or len(eef_quat) < num_history_rows + 1:
                    raise ValueError("Not enough eef_pos/eef_quat rows for requested history_length")
                poses_abs = build_abs_pose_from_components(eef_pos, eef_quat, "quat_xyzw")
                poses_rel = pose_abs_to_rel(poses_abs, rotation_format="rot6d", pose_convention="backward_framewise")
                history_np = np.concatenate(
                    [poses_rel[-num_history_rows:], gripper_position[-num_history_rows:]],
                    axis=-1,
                )
                history_action = torch.from_numpy(history_np).float()  # [H,D]

        sample: dict[str, Any] = {
            "ai_caption": prompt,
            "video": video,
            "action": action,
            "conditioning_fps": torch.tensor(self.cfg.conditioning_fps, dtype=torch.long),  # []
            "mode": "policy",
            "domain_id": torch.tensor(get_domain_id(self.cfg.domain_name), dtype=torch.long),  # []
            "viewpoint": "concat_view",
            "additional_view_description": _CONCAT_VIEW_DESCRIPTION,
        }
        if history_action is not None:
            sample["history_action"] = history_action
        return self._transform(sample, self.cfg.resolution)

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        sample = self._build_sample(obs)
        data_batch = _build_data_batch_from_sample(sample)
        seed = self._next_seed()
        log.info(f"[robolab-policy-server] prompt={data_batch['ai_caption'][0]!r} seed={seed}")

        with self._lock:
            with torch.inference_mode():
                samples = self.model.generate_samples_from_batch(
                    data_batch,
                    guidance=self.cfg.guidance,
                    seed=[seed],
                    num_steps=self.cfg.num_steps,
                    shift=self.cfg.shift,
                )

        action = samples["action"][0][:, : self.cfg.action_dim]  # [T,D]
        action = action[self.cfg.history_length :]  # [T2,D]
        action_np = action.detach().cpu().numpy()  # [T2,D]
        action_np[:, -1] = 1.0 - action_np[:, -1]

        if self.cfg.action_space == "midtrain":
            eef_pos = _ensure_2d_float_array(obs["observation/eef_pos"], "observation/eef_pos", 3)
            eef_quat = _ensure_2d_float_array(obs["observation/eef_quat"], "observation/eef_quat", 4)
            initial_pose = np.eye(4, dtype=np.float32)
            initial_pose[:3, :3] = convert_rotation(eef_quat[-1], "quat_xyzw", "matrix")
            initial_pose[:3, 3] = eef_pos[-1]
            abs_pose = pose_rel_to_abs(
                action_np[:, :9],
                rotation_format="rot6d",
                pose_convention="backward_framewise",
                initial_pose=initial_pose,
            )
            position = abs_pose[1:, :3, 3]
            quat_xyzw = convert_rotation(abs_pose[1:, :3, :3], "matrix", "quat_xyzw")
            action_np = np.concatenate([position, quat_xyzw, action_np[:, 9:]], axis=-1)

        outputs: dict[str, Any] = {"action": action_np}
        if self.cfg.decode_video:
            pred_vision_latent = samples["vision"][0]  # [C,T,H,W]
            video = self.model.decode(pred_vision_latent)  # [1,C,T,H,W]
            video = ((video[0].clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 3, 0)  # [T,H,W,3]
            outputs["video"] = video.detach().cpu().numpy()
        return outputs


def serve(args: RobolabServerArgs) -> None:
    hostname = socket.gethostname()
    log.info(f"[robolab-policy-server] starting host={hostname} bind={args.host}:{int(args.port)}")
    service = RobolabPolicyService(args)
    local_ip = get_local_ip()
    log.info(f"[robolab-policy-server] Server accessible at: ws://{local_ip}:{int(args.port)}/")
    log.info(f"[robolab-policy-server] Health check: http://{local_ip}:{int(args.port)}/healthz")
    server_cls = _load_openpi_websocket_policy_server()
    server_cls(policy=service, host=args.host, port=int(args.port), metadata={}).serve_forever()


def main() -> None:
    cascade_subcommand_args = getattr(
        tyro.conf,
        "CascadeSubcommandArgs",
        tyro.conf.ConsolidateSubcommandArgs,
    )
    args = tyro_cli(
        RobolabServerArgs,
        description=__doc__,
        config=(
            tyro.conf.OmitArgPrefixes,
            cascade_subcommand_args,
            tyro.conf.OmitSubcommandPrefixes,
        ),
    )
    serve(args)


if __name__ == "__main__":
    main()
