# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from pathlib import Path
from typing import get_args

import pydantic
import pytest
import safetensors.torch
import torch
import torchvision.transforms.functional
from typing_extensions import Self

from cosmos_framework.data.vfm.utils import VIDEO_RES_SIZE_INFO
from cosmos_framework.inference.metrics import compute_action_mse, compute_psnr
from cosmos_framework.inference.args import (
    IMAGE_ONLY_RESOLUTIONS,
    AspectRatio,
    InferenceResolution,
    ModelMode,
    OmniSampleArgs,
    OmniSampleOverrides,
    _load_modality_defaults,
)
from cosmos_framework.inference.common.args import SampleOutputs
from cosmos_framework.inference.common.checkpoints import CHECKPOINTS, DATASETS, DatasetConfig
from cosmos_framework.inference.fixtures.args import MAX_GPUS
from cosmos_framework.inference.fixtures.script import INPUT_DIR, ScriptConfig, ScriptRunner, script_test
from cosmos_framework.utils.checkpoint_db import CheckpointConfig

_CURRENT_DIR = Path(__file__).parent.absolute()
_TEST_DIR = _CURRENT_DIR / "_test"

_TEMPORAL_COMPRESSION = 4  # wan2pt2_vae_4x16x16
_T2V_DEFAULTS = _load_modality_defaults("text2video")


def _vae_output_frames(num_frames: int) -> int:
    """Round up to the nearest valid output frame count (t*k + 1)."""
    t = _TEMPORAL_COMPRESSION
    return ((num_frames - 1 + t - 1) // t) * t + 1


# Excluded from the sweep: Cosmos-Guardrail1 resolution-sensitive FP on 720p t2v.
_GUARDRAIL_BLOCKED_RESOLUTIONS: frozenset[str] = frozenset({"720"})

# Image-only resolutions can't run in the t2v sweep (num_frames > 1 is rejected).
_T2V_BLOCKED_RESOLUTIONS: frozenset[str] = _GUARDRAIL_BLOCKED_RESOLUTIONS | IMAGE_ONLY_RESOLUTIONS

_SWEEP_CASES: list[tuple[str, dict]] = [
    *((f"res_{r}", {"resolution": r}) for r in get_args(InferenceResolution) if r not in _T2V_BLOCKED_RESOLUTIONS),
    *((f"ar_{ar}", {"resolution": "256", "aspect_ratio": ar}) for ar in get_args(AspectRatio)),
    *((f"nframes_{n}", {"resolution": "256", "num_frames": n}) for n in (None, 189)),
]

_OMNI_SUPER_MODALITIES: list[str] = ["t2v", "t2i", "i2v", "i2i", "v2v"]


def test_assets():
    overrides_list = OmniSampleOverrides.from_files([INPUT_DIR / "omni/*.json*"])
    assert overrides_list
    reasoner_overrides = OmniSampleOverrides.from_files([INPUT_DIR / "reasoner/*.json*"])
    assert reasoner_overrides
    assert all(o.model_mode == ModelMode.REASONER for o in reasoner_overrides)


def _check_inference_output(input_files: list[Path], output_dir: Path) -> list[SampleOutputs]:
    sample_args_list = OmniSampleArgs.from_files([output_dir / "*/sample_args.json"])
    assert sample_args_list
    sample_outputs_list: list[SampleOutputs] = []
    for sample_args in sample_args_list:
        (sample_outputs,) = SampleOutputs.from_files([sample_args.output_dir / "sample_outputs.json"])
        sample_outputs_list.append(sample_outputs)

        assert len(sample_outputs.outputs) == sample_args.num_outputs
        for output in sample_outputs.outputs:
            for file in output.files:
                assert file.is_file()

            vision_files = [f for f in output.files if f.stem == "vision"]
            assert len(vision_files) == 1
    return sample_outputs_list


def _get_video_dims(path: Path) -> tuple[int, int, int]:
    """Return (width, height, frame_count) of a video file."""
    import av

    with av.open(str(path), mode="r") as container:
        stream = container.streams.video[0]
        frame_count = int(stream.frames)
        if frame_count == 0:
            frame_count = sum(1 for _ in container.decode(video=0))
        return int(stream.width), int(stream.height), frame_count


def _compute_video_metrics(gt_video_cthw_uint8: torch.Tensor, pred_path: Path, mode: str) -> dict[str, float]:
    from cosmos_framework.inference.vision import read_media_frames
    from cosmos_framework.utils import log

    pred, _ = read_media_frames(pred_path, max_frames=gt_video_cthw_uint8.shape[1] + 1)
    pred = pred[..., : gt_video_cthw_uint8.shape[-2], : gt_video_cthw_uint8.shape[-1]]
    gt = gt_video_cthw_uint8

    if pred.shape != gt.shape:
        if pred.shape[-2:] != gt.shape[-2:]:
            raise ValueError(f"video spatial mismatch: gt {tuple(gt.shape)} vs pred {tuple(pred.shape)} ({pred_path})")
        if mode == "vision":
            min_t = min(gt.shape[1], pred.shape[1])
            log.info(
                f"vision frame-count mismatch trimmed to {min_t} (gt T={gt.shape[1]}, pred T={pred.shape[1]}, "
                f"{pred_path}); likely due to generation aligned to 4k+1 frames "
                "(latent temporal factor 4 + 1 conditioning frame), while the on-disk GT "
                "keeps the raw clip length."
            )
            gt = gt[:, :min_t]
            pred = pred[:, :min_t]
        else:
            raise ValueError(f"video shape mismatch: gt {tuple(gt.shape)} vs pred {tuple(pred.shape)} ({pred_path})")

    return {"psnr": compute_psnr(gt, pred)}


def _compute_action_metrics(gt_action_td: torch.Tensor, pred_action_list: list) -> dict[str, float]:
    pred = torch.tensor(pred_action_list, dtype=torch.float32)
    if pred.shape != gt_action_td.shape:
        raise ValueError(f"action shape mismatch: gt {tuple(gt_action_td.shape)} vs pred {tuple(pred.shape)}")
    return {"action_mse": compute_action_mse(gt_action_td, pred)}


def _compute_sample_metrics(
    mode: ModelMode,
    gt_video_cthw: torch.Tensor | None,
    gt_action_td: torch.Tensor | None,
    sample_output: SampleOutputs,
    sample_dir: Path,
    vision_extension: str,
) -> dict[str, float]:
    out: dict[str, float] = {"mode": mode.value, "name": sample_dir.name}
    if mode in (ModelMode.FORWARD_DYNAMICS, ModelMode.POLICY):
        if gt_video_cthw is None:
            raise ValueError(f"mode={mode.value!r} requires GT video but data_batch had none")
        out.update(_compute_video_metrics(gt_video_cthw, sample_dir / f"vision{vision_extension}", mode.value))
    if mode in (ModelMode.INVERSE_DYNAMICS, ModelMode.POLICY):
        pred_action = sample_output.outputs[0].content.get("action") if sample_output.outputs else None
        if pred_action is None:
            raise ValueError(f"mode={mode.value!r} requires predicted action but content has none")
        if gt_action_td is None:
            raise ValueError(f"mode={mode.value!r} requires GT action but data_batch had none")
        out.update(_compute_action_metrics(gt_action_td, pred_action))
    return out


def _omni_after_script(runner: ScriptRunner, cfg: ScriptConfig) -> None:
    inference_dir = runner.output_dir / "inference"
    sample_outputs_list = _check_inference_output([runner.input_dir / "omni/*json"], inference_dir)
    # Skip golden PSNR/MSE at L0 — SMOKE mode runs a degenerate model that won't hit real thresholds.
    if runner.level == 0:
        return
    failures: list[str] = []
    for sample_outputs in sample_outputs_list:
        failures.extend(_check_action_golden(sample_outputs, inference_dir / sample_outputs.name))
    assert not failures, "Golden checks failed:\n  " + "\n  ".join(failures)


def _reasoner_after_script(runner: ScriptRunner, cfg: ScriptConfig) -> None:
    inference_dir = runner.output_dir / "inference"
    sample_args_list = OmniSampleArgs.from_files([inference_dir / "*/sample_args.json"])
    assert sample_args_list
    for sample_args in sample_args_list:
        assert sample_args.model_mode.is_reasoner
        (sample_outputs,) = SampleOutputs.from_files([sample_args.output_dir / "sample_outputs.json"])
        assert len(sample_outputs.outputs) == sample_args.num_outputs
        for output in sample_outputs.outputs:
            for file in output.files:
                assert file.is_file()
            text_files = [f for f in output.files if f.stem == "reasoner_text"]
            assert len(text_files) == 1


def _omni_super_before_script(runner: ScriptRunner, cfg: ScriptConfig) -> None:
    """Stage only the omni modalities supported by Cosmos3-Super (32B)."""
    src_dir = runner.input_dir / "omni"
    dst_dir = runner.tmp_input_dir / "omni"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(src_dir.glob("*.json")):
        if not any(src.stem == m or src.stem.startswith(f"{m}_") for m in _OMNI_SUPER_MODALITIES):
            continue
        dst = dst_dir / src.name
        dst.unlink(missing_ok=True)
        dst.symlink_to(src)


def _omni_param_before_script(runner: ScriptRunner, cfg: ScriptConfig) -> None:
    """Stage t2v-based parameter-sweep cases into a temp input root."""
    input_root = runner.tmp_input_dir
    input_dir = input_root / "omni"
    base_case = json.loads((runner.input_dir / "omni" / "t2v.json").read_text())

    input_dir.mkdir(parents=True, exist_ok=True)

    prompt_file = runner.input_dir / "t2v_prompt.txt"
    prompt_target = input_root / prompt_file.name
    prompt_target.unlink(missing_ok=True)
    prompt_target.symlink_to(prompt_file)

    for name, payload in _SWEEP_CASES:
        (input_dir / f"{name}.json").write_text(json.dumps({**base_case, "name": name, **payload}, indent=4) + "\n")


def _omni_param_after_script(runner: ScriptRunner, cfg: ScriptConfig) -> None:
    """Validate parameter-sweep outputs."""
    inference_dir = runner.output_dir / "inference"
    for name, payload in _SWEEP_CASES:
        resolution = payload["resolution"]
        aspect_ratio = payload.get("aspect_ratio", _T2V_DEFAULTS["aspect_ratio"])
        num_frames = payload.get("num_frames") or _T2V_DEFAULTS["num_frames"]
        expected_frames = _vae_output_frames(num_frames)

        vision_path = inference_dir / name / "vision.mp4"
        assert vision_path.is_file(), f"{name}: missing {vision_path.relative_to(inference_dir)}"
        width, height, frame_count = _get_video_dims(vision_path)
        expected_width, expected_height = VIDEO_RES_SIZE_INFO[resolution][aspect_ratio]
        assert (width, height) == (expected_width, expected_height), (
            f"{name}: {width}x{height} != {expected_width}x{expected_height}"
        )
        assert frame_count == expected_frames, f"{name}: {frame_count} frames != {expected_frames}"


def _dcp_checkpoint_after_script(runner: ScriptRunner, cfg: ScriptConfig) -> None:
    _check_inference_output([runner.input_dir / "omni/t2v.json"], runner.output_dir / "inference")


def _action_after_script(runner: ScriptRunner, cfg: ScriptConfig) -> None:
    st_files = list(runner.output_dir.rglob("output.safetensors"))
    assert st_files, f"No output.safetensors found under {runner.output_dir}"
    for f in st_files:
        tensors = safetensors.torch.load_file(f)
        if "action" in tensors:
            action = tensors["action"]
            assert action.ndim >= 2, (
                f"{f}: expected action to have >= 2 dims (T, D), got {action.ndim} dims: {tuple(action.shape)}"
            )
            print(f"PASS: '{f.relative_to(runner.output_dir)}' action shape={tuple(action.shape)}")


def _load_canonical_gt_video(sample_dir: Path) -> torch.Tensor:
    """Load the raw GT video and resize it to the canonical (unpadded) shape."""
    from cosmos_framework.inference.vision import read_media_frames

    image_size = safetensors.torch.load_file(sample_dir / "sample_data.safetensors")["image_size"][0]
    orig_h, orig_w = int(image_size[2].item()), int(image_size[3].item())
    raw, _ = read_media_frames(sample_dir / "inputs" / "vision.mp4", max_frames=1024)
    return torchvision.transforms.functional.resize(
        raw,
        [orig_h, orig_w],
        interpolation=torchvision.transforms.functional.InterpolationMode.BICUBIC,
        antialias=True,
    )


def _load_gt_action(url_or_path: str, sample_dir: Path) -> torch.Tensor:
    """Download (if URL) and load the GT action JSON as a float32 tensor."""
    from cosmos_framework.inference.common.args import download_file

    path = download_file(url_or_path, sample_dir, "golden_action")
    return torch.tensor(json.loads(Path(path).read_text()), dtype=torch.float32)


def _check_action_golden(sample_outputs: SampleOutputs, sample_dir: Path) -> list[str]:
    """Run PSNR/MSE checks against thresholds in the sample's `extra` block."""
    extra = sample_outputs.args.get("extra") or {}
    psnr_min: float | None = extra.get("golden_psnr_min")
    mse_max: float | None = extra.get("golden_mse_max")
    if psnr_min is None and mse_max is None:
        return []

    mode = ModelMode(sample_outputs.args["model_mode"])
    gt_video = _load_canonical_gt_video(sample_dir) if mode in (ModelMode.FORWARD_DYNAMICS, ModelMode.POLICY) else None
    gt_action = _load_gt_action(extra["golden_action_path"], sample_dir) if mse_max is not None else None
    metrics = _compute_sample_metrics(mode, gt_video, gt_action, sample_outputs, sample_dir, ".mp4")

    failures: list[str] = []
    if psnr_min is not None and "psnr" in metrics:
        psnr = metrics["psnr"]
        print(f"[{sample_dir.name}] PSNR={psnr:.2f} dB (min {psnr_min:.2f})")
        if psnr < psnr_min:
            failures.append(f"{sample_dir.name}: PSNR {psnr:.2f} < {psnr_min:.2f}")
    if mse_max is not None and "action_mse" in metrics:
        mse = metrics["action_mse"]
        print(f"[{sample_dir.name}] MSE={mse:.4f} (max {mse_max:.4f})")
        if mse > mse_max:
            failures.append(f"{sample_dir.name}: MSE {mse:.4f} > {mse_max:.4f}")
    return failures


class SftGetEnv(pydantic.BaseModel):
    base_checkpoint_name: str
    config_file: str
    job_name: str
    dataset_name: str

    _dataset: DatasetConfig | None = None
    _base_checkpoint: CheckpointConfig | None = None

    @pydantic.model_validator(mode="after")
    def validate_model(self) -> Self:
        self._dataset = DATASETS[self.dataset_name]
        self._base_checkpoint = CHECKPOINTS[f"{self.base_checkpoint_name}-Train"]
        return self

    def __call__(self, runner: ScriptRunner, cfg: ScriptConfig) -> dict[str, str]:
        assert self._dataset is not None
        assert self._base_checkpoint is not None
        return {
            "BASE_CHECKPOINT_NAME": self.base_checkpoint_name,
            "CONFIG_FILE": self.config_file,
            "JOB_NAME": self.job_name,
            "DATASET_PATH": self._dataset.hf.download(),
            "BASE_CHECKPOINT_PATH": self._base_checkpoint.hf.download(),
        }


_SFT_GET_ENV = {
    "vision": SftGetEnv(
        base_checkpoint_name="Cosmos3-Nano",
        config_file="cosmos3/configs/experiment/vision_sft_nano.yaml",
        job_name="vision_sft_nano",
        dataset_name="nvidia/bridge-v2-subset-synthetic-captions",
    ),
    "vision_super": SftGetEnv(
        base_checkpoint_name="Cosmos3-Super",
        config_file="cosmos3/configs/experiment/vision_sft_super.yaml",
        job_name="vision_sft_super",
        dataset_name="nvidia/bridge-v2-subset-synthetic-captions",
    ),
}
_DEFAULT_SFT_NAME = "vision"

_script_configs = [
    ScriptConfig(
        script=_TEST_DIR / "omni.sh",
        levels=(0, 1, 2),
        gpus=(1, MAX_GPUS, MAX_GPUS),
        after_script=_omni_after_script,
    ),
    ScriptConfig(
        script=_TEST_DIR / "reasoner.sh",
        levels=(0, 1, 2),
        gpus=(1, MAX_GPUS, MAX_GPUS),
        after_script=_reasoner_after_script,
    ),
    ScriptConfig(
        script=_TEST_DIR / "latency.sh",
        levels=(1, 2),
        gpus=(0, MAX_GPUS, MAX_GPUS),
        after_script=_omni_after_script,
    ),
    ScriptConfig(
        script=_TEST_DIR / "omni-super.sh",
        use_tmp_input_dir=True,
        levels=(1, 2),
        gpus=(1, MAX_GPUS, MAX_GPUS),
        before_script=_omni_super_before_script,
        after_script=_omni_after_script,
    ),
    ScriptConfig(
        script=_TEST_DIR / "sft.sh",
        levels=(0, 1, 2),
        gpus=(1, MAX_GPUS, MAX_GPUS),
        get_env=_SFT_GET_ENV[_DEFAULT_SFT_NAME],
    ),
    ScriptConfig(
        name="sft_super",
        script=_TEST_DIR / "sft.sh",
        levels=(0, 1, 2),
        gpus=(1, MAX_GPUS, MAX_GPUS),
        get_env=_SFT_GET_ENV["vision_super"],
    ),
]

for name, get_env in _SFT_GET_ENV.items():
    marks = []
    if name in (_DEFAULT_SFT_NAME, "vision_super"):
        # Already tested in the dedicated `sft.sh` entries above.
        marks.append(pytest.mark.manual)
    _script_configs += [
        ScriptConfig(
            name=f"sft_{name}",
            script=_TEST_DIR / "sft_param.sh",
            levels=(0, 2),
            gpus=(1, MAX_GPUS, MAX_GPUS),
            marks=marks,
            get_env=get_env,
        ),
        ScriptConfig(
            name=f"eval_{name}",
            script=_TEST_DIR / "eval.sh",
            levels=(0, 2),
            gpus=(1, MAX_GPUS, MAX_GPUS),
            marks=marks,
            get_env=get_env,
            after_script=_action_after_script,
        ),
    ]


@script_test(_script_configs)
class TestScript: ...
