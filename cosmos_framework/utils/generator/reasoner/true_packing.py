# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Dense Qwen3-VL position helpers used only by true sequence packing."""

from types import SimpleNamespace
from typing import Any

import torch

_UNSUPPORTED_PACKED_TEMPORAL_KEYS: tuple[str, ...] = (
    "video_fps",
    "fps",
    "audio_grid_thw",
    "action_grid_thw",
)
TRUE_PACKING_CPU_PREPARED_KEY = "true_packing_cpu_prepared"


def assert_packing_temporal_inputs_supported(data: dict[str, Any]) -> None:
    """Reject physical-time inputs whose packed M-RoPE semantics have not been proved.

    ``second_per_grid_ts`` is intentionally allowed: dense Qwen3-VL forwards it to the
    processor/model, but its training-time ``get_rope_index`` does not consume it.
    """
    present = [key for key in _UNSUPPORTED_PACKED_TEMPORAL_KEYS if data.get(key) is not None]
    if present:
        raise NotImplementedError(
            f"True-packing position parity is unproven for physical-time inputs {present}. "
            "Extend build_packed_position_ids and add packed-vs-padded position parity tests "
            "before enabling COSMOS_PACK_TRUE_PACKING for them."
        )


def build_packed_position_ids(
    config: Any,
    input_ids: torch.LongTensor,  # [1,S_pad]
    seq_lens: list[int],
    image_grid_thw: torch.LongTensor | None = None,  # [N_image,3]
    video_grid_thw: torch.LongTensor | None = None,  # [N_video,3]
) -> torch.Tensor:
    """Build per-sample-reset M-RoPE ids for one true-packed dense Qwen3-VL row on CPU.

    The unmodified dense Qwen3-VL position routine already resets each row independently
    and consumes concatenated image/video grids in row-major order. Reconstructing a small
    integer-only ``[num_samples,max_length]`` view therefore gives exactly the padded-path
    positions without changing the temporal or media ordering. This helper must run before
    the trainer's H2D copy: the upstream routine uses Python scalar/list reads that would
    otherwise synchronize CUDA. Only position tensors are reconstructed; model activations
    remain true-packed.

    Returns:
        Position ids with shape ``[3,1,S_pad]``.
    """
    if config.model_type != "qwen3_vl":
        raise NotImplementedError(
            "True-packing position ids are implemented for dense 'qwen3_vl' only; "
            f"got model_type={config.model_type!r}."
        )
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"True packing requires input_ids with shape [1,S_pad], got {tuple(input_ids.shape)}")
    if input_ids.device.type != "cpu":
        raise ValueError(
            f"build_packed_position_ids must run on CPU before the batch H2D copy; got input_ids on {input_ids.device}"
        )
    for name, grid in (("image_grid_thw", image_grid_thw), ("video_grid_thw", video_grid_thw)):
        if grid is not None and not isinstance(grid, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if grid is not None and grid.device.type != "cpu":
            raise ValueError(
                "build_packed_position_ids must receive CPU grid metadata before the batch H2D copy; "
                f"got {name} on {grid.device}"
            )
    if not seq_lens or any(length <= 0 for length in seq_lens):
        raise ValueError(f"seq_lens must contain positive logical lengths, got {seq_lens}")

    seq_total = sum(seq_lens)
    seq_padded = input_ids.shape[1]
    if seq_total > seq_padded:
        raise ValueError(f"sum(seq_lens)={seq_total} exceeds packed row length {seq_padded}")

    # Lazy import avoids a module-load cycle through the reasoner model package.
    from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import get_rope_index

    real_input_ids = input_ids[0, :seq_total]  # [S]
    spatial_merge_size = int(config.vision_config.spatial_merge_size)

    def _grid_rows(grid: torch.Tensor | None, name: str) -> list[tuple[int, int, int]]:
        if grid is None:
            return []
        if grid.ndim != 2 or grid.shape[1] != 3:
            raise ValueError(f"{name} must have shape [N,3], got {tuple(grid.shape)}")
        if bool((grid <= 0).any()):
            raise ValueError(f"{name} entries must be positive")
        if bool((grid[:, 1:] % spatial_merge_size != 0).any()):
            raise ValueError(f"{name} spatial dimensions must be divisible by spatial_merge_size")
        return [tuple(int(value) for value in row) for row in grid.tolist()]

    image_runs = [
        temporal * (height // spatial_merge_size) * (width // spatial_merge_size)
        for temporal, height, width in _grid_rows(image_grid_thw, "image_grid_thw")
    ]
    video_runs = [
        (height // spatial_merge_size) * (width // spatial_merge_size)
        for temporal, height, width in _grid_rows(video_grid_thw, "video_grid_thw")
        for _ in range(temporal)
    ]
    image_index = video_index = 0
    for sample_index, segment in enumerate(torch.split(real_input_ids, seq_lens)):
        tokens = [int(token) for token in segment.tolist()]
        for position, token in enumerate(tokens):
            if token != int(config.vision_start_token_id):
                continue
            if position + 1 >= len(tokens):
                raise ValueError(f"sample {sample_index} ends with a vision-start token")
            modality_token = tokens[position + 1]
            run_end = position + 1
            while run_end < len(tokens) and tokens[run_end] == modality_token:
                run_end += 1
            actual_run = run_end - (position + 1)
            if modality_token == int(config.image_token_id):
                if image_index >= len(image_runs) or actual_run != image_runs[image_index]:
                    expected = image_runs[image_index] if image_index < len(image_runs) else None
                    raise ValueError(
                        f"True-packing visual injection mismatch in sample {sample_index}: "
                        f"image placeholder run={actual_run}, next grid expansion={expected}."
                    )
                image_index += 1
            elif modality_token == int(config.video_token_id):
                if video_index >= len(video_runs) or actual_run != video_runs[video_index]:
                    expected = video_runs[video_index] if video_index < len(video_runs) else None
                    raise ValueError(
                        f"True-packing visual injection mismatch in sample {sample_index}: "
                        f"video placeholder run={actual_run}, next frame-grid expansion={expected}."
                    )
                video_index += 1
            else:
                raise ValueError(
                    f"sample {sample_index} vision-start token is followed by unsupported token {modality_token}"
                )
    if image_index != len(image_runs) or video_index != len(video_runs):
        raise ValueError(
            "True-packing visual injection mismatch: every image/video grid row must be consumed in "
            f"logical sample order; images consumed/available={image_index}/{len(image_runs)}, "
            f"video frames consumed/available={video_index}/{len(video_runs)}."
        )

    segments = torch.split(real_input_ids, seq_lens)  # tuple[[L_i]]
    max_length = max(seq_lens)
    logical_input_ids = real_input_ids.new_zeros((len(seq_lens), max_length))  # [K,max_L]
    logical_attention_mask = torch.zeros(
        (len(seq_lens), max_length), dtype=torch.long, device=input_ids.device
    )  # [K,max_L]
    for sample_index, segment in enumerate(segments):
        length = seq_lens[sample_index]
        logical_input_ids[sample_index, :length] = segment
        logical_attention_mask[sample_index, :length] = 1

    # ``get_rope_index`` only reads ``model.config``. Grids stay in the exact sample/media
    # order produced by the collate; packing never sorts or retimes media.
    model_shim = SimpleNamespace(config=config)
    logical_position_ids, _ = get_rope_index(
        model_shim,
        logical_input_ids,
        image_grid_thw,
        video_grid_thw,
        attention_mask=logical_attention_mask,
    )  # [3,K,max_L]
    position_parts = [
        logical_position_ids[:, sample_index, :length] for sample_index, length in enumerate(seq_lens)
    ]  # list[[3,L_i]]
    packed_position_ids = torch.cat(position_parts, dim=1)  # [3,S]
    if seq_padded > seq_total:
        pad_position_ids = packed_position_ids.new_zeros((3, seq_padded - seq_total))  # [3,S_pad-S]
        packed_position_ids = torch.cat([packed_position_ids, pad_position_ids], dim=1)  # [3,S_pad]
    return packed_position_ids.unsqueeze(1)  # [3,1,S_pad]


def prepare_true_packing_batch_on_cpu(data: dict[str, Any], config: Any) -> None:
    """Validate and add packed positions while every batch tensor is still on CPU.

    ``ImaginaireTrainer`` recursively moves the returned batch to CUDA after dataloading.
    Preparing here keeps Qwen3-VL's Python-based M-RoPE scan off the GPU hot path and turns
    ``position_ids`` into one ordinary asynchronous H2D tensor copy.
    """
    if data.get("true_packing") is not True:
        raise ValueError("prepare_true_packing_batch_on_cpu requires a true-packed batch")
    if config.model_type != "qwen3_vl":
        raise NotImplementedError(
            f"True packing is validated only for dense qwen3_vl; got model_type={config.model_type!r}"
        )

    input_ids = data.get("input_ids")
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("true-packed input_ids must be a tensor")
    if input_ids.device.type != "cpu":
        raise ValueError(
            f"true-packed batches must be prepared before the trainer H2D copy; got input_ids on {input_ids.device}"
        )
    packed_cu_seq_lens = data.get("packed_cu_seq_lens")
    if not isinstance(packed_cu_seq_lens, torch.Tensor) or packed_cu_seq_lens.dtype != torch.int32:
        raise TypeError("packed_cu_seq_lens must be one int32 tensor")
    if packed_cu_seq_lens.device.type != "cpu":
        raise ValueError("packed_cu_seq_lens must still be on CPU during true-packing preparation")

    seq_lens = data.get("seq_lens")
    if not isinstance(seq_lens, list) or not all(isinstance(length, int) for length in seq_lens):
        raise TypeError("seq_lens must be a list of Python ints")

    assert_packing_temporal_inputs_supported(data)
    video_grid_thw = data.get("video_grid_thw")
    if video_grid_thw is not None and not isinstance(video_grid_thw, torch.Tensor):
        raise TypeError("video_grid_thw must be a tensor")
    if video_grid_thw is not None and bool((video_grid_thw[:, 0] != 1).any()):
        raise NotImplementedError(
            "True-packed multi-frame video is not enabled until padded-vs-packed GPU logit and "
            "gradient parity are certified; use one-frame video grids or padded batching."
        )

    data["position_ids"] = build_packed_position_ids(
        config,
        input_ids=input_ids,
        seq_lens=seq_lens,
        image_grid_thw=data.get("image_grid_thw"),
        video_grid_thw=video_grid_thw,
    )  # [3,1,S_pad]
    data[TRUE_PACKING_CPU_PREPARED_KEY] = True
