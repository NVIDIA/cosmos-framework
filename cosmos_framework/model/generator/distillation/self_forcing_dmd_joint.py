# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Multiview RGB or RGB/LiDAR SF-DMD with detached causal history and per-chunk replay."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.distributed as dist

from cosmos_framework.checkpoint.strict_checkpoint import load_regular_checkpoint
from cosmos_framework.data.generator.sequence_packing import PackedSequence, SequencePlan
from cosmos_framework.model.generator.distillation.self_forcing_dmd import SelfForcingDMDModel
from cosmos_framework.model.generator.mot.causal_cosmos3_vfm_network import build_interactive_multiview_mask_items
from cosmos_framework.model.generator.mot.context_parallel_utils import context_parallel_broadcast_tensor_list
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean, GenerationDataNoised
from cosmos_framework.model.generator.utils.kv_cache import JointChunkMemory, TeacherForcingMemoryState
from cosmos_framework.model.generator.utils.multiview_ar import (
    JointARCache,
    JointARChunk,
    _build_persistent_branch,
    _build_replay_branch,
    _commit_persistent_joint_chunk,
    _conditioned_target,
    _current_chunk_data,
    _prefix_data,
    _validate_persistent_chunks,
    joint_ar_chunks,
    multiview_conditioned_prefix_length,
)
from cosmos_framework.model.generator.utils.multiview_negative_prompt import multiview_negative_captions
from cosmos_framework.utils.generator.multiview import slice_camera_major_frames


@dataclass(frozen=True)
class JointReplayFrame:
    """Detached last-denoise inputs on the common sensor clock."""

    chunk: JointARChunk
    prefix_indexes: dict[str, list[int]]
    output_indexes: dict[str, list[int]]
    noisy: dict[str, torch.Tensor]  # modality -> [1,C,T_chunk,H,W]
    sigma: float


@dataclass
class JointRollout:
    data: GenerationDataClean
    packed: PackedSequence
    frames: list[JointReplayFrame]
    num_views: int
    cache: JointARCache | None = None
    persistent_kv_used: bool = False
    persistent_kv_fallback: bool = False


def active_mean_square(
    prediction: torch.Tensor,  # [1,C,T,H,W]
    target: torch.Tensor,  # [1,C,T,H,W]
    mask: torch.Tensor,  # [T,1,1]
) -> torch.Tensor:  # []
    """Average only generated elements, retaining a graph edge when none exist."""
    expanded = mask.to(device=prediction.device, dtype=torch.float32).expand_as(prediction)  # [1,C,T,H,W]
    error = (prediction.float() - target.float()).square()  # [1,C,T,H,W]
    return (error * expanded).sum() / expanded.sum().clamp(min=1)  # []


class JointSelfForcingDMDModel(SelfForcingDMDModel):
    """Train a causal multiview RGB or RGB/LiDAR student against bidirectional score networks.

    Student backward reuses detached clean target K/V and replays the full noisy prefix. This permits
    controls to read generated past sensor values without retaining all chunks'
    activation graphs or treating LiDAR sweeps as camera frames. Student and critic
    no-grad rollouts reuse persistent clean K/V for supported joint layouts. Teacher and fake
    score instead see the complete noisy clip through the original bidirectional
    teacher architecture, without a clean teacher-forcing cache.
    """

    def _joint_modalities(self, data: GenerationDataClean) -> tuple[str, ...]:
        """Use each sample's sensor presence, including whole-stream LiDAR dropout."""
        return tuple(
            modality
            for modality in self.config.joint_sensor_modalities
            if getattr(data, f"x0_tokens_{modality}") is not None
        )

    @staticmethod
    def _sync_transfer_conditioning_plans(
        sequence_plans: list[SequencePlan], gen_data_clean: GenerationDataClean
    ) -> list[SequencePlan]:
        """Broadcast an aligned prefix pair even when rank zero dropped LiDAR."""
        vision = gen_data_clean.x0_tokens_vision
        lidar = gen_data_clean.x0_tokens_lidar
        if gen_data_clean.batch_size != 1 or len(sequence_plans) != 1 or vision is None or len(vision) != 2:
            raise ValueError("Joint self-forcing requires one aligned [control, target] camera pair per rank.")
        plan = sequence_plans[0]
        has_lidar = lidar is not None
        if plan.has_lidar != has_lidar or (
            lidar is not None and (len(lidar) != 2 or gen_data_clean.num_lidar_items_per_sample != [2])
        ):
            raise ValueError("Joint self-forcing LiDAR presence must match its plan and control/target pair.")
        counts = []
        for modality in ("vision", "lidar"):
            indexes = getattr(plan, f"condition_frame_indexes_{modality}")
            if indexes != list(range(len(indexes))):
                raise ValueError(f"Joint self-forcing requires contiguous {modality} prefix indexes; got {indexes}.")
            counts.append(len(indexes))
        if dist.is_available() and dist.is_initialized():
            # Every rank participates, including camera-only replicas. Select a
            # joint owner when one exists so the formatter-derived sensor pair
            # survives dropout; an all-RGB batch uses world rank zero.
            owner = torch.tensor(
                [dist.get_rank() if has_lidar else dist.get_world_size()], dtype=torch.long, device=vision[1].device
            )  # [1]
            dist.all_reduce(owner, op=dist.ReduceOp.MIN)  # [1]
            source = int(owner.item())
            if source == dist.get_world_size():
                source = 0
            prefixes = torch.tensor(counts, dtype=torch.long, device=vision[1].device)  # [2]
            dist.broadcast(prefixes, src=source)  # [2]
            counts = prefixes.tolist()
        views = gen_data_clean.num_views_per_vision_item
        if views is None or len(views) != 2 or views[1] < 1:
            raise ValueError("Joint self-forcing requires per-item camera counts.")
        target = vision[1]  # [1,C,V*T,H,W] or [C,V*T,H,W]
        vision_frames = (target.shape[2] if target.ndim == 5 else target.shape[1]) // views[1]
        if not 0 <= counts[0] < vision_frames:
            raise ValueError("Synchronized camera conditioning must leave at least one generated frame per view.")
        if lidar is not None:
            target = lidar[1]  # [1,C,T,H,W] or [C,T,H,W]
            lidar_frames = target.shape[2] if target.ndim == 5 else target.shape[1]
            if not 0 <= counts[1] < lidar_frames:
                raise ValueError("Synchronized LiDAR conditioning must leave at least one generated sweep.")
        return [
            replace(
                plan,
                condition_frame_indexes_vision=list(range(counts[0])),
                condition_frame_indexes_lidar=list(range(counts[1])) if has_lidar else [],
            )
        ]

    def build_net(
        self,
        dtype: torch.dtype,
        *,
        mp_policy: Any | None = None,
        lora_enabled: bool | None = None,
    ) -> torch.nn.Module:
        """Keep student replay geometry out of the bidirectional scorer builds."""
        if self.config.video_temporal_causal:
            return super().build_net(dtype, mp_policy=mp_policy, lora_enabled=lora_enabled)
        return OmniMoTModel.build_net(self, dtype, mp_policy=mp_policy, lora_enabled=lora_enabled)

    def _load_checkpoint_to_net(
        self,
        net: torch.nn.Module,
        ckpt_path: str,
        prefix: str = "net",
        credential_path: str | None = None,
    ) -> None:
        """Use the regular networks from the explicitly selected joint lineage."""
        if self.config.strict_regular_checkpoint_loading:
            load_regular_checkpoint(net, ckpt_path, prefix=prefix, credential_path=credential_path)
            return
        super()._load_checkpoint_to_net(net, ckpt_path, prefix=prefix, credential_path=credential_path)

    def _make_teacher_negative_captions(self, batch_size: int, data_batch: dict[str, Any]) -> list[str]:
        """Opt into camera-local metadata without changing other recipes' captions."""
        if self.config.teacher_negative_prompt_keep_metadata and self.config.teacher_negative_prompt_per_view_metadata:
            return multiview_negative_captions(
                data_batch, batch_size, self.config.teacher_negative_prompt, video_key=self.input_video_key
            )
        return super()._make_teacher_negative_captions(batch_size, data_batch)

    def _joint_layout(
        self, data: GenerationDataClean, plans: list[SequencePlan], text: list[list[int]]
    ) -> tuple[PackedSequence, int, list[JointARChunk]]:
        modalities = self._joint_modalities(data)
        has_lidar = data.x0_tokens_lidar is not None
        if data.batch_size != 1 or len(plans) != 1 or not plans[0].has_vision or plans[0].has_lidar != has_lidar:
            raise ValueError("Joint SF-DMD requires one sample with the configured sensors per context-parallel group.")
        if "vision" not in modalities or (has_lidar and "lidar" not in modalities):
            raise ValueError("Joint SF-DMD sample sensors must be supported by the model.")
        if plans[0].has_action or plans[0].has_sound or self.config.action_gen or self.config.sound_gen:
            raise ValueError("Joint SF-DMD supports only RGB and LiDAR modalities.")
        if any(getattr(data, f"num_{modality}_items_per_sample") != [2] for modality in modalities):
            raise ValueError("Joint SF-DMD requires a control and target item for each sensor.")
        if not self._uses_multiview_replay_kv() or self.config.compile.enabled:
            raise ValueError("Joint SF-DMD requires eager multiview replay.")
        if self._get_teacher_forcing_replay_policy().control_visibility != "causal":
            raise ValueError("Joint SF-DMD prefix replay requires causal controls.")
        if self.parallel_dims is not None and self.parallel_dims.cfgp_enabled:
            raise ValueError("Joint SF-DMD uses serial guidance; set cfg_parallel_shard_degree=1.")
        if self.config.backward_grad_steps != 1 or not self.config.sf_replayed_training:
            raise ValueError("Joint SF-DMD requires last-step gradient replay.")
        if self.config.fixed_step_sampler_config.sample_type not in ("sde", "ode"):
            raise ValueError("Joint SF-DMD supports only SDE or ODE rollout updates.")
        views = data.num_views_per_vision_item
        if views is None or len(views) != 2 or views[0] < 1 or views[0] != views[1]:
            raise ValueError("Joint SF-DMD requires matching camera-major view metadata.")
        for modality in modalities:
            values = getattr(data, f"x0_tokens_{modality}")
            if values is None or len(values) != 2 or values[0].ndim != 5 or values[0].shape[0] != 1:
                raise ValueError(f"Expected two [1,C,T,H,W] {modality} items.")
            if values[0].shape != values[1].shape:
                raise ValueError(f"Control and target {modality} shapes must match.")
        pack = self._pack_input_sequence(plans, text, data, torch.zeros((1, 1)))  # timestep: [1,1]
        self._validate_teacher_forcing_pack(pack)
        sensors = build_interactive_multiview_mask_items(pack)
        if len(sensors) != 1 or len(sensors[0]) != 2 * len(modalities):
            raise ValueError("Expected one control and one target for every configured sensor.")
        for index in range(0, len(sensors[0]), 2):
            if not sensors[0][index].is_control or sensors[0][index + 1].is_control:
                raise ValueError("Joint SF-DMD sensor ownership differs from the training layout.")
        rgb = sensors[0][1]
        assert data.x0_tokens_vision is not None
        if data.x0_tokens_vision[1].shape[2] % views[0]:
            raise ValueError("Camera-major time must be divisible by the number of views.")
        if has_lidar:
            assert data.x0_tokens_lidar is not None
            chunks = joint_ar_chunks(
                vision_frames=data.x0_tokens_vision[1].shape[2] // views[0],
                lidar_frames=data.x0_tokens_lidar[1].shape[2],
                vision_seconds_per_frame=rgb.seconds_per_frame,
                lidar_seconds_per_frame=sensors[0][3].seconds_per_frame,
                frames_per_chunk=self.config.teacher_forcing_frames_per_chunk,
            )
        else:
            # The RGB-only arm shares the joint replay engine, including controls
            # reading generated history. It does not invent or encode LiDAR.
            count = data.x0_tokens_vision[1].shape[2] // views[0]
            width = self.config.teacher_forcing_frames_per_chunk
            if width < 1:
                raise ValueError("RGB replay requires a positive chunk size.")
            chunks = [JointARChunk(0, (0,), (), 1, 0)]
            chunks.extend(
                JointARChunk(
                    index + 1, tuple(range(start, min(start + width, count))), (), min(start + width, count), 0
                )
                for index, start in enumerate(range(1, count, width))
            )
        self._validate_joint_schedule(data, pack, views[0], chunks)
        return pack, views[0], chunks

    def _validate_joint_schedule(
        self, data: GenerationDataClean, pack: PackedSequence, views: int, chunks: list[JointARChunk]
    ) -> None:
        """Require equal replay schedules on both HSDP axes before any rollout."""
        if not self.config.sf_require_equal_rollout_lengths:
            raise ValueError("Joint SF-DMD requires equal rollout lengths; truncation is unsupported.")
        assert data.x0_tokens_vision is not None and pack.vision is not None
        vision_count = multiview_conditioned_prefix_length(pack.vision.condition_mask[1], views)
        lidar_count = 0 if pack.lidar is None else multiview_conditioned_prefix_length(pack.lidar.condition_mask[1], 1)
        replay_steps = [
            chunk.step
            for chunk in chunks
            if any(index >= vision_count for index in chunk.vision_frames)
            or (pack.lidar is not None and any(index >= lidar_count for index in chunk.lidar_frames))
        ]
        signature = [
            data.x0_tokens_vision[1].shape[2] // views,
            vision_count,
            len(chunks),
            len(replay_steps),
            replay_steps[0] if replay_steps else -1,
        ]
        if dist.is_available() and dist.is_initialized():
            # Replay closures synchronize gradients across shard and replicate
            # axes. Checking only CP/FSDP shards can miss unequal replica horizons.
            # Independent replicas may retain different camera rigs/resolutions;
            # their temporal schedule, rather than their token count, must match.
            # Missing LiDAR contributes no sensor bound, but must execute the
            # same replay closures. Present LiDAR replicas still need identical
            # sweep horizons/prefixes; dropout must not hide a real mismatch.
            missing = torch.iinfo(torch.long).max
            lidar_signature = (
                [missing, missing] if data.x0_tokens_lidar is None else [data.x0_tokens_lidar[1].shape[2], lidar_count]
            )
            bounds = torch.tensor(
                [
                    signature + lidar_signature,
                    [-value for value in signature]
                    + [missing if value == missing else -value for value in lidar_signature],
                ],
                dtype=torch.long,
                device=self.tensor_kwargs["device"],
            )  # [2,7]
            dist.all_reduce(bounds, op=dist.ReduceOp.MIN)
            lower, upper = bounds[0].tolist(), (-bounds[1]).tolist()
            if lower[:5] != upper[:5] or (lower[5] != missing and lower[5:] != upper[5:]):
                raise ValueError(
                    "Joint SF-DMD requires equal RGB/LiDAR horizons and conditioned prefixes across HSDP; "
                    f"schedule minima={lower}, maxima={upper}. Align batches before tokenization."
                )

    def _joint_prefix(self, rollout: JointRollout, frame: JointReplayFrame) -> GenerationDataClean:
        assert rollout.data.x0_tokens_vision is not None
        if rollout.data.x0_tokens_lidar is not None:
            assert rollout.data.x0_tokens_lidar is not None
            prefix = _prefix_data(
                rollout.data,
                vision_target=rollout.data.x0_tokens_vision[1],  # [1,Cv,V*Tv,Hv,Wv]
                lidar_target=rollout.data.x0_tokens_lidar[1],  # [1,Cl,Tl,Hl,Wl]
                num_views=rollout.num_views,
                chunk=frame.chunk,
            )
        else:
            if rollout.data.temporal_positions_vision is not None:
                raise ValueError("RGB replay requires training latent-index temporal positions.")
            prefix = replace(
                rollout.data,
                raw_state_vision=None,
                x0_tokens_vision=[
                    slice_camera_major_frames(value, rollout.num_views, 0, frame.chunk.vision_prefix_end)
                    for value in rollout.data.x0_tokens_vision
                ],  # list[[1,C,V*T_prefix,H,W]]
            )
        for modality in self._joint_modalities(prefix):
            values = getattr(prefix, f"x0_tokens_{modality}")
            target = values[1].clone()  # [1,C,T_prefix,H,W]
            # Replayed clean history must match rollout time, when the current
            # generated chunk did not exist. Conditioned positions are preserved.
            target[:, :, frame.prefix_indexes[modality]] = 0.0
            setattr(prefix, f"x0_tokens_{modality}", [values[0], target])
        return prefix

    def _joint_chunk_velocity(
        self,
        prefix: GenerationDataClean,
        pack: PackedSequence,
        memory: TeacherForcingMemoryState | JointChunkMemory,
        frame: JointReplayFrame,
        noisy: dict[str, torch.Tensor],  # modality -> [1,C,T_chunk,H,W]
        sigma: float,
    ) -> dict[str, torch.Tensor]:  # modality -> [1,C,T_chunk,H,W]
        inputs: dict[str, list[torch.Tensor]] = {}
        modalities = self._joint_modalities(prefix)
        for modality in modalities:
            values = getattr(prefix, f"x0_tokens_{modality}")
            target = values[1].clone()  # [1,C,T_prefix,H,W]
            target[:, :, frame.prefix_indexes[modality]] = noisy[modality].to(target)  # [1,C,T_chunk,H,W]
            inputs[modality] = [values[0], target]  # list[[1,C,T_prefix,H,W]]
        timestep = torch.full((1, 1), sigma * self.config.rectified_flow_inference_config.num_train_timesteps)  # [1,1]
        self._update_inference_pack_template(
            packed_sequence=pack,
            noise_x_vision=inputs["vision"],  # list[[1,Cv,T_prefix,Hv,Wv]]
            noise_x_lidar=inputs.get("lidar"),  # list[[1,Cl,T_prefix,Hl,Wl]] or None
            noise_x_radar=None,
            noise_x_action=None,
            noise_x_sound=None,
            timestep=timestep,  # [1,1]
        )
        output = self.denoise(net=self.net, data_batch_packed=pack, memory=memory)
        velocity = {
            modality: output[f"preds_{modality}"][1][:, :, frame.prefix_indexes[modality]].float()
            for modality in modalities
        }  # modality -> [1,C,T_chunk,H,W]
        if "lidar" in self.config.joint_sensor_modalities and prefix.x0_tokens_lidar is None:
            # The network probes absent sensor heads for FSDP. Keep their zero
            # gradient edges so dropout cannot skip a head's backward collective.
            velocity["vision"] = velocity["vision"] + 0.0 * self._sum_optional_preds(
                output, "preds_lidar", velocity["vision"]
            )  # [1,C,T_chunk,H,W]
        return velocity

    def _can_use_persistent_joint_kv(self, pack: PackedSequence, views: int, chunks: list[JointARChunk]) -> bool:
        """Choose one rollout backend across both HSDP axes before any forward."""
        eligible = (
            self._get_teacher_forcing_kv_implementation() == "multiview_maskless_kv"
            and self.config.kv_cache_inference_size is None
            and self.config.attention_sink_size == 0
            and self.config.rolling_kv_cache_chunks is None
        )
        if eligible:
            assert pack.vision is not None
            vision_count = multiview_conditioned_prefix_length(pack.vision.condition_mask[1], views)
            lidar_count = (
                multiview_conditioned_prefix_length(pack.lidar.condition_mask[1], 1) if pack.lidar is not None else 0
            )
            try:
                _validate_persistent_chunks(chunks, vision_count, lidar_count)
            except ValueError:
                # Later conditioned queries can read noisy replay history; copying
                # clean K/V would change that behavior. Missing sensor chunks also
                # require replay until the persistent backend supports their layout.
                eligible = False
        if dist.is_available() and dist.is_initialized():
            # Sensor layouts can differ across replicas. All FSDP participants must
            # agree before persistent clean commits change the forward schedule.
            supported = torch.tensor(int(eligible), dtype=torch.int32, device=self.tensor_kwargs["device"])  # []
            dist.all_reduce(supported, op=dist.ReduceOp.MIN)
            eligible = bool(supported.item())
        return eligible

    def _joint_rollout(
        self,
        data: GenerationDataClean,
        plans: list[SequencePlan],
        text: list[list[int]],
        effective_iteration: int | None = None,
        *,
        use_persistent_kv: bool = False,
        save_replay_frames: bool = False,
    ) -> JointRollout:
        """Use the selected denoising exit, discarding unconditioned ground truth.

        Persistent student rollouts save detached last-denoise inputs for subsequent
        full-prefix gradient replay. When enabled, its clean target K/V also supplies
        backward replay without recomputing the clean prefix.
        """
        packed, views, chunks = self._joint_layout(data, plans, text)
        cache = None
        if use_persistent_kv:
            if self._can_use_persistent_joint_kv(packed, views, chunks):
                cache = JointARCache(packed, self.net.num_hidden_layers)
        modalities = self._joint_modalities(data)
        generated = replace(data, raw_state_vision=None, raw_state_lidar=None)
        for modality in modalities:
            original = getattr(data, f"x0_tokens_{modality}")
            target = _conditioned_target(original[1], getattr(packed, modality).condition_mask[1]).to(
                **self.tensor_kwargs_fp32
            )
            # [1,C,T,H,W]
            control = original[0].detach().to(device=self.tensor_kwargs["device"])  # [1,C,T,H,W]
            setattr(generated, f"x0_tokens_{modality}", [control, target])
        rollout = JointRollout(
            generated,
            packed,
            [],
            views,
            persistent_kv_used=cache is not None,
            persistent_kv_fallback=use_persistent_kv and cache is None,
        )
        assert generated.x0_tokens_vision is not None
        total_frames = generated.x0_tokens_vision[1].shape[2] // views
        step_counts = self._self_forcing_step_counts_for_iteration(total_frames, effective_iteration)
        generated_chunks = 0
        for chunk in chunks:
            # Match the RGB rollout's per-frame schedule at the chunk start.
            # A final LiDAR-only chunk uses the last RGB frame's schedule.
            first_frame = chunk.vision_frames[0] if chunk.vision_frames else total_frames - 1
            schedule = self._self_forcing_timestep_schedule_for_frame(first_frame, total_frames, step_counts)
            if schedule[0] != 1.0 or schedule[-1] != 0.0 or any(a <= b for a, b in zip(schedule, schedule[1:])):
                raise ValueError("Joint SF-DMD requires a strictly decreasing schedule from one to zero.")
            rgb_indices = [view * total_frames + frame for view in range(views) for frame in chunk.vision_frames]
            rgb_prefix = [
                view * chunk.vision_prefix_end + frame for view in range(views) for frame in chunk.vision_frames
            ]
            lidar_prefix = list(chunk.lidar_frames)
            if cache is not None:
                rgb_prefix = list(range(views * len(chunk.vision_frames)))
                lidar_prefix = list(range(len(chunk.lidar_frames)))
            output_indexes: dict[str, list[int]] = {}
            prefix_indexes: dict[str, list[int]] = {}
            for modality, final, local in (
                ("vision", rgb_indices, rgb_prefix),
                ("lidar", list(chunk.lidar_frames), lidar_prefix),
            ):
                if modality not in modalities:
                    continue
                mask = getattr(packed, modality).condition_mask[1].flatten()  # [T]
                pairs = [(i, j) for i, j in zip(final, local, strict=True) if not bool(mask[i])]
                output_indexes[modality] = [i for i, _ in pairs]
                prefix_indexes[modality] = [j for _, j in pairs]
            if not any(output_indexes.values()):
                if cache is not None and chunk is not chunks[-1]:
                    current = _current_chunk_data(
                        generated,
                        generated.x0_tokens_vision[1],
                        generated.x0_tokens_lidar[1] if generated.x0_tokens_lidar is not None else None,
                        views,
                        chunk,
                    )
                    # A fully conditioned first chunk still supplies clean history
                    # to every generated chunk that follows it.
                    _commit_persistent_joint_chunk(self, plans, current, [(text, cache)], chunk)
                continue
            noisy = {
                modality: torch.randn_like(
                    getattr(generated, f"x0_tokens_{modality}")[1][:, :, output_indexes[modality]],
                    **self.tensor_kwargs_fp32,
                )
                for modality in modalities
            }  # modality -> [1,C,T_chunk,H,W]
            self._broadcast_joint_noise(noisy)
            frame = JointReplayFrame(chunk, prefix_indexes, output_indexes, noisy, schedule[-2])
            if cache is None:
                prefix = self._joint_prefix(rollout, frame)
                pack, memory = _build_replay_branch(self, plans, prefix, text)
            else:
                prefix = _current_chunk_data(
                    generated,
                    generated.x0_tokens_vision[1],
                    generated.x0_tokens_lidar[1] if generated.x0_tokens_lidar is not None else None,
                    views,
                    chunk,
                )
                pack, memory = _build_persistent_branch(self, plans, prefix, text, cache, chunk)
            for index, (sigma, next_sigma) in enumerate(zip(schedule, schedule[1:])):
                if (cache is None or save_replay_frames) and index == len(schedule) - 2:
                    frame = replace(frame, noisy={key: value.detach().clone() for key, value in noisy.items()})
                    # noisy: modality -> [1,C,T_chunk,H,W]
                velocity = self._joint_chunk_velocity(prefix, pack, memory, frame, noisy, sigma)
                endpoints = {key: noisy[key] - sigma * velocity[key] for key in modalities}
                # endpoints: modality -> [1,C,T_chunk,H,W]
                if next_sigma:
                    if self.config.fixed_step_sampler_config.sample_type == "ode":
                        noisy = {key: noisy[key] + (next_sigma - sigma) * velocity[key] for key in modalities}
                        # noisy: modality -> [1,C,T_chunk,H,W]
                    else:
                        noise = {key: torch.randn_like(value) for key, value in endpoints.items()}
                        # noise: modality -> [1,C,T_chunk,H,W]
                        self._broadcast_joint_noise(noise)
                        noisy = {key: (1 - next_sigma) * endpoints[key] + next_sigma * noise[key] for key in modalities}
                        # noisy: modality -> [1,C,T_chunk,H,W]
            for modality in modalities:
                if not torch.isfinite(endpoints[modality]).all():
                    raise FloatingPointError(f"Nonfinite {modality} rollout at chunk {chunk.step}.")
                target = getattr(generated, f"x0_tokens_{modality}")[1]  # [1,C,T,H,W]
                target[:, :, output_indexes[modality]] = endpoints[modality]  # [1,C,T_chunk,H,W]
            generated_chunks += 1
            if cache is None or save_replay_frames:
                if cache is not None:
                    # Denoising uses chunk-local indexes, but backward rebuilds the
                    # full prefix with each camera's history contiguous in time.
                    frame = replace(
                        frame,
                        prefix_indexes={
                            modality: [i // total_frames * chunk.vision_prefix_end + i % total_frames for i in indexes]
                            if modality == "vision"
                            else indexes
                            for modality, indexes in output_indexes.items()
                        },
                    )
                rollout.frames.append(frame)
            if cache is not None and chunk is not chunks[-1]:
                current = _current_chunk_data(
                    generated,
                    generated.x0_tokens_vision[1],
                    generated.x0_tokens_lidar[1] if generated.x0_tokens_lidar is not None else None,
                    views,
                    chunk,
                )
                # Commit a separate clean forward, never the final noisy denoise
                # K/V. Only student rollouts save gradient replay frames and last-step copies.
                _commit_persistent_joint_chunk(self, plans, current, [(text, cache)], chunk)
            del pack, memory, prefix
        if generated_chunks == 0:
            raise ValueError("Joint SF-DMD requires at least one generated sensor chunk.")
        if save_replay_frames and self.config.sf_student_backward_use_persistent_kv:
            rollout.cache = cache
        return rollout

    def _broadcast_joint_noise(self, noise: dict[str, torch.Tensor]) -> None:  # each [1,C,T_chunk,H,W]
        context_parallel_broadcast_tensor_list([value for value in noise.values() if value.numel()], self.parallel_dims)

    def _joint_scoring_inputs(
        self, rollout: JointRollout, resolutions: list[str]
    ) -> tuple[GenerationDataNoised, torch.Tensor, torch.Tensor]:  # data, [1,1], [1,1]
        timestep, sigma = self._get_train_noise_level_vision(
            batch_size=1,
            is_image_batch=False,
            num_vision_latent_frames=[1],
            resolutions=resolutions,
            num_tokens=self._num_vision_tokens_per_sample(rollout.data),
        )  # [1,1], [1,1]
        if sigma.shape != (1, 1):
            raise ValueError("Joint scoring requires one diffusion time shared by both sensors.")
        context_parallel_broadcast_tensor_list([timestep, sigma], self.parallel_dims)
        noised = self._add_noise_to_input(
            rollout.data,
            rollout.packed,
            sigma.expand(2, 1),  # [2,1]
            sigmas_lidar=sigma.expand(2, 1) if rollout.data.x0_tokens_lidar is not None else None,  # [2,1] or None
        )
        return noised, timestep, sigma

    def _joint_score(
        self,
        rollout: JointRollout,
        noised: GenerationDataNoised,
        timestep: torch.Tensor,  # [1,1]
        plans: list[SequencePlan],
        text: list[list[int]],
        net_type: str,
    ) -> dict[str, Any]:
        if net_type not in ("teacher", "fake_score"):
            raise ValueError("Full-clip joint scoring requires a teacher or fake-score network.")
        # Preserve all joint/caption metadata while packing the actual noisy
        # inputs. Both CFG branches use bidirectional full-clip scoring. Clean
        # student replay caches would expose unnoised generated targets here.
        video_temporal_causal = self.config.video_temporal_causal
        self.config.video_temporal_causal = False
        try:
            pack = self._pack_input_sequence(plans, text, rollout.data, timestep.cpu())
        finally:
            self.config.video_temporal_causal = video_temporal_causal
        # Critic inputs are detached generated samples, but its complete noisy
        # forward retains parameter gradients. Teacher and DMD target scoring
        # run under their caller's no-grad context; only the student uses replay.
        self._replace_clean_with_noised(pack, noised)
        pack.to_cuda()
        self._cast_generated_tokens_to_precision(pack)
        return self.denoise(
            net=self.denoiser_nets[net_type],
            data_batch_packed=pack,
            memory=None,
            video_temporal_causal=False,
        )

    def _joint_sensor_weight(self, modality: str) -> float:
        rf = self.config.rectified_flow_training_config
        return 1.0 if modality == "vision" or rf.lidar_loss_scale is None else rf.lidar_loss_scale / rf.loss_scale

    def _joint_critic(
        self, rollout: JointRollout, plans: list[SequencePlan], text: list[list[int]], resolutions: list[str]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        with torch.no_grad():
            noised, timestep, sigma = self._joint_scoring_inputs(rollout, resolutions)
        predictions = self._joint_score(rollout, noised, timestep, plans, text, "fake_score")
        metrics: dict[str, torch.Tensor] = {"sigma": sigma.detach()}  # [1,1]
        # Record the selected backend after all ranks agree. An explicit opt-out
        # is replay, but is not an unsupported-layout fallback.
        metrics["sf_critic_persistent_kv_used"] = torch.tensor(
            float(rollout.persistent_kv_used), **self.tensor_kwargs_fp32
        )  # []
        metrics["sf_critic_persistent_kv_fallback"] = torch.tensor(
            float(rollout.persistent_kv_fallback), **self.tensor_kwargs_fp32
        )  # []
        total = torch.zeros((), **self.tensor_kwargs_fp32)  # []
        for modality in self._joint_modalities(rollout.data):
            mask = 1.0 - getattr(rollout.packed, modality).condition_mask[1]  # [T,1,1]
            loss = active_mean_square(
                predictions[f"preds_{modality}"][1],
                getattr(noised, f"vt_target_{modality}")[1],
                mask,
            )  # []
            total = total + loss * self._joint_sensor_weight(modality) * self.config.loss_scale_fake_score  # []
            metrics[f"flow_matching_loss_{modality}_per_instance"] = loss.detach().reshape(1)  # [1]
            metrics[f"joint_critic_loss_{modality}"] = loss.detach()  # []
        if "lidar" in self.config.joint_sensor_modalities and rollout.data.x0_tokens_lidar is None:
            total = total + 0.0 * self._sum_optional_preds(predictions, "preds_lidar", total)  # []
        metrics.update(loss_fake_score=total.detach(), total_critic_loss=total.detach(), dmd_loss=total.detach())
        metrics["dmd_loss_critic"] = total.detach()  # []
        return metrics, total

    def _joint_pseudo_targets(
        self,
        rollout: JointRollout,
        plans: list[SequencePlan],
        text: list[list[int]],
        negative: list[list[int]] | None,
        resolutions: list[str],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:  # modality -> [1,C,T,H,W], scoring sigma: [1,1]
        if self.config.teacher_guidance > 1 and negative is None:
            raise ValueError("Joint SF-DMD teacher CFG requires its CP owner's cached unconditional tokens.")
        noised, timestep, scoring_sigma = self._joint_scoring_inputs(rollout, resolutions)  # data, [1,1], [1,1]
        fake = self._joint_score(rollout, noised, timestep, plans, text, "fake_score")
        teacher = self._joint_score(rollout, noised, timestep, plans, text, "teacher")
        guidance = self.config.teacher_guidance
        unconditional = (
            self._joint_score(rollout, noised, timestep, plans, negative, "teacher")
            if guidance > 1 and negative is not None
            else None
        )
        targets: dict[str, torch.Tensor] = {}
        for modality in self._joint_modalities(rollout.data):
            velocity = teacher[f"preds_{modality}"][1]  # [1,C,T,H,W]
            if unconditional is not None:
                velocity = velocity + (guidance - 1) * (velocity - unconditional[f"preds_{modality}"][1])
                # [1,C,T,H,W]
            noisy = getattr(noised, f"xt_tokens_{modality}")[1].float()  # [1,C,T,H,W]
            sigma = getattr(noised, f"sigmas_{modality}")[1].to(noisy)  # [T,1,1]
            teacher_x0 = noisy - sigma * velocity.float()  # [1,C,T,H,W]
            fake_x0 = noisy - sigma * fake[f"preds_{modality}"][1].float()  # [1,C,T,H,W]
            generated = getattr(rollout.data, f"x0_tokens_{modality}")[1]  # [1,C,T,H,W]
            mask = 1.0 - getattr(rollout.packed, modality).condition_mask[1]  # [T,1,1]
            targets[modality] = self._make_vsd_pseudo_target([generated], [teacher_x0], [fake_x0], [mask])[0]
            # [1,C,T,H,W]
        return targets, scoring_sigma.detach()  # modality -> [1,C,T,H,W], [1,1]

    def _joint_replay_loss(
        self,
        rollout: JointRollout,
        frame: JointReplayFrame,
        plans: list[SequencePlan],
        text: list[list[int]],
        targets: dict[str, torch.Tensor],  # modality -> [1,C,T,H,W]
        *,
        scoring_sigma: torch.Tensor | None = None,  # [1,1]; only the first closure emits step metadata
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        with torch.no_grad():
            prefix = self._joint_prefix(rollout, frame)
            pack, memory = _build_replay_branch(self, plans, prefix, text, persistent_cache=rollout.cache)
        velocity = self._joint_chunk_velocity(prefix, pack, memory, frame, frame.noisy, frame.sigma)
        total = torch.zeros((), **self.tensor_kwargs_fp32)  # []
        metrics: dict[str, torch.Tensor] = {}
        for modality in self._joint_modalities(rollout.data):
            endpoint = frame.noisy[modality] - frame.sigma * velocity[modality]  # [1,C,T_chunk,H,W]
            target = targets[modality][:, :, frame.output_indexes[modality]]  # [1,C,T_chunk,H,W]
            mask = 1.0 - getattr(rollout.packed, modality).condition_mask[1]  # [T,1,1]
            # Sum chunk contributions using the complete generated sensor's
            # denominator; changing chunk size must not change loss normalization.
            denominator = mask.sum().to(endpoint) * endpoint.shape[1] * endpoint.shape[3] * endpoint.shape[4]  # []
            loss = 0.5 * (endpoint - target).square().sum() / denominator.clamp(min=1)  # []
            total = total + loss * self._joint_sensor_weight(modality) * self.config.loss_scale_sid  # []
            metrics[f"joint_generator_loss_{modality}"] = loss.detach()  # []
            metrics[f"flow_matching_loss_{modality}_per_instance"] = loss.detach().reshape(1)  # [1]
        metrics.update(vsd_loss=total.detach(), total_generator_loss=total.detach(), dmd_loss=total.detach())
        metrics["dmd_loss_generator"] = total.detach()  # []
        if scoring_sigma is not None:
            # The trainer sums repeated tensor fields across replay closures.
            # Log the teacher/fake scoring draw once, not the student SDE sigma.
            metrics["sigma"] = scoring_sigma.detach()  # [1,1]
            metrics["teacher_guidance"] = torch.tensor(self.config.teacher_guidance, **self.tensor_kwargs_fp32)  # []
        return metrics, total

    def training_step_closures(
        self, data_batch: dict[str, Any], iteration: int
    ) -> Iterator[tuple[str, Callable[[], tuple[dict[str, torch.Tensor], torch.Tensor]], bool]]:
        self._setup_grad_requirements(iteration)
        student_phase = self.get_phase(iteration) == "student"
        text, plans, data, resolutions, negative = self._prepare_self_forcing_inputs(data_batch, iteration)
        resolutions = resolutions or [self.config.resolution]
        with torch.no_grad():
            effective_iteration = (
                self.get_student_iteration(iteration) if student_phase else self.get_fake_score_iteration(iteration)
            )
            rollout = self._joint_rollout(
                data,
                plans,
                text,
                effective_iteration=effective_iteration,
                use_persistent_kv=(
                    self.config.sf_student_use_persistent_kv
                    if student_phase
                    else self.config.sf_critic_use_persistent_kv
                ),
                save_replay_frames=student_phase,
            )
        if not student_phase:
            yield "joint_sf_dmd_critic", lambda: self._joint_critic(rollout, plans, text, resolutions), True
            return
        # CP rotates the owner payload; use the unconditional tokens captured
        # with that sample by the native Stage 3 prompt path before rotation.
        with torch.no_grad():
            targets, scoring_sigma = self._joint_pseudo_targets(rollout, plans, text, negative, resolutions)
            # targets: modality -> [1,C,T,H,W]; scoring_sigma: [1,1]
        for index, frame in enumerate(rollout.frames):
            yield (
                f"joint_sf_dmd_chunk_{frame.chunk.step}",
                lambda frame=frame, index=index: self._joint_replay_loss(
                    rollout, frame, plans, text, targets, scoring_sigma=scoring_sigma if index == 0 else None
                ),
                index == len(rollout.frames) - 1,
            )

    def training_step(self, data_batch: dict[str, Any], iteration: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        raise RuntimeError("Joint SF-DMD requires the interactive trainer's training_step_closures interface.")
