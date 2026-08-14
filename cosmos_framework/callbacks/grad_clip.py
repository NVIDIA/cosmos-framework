# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import math
from collections import defaultdict

import torch
import wandb
from torch.distributed.tensor import DTensor

from cosmos_framework.utils import log
from cosmos_framework.utils.callback import Callback
from cosmos_framework.utils.generator.input_probe import maybe_dump_post_clip


def _fused_nan_to_num(grads: list[torch.Tensor]) -> None:
    """Replace NaN/Inf entries with 0.0 in every floating-point grad in-place.

    Runs eager, NOT ``@torch.compile``. Compiling this generates a GPU-only Triton
    ``nan_to_num`` kernel, which crashes whenever any grad in the list is a CPU tensor.
    Eager ``torch.nan_to_num`` handles CPU and CUDA grads alike; the fusion win
    from compiling a handful of per-tensor ops once per step is negligible next
    to that fragility.
    """
    grads = [g for g in grads if torch.is_floating_point(g)]
    for g in grads:
        torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0, out=g)


class _MagnitudeRecord:
    def __init__(self) -> None:
        self.state: torch.Tensor | None = None
        self.iter_count: int = 0

    def reset(self) -> None:
        self.state = None
        self.iter_count = 0

    def update(self, cur_state: torch.Tensor) -> None:
        if self.state is None:
            self.state = cur_state.detach().clone()
        else:
            self.state.add_(cur_state)
        self.iter_count += 1

    def get_stat(self) -> float:
        if self.state is not None and self.iter_count > 0:
            avg_state = (self.state / self.iter_count).item()
        else:
            avg_state = 0.0
        self.reset()
        return avg_state


def _mesh_key(param: torch.Tensor) -> str:
    """Mesh-group key for ``param``: its mesh-dim names joined, else ``"default"``.

    If one parameter belongs to multiple meshes, use a flattened mesh name
    by concatenating all the mesh-dim names together.  ``mesh_dim_names``
    is ``tuple[str, ...] | None`` on DeviceMesh — fall back to ``default``
    when names weren't assigned.  Plain (non-DTensor) params also map to
    ``"default"``.
    """
    if hasattr(param, "device_mesh"):
        names = param.device_mesh.mesh_dim_names
        return "-".join(names) if names else "default"
    return "default"


def _group_params_by_mesh(parameters: list[torch.Tensor]) -> dict[str, list[torch.Tensor]]:
    """Group the parameters by their device meshes.

    A parameter's mesh assignment is fixed once the model is built, so hot-path
    callers are expected to build this mapping once and reuse it rather than
    re-deriving mesh keys every optimizer step.
    """
    parameters_by_mesh: dict[str, list[torch.Tensor]] = defaultdict(list)
    for param in parameters:
        parameters_by_mesh[_mesh_key(param)].append(param)
    return parameters_by_mesh


@torch.no_grad()
def _total_norm_by_mesh(
    parameters_by_mesh: dict[str, list[torch.Tensor]],
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute one global grad norm across pre-grouped mesh buckets.

    Gradient norm clipping requires computing the gradient norm over the entire model.
    `torch.nn.utils.clip_grad_norm_` only computes gradient norm along DP/FSDP/TP dimensions.
    We need to manually reduce the gradient norm across PP stages.
    See https://github.com/pytorch/torchtitan/issues/596 for details.

    A scalar L2 norm is computed per mesh group, DTensor results are reduced to
    local scalars via ``.full_tensor()``, and the per-mesh scalars are combined
    into one global norm.

    Every parameter in every group must have a gradient; callers are expected to
    pre-filter, as :meth:`GradClip.on_before_optimizer_step` does when it builds
    the groups.

    ``parameters_by_mesh`` must be non-empty. Zero parameters have no meaningful
    norm, and the caller that can hit that (a no-grad / all-frozen step) wants to
    skip clipping entirely rather than rescale by some placeholder, so it is
    rejected here instead of being given one.

    Returns:
        ``(total_norm, per_mesh_norms)`` where ``total_norm`` is the global
        scalar norm, and ``per_mesh_norms`` maps each mesh-dim-names key
        (or ``"default"`` for plain params) to its per-mesh L2 norm.
    """
    if not parameters_by_mesh:
        raise ValueError(
            "_total_norm_by_mesh received no mesh groups; a step with no gradients has no global "
            "norm to compute, and the caller should skip clipping instead."
        )

    # Compute the norm for each mesh group
    per_mesh_norms: dict[str, torch.Tensor] = {}
    per_mesh_norm_list = []
    for mesh, params in parameters_by_mesh.items():
        grads = [p.grad for p in params]
        mesh_norm = torch.nn.utils.get_total_norm(grads, norm_type, error_if_nonfinite, foreach)

        # If mesh_norm is a DTensor, the placements must be
        # `torch.distributed._tensor.ops.math_ops._NormPartial`.
        # We can simply reduce the DTensor to get the total norm in this
        # tensor's process group and then convert it to a local tensor.
        if isinstance(mesh_norm, DTensor):
            # Will reach here if any non-PP parallelism is used.
            # If only using PP, mesh_norm will be a local tensor.

            # Reduce over every mesh dimension — including an FT replicate dim if
            # one exists — into a plain, rank-replicated tensor.
            mesh_norm = mesh_norm.full_tensor()

        # Both paths are supposed to land on one number per group: ``get_total_norm``
        # reduces to 0-dim and ``full_tensor()`` preserves that. A norm that arrives
        # un-reduced instead (some future DTensor path keeping a replica dimension)
        # would be logged as a meaningless per-mesh magnitude and, on the
        # single-group path below, become a non-scalar ``total_norm`` that rescales
        # grads elementwise. Raise rather than assert, so ``python -O`` cannot strip
        # the check; one ``numel()`` per mesh group costs nothing.
        if mesh_norm.numel() != 1:
            raise ValueError(
                f"grad norm for mesh {mesh!r} is not a scalar (shape {tuple(mesh_norm.shape)}); "
                "it was not fully reduced over that mesh."
            )

        # Expose the (rank-replicated) per-mesh scalar for diagnostic logging.
        per_mesh_norms[mesh] = mesh_norm
        per_mesh_norm_list.append(mesh_norm)

    # Compute the total norm among all meshes. ``get_total_norm`` always reduces
    # to 0-dim and ``full_tensor()`` preserves that, so stack (rather than cat)
    # turns the per-mesh scalars into the 1-D tensor to reduce over without
    # reshaping any of them first.
    if len(per_mesh_norm_list) > 1:
        per_mesh_norm_tensor = torch.stack(per_mesh_norm_list)
        if math.isinf(norm_type):
            total_norm = torch.max(per_mesh_norm_tensor)
        else:
            # In place on the freshly stacked copy, so per_mesh_norms is unaffected.
            per_mesh_norm_tensor **= norm_type
            total_norm = torch.sum(per_mesh_norm_tensor)
            total_norm **= 1.0 / norm_type
    else:
        # A single mesh group's scalar — already checked to be one above — IS the
        # global norm; skip the pow/sum/pow round trip. This is VFM's current
        # FSDP-only path.
        total_norm = per_mesh_norm_list[0]

    return total_norm, per_mesh_norms


@torch.no_grad()
def _clip_grads_with_global_norm(
    parameters_by_mesh: dict[str, list[torch.Tensor]],
    max_norm: float,
    total_norm: torch.Tensor,
    foreach: bool | None = None,
) -> None:
    """Do the clipping: scale every gradient in place by ``min(1, max_norm / total_norm)``.

    This is where gradients are actually modified. ``total_norm`` is the norm the
    caller already measured, not one recomputed here, and the clip is conditional
    in the usual way — torch clamps the coefficient at 1.0, so a step whose norm
    is already within ``max_norm`` comes out untouched.

    What matters is that every group is passed the SAME ``total_norm``, which is
    what makes this one global rescale of the whole model rather than one clip per
    mesh. Giving each group its own norm would be stock ``clip_grad_norm_``
    behaviour and would apply a different factor to, say, dense FSDP-sharded
    params and EP-sharded MoE experts, distorting their relative magnitudes.

    The loop exists because groups are the form the caller's parameters are
    already in, having been grouped to compute the per-mesh norms; the factor
    itself does not depend on the group. Flattening them into a single call is
    therefore not obviously equivalent — one fused ``foreach`` multiply spanning
    tensors from different meshes is not something to assume works.
    """
    for params in parameters_by_mesh.values():
        torch.nn.utils.clip_grads_with_norm_(params, max_norm, total_norm, foreach)


class GradClip(Callback):
    """Unified gradient-clipping callback for both VFM (diffusion) and VLM training.

    Params are grouped by their ``device_mesh`` (using mesh-dim-names as the
    key) once and cached; each step ``_total_norm_by_mesh`` computes a scalar
    L2 norm per mesh group (reducing any DTensor result via ``.full_tensor()``)
    and combines the per-mesh scalars into ONE global norm via
    ``sqrt(sum(per_mesh_norm**2))``, then ``_clip_grads_with_global_norm``
    scales the gradients by ``min(1, clip_norm / global_norm)``, passing every
    mesh group that same global scalar — a SINGLE GLOBAL rescale across every
    parameter.

    This is necessary for correctness when parameters live on multiple device
    meshes (e.g. dense FSDP-shard + EP-shard MoE experts): clipping each
    mesh independently with stock ``torch.nn.utils.clip_grad_norm_`` would
    assign a different rescale factor per mesh and distort the relative
    magnitudes of dense vs MoE updates. Under VFM's current FSDP-only
    setup the math reduces to a single mesh group and is identical to
    stock ``clip_grad_norm_``; this implementation is forward-correct
    once EP is enabled.

    For diagnostics, the callback ALSO records pre-clip per-mesh sub-norms
    alongside the actual global norm. When ``track_per_modality=True`` (VFM),
    samples are bucketed by image/video via ``model.is_image_batch(data_batch)``,
    producing wandb keys ``clip_grad_norm/{image|video}/{mesh_key}`` plus a
    ``.../global`` synthetic key carrying the actual rescale norm. When False
    (VLM), keys are ``clip_grad_norm/{mesh_key}`` plus ``clip_grad_norm/global``.

    Param-source semantics:
      * ``track_per_modality=True`` (VFM): caller passes the ``OmniMoTModel``;
        only ``model.net.parameters()`` is iterated, matching legacy VFM
        behavior (the optimizer is built from ``self.net``).
      * ``track_per_modality=False`` (VLM): caller passes a single
        ``ImaginaireModel`` or a list of model parts; ``parameters()`` is
        iterated and filtered by grad-presence.

    Args:
      clip_norm: max norm to clip to.
      force_finite: if True, NaN/Inf in any grad is zeroed in-place and the
        norm re-measured, on the steps where the computed norm comes back
        non-finite (which is exactly the steps where some grad entry is).
      track_per_modality: if True, route stats into image/video buckets via
        ``model.is_image_batch(data_batch)``. If False, accumulate into a
        single un-bucketed log group.
    """

    def __init__(
        self,
        clip_norm: float = 1.0,
        force_finite: bool = True,
        track_per_modality: bool = False,
    ):
        self.clip_norm = clip_norm
        self.force_finite = force_finite
        self.track_per_modality = track_per_modality

        # Outer key: modality bucket name. For VLM we use a single bucket "" so
        # wandb keys are short (`clip_grad_norm/{mesh}`); for VFM the bucket is
        # "image" or "video" (`clip_grad_norm/image/{mesh}`).
        # Inner key: mesh string, plus the synthetic "global" key for the
        # actual global norm used for the rescale.
        self._states: dict[str, dict[str, _MagnitudeRecord]] = defaultdict(lambda: defaultdict(_MagnitudeRecord))
        self._state_key: str = ""

        # The model parts the cache was built from, paired with the mesh grouping
        # of every one of their parameters; filled on the first optimizer step and
        # reused afterwards (see ``_mesh_groups``). Both the module tree walk and
        # the mesh-key derivation are pure-Python and O(#params); rebuilding them
        # every step put a few thousand Python frames on the critical path.
        self._cached_mesh_groups: tuple[list[torch.nn.Module], dict[str, list[torch.Tensor]]] | None = None

    def _mesh_groups(self, model_parts: list[torch.nn.Module]) -> dict[str, list[torch.Tensor]]:
        """Mesh-grouped parameters for ``model_parts``, walking the tree only once.

        ``Module.parameters()`` is a recursive generator over the whole module
        tree, so calling it per step is expensive for a large model — and most of
        that tree is frozen here anyway (``_build_params_with_metadata`` clears
        ``requires_grad`` for everything outside ``keys_to_select``). Parameter
        identity and mesh membership are both fixed once the model is built, so
        the walk and the grouping are cached; only the per-step
        ``grad is not None`` filter in ``on_before_optimizer_step`` stays dynamic,
        because which params receive a gradient can vary by batch (e.g. image vs
        video, audio vs no audio).

        One callback instance is expected to serve one model for the life of a
        run. Silently re-serving a cache built from some other model would clip
        the wrong parameter set, so the parts are held and identity-checked
        rather than compared by ``id()``, which is only unique among live
        objects. Note that this cannot catch parameters being replaced *inside*
        an unchanged module (progressive unfreezing, a LoRA merge, any
        reparametrization); such a caller has to invalidate the cache itself.
        """
        if self._cached_mesh_groups is None:
            params = [p for part in model_parts for p in part.parameters()]
            self._cached_mesh_groups = (model_parts, _group_params_by_mesh(params))

        cached_parts, mesh_groups = self._cached_mesh_groups
        if len(cached_parts) != len(model_parts) or any(
            cached is not part for cached, part in zip(cached_parts, model_parts, strict=True)
        ):
            raise RuntimeError(
                f"{type(self).__name__} was called with different model parts than the ones its parameter "
                "cache was built from; one callback instance must serve one model."
            )
        return mesh_groups

    def on_training_step_start(
        self,
        model: torch.nn.Module,
        data_batch: dict[str, torch.Tensor],
        iteration: int = 0,
    ) -> None:
        if not self.track_per_modality:
            return
        self._state_key = "image" if model.is_image_batch(data_batch) else "video"

    def on_before_optimizer_step(
        self,
        model: torch.nn.Module | list[torch.nn.Module],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int = 0,
    ) -> None:
        del optimizer, scheduler, grad_scaler

        # 1. Resolve which parameters to clip.
        if self.track_per_modality:
            # VFM: only clip `.net` params, matching legacy semantics + the
            # optimizer's actual param set.
            assert not isinstance(model, list), "track_per_modality=True expects a single OmniMoTModel, not a list"
            model_parts = [model.net]
        else:
            # VLM: frozen vlm/trainer/sft_trainer_cosmos_rl.py types
            # model_parts as list[ImaginaireModel] (FSDP + PP, no DDP).
            model_parts = model if isinstance(model, list) else [model]

        # 2. Collect params with grads, per mesh group. The grouping itself is
        #    cached; only the grad-presence filter is redone each step.
        grouped_params: dict[str, list[torch.Tensor]] = {}
        for mesh, params in self._mesh_groups(model_parts).items():
            with_grad = [p for p in params if p.grad is not None]
            if with_grad:
                grouped_params[mesh] = with_grad

        # 3. No-grad / all-frozen step → skip. There is nothing to clip, and
        #    ``_total_norm_by_mesh`` rejects an empty grouping rather than
        #    inventing a norm; deeper down, ``get_total_norm``'s own empty
        #    fallback uses torch.cuda.current_device() and would crash on CPU.
        if not grouped_params:
            return

        # 4. Compute per-mesh L2 norms (reducing DTensor results to local
        #    scalars) and combine them into the single global norm that will
        #    rescale every mesh group.
        global_norm, per_mesh_norms = _total_norm_by_mesh(
            grouped_params,
            error_if_nonfinite=False,
            foreach=True,
        )

        # 5. Optionally zero NaN/Inf in grads, but only when the norm says
        #    there is something to zero: any non-finite grad entry makes the
        #    global norm non-finite, so this is exact, and it keeps the common
        #    (finite) step from issuing one eager ``nan_to_num`` per gradient
        #    tensor. The finiteness test syncs on a scalar; that is affordable
        #    here because SkipNaNStep already all-reduces and ``.item()``s a
        #    flag immediately before this callback, so the device is drained
        #    either way. Drop SkipNaNStep and this becomes the step's first
        #    sync point.
        if self.force_finite and not bool(torch.isfinite(global_norm)):
            _fused_nan_to_num([p.grad for params in grouped_params.values() for p in params])
            # Re-measure so both the rescale and the logged diagnostics reflect
            # the sanitized grads, matching the pre-sanitize-always behavior.
            global_norm, per_mesh_norms = _total_norm_by_mesh(
                grouped_params,
                error_if_nonfinite=False,
                foreach=True,
            )

        # 6. Clip: scale every grad by min(1, clip_norm / global_norm), one factor
        #    for the whole model.
        _clip_grads_with_global_norm(grouped_params, self.clip_norm, global_norm, foreach=True)

        if not isinstance(model, list):
            raw_model = getattr(getattr(model, "model", None), "model", model)
            maybe_dump_post_clip(raw_model, global_norm, self.clip_norm, iteration, tag="i4")

        # 7. Record diagnostic stats: pre-clip per-mesh sub-norms plus the
        #    actual global rescale norm.
        cur_state = self._states[self._state_key]
        for mesh_str, mesh_norm in per_mesh_norms.items():
            cur_state[mesh_str].update(mesh_norm)
        cur_state["global"].update(global_norm)

        # 8. Log after the optimizer-step counter advances, in
        #    ``on_training_step_end``.

    def on_training_step_end(
        self,
        model: torch.nn.Module | list[torch.nn.Module],
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        """Log the stats accumulated by ``on_before_optimizer_step`` every logging_iter."""
        del model, data_batch, output_batch, loss

        if iteration % self.config.trainer.logging_iter != 0:
            return
        # The reset is intentionally *outside* the ``wandb.run`` gate:
        # ``_MagnitudeRecord.get_stat`` is the consumer that flushes the
        # windowed accumulator, so coupling it to wandb being live would let
        # stats accumulate unboundedly whenever wandb is disabled (smoke tests,
        # ``job.wandb_mode=disabled``, wandb init failure) and would back-fill
        # any later wandb enablement with the entire pre-enable history.
        log_dict: dict[str, float | int] = {"iteration": iteration}
        for modality, state in self._states.items():
            for mesh_str, record in state.items():
                avg = record.get_stat()
                if self.track_per_modality:
                    key = f"clip_grad_norm/{modality}/{mesh_str}"
                else:
                    key = f"clip_grad_norm/{mesh_str}"
                log_dict[key] = avg
                if mesh_str == "global":
                    log.info(f"{key}: {avg:.5f} (iteration {iteration})", rank0_only=False)
        if wandb.run:
            wandb.log(log_dict, step=iteration)
