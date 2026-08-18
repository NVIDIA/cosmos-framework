# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Turn a measured Cosmos3 VFM benchmark into a per-sample cost model.

The model is INCREMENTAL: it prices the marginal sample added to a step that is
already large, not a sample run on an otherwise idle GPU. A training step pays
two structurally different costs:

* Token-proportional work -- attention, projections, expert GEMMs, the VAE
  encode, the expert-parallel all-to-all. Doubling the tokens in a step roughly
  doubles this, so it is genuinely caused by the sample.
* Fixed per-step work -- streaming every parameter and optimizer-state byte
  through HBM, and the FSDP all-gather / reduce-scatter whose payload is set by
  the parameter count alone. A sparse MoE pays this in full whether the step
  carries one sample or a thousand: every expert's weights are read even if only
  a handful of tokens route to it, and the collectives move the same bytes
  either way. This is the memory- and network-bandwidth floor of a step.

Charging a sample the *average* cost of a small step bills it for bandwidth it
did not cause, and makes samples look more expensive the smaller the batch they
were measured in. So cost is quoted as the sample's marginal cost alone: its own
token terms plus its own per-sample constant, with no share of the per-step
constant, which the step pays regardless. That is exactly the incremental cost,
and it is exact rather than asymptotic -- adding a sample to a step holding N
tokens costs the same for any N, because the per-step constant cancels in the
difference and block-diagonal attention contributes no cross term. Nothing about
the batch needs to be known to price the next sample.

One thing qualifies that. A step can only hold
``max_num_tokens_after_packing`` tokens, so the sample that does not fit opens a
new step and pays a whole ``c0``: :meth:`TokenCostModel.new_step_gpu_sec` is that
upper bound, and the two together bracket the real incremental cost.

``c0`` is reported on its own and is never spread across a step to give a sample
an "absorbed" price. Doing that needs the size of the step the sample will join,
which is a property of the training config rather than of anything measured here,
and quoting it against an assumed size produces a number whose sensitivity to the
assumption is invisible once printed. The benchmark instead fills every measured
step to a token floor, so ``c0`` is fitted against steps that are not dominated by
it, and the two bracketing prices above are what gets quoted.

Cost is a function of TWO variables: the understanding (text) tokens ``x`` and
the generation (vision + sound) tokens ``y`` a sample contributes.
:func:`fit_token_cost_model` regresses measured step time on both as

    gpu_sec_per_step = c0 + sum_i [ A*x_i**2 + B*y_i**2 + C*x_i*y_i
                                    + D*x_i + E*y_i + F ]

which is not a chosen curve shape but the analytic form of the work. ``c0`` is
the per-step floor described above and ``F`` its per-sample counterpart: the two
are separable only when batch size varies, and confusing them misprices exactly
the case that matters, a lone sample versus one joining a step already underway.
``D`` and ``E`` are the work linear in each stream's tokens, and they differ:
understanding tokens run forward-only through a frozen tower, while generation
tokens pay forward and backward plus the VAE encode.

``A``, ``B`` and ``C`` are the three quadrants of the attention mask, kept
separate because the attention is asymmetric. The reasoner attends causally over
its own understanding stream, area ``x**2/2``, giving ``A``. Generation attends
over its own stream, area ``y**2``, giving ``B``, and over the text as well, area
``x*y``, giving ``C``. Collapsing all three into a single quadratic in ``x + y``
charges ``(x + y)**2``, which overcharges the understanding quadrant and cannot
express a caption costing less per token than a latent.

The quadratic terms are what make clip length expensive non-linearly: below the
crossover (:attr:`TokenCostModel.crossover_gen_tokens`, where ``E == B * y``) a
sample is priced essentially linearly, and above it attention dominates and
doubling the frames roughly quadruples the cost.

Two properties make the per-sample decomposition legitimate rather than
convenient. Both attention passes are bounded by ``sample_offsets``, so no token
attends across a sample boundary and a sample's own cost never depends on what
shares its step -- hence sums of per-sample areas rather than areas of the packed
total. And a token is a fixed 4096 pixels at every resolution (32x32 spatially
after VAE and patchify, times 4 temporally), so even the VAE encode is linear in
``y`` and 256p, 480p and 720p should fall on ONE surface. That last one is a
prediction the benchmark tests rather than an assumption:
:attr:`CostEstimate.residual_fraction` exposes any resolution that misses it.

Which coefficients a given calibration can support is decided by the data, not
assumed: :func:`fit_token_cost_model` rank-checks each candidate design and
reports what survived in ``fitted_terms``. A sweep that varies clip length at one
fixed caption length cannot identify ``a`` or ``b`` at all, and says so rather
than returning numbers for them.

Analytic FLOPs (``flops_per_sample``, over the repo's own
:mod:`cosmos_framework.tools.flops` estimator) are still computed, but only to report MFU.
They are deliberately NOT in the pricing path: quoted cost rests on measured
time alone.

Cost is reported as GPU-time -- GPU-seconds and GPU-hours per sample -- and never
as currency. A rate would be deployment-specific and would only obscure what was
actually measured, so converting is left to the caller.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from cosmos_framework.tools.flops import (
    OmniMoTModelDescriptor,
    compute_omni_mot_flops_per_batch,
    compute_wan_vae_encoder_flops,
    get_omni_mot_model_descriptor,
)
from cosmos_framework.utils import log
from cosmos_framework.utils.generator.cost_model.spec import SampleSpec

SECONDS_PER_HOUR = 3600.0

# Dense BF16 peak per GPU, no sparsity. Used only to express MFU; cost itself is
# derived from measured throughput and never from peak.
PEAK_TFLOPS_BY_DEVICE: dict[str, float] = {
    "GB200": 2250.0,
    "B200": 2250.0,
    "H200": 989.0,
    "H100": 989.0,
    "A100": 312.0,
}


@dataclass(frozen=True)
class FlopFlags:
    """Which FLOP terms the model counts, mirroring the live training config.

    Attributes:
        freeze_und: Understanding-tower gradients detached (no und backward).
        activation_checkpointing: Any non-``none`` AC mode -- adds one
            transformer-block forward recompute during backward.
        backwardpass_ratio: Backward FLOPs relative to forward.
        include_vae_encoder: Count the frozen Wan VAE encoder forward pass.
        vae_latent_channels: VAE latent channel count (``z_dim``).
    """

    freeze_und: bool = False
    activation_checkpointing: bool = True
    backwardpass_ratio: float = 2.0
    include_vae_encoder: bool = True
    vae_latent_channels: int = 48


def descriptor_from_hf_text_config(
    text_config: dict[str, Any],
    *,
    latent_patch_size: int = 2,
    latent_channel_size: int = 48,
    sound_dim: int = 64,
    frequency_embedding_size: int = 256,
    predict_text_tokens: bool = False,
    gen_moe_shared_expert: bool = False,
    gen_moe_shared_expert_intermediate_scale: int = 1,
    gen_moe_top_k: int | None = None,
) -> OmniMoTModelDescriptor:
    """Build a FLOP descriptor from a HuggingFace ``text_config`` block.

    Lets the cost model be evaluated offline from the checked-in architecture
    JSON (e.g. ``Qwen3-VL-30B-A3B-Instruct.json``) without instantiating the
    model. Mirrors the field mapping in
    ``cosmos_framework/callbacks/mfu.py::_ensure_initialised``.

    Args:
        text_config: The ``text_config`` dict from the HF model config.
        latent_patch_size: Generation-tower spatial patch size.
        latent_channel_size: VAE latent channels seen by ``vae2llm``/``llm2vae``.
        sound_dim: Audio latent dimension.
        frequency_embedding_size: Timestep sinusoidal embedding width.
        predict_text_tokens: Whether ``lm_head`` runs for a text CE loss.
        gen_moe_shared_expert: Gen-tower MoE always-on shared expert.
        gen_moe_shared_expert_intermediate_scale: Shared-expert width scale.
        gen_moe_top_k: Gen-tower top-k; ``None`` means match the und tower.

    Returns:
        A descriptor consumable by :func:`compute_omni_mot_flops_per_batch`.
    """
    num_experts = int(text_config.get("num_experts", 0))
    return get_omni_mot_model_descriptor(
        hidden_size=int(text_config["hidden_size"]),
        num_hidden_layers=int(text_config["num_hidden_layers"]),
        num_attention_heads=int(text_config["num_attention_heads"]),
        num_key_value_heads=int(text_config["num_key_value_heads"]),
        head_dim=text_config.get("head_dim"),
        intermediate_size=int(text_config["intermediate_size"]),
        vocab_size=int(text_config["vocab_size"]),
        use_moe=num_experts > 0,
        num_experts=num_experts,
        num_experts_per_tok=int(text_config.get("num_experts_per_tok", 0)),
        moe_intermediate_size=int(text_config.get("moe_intermediate_size", 0)),
        decoder_sparse_step=int(text_config.get("decoder_sparse_step", 1)),
        mlp_only_layers=list(text_config.get("mlp_only_layers", [])),
        gen_moe_shared_expert=gen_moe_shared_expert,
        gen_moe_shared_expert_intermediate_scale=gen_moe_shared_expert_intermediate_scale,
        gen_moe_top_k=gen_moe_top_k,
        latent_patch_size=latent_patch_size,
        latent_channel_size=latent_channel_size,
        sound_dim=sound_dim,
        frequency_embedding_size=frequency_embedding_size,
        predict_text_tokens=predict_text_tokens,
    )


def descriptor_from_model(model: Any) -> tuple[OmniMoTModelDescriptor, FlopFlags]:
    """Read the FLOP descriptor and matching flags off a live ``OmniMoTModel``.

    Mirrors ``cosmos_framework/callbacks/mfu.py::_ensure_initialised``,
    which is the reference for this mapping. Keeping the two in step is what
    makes a cost estimate comparable with the MFU a training run reports: both
    then count the same terms for the same architecture.

    Args:
        model: An instantiated ``OmniMoTModel``.

    Returns:
        ``(descriptor, flags)``. ``flags`` carries what the model config
        determines (understanding-tower freezing, activation checkpointing, VAE
        latent width); the remaining fields keep their :class:`FlopFlags`
        defaults because they are accounting choices, not model properties.
    """
    hf_vlm_cfg = model.net.language_model.config
    mot_cfg = model.config.vlm_config.model_instance.config
    net_cfg = model.net.config
    text_config = hf_vlm_cfg.text_config if hasattr(hf_vlm_cfg, "text_config") else hf_vlm_cfg

    num_experts = int(getattr(text_config, "num_experts", 0))
    descriptor = get_omni_mot_model_descriptor(
        hidden_size=text_config.hidden_size,
        num_hidden_layers=text_config.num_hidden_layers,
        num_attention_heads=text_config.num_attention_heads,
        num_key_value_heads=text_config.num_key_value_heads,
        head_dim=getattr(text_config, "head_dim", None),
        intermediate_size=text_config.intermediate_size,
        vocab_size=text_config.vocab_size,
        use_moe=num_experts > 0,
        num_experts=num_experts,
        num_experts_per_tok=int(getattr(text_config, "num_experts_per_tok", 0)),
        moe_intermediate_size=int(getattr(text_config, "moe_intermediate_size", 0)),
        decoder_sparse_step=int(getattr(text_config, "decoder_sparse_step", 1)),
        mlp_only_layers=list(getattr(text_config, "mlp_only_layers", [])),
        gen_moe_shared_expert=getattr(mot_cfg, "gen_moe_shared_expert", False),
        gen_moe_shared_expert_intermediate_scale=getattr(mot_cfg, "gen_moe_shared_expert_intermediate_scale", 1),
        gen_moe_top_k=getattr(mot_cfg, "gen_moe_top_k", None),
        latent_patch_size=getattr(net_cfg, "latent_patch_size", 2),
        latent_channel_size=getattr(net_cfg, "latent_channel_size", 48),
        action_dim=getattr(net_cfg, "action_dim", 32),
        sound_dim=getattr(net_cfg, "sound_dim", 64),
        frequency_embedding_size=getattr(net_cfg, "frequency_embedding_size", 256),
        predict_text_tokens=getattr(net_cfg, "predict_text_tokens", False),
    )

    # Any non-"none" mode recomputes a block forward during backward.
    ac_mode = getattr(getattr(model.config, "activation_checkpointing", None), "mode", "none")
    flags = FlopFlags(
        freeze_und=getattr(hf_vlm_cfg, "freeze_und", False),
        activation_checkpointing=ac_mode != "none",
        vae_latent_channels=int(getattr(net_cfg, "latent_channel_size", 48)),
    )
    return descriptor, flags


def descriptor_from_json_file(path: str | Path, **kwargs: Any) -> OmniMoTModelDescriptor:
    """Load a HF model-config JSON file and build a FLOP descriptor from it.

    Args:
        path: Path to the HF config JSON (must contain a ``text_config`` block,
            or be the text config itself).
        **kwargs: Forwarded to :func:`descriptor_from_hf_text_config`.

    Returns:
        A descriptor consumable by :func:`compute_omni_mot_flops_per_batch`.
    """
    with open(path) as handle:
        config = json.load(handle)
    text_config = config.get("text_config", config)
    return descriptor_from_hf_text_config(text_config, **kwargs)


def flops_per_sample(
    descriptor: OmniMoTModelDescriptor,
    spec: SampleSpec,
    flags: FlopFlags | None = None,
) -> float:
    """Training FLOPs (forward + backward) for one sample.

    Sequence packing keeps attention block-diagonal per sample, so per-sample
    FLOPs are additive across a packed step and this number is the correct unit
    of cost. The sample is described to the FLOP estimator as one
    ``(causal text, full generation)`` split pair, which is exactly the MoT
    attention pattern for a single packed sample.

    Args:
        descriptor: Model architecture descriptor.
        spec: The sample being priced.
        flags: Which FLOP terms to count. Defaults to :class:`FlopFlags`.

    Returns:
        Total FLOPs for this sample, including the frozen VAE encoder forward
        pass when ``flags.include_vae_encoder`` is set.
    """
    flags = flags or FlopFlags()
    gen_tokens = spec.gen_tokens

    total = compute_omni_mot_flops_per_batch(
        cfg=descriptor,
        B=1,
        text_tokens=spec.text_tokens,
        vision_tokens=spec.vision_tokens,
        sound_tokens=spec.sound_tokens,
        freeze_und=flags.freeze_und,
        vision_gen=True,
        sound_gen=spec.sound_tokens > 0,
        backwardpass_ratio=flags.backwardpass_ratio,
        # One packed sample = one causal understanding split followed by one
        # full-attention generation split.
        split_lens=[spec.text_tokens, gen_tokens],
        attn_modes=["causal", "full"],
        include_padding=False,
        use_activation_checkpointing=flags.activation_checkpointing,
    )

    if flags.include_vae_encoder:
        height, width = spec.pixel_shape
        total += compute_wan_vae_encoder_flops(
            B=1,
            T=spec.num_pixel_frames,
            H=height,
            W=width,
            z_dim=flags.vae_latent_channels,
        )

    return float(total)


@dataclass
class BucketMeasurement:
    """Measured throughput for one sample bucket on real hardware.

    Attributes:
        name: Bucket key (``SampleSpec.name``).
        spec: The sample shape that was run.
        steps: Timed steps (warm-up excluded).
        samples_per_step: Samples per step summed across all ranks.
        sec_per_step: Mean wall-clock seconds per step.
        gpu_sec_per_sample: ``sec_per_step * world_size / samples_per_step`` --
            the headline measured quantity.
        flops_per_sample: Analytic FLOPs for one sample of this shape.
        achieved_tflops_per_gpu: Retired TFLOP/s per GPU over the timed window.
        peak_mem_gb: Peak allocated device memory.
        world_size: Number of ranks that produced the measurement.
        note: Free-form annotation (e.g. why a bucket was skipped).
    """

    name: str
    spec: SampleSpec
    steps: int
    samples_per_step: float
    sec_per_step: float
    gpu_sec_per_sample: float
    flops_per_sample: float
    achieved_tflops_per_gpu: float
    peak_mem_gb: float = 0.0
    world_size: int = 1
    note: str = ""

    def mfu(self, peak_tflops_per_gpu: float) -> float:
        """Fraction of dense BF16 peak actually retired."""
        if peak_tflops_per_gpu <= 0:
            return 0.0
        return self.achieved_tflops_per_gpu / peak_tflops_per_gpu

    @property
    def gpu_sec_per_step(self) -> float:
        """GPU-seconds the whole job spends on one step (every rank is busy)."""
        return self.sec_per_step * self.world_size

    @property
    def tokens_per_step(self) -> float:
        """Packed tokens in one step, summed over all ranks."""
        return self.samples_per_step * self.spec.total_tokens

    @property
    def und_tokens_per_step(self) -> float:
        """Understanding-stream (text) tokens in one step, summed over all ranks."""
        return self.samples_per_step * self.spec.text_tokens

    @property
    def gen_tokens_per_step(self) -> float:
        """Generation-stream (vision + sound) tokens in one step, summed over all ranks."""
        return self.samples_per_step * self.spec.gen_tokens

    @property
    def und_area_per_step(self) -> float:
        """Understanding self-attention area: the sum of per-sample ``x**2``.

        The reasoner attends causally over its own sample's understanding stream
        and nothing else, so the true area is ``x*(x+1)/2``. The fitted
        coefficient absorbs the half; what has to be right here is the ``x**2``
        scaling and the fact that this is a per-sample sum.
        """
        return self.samples_per_step * float(self.spec.text_tokens) ** 2

    @property
    def gen_area_per_step(self) -> float:
        """Generation self-attention area: the sum of per-sample ``y**2``.

        The generation-over-generation quadrant alone. Its cost was previously
        welded to the generation-over-understanding quadrant by charging a single
        coefficient over ``(x + y) * y``; the two are now separate columns, so
        the fit can find them to be priced differently instead of being told they
        are not.
        """
        return self.samples_per_step * float(self.spec.gen_tokens) ** 2

    @property
    def cross_area_per_step(self) -> float:
        """Generation-over-understanding attention area: the sum of per-sample ``x * y``.

        Generation queries are keyed against their whole sample -- understanding
        tokens and latents alike -- so each of a sample's ``y`` generation tokens
        attends over ``x`` text keys on top of its own stream. This is the term
        that makes cost genuinely two-dimensional: it is why a sample's price
        cannot be recovered from ``x + y`` alone, since the understanding stream
        is attended TO by generation while only attending causally over itself.

        Deliberately a sum of per-sample areas, never an area of the packed
        total. Both attention passes are bounded by ``sample_offsets``, so no
        token attends across a sample boundary; charging the packed total would
        invent cross-sample attention the mask forbids and would grow as the
        square of the batch instead of linearly in it.
        """
        und = float(self.spec.text_tokens)
        gen = float(self.spec.gen_tokens)
        return self.samples_per_step * und * gen

    @property
    def flops_per_step(self) -> float:
        """Analytic FLOPs in one step, summed over all ranks."""
        return self.samples_per_step * self.flops_per_sample


@dataclass
class Calibration:
    """A benchmark run's measurements plus everything needed to reuse them.

    No run includes the EMA average, the benchmark building no ``net_ema`` at all:
    that frees an fp32 copy of the whole network and skips the per-step
    ``update_average``. The averaging is elementwise over parameters, so its absence
    leaves every per-token coefficient intact and lowers only ``c0``, which is
    therefore a lower bound for a job that trains with EMA enabled. It is stated
    here rather than recorded per run because no run can differ on it.

    Attributes:
        model_name: Human label for the model that was measured.
        device_name: CUDA device name reported by the benchmark host.
        peak_tflops_per_gpu: Dense BF16 peak for the measured device.
        world_size: Ranks in the benchmark job.
        context_parallel_shard_degree: Context-parallel group size the ranks were
            overlaid into. CP ranks cooperate on the SAME samples (splitting one
            packed sequence across the group via all-to-all) rather than each
            contributing a distinct sample, so it changes what ``samples_per_step``
            means and adds communication that a plain world-size change does not.
            Two runs that agree on ``world_size`` but not on this are not
            comparable, hence it is checked alongside ``world_size`` when pooling.
        gpus_per_node: GPUs per node in the benchmark job.
        descriptor_fields: Serialized :class:`OmniMoTModelDescriptor` fields, so
            the analytic layer can be re-evaluated offline for new shapes.
        flags: FLOP terms that were counted.
        measurements: Bucket name to measurement.
        measured_optimizer_step: Whether the timed steps included the parameter
            update. When false the benchmark ran forward and backward only, which
            leaves the per-token coefficients intact -- the update is elementwise
            over parameters, so it cannot depend on what the batch held -- but
            lowers ``c0`` by the update's cost. Incremental prices are unaffected;
            new-step prices are understated.
        metadata: Free-form provenance (git sha, experiment name, timestamp).
    """

    model_name: str
    device_name: str = ""
    peak_tflops_per_gpu: float = 0.0
    world_size: int = 1
    context_parallel_shard_degree: int = 1
    gpus_per_node: int = 4
    descriptor_fields: dict[str, Any] = field(default_factory=dict)
    flags: FlopFlags = field(default_factory=FlopFlags)
    measurements: dict[str, BucketMeasurement] = field(default_factory=dict)
    measured_optimizer_step: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def descriptor(self) -> OmniMoTModelDescriptor:
        """Rebuild the FLOP descriptor recorded at benchmark time.

        ``descriptor_fields`` is the descriptor's own ``_asdict()``, so every
        field is already resolved (``head_dim``, ``gen_moe_top_k``) and the
        NamedTuple can be reconstructed directly rather than re-running the
        defaulting factory.
        """
        return OmniMoTModelDescriptor(**self.descriptor_fields)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "device_name": self.device_name,
            "peak_tflops_per_gpu": self.peak_tflops_per_gpu,
            "world_size": self.world_size,
            "context_parallel_shard_degree": self.context_parallel_shard_degree,
            "gpus_per_node": self.gpus_per_node,
            "descriptor_fields": self.descriptor_fields,
            "flags": asdict(self.flags),
            "measurements": {
                name: {**asdict(measurement), "spec": asdict(measurement.spec)}
                for name, measurement in self.measurements.items()
            },
            "measured_optimizer_step": self.measured_optimizer_step,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Calibration:
        measurements: dict[str, BucketMeasurement] = {}
        for name, raw in payload.get("measurements", {}).items():
            raw = dict(raw)
            measurements[name] = BucketMeasurement(**{**raw, "spec": SampleSpec(**raw["spec"])})
        return cls(
            model_name=payload["model_name"],
            device_name=payload.get("device_name", ""),
            peak_tflops_per_gpu=float(payload.get("peak_tflops_per_gpu", 0.0)),
            world_size=int(payload.get("world_size", 1)),
            # Absent in calibrations written before CP was tracked, and those all
            # measured cp=1 (the field did not exist for the benchmark to vary it).
            context_parallel_shard_degree=int(payload.get("context_parallel_shard_degree", 1)),
            gpus_per_node=int(payload.get("gpus_per_node", 4)),
            descriptor_fields=payload.get("descriptor_fields", {}),
            flags=FlopFlags(**payload.get("flags", {})),
            measurements=measurements,
            # Absent in calibrations written before the update could be skipped, and
            # those all included it.
            measured_optimizer_step=bool(payload.get("measured_optimizer_step", True)),
            metadata=payload.get("metadata", {}),
        )

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str | Path) -> Calibration:
        with open(path) as handle:
            return cls.from_dict(json.load(handle))

    @classmethod
    def load_all(cls, paths: Sequence[str | Path]) -> Calibration:
        """Load one or more calibrations and pool their measurements into one fit.

        Sweeps are expensive, and the axes that matter are not all worth crossing:
        the clip-length curve wants every resolution tier, while identifying the
        understanding coefficients needs several caption lengths at only a handful
        of shapes. Crossing them wholesale multiplies GPU hours for rows that add
        nothing. Pooling separate runs gets the same rank at a fraction of the cost,
        and lets a calibration be extended later without re-measuring what is
        already known.

        Runs are only poolable if they describe the same machine and the same model,
        so the fields that would corrupt a shared fit are checked rather than
        assumed. Step time is not comparable across a different device, a different
        model, or a different world size -- ``c0`` in particular scales with the
        collectives, so merging two world sizes would fit a fixed cost that neither
        run ever paid. The same is true of the context-parallel degree: it changes
        both the all-to-all traffic every step pays and what ``samples_per_step``
        means (CP ranks share samples rather than each contributing one), so a run
        at cp=1 and one at cp=4 are not the same experiment even at equal world
        size. Whether the parameter update was timed is checked for the
        same reason: pooling a run that stepped the optimizer with one that did not
        fits a single intercept to two different fixed costs, landing between them
        and describing neither.

        Colliding bucket names are kept, not overwritten: the same shape measured
        twice is two pieces of evidence, and least squares is the right place to
        reconcile them. The later one is suffixed with its file's stem.

        Args:
            paths: Calibration JSON files, in the order they should be pooled.

        Returns:
            A single calibration whose measurements are the union of theirs.

        Raises:
            ValueError: If ``paths`` is empty, or if two runs disagree on the model,
                the device, the peak throughput, the world size, or which per-step
                work was timed.
        """
        if not paths:
            raise ValueError("No calibration files given")
        loaded = [(Path(path), cls.load(path)) for path in paths]
        first_path, merged = loaded[0]
        if len(loaded) == 1:
            return merged

        incompatible = (
            "model_name",
            "device_name",
            "peak_tflops_per_gpu",
            "world_size",
            "context_parallel_shard_degree",
            "measured_optimizer_step",
        )
        measurements = dict(merged.measurements)
        sources = {str(first_path): merged.metadata}
        for path, other in loaded[1:]:
            for attribute in incompatible:
                mine, theirs = getattr(merged, attribute), getattr(other, attribute)
                if mine != theirs:
                    raise ValueError(
                        f"Cannot pool {path} with {first_path}: {attribute} differs "
                        f"({theirs!r} vs {mine!r}). Step times from different models, devices, "
                        "world sizes or context-parallel degrees are not comparable, and neither "
                        "are steps that differ on whether the parameter update was timed, so a "
                        "shared fit would be meaningless."
                    )
            if other.descriptor_fields != merged.descriptor_fields or other.flags != merged.flags:
                log.warning(
                    f"{path} recorded a different FLOP descriptor or flag set than {first_path}. Pooling anyway, "
                    "since cost never depends on them, but the TFLOP and MFU columns will mix conventions."
                )
            for name, measurement in other.measurements.items():
                key = name if name not in measurements else f"{name}@{path.stem}"
                measurements[key] = replace(measurement, name=key)
            sources[str(path)] = other.metadata

        return replace(
            merged,
            measurements=measurements,
            metadata={**merged.metadata, "pooled_from": sources},
        )


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float] | None:
    """Gaussian elimination with partial pivoting. ``None`` when singular."""
    size = len(vector)
    rows = [list(matrix[i]) + [vector[i]] for i in range(size)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda row: abs(rows[row][col]))
        if abs(rows[pivot][col]) < 1e-30:
            return None
        rows[col], rows[pivot] = rows[pivot], rows[col]
        for below in range(col + 1, size):
            factor = rows[below][col] / rows[col][col]
            if factor:
                for cell in range(col, size + 1):
                    rows[below][cell] -= factor * rows[col][cell]
    out = [0.0] * size
    for row in reversed(range(size)):
        total = rows[row][size] - sum(rows[row][col] * out[col] for col in range(row + 1, size))
        out[row] = total / rows[row][row]
    return out


def _rank(design: list[list[float]], tolerance: float = 1e-9) -> int:
    """Numerical rank of a design matrix, after scaling every column to unit magnitude.

    Scaling first is essential rather than tidy: the token and area columns
    differ by four or five orders of magnitude, so an unscaled pivot test would
    either dismiss a perfectly good column as negligible or miss a genuine
    collinearity between two large ones.

    A design whose rank is below its width contains a coefficient the data
    cannot identify, and the least-squares solve will still return a number for
    it -- whatever the pivoting happens to produce. The caller checks rank
    instead of trusting the solve to fail, because silently invented
    coefficients are the failure mode that looks most like success.
    """
    width = len(design[0])
    scales = [max((abs(row[col]) for row in design), default=0.0) or 1.0 for col in range(width)]
    rows = [[row[col] / scales[col] for col in range(width)] for row in design]
    pivot_row = 0
    for col in range(width):
        if pivot_row >= len(rows):
            break
        pivot = max(range(pivot_row, len(rows)), key=lambda row: abs(rows[row][col]))
        if abs(rows[pivot][col]) < tolerance:
            continue
        rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
        base = rows[pivot_row]
        for below in range(pivot_row + 1, len(rows)):
            factor = rows[below][col] / base[col]
            if factor:
                rows[below] = [cell - factor * base_cell for cell, base_cell in zip(rows[below], base, strict=True)]
        pivot_row += 1
    return pivot_row


def _least_squares(design: list[list[float]], targets: list[float]) -> list[float] | None:
    """Least-squares solve of ``design @ x = targets`` via scaled normal equations.

    Columns are normalized to unit scale before forming ``A^T A`` and the
    solution is unscaled afterwards. Without that the token and token-squared
    columns differ by four or five orders of magnitude, and squaring them into
    the normal equations costs enough precision to turn a well-posed fit into a
    singular one.
    """
    width = len(design[0])
    scales = [max((abs(row[col]) for row in design), default=0.0) or 1.0 for col in range(width)]
    scaled = [[row[col] / scales[col] for col in range(width)] for row in design]
    normal = [[sum(row[i] * row[j] for row in scaled) for j in range(width)] for i in range(width)]
    moment = [sum(row[i] * target for row, target in zip(scaled, targets, strict=True)) for i in range(width)]
    solution = _solve(normal, moment)
    if solution is None:
        return None
    return [value / scale for value, scale in zip(solution, scales, strict=True)]


@dataclass(frozen=True)
class TokenCostModel:
    """Measured step time as a general quadratic in understanding and generation tokens.

    Writing ``x`` for a sample's understanding (text) tokens and ``y`` for its
    generation (vision + sound) tokens, step time is modelled as::

        gpu_sec_per_step = c0 + sum_i [ A*x_i**2 + B*y_i**2 + C*x_i*y_i
                                        + D*x_i + E*y_i + F ]

    Two constants, deliberately. ``c0`` is paid once per step and ``F`` once per
    sample, and they are different physical things: the optimizer step, the EMA
    update and the FSDP collectives all have the parameter count as their payload
    and so cost the same whether the step held one sample or sixteen, while a
    sample's own setup -- its VAE encode launch, its timestep embedding, its slice
    of the packing -- recurs per sample. Only a sweep that varies batch size can
    tell the two apart; ``F`` is what makes an isolated sample cost more than its
    tokens alone, and ``c0`` is what a sample avoids by joining a step that exists
    already.

    The three quadratic terms are the three quadrants of the attention mask, and
    keeping them separate is the point of the form. The reasoner attends CAUSALLY
    over its own understanding stream, area ``x**2/2``, giving ``A``. Generation
    attends over its own stream, area ``y**2``, giving ``B``. And generation also
    attends over the text, area ``x*y``, giving ``C``. A single quadratic in
    ``x + y`` would charge ``(x + y)**2`` and so overcharge the understanding
    quadrant, which no token ever attends over bidirectionally; tying ``C`` to
    ``B``, as an earlier form did by charging one coefficient over ``(x + y)*y``,
    asserts that the two quadrants cost the same per element rather than measuring
    whether they do.

    The two linear terms are separate for a different reason: understanding tokens
    run forward-only through a frozen tower, while generation tokens pay forward
    and backward plus the VAE encode. Expecting ``D`` to come out well below ``E``
    is a prediction of the model, not an artifact of it.

    Every token term is a per-sample sum, so a sample's own cost does not depend
    on what shares its step. That is what makes the model composable, and it is
    only correct because all three attention quadrants are bounded by
    ``sample_offsets`` and no token attends across a sample boundary.

    Attributes:
        fixed_gpu_sec_per_step: ``c0`` -- paid once per step however many samples
            it holds: optimizer step, EMA update, and the FSDP collectives whose
            payload is the parameter count. A sparse MoE pays it in full at any
            batch size. In GPU-seconds summed over the ranks that ran it at the
            same time, which makes it the one coefficient whose value depends on
            ``world_size``; :attr:`fixed_sec_per_step_per_rank` is the form to
            compare against a wall clock.
        fixed_gpu_sec_per_sample: ``F`` -- paid once per sample regardless of its
            size. Only separable from ``c0`` when batch size varies; when it is
            not fitted, per-sample setup has been absorbed into ``c0`` and the
            linear rates.
        gpu_sec_per_und_area: ``A`` -- understanding causal self-attention over
            ``x**2``, absorbing the factor of a half.
        gpu_sec_per_gen_area: ``B`` -- generation self-attention over ``y**2``.
        gpu_sec_per_cross_area: ``C`` -- generation attending over the
            understanding stream, area ``x*y``. Independent of ``B`` rather than
            tied to it.
        gpu_sec_per_und_token: ``D`` -- work linear in understanding tokens.
        gpu_sec_per_gen_token: ``E`` -- work linear in generation tokens:
            projections, expert GEMMs, and the VAE encode, which is linear in
            tokens because a token is a fixed 4096 pixels at every resolution.
        world_size: Ranks that shared the step these coefficients were measured
            over. Every per-sample term is already a one-GPU quantity, because
            one rank runs a sample from end to end; ``c0`` is not, because it is
            replicated work that all the ranks did at once. Recording the scale
            is what lets a consumer at a different one recover the per-rank
            floor instead of dividing by its own rank count.
        fitted_terms: Which columns the fit actually used. A coefficient whose
            column is absent is fixed at zero rather than guessed, and the label
            ``DE_shared_linear`` means ``D`` and ``E`` were tied to a single rate.
        und_identifiable: Whether the sweep separated ``D`` from ``E``, which
            needs either caption length or batch size to vary. When false, ``D``
            was tied to ``E`` rather than measured, so understanding tokens are
            charged at the generation rate, which overcharges them since
            ``D <= E`` physically. That is not the whole story, though: such a
            sweep also loses ``F``, which incremental pricing would have charged,
            so the errors have opposite signs and the quoted price is neither an
            upper nor a lower bound. Either way, another caption length is
            extrapolation -- see :meth:`extrapolates_und`.
        num_points: Measured steps the fit used.
        distinct_und_tokens: Distinct per-sample ``x`` among them. Separating
            ``D`` from ``A`` needs three.
        distinct_gen_tokens: Distinct per-sample ``y`` among them.
        r_squared: Fraction of step-time variance explained. Nearly useless on a
            wide sweep, where the longest clips set almost all the variance and
            can hide order-of-magnitude errors on the shortest.
        max_residual_fraction: Largest per-point relative error on whole-step
            time. The statistic to judge the fit by.
        und_token_range: Smallest and largest ``x`` the fit saw.
        gen_token_range: Smallest and largest ``y`` the fit saw.
        source: How the coefficients were arrived at, so a degenerate fit is
            never mistaken for a confident one.
    """

    fixed_gpu_sec_per_step: float = 0.0
    fixed_gpu_sec_per_sample: float = 0.0
    gpu_sec_per_und_area: float = 0.0
    gpu_sec_per_gen_area: float = 0.0
    gpu_sec_per_cross_area: float = 0.0
    gpu_sec_per_und_token: float = 0.0
    gpu_sec_per_gen_token: float = 0.0
    world_size: int = 1
    fitted_terms: tuple[str, ...] = ()
    und_identifiable: bool = False
    num_points: int = 0
    distinct_und_tokens: int = 0
    distinct_gen_tokens: int = 0
    r_squared: float = 0.0
    max_residual_fraction: float = 0.0
    und_token_range: tuple[int, int] = (0, 0)
    gen_token_range: tuple[int, int] = (0, 0)
    source: str = "unfitted"

    @property
    def fixed_sec_per_step_per_rank(self) -> float:
        """``c0`` as seconds on one rank's clock -- the form that carries across scale.

        ``c0`` is replicated work: the optimizer update and the collectives whose
        payload is the parameter count, which every rank performs at the same time
        on its own shard. Its GPU-second total therefore grows with the world size
        it was measured at, while what a single rank waits for does not -- at eight
        ranks or eight hundred, each still streams the same parameters and moves
        about the same bytes. Dividing the total by the scale it was measured at
        leaves that per-rank floor, which is what any consumer reasoning about wall
        clock wants, at any world size.

        This is a first-order model rather than an identity, and it errs low: the
        latency of a collective grows slowly with the rank count and straggler
        spread grows faster, so a job far above the measured scale waits somewhat
        longer than this. What it is emphatically not is ``c0`` divided by the
        *consumer's* rank count, which is what treating the fitted total as though
        it had been measured at that scale gives; that shrinks the floor by the
        ratio between the two scales, which at a 4-rank calibration and a 256-rank
        job is a factor of 64.

        This is the quantity a time budget is denominated in, and the key
        ``cost_model.report --json`` emits it under, so that pinning a fit into a
        config stays a copy rather than an arithmetic step. The config field it
        feeds is named ``fixed_gpu_sec_per_step`` after the fitted total, so on
        this one term the names cross over while the units do not.
        """
        return self.fixed_gpu_sec_per_step / self.world_size

    def marginal_gpu_sec(self, und_tokens: int, gen_tokens: int) -> float:
        """GPU-seconds one sample adds to a step, excluding the step's ``c0``.

        This is exactly the incremental cost of adding the sample to a step that
        has room for it, and it does not depend on what that step already
        carries::

            T(B + {x,y}) - T(B) = A*x**2 + B*y**2 + C*x*y + D*x + E*y + F

        ``c0`` cancels because the step pays it either way, while ``F`` does NOT
        cancel: it is charged per sample, so the new sample brings its own. No
        cross term with the existing samples appears because all three attention
        quadrants are bounded by ``sample_offsets`` -- the new sample neither
        attends to those already in the step nor they to it. So a large existing
        batch does not make the next sample any cheaper or dearer, and the caller
        needs to know nothing about the batch to price it.

        Note that ``C*x*y`` is a cross term WITHIN the sample, between its own
        two streams, which is a different thing entirely and is real.

        The one case this does not cover is a step already at its packing
        budget, where the sample cannot join and must open a new one; see
        :meth:`new_step_gpu_sec`.
        """
        und = float(und_tokens)
        gen = float(gen_tokens)
        return (
            self.fixed_gpu_sec_per_sample
            + self.gpu_sec_per_und_area * und * und
            + self.gpu_sec_per_gen_area * gen * gen
            + self.gpu_sec_per_cross_area * und * gen
            + self.gpu_sec_per_und_token * und
            + self.gpu_sec_per_gen_token * gen
        )

    def new_step_gpu_sec(self, und_tokens: int, gen_tokens: int) -> float:
        """Incremental cost when the sample has to open a fresh step.

        A step can only hold ``max_num_tokens_after_packing`` tokens. Up to that
        point admitting a sample costs :meth:`marginal_gpu_sec`; the sample that
        does not fit forces another step and pays its ``c0`` in full. That makes
        the true incremental cost a step function of how full the batch is, and
        these two methods are its two values -- the second being the worst case
        for a batch packed right up to the budget.
        """
        return self.fixed_gpu_sec_per_step + self.marginal_gpu_sec(und_tokens, gen_tokens)

    def predict_step_gpu_sec(self, und_tokens: int, gen_tokens: int, samples_per_step: float) -> float:
        """Predicted whole-step GPU-seconds for ``samples_per_step`` identical samples."""
        return self.fixed_gpu_sec_per_step + samples_per_step * self.marginal_gpu_sec(und_tokens, gen_tokens)

    @property
    def crossover_gen_tokens(self) -> float:
        """Clip length at which generation self-attention costs as much as linear generation work.

        Solves ``B*y**2 == E*y``. Below it a clip is priced essentially linearly
        in its own tokens; above it attention dominates and doubling the frames
        roughly quadruples the cost. Deliberately ignores the understanding
        stream, so it is a rule of thumb about clip length rather than the exact
        point where all quadratic terms overtake all linear ones. ``0.0`` when no
        generation attention term was fitted.
        """
        if self.gpu_sec_per_gen_area <= 0 or self.gpu_sec_per_gen_token <= 0:
            return 0.0
        return self.gpu_sec_per_gen_token / self.gpu_sec_per_gen_area

    def extrapolates(self, und_tokens: int, gen_tokens: int) -> bool:
        """Whether pricing this shape means leaving the measured range on either axis."""
        return self.extrapolates_und(und_tokens) or self._extrapolates_gen(gen_tokens)

    def extrapolates_und(self, und_tokens: int) -> bool:
        """Whether this understanding length is one the fit cannot speak to.

        This bites much harder than the generation check on a sweep that never
        varied caption length. There the range collapses to the single observed
        ``x``, so every other ``x`` is flagged -- correctly, because the
        understanding terms were folded into the constants and ``E`` at that one
        value and carry no information about any other.
        """
        low, high = self.und_token_range
        return high > 0 and not (low <= und_tokens <= high)

    def _extrapolates_gen(self, gen_tokens: int) -> bool:
        low, high = self.gen_token_range
        return high > 0 and not (low <= gen_tokens <= high)


@dataclass(frozen=True)
class _Column:
    """One column of the design matrix and the coefficients it fills in.

    ``sets`` is usually a single field, but a column can drive two when the data
    cannot tell them apart and the honest response is to tie them together rather
    than to invent one or discard the other.
    """

    label: str
    value: Any
    sets: tuple[str, ...]


# Named for the coefficient each fills rather than by letter, because the letters
# are easy to transpose and a swapped pair here would be a silent mispricing.
_COL_C0 = _Column("c0_per_step", lambda m: 1.0, ("fixed_gpu_sec_per_step",))
_COL_F = _Column("F_per_sample", lambda m: m.samples_per_step, ("fixed_gpu_sec_per_sample",))
_COL_A = _Column("A_und_area", lambda m: m.und_area_per_step, ("gpu_sec_per_und_area",))
_COL_B = _Column("B_gen_area", lambda m: m.gen_area_per_step, ("gpu_sec_per_gen_area",))
_COL_C = _Column("C_cross_area", lambda m: m.cross_area_per_step, ("gpu_sec_per_cross_area",))
_COL_D = _Column("D_und_token", lambda m: m.und_tokens_per_step, ("gpu_sec_per_und_token",))
_COL_E = _Column("E_gen_token", lambda m: m.gen_tokens_per_step, ("gpu_sec_per_gen_token",))
# One rate for both streams, over the packed length. The fallback for a sweep
# that held caption length fixed; see ``_DESIGNS``.
_COL_DE_TIED = _Column(
    "DE_shared_linear", lambda m: m.tokens_per_step, ("gpu_sec_per_und_token", "gpu_sec_per_gen_token")
)

# Candidate designs, richest first. Every one is rank-checked against the data
# before being fitted, so this is a preference order rather than a set of
# assumptions baked in.
#
# The order in which terms are given up is by how well each is grounded, so that
# what survives a degenerate sweep is what the data can actually support:
#
#   E, B, c0   never dropped -- linear generation work, generation self-attention,
#              and the per-step floor are the backbone, and all three are
#              identifiable from a clip-length sweep alone
#   A          dropped first -- understanding causal self-attention over a caption
#              two orders of magnitude shorter than a clip is the smallest real
#              effect here, and the easiest for noise to swamp
#   F          dropped next -- needs batch size to vary at more than one caption
#              length, and competes directly with D for the same signal
#   C          then -- generation attending over text is physically certain to
#              exist, but its magnitude is hard to separate from B when x barely
#              moves
#   D          last -- when even this cannot be had, it is tied to E rather than
#              discarded
#
# The tied designs at the end are the interesting case. A sweep that varies clip
# length but not caption length cannot separate the understanding terms from the
# constants, because ``x``, ``x**2`` and the intercept are then all constant
# columns. There are two ways to respond:
#
#   drop the und terms   assume D = 0, and their cost lands in the constants
#   tie them to gen      assume D = E, and their cost stays in the per-token rate
#
# These are NOT distinguishable by fit quality. At fixed ``x`` and fixed batch the
# two designs span the same subspace, so they reproduce the measured step times
# identically, to floating-point. Any apparent accuracy difference between them is
# bookkeeping, not evidence -- which is exactly what rank deficiency means.
#
# They price very differently all the same, because incremental pricing charges
# the per-token rate and does not charge ``c0``. So the choice is a policy one, and
# it is made on the one thing that IS provable: physically ``0 <= D <= E``, since
# understanding tokens run forward-only through a frozen tower and pay no VAE
# encode, and they cannot cost more per token than a trained one unless the
# understanding tower is the wider of the two. Tying therefore keeps more of the
# real cost in the quoted price than dropping does, and that is why it is
# preferred. ``und_identifiable`` records that ``D`` was assumed, not measured.
#
# Tying is NOT an upper bound, though, and it is worth being exact about why. That
# claim would follow from ``D <= E`` alone, but a sweep degenerate enough to force
# tying also loses ``F`` -- and ``F`` is charged incrementally, so losing it
# understates. One error is positive and the other negative, so the net is
# unsigned. Neither fallback brackets the truth once there is a per-sample cost.
# Note that no design drops ``D`` outright while keeping ``E`` free. Dropping it
# would book the understanding cost to ``c0``, which incremental pricing never
# charges, so a still would be quoted as very nearly free. Tying is the fallback
# instead, and the last three designs are where that happens.
_DESIGNS: tuple[tuple[_Column, ...], ...] = (
    (_COL_C0, _COL_F, _COL_A, _COL_B, _COL_C, _COL_D, _COL_E),
    (_COL_C0, _COL_F, _COL_B, _COL_C, _COL_D, _COL_E),
    (_COL_C0, _COL_A, _COL_B, _COL_C, _COL_D, _COL_E),
    (_COL_C0, _COL_B, _COL_C, _COL_D, _COL_E),
    (_COL_C0, _COL_F, _COL_B, _COL_D, _COL_E),
    (_COL_C0, _COL_B, _COL_D, _COL_E),
    (_COL_C0, _COL_DE_TIED, _COL_B),
    (_COL_C0, _COL_DE_TIED),
    (_COL_DE_TIED,),
)


def fit_token_cost_model(calibration: Calibration) -> TokenCostModel:
    """Fit the general quadratic cost model to the measured steps.

    Every measurement contributes one row, with columns for the seven mechanisms
    described on :class:`TokenCostModel`. The sweep axes probe independent
    directions: clip length moves ``y``, ``y**2`` and the cross area, caption
    length moves ``x``, ``x**2`` and the cross area again, and batch size scales
    every per-sample column while leaving the intercept alone, which is what
    separates ``c0`` from ``F`` and pins both down instead of leaving them an
    extrapolation to zero tokens.

    ``F`` is the term with the sharpest requirement: batch size has to vary at all,
    or the intercept and the per-sample column are proportional and only their sum
    ``c0 + n*F`` is determined. Once it does vary the design is full rank and well
    conditioned, and the two axes need NOT be crossed -- a batch ladder at a single
    caption length suffices, given other captions elsewhere in the sweep. What
    makes ``F`` hard in practice is therefore not the design but noise: it competes
    with ``D`` for the same modest signal, and on real measurements the loser of
    that competition tends to come out negative, which the second guard catches.

    Two guards decide which coefficients the fit is allowed to report.

    The first is rank. A design with a column the data cannot separate still
    yields a solution -- an arbitrary one, since any of infinitely many
    coefficient vectors fits equally well -- and it looks exactly like a good
    fit. So each candidate design is rank-checked first and skipped if
    degenerate. This is what stops a sweep at one fixed caption length from
    inventing an understanding-attention coefficient it has no evidence for.

    The second is non-negativity. A negative coefficient is physically
    impossible; it would mean longer sequences run faster. When the
    unconstrained solve produces one it is a symptom of efficiency drift across
    the sweep -- short steps are launch- and bandwidth-bound and retire tokens
    more slowly per unit of work than long ones -- not a real measurement, so
    the fit degrades to a simpler design instead of keeping it.

    Args:
        calibration: Measurements from one or more benchmark runs.

    Returns:
        The fitted :class:`TokenCostModel`. ``fitted_terms`` says which
        coefficients came from the data; the rest are zero.
    """
    usable = [
        measurement
        for measurement in calibration.measurements.values()
        if measurement.gpu_sec_per_step > 0 and measurement.spec.total_tokens > 0 and measurement.samples_per_step > 0
    ]
    if not usable:
        return TokenCostModel(world_size=calibration.world_size, source="unfitted (no usable measurements)")

    targets = [measurement.gpu_sec_per_step for measurement in usable]
    und_lengths = [measurement.spec.text_tokens for measurement in usable]
    gen_lengths = [measurement.spec.gen_tokens for measurement in usable]
    distinct_und = len(set(und_lengths))
    distinct_gen = len(set(gen_lengths))

    base = TokenCostModel(
        # ``c0`` comes out of the fit in GPU-seconds over this many ranks, and is
        # meaningless as a wall-clock figure without it.
        world_size=calibration.world_size,
        num_points=len(usable),
        distinct_und_tokens=distinct_und,
        distinct_gen_tokens=distinct_gen,
        und_token_range=(min(und_lengths), max(und_lengths)),
        gen_token_range=(min(gen_lengths), max(gen_lengths)),
    )

    def finish(columns: tuple[_Column, ...], coefficients: list[float], rejected: list[str]) -> TokenCostModel:
        fields = {
            "fixed_gpu_sec_per_step": 0.0,
            "fixed_gpu_sec_per_sample": 0.0,
            "gpu_sec_per_und_area": 0.0,
            "gpu_sec_per_gen_area": 0.0,
            "gpu_sec_per_cross_area": 0.0,
            "gpu_sec_per_und_token": 0.0,
            "gpu_sec_per_gen_token": 0.0,
        }
        for column, coefficient in zip(columns, coefficients, strict=True):
            for name in column.sets:
                fields[name] = coefficient
        labels = tuple(column.label for column in columns)
        source = f"{len(columns)}-term fit over {len(usable)} steps"
        if _COL_DE_TIED.label in labels:
            source += "; the caption never varied, so D was tied to E rather than measured"
        # Say WHY each richer design lost, not just that it did. "Unidentifiable"
        # and "came out negative" call for different fixes -- another sweep axis
        # versus a suspect measurement -- so collapsing them into one message sends
        # the reader after the wrong problem.
        if rejected:
            source += "; passed over " + "; ".join(rejected)
        model = replace(
            base,
            **fields,
            fitted_terms=labels,
            und_identifiable=_COL_D.label in labels,
            source=source,
        )
        predictions = [
            model.predict_step_gpu_sec(m.spec.text_tokens, m.spec.gen_tokens, m.samples_per_step) for m in usable
        ]
        mean = sum(targets) / len(targets)
        total_variance = sum((target - mean) ** 2 for target in targets)
        residual = sum((target - prediction) ** 2 for target, prediction in zip(targets, predictions, strict=True))
        worst = max(abs(target - prediction) / target for target, prediction in zip(targets, predictions, strict=True))
        return replace(
            model,
            r_squared=1.0 - residual / total_variance if total_variance > 0 else 0.0,
            max_residual_fraction=worst,
        )

    rejected: list[str] = []
    for columns in _DESIGNS:
        width = len(columns)
        design = [[column.value(m) for column in columns] for m in usable]
        if len(usable) < width:
            rejected.append(f"the {width}-term design ({len(usable)} steps cannot fit {width} coefficients)")
            continue
        if _rank(design) < width:
            rejected.append(
                f"the {width}-term design (rank {_rank(design)} of {width}; the sweep cannot separate them)"
            )
            continue
        coefficients = _least_squares(design, targets)
        if coefficients is None:
            rejected.append(f"the {width}-term design (singular)")
            continue
        # Note that c0 is deliberately NOT capped against the shortest measured
        # step. On a short clip the fixed cost genuinely is most of the step, and
        # clamping it there would only inflate the per-token terms to compensate.
        negative = [column.label for column, value in zip(columns, coefficients, strict=True) if value < 0]
        if negative:
            rejected.append(f"the {width}-term design ({', '.join(negative)} came out negative)")
            continue
        return finish(columns, coefficients, rejected)

    return replace(base, source="unfitted (every candidate design was degenerate or unphysical)")


@dataclass(frozen=True)
class CostEstimate:
    """Per-sample cost for one sample shape.

    Attributes:
        name: Bucket key.
        spec: The sample that was priced.
        total_tokens: Packed sequence length the sample occupies.
        flops_per_sample: Analytic training FLOPs for the sample.
        marginal_gpu_sec_per_sample: The quoted cost, and what the GPU-hour and
            throughput fields derive from: the cost caused by the sample's own
            tokens, which is exactly what adding it to a step with room costs,
            whatever else that step already holds. See
            :meth:`TokenCostModel.marginal_gpu_sec`.
        new_step_gpu_sec_per_sample: What the sample costs if it does not fit in
            the current step and has to open a new one: the same marginal cost
            plus a whole ``c0``. Upper bound on the incremental cost, and the
            right number for a batch already packed to its token budget.
        marginal_tflops_per_gpu: Per-GPU throughput of the token-proportional
            work. Diagnostic: analytic FLOPs over the priced marginal time.
        source: How the marginal term was obtained.
        mfu: Marginal FLOPs retired as a fraction of dense BF16 peak, when a
            peak is known. Higher than whole-step MFU by construction, since
            the bandwidth floor is excluded.
        measured_marginal_gpu_sec_per_sample: The same quantity taken straight
            from this shape's own measurement, when it has one. ``None`` for a
            shape that was never run. Carried alongside the modelled value so
            the fit can be checked against the point it is meant to reproduce
            instead of silently replacing it.
        quadratic_fraction: Share of the marginal cost attributable to the three
            attention terms. Near zero means this shape is priced essentially
            linearly in tokens; above a half means sequence length is the
            dominant cost driver.
        und_fraction: Share of the marginal cost attributable to the
            understanding stream -- its linear term, its causal self-attention,
            and the whole cross area, since that exists only because generation
            attends over the text. Large on images and short clips, negligible on
            long ones. Does not reach 100% even on a caption-only sample, because
            the per-sample constant ``F`` belongs to neither stream.
        extrapolated_tokens: Whether this shape falls outside the range the fit
            actually observed, on either axis.
        extrapolated_und_tokens: Whether the understanding length specifically is
            outside it. Worth separating because a sweep that never varied
            caption length cannot price any other caption length, however well
            it does on clip length.
    """

    name: str
    spec: SampleSpec
    total_tokens: int
    flops_per_sample: float
    marginal_gpu_sec_per_sample: float
    marginal_tflops_per_gpu: float
    source: str
    new_step_gpu_sec_per_sample: float = 0.0
    mfu: float = 0.0
    measured_marginal_gpu_sec_per_sample: float | None = None
    quadratic_fraction: float = 0.0
    und_fraction: float = 0.0
    extrapolated_tokens: bool = False
    extrapolated_und_tokens: bool = False

    @property
    def residual_fraction(self) -> float | None:
        """Relative gap between the modelled and measured marginal cost.

        ``None`` when this shape was never measured. A large value on a shape
        that WAS measured means the quadratic form is not capturing it, which
        matters more than the fit's aggregate r-squared: a handful of very long
        clips dominate that statistic and can mask poor accuracy on short ones.
        """
        measured = self.measured_marginal_gpu_sec_per_sample
        if measured is None or measured <= 0:
            return None
        return (self.marginal_gpu_sec_per_sample - measured) / measured

    @property
    def gpu_hours_per_sample(self) -> float:
        return self.marginal_gpu_sec_per_sample / SECONDS_PER_HOUR

    @property
    def gpu_hours_per_1k_samples(self) -> float:
        return self.gpu_hours_per_sample * 1_000.0

    @property
    def samples_per_gpu_hour(self) -> float:
        if self.marginal_gpu_sec_per_sample <= 0:
            return 0.0
        return SECONDS_PER_HOUR / self.marginal_gpu_sec_per_sample


def _measured_for(spec: SampleSpec, calibration: Calibration) -> BucketMeasurement | None:
    """This exact shape's own measurement, if the benchmark ran it.

    Matched on the two token counts rather than on the bucket name. A sweep can
    measure one shape at several caption lengths and several batch sizes, and
    those rows share a name stem, so the name alone no longer identifies a shape.

    When more than one row matches, the largest step wins: its fixed cost is
    spread over the most samples, so the marginal cost implied by subtracting
    ``c0`` is the least sensitive to how well ``c0`` itself was fitted.
    """
    matches = [
        measurement
        for measurement in calibration.measurements.values()
        if measurement.spec.text_tokens == spec.text_tokens
        and measurement.spec.gen_tokens == spec.gen_tokens
        and measurement.gpu_sec_per_step > 0
        and measurement.samples_per_step > 0
    ]
    if not matches:
        return None
    return max(matches, key=lambda measurement: measurement.samples_per_step)


def estimate(
    spec: SampleSpec,
    calibration: Calibration,
    model: TokenCostModel | None = None,
) -> CostEstimate:
    """Price one sample shape from the fitted token cost model.

    Pricing is incremental: the sample is charged
    ``A*x**2 + B*y**2 + C*x*y + D*x + E*y + F``, what adding it to a step with
    room for it actually costs, and none of the step's ``c0``, which is paid
    whether or not the sample is there. That figure is exact rather than
    asymptotic, and
    holds however many tokens the step already carries -- see
    :meth:`TokenCostModel.marginal_gpu_sec`.

    The estimate also carries ``new_step_gpu_sec_per_sample``, the cost when the
    batch is packed to its token budget and the sample has to open a step of its
    own. The two bracket what the sample really costs; nothing here divides ``c0``
    across an assumed step size to land somewhere in between.

    Marginal cost comes from the fit for EVERY shape, measured or not. Using the
    fit even where a direct measurement exists is deliberate: it makes cost a
    single continuous function of the two token counts rather than a lookup table
    with interpolation between entries, so a 200-frame clip is priced
    consistently with the 181- and 241-frame clips that bracket it. The
    shape's own measurement is still carried on the estimate, so the two can be
    compared -- see :attr:`CostEstimate.residual_fraction`.

    Analytic FLOPs are reported but never priced: they feed MFU only, so the
    quoted cost rests on measured time alone.

    Args:
        spec: Sample shape to price.
        calibration: Measurements from a benchmark run.
        model: Fitted token cost model. Defaults to fitting it from
            ``calibration``; pass one explicitly to price many shapes against a
            single fit instead of refitting per shape.

    Returns:
        A :class:`CostEstimate`.
    """
    model = model if model is not None else fit_token_cost_model(calibration)
    sample_flops = flops_per_sample(calibration.descriptor, spec, calibration.flags)

    und = spec.text_tokens
    gen = spec.gen_tokens
    marginal_gpu_sec = model.marginal_gpu_sec(und, gen)
    attention_term = (
        model.gpu_sec_per_und_area * float(und) ** 2
        + model.gpu_sec_per_gen_area * float(gen) ** 2
        + model.gpu_sec_per_cross_area * float(und) * gen
    )
    # Everything the understanding stream is responsible for: its linear term, its
    # own causal self-attention, and the whole cross area, which exists only
    # because generation queries attend over the text. The per-sample constant F
    # is charged to neither stream -- it is there precisely because it is the part
    # of the cost that no token explains -- so it sits in the denominator only,
    # and und% plus gen% falls short of 100% by exactly F's share.
    und_term = (
        model.gpu_sec_per_und_token * und
        + model.gpu_sec_per_und_area * float(und) ** 2
        + model.gpu_sec_per_cross_area * float(und) * gen
    )
    quadratic_fraction = attention_term / marginal_gpu_sec if marginal_gpu_sec > 0 else 0.0
    und_fraction = und_term / marginal_gpu_sec if marginal_gpu_sec > 0 else 0.0

    measured = _measured_for(spec, calibration)
    measured_marginal: float | None = None
    if measured is not None and measured.gpu_sec_per_step > 0 and measured.samples_per_step > 0:
        # Strip the fixed term before dividing among the step's samples;
        # ``measured.gpu_sec_per_sample`` would hand each sample a full share of
        # that step's floor, which is the non-incremental answer and is inflated
        # whenever the benchmark step was small.
        measured_step = max(measured.gpu_sec_per_step - model.fixed_gpu_sec_per_step, 0.0)
        measured_marginal = measured_step / measured.samples_per_step

    marginal_tflops = sample_flops / marginal_gpu_sec / 1e12 if marginal_gpu_sec > 0 else 0.0
    mfu = marginal_tflops / calibration.peak_tflops_per_gpu if calibration.peak_tflops_per_gpu > 0 else 0.0

    return CostEstimate(
        name=spec.name,
        spec=spec,
        total_tokens=spec.total_tokens,
        flops_per_sample=sample_flops,
        marginal_gpu_sec_per_sample=marginal_gpu_sec,
        marginal_tflops_per_gpu=marginal_tflops,
        source=model.source,
        new_step_gpu_sec_per_sample=model.new_step_gpu_sec(und, gen),
        mfu=mfu,
        measured_marginal_gpu_sec_per_sample=measured_marginal,
        quadratic_fraction=quadratic_fraction,
        und_fraction=und_fraction,
        extrapolated_tokens=model.extrapolates(und, gen),
        extrapolated_und_tokens=model.extrapolates_und(und),
    )
