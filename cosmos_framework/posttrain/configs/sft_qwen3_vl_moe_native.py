# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
r"""Qwen3-VL-MoE full-parameter SFT through the i4-native policy builder (the "native seat").

"Seat" here means the model construction path the SFT objective sits on. The HF backend refuses MoE VLMs
(``model/hf.py::_is_moe_vlm``: the general weight loader has no fused-expert handling, and the Transformers 5
``Qwen3VLMoeTextExperts`` stores ``gate_up_proj`` / ``down_proj`` transposed relative to a Transformers 4
checkpoint). ``model/qwen3_vl.py::build_qwen3_vl_native_policy`` is the construction the online algorithms
train and its state dict matches such a checkpoint key for key, so SFT selects it with ``model_backend='i4'``
and the ``NativePolicyBuildSpec`` shape of ``ppo_qwen3vl30b_a3b.py``.

Training stack, each field with its reason:
- ``attn_impl='sdpa'`` (the fleet's ``ppo_qwen3vl30b_a3b.py``): the native eager path builds a 4-D additive
  mask from the collated ``attention_mask``; sdpa/flash let the kernel handle right padding.
- ``dtype='bfloat16'``: ``build_qwen3_vl_model`` requires bfloat16 storage for qwen3_vl_moe (the MoE expert
  kernel); ``rl.master_dtype`` / ``rl.param_dtype`` mirror it so the recorded precision identity describes the
  model that trains. fp32 master weights therefore live INSIDE the optimizer:
- ``OptimizerConfig(name='fusedadam')`` (apex, fp32 master weights and moments, ``config/build.py``): torch
  ``adamw`` would keep moments and updates in bfloat16, where 84.5 % of lr-1e-5 updates on std-0.02 weights
  round to zero; ``entrypoints/sft.py::validate_config`` refuses ``adamw`` on this seat. Known limit of the
  bf16 storage (review R125-B-02): the parameters the forward reads are the bf16 copy the optimizer refreshes
  from its fp32 master each step, so an element whose accumulated update stays below half its bf16 ulp never
  changes -- not in the saved artifact AND not in the training forward, which reads the same copy (review
  R134-12): the final RMSNorm (median |w| 3.08, ulp 0.0156) changed in 0 of 2,048 elements over the 1,401
  steps of A1a and 36 of 882 tensors (all norm scales) stayed bitwise the base's, i.e. those 36 tensors did not
  train. Trade-off: storing the 1-D parameters (the 426 norm scales and biases, a few MB) in fp32 would
  un-freeze them at negligible memory cost and the export would round each once, losing at most half an ulp;
  the served model is bf16 either way. The limit is recorded, not fixed; the fp32-1-D-storage arm is owed
  before a long or a higher-lr run (``docs/guides/sft.md`` section 5).
- ``rl.max_length=16384``: the Codex training rows render 8.5-13k student tokens; the packer refuses any row
  over the budget instead of truncating it (research document section 5).
- ``images_kwargs.size.longest_edge=4194304``, ``shortest_edge=65536`` (pixel AREA): 2048 x 2048, the long-side
  cap of the Codex harness that produced the trajectories (its images are at most 2,048 px on the long side, up to
  4,194,304 px), so no training image is downscaled relative to what the agent saw; the earlier 4,147,200 (the
  largest image of one training set) was below that cap (review R125-A-06). The serve applies the checkpoint's own
  default, 16,777,216 px, so train and serve see the same grid for every image under the cap.
- ``checkpoint_format='consolidated'``: the HF materializer consumes consolidated saves only.
- ``micro_batch_size=1``, ``grad_accum=1``, ``max_steps=2``, ``save_every=1``, validation before training and
  every 50 steps: the 2-step canary of research document section 5; arms override these on the command line.

Objective-level knobs and the recorded decision per knob (review R125-D-01; written 2026-10-02). Terms:
- "the Cosmos post-training recipe" = the configuration the Cosmos team's own ``reasoner_posttraining``
  experiments of this backbone use in this tree: ``cosmos_framework/configs/base/experiment/
  reasoner_posttraining/_submit_posttrain.py`` entry ``pre_exp020_113_..._sqrt_lr1e5_lbl8e3_n128`` on the
  reasoner defaults ``configs/base/reasoner/defaults/optimizer.py`` (``VLM_OPTIMIZER_KWARGS``,
  ``VLM_LAMBDACOSINE_KWARGS``) and ``policy_config.py`` -- the lab's nearest record of how the base was
  post-trained;
- "A1 / A1a" = the two SFT arms of 2026-10-01 (jobs 734544 and 747502) that trained with every knob at its
  pre-existing framework default; the two differ in one field only, the validation cadence (A1 every 50 steps,
  A1a every step), so ``RECIPE_VARIANTS["a1"]`` and ``RECIPE_VARIANTS["a1a"]`` reproduce them and the arms of
  record stay re-runnable (``SFT_RECIPE_VARIANT=a1`` / ``a1a``; review R134-09);
- "the serving path of record" = how the lab's reads of record serve the student: Codex with ``--thinking high``
  -> a Responses request that carries ``reasoning.effort`` AND the harness's tools array (8 tools) on every call
  -> vLLM 0.24, which with a tools array present appends ``<|im_start|>assistant\n`` and NO ``<think>`` after the
  rendered history (review R138-01, measured live on the A1a deployment: the input token count is the same with and
  without the effort field, 157 = 157 for one synthetic tool (round 138 P4 / P7) and 7,862 = 7,862 for the 8-tool list
  with the wire instructions (review round 142 Q3 / Q4, ``review_r142/ev/02_live_probes.out``, the first live pair
  with the 8 tools; round 138's P3 measured the effort-present member live and its second 7,862 was the local
  rendering without ``<think>``, review R142-02); a Codex-shaped request counts 7,880 = the rendering without a
  ``<think>``; ``/tokenize`` with tools and ``enable_thinking: true`` ends in ``assistant\n``; the ids and both pairs
  are ``tests/fixtures/served_prefix_with_tools.json``);
  "the prefill" = the tokens the server appends after the rendered history: on the path of record
  ``<|im_start|>assistant\n`` (ids 151644, 77091, 198) and nothing more, so the student must emit ``<think>``
  itself; a ``<think>`` prefill exists only for requests WITHOUT tools, where a present effort sets
  ``enable_thinking`` (``vllm/entrypoints/openai/responses/protocol.py:314-315``) and the template appends
  ``<think>`` (10 input tokens where 9 are rendered without the flag; ``fix_r125/S/C04_generation_prompt.md``
  section 2, ``review_r133/10_live_probe_r133.out``) -- the measurement of rounds 133 / 134, which holds for
  tool-less requests only (review R134-01's ruling of a prefill on every turn is superseded); "a no-think turn"
  = an assistant turn whose completion carries no ``<think>...</think>`` span (45.5 % of the A1 file's turns);
- "pin" vs "fixture" (tests): a fixture FAILS on the parent commit and PASSES on the child (it proves a change);
  a pin passes on both and records a mechanism the change did not touch.

| knob | Cosmos post-training recipe | A1 / A1a | this recipe's default | why |
|---|---|---|---|---|
| MoE load-balancing loss | ``lbl.coeff 8e-3``, ``lbl.method global`` | none (the adapter discarded ``LBLMetadata``) | ``moe_lbl_coeff=8e-3``, ``moe_lbl_method='global'`` | the base was post-trained with it; without it the router (the parameter group that moved most in A1a, mean relative change 0.027) drifts unobserved; ``train/lbl_loss`` and ``train/moe_load_max_rel`` record it every step |
| learning-rate schedule | LambdaCosine: warmup 1,000, ``f_start 0.01``, ``f_min 0.1`` over ``max_iter`` | constant 1e-5, warmup 0 | ``lr_schedule.kind='cosine'``, ``f_start 0.01``, ``f_min 0.1``, cycle = ``max_steps``; ``optimizer.warmup_steps=1000`` | the base's schedule; with the full lr from step 1 the 1.0 clip set the step size on 73.5 % of A1's updates (pre-clip norm p50 1.10, p90 1.62, first steps 12.9-32.8); ``train/grad_clipped`` records the clip per step |
| Adam betas | (0.9, 0.95) | (0.9, 0.999) | ``betas=[0.9, 0.95]`` | the base's; the second moment averages about 20 updates instead of 1,000 |
| weight decay | 0.1 on every tensor: ``cosmos3/utils/optimizer.py::_build_params_with_metadata`` lines 326-331 give every parameter the base decay unless a skip pattern or ``disable_weight_decay_for_1d_params`` says otherwise, and the recipe entry (``_submit_posttrain.py`` lines 330-350) sets neither, so all 882 tensors decay | 0.01 on every tensor (``decay_groups=''``) | ``weight_decay=0.1``, ``decay_groups='qwen_moe'`` | the VALUE is the base's; the GROUPING is this seat's choice and matches neither reference (review R134-03): the backbone decays all 882 tensors; transformers 5.13 ``Trainer.get_decay_parameter_names`` exempts LayerNorm modules, biases and ``*norm*`` names only and DECAYS ``embed_tokens``, ``lm_head`` and every ``mlp.gate.weight``; ``qwen_moe`` exempts the 426 1-D tensors (193 text norm scales, 58 vision norm scales, 175 vision biases: the HF exemption) and additionally the embedding, the head and the 48 router gates, 50 tensors neither reference exempts. What that protects: a router gate's magnitude sets its layer's routing sharpness, and decay pulls every gate toward zero (flatter routing) independent of the data while the LBL term already pulls the shares toward uniform; the embedding and head norms set the vocabulary logit scale. What it costs: at the peak lr the exempted decay is lr x wd = 1e-6 of the weight per step (0.3 % over 2,847 steps, 0.8 % over the backbone's 8,200), so the effect is small either way, and a readout of this recipe against the base cannot attribute the difference to the base's recipe alone; the ``decay_groups=''`` arm at wd 0.1 (the backbone's grouping) is the ablation owed before the grouping is called a gain. Groups on the real names: 288 decayed 1x / 243 undecayed 1x / 118 decayed 0.1x / 233 undecayed 0.1x (``tests/test_config_sft.py``) |
| vision-tower learning rate | 0.1x (``lr_multipliers {"model.visual": 0.1}``) | 1.0x | ``lr_multipliers={'model.visual': 0.1}`` | the base's; the tower adapts at a tenth of the language model's rate |
| cross-entropy weighting | square-root per-token (``weighted_ce_exponent 0.5``) | token mean | ``ce_exponent=0.5`` | the base's: under the token mean a 3,651-token row weighed 5.0x a 725-token row of the same batch; here sqrt(5) = 2.2x |
| ``<think>`` opener | masked together with an empty block (the Nemo-Chat processor rule: its generation prompt supplies the prefix) | supervised | ``mask_think_opener=False`` | kept at the A1 value, for the stronger reason measured in review R138-01 (live on the A1a deployment, vLLM 0.24; it supersedes review R134-01's ruling of a prefill on every turn): on the serving path of record there is no prefill of the opener -- Codex sends the tools array on every call, and with tools present vLLM 0.24 appends ``<|im_start|>assistant\n`` and NO ``<think>`` (``reasoning.effort`` present or absent: 157 = 157 and 7,862 = 7,862 input tokens, round 138 P4 / P7 and review round 142 Q3 / Q4; ``/tokenize`` with tools + ``enable_thinking`` ends in ``assistant\n``; ids and both pairs in ``tests/fixtures/served_prefix_with_tools.json``) -- so the student must emit ``<think>`` itself as the first token of every turn, and the opener's loss is NECESSARY, not merely harmless. ``mask_think_opener=True`` would remove the loss from the one token the student has to produce first; what a ``True``-trained student would emit there is UNMEASURED with tools present (no student was trained with ``True``, and the first token after ``assistant\n`` with tools present was never sampled), so the consequence is a RISK, not a measurement (review R142-01): the failure it could reopen is C-04 (the lab's live probe record of the served generation prompt on the A1a deployment, 2026-10-02, outside the repo) section 3's measured trigger, quoted -- "The qwen3 parser treats an output that carries no ``</think>`` as reasoning in full: a no-think TEXT turn of the student never reaches Codex as an agent message" and "A no-think TOOL-CALL turn is unaffected" (vLLM 0.24 ``reasoning/basic_parsers.py`` 173-181, read in review round 142: without ``</think>`` the whole output is returned as reasoning; with ``</think>`` it is split even without ``<think>``) -- and under the exporter's rendering the closer ``\n\n</think>\n\n`` stays supervised under ``True``, so a turn that dropped only the opener would still carry ``</think>``. The decision does not rest on that risk: ``False`` stays the default because the opener's loss is the only training signal for a token the student must emit on the path of record. The ``<think>`` prefill exists only for requests without tools (a present effort -> ``enable_thinking`` -> the template appends ``<think>``; 10 input tokens vs 9; the measurement of rounds 133 / 134), which is not a path of record; under such a serve the supervised opener costs one deterministic token per turn and the student continues after the prefilled ``<think>``, so the rendering is right under both conditions. Consequence for the data: 45.5 % of A1a's assistant turns carry no think block; the exporter renders every no-think turn as ``<think>\n\n</think>\n\n`` (C-04 section 5, fix round 8b), under which the opener ``<think>`` AND the closer tokens ``\n\n``, ``</think>``, ``\n\n`` MUST stay supervised: the student opens and closes the empty block itself (fixture ``tests/test_config_sft.py::test_default_recipe_supervises_the_opener_the_student_emits_after_the_served_prefix_with_tools``). ``RECIPE_VARIANTS['prefilled_opener']`` masks the opener and only the opener; it is kept for a serve that prefills the opener (a tool-less request with the effort field or ``enable_thinking``) and is not the path of record: it must not be selected for a student served through Codex; selecting it sets the typed field ``rl.serve_contract=prefills_opener`` beside the mask (review threads b196e5e99f / 12f2ce563b: the acknowledgement is a field of the resolved config and of the objective identity, never an environment variable; ``SFTConfig`` refuses ``mask_think_opener=True`` under the default contract ``tools_no_prefill`` from any source, a ``--`` override included), and the resolved variant name is recorded in the attempt's provenance as ``recipe_variant`` (review R142-04; review R134-02: the earlier ``nemo_chat_opener`` rule also masked an empty block, which under a single-newline rendering ``<think>\n</think>`` removed the closer's loss; with the exporter's ``\n\n``, one token, it masked only the opener by accident). The packer applies either setting to the token stream of the base's own template |
| validation cadence | -- | every 50 steps (A1) / every step (A1a) | ``validation.every_steps=50`` | review R125-D-05: the canary default 1 validated 1,402 times in A1a; ``RECIPE_VARIANTS['a1']`` keeps 50, ``['a1a']`` sets 1 (review R134-09) |
| image budget | 2,048 visual tokens | 4,147,200 px | 4,194,304 px | review R125-A-06, above |
| bit-reproducible kernels | -- | off | ``rl.deterministic=False`` | ``True`` (cuDNN deterministic, deterministic algorithms, benchmark off) for paired readouts between arms: the same rows and weights gave ``val/loss_avg`` 0.5745 vs 0.5790 at step 0 of A1a and A1 (review R125-D-12); it costs throughput, so it is a per-arm choice, documented here |

Set ``SFT_MODEL_PATH`` to a complete local HF snapshot (``config.json`` with ``model_type`` ``qwen3_vl_moe`` or
``qwen3_vl`` plus the ``Qwen3VLProcessor`` sidecars). ``SFT_RECIPE_VARIANT`` selects a ``RECIPE_VARIANTS`` entry
(``default`` when unset); ``prefilled_opener`` sets ``rl.serve_contract=prefills_opener``, the typed declaration that the
student will be served by a path that prefills ``<think>`` for it, which the Codex-served path of record does not
(review R142-04; review threads b196e5e99f / 12f2ce563b: ``SFTConfig`` refuses the mask under any other contract and no
environment variable is read), and the resolved variant name is stamped into the attempt's provenance as
``recipe_variant`` (``utils/provenance.py``). Torchrun chooses the DP world; no GPU allocation is made by this file.
"""

from __future__ import annotations

import os
from collections.abc import Callable

from cosmos_framework.posttrain.config.config_sft import PosttrainSFTConfig
from cosmos_framework.posttrain.config.config_sft import make_config as make_root_config
from cosmos_framework.posttrain.config.model import (
    NativePolicyBuildSpec,
    NativePolicyParallelismConfig,
    Qwen3VLModelConfig,
)
from cosmos_framework.posttrain.config.rl import OptimizerConfig
from cosmos_framework.posttrain.config.sft import (
    SFT_SERVE_CONTRACT_DEFAULT,
    SFT_SERVE_CONTRACT_PREFILLS_OPENER,
    SFTDatasetConfig,
    SFTLRScheduleConfig,
    SFTValidationConfig,
)
from cosmos_framework.posttrain.utils.provenance import record_recipe_variant

NATIVE_POLICY_BUILDER = "cosmos_framework.posttrain.model.qwen3_vl.build_qwen3_vl_native_policy"

# The processor-resized pixel AREA budget: 2048 x 2048, the Codex harness's long-side cap (review R125-A-06).
IMAGE_LONGEST_EDGE_PX = 4_194_304
IMAGE_SHORTEST_EDGE_PX = 65_536


def _variant_a1(config: PosttrainSFTConfig) -> None:
    """The A1 arm of 2026-10-01 (job 734544): every objective-level knob at its pre-existing framework default;
    validation every 50 steps (its ``config_resolved.yaml``), which is this recipe's default too."""
    config.rl.optimizer = OptimizerConfig(
        name="fusedadam", lr=1e-5, weight_decay=0.01, grad_clip_norm=1.0, warmup_steps=0
    )
    config.rl.lr_schedule = SFTLRScheduleConfig(kind="constant")
    config.rl.ce_exponent = 0.0
    config.rl.mask_think_opener = False
    config.rl.moe_lbl_coeff = 0.0
    config.rl.validation.every_steps = 50
    config.rl.processor_kwargs["images_kwargs"]["size"]["longest_edge"] = 4_147_200


def _variant_a1a(config: PosttrainSFTConfig) -> None:
    """The A1a arm of 2026-10-01 (job 747502): A1's knobs with validation every step (review R134-09: the one field the
    two arms' resolved configs differ in)."""
    _variant_a1(config)
    config.rl.validation.every_steps = 1


def _variant_prefilled_opener(config: PosttrainSFTConfig) -> None:
    r"""This recipe's defaults with the ``<think>`` opener unsupervised. NOT the path of record: it fits only a serve
    that prefills ``<think>`` for the student (a request without tools, whose effort field or ``enable_thinking`` makes
    the template append ``<think>``; 10 input tokens vs 9), which the Codex-served path of record does not: there
    (tools present) vLLM 0.24 prefills no opener and the student must emit ``<think>`` itself (knob 7, review R138-01).
    What a ``True``-trained student would emit as that first token is UNMEASURED with tools present (no such student
    exists; the first token after ``assistant\n`` with tools present was never sampled); the RISK is C-04 section 3's
    measured trigger (C-04: the lab's live probe record of the served generation prompt on the A1a deployment,
    2026-10-02, outside the repo; primary pointer vLLM 0.24 ``reasoning/basic_parsers.py`` 173-181: ``extract_reasoning``
    returns all-reasoning only when ``</think>`` is absent), "an output that carries no ``</think>``" is returned as a
    reasoning item (a no-think TEXT turn
    never reaches Codex; tool-call turns unaffected), while the closer ``\n\n</think>\n\n`` of an empty block stays
    supervised under this variant (review R142-01). Guard (review R142-04, re-shaped by review threads b196e5e99f /
    12f2ce563b): the variant sets ``rl.serve_contract=prefills_opener`` beside the mask, the typed declaration that the
    student WILL be served by a path that prefills the opener; ``SFTConfig`` refuses ``mask_think_opener=True`` under any
    other contract, the field is in the resolved config and in the objective identity, and the resolved variant name is
    recorded in provenance (``recipe_variant``).
    """
    config.rl.serve_contract = SFT_SERVE_CONTRACT_PREFILLS_OPENER
    config.rl.mask_think_opener = True


RECIPE_VARIANTS: dict[str, Callable[[PosttrainSFTConfig], None]] = {
    "default": lambda config: None,
    "a1": _variant_a1,
    "a1a": _variant_a1a,
    "prefilled_opener": _variant_prefilled_opener,
}


def build_config(variant: str | None = None) -> PosttrainSFTConfig:
    variant = variant if variant is not None else os.environ.get("SFT_RECIPE_VARIANT", "default")
    if variant not in RECIPE_VARIANTS:
        raise ValueError(f"SFT_RECIPE_VARIANT={variant!r} must be one of {sorted(RECIPE_VARIANTS)}")
    config = make_root_config()
    model_id = os.environ.get("SFT_MODEL_PATH", "Qwen/Qwen3-VL-30B-A3B-Instruct")
    config.job.project = "cosmos3_reasoner"
    config.job.group = "qwen3_vl_moe_sft_example"
    config.job.name = "native_seat_canary_seed42"
    config.job.wandb_mode = "disabled"
    config.model_backend = "i4"
    config.model = NativePolicyBuildSpec(
        builder=NATIVE_POLICY_BUILDER,
        model_id=model_id,
        model_config=Qwen3VLModelConfig(
            # NativePolicyBuildSpec.model_id is the single artifact identity; the builder injects it.
            name_or_path="",
            trust_remote_code=False,
            checkpoint_path="",
            credential_path="",
            dtype="bfloat16",
            attn_impl="sdpa",
            activation_checkpointing="full",
            fsdp_reduce_dtype="float32",
        ),
        parallelism=NativePolicyParallelismConfig(dp_replicate=1, dp_shard=-1),
        requires_collated_policy_forward=True,
    )
    config.rl.seed = 42
    config.rl.deterministic = False
    config.rl.master_dtype = "bfloat16"
    config.rl.param_dtype = "bfloat16"
    config.rl.reduce_dtype = "float32"
    config.rl.mixed_precision_mode = "fsdp"
    config.rl.max_steps = 2
    config.rl.micro_batch_size = 1
    config.rl.grad_accum = 1
    config.rl.max_length = 16384
    config.rl.checkpoint_format = "consolidated"
    config.rl.save_async = False
    config.rl.save_every = 1
    # The optimizer of the Cosmos post-training recipe: AdamW lr 1e-5, betas (0.9, 0.95), weight decay 0.1 with
    # the qwen_moe groups, the vision tower at 0.1x, 1,000 linear warmup steps into the cosine schedule below.
    config.rl.optimizer = OptimizerConfig(
        name="fusedadam",
        lr=1e-5,
        betas=[0.9, 0.95],
        weight_decay=0.1,
        decay_groups="qwen_moe",
        lr_multipliers={"model.visual": 0.1},
        grad_clip_norm=1.0,
        warmup_steps=1000,
    )
    config.rl.lr_schedule = SFTLRScheduleConfig(kind="cosine", f_start=0.01, f_min=0.1, cycle_steps=0)
    config.rl.ce_exponent = 0.5
    config.rl.mask_think_opener = False
    config.rl.serve_contract = SFT_SERVE_CONTRACT_DEFAULT
    config.rl.moe_lbl_coeff = 8e-3
    config.rl.moe_lbl_method = "global"
    data_root = os.environ.get("SFT_DATA_ROOT", ".ene/scratch/sft-example")
    config.rl.data = SFTDatasetConfig(path=os.path.join(data_root, "train.jsonl"))
    config.rl.validation = SFTValidationConfig(
        enabled=True,
        before_training=True,
        every_steps=50,
        batch_size=1,
        seed=42,
        data=SFTDatasetConfig(path=os.path.join(data_root, "validation.jsonl"), shuffle=False),
    )
    config.rl.processor_kwargs = {
        "images_kwargs": {"size": {"shortest_edge": IMAGE_SHORTEST_EDGE_PX, "longest_edge": IMAGE_LONGEST_EDGE_PX}},
        # Video rows (SOP) are not in the first canary; the dense recipe's budget is kept for them.
        "videos_kwargs": {"size": {"shortest_edge": 4096, "longest_edge": 65536}, "num_frames": 8, "fps": None},
    }
    RECIPE_VARIANTS[variant](config)
    # The resolved NAME goes into the attempt's provenance (review R142-04): the resolved config carries the values only.
    record_recipe_variant(variant)
    return config


def make_config() -> PosttrainSFTConfig:
    """The cosmos_framework.utils.config.load_config entry point."""
    return build_config()
