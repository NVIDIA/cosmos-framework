# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Built-in capability registrations for the DPO and GRPO recipes.

Descriptors are owned by the impl_key: the DPO env produces a preference `targets` field and the DPO
trainer consumes it, both on the DATA plane. The `'targets'` kind derives from `SUPERVISION_KINDS`,
so it tracks the real `Preference` type rather than a free-authored literal.

These registrations belong with the env/trainer/reward impls (capabilities are impl-owned) and will
move there when the algorithm packages are reorganized; centralized here for now as the single
registration trigger the recipe imports.
"""

from __future__ import annotations

from cosmos_framework.posttrain.control.capability import (
    SUPERVISION_KINDS,
    CapabilityDescriptor,
    FieldSpec,
    register_capability,
)
from cosmos_framework.posttrain.control.manifest import Plane
from cosmos_framework.posttrain.schema import FINAL_REWARD_KEY, Preference, SFTTarget, TeacherSignal

_PREFERENCE_FIELD = FieldSpec("targets", SUPERVISION_KINDS[Preference])

register_capability(
    "dpo_env",
    CapabilityDescriptor(produces=(_PREFERENCE_FIELD,), planes=frozenset({Plane.DATA})),
)
register_capability(
    "dpo_trainer",
    CapabilityDescriptor(consumes=(_PREFERENCE_FIELD,), planes=frozenset({Plane.DATA})),
)

# Offline SFT uses the supervised channel, never synthetic rewards or advantages.
_SFT_FIELD = FieldSpec("targets", SUPERVISION_KINDS[SFTTarget])
register_capability("sft_env", CapabilityDescriptor(produces=(_SFT_FIELD,), planes=frozenset({Plane.DATA})))
register_capability("sft_trainer", CapabilityDescriptor(consumes=(_SFT_FIELD,), planes=frozenset({Plane.DATA})))

# --------------------------------------------------------------------------- GRPO
# The observation fields a prompt source emits and a rule reward may declare in its `needs`; an
# unregistered need raises here rather than registering a reward the check under-constrains.
# The canonical observation keys. Named for their ROLE, not a modality: a prompt may be a bare
# string or a multimodal conversation, and a reference may be text, a label, or media.
_PROMPT = FieldSpec("prompt", "prompt")
_GROUND_TRUTH = FieldSpec(
    "reference",
    "ground_truth",
    remedy=("name it in rl.data.response_column, or select a reward provider whose needs omit 'reference'"),
)
_COMPLETION = FieldSpec("completion_text", "completion")
_FINAL_REWARD = FieldSpec(FINAL_REWARD_KEY, "reward")
_OBS_FIELDS = {_PROMPT.name: _PROMPT, _GROUND_TRUTH.name: _GROUND_TRUTH}
_DATA = frozenset({Plane.DATA})
_DATA_AND_WEIGHT = frozenset({Plane.DATA, Plane.WEIGHT})

for _key, _produces in (
    ("grpo_env_arithmetic", (_PROMPT, _GROUND_TRUTH)),  # synthetic: answer generated with the prompt
    ("grpo_env_jsonl", (_PROMPT,)),
    ("grpo_env_jsonl_with_answers", (_PROMPT, _GROUND_TRUTH)),
    # ANY HF dataset (openai/gsm8k, and anything else the columns or an items_fn can express).
    # The distinguishing fact was never which dataset it is -- it is whether the source ships a
    # reference answer, which is what lets a verifiable reward pass the compatibility check.
    ("grpo_env_hf", (_PROMPT,)),
    # NOTE: `grpo_env_hf` (the no-ground-truth variant) is NOT reachable today. `DatasetConfig`
    # requires `response_column` whenever `name` is set, so `prompt_source` returns ground_truth=True
    # for every HF source and the `_with_answers` key below always wins. Kept deliberately: a
    # prompt-only HF source is the shape a non-verifiable reward needs. A CHOICE, not an oversight.
    ("grpo_env_hf_with_answers", (_PROMPT, _GROUND_TRUTH)),
    # Custom sources enter the same canonical prompt/reference view as JSONL and HF sources. Their
    # storage factory changes construction, not the fields declared to downstream roles.
    ("grpo_env_custom", (_PROMPT,)),
    ("grpo_env_custom_with_answers", (_PROMPT, _GROUND_TRUTH)),
):
    register_capability(_key, CapabilityDescriptor(produces=_produces, planes=_DATA))

# A rollout carries its input observation through to each trajectory and adds the decoded completion.
# Two descriptors make the source/reward compatibility check honest: an answer-bearing source selects
# the carry-through variant, while a prompt-only source cannot appear to manufacture ground truth.
register_capability(
    "grpo_rollout",
    CapabilityDescriptor(consumes=(_PROMPT,), produces=(_PROMPT, _COMPLETION), planes=_DATA_AND_WEIGHT),
)
register_capability(
    "grpo_rollout_with_answers",
    CapabilityDescriptor(
        consumes=(_PROMPT, _GROUND_TRUTH),
        produces=(_PROMPT, _GROUND_TRUTH, _COMPLETION),
        planes=_DATA_AND_WEIGHT,
    ),
)

_REWARD_CAPABILITY_KEYS: dict[frozenset[str], str] = {
    frozenset(): "grpo_reward_completion",
    frozenset({"prompt"}): "grpo_reward_prompt",
    frozenset({"reference"}): "grpo_reward_reference",
    frozenset({"prompt", "reference"}): "grpo_reward_prompt_reference",
}


def grpo_reward_capability_key(needs: tuple[str, ...]) -> str:
    """Resolve one of the bounded canonical reward capability contracts."""
    if len(needs) != len(set(needs)):
        raise ValueError(f"reward provider needs must be unique, got {needs!r}")
    key = frozenset(needs)
    if key not in _REWARD_CAPABILITY_KEYS:
        unknown = sorted(key.difference(_OBS_FIELDS))
        raise ValueError(
            f"reward provider needs unsupported observation fields {unknown}; supported: {sorted(_OBS_FIELDS)}"
        )
    return _REWARD_CAPABILITY_KEYS[key]


# Custom provider names never mutate the process-global capability registry. Four bounded descriptors
# cover every supported subset of the canonical prompt/reference observation contract.
for _needs, _key in _REWARD_CAPABILITY_KEYS.items():
    register_capability(
        _key,
        CapabilityDescriptor(
            consumes=(_COMPLETION, *(tuple(_OBS_FIELDS[name] for name in _OBS_FIELDS if name in _needs))),
            produces=(_FINAL_REWARD,),
            planes=_DATA,
        ),
    )

register_capability(
    "grpo_buffer",
    CapabilityDescriptor(consumes=(_FINAL_REWARD,), produces=(_FINAL_REWARD,), planes=_DATA),
)
register_capability("grpo_trainer", CapabilityDescriptor(consumes=(_FINAL_REWARD,), planes=_DATA_AND_WEIGHT))


# --------------------------------------------------------------------------- OPD
# The teacher consumes a scored trajectory and adds the distillation signal on `targets`. The same
# trajectory, including its final reward, continues on the DATA plane. Algorithmic credit assignment
# is distinct from the transport engine's flow-control credits. The field kind derives from
# `SUPERVISION_KINDS[TeacherSignal]`, so it tracks the real dataclass rather than a literal that could
# drift from it.
_TEACHER_SIGNAL = FieldSpec("targets", SUPERVISION_KINDS[TeacherSignal])

register_capability(
    "opd_teacher_vllm",
    CapabilityDescriptor(
        # `final` is the immediate-edge input contract, matching `grpo_buffer`/`grpo_trainer`. The
        # teacher also reads completion token ids carried through from rollout, but the reward role
        # does not declare those as outputs. The teacher returns the same trajectory, so it re-emits
        # `final` while adding the new signal.
        consumes=(_FINAL_REWARD,),
        produces=(_FINAL_REWARD, _TEACHER_SIGNAL),
        planes=_DATA,
    ),
)
# The buffer and trainer carry the signal through on the OPD path; separate impl_keys from their
# GRPO namesakes so a manifest cannot claim a GRPO buffer transports a teacher signal it never saw.
register_capability(
    "opd_buffer",
    CapabilityDescriptor(
        consumes=(_FINAL_REWARD, _TEACHER_SIGNAL),
        produces=(_FINAL_REWARD, _TEACHER_SIGNAL),
        planes=_DATA,
    ),
)
register_capability(
    "opd_trainer",
    CapabilityDescriptor(
        consumes=(_FINAL_REWARD, _TEACHER_SIGNAL),
        planes=_DATA_AND_WEIGHT,
    ),
)


# --------------------------------------------------------------------------- PPO
# PPO enriches the same Trajectory in several model roles. The v1 capability schema is role-wide,
# not method/port-specific, so the actor-critic target return edge declares the shared scored
# trajectory envelope here; exact reference/value/target annotations remain role-local validation.
register_capability("ppo_env", CapabilityDescriptor(produces=(_PROMPT,), planes=_DATA))
register_capability(
    "ppo_env_with_answers",
    CapabilityDescriptor(produces=(_PROMPT, _GROUND_TRUTH), planes=_DATA),
)
register_capability(
    "ppo_rollout",
    CapabilityDescriptor(consumes=(_PROMPT,), produces=(_PROMPT, _COMPLETION), planes=_DATA_AND_WEIGHT),
)
register_capability(
    "ppo_rollout_with_answers",
    CapabilityDescriptor(
        consumes=(_PROMPT, _GROUND_TRUTH),
        produces=(_PROMPT, _GROUND_TRUTH, _COMPLETION),
        planes=_DATA_AND_WEIGHT,
    ),
)
register_capability(
    "ppo_reward",
    CapabilityDescriptor(consumes=(_PROMPT, _COMPLETION), produces=(_FINAL_REWARD,), planes=_DATA),
)
register_capability(
    "ppo_reward_with_answers",
    CapabilityDescriptor(consumes=(_PROMPT, _COMPLETION, _GROUND_TRUTH), produces=(_FINAL_REWARD,), planes=_DATA),
)
for _ppo_key in ("ppo_buffer", "ppo_reference", "ppo_critic"):
    register_capability(
        _ppo_key,
        CapabilityDescriptor(consumes=(_FINAL_REWARD,), produces=(_FINAL_REWARD,), planes=_DATA),
    )
register_capability(
    "ppo_trainer",
    CapabilityDescriptor(consumes=(_FINAL_REWARD,), produces=(_FINAL_REWARD,), planes=_DATA_AND_WEIGHT),
)


# --------------------------------------------------------------------------- embodied sim-eval
# The sim role and the policy it queries. ONE field crosses this edge -- the observation -- and
# that is not under-specification, it is the layering. The per-key contract (cameras, proprio,
# extrinsics) is `obs_spec()` + `check_obs()` in `embodied/embodiment.py`, enforced inside the
# loop on every unit; a second copy of the key space here is how the two come to disagree.
_OBSERVATION = FieldSpec(
    "observation",
    "observation",
    remedy="the sim role must emit the embodiment's declared observation; see embodied/embodiment.py::obs_spec",
)

#: ONE NAME PER IMPLEMENTATION, ACROSS THREE REGISTRIES. An ``impl_key`` is meant to resolve
#: the same thing in all of them: the ``CapabilityDescriptor`` here, the ``RoleDecl`` in
#: ``control/recipe.py::sim_manifest``, and the body builder in ``roles/core/registry.py``.
#: Spelling it three times as a literal is what lets those drift -- a manifest can name a role
#: whose capability is registered under a near-miss and ``bind_check`` then validates a
#: different contract than the one the run builds. Named here, next to the descriptor, because
#: this is the module every recipe already imports for its side effects.
SIM_IMPL_KEY = "behavior1k_sim"
POLICY_IMPL_KEY = "cosmos3_policy"

register_capability(
    SIM_IMPL_KEY,
    CapabilityDescriptor(produces=(_OBSERVATION,), planes=_DATA),
)
register_capability(
    POLICY_IMPL_KEY,
    CapabilityDescriptor(consumes=(_OBSERVATION,), planes=_DATA),
)

# --------------------------------------------------------------------------- online WFM NFT
_WFM_CONDITIONING = FieldSpec("wfm_conditioning", "wfm_conditioning")
_WFM_LATENT = FieldSpec("action", "video_latent")
_WFM_ADVANTAGE = FieldSpec("advantage", "advantage")

register_capability("nft_env", CapabilityDescriptor(produces=(_WFM_CONDITIONING,), planes=_DATA))
register_capability(
    "nft_rollout",
    CapabilityDescriptor(
        consumes=(_WFM_CONDITIONING,),
        produces=(_WFM_CONDITIONING, _WFM_LATENT),
        planes=_DATA_AND_WEIGHT,
    ),
)
register_capability(
    "nft_reward",
    CapabilityDescriptor(
        consumes=(_WFM_CONDITIONING, _WFM_LATENT),
        produces=(_WFM_CONDITIONING, _WFM_LATENT, _FINAL_REWARD, _WFM_ADVANTAGE),
        planes=_DATA,
    ),
)
register_capability(
    "nft_buffer",
    CapabilityDescriptor(
        consumes=(_WFM_CONDITIONING, _WFM_LATENT, _FINAL_REWARD, _WFM_ADVANTAGE),
        produces=(_WFM_CONDITIONING, _WFM_LATENT, _FINAL_REWARD, _WFM_ADVANTAGE),
        planes=_DATA,
    ),
)
register_capability(
    "nft_trainer",
    CapabilityDescriptor(
        consumes=(_WFM_CONDITIONING, _WFM_LATENT, _FINAL_REWARD, _WFM_ADVANTAGE),
        planes=_DATA_AND_WEIGHT,
    ),
)


# Joint actor/value PPO: environment decisions never advertise token fields.
_JOINT_OBSERVATION = FieldSpec("observation", "object")
_JOINT_DECISION = FieldSpec("decision_feedback", "object")
for _key, _consumes, _produces, _planes in (
    ("joint_ppo_sim", (), (_JOINT_OBSERVATION,), _DATA),
    ("joint_ppo_env", (_JOINT_OBSERVATION,), (_JOINT_OBSERVATION, _JOINT_DECISION), _DATA),
    ("joint_ppo_rollout", (_JOINT_OBSERVATION,), (), _DATA_AND_WEIGHT),
    ("joint_ppo_reward", (_JOINT_DECISION,), (_JOINT_DECISION, _FINAL_REWARD), _DATA),
    ("joint_ppo_buffer", (_JOINT_DECISION, _FINAL_REWARD), (_JOINT_DECISION, _FINAL_REWARD), _DATA),
    ("joint_ppo_trainer", (_JOINT_DECISION, _FINAL_REWARD), (), _DATA_AND_WEIGHT),
):
    register_capability(_key, CapabilityDescriptor(consumes=_consumes, produces=_produces, planes=_planes))
