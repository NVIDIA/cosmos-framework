# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""HuggingFace rollout body (DESIGN §1.2 body=(a) local compute).

`HFRollout`: HF `generate` with group sampling. In the training loop the POLICY pushes weights
into this PASSIVE receiver (producer-driven, design 2026-07-27 §2): `apply_bucket` loads each
pushed bucket and the controller's prepare/commit barrier publishes the version. `apply_weights` is the KEPT
consumer-pull surface (manual / NCCL rigs), NOT the loop entry, and a no-op only when trainer &
rollout share one model instance. A real inference engine (SGLang/vLLM/TRT-LLM, #6) drops in as a
sibling variant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from cosmos_framework.posttrain.config.build import replica_coords
from cosmos_framework.posttrain.roles.rollout.base import (
    ShardSpec,
    WeightReceiverBase,
    group_ids_for_prompt,
)
from cosmos_framework.posttrain.roles.rollout.validation import (
    ValidationRolloutContext,
    distributed_validation_rng,
    normalize_validation_semantics_identity,
    rollout_validation_semantics_identity,
    run_validation_stage,
    seed_validation_rng,
    validation_generation_seed,
    validation_prompt_seed,
    validation_semantics_equal,
)
from cosmos_framework.posttrain.schema import ObsBundle, Status, Trajectory, Transition

if TYPE_CHECKING:
    from cosmos_framework.posttrain.data.packer.base import DataPacker


def _trim_at_stop(comp_ids, stop_ids: "set[int]"):
    """Cut a completion at (and including) its FIRST stop token, dropping `generate`'s right padding.

    The stop token itself is KEPT — it is a real generated token the policy chose, and dropping it
    would train a model that never learns to end its own generation. Only the padding that follows
    is dropped.

    NOTE: an earlier revision of this docstring claimed cosmos-rl "likewise trains on it". MEASURED
    FALSE on its default path. vLLM emits the stop token in `token_ids` but omits it from `.text`
    unless BOTH `skip_special_tokens=False` AND `include_stop_str_in_output=True` are set; cosmos-rl
    sets neither on its rollout SamplingParams, and its `DecoderOnlyLLMDataPacker` re-tokenizes the
    TEXT. So cosmos-rl trains one token shorter than we do on every stopped completion. That single
    token was the entire residual loss gap between the two stacks — see
    docs/design/grpo-cosmos-rl-parity.md.

    Args:
        comp_ids: 1-D tensor of completion token ids, right-padded by `generate`.
        stop_ids: token ids that terminate a completion (eos + pad, plus any extra eos the
            generation config declares — Qwen has two: <|im_end|> and <|endoftext|>).

    Returns:
        The completion up to and including the first stop token, or `comp_ids` unchanged when the
        sequence ran to `max_new_tokens` without stopping.
    """
    for i, t in enumerate(comp_ids.tolist()):
        if t in stop_ids:
            return comp_ids[: i + 1]
    return comp_ids


def _require_identical_validation_prompt(
    *,
    sample_id: str,
    encoded: Any,
    sampling: dict[str, Any],
    pad_token_id: int | None,
    validation: ValidationRolloutContext,
) -> None:
    """Prove every HF/FSDP rank will enter generation with identical model inputs."""
    from cosmos_framework.posttrain.utils.rank_state import gather_rank_states

    items = getattr(encoded, "items", None)
    if not callable(items):
        raise TypeError(f"HF validation tokenizer output must be a mapping, got {type(encoded).__name__}")
    model_inputs: dict[str, Any] = {}
    for key, value in sorted(items()):
        if not isinstance(key, str):
            raise TypeError(f"HF validation tokenizer output key must be str, got {key!r}")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"HF validation tokenizer output {key!r} must be a tensor, got {value!r}")
        local = value.detach().cpu()  # [*model input]
        model_inputs[key] = {
            "dtype": str(local.dtype),
            "shape": list(local.shape),
            "values": local.tolist(),
        }
    identity = normalize_validation_semantics_identity(
        {
            "sample_id": sample_id,
            "model_inputs": model_inputs,
            "sampling": sampling,
            "pad_token_id": pad_token_id,
            "validation_seed": validation.seed,
            "n_generation": validation.n_generation,
        },
        context="HF validation request identity",
    )
    by_rank = gather_rank_states(
        identity,
        context="HF validation request identity preparation",
    )
    if any(not validation_semantics_equal(candidate, identity) for candidate in by_rank.values()):
        raise RuntimeError(f"HF validation request differs across FSDP ranks: {by_rank!r}")


class HFRollout(WeightReceiverBase):
    """HF `generate` rollout with group sampling. Shares its model instance with LLMTrainer when
    colocated (DESIGN §7); receives pushed weights as a passive producer-driven receiver
    (`apply_bucket` + rollout-wide version prepare/commit, design 2026-07-27 §2)."""

    def __init__(
        self,
        model: Any,
        *,
        packer: "DataPacker",
        sampling: dict | None = None,
        capture_logprobs: bool = False,
        model_identifier: str | None = None,
        backend_config: dict[str, Any] | None = None,
    ) -> None:
        """Bind a model, its packer and its sampling overrides to this body.

        Everything after ``model`` is KEYWORD-ONLY, so the meaning of a call never depends on
        argument order (the 2nd positional slot has changed meaning more than once as this seam
        landed). A stale positional call raises immediately instead of silently mis-binding.

        Args:
            model: the HF model this rollout generates with (shared with LLMTrainer when colocated).
            packer: REQUIRED (keyword-only) DataPacker — the single prompt-shaping path and the owner
                of the tokenizer. It is required HERE, not at ``generate``, because it is decidable at
                construction: a packer-less rollout would bind and spawn every Ray actor fine, then
                die on the first generate() with the whole cluster already up. Weight-only rigs
                construct one explicitly (any tokenizer will do — they never generate).
            sampling: overrides for the default HF ``generate`` kwargs
                (do_sample/temperature/top_p/top_k/repetition_penalty/max_new_tokens/...). Pass every
                knob that matters — whatever is absent is inherited from the checkpoint's
                generation_config.json, NOT from a framework default.
            capture_logprobs: record per-token pi_old in ``policy_info["rollout_logprobs"]``.
                OFF by default because it is EXPENSIVE and usually unread: it forces
                ``output_scores=True``, so `generate` retains a `[group, vocab]` score tensor for
                every generated step and the full-vocab log_softmax duplicates it — for
                Qwen2.5-3B at group=16/max_tokens=512 that is ~10 GB per prompt. The clipped
                objective reads recorded values for the ``train/logprob_gap_*`` diagnostics and, only
                under ``GRPOConfig.use_rollout_logprobs_for_loss`` (default False, matching
                cosmos-rl), as pi_old; paying for them by default made the documented
                `--rollout hf` recipe OOM for nothing. The values are the log-softmax of the
                PROCESSED scores, so away from temperature 1 they are not the raw logprobs the
                vLLM rollout records.
            model_identifier: Explicit immutable model artifact identifier. Normally inferred from
                ``model.config``; custom wrappers must pass it when the live config has no source id.
            backend_config: Explicit finite JSON model/generation configuration for a custom HF
                wrapper. Standard ``PreTrainedModel`` instances derive it from their live configs.

        Raises:
            TypeError: ``packer`` was omitted (or a stale positional argument was passed).
            ValueError: ``packer`` was passed explicitly as None.
        """
        if packer is None:  # explicit packer=None: a bare TypeError would not say what to do
            raise ValueError(
                "HFRollout requires a DataPacker at construction — build one with "
                "init_data_packer(name_or_path) and pass HFRollout(model, packer=..., sampling=...)"
            )
        self.model = model
        self.packer = packer
        # The packer owns the tokenizer (rollout-side shaping seam, design 2026-07-28).
        self.tok = packer.tokenizer
        # top_k/repetition_penalty are spelled out rather than left off — an omitted knob inherits
        # from the model object, not from a framework default (`RolloutConfig.top_k`). This fallback
        # is for the weight-only and toy rigs; config-driven callers pass `hf_sampling_params(...)`.
        self.sampling = sampling or {
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": None,
            "repetition_penalty": 1.0,
            "max_new_tokens": 16,
        }
        self.capture_logprobs = bool(capture_logprobs)
        self._validation_model_identifier = model_identifier
        self._validation_backend_config = None if backend_config is None else dict(backend_config)
        self._weight_version = 0
        # Stop tokens used to trim `generate`'s right padding (see `_trim_at_stop`). eos + pad, plus
        # any EXTRA eos the model's generation config declares as a list — Qwen ships two
        # (<|im_end|> stops a chat turn, <|endoftext|> is also the pad), and trimming on only one of
        # them leaves the other as scored padding.
        # Tolerates `packer.tokenizer is None`: the weight-only rigs (NCCL/loopback sync tests) bind a
        # stub packer to stay HF-free and never call generate(), so there is nothing to trim.
        stop: set[int] = set()
        tok = packer.tokenizer
        if tok is not None:
            stop |= {i for i in (tok.eos_token_id, tok.pad_token_id) if i is not None}
        gen_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
        if isinstance(gen_eos, int):
            stop.add(gen_eos)
        elif gen_eos is not None:
            stop.update(int(i) for i in gen_eos)
        self._stop_ids = stop

    def validation_semantics_identity(self) -> dict[str, Any]:
        """Bind live HF model, tokenizer, packer, and generation semantics for exact resume."""
        backend_config = self._validation_backend_config
        model_config = getattr(self.model, "config", None)
        if backend_config is None:
            model_to_dict = getattr(model_config, "to_dict", None)
            generation_config = getattr(self.model, "generation_config", None)
            generation_to_dict = getattr(generation_config, "to_dict", None)
            if not callable(model_to_dict) or not callable(generation_to_dict):
                raise TypeError(
                    "HF validation identity requires live model.config.to_dict() and "
                    "model.generation_config.to_dict(), or an explicit backend_config"
                )
            backend_config = {
                "model_config": model_to_dict(),
                "generation_config": generation_to_dict(),
            }
        backend_config = normalize_validation_semantics_identity(
            backend_config,
            context="HF validation backend config",
        )
        model_identifier = self._validation_model_identifier
        if model_identifier is None:
            for owner in (model_config, self.model):
                for field in ("_name_or_path", "name_or_path"):
                    candidate = getattr(owner, field, None)
                    if isinstance(candidate, str) and candidate.strip():
                        model_identifier = candidate
                        break
                if model_identifier is not None:
                    break
        return rollout_validation_semantics_identity(
            backend="huggingface",
            model=self.model,
            packer=self.packer,
            sampling=dict(self.sampling),
            backend_config=backend_config,
            model_identifier="" if model_identifier is None else model_identifier,
            extra={"stop_token_ids": sorted(self._stop_ids)},
        )

    def apply_weights(self, source: "Any | None" = None) -> None:
        # PEER-ONLY surface: ``source`` is the policy peer (what weight_sync(rollout, policy)
        # sends); pull_stream_from_peer streams the sync per bucket, transport from the bound
        # mover. None keeps the degenerate contracts (no backend / SharedMover => stamp-only).
        # The WeightPayload batch form was REMOVED — drive WeightSync.apply/export directly for
        # manual whole-model syncs (as the loopback/NCCL rigs do).
        from cosmos_framework.posttrain.comm.weight_transfer import pull_stream_from_peer

        return pull_stream_from_peer(self, source)

    def prepare_recv(self) -> dict[str, ShardSpec]:
        # v1: no rollout-side sharding — receive each param whole, dim 0.
        return {k: ShardSpec(0, (tuple(v.shape)[0] if v.dim() else 0), 0) for k, v in self.model.state_dict().items()}

    shard_map = prepare_recv

    # --- streaming weight sync (per-bucket); set_bucket_plan + check_plan_coverage on WeightReceiverBase ---
    def apply_bucket(self, payload, i: int) -> None:
        # Producer-driven landing: fetch (import) + load ONLY bucket i in place (strict=False partial).
        # PURE LOAD — readiness and publication are separate from the per-bucket sink.
        be = getattr(self, "_weight_backend", None)
        if be is not None:
            be.apply_bucket(self.model, i, payload)

    @torch.no_grad()
    def generate(self, prompts: list[ObsBundle], **knobs: Any) -> list[Trajectory]:
        # Per-call knobs ride in **knobs so all rollout bodies share ONE `generate` surface
        # (roles/rollout/base.py). `group`: completions per prompt (GRPO group size). `seed`
        # (optional): seed the sampling RNG for reproducibility — under FSDP the ranks of one
        # replica share the model and MUST sample identically to keep the forward all-gathers
        # aligned, so passing the same seed to every rank guarantees that (None -> non-deterministic).
        validation = knobs.get("validation")
        if validation is not None:
            return self._generate_validation(prompts, validation)
        group = int(knobs.get("group", 8))
        seed = knobs.get("seed")
        sampling = dict(self.sampling)
        capture_logprobs = self.capture_logprobs
        # No packer check here: __init__ makes it REQUIRED, so self.packer/self.tok are always set.
        device = next(self.model.parameters()).device
        if seed is not None:
            # Replica-local deterministic stream: every TP/FSDP rank receives the same seed, while
            # different data replicas offset it to avoid cloning stochastic generations.
            replica_index = replica_coords()[0]
            torch.manual_seed(int(seed) + replica_index)
        pad_id = self.tok.pad_token_id or self.tok.eos_token_id
        trajs: list[Trajectory] = []
        module_training_flags = [(module, bool(module.training)) for module in self.model.modules()]
        self.model.eval()
        try:
            for obs in prompts:
                # Tokenize the packer prompt with add_special_tokens=False to match
                # DecoderOnlyLLMDataPacker.get_policy_input (which re-tokenizes the prompt the same
                # way); otherwise a default-added BOS (llama/qwen) would make rollout generate from a
                # different prefix than the objective scores.
                text = self.packer.get_rollout_input(obs)
                enc = self.tok(  # input_ids: [1,P]
                    text,
                    add_special_tokens=False,
                    return_tensors="pt",
                ).to(device)
                plen = enc["input_ids"].shape[1]
                # group_id = obs.id (per-prompt uuid), NOT the batch index: batch-local ids collide across
                # replicas at fan-in (roles/rollout/base.py::group_ids_for_prompt).
                group_id, traj_ids = group_ids_for_prompt(obs, group)
                # return_dict_in_generate + output_scores gives per-step logits so we can capture the
                # SAMPLED token's logprob (π_old) for the clipped objective's ratio (algorithm/grpo.py).
                out = self.model.generate(  # sequences: [G,P+T]
                    **enc,
                    num_return_sequences=group,
                    pad_token_id=pad_id,
                    return_dict_in_generate=True,
                    output_scores=capture_logprobs,
                    **sampling,
                )
                seqs = out.sequences  # [G,P+T]
                # out.scores: tuple(gen_len) of [G,V] PROCESSED scores, one per generated step.
                step_logps = (  # each [G,V]
                    [torch.log_softmax(score, dim=-1) for score in out.scores] if capture_logprobs else []
                )
                gen_len = len(step_logps)
                for ki in range(group):
                    comp_ids = seqs[ki][plen:].detach().cpu()  # [T]
                    # TRIM AT THE FIRST STOP TOKEN. `generate` right-pads every sequence in the group to
                    # the longest one, so a completion that stopped early carries trailing pad ids. Those
                    # pads are NOT generated tokens: keeping them puts pad positions inside
                    # `logprob_masks` (the mask is built from len(completion)), so the objective trains on
                    # padding, and their π_old is -inf — the sampler's logit processors (Qwen's
                    # generation_config sets top_k=20) push every filtered token to -inf, and a pad is
                    # always filtered. Downstream that becomes exp(cur - (-inf)) = inf, and inf * 0 = NaN
                    # the moment a group has zero advantage (all rewards equal), which is the common case
                    # early in training. Decoding hid this: `skip_special_tokens=True` makes
                    # `completion_text` look clean while the ids still carry the pads.
                    # cosmos-rl never sees this — its vLLM rollout passes stop_token_ids so the engine
                    # stops instead of padding, and its HF example rollout returns DECODED TEXT
                    # (completion_token_ids=None), which drops the pads on the way out.
                    comp_ids = _trim_at_stop(comp_ids, self._stop_ids)  # [T]
                    comp_text = self.tok.decode(comp_ids, skip_special_tokens=True)
                    # Per-token logprob of the SAMPLED completion token at each generated step. Truncate
                    # to min(len(comp_ids), gen_len) so early-EOS/pad never indexes past the produced
                    # scores. After the trim above len(comp_ids) <= gen_len, and the list length stays
                    # == len(action) — the parity behavior-logprob alignment needs.
                    n_lp = min(len(comp_ids), gen_len)
                    rollout_logprobs = [
                        float(step_logps[token_index][ki, int(comp_ids[token_index])]) for token_index in range(n_lp)
                    ]
                    # Absent (not empty) when disabled, so a consumer that opts into pi_old fails loud
                    # rather than silently seeing "no logprobs recorded".
                    pi = {
                        "prompt_ids": enc["input_ids"][0].detach().cpu(),  # [P]
                        "completion_text": comp_text,
                        "completion_token_count": len(comp_ids),
                        "weight_version": self._weight_version,
                    }
                    if capture_logprobs:
                        pi["rollout_logprobs"] = rollout_logprobs
                    tr = Transition(
                        obs=obs,
                        action=comp_ids,
                        policy_info=pi,
                        done=True,
                    )
                    trajs.append(
                        Trajectory(
                            transitions=[tr],
                            id=traj_ids[ki],
                            group_id=group_id,
                            status=Status.COMPLETED,
                        )
                    )
        finally:
            # ``model.train(was_training)`` recursively overwrites deliberately frozen/eval
            # submodules. Restore every module's exact incoming flag without recursive calls.
            for module, was_training in module_training_flags:
                module.training = was_training
        return trajs

    def _generate_validation(
        self,
        prompts: list[ObsBundle],
        validation: ValidationRolloutContext,
    ) -> list[Trajectory]:
        """Generate held-out completions with a failure boundary around every collective forward.

        HF/FSDP generation performs collectives on the rollout replica's default process group.
        Rank-local preparation and parsing must therefore converge before another rank can enter
        ``model.generate``. Training generation deliberately stays on the existing path above and
        pays none of these validation-only collectives.
        """

        def _prepare_runtime() -> tuple[dict[str, Any], torch.device, int | None, list[tuple[Any, bool]]]:
            if not isinstance(validation, ValidationRolloutContext):
                raise TypeError(f"validation must be ValidationRolloutContext or None, got {type(validation).__name__}")
            self.require_validation_weight_version(validation)
            sampling = dict(self.sampling)
            sampling["repetition_penalty"] = validation.repetition_penalty
            if validation.temperature == 0.0:
                # HF rejects temperature=0 even with sampling disabled; greedy is expressed by
                # do_sample=False and omitting every sampling warper altogether unless validation
                # explicitly names it below.
                sampling["do_sample"] = False
                sampling.pop("temperature", None)
                sampling.pop("top_p", None)
                sampling.pop("top_k", None)
            else:
                sampling["do_sample"] = True
                sampling["temperature"] = validation.temperature
            if validation.top_p is not None:
                sampling["top_p"] = validation.top_p
            elif sampling.get("do_sample", True):
                sampling["top_p"] = 1.0
            if validation.top_k is not None:
                sampling["top_k"] = validation.top_k if validation.top_k > 0 else None
            elif sampling.get("do_sample", True):
                sampling["top_k"] = None
            if validation.max_tokens is not None:
                sampling["max_new_tokens"] = validation.max_tokens
            device = next(self.model.parameters()).device
            pad_id = self.tok.pad_token_id or self.tok.eos_token_id
            module_training_flags = [(module, bool(module.training)) for module in self.model.modules()]
            return sampling, device, pad_id, module_training_flags

        sampling, device, pad_id, module_training_flags = run_validation_stage(
            _prepare_runtime,
            "HF validation runtime preparation",
        )
        trajectories: list[Trajectory] = []
        with distributed_validation_rng("HF validation rollout"):
            try:
                run_validation_stage(self.model.eval, "HF validation model-mode setup")
                for prompt_index, obs in enumerate(prompts):
                    enc, prompt_length, group_id, trajectory_ids = run_validation_stage(
                        lambda obs=obs: self._prepare_validation_prompt(
                            obs,
                            validation=validation,
                            device=device,
                        ),
                        f"HF validation prompt preparation {prompt_index}",
                    )
                    run_validation_stage(
                        lambda obs=obs, enc=enc: _require_identical_validation_prompt(
                            sample_id=obs.id,
                            encoded=enc,
                            sampling=sampling,
                            pad_token_id=pad_id,
                            validation=validation,
                        ),
                        f"HF validation prompt convergence {prompt_index}",
                    )
                    for generation_index in range(validation.n_generation):
                        run_validation_stage(
                            lambda obs=obs, generation_index=generation_index: seed_validation_rng(
                                validation_generation_seed(validation.seed, obs.id, generation_index)
                            ),
                            f"HF validation generation preflight {prompt_index}:{generation_index}",
                        )
                        output = run_validation_stage(
                            lambda enc=enc: self.model.generate(
                                **enc,
                                num_return_sequences=1,
                                pad_token_id=pad_id,
                                return_dict_in_generate=True,
                                output_scores=False,
                                **sampling,
                            ),
                            f"HF validation model generation {prompt_index}:{generation_index}",
                        )
                        trajectory = run_validation_stage(
                            lambda output=output, generation_index=generation_index: self._parse_validation_output(
                                obs=obs,
                                output=output,
                                prompt_ids=enc["input_ids"][0],
                                prompt_length=prompt_length,
                                group_id=group_id,
                                trajectory_id=trajectory_ids[generation_index],
                                validation=validation,
                            ),
                            f"HF validation output parsing {prompt_index}:{generation_index}",
                        )
                        trajectories.append(trajectory)
            finally:

                def _restore_modes() -> None:
                    # ``model.train(was_training)`` recursively overwrites deliberately frozen/eval
                    # submodules. Restore every module's exact incoming flag without recursive calls.
                    for module, was_training in module_training_flags:
                        module.training = was_training

                run_validation_stage(_restore_modes, "HF validation model-mode restoration")
        return trajectories

    def _prepare_validation_prompt(
        self,
        obs: ObsBundle,
        *,
        validation: ValidationRolloutContext,
        device: torch.device,
    ) -> tuple[Any, int, str, list[str]]:
        """Render and tokenize one held-out prompt before the next FSDP generate collective."""
        seed_validation_rng(validation_prompt_seed(validation.seed, obs.id))
        text = self.packer.get_rollout_input(obs)
        enc = self.tok(  # enc["input_ids"]: [1,P]
            text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(device)
        prompt_length = int(enc["input_ids"].shape[1])
        group_id, trajectory_ids = group_ids_for_prompt(obs, validation.n_generation)
        return enc, prompt_length, group_id, trajectory_ids

    def _parse_validation_output(
        self,
        *,
        obs: ObsBundle,
        output: Any,
        prompt_ids: torch.Tensor,  # [P]
        prompt_length: int,
        group_id: str,
        trajectory_id: str,
        validation: ValidationRolloutContext,
    ) -> Trajectory:
        """Parse one validation output locally; its caller converges before the next forward."""
        sequences = output.sequences  # [1,P+T]
        completion_ids = sequences[0][prompt_length:].detach().cpu()  # [T]
        completion_ids = _trim_at_stop(completion_ids, self._stop_ids)  # [T]
        completion_text = self.tok.decode(completion_ids, skip_special_tokens=True)
        policy_info: dict[str, Any] = {
            "prompt_ids": prompt_ids.detach().cpu(),  # [P]
            "completion_text": completion_text,
            "completion_token_count": len(completion_ids),
            "weight_version": self._weight_version,
            "validation_step": validation.model_step,
        }
        transition = Transition(
            obs=obs,
            action=completion_ids,
            policy_info=policy_info,
            done=True,
        )
        return Trajectory(
            transitions=[transition],
            id=trajectory_id,
            group_id=group_id,
            status=Status.COMPLETED,
        )
