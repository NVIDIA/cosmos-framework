# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Concrete GRPO config: Qwen2.5-3B-Instruct on GSM8K, mirroring the cosmos-rl reference TOML.

`build_config()` returns a fully-populated :class:`PosttrainGRPOConfig` — the same ROOT the GRPO
entrypoint uses (`config/config_grpo.py`) — whose values are copied verbatim from cosmos-rl's
``qwen2-5-3b-p-fsdp2-colocated-grpo.toml`` (the real bf16 3B GRPO recipe), so the example (Task 6)
runs the same algorithm/optimizer/rollout knobs the reference does. The few knobs that are
example-run sizing rather than TOML-exact (step count / prompts-per-step) are marked below.

Why the root and not a bare ``GRPOConfig``: in the landed config layout the model and the rollout
engine are **top-level** sections (``config.model_hf``, ``config.rollout``), NOT ``rl`` fields —
``rl`` holds the algorithm + lifecycle knobs only. A recipe has to pin all three, so it returns the
root that owns them.

Porting another cosmos-rl TOML? Read ``docs/design/grpo-cosmos-rl-parity.md`` §42 first. It lists the fields
whose names match while their units do not, the ones with no counterpart that must not be mapped to
something adjacent, and the settings that pass validation and only fail at launch. Two rows of the
table below once asserted values the TOML never declared (the sampling temperature and
``transfer_dtype``, both of which then silently defaulted): a mapping table is a CLAIM about the other
stack, so resolve both configs into objects and diff the RESOLVED values rather than the files.

TOML key -> config field mapping
--------------------------------
| cosmos-rl TOML key                          | posttrain field                    | value                      |
| ------------------------------------------- | ---------------------------------- | -------------------------- |
| policy.model_name_or_path                   | model_hf.name_or_path              | "Qwen/Qwen2.5-3B-Instruct" |
| train.enable_gradient_checkpointing         | model_hf.grad_ckpt                 | True                       |
| train.master_dtype                          | rl.master_dtype                    | "float32"  (STORAGE)       |
| train.param_dtype                           | rl.param_dtype                     | "bfloat16" (COMPUTE)       |
| train.fsdp_reduce_dtype                     | rl.reduce_dtype                    | "float32"                  |
| train.transfer_dtype                        | rl.transfer_dtype                  | "bfloat16" (see note)      |
| rollout.model_name_or_path (= policy)       | rollout.name_or_path               | "Qwen/Qwen2.5-3B-Instruct" |
| rollout.max_response_length                 | rollout.max_tokens                 | 512                        |
| (cosmos-rl's hardcoded engine seed)         | rollout.seed                       | 42                         |
| rollout.sampling_config.temperature         | rollout.temperature                | 0.9 (see dual-temp note)   |
| rollout.gpu_memory_utilization              | rollout.gpu_memory_utilization     | 0.25 (DELIBERATE, see below)|
| policy.model_max_length                     | rollout.max_model_len              | 4096                       |
| rollout.parallelism.tp_size                 | rollout.tensor_parallel_size       | 2                          |
| policy.parallelism.dp_shard_size            | model_hf.parallelism.dp_shard      | -1 (infer == 2)            |
| train.optm_name                             | rl.optimizer.name                  | "adamw"                    |
| train.optm_impl                             | rl.optimizer.impl                  | "fused" (torch KERNEL)     |
| train.optm_lr                               | rl.optimizer.lr                    | 1e-6                       |
| train.optm_weight_decay                     | rl.optimizer.weight_decay          | 0.01                       |
| train.optm_betas                            | rl.optimizer.betas                 | [0.9, 0.999]               |
| train.epsilon                               | rl.optimizer.eps                   | 1e-6                       |
| train.optm_grad_norm_clip                   | rl.optimizer.grad_clip_norm        | 1.0                        |
| train.optm_warmup_steps                     | rl.optimizer.warmup_steps          | 20                         |
| train.train_policy.epsilon_low              | rl.epsilon_low                     | 0.2                        |
| train.train_policy.epsilon_high             | rl.epsilon_high                    | 0.2                        |
| train.train_policy.kl_beta                  | rl.kl_beta                         | 0.0                        |
| train.train_policy.mu_iterations            | rl.mu_iterations                   | 1                          |
| train.train_policy.temperature              | rl.temperature                     | 0.9                        |
| train.train_policy.min_filter_prefix_tokens | rl.min_filter_prefix_tokens        | 1  (threshold, not a count) |
| train.train_policy.dataset.name             | rl.data.name                       | "openai/gsm8k"             |
| train.train_policy.dataset.subset           | rl.data.subset                     | "main"                     |
| train.train_policy.dataset.split            | rl.data.split                      | "train"                    |
| train.train_policy.dataset.prompt_column    | rl.data.prompt_column              | "question"                 |
| train.train_policy.dataset.response_column  | rl.data.response_column            | "answer"                   |
| (dataset.shuffle)                           | rl.data.shuffle                    | False                      |
| train.train_policy.n_generation             | rl.group_size                      | 16                         |
| train.ckpt.enable_checkpoint                | rl.save_every                      | 0  (disabled; see note)    |

Notes
-----
* Dual temperature: the rollout SAMPLING temperature (``rollout.temperature``) and the policy-loss
  (logit) temperature (``rl.temperature``) are NOT auto-synced, so we set BOTH explicitly to 0.9
  (leaving one at the default silently desyncs sampling from the update). The reference TOML sets
  only ``train.train_policy.temperature = 0.9`` and omits ``[rollout.sampling_config]`` entirely, so
  it SAMPLED at the vLLM default 1.0 while SCORING at 0.9 -- the importance ratio is then not a
  ratio over the sampling distribution. Measured on a live 400-step pair: at T=1.0 cosmos-rl's mean
  completion ran 291 tokens against posttrain's 210 at T=0.9, hitting the 512 cap on nearly every
  early step. The REFERENCE TOML still carries that default -- it is left as it is so the finding
  stays visible in it. A separate
  ``packages/cosmos-rl/configs/qwen2-5/qwen2-5-3b-p-fsdp2-grpo-posttrain-parity.toml`` declares
  ``[rollout.sampling_config]`` at 0.9 for the cross-stack comparison, so this row describes what
  that comparison pins, not what the reference recipe does.
* `transfer_dtype`: the reference TOML did not set this field, so it defaulted to `master_dtype`
  (`cosmos_rl/policy/config/__init__.py:2047`) = float32, while this recipe pushes bf16 -- the
  dtype the rollout engine actually runs. A live 400-step pair therefore differed in the
  precision of every weight push, invisibly, because neither config mentioned the field. The
  parity TOML sets it explicitly; the reference is unchanged. This is the SAME failure as the
  dual-temperature note above: this table asserted a TOML value the TOML never declared.
* Example-run sizing (NOT TOML-exact): ``rl.max_steps`` / ``rl.batch_per_replica`` are chosen for a
  reasonable single-run example, not copied from the TOML's large-scale schedule.
* Policy data parallelism lives at ``model_hf.parallelism``; rollout TP remains the separate
  ``rollout.tensor_parallel_size``. The entrypoint validates
  ``dp_replicate * dp_shard == POLICY_RANKS`` and placement never synthesizes that factorization.
* **Checkpoint cadence**: the TOML's ``save_mode="async"`` maps to shared ``rl.save_async=True``.
  The TOML disables checkpointing (``enable_checkpoint=False``), which is ``rl.save_every=0``;
  ``save_async`` is therefore inert for normal cadence and the TOML's ``save_freq=5`` is moot.
"""

from __future__ import annotations

from cosmos_framework.posttrain.config.config_grpo import PosttrainGRPOConfig
from cosmos_framework.posttrain.config.config_grpo import make_config as _make_root_config
from cosmos_framework.posttrain.config.model import HF
from cosmos_framework.posttrain.config.rl import DatasetConfig, OptimizerConfig

# Reference checkpoint mirrored by both the policy (FSDP trainer) and the vLLM rollout engine.
_QWEN_3B = "Qwen/Qwen2.5-3B-Instruct"

# The GSM8K format instruction lives HERE, not in an env, because it is CONSTRUCTION rather than
# selection -- and because the prompt and the `#### N` verifier (`rl.reward = "gsm8k"`, set below)
# are ONE contract split across two roles (docs/design/grpo-cosmos-rl-parity.md section 2.1). This file is
# the only place that pins both.
#
# Deliberately stricter than cosmos-rl's inline `Let's think step by step and output the final
# answer after "####".` -- ours also forbids units and commas, which matters because the verifier is
# EXACT-MATCH: `#### 72.` or `#### 1,234` scores 0 against gold `72`/`1234`. Measured worth roughly
# +0.20 mean reward (0.82 vs 0.62, docs/design/grpo-cosmos-rl-parity.md section 2.1).
# The column holding the problem text. ONE definition, read by `gsm8k_items` (which rewrites it in
# place) and by `build_config` (which names it as `rl.data.prompt_column`). Two copies of this would
# be exactly the drift the prompt/verifier note below warns about: the map writes the conversation
# to one column while the framework reads the prompt from another, so training silently sees the
# RAW question with no system prompt. Overriding `rl.data.prompt_column` on this recipe without
# changing this constant is out of contract for that reason -- the map encodes GSM8K's own schema.
_PROMPT_COLUMN = "question"

GSM8K_SYSTEM_PROMPT = (
    "Solve the math problem step by step. On the last line, output the final answer "
    "as '#### <number>' with digits only (no units, no commas)."
)


def gsm8k_items(row: dict) -> dict:
    """`rl.data.items_fn` target: a MAP over one GSM8K row, rewriting the prompt column in place.

    Row in, row out, same keys: the conversation is written back to `question`, which
    `rl.data.prompt_column` then names as the prompt. Nothing is renamed or dropped here -- the
    column fields on `DatasetConfig` stay the single description of which column plays which role.

    Emits the CONVERSATION, never a rendered string: the DataPacker applies the chat template, so
    no env and no dataset needs a tokenizer.

    Args:
        row: a raw dataset row carrying the `_PROMPT_COLUMN` column.

    Returns:
        The row with the prompt column replaced by the system+user message list.
    """
    return {
        **row,
        _PROMPT_COLUMN: [
            {"role": "system", "content": GSM8K_SYSTEM_PROMPT},
            {"role": "user", "content": row[_PROMPT_COLUMN]},
        ],
    }


def gsm8k_items_cosmos_rl(row: dict) -> dict:
    """`rl.data.items_fn` target reproducing cosmos-rl's GSM8K wording BYTE FOR BYTE.

    Same contract as `gsm8k_items` (row in, row out, conversation written back to the prompt
    column), but emits cosmos-rl's single user turn with its inline suffix instead of this recipe's
    system message. Copied from `cosmos_rl/tools/dataset/gsm8k_grpo.py::GSM8kDataset.__getitem__`.

    This exists ONLY so a cross-stack comparison can hold the prompt constant. The prompt is worth
    ~0.20 mean reward on GSM8K (parity doc section 2.1: 0.8233 with this recipe's wording against
    0.6237 with cosmos-rl's), which is larger than anything the trainer-side differences produce --
    a curve comparison that leaves it uncontrolled measures the prompt, not the stacks. It is NOT
    the recipe default and should not become one: the default wording scores better.

    Args:
        row: a raw dataset row carrying the `_PROMPT_COLUMN` column.

    Returns:
        The row with the prompt column replaced by cosmos-rl's single-user-turn message list.
    """
    return {
        **row,
        _PROMPT_COLUMN: [
            {
                "role": "user",
                "content": f'{row[_PROMPT_COLUMN]} Let\'s think step by step and output the final answer after "####".',
            }
        ],
    }


def build_config() -> PosttrainGRPOConfig:
    """Build the Qwen2.5-3B / GSM8K GRPO config mirroring the cosmos-rl reference TOML.

    Starts from `config_grpo.make_config()` so the recipe inherits the registered job/trainer
    defaults (`build_root`) instead of re-deriving them, then overrides the recipe-specific fields.

    Returns:
        PosttrainGRPOConfig: fully-populated root (`rl` + `model_hf` + `rollout`); see the module
        docstring for the TOML->field mapping.
    """
    # This file fully specifies the runnable recipe and does not consume Hydra's `experiment=`
    # catalog. Importing that catalog would recursively load every unrelated Cosmos3 experiment
    # and make their optional dependencies (for example LeRobot) mandatory for this HF-only run.
    cfg = _make_root_config(register_experiment_configs=False)
    # This HF recipe supplies its own optimizer, data and sampling configuration.
    # Generator defaults include a scheduler group absent from the standalone checkout.
    cfg.defaults = ["_self_"]
    cfg.model_backend = HF

    # --- job identity (W&B project/group/name AND the output namespace) ---
    # `build_root` stamps the generator-side defaults: project `cosmos3_vfm` (which
    # `get_wandb_project` maps to `cosmos3_generator`) and the throwaway name
    # `delete_<timestamp>`. Both are wrong for a reasoner GRPO recipe. `cosmos_reason2` is the
    # legacy reasoner STORAGE namespace and is what maps to the `cosmos3_reasoner` W&B project, so
    # this recipe lands in the right workspace without every launcher having to export
    # `WANDB_PROJECT`. Group/name follow the Cosmos Lab contract (group = comparison cohort, name =
    # `<variant>_seed<seed>`); a launcher can still redirect a run into a study cohort with
    # `WANDB_RUN_GROUP` / `WANDB_NAME`, which `utils/report.py` honors.
    cfg.job.project = "cosmos_reason2"
    cfg.job.group = "reasoner_grpo_gsm8k"
    cfg.job.name = "qwen3b_bf16_seed42"

    # --- the FSDP-sharded HF trainer weights (top-level, not an rl field) ---
    # NO `model_hf.dtype` / `fsdp_reduce_dtype` here: precision moved to `rl.*` (see below and
    # config/rl.py::HFRunConfig). Setting them would still work — they are the deprecated fallback —
    # but they cannot express this recipe, which is the reason the split exists.
    cfg.model_hf.name_or_path = _QWEN_3B
    cfg.model_hf.attn_impl = "eager"  # GRPO stack default (flash-attn ABI break on the training image)
    cfg.model_hf.grad_ckpt = True
    cfg.model_hf.parallelism.dp_replicate = 1
    cfg.model_hf.parallelism.dp_shard = -1  # infer pure FSDP over POLICY_RANKS (== 2 here)

    # --- precision: STORAGE vs COMPUTE, matching cosmos-rl's [train] section ---
    # cosmos-rl runs `master_dtype=float32` (its default) with `param_dtype=bfloat16`: the parameter
    # is held in fp32 and cast to bf16 for the matmuls. That is what makes plain `torch.optim.AdamW`
    # correct there — `zeros_like(p)` on an fp32 param gives fp32 moments, so no master-weight
    # optimizer is needed. This recipe previously could not express that (one `model_hf.dtype` drove
    # both), which is why it reached for apex `fusedadam` to rebuild an fp32 master inside the
    # optimizer. With the split, the reference configuration is stated directly and apex is not
    # needed on this path at all.
    cfg.rl.master_dtype = "float32"  # cosmos-rl train.master_dtype (its default)
    cfg.rl.param_dtype = "bfloat16"  # cosmos-rl train.param_dtype
    cfg.rl.reduce_dtype = "float32"  # cosmos-rl train.fsdp_reduce_dtype
    # The policy holds fp32 while the vLLM engine is bf16, so this hop is exactly where the cast
    # belongs. Left at "" it resolves to master_dtype (fp32) and the plan would deliberately size,
    # stage and send fp32 buffers to a bf16 engine — double the weight-sync traffic and staging
    # memory, on a COLOCATED recipe where that competes with the engine's own allocation.
    cfg.rl.transfer_dtype = "bfloat16"  # == the rollout engine's dtype, not the policy's storage

    # --- the separate vLLM inference engine + its sampling (top-level, not an rl field) ---
    cfg.rollout.name_or_path = _QWEN_3B
    # 2 GPUs, colocated: `rollout.tensor_parallel_size = 2` makes POLICY_RANKS = 2, and
    # `model_hf.parallelism.dp_shard = -1` above infers a pure-FSDP shard of 2 over them. The
    # single-device default (TP=1) leaves `parallelize()` installing no FSDP, hence no
    # MixedPrecisionPolicy, hence `param_dtype` cannot be delivered and the precision guard refuses
    # to start. The dtypes above are the point of this recipe, so the topology has to honour them.
    cfg.grid.backend = "ray"
    cfg.grid.num_replicas = 1
    # `grid.ranks` is the K of the fallback shapes: with `shapes` left empty, policy falls back to
    # `(1, ranks)` and rollout to `(num_replicas, ranks)` (config/rl.py:306-310). It defaults to 1,
    # so a recipe that asks for `rollout.tensor_parallel_size = 2` without setting this dies at
    # startup with `ROLLOUT_RANKS=1 must equal rollout.tensor_parallel_size=2` — the two GPUs the
    # precision split needs are declared in one place and not the other.
    cfg.grid.ranks = 2
    cfg.rollout.tensor_parallel_size = 2  # 2 GPUs total, colocated
    # DELIBERATE divergence from the reference TOML's 0.4, together with
    # `rl.micro_batch_per_replica = 8` below. Reserving 0.4 of each card for the colocated vLLM
    # engine leaves the policy too little for its fp32 `log_softmax` over Qwen's 151k vocabulary,
    # and the run OOMs on two 80 GB H100s — the topology this recipe pins. The qualified profile
    # completed all 300 steps and improved mean reward 0.8400 -> 0.9264 (docker/README.md). Defaults
    # that cannot run on the recipe's own topology are not useful defaults.
    cfg.rollout.gpu_memory_utilization = 0.25
    cfg.rollout.max_model_len = 4096
    cfg.rollout.temperature = 0.9  # rollout SAMPLING temperature (NOT auto-synced with rl.temperature)
    cfg.rollout.max_tokens = 512
    # cosmos-rl ALWAYS seeds its engine (`vllm_rollout.py:251,321`: `seed: int = 42` then
    # `seed=seed or 42`), while `RolloutConfig.seed` defaults to None and `config/build.py:500`
    # passes a seed only when it is set — so this recipe used to run UNSEEDED against a fixed-seed
    # reference. That made every cross-stack reward comparison a seeded-vs-unseeded one; the parity
    # doc (section 14) measured it as roughly a quarter of the original gap. Pinned to the reference
    # value so the two stacks start from the same sampling stream.
    cfg.rollout.seed = 42

    # --- lifecycle (RunConfig) ---
    cfg.rl.max_steps = 300  # example-run sizing (NOT TOML-exact)
    cfg.rl.save_every = 0  # TOML enable_checkpoint=False
    cfg.rl.optimizer = OptimizerConfig(
        # TOML `optm_name = "AdamW"`. This was `fusedadam`, which mis-read the TOML: cosmos-rl's
        # `optm_impl = "fused"` selects a torch KERNEL (`fused=True` on torch.optim.AdamW), it does
        # NOT select apex. cosmos-rl's LLM GRPO optimizer registry is {Adam, AdamW, Adam8bit,
        # AdamW8bit} — apex appears only on its WFM/diffusion path — so `fusedadam` silently ran a
        # different optimizer than the reference. With `rl.master_dtype="float32"` above, plain
        # AdamW gets fp32 moments from the fp32 parameter, which is what apex's master_weights was
        # emulating.
        name="adamw",
        impl="fused",  # TOML optm_impl = "fused" -> torch.optim.AdamW(fused=True)
        lr=1e-6,
        weight_decay=0.01,
        betas=[0.9, 0.999],
        eps=1e-6,
        grad_clip_norm=1.0,
        warmup_steps=20,
        # Full-finetune the LLM: train ALL params. Stated explicitly even though it now MATCHES the
        # OptimizerConfig default — an earlier draft defaulted this to the i4/diffusion
        # generation-adapter keys (["moe_gen", "time_embedder", ...]), which match no Qwen causal-LM
        # parameter and would have frozen the whole net; the landed schema keeps a freeze subset a
        # per-recipe choice and defaults to []. Pinning it here makes the recipe independent of that
        # default (cosmos-rl's TOML likewise updates the whole policy).
        train_keys=[],
    )

    # --- GRPO loop + clipped-objective knobs (FLAT on GRPOConfig, as cosmos-rl keeps them flat) ---
    cfg.rl.group_size = 16  # TOML n_generation (GRPO group size G)
    cfg.rl.batch_per_replica = 128  # example-run sizing (NOT TOML-exact): 8 prompts x 16 group
    # Memory-only split of the same 128 trajectories into 8-sample chunks: the collected batch and
    # the single optimizer step per batch are unchanged, only the activation footprint shrinks.
    # Part of the qualified two-GPU profile described at `gpu_memory_utilization` above.
    cfg.rl.micro_batch_per_replica = 8
    cfg.rl.epsilon_low = 0.2
    cfg.rl.epsilon_high = 0.2
    cfg.rl.kl_beta = 0.0
    cfg.rl.mu_iterations = 1
    cfg.rl.temperature = 0.9  # policy-loss (logit) temperature; matches rollout.temperature
    # TOML min_filter_prefix_tokens=1 — a THRESHOLD for shared-prefix loss filtering, not a count.
    # The tokens actually masked are the COMPUTED shared-prefix length, and only for completions in
    # a same-prefix/different-reward set (algorithm/annotators/prefix_filter.py).
    cfg.rl.min_filter_prefix_tokens = 1
    # The production clipped surrogate (ratio + eps-clip + dual-clip), not legacy_pg.
    # This is what makes `entrypoints/grpo.py --config <this file>` a GSM8K GRPO run.
    cfg.rl.grpo_surrogate = "clipped"

    # --- dataset (standalone `data`, same field name DPO uses) ---
    # `rl.reward` selects the verifier from the built-in registry (main's reward selection), and
    # `bind_check` validates it against the prompt source -- gsm8k is registered in algorithm/reward
    # with needs=("reference",), which `response_column` supplies.
    cfg.rl.reward = "gsm8k"
    cfg.rl.data = DatasetConfig(
        name="openai/gsm8k",
        subset="main",
        split="train",
        prompt_column=_PROMPT_COLUMN,
        response_column="answer",
        shuffle=False,
        items_fn="cosmos_framework.posttrain.configs.grpo_qwen3b_gsm8k.gsm8k_items",
    )
    return cfg


def make_config() -> PosttrainGRPOConfig:
    """The i4 `--config` entry point (`cosmos_framework.utils.config.load_config` calls `make_config()`).

    Alias of `build_config` so this recipe can be launched directly:

        PYTHONPATH=. python -m cosmos_framework.posttrain.entrypoints.grpo \
            --config cosmos_framework/posttrain/configs/grpo_qwen3b_gsm8k.py

    which gets resume, run_ts layout, save cadence, preemption save and W&B from the entrypoint —
    none of which the standalone example has.
    """
    return build_config()
