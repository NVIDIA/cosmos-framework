# DiffusionNFT

Online RL for Cosmos world-foundation diffusion models: sample a group of generations per prompt,
score them with an image or video reward, and train the flow model on the resulting advantage.
Facts here are read from `config/rl.py::NFTConfig`, `entrypoints/nft.py`, and `algorithm/nft.py`.

## 1. When to choose it

Use DiffusionNFT when the thing you are improving is a **diffusion generator** and you have a
reward that scores its output — an aesthetic or preference model on decoded images, or a video
reward on decoded frames. It is the online counterpart to DPO for these models: DPO needs
preference pairs collected in advance, NFT generates and scores its own samples every step.

Choose [`dpo.md`](dpo.md) instead when you already have preference pairs — it needs no rollout and
no reward model at training time, so it is far cheaper. NFT is for the case where you can score but
cannot pair.

NFT is the most constrained entry in the framework. Production runs are **required** to use one
topology (see §4), so it has fewer degrees of freedom than GRPO by design.

## 2. Quick start

NFT runs are `experiment=` configurations, not `--config` recipes. The public four-GPU profile is
`cosmos3_nft_t2i_public_4gpu`: Cosmos3-Nano, four-rank FSDP, four prompt rows, and an HTTP
image-reward service. Set up the container and Posttrain environment from
[sections 1 and 2 of the run guide](../howto/run-posttrain.md#1-start-the-container), then prepare
the Nano DCP, the VAE, the prompt file, the reward-service variables and the local toy reward
service exactly as in [section 7](../howto/run-posttrain.md#7-nft).

Check the wiring first — NFT's preflight validates far more than the manifest, including the
sampler contract, the reward provider revision, and the moving-old schedule. **`--validate-manifest`
must come before the `--` separator**: `cli()` keeps only `key=value` items after it, so the flag
would be silently dropped and the command would start training instead.

```bash
"$POSTTRAIN_PY" -m cosmos_framework.posttrain.entrypoints.nft \
  --validate-manifest -- experiment=cosmos3_nft_t2i_public_4gpu

"$POSTTRAIN_PY" -m cosmos_framework.posttrain.entrypoints.nft -- \
  experiment=cosmos3_nft_t2i_public_4gpu \
  job.name=nft_public_4gpu
```

The profile runs 400 steps; append `rl.max_steps=5` for a short check. The
`cosmos3_nft_t2i_public` recipe it derives from is described in §9.

## 3. Data contract

`rl.modality` selects the decoded media type, `image` or `video` (default `video`). It identifies
the media type **only**; it does not implicitly select production coordinates.

The two modalities take different contracts:

| Modality | `rl.profile`                                                                       | Reward              |
| -------- | ---------------------------------------------------------------------------------- | ------------------- |
| `video`  | Not included in this release                                                       | Not included        |
| `image`  | Must name a known T2I profile; an unknown name is rejected with the supported list | `ImageRewardConfig` |

`rl.init_from` is **required** — NFT always starts from a base checkpoint, and an empty value fails
preflight rather than training from scratch.

## 4. Topology and launch

Production online NFT is required to use exactly one shape. Preflight rejects anything else:

| Setting              | Required value | Why                                                                  |
| -------------------- | -------------- | -------------------------------------------------------------------- |
| `grid.backend`       | `ray`          | The five roles are one Ray ActorGroup                                |
| `grid.grouping`      | `single`       | data, rollout, reward, replay, and policy share one process per rank |
| `grid.gpu_placement` | `colocated`    | That actor mesh forms one distributed FSDP world                     |
| `grid.gpus_per_role` | unset          | Colocated NFT does not accept per-role GPU overrides                 |

Shapes are equally fixed: **policy and rollout must both be `(1, group_size)`** — one execution
replica, and one WFM member per model rank. `_nft_role_shapes` derives this from
`grid.shapes` when given, or from `grid.num_replicas` / `grid.ranks` otherwise, and rejects any
other combination.

Concretely, `rl.group_size=8` means eight model ranks, one generated sample per rank.

## 5. Key configuration

Complete table: [`../reference/config.md`](../reference/config.md). Defaults below are read from
`NFTConfig`.

| Knob                                  | Default        | What it controls                                                              |
| ------------------------------------- | -------------- | ----------------------------------------------------------------------------- |
| `rl.group_size`                       | `4`            | Samples per prompt — and, per §4, the model rank count                        |
| `rl.beta`                             | `1.0`          | Objective sharpness                                                           |
| `rl.kl_beta`                          | `1.0`          | KL penalty weight. The qualified T2I recipe sets this to 0                    |
| `rl.policy_loss_scale`                | `5.0`          | Scale on the policy loss term                                                 |
| `rl.advantage_low` / `advantage_high` | `-5.0` / `5.0` | Advantage clamp                                                               |
| `rl.r_map`                            | `"linear"`     | Optimality map, `linear` or `sigmoid`                                         |
| `rl.r_tau`                            | `0.5`          | Temperature for the sigmoid map                                               |
| `rl.uniform_timestep`                 | `False`        | Sample the RF timestep uniformly                                              |
| `rl.sample_steps`                     | `20`           | Sampler steps per generation                                                  |
| `rl.sampler_shift`                    | `5.0`          | RF sampler shift                                                              |
| `rl.guidance`                         | `3.0`          | Classifier-free guidance                                                      |
| `rl.rollout_seed_clock`               | `"source_row"` | Whether the rank-local seed set advances per source row or per optimizer step |
| `rl.old.update_every`                 | —              | Moving-old refresh cadence. Must be >= 1 with `build_old` enabled             |
| `rl.ref.update_every`                 | `0`            | The reference is frozen; a nonzero value is rejected                          |

> **Guidance does not mean what it means in the reference implementation.** Native OmniMoT computes
> `v_uncond + guidance * (v_cond - v_uncond)`, so **guidance 1 is conditional-only** and guidance 0
> is unconditional. The reference sampler special-cases zero as conditional-only. Do not copy a
> numeric `guidance=0` across.

The scheduler type must be `unipc`; preflight rejects other samplers.

## 6. Is it training correctly

NFT's objective uses `MEAN` reduction, not `TOKEN_SUM`, so its metrics aggregate as plain means.

| Metric                                                     | Read it for                                                                                                             |
| ---------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `train/reward_mean`                                        | The task signal                                                                                                         |
| `train/policy_loss` / `train/kl_loss`                      | The two loss components separately — a KL term that dominates means the policy is being held at the reference           |
| `train/advantage_mean`                                     | Should stay well inside the `advantage_low`/`advantage_high` clamp; pinned at a bound means the clamp is doing the work |
| `train/optimality_mean` / `train/optimality_second_moment` | The optimality ratio's first and second moment                                                                          |
| `train/positive_error` / `train/negative_error`            | Error on each branch                                                                                                    |

Full table: [`../reference/metrics.md`](../reference/metrics.md).

## 7. Checkpoints

Shared mechanics: [`../reference/checkpoint.md`](../reference/checkpoint.md). NFT additionally
persists the **moving old policy**, the **prompt cursor**, and the **seed contract**. Exact resume
rejects a changed seed contract before advancing the prompt cursor, so a run cannot silently
continue under different sampling.

`rl.init_keys_to_skip_loading` allows a partial load from the base checkpoint; the video contract
requires it to be empty.

## 8. Failure semantics and limits

- **Preflight is strict and refuses rather than warns.** Backend, grouping, GPU placement, role
  shapes, profile/modality agreement, reward provider revision, sampler type, moving-old cadence,
  and a frozen reference are all validated before any GPU work.
- **Video NFT rejects a non-empty `rl.profile`**; image NFT rejects an unknown one.
- **A missing reward provider revision is fatal.** The provider's declared revision participates in
  run identity, so an unpinned provider cannot be used.
- **The topology is not configurable** (§4). This is a deliberate production constraint, not an
  unimplemented feature.

## 9. Public T2I recipe

`cosmos3_nft_t2i_public` is the release-facing image NFT recipe. It replaces
internal experiment, dataset, cluster and checkpoint defaults with explicit
operator-provided artifacts while retaining the T2I model, objective, LoRA,
sampling and reward contracts. See the
[public run guide](../howto/run-posttrain.md#7-nft).
