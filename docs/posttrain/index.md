# Posttrain documentation

Post-training for Cosmos models: offline SFT, DPO, GRPO, actor-critic PPO, on-policy distillation (OPD), and
DiffusionNFT on one substrate that scales from a single GPU to a multinode Ray cluster.

New public users should start with the
[`general run guide`](howto/run-posttrain.md), then read the reference guide for the
algorithm they need. The [`glossary.md`](glossary.md) is worth five minutes before anything else — `replica`,
`rank`, `dp_shard`, `ActorGroup`, and `DeviceGroup` all mean specific things here.

## Getting started

| Document                                                 | What it covers                                                                                                            |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| [`howto/run-posttrain.md`](howto/run-posttrain.md)       | Step-by-step container recipes for GRPO, DPO, PPO, OPD, NFT, and Joint PPO, using a four-GPU showcase that can be adapted |
| [`howto/run-posttrain-qa.md`](howto/run-posttrain-qa.md) | Run length and short runs, Joint PPO's silent pre-training evaluation, checkpoint intervals, and first-run errors         |
| [`glossary.md`](glossary.md)                             | Terminology, including the three unrelated meanings of "validation"                                                       |

## Run a capability

Each guide follows the same skeleton, so a reader who has used one can navigate the others.

| Guide                                                                  | Use it for                                                                    |
| ---------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| [`guides/dpo.md`](guides/dpo.md)                                       | Offline preference optimization on preference pairs                           |
| [`guides/grpo.md`](guides/grpo.md)                                     | Online RL with a rule or model reward and a vLLM rollout                      |
| [`guides/ppo.md`](guides/ppo.md)                                       | Actor-critic PPO with frozen reference and critic roles                       |
| [`guides/opd.md`](guides/opd.md)                                       | Distilling a frozen teacher into a policy on-policy                           |
| [`guides/nft.md`](guides/nft.md)                                       | DiffusionNFT — online RL for diffusion world models                           |
| [`guides/joint-ppo-pi05-public.md`](guides/joint-ppo-pi05-public.md)   | Public π0.5 checkpoint, LIBERO runtime, assets, and four-GPU Joint PPO launch |
| [`guides/joint-ppo-gr00t-public.md`](guides/joint-ppo-gr00t-public.md) | Public GR00T-N1.7 checkpoint, pinned upstreams, and four-GPU Joint PPO launch |

## Customize

| Document                                                           | Use it for                                                                                                                    |
| ------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| [`howto/custom-data.md`](howto/custom-data.md)                     | Pointing a run at your own dataset                                                                                            |
| [`howto/custom-datapacker.md`](howto/custom-datapacker.md)         | Supporting a new model architecture's tensor shaping                                                                          |
| [`howto/custom-reward.md`](howto/custom-reward.md)                 | Writing a reward provider                                                                                                     |
| [`howto/custom-objective.md`](howto/custom-objective.md)           | Changing the training objective                                                                                               |
| [`howto/sim-integration.md`](howto/sim-integration.md)             | Putting your own simulator behind the `Simulator`/`SimSession` contract                                                       |
| [`howto/batching.md`](howto/batching.md)                           | Sizing mini/micro batches, fixing OOM, and what the colocated rollout/trainer offload holds on the card                       |
| [`howto/data-plane.md`](howto/data-plane.md)                       | The opt-in Ray/NIXL pooled data plane for large trajectory fields                                                             |
| [`howto/preflight.md`](howto/preflight.md)                         | Checking wiring before allocating GPUs                                                                                        |
| [`howto/train-time-validation.md`](howto/train-time-validation.md) | Held-out evaluation during training                                                                                           |
| [`howto/checkpoint-artifacts.md`](howto/checkpoint-artifacts.md)   | Shared archive publication and verified fetch/pruning; `checkpoint.tools` commands for archives, expert merge and HF teachers |

## Look something up

| Document                                                   | Contents                                                                                                            |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| [`reference/config.md`](reference/config.md)               | Every `rl.*` knob, per algorithm                                                                                    |
| [`reference/metrics.md`](reference/metrics.md)             | Every reported metric, and how it aggregates                                                                        |
| [`reference/checkpoint.md`](reference/checkpoint.md)       | Formats, resume, preemption, retention                                                                              |
| [`reference/compatibility.md`](reference/compatibility.md) | API maturity and migration windows                                                                                  |
| [`reference/embodied.md`](reference/embodied.md)           | The simulator-side contracts: `Simulator`/`SimSession`, `EmbodimentMap`, `WorkUnit`, and the plugin/import boundary |
| [`troubleshooting.md`](troubleshooting.md)                 | Symptom-first failure catalogue                                                                                     |

## Qualified commands

Each command below was run end to end in the runtime image of the
[run guide](howto/run-posttrain.md) and completed successfully. Rerun them after a dependency or
image change.

| Command                                                                      | Result                                                                                                                     |
| ---------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `--validate-manifest` for GRPO, PPO, OPD                                     | exit 0, `bind_check: ok`                                                                                                   |
| GRPO GSM8K public recipe ([`run guide`](howto/run-posttrain.md#3-grpo))      | Completed optimizer updates with public Qwen and GSM8K artifacts                                                           |
| DPO public T2I recipe ([`run guide`](howto/run-posttrain.md#4-dpo))          | Four-rank FSDP initialized from the converted Cosmos3-Nano and completed optimizer updates over synthetic preference pairs |
| PPO Qwen3-0.6B public recipe ([`run guide`](howto/run-posttrain.md#5-ppo))   | Completed five policy/critic update iterations                                                                             |
| OPD Countdown public recipe ([`run guide`](howto/run-posttrain.md#6-opd))    | Completed five disaggregated policy/rollout/teacher updates over NIXL                                                      |
| NFT Cosmos3-Nano public recipe ([`run guide`](howto/run-posttrain.md#7-nft)) | Completed five updates against the deterministic toy reward service                                                        |
| π0.5 Joint PPO ([`public guide`](guides/joint-ppo-pi05-public.md))           | Completed five LIBERO collection/update windows                                                                            |
| GR00T-N1.7 Joint PPO ([`public guide`](guides/joint-ppo-gr00t-public.md))    | Completed five LIBERO collection/update windows with exit code 0                                                           |
