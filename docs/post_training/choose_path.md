# Choose a post-training path

Most paths start from a released Cosmos3 checkpoint; experimental native QLoRA uses its pinned Qwen3-VL base. The sections below differ by the training signal you bring.

| You have                                                                   | Method       | Where                         |
| -------------------------------------------------------------------------- | ------------ | ----------------------------- |
| Paired examples: prompt → video, video → text answer, observation → action | SFT          | [SFT](#sft)                   |
| A teacher model you want to make faster (fewer steps) or causal            | Distillation | [Distillation](#distillation) |
| A reward function, verifier, simulator or preference pairs                 | RL           | [RL](#rl)                     |

## SFT

Supervised fine-tuning runs on the native trainer. Each recipe is a TOML file under `examples/toml/sft_config/`; most have a paired `examples/launch_sft_*.sh` that runs the trainer under `torchrun`.

```shell
python -m cosmos_framework.scripts.train --sft-toml=examples/toml/sft_config/<recipe>.toml
```

- [SFT recipes](../training.md): Generator SFT for Cosmos3-Nano, Cosmos3-Super (LoRA) and Cosmos3-Edge, Reasoner SFT for all three, plus checkpoint conversion and export.
- [Native QLoRA (experimental)](native_qlora.md): frozen NF4 language weights and native LoRA on Qwen3-VL-8B, initially restricted to one A100 80GB; runtime and quality qualification are pending.
- Your own data: [JSONL datasets](../dataset_jsonl.md) and [custom datasets](../custom_dataset.md).
- Robot actions from Cosmos3-Nano: [DROID policy](../action_policy_droid_posttrain.md) (reproduces Cosmos3-Nano-Policy-DROID), [DROID forward dynamics](../action_fd_droid_posttrain.md) and [LIBERO-10 policy](../action_policy_libero_posttrain.md).
- [Transfer-Auto](../training_transfer_auto.md): a four-phase RGB/LiDAR curriculum from Cosmos3-Nano, run with `cosmos_framework.scripts.train_experiment` instead.

## Distillation

- [Sim-Auto](../training_sim_auto.md): three stages from Transfer-Auto, TF → TF-dCM → SF-DMD. To adapt Sim-Auto to your domain, adapt Transfer-Auto first and then distill, rather than running SFT on the released Sim-Auto checkpoint; this preserves its few-step generation.
- Sim-Transfer: a bidirectional [Stage 0 teacher](../training_sim_transfer_stage0.md), then [Stages 1–3](../../.agents/skills/cosmos3-nano-sim-transfer/post-training/SKILL.md): teacher forcing, four-step distillation and self-forcing.
- Sim-Bimanual: [Stages 0–3](../../.agents/skills/cosmos3-nano-sim-bimanual/post-training/SKILL.md) from Cosmos3-Nano. [Check the training data](../bimanual_data_preflight.md) before loading weights.
- [Super four-step (DMD2)](../../.agents/skills/cookbooks/cosmos3/generator/audiovisual/distill/README.md): Cosmos3-Super-Text2Image or Cosmos3-Super-Image2Video as the teacher, the matching `-4Step` checkpoint as the initial student. A six-iteration functional recipe, not a production reproduction.

## RL

RL runs on the `cosmos_framework.posttrain` runtime in its own environment: the Framework base plus a locked dependency set. Start with the [overview](../posttrain/index.md) and the [run guide](../posttrain/howto/run-posttrain.md).

- Generator: [DPO](../posttrain/guides/dpo.md) learns from preference pairs collected in advance; [DiffusionNFT](../posttrain/guides/nft.md) generates and scores its own samples with a reward. The run-guide recipes start from Cosmos3-Nano.
- Reasoner: [GRPO](../posttrain/guides/grpo.md), [PPO](../posttrain/guides/ppo.md) and [OPD](../posttrain/guides/opd.md) (on-policy distillation from a frozen teacher). The run-guide recipes use public Qwen models.

To start the Reasoner side from Cosmos3-Nano or Cosmos3-Super, export the checkpoint as a Qwen3-VL Hugging Face directory. For Cosmos3-Super, add `--vlm-model-name Qwen/Qwen3-VL-32B-Instruct`.

```shell
python -m cosmos_framework.scripts.convert_model_to_vlm_safetensors \
    --checkpoint-path Cosmos3-Nano \
    -o examples/checkpoints/Cosmos3-Nano-VLM
```

Joint PPO also trains GR00T-N1.7 and π0.5 robot policies on LIBERO-Spatial; see the [GR00T-N1.7](../posttrain/guides/joint-ppo-gr00t-public.md) and [π0.5](../posttrain/guides/joint-ppo-pi05-public.md) guides. The runtime also has an [SFT objective](../posttrain/guides/sft.md) for Qwen text and vision-language models.

## AutoML (preview)

AutoML searches Reasoner SFT hyperparameters against held-out evaluation; its GEPA Auto-Prompter optimizes evaluation prompts without training model weights. It requires the `nvidia-automl-core` wheel, which is not yet published to a package index. See [AutoML](../automl.md).
