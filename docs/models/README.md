# Models

The Cosmos3 checkpoints that Cosmos-Framework runs and trains, with their Hugging Face repositories and docs.

| Model                             | What it does                                                           | Hugging Face                                                                                                                                   | Docs                                                                                                                                                                      |
| --------------------------------- | ---------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Cosmos3-Nano                      | Generator (image, video, audio, action) and Reasoner (text)            | [Model card](https://huggingface.co/nvidia/Cosmos3-Nano)                                                                                       | [Run](../inference.md#cosmos3-nano) · [Train](../training.md)                                                                                                             |
| Cosmos3-Super                     | Generator and Reasoner, largest checkpoint                             | [Model card](https://huggingface.co/nvidia/Cosmos3-Super)                                                                                      | [Run](../inference.md#cosmos3-super) · [Train](../training.md)                                                                                                            |
| Cosmos3-Edge                      | Generator (no audio) and Reasoner, smallest checkpoint                 | [Model card](https://huggingface.co/nvidia/Cosmos3-Edge)                                                                                       | [Run](../inference.md#cosmos3-edge) · [Train](../training.md)                                                                                                             |
| Cosmos3-Super-Text2Image, -4Step  | Text to image; -4Step is a four-step distilled student                 | [Model card](https://huggingface.co/nvidia/Cosmos3-Super-Text2Image) · [4Step](https://huggingface.co/nvidia/Cosmos3-Super-Text2Image-4Step)   | [Run](../../.agents/skills/cookbooks/cosmos3/generator/audiovisual/README.md) · [Distill](../../.agents/skills/cookbooks/cosmos3/generator/audiovisual/distill/README.md) |
| Cosmos3-Super-Image2Video, -4Step | Image and prompt to video; -4Step is a four-step distilled student     | [Model card](https://huggingface.co/nvidia/Cosmos3-Super-Image2Video) · [4Step](https://huggingface.co/nvidia/Cosmos3-Super-Image2Video-4Step) | [Run](../../.agents/skills/cookbooks/cosmos3/generator/audiovisual/README.md) · [Distill](../../.agents/skills/cookbooks/cosmos3/generator/audiovisual/distill/README.md) |
| Cosmos3-Nano-Policy-DROID         | DROID robot policy                                                     | [Model card](https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID)                                                                          | [Serve](../action_policy_droid_server.md) · [Train](../action_policy_droid_posttrain.md)                                                                                  |
| Cosmos3-Edge-Policy-DROID         | DROID robot policy                                                     | [Model card](https://huggingface.co/nvidia/Cosmos3-Edge-Policy-DROID)                                                                          | [Serve](../action_policy_droid_server.md)                                                                                                                                 |
| Cosmos3-Nano-Transfer-Auto        | Multi-camera driving video, optionally with LiDAR, from control videos | [Model card](https://huggingface.co/nvidia/Cosmos3-Nano-Transfer-Auto)                                                                         | [Run](../inference_auto.md) · [Train](../training_transfer_auto.md)                                                                                                       |
| Cosmos3-Nano-Sim-Auto             | Distilled Transfer-Auto for long chunk-by-chunk rollouts               | [Model card](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Auto)                                                                              | [Run](../inference_sim_auto.md) · [Train](../training_sim_auto.md)                                                                                                        |
| Cosmos3-Nano-Sim-Transfer         | Video from a caption and one spatial control video                     | [Model card](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Transfer)                                                                          | [Run](../inference_transfer.md) · [Train](../training_sim_transfer_stage0.md)                                                                                             |
| Cosmos3-Nano-Sim-Bimanual         | Robot video from an initial image and bimanual actions                 | [Model card](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Bimanual)                                                                          | [Run](../inference_bimanual.md) · [Train](../../.agents/skills/cosmos3-nano-sim-bimanual/post-training/SKILL.md)                                                          |

## Cosmos3 Nano, Super and Edge

Each checkpoint has two capabilities: the Generator produces images, video, audio and actions, and the Reasoner generates text from a prompt and an optional image or video. There is no separate Reasoner repository. Select a checkpoint by name, for example `--checkpoint-path Cosmos3-Edge`.

| Checkpoint    | Total parameters (Hugging Face) | Generator modes in the framework CLI      |
| ------------- | ------------------------------- | ----------------------------------------- |
| Cosmos3-Nano  | 15.75B                          | All                                       |
| Cosmos3-Super | 64.6B                           | `text2image`, `text2video`, `image2video` |
| Cosmos3-Edge  | 3.86B                           | All except audio                          |

- Run: [Inference](../inference.md). Edge fits on a single GPU; Super does not fit on one 80 GB H100 and is sharded across GPUs.
- Reasoner: [Reasoner notebooks](../../.agents/skills/cookbooks/cosmos3/reasoner/README.md) · [Prompt guide](../../.agents/skills/cookbooks/cosmos3/reasoner/reasoner_prompt_guide.md). The vLLM, TensorRT-LLM, Transformers and NIM backends load only the Reasoner weights and use less GPU memory than the framework backend.
- Train: [SFT recipes](../training.md) (Generator and Reasoner SFT for each checkpoint) · [RL post-training](../posttrain/index.md)

## Super Text2Image and Image2Video

Cosmos3-Super-Text2Image and Cosmos3-Super-Image2Video are task-specialized Cosmos3-Super checkpoints. All four names, including -4Step, work with `--checkpoint-path`. The -4Step variants are DMD2-distilled students: they run a fixed four-step schedule with classifier-free guidance baked into the weights, and each serves on a single GPU with TensorRT-LLM.

- Run: [Generation notebooks](../../.agents/skills/cookbooks/cosmos3/generator/audiovisual/README.md#distilled-4-step-notebook) · [Prompt upsampling for Image2Video](../prompt_upsampling.md#posttrained-i2v)
- Distill: [Super four-step (DMD2)](../../.agents/skills/cookbooks/cosmos3/generator/audiovisual/distill/README.md), a short smoke recipe with each model as teacher and its -4Step checkpoint as the initial student

## Robot policy

Cosmos3-Nano-Policy-DROID and Cosmos3-Edge-Policy-DROID are post-trained DROID policies: they predict robot actions from camera observations, a task instruction and robot state. A policy server streams actions to a client that drives a simulated or real robot.

- Serve: [DROID policy server](../action_policy_droid_server.md) · [Action notebooks](../../.agents/skills/cookbooks/cosmos3/generator/action/README.md)
- Train: [DROID policy](../action_policy_droid_posttrain.md) reproduces Cosmos3-Nano-Policy-DROID from Cosmos3-Nano
- Other robot recipes: [DROID forward dynamics](../action_fd_droid_posttrain.md) · [LIBERO-10 policy](../action_policy_libero_posttrain.md)

## World simulation

Four checkpoints built on Cosmos3-Nano, each listed as input → output.

- **Transfer-Auto**: per-camera WSM (World Scenario Map) control videos and per-camera captions → synchronized multi-camera video, optionally with LiDAR. [Run](../inference_auto.md) · [Train](../training_transfer_auto.md)
- **Sim-Auto**: the same inputs → the same outputs, generated chunk by chunk with a rolling key/value cache and a fixed four-step sampler; distilled from Transfer-Auto. [Run](../inference_sim_auto.md) · [Train](../training_sim_auto.md)
- **Sim-Transfer**: a caption and one edge, blur, depth or segmentation control video → video, generated autoregressively. [Run](../inference_transfer.md) · [Train: Stage 0](../training_sim_transfer_stage0.md) · [Stages 1-3](../../.agents/skills/cosmos3-nano-sim-transfer/post-training/SKILL.md)
- **Sim-Bimanual**: an initial image and recorded robot and camera actions → an open-loop video rollout. [Run](../inference_bimanual.md) · [Live sessions](../bimanual_sessions.md) · [Check training data](../bimanual_data_preflight.md) · [Train: Stages 0-3](../../.agents/skills/cosmos3-nano-sim-bimanual/post-training/SKILL.md)

## Deploy with NIM

The [Cosmos3 NIM](../../.agents/skills/cookbooks/cosmos3/nim/README.md) serves the Nano and Super Generator and Reasoner, the Super Text2Image and Image2Video variants including -4Step, and a Nano DROID policy.
