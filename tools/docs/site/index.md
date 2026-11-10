---
hide:
  - navigation
  - toc
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

<!-- Site-only landing page, staged to index.md: links are site paths, so the repo link check is off. -->
<!-- rumdl-disable-file MD057 -->

# Cosmos-Framework

One Python package to train, post-train, evaluate and serve Cosmos3 omnimodal world models.

[Quickstart](get_started/quickstart.md){ .md-button .md-button--primary }
[Models](models/README.md){ .md-button }
[GitHub](https://github.com/NVIDIA/cosmos-framework){ .md-button }

**New:** [four world-simulation models](models/README.md#world-simulation): Transfer-Auto, Sim-Auto, Sim-Transfer and Sim-Bimanual.

**Using a coding agent?** Load the [agent skills](get_started/agent_skills.md) from a clone, or point it at [`llms.txt`](llms.txt).

## Choose your path

<div class="grid cards" markdown>

- **Generate video, images, text and actions**

    Text, images, video or actions in; images, video with optional audio, text or actions out.

    [Run inference](inference.md) · [Generation skill](skills/cosmos3-generate-media/SKILL.md)

- **Simulate driving and robot scenes**

    Multi-camera driving video from control videos; robot rollouts from recorded actions.

    [Transfer-Auto](inference_auto.md) · [Sim-Bimanual](inference_bimanual.md)

- **Fine-tune on your data**

    Supervised fine-tuning on JSONL or custom datasets, including robot policies.

    [Choose a path](post_training/choose_path.md) · [SFT recipes](training.md)

- **Improve with RL and preferences**

    DPO, DiffusionNFT, GRPO, PPO and OPD, from a single GPU to a multinode Ray cluster.

    [RL overview](posttrain/index.md) · [RL quickstart](posttrain/howto/run-posttrain.md)

- **Evaluate**

    Score a checkpoint against ground truth, or reproduce PAI-Bench, Physics-IQ and R-Bench.

    [Evaluate and score](evaluation.md) · [PAI-Bench-G](skills/evaluation/cosmos3/generator/paibench_g/README.md)

- **Deploy**

    Serve with Ray and Gradio, vLLM, TensorRT-LLM or NIM, or stream robot actions from a policy server.

    [Choose a serving path](deploy/serving.md) · [Deploy with NIM](skills/cookbooks/cosmos3/nim/README.md)

</div>
