<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

<!-- Site-only page: links are site paths, so the repo link check is off. stage.py replaces the
agent-skills-table marker with a table built from every .agents/skills/**/SKILL.md. -->
<!-- rumdl-disable-file MD057 -->

# Agent skills

Cosmos-Framework ships skills for coding agents such as Claude Code and Codex: step-by-step instructions to install, run, post-train and troubleshoot Cosmos3 models, and to work on this repository. Each skill is a folder with a `SKILL.md` file; see [Agent Skills](https://agentskills.io).

## Load the skills

The skills refer to files in the repository, so work in a clone:

```shell
git clone https://github.com/NVIDIA/cosmos-framework.git
cd cosmos-framework
```

- **Codex** loads the skills in `.agents/skills/` when you start it in the clone, except those in hidden folders such as `cosmos3-nim-user`.
- **Claude Code** loads only the skills in `.claude/skills/`, a subset of the table below. To load them while working in another project, start it with `claude --add-dir <path to the clone>`.
- **For any skill your agent does not load**, ask it to read the skill's `SKILL.md`.

Agents reading this site can start from [`llms.txt`](../llms.txt). It links the Markdown of every page and every skill below.

## Skills

<!-- agent-skills-table -->
