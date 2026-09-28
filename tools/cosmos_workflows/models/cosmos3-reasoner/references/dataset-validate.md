<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Validate Cosmos datasets

The native `cosmos-dataset` CLI validates schemas, media references, and
format-specific consistency using the bundled Framework schemas.

```bash
cosmos-dataset validate --help
cosmos-dataset validate cosmos-video-reasoning-v1.0 --help
cosmos-dataset validate cosmos-video-reasoning-v1.0 --path /data/reasoning
```

Supported formats are `metropolis-v3.0`, `cosmos-reason-v1.0`, and
`cosmos-video-reasoning-v1.0`. Select the actual format; do not infer solely from
a directory name. `--path` accepts the format's dataset directory or parent tree.

Use the leaf `--help` for format-specific restrictions. `--strict` treats warnings
as failures where supported. Read the final validation summary and exit status;
a syntactically valid annotation can still reference missing media or task IDs.

Implementations and schemas live in `cosmos_framework/data/reasoner/formats/`.
Install Framework's optional `workflows` dependencies to use the command.
For conversion, see [dataset-convert.md](dataset-convert.md).
