<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Convert Cosmos datasets

Conversion is implemented in Framework, under
`cosmos_framework/data/reasoner/formats/`. Install the optional `workflows`
dependencies; no external dataset toolkit or service is needed.

```bash
cosmos-dataset convert --help
cosmos-dataset convert metropolis-v3.0 cosmos-reason-v1.0 --help
cosmos-dataset convert metropolis-v3.0 cosmos-video-reasoning-v1.0 --help
cosmos-dataset convert metropolis-v3.0 cosmos-video-reasoning-v1.0 \
  --path /data/metropolis --output /data/reasoning
cosmos-dataset validate cosmos-video-reasoning-v1.0 --path /data/reasoning
```

Source and target are positional. `--path` and `--output` are required flags.
The two conversion targets preserve ordinary video conversations and task-aware
reasoning QA respectively. Task filtering and media-copy options are exposed by
each target's `--help`; do not assume options are identical.

Inspect the source layout and confirm the requested conversion before writing.
The output must be absent or empty; existing datasets are not overwritten.
Validate the result before training. Nonzero exit means conversion or validation
failed; retain logs and partial output for diagnosis.

For ordinary native Framework JSONL assembly, use `docs/dataset_jsonl.md`
through `cosmos3-post-training`. Unrelated dataset formats are outside this port.
