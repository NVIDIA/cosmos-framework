# Modular Dataflow — videophy2 Migration (Plan 4 of N)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Migrate the videophy2 VLM SFT recipe (`videophy2_sft_nano`) onto the four-role `CosmosDataLoader`, behavior-preserving, validated by golden-batch equality + a loss-curve regression run vs. the untouched baseline.

**Architecture:** videophy2 already runs on the legacy `DataPackerDataLoader` with an **iterable** source (`_UnshardedLocalSFTDataset`, non-resumable) and `VideoPhy2DataPacker`. Migration: `IterableDistributor(_UnshardedLocalSFTDataset(...))` + a new `VideoPhy2Processor` (extracts `VideoPhy2DataPacker.sft_process_sample`; the media-materialization differs from VLM) + `PoolPackingBatcher` + **reuse `VLMCollator`** (the videophy2 `sft_collate_fn` is byte-identical to VLM's). The legacy recipe + classes stay untouched (baseline). videophy2 has BOTH train and val loaders; mirror both.

**Tech Stack:** Python, PyTorch, Hydra `LazyCall`, pytest, `--sft-toml` launch.

**Spec:** `docs/superpowers/specs/2026-06-04-modular-dataflow-refactor-design.md` ("Goal 1.5 — videophy2"). Builds on Plans 1–3.

> **HARD INVARIANT:** videophy2's source is iterable → non-resumable today; the mirror must preserve that (no resume, callbacks untouched). Checkpoint *saving* (model/optim) is unaffected.

**Source references:**
- `cosmos_framework/configs/base/vlm/experiment/videophy2_sft_nano.py` — `_UnshardedLocalSFTDataset` (`:33-49`), `VideoPhy2DataPacker` (`:114-245`: `sft_process_sample` `:185-213`, `compute_num_tokens` `:215-216`, `sft_collate_fn` `:218-245`), `build_videophy2_local_dataset` (`:52-75`), `build_videophy2_datapacker_dataloader` (`:78-81`), dataloader wiring (`:315-356`).
- `cosmos_framework/data/vlm/local_sft_dataset.py:190` — `LocalSFTDataset`.
- `examples/toml/sft_config/videophy2_sft_nano.toml`, `examples/launch_sft_videophy2_nano.sh`.

---

## File Structure

| File | Change |
|---|---|
| `cosmos_framework/configs/base/vlm/experiment/videophy2_dataflow_roles.py` (create) | `VideoPhy2Processor` (extracted) |
| `cosmos_framework/configs/base/vlm/experiment/videophy2_dataflow_roles_test.py` (create) | processor unit test |
| `cosmos_framework/data/vfm/dataflow/golden_videophy2_test.py` (create) | legacy-vs-new batch equality |
| `cosmos_framework/configs/base/vlm/experiment/videophy2_sft_nano_v2_experiment.py` (create) | mirror experiment (train+val) |
| `cosmos_framework/configs/base/vlm/experiment/videophy2_sft_nano_v2_test.py` (create) | registration smoke |
| `examples/toml/sft_config/videophy2_sft_nano_v2.toml` (create) | mirror recipe TOML |
| `examples/launch_sft_videophy2_datapacker.sh` (create) | launch wrapper for the mirror |

---

### Task 1: `VideoPhy2Processor` (extract from VideoPhy2DataPacker)

**Files:**
- Create: `cosmos_framework/configs/base/vlm/experiment/videophy2_dataflow_roles.py`
- Test: `cosmos_framework/configs/base/vlm/experiment/videophy2_dataflow_roles_test.py`

`VideoPhy2DataPacker.sft_process_sample` differs from VLM only in turning the
LocalSFT `{"texts", "media"}` sample into messages via
`_materialize_media_in_conversation` (instead of `_sharegpt_to_openai`). Extract
that helper verbatim and wrap it as a `RawItemProcessor`.

- [ ] **Step 1: Write the failing test**

`videophy2_dataflow_roles_test.py`:
```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tests for VideoPhy2Processor (extracted from VideoPhy2DataPacker)."""

from __future__ import annotations

import torch

from cosmos_framework.configs.base.vlm.experiment.videophy2_dataflow_roles import VideoPhy2Processor


class _FakeProcessor:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return {"input_ids": torch.arange(6), "pixel_values_videos": torch.zeros(3, 8)}

    def add_assistant_tokens_mask(self, input_ids):
        m = torch.zeros_like(input_ids, dtype=torch.bool)
        m[3:] = True
        return m


def _item():
    # LocalSFT sample shape: texts (conversation list) + media (key->bytes).
    return {
        "texts": [
            {"from": "human", "value": "describe <video_0>"},
            {"from": "gpt", "value": "a ball falls"},
        ],
        "media": {"video_0": b"\x00\x00"},
    }


def test_videophy2_processor_builds_masked_labels():
    p = VideoPhy2Processor(processor=_FakeProcessor(), ignore_index=-100)
    s = p.process(_item())
    assert s["input_ids"].tolist() == [0, 1, 2, 3, 4, 5]
    assert s["labels"].tolist() == [-100, -100, -100, 3, 4, 5]
    assert "pixel_values_videos" in s


def test_videophy2_processor_rejects_non_list_texts():
    p = VideoPhy2Processor(processor=_FakeProcessor())
    import pytest

    with pytest.raises(TypeError):
        p.process({"texts": "not-a-list", "media": {}})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest cosmos_framework/configs/base/vlm/experiment/videophy2_dataflow_roles_test.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

`videophy2_dataflow_roles.py` — copy `VideoPhy2DataPacker.sft_process_sample`
(`videophy2_sft_nano.py:185-213`) and its helper `_materialize_media_in_conversation`
**verbatim** into a `RawItemProcessor`:
```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""videophy2 RawItemProcessor extracted 1:1 from VideoPhy2DataPacker."""

from __future__ import annotations

from typing import Any

from cosmos_framework.data.vfm.dataflow.base import RawItemProcessor
from cosmos_framework.utils.vlm.constant import IGNORE_INDEX, PROCESSOR_KEYS_TO_ADD


class VideoPhy2Processor(RawItemProcessor):
    """LocalSFT {"texts","media"} record -> VLM training tensors."""

    def __init__(self, processor: Any, ignore_index: int = IGNORE_INDEX) -> None:
        self._processor = processor
        self._ignore_index = ignore_index

    # --- COPY VERBATIM from videophy2_sft_nano.py VideoPhy2DataPacker ---
    # Paste `_materialize_media_in_conversation` here exactly as defined on the
    # original packer (it decodes media bytes per conversation turn). Keep its
    # signature `_materialize_media_in_conversation(self, conversation, media_bytes_by_key)`.
    def _materialize_media_in_conversation(self, conversation, media_bytes_by_key):
        ...  # <-- replace with the verbatim body from the source file

    def process(self, item: dict) -> dict:
        conversation = item.get("texts")
        if not isinstance(conversation, list):
            raise TypeError(
                f"LocalSFTDataset sample expected 'texts' to be a list, got {type(conversation).__name__}"
            )
        media_bytes_by_key = item.get("media") or {}
        messages = self._materialize_media_in_conversation(conversation, media_bytes_by_key)
        inputs = self._processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )
        input_ids = inputs["input_ids"]
        token_mask = self._processor.add_assistant_tokens_mask(input_ids)
        labels = input_ids.clone()
        labels[~token_mask] = self._ignore_index
        result: dict = {"input_ids": input_ids, "labels": labels}
        for key in PROCESSOR_KEYS_TO_ADD:
            if key in inputs and inputs[key] is not None:
                result[key] = inputs[key]
        return result
```

IMPORTANT: replace the `_materialize_media_in_conversation` stub body with the
exact code from `videophy2_sft_nano.py` (the only non-trivial difference from
VLM). The test's `_FakeProcessor` bypasses real media decoding, so for the unit
test the helper must at least pass the conversation through to messages; verify
against the real helper's behavior. If the real helper needs PIL/byte decoding
that the fake bypasses, adjust the test's `_item()` to provide a decodable media
blob, or assert only on the tokenization path.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest cosmos_framework/configs/base/vlm/experiment/videophy2_dataflow_roles_test.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cosmos_framework/configs/base/vlm/experiment/videophy2_dataflow_roles.py \
        cosmos_framework/configs/base/vlm/experiment/videophy2_dataflow_roles_test.py
git commit -m "feat(videophy2): extract VideoPhy2Processor from VideoPhy2DataPacker"
```

---

### Task 2: Golden-batch equality (legacy vs new)

**Files:**
- Create: `cosmos_framework/data/vfm/dataflow/golden_videophy2_test.py`

Same pattern as Plan 2 Task 3, but with `VideoPhy2DataPacker` (legacy) vs
`VideoPhy2Processor` + `PoolPackingBatcher` + `VLMCollator` (new). Confirms the
videophy2 `sft_collate_fn` truly equals `VLMCollator` by reusing the latter.

- [ ] **Step 1: Write the failing/characterization test**

`golden_videophy2_test.py`:
```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Golden-batch equality: legacy DataPackerDataLoader+VideoPhy2DataPacker vs the
new CosmosDataLoader (VideoPhy2Processor + PoolPackingBatcher + VLMCollator)."""

from __future__ import annotations

import random

import torch

from cosmos_framework.configs.base.vlm.experiment.dataflow_roles import VLMCollator
from cosmos_framework.configs.base.vlm.experiment.videophy2_dataflow_roles import VideoPhy2Processor
from cosmos_framework.configs.base.vlm.experiment.videophy2_sft_nano import VideoPhy2DataPacker
from cosmos_framework.data.vfm.data_packer_dataloader import DataPackerDataLoader as LegacyLoader
from cosmos_framework.data.vfm.dataflow import (
    CosmosDataLoader as NewLoader,
    IterableDistributor,
    PoolPackingBatcher,
)


class _FakeProcessor:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        n = 4 + len(messages)
        return {"input_ids": torch.arange(n)}

    def add_assistant_tokens_mask(self, input_ids):
        m = torch.zeros_like(input_ids, dtype=torch.bool)
        m[len(input_ids) // 2 :] = True
        return m


class _FixedIterable(torch.utils.data.IterableDataset):
    def __init__(self, items):
        self._items = items

    def __iter__(self):
        yield from self._items


def _items(k):
    return [
        {"texts": [{"from": "human", "value": f"q{i}"}, {"from": "gpt", "value": "a"}], "media": {}}
        for i in range(k)
    ]


def test_videophy2_golden_batches_match():
    proc = _FakeProcessor()
    items = _items(30)

    random.seed(0)
    legacy = LegacyLoader(
        data_source=_FixedIterable(list(items)),
        data_packer=VideoPhy2DataPacker(tokenizer_config=proc, max_seq_len=200),
        max_tokens=200, pool_size=8, max_batch_size=1, long_threshold=6400, num_workers=0,
    )
    random.seed(0)
    new = NewLoader(
        distributor=IterableDistributor(list(items)),
        processor=VideoPhy2Processor(processor=proc),
        batcher=PoolPackingBatcher(max_tokens=200, pool_size=8, max_batch_size=1, long_threshold=6400),
        collator=VLMCollator(),
        num_workers=0,
    )

    a = [next(iter(legacy)) for _ in range(8)]
    b = [next(iter(new)) for _ in range(8)]
    for ba, bb in zip(a, b):
        assert ba.keys() == bb.keys()
        for k in ba:
            assert torch.equal(ba[k], bb[k]), f"mismatch at {k}"
```

Note: this assumes `VideoPhy2DataPacker(tokenizer_config=..., max_seq_len=...)` and
that `_materialize_media_in_conversation` tolerates empty `media={}` with no
`<media>` placeholders. If the real helper requires a media token, give `_items`
a benign placeholder + a fake decodable blob, and mirror it in both paths.

- [ ] **Step 2: Run / fix until identical**

Run: `pytest cosmos_framework/data/vfm/dataflow/golden_videophy2_test.py -v`
Fix `VideoPhy2Processor` (Task 1) until the new batches equal legacy.

- [ ] **Step 3: Commit**

```bash
git add cosmos_framework/data/vfm/dataflow/golden_videophy2_test.py
git commit -m "test(dataflow): golden-batch equality videophy2 legacy vs new"
```

---

### Task 3: Mirror experiment (train + val) + registration smoke

**Files:**
- Create: `cosmos_framework/configs/base/vlm/experiment/videophy2_sft_nano_v2_experiment.py`
- Create: `cosmos_framework/configs/base/vlm/experiment/videophy2_sft_nano_v2_test.py`

Copy the `videophy2_sft_nano` experiment, swapping BOTH `dataloader_train` and
`dataloader_val` to the four-role `CosmosDataLoader`. Reuse the original
`build_videophy2_local_dataset` and `build_processor`; val keeps `num_workers=0`.

- [ ] **Step 1: Write the registration smoke test**

`videophy2_sft_nano_v2_test.py`:
```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

from hydra.core.config_store import ConfigStore


def test_videophy2_v2_registered():
    import cosmos_framework.configs.base.vlm.experiment.videophy2_sft_nano_v2_experiment  # noqa: F401

    names = set(ConfigStore.instance().repo["experiment"].keys())
    assert "videophy2_sft_nano_v2.yaml" in names, sorted(names)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest cosmos_framework/configs/base/vlm/experiment/videophy2_sft_nano_v2_test.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the mirror experiment**

`videophy2_sft_nano_v2_experiment.py` — duplicate the original's `LazyDict`
(defaults, job, trainer, model, checkpoint, data_setting) verbatim from
`videophy2_sft_nano.py`, changing only `job.name` to
`"videophy2_sft_nano_v2_${now:...}"` and replacing the two dataloaders:
```python
# (imports: LazyCall as L, LazyDict, ConfigStore, build_processor, IGNORE_INDEX,
#  the new dataflow roles, and the original build_videophy2_local_dataset)
from cosmos_framework.data.vfm.dataflow import CosmosDataLoader, IterableDistributor, PoolPackingBatcher
from cosmos_framework.configs.base.vlm.experiment.dataflow_roles import VLMCollator
from cosmos_framework.configs.base.vlm.experiment.videophy2_dataflow_roles import VideoPhy2Processor
from cosmos_framework.configs.base.vlm.experiment.videophy2_sft_nano import build_videophy2_local_dataset

def _dl(dataset_key, split, num_workers):
    return L(CosmosDataLoader)(
        distributor=L(IterableDistributor)(
            iterable=L(build_videophy2_local_dataset)(dataset_key=dataset_key, split=split),
        ),
        processor=L(VideoPhy2Processor)(
            processor=L(build_processor)(
                tokenizer_type="${model.config.policy.backbone.model_name}",
                config_variant="hf",
            ),
            ignore_index=IGNORE_INDEX,
        ),
        batcher=L(PoolPackingBatcher)(
            max_tokens="${data_setting.max_tokens}", pool_size=16, max_batch_size=1, long_threshold=6400,
        ),
        collator=L(VLMCollator)(),
        num_workers=num_workers,
    )

# videophy2_sft_nano_v2 = LazyDict(dict(... copied blocks ...,
#     dataloader_train=_dl("videophy2_train", "train", 2),
#     dataloader_val=_dl("videophy2_val", "val", 0),
# ))
# cs.store(group="experiment", package="_global_", name="videophy2_sft_nano_v2", node=videophy2_sft_nano_v2)
```
Copy the non-dataloader blocks (job/trainer/model/checkpoint/data_setting/defaults)
exactly from `videophy2_sft_nano.py:248-370` so only the dataloaders differ.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest cosmos_framework/configs/base/vlm/experiment/videophy2_sft_nano_v2_test.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cosmos_framework/configs/base/vlm/experiment/videophy2_sft_nano_v2_experiment.py \
        cosmos_framework/configs/base/vlm/experiment/videophy2_sft_nano_v2_test.py
git commit -m "feat(videophy2): mirror experiment videophy2_sft_nano_v2 on dataflow loader"
```

---

### Task 4: Mirror TOML + launch wrapper + regression run

**Files:**
- Create: `examples/toml/sft_config/videophy2_sft_nano_v2.toml`
- Create: `examples/launch_sft_videophy2_datapacker.sh`

- [ ] **Step 1: Write the mirror TOML**

Copy `examples/toml/sft_config/videophy2_sft_nano.toml` to
`videophy2_sft_nano_v2.toml`, changing `[job].experiment` to
`"videophy2_sft_nano_v2"`, `[job].name` likewise, `[job].project = "cosmos_oss_alignment"`,
`[job].wandb_mode = "online"`, `[trainer].logging_iter = 1`, `[trainer].max_iter = 500`.

- [ ] **Step 2: Write the launch wrapper**

`examples/launch_sft_videophy2_datapacker.sh`:
```bash
#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Dataflow-loader mirror of videophy2_sft_nano for loss-curve regression.
TOML_FILE="examples/toml/sft_config/videophy2_sft_nano_v2.toml"
: "${RUN_NAME:=videophy2_datapacker_v2_$(date +%Y%m%d_%H%M%S)}"
TAIL_OVERRIDES=(
    "data_setting.max_tokens=16000"
    "trainer.logging_iter=1"
    "trainer.max_iter=500"
    "job.project=cosmos_oss_alignment"
    "job.wandb_mode=online"
    "job.name=${RUN_NAME}"
)
source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
```

- [ ] **Step 3: Run baseline + mirror (env via cosmos3-run-env, node via slurm-node)**

Baseline: `bash examples/launch_sft_videophy2_nano.sh` with the same
`TAIL_OVERRIDES` (export before sourcing the common launcher) + fresh
`RUN_NAME=videophy2_baseline_...`, `CUDA_VISIBLE_DEVICES=0,1,2,3`.
Mirror: `bash examples/launch_sft_videophy2_datapacker.sh`,
`CUDA_VISIBLE_DEVICES=4,5,6,7`. Both need `WANDB_API_KEY` + venv per
`cosmos3-run-env`, and the videophy2 dataset present (`VIDEOPHYSICS_ROOT`).
Run both via `srun --overlap ... bash -s < /tmp/run_*.sh` (see Plan 2 Task 5).

- [ ] **Step 4: Verify equivalence**

Compare baseline vs mirror `loss` in wandb `cosmos_oss_alignment`. Under
`--deterministic` + `PYTHONHASHSEED=42` the curves must match. Record run URLs.

- [ ] **Step 5: Commit**

```bash
git add examples/toml/sft_config/videophy2_sft_nano_v2.toml \
        examples/launch_sft_videophy2_datapacker.sh
git commit -m "chore(videophy2): mirror TOML + launch wrapper for dataflow regression"
```

---

## Self-Review

**Spec coverage:** Goal 1.5 videophy2 migration → Tasks 1–4. `VideoPhy2Processor` extraction (spec "Processor placement: videophy2 = extract augmentation") → Task 1. Collator reuse confirmed (videophy2 `sft_collate_fn` == VLM) → Task 2 uses `VLMCollator`. Train + val both mirrored → Task 3. Regression run config (launch scripts, `logging_iter=1`, `max_iter=500`, wandb `cosmos_oss_alignment`, fresh names) → Task 4. Iterable/non-resumable preserved (no resume wiring) → consistent with HARD INVARIANT.

**Placeholder scan:** One intentional cite-and-copy: `_materialize_media_in_conversation` must be pasted verbatim from `videophy2_sft_nano.py` (Task 1 Step 3 calls this out explicitly with the exact source). Run-time secrets/paths in Task 4 resolved via named skills, not committed.

**Type consistency:** `VideoPhy2Processor(processor, ignore_index)` consistent across Tasks 1–3. `CosmosDataLoader(distributor, processor, batcher, collator, num_workers)` matches Plan 1. `VLMCollator()` reused from Plan 2 Task 2.
