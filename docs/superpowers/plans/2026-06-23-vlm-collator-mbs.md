# VLMCollator `max_batch_size > 1` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `VLMCollator` produce correct VLM training batches for any `max_batch_size ≥ 1` by porting i4's `custom_collate` pad-and-stack logic.

**Architecture:** All batch sizes route through a single pad-and-stack collate path: variable-length sequence tensors are right-padded to the batch max (rounded up to a multiple of 16) and stacked on a new batch axis; vision tensors are flat-concatenated on dim 0. `VLMProcessor` emits the per-sample constants (`pad_token_id`, `ignore_index`, `token_mask`) the collate consumes. The batcher (`PoolPackingBatcher`) and model forward (`HFModel`/`get_position_ids`) already support batched/padded input and need no changes.

**Tech Stack:** Python, PyTorch, pytest. Cosmos3 dataflow (`cosmos_framework.data.vfm.dataflow`). Training launched in the `bob_echo_dev` i4 container via the `cosmos3-run-env` + `slurm-node` skills.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-06-23-vlm-collator-mbs-design.md`.
- Only two files change: `cosmos_framework/configs/base/vlm/experiment/dataflow_roles.py` (`VLMProcessor`, `VLMCollator`). No batcher/model edits.
- Right-padding only (causal attention + `labels=-100` correctness depends on it).
- Padded length is always rounded up: `max_seq_length = (max_len + 15) // 16 * 16`.
- `IGNORE_INDEX = -100` (from `cosmos_framework.utils.vlm.constant`).
- SPDX header on every new file; `from __future__ import annotations`; test files named `*_test.py` co-located with the module.
- mbs>1 + `MapDistributor` resume is out of scope; validation recipes use `IterableDistributor`.

---

### Task 1: `VLMProcessor` emits per-sample constants

**Files:**
- Modify: `cosmos_framework/configs/base/vlm/experiment/dataflow_roles.py` (`VLMProcessor.__init__` lines 20–22, `VLMProcessor.process` lines 74–87)
- Test: `cosmos_framework/configs/base/vlm/experiment/dataflow_roles_test.py` (create)

**Interfaces:**
- Produces: `VLMProcessor.process(item)` returns a dict that now additionally contains
  `"token_mask"` (1-D bool tensor), `"pad_token_id"` (int), and `"ignore_index"` (int),
  alongside the existing `"input_ids"`, `"labels"`, and `PROCESSOR_KEYS_TO_ADD` keys.
  `VLMProcessor.__init__` resolves `self._pad_token_id: int` from the processor's tokenizer.

- [ ] **Step 1: Write the failing test**

Create `cosmos_framework/configs/base/vlm/experiment/dataflow_roles_test.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import torch

from cosmos_framework.configs.base.vlm.experiment.dataflow_roles import (
    VLMProcessor,
    VLMCollator,
)
from cosmos_framework.utils.vlm.constant import IGNORE_INDEX


class _FakeTok:
    pad_token_id = 7


class _FakeProcessor:
    tokenizer = _FakeTok()

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return {
            "input_ids": torch.tensor([1, 2, 3, 4], dtype=torch.long),
            "attention_mask": torch.ones(4, dtype=torch.bool),
        }

    def add_assistant_tokens_mask(self, input_ids):
        # first two tokens are prompt (masked), last two are assistant (kept)
        return torch.tensor([False, False, True, True])


def _item():
    return {
        "conversations": [
            {"from": "human", "value": "hello"},
            {"from": "gpt", "value": "world"},
        ],
        "image": None,
    }


def test_processor_emits_per_sample_constants():
    proc = VLMProcessor(processor=_FakeProcessor())
    out = proc.process(_item())

    assert out["pad_token_id"] == 7
    assert out["ignore_index"] == IGNORE_INDEX
    assert "token_mask" in out
    assert out["token_mask"].dtype == torch.bool
    # labels are -100 where token_mask is False
    assert out["labels"].tolist() == [IGNORE_INDEX, IGNORE_INDEX, 3, 4]
```

- [ ] **Step 2: Run test to verify it fails**

Run (in the i4 container — see `cosmos3-run-env` skill):
`pytest cosmos_framework/configs/base/vlm/experiment/dataflow_roles_test.py::test_processor_emits_per_sample_constants -v`
Expected: FAIL with `KeyError: 'pad_token_id'`.

- [ ] **Step 3: Resolve the pad id in `__init__`**

Replace `VLMProcessor.__init__` (lines 20–22):

```python
    def __init__(self, processor: Any, ignore_index: int = IGNORE_INDEX) -> None:
        self._processor = processor
        self._ignore_index = ignore_index
        # Resolve pad token id once; VLMCollator uses it to right-pad input_ids.
        tok = getattr(processor, "tokenizer", processor)
        pad_id = getattr(tok, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(tok, "eos_token_id", 0)
        self._pad_token_id = int(pad_id)
```

- [ ] **Step 4: Emit the keys in `process`**

Replace the `result` construction in `VLMProcessor.process` (lines 83–87):

```python
        result: dict = {
            "input_ids": input_ids,
            "labels": labels,
            "token_mask": token_mask,
            "pad_token_id": self._pad_token_id,
            "ignore_index": self._ignore_index,
        }
        for key in PROCESSOR_KEYS_TO_ADD:
            if key in inputs and inputs[key] is not None:
                result[key] = inputs[key]
        return result
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest cosmos_framework/configs/base/vlm/experiment/dataflow_roles_test.py::test_processor_emits_per_sample_constants -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add cosmos_framework/configs/base/vlm/experiment/dataflow_roles.py \
        cosmos_framework/configs/base/vlm/experiment/dataflow_roles_test.py
git commit -m "feat(vlm): VLMProcessor emits pad_token_id/ignore_index/token_mask per sample"
```

---

### Task 2: `VLMCollator` pad-and-stack parity port

**Files:**
- Modify: `cosmos_framework/configs/base/vlm/experiment/dataflow_roles.py` (top-of-file imports; `VLMCollator.collate` lines 94–114)
- Test: `cosmos_framework/configs/base/vlm/experiment/dataflow_roles_test.py` (append)

**Interfaces:**
- Consumes: sample dicts from Task 1 (`input_ids`, `labels`, `token_mask`, `attention_mask`,
  `pad_token_id`, `ignore_index`, optional vision keys).
- Produces: `VLMCollator.collate(samples)` returns a batch dict with `input_ids`/`labels`/
  `attention_mask`/`token_mask` shaped `[B, L]` (`L` a multiple of 16, right-padded),
  vision keys (`pixel_values`, `pixel_values_videos`, `image_grid_thw`, `video_grid_thw`,
  `second_per_grid_ts`, `image_sizes`) flat-concatenated on dim 0, `sample_worker_id`/
  `sample_epoch`/`sample_index` length-B, and `collated=True`.

- [ ] **Step 1: Write the failing tests**

Append to `cosmos_framework/configs/base/vlm/experiment/dataflow_roles_test.py`:

```python
def _sample(n: int, pad_id: int = 0, vision: bool = False, img_tokens: int = 4):
    s = {
        "input_ids": torch.arange(1, n + 1, dtype=torch.long),
        "labels": torch.arange(1, n + 1, dtype=torch.long),
        "attention_mask": torch.ones(n, dtype=torch.bool),
        "token_mask": torch.ones(n, dtype=torch.bool),
        "pad_token_id": pad_id,
        "ignore_index": IGNORE_INDEX,
    }
    if vision:
        s["pixel_values"] = torch.randn(img_tokens, 8)
        s["image_grid_thw"] = torch.tensor([[1, 2, 2]])
    return s


def test_collate_bs1_padded_to_multiple_of_16():
    out = VLMCollator().collate([_sample(5, pad_id=9)])
    assert out["input_ids"].shape == (1, 16)        # 5 -> 16
    assert out["labels"].shape == (1, 16)
    # real tokens preserved, then pad_token_id
    assert out["input_ids"][0, :5].tolist() == [1, 2, 3, 4, 5]
    assert out["input_ids"][0, 5:].eq(9).all()
    # labels padded with ignore_index
    assert out["labels"][0, 5:].eq(IGNORE_INDEX).all()
    assert out["sample_worker_id"].shape == (1,)
    assert out["collated"] is True


def test_collate_bs4_shapes_and_right_pad_fill():
    samples = [_sample(3, pad_id=9), _sample(20, pad_id=9),
               _sample(7, pad_id=9), _sample(12, pad_id=9)]
    out = VLMCollator().collate(samples)
    assert out["input_ids"].shape == (4, 32)        # max 20 -> 32
    # attention_mask / token_mask padded with False past real length
    assert out["attention_mask"][0, :3].all()
    assert (~out["attention_mask"][0, 3:]).all()
    # row 1 (len 20) real tokens intact
    assert out["input_ids"][1, :20].tolist() == list(range(1, 21))
    assert out["input_ids"][1, 20:].eq(9).all()
    # length-B resume meta
    assert out["sample_epoch"].tolist() == [0, 0, 0, 0]
    assert out["sample_index"].shape == (4,)


def test_collate_vision_flat_concat():
    samples = [_sample(5, vision=True, img_tokens=4),
               _sample(6, vision=True, img_tokens=9)]
    out = VLMCollator().collate(samples)
    # pixel_values concatenated on dim 0 (4 + 9), NOT stacked on a batch axis
    assert out["pixel_values"].shape[0] == 13
    # image_grid_thw concatenated: 2 rows of [1,2,2]
    assert out["image_grid_thw"].shape == (2, 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest cosmos_framework/configs/base/vlm/experiment/dataflow_roles_test.py -k collate -v`
Expected: FAIL — `test_collate_bs4_*` fails at the `assert len(samples) == 1` in the current collator.

- [ ] **Step 3: Add the `default_collate` import**

At the top of `dataflow_roles.py`, below `import torch` (line 11), add:

```python
from torch.utils.data._utils.collate import default_collate
```

- [ ] **Step 4: Replace `VLMCollator.collate`**

Replace the body of `VLMCollator.collate` (lines 94–114) with:

```python
    def collate(self, samples: list[dict]) -> dict:
        # Parity with i4 custom_collate: skip if already collated.
        if samples and samples[0].get("collated"):
            return samples[0]

        # All sequence tensors must be 1-D per sample before padding/stacking.
        for key in ("input_ids", "token_mask", "attention_mask", "labels"):
            assert all(s[key].ndim == 1 for s in samples if key in s), (
                f"VLMCollator: {key} must be 1-D per sample"
            )

        # Right-pad target length, rounded up to a multiple of 16 (FP8 support).
        max_seq_length = max(s["input_ids"].shape[0] for s in samples)
        max_seq_length = (max_seq_length + 15) // 16 * 16

        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        batch_size = len(samples)

        regular: dict = {}
        special: dict = {}

        def _pad_stack(key: str, fill, dtype) -> torch.Tensor:
            rows = []
            for s in samples:
                t = s[key]
                pad = torch.full((max_seq_length - t.shape[0],), fill, dtype=dtype)
                rows.append(torch.cat([t, pad]))
            return torch.stack(rows, dim=0)

        # input_ids: pad with each sample's pad_token_id.
        regular["input_ids"] = torch.stack(
            [
                torch.cat([
                    s["input_ids"],
                    torch.full((max_seq_length - s["input_ids"].shape[0],),
                               s["pad_token_id"], dtype=torch.long),
                ])
                for s in samples
            ],
            dim=0,
        )

        # token_mask / attention_mask: pad with False.
        for key in ("token_mask", "attention_mask"):
            if all(key in s for s in samples):
                regular[key] = _pad_stack(key, False, torch.bool)

        # labels: pad with each sample's ignore_index.
        regular["labels"] = torch.stack(
            [
                torch.cat([
                    s["labels"],
                    torch.full((max_seq_length - s["labels"].shape[0],),
                               s["ignore_index"], dtype=torch.long),
                ])
                for s in samples
            ],
            dim=0,
        )

        # raw_image / raw_video: keep per-sample, per-item boundaries (parity).
        if any("raw_image" in s for s in samples):
            ri: list = []
            for s in samples:
                img = s.get("raw_image", [])
                if isinstance(img, torch.Tensor):
                    if img.ndim == 3:
                        img = img[:, None]
                    img = [img[:, i:i + 1] for i in range(img.shape[1])]
                ri.append(img)
            regular["raw_image"] = ri
        if any("raw_video" in s for s in samples):
            rv: list = []
            for s in samples:
                vid = s.get("raw_video", [])
                if isinstance(vid, torch.Tensor):
                    vid = [vid]
                rv.append(vid)
            regular["raw_video"] = rv

        # Vision tensors: flat-concatenate on dim 0 (Qwen3-VL addresses them via
        # placeholder tokens in input_ids, not by batch position).
        vision_cat_keys = (
            "image_grid_thw", "video_grid_thw", "second_per_grid_ts",
            "pixel_values", "pixel_values_videos", "image_sizes",
        )
        all_keys = {k for s in samples for k in s}
        for key in all_keys:
            if key in regular:
                continue
            if key in vision_cat_keys:
                special[key] = torch.cat([s[key] for s in samples if key in s], dim=0)
            else:
                regular[key] = default_collate([s[key] for s in samples])

        batch = {**regular, **special, "collated": True}
        # Resume meta (streaming source has no position -> zeros), length-B.
        batch["sample_worker_id"] = torch.tensor([worker_id] * batch_size)
        batch["sample_epoch"] = torch.tensor([0] * batch_size)
        batch["sample_index"] = torch.tensor([0] * batch_size)
        return batch
```

Also update the `VLMCollator` docstring (lines 91–92) to reflect pad-and-stack for any batch size.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest cosmos_framework/configs/base/vlm/experiment/dataflow_roles_test.py -v`
Expected: PASS (all 4 tests).

- [ ] **Step 6: Commit**

```bash
git add cosmos_framework/configs/base/vlm/experiment/dataflow_roles.py \
        cosmos_framework/configs/base/vlm/experiment/dataflow_roles_test.py
git commit -m "feat(vlm): VLMCollator supports max_batch_size>1 (i4-parity pad-and-stack)"
```

---

### Task 3: Enable and validate `max_batch_size=4` trainings

**Files:**
- Modify: `examples/toml/sft_config/llava_ov.toml:107` (`max_samples_per_batch = 1` -> `4`)
- (No file edit for videophy2 — override `max_batch_size` via Hydra CLI.)

**Interfaces:**
- Consumes: the working `VLMProcessor` + `VLMCollator` from Tasks 1–2.
- Produces: two completed 200-step training runs (videophy2, llava_ov) at batch size 4 with fresh wandb runs, demonstrating real multi-sample batches.

- [ ] **Step 1: GPU smoke check (fail fast)**

Use the `cosmos3-run-env` skill to author a run wrapper and the `slurm-node` skill to
execute it in the `bob_echo_dev` i4 container. Launch llava_ov for **5 iterations** at
batch size 4:

```
python -m cosmos_framework.scripts.train \
    --sft-toml examples/toml/sft_config/llava_ov.toml -- \
    data_setting.max_tokens=16000 \
    dataloader_train.batcher.max_batch_size=4 \
    trainer.max_iter=5 trainer.logging_iter=1 \
    checkpoint.load_path="" job.wandb_mode=disabled
```

Expected: runs 5 steps without crashing; the `log_tensor_shape` callback prints
`input_ids` with a leading dim of `4` (or fewer only if the pool drained), not `1`.
If it crashes on a vision-token mismatch, stop — that means `pad_token_id` collides with a
vision placeholder id (see spec §7) and Task 1's pad-id resolution must be revisited.

- [ ] **Step 2: Set llava_ov TOML to batch size 4**

Edit `examples/toml/sft_config/llava_ov.toml` line 107:

```toml
max_samples_per_batch = 4
```

- [ ] **Step 3: Run the videophy2 validation training**

Via `cosmos3-run-env` + `slurm-node`, launch `videophy2_sft_nano` with the overrides
(its batcher `max_batch_size` is hardcoded to 1 at `videophy2_sft_nano.py:124`, so override
on the CLI). Use the recipe's launch shell (`examples/launch_sft_videophy2_nano.sh`) or a
direct `train` invocation with:

```
trainer.max_iter=200 trainer.logging_iter=1 \
dataloader_train.batcher.max_batch_size=4 \
job.wandb_mode=online job.project=cosmos3_reasoner \
job.group=mbs_validation job.name=videophy2_mbs4_2026-06-23
```

Expected: reaches iteration 200 without crashing; wandb shows a fresh run logging every
iteration; loss is finite and trends down.

- [ ] **Step 4: Run the llava_ov validation training**

Via `cosmos3-run-env` + `slurm-node`:

```
python -m cosmos_framework.scripts.train \
    --sft-toml examples/toml/sft_config/llava_ov.toml -- \
    data_setting.max_tokens=16000 \
    trainer.max_iter=200 trainer.logging_iter=1 \
    checkpoint.load_path="" \
    job.wandb_mode=online job.project=cosmos3 \
    job.group=mbs_validation job.name=llava_ov_mbs4_2026-06-23
```

(`max_samples_per_batch=4` now comes from the TOML edit in Step 2.)
Expected: reaches iteration 200 without crashing; fresh wandb run, per-iteration logging;
loss finite and trending down.

- [ ] **Step 5: Confirm success criteria and record run URLs**

For BOTH runs verify: (a) 200 steps completed, no crash; (b) `log_tensor_shape` shows
`input_ids` leading dim `> 1` (batches actually pack multiple samples); (c) loss finite and
decreasing. Paste the two wandb run URLs into the PR description.

- [ ] **Step 6: Commit the TOML change**

```bash
git add examples/toml/sft_config/llava_ov.toml
git commit -m "chore(vlm): set llava_ov SFT max_samples_per_batch=4"
```

---

## Self-Review

**Spec coverage:**
- §5.1 processor emits `pad_token_id`/`ignore_index`/`token_mask` → Task 1. ✓
- §5.2 collate pad+stack, ×16, flat vision cat, raw_image/raw_video, length-B meta, `collated` → Task 2. ✓
- §5.2 step 2 assert replaces `len==1` → Task 2 Step 4. ✓
- §6 no batcher/model edits → enforced by Global Constraints; no task touches them. ✓
- §8.1 unit tests (B=1 padded, B=4, mixed lengths, fill values, flat cat, meta) → Task 2 Steps 1–5 (B=1, B=4 incl. mixed lengths, vision cat, meta). ✓
- §8.2 GPU smoke run → Task 3 Step 1. ✓
- §8.3 mbs=4 validation trainings (videophy2 + llava_ov, max_iter=200, logging_iter=1, fresh wandb) → Task 3 Steps 2–5. ✓
- §7 pad_token_id vs vision placeholder collision risk → surfaced as the smoke-run failure mode in Task 3 Step 1. ✓

**Placeholder scan:** No TBD/TODO; all code shown in full; commands have expected output.

**Type consistency:** `pad_token_id` is an int in the sample dict (Task 1) and consumed via `torch.full(..., s["pad_token_id"], dtype=torch.long)` (Task 2). `token_mask`/`attention_mask` are 1-D bool tensors in both. `collate` returns `collated=True` (bool) — test asserts `is True`. Meta keys are length-B tensors. Consistent across tasks.
