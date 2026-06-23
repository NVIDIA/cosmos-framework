# VLMCollator: support `max_batch_size > 1` (i4-parity collate)

**Date:** 2026-06-23
**Status:** Design — awaiting review
**Scope:** `cosmos_framework/configs/base/vlm/experiment/dataflow_roles.py` (2 classes) + validation runs

## 1. Problem & goal

`VLMCollator.collate` (`dataflow_roles.py:90–114`) only supports batches of exactly one
sample — it asserts `len(samples) == 1`, then `unsqueeze(0)`s the single sample's
sequence tensors and passes vision tensors through flat. It performs no padding or
stacking. As a result every VLM recipe is pinned to `max_batch_size=1`
(`videophy2_sft_nano.py:124`, `llava_ov_vlm.py:224,281`), and raising
`max_samples_per_batch` in a recipe TOML crashes at the assert.

The original imaginaire4 (i4) repo supported multi-sample VLM batches (e.g.
`pre_exp011_030_qwen3_vl_2b_vit2k8k_mbs8`, batch size 8) via `custom_collate`
(`imaginaire4/.../vlm/datasets/collate_fn.py:18–111`). That collate pads variable-length
sequences and stacks them on the batch axis (dynamic right-padding, NOT sequence
packing), while flat-concatenating vision tensors.

**Goal:** make `VLMCollator` produce correct training batches for any
`max_batch_size ≥ 1`, as a faithful port of i4's `custom_collate`, with all batch
sizes (including 1) routed through the same pad-and-stack path.

## 2. Background: how the dataflows correspond

Our dataflow is a re-homing of i4's. The batch-assembly engine is already ported and
already supports `max_batch_size > 1`:

| Stage | i4 | Ours |
| --- | --- | --- |
| Per-sample tokenize | `TokenizeData` augmentor | `VLMProcessor.process` |
| Batch assembly (bin-pack) | `_JointIterableDataset._best_fit_batch` | `PoolPackingBatcher._best_fit_batch` (`batchers.py:110–127`) — **already supports mbs>1** |
| Collate | `custom_collate` (1–8 samples) | `VLMCollator.collate` — **bs=1 only** ← this spec |

"mbs8" in i4 means **8 samples on the batch axis with dynamic right-padding**, not
sequence packing — no `cu_seqlens`, no document attention masks.

## 3. Key finding: the model path is already wired for this

Investigation of the current repo shows the model side already anticipates the
i4-parity collate output, so **no model changes are needed**:

- `HFModel.forward` (`hf_model.py:327–347`) strips
  `_COLLATE_NON_MODEL_KEYS = {token_mask, pad_token_id, ignore_index, collated,
  raw_image, raw_video, image_sizes}` (`hf_model.py:311–325`) before calling the HF
  model — exactly the extra keys the parity collate produces.
- `get_position_ids` / `get_rope_index_qwen3_vl` (`create_position_ids.py`) is already
  batched and padding-aware: it takes `attention_mask` `(B, N)`, drops padding per row
  before computing M-RoPE positions, and scatters them back into padded slots
  (`:86–201`).
- The forward computes `position_ids` from `attention_mask`, then pops the mask
  (`vlm_model.py:495–497, 532–534`). This is correct **specifically for right-padding +
  causal attention + `labels=-100` on pads**: real tokens never attend to pads (pads are
  causally in the future) and pad logits are loss-masked.
- `sample_worker_id` / `sample_epoch` / `sample_index` meta keys are already tolerated by
  the forward at bs=1 today; length-B versions are no different.

## 4. Design decisions (locked)

1. **Full i4 parity.** Reproduce `custom_collate` behavior, including `raw_image`/
   `raw_video` per-sample list handling, `token_mask`, the `collated` early-return guard,
   and FP8 ×16 length rounding.
2. **Processor emits per-sample constants.** `VLMProcessor.process` additionally emits
   `pad_token_id`, `ignore_index`, and `token_mask`, matching i4's per-sample-dict
   contract (collate reads `item[...]`).
3. **Unify all batch sizes through the parity path.** Even bs=1 goes through pad+stack and
   gets ×16 padding. This **changes existing mbs=1 numerics** (adds masked pad tokens) and
   requires re-validation of current recipes — accepted.
4. **Verification:** collator unit test + GPU smoke run + full mbs=4 validation trainings.

## 5. Changes

Both changes are in `cosmos_framework/configs/base/vlm/experiment/dataflow_roles.py`.

### 5.1 `VLMProcessor.process` (`dataflow_roles.py:74–87`)

Add three keys to the returned sample dict:

- `token_mask` — already computed at line 80; stop discarding it.
- `pad_token_id` — resolved once from the processor/tokenizer (e.g.
  `self._processor.tokenizer.pad_token_id`) in `__init__`, emitted per sample.
- `ignore_index` — `self._ignore_index` (already held), emitted per sample.

### 5.2 `VLMCollator.collate` (`dataflow_roles.py:90–114`)

Replace the bs=1-only body with the `custom_collate` port (single path, all batch sizes):

1. Optional early-return if `samples[0].get("collated")` (parity with i4).
2. Assert the four sequence keys (`input_ids`, `token_mask`, `attention_mask`, `labels`)
   are 1-D per sample. **This replaces the old `len(samples) == 1` assert.**
3. `max_seq_length = ((max_len + 15) // 16) * 16` (FP8 ×16 rounding; applies at bs=1 too).
4. Right-pad then `torch.stack` → `[B, L]`:
   - `input_ids` padded with each sample's `pad_token_id`
   - `labels` padded with each sample's `ignore_index` (−100)
   - `attention_mask` padded with `0`/`False`
   - `token_mask` padded with `0`/`False`
5. Flat-`cat` on dim 0 (no batch axis): `image_grid_thw`, `video_grid_thw`,
   `second_per_grid_ts`, `pixel_values`, `pixel_values_videos`, `image_sizes`.
6. `raw_image` / `raw_video` → per-sample lists (inert for our pipeline, which does not
   emit them; included for parity completeness).
7. Any remaining key → `default_collate`.
8. Stamp resume meta as **length-B** vectors: `sample_worker_id` (from
   `torch.utils.data.get_worker_info()`), `sample_epoch = 0`, `sample_index = 0` (the
   streaming "no position" sentinel). Add `collated = True`.

## 6. What is explicitly NOT changing (non-goals)

- **`PoolPackingBatcher`** — already supports `max_batch_size > 1`. No change.
- **`VLMModel` / `HFModel` / `get_position_ids`** — already batched + padding-aware and
  strip non-model keys. No change (validated by smoke run, not edited).
- **Resume at mbs>1 with `MapDistributor`.** `PoolPackingBatcher` reorders
  (`prefer_closest`), so the loader's resume-stamp guard (`loader.py:76–87`) raises on
  multi-sample reordered batches. The validation recipes (`llava_ov`, `videophy2`) use
  `IterableDistributor` (not resumable), so this is moot for them. Making reordered
  map-style resume work at mbs>1 is out of scope.

## 7. Edge cases & risks

- **`pad_token_id` vs vision placeholder id.** Padding must use a `pad_token_id` distinct
  from the Qwen3-VL image/video placeholder token id; otherwise padding would inject
  phantom vision placeholders and the vision-count check would crash. Distinct for
  Qwen3-VL; confirmed by the smoke run.
- **bs=1 numeric drift.** ×16 padding changes existing mbs=1 sequences (extra masked
  tokens). Loss is masked so real-token gradients are unaffected, but loss-curve
  comparisons against pre-change goldens are no longer byte-equal by design.
- **Single-modality batches.** `PoolPackingBatcher` never mixes modalities in a batch
  (`_get_modality`), so the flat vision-cat is always within one modality — consistent
  with the collate's assumptions.
- **Padded forward unexercised in our repo.** Today's bs=1 path never pads, so the
  padded-forward path (mask with zeros, ×16 lengths) is new here even though the code
  supports it. The smoke run + validation trainings are the gate.

## 8. Testing & validation

### 8.1 Unit tests (no GPU)
- `VLMProcessor` emits `token_mask`, `pad_token_id`, `ignore_index`.
- `VLMCollator` over synthetic samples (B=1 padded ×16, B=4, mixed lengths):
  - shapes `[B, L]` with `L` a multiple of 16;
  - right-pad fill values: `pad_token_id` in `input_ids`, `-100` in `labels`, `0` in
    `attention_mask`/`token_mask`;
  - real-token slots unchanged vs input;
  - vision tensors flat-`cat`: `pixel_values.shape[0] == sum(per-sample)`;
  - `sample_*` meta length-B; `collated=True`.

### 8.2 GPU smoke run
Quick mbs>1 forward+backward on a slurm node (i4 container) to confirm position ids,
vision merge, and loss run before the full trainings.

### 8.3 Validation trainings (acceptance)
Run both recipes after implementation:

| Recipe | `max_batch_size` | `max_iter` | `logging_iter` | wandb |
| --- | --- | --- | --- | --- |
| videophy2 (`videophy2_sft_nano`) | 4 | 200 | 1 | fresh run, enabled |
| llava_ov (`pre_exp012_llava_ov` via `llava_ov.toml`) | 4 | 200 | 1 | fresh run, enabled |

- llava_ov: set `max_samples_per_batch = 4` in `examples/toml/sft_config/llava_ov.toml`
  (routed to `batcher.max_batch_size` via `PATH_REMAPS["vlm"]`).
- videophy2: batcher `max_batch_size` is hardcoded to 1 (`videophy2_sft_nano.py:124`) —
  override to 4 via Hydra CLI or a temp edit (plan picks one).
- Enable wandb (both currently `wandb_mode="disabled"`) with new run names; WANDB key via
  the `cosmos3-run-env` skill.
- **Success criteria:** both reach 200 steps without crashing; loss finite and trending
  down; `log_tensor_shape` shows `input_ids` is `[4, L]` (batches actually pack 4
  samples), not `[1, L]`.

## 9. Reference

- i4 collate: `imaginaire4/projects/cosmos3/vlm/datasets/collate_fn.py:18–111`
- i4 experiment: `imaginaire4/projects/cosmos3/vfm/configs/base/vlm/experiment/pre_exp01x.py:165–205`
- Our collator/processor: `cosmos_framework/configs/base/vlm/experiment/dataflow_roles.py`
- Our batcher: `cosmos_framework/data/vfm/dataflow/batchers.py:41–181`
- Our forward: `cosmos_framework/model/vfm/vlm_model.py:483–536`, `hf_model.py:311–347`
- Position ids: `cosmos_framework/utils/vfm/vlm/create_position_ids.py`
