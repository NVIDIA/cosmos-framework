# LanceDB-powered Cosmos Dataloaders

This directory contains LanceDB-backed implementations of the three main dataloaders used in Cosmos training:
- **Action (DROID/LeRobot)**: `LanceDROIDComposedDataset`
- **Vision-SFT (Local clips)**: `LanceVisionSFTDataset`
- **VLM (LLaVA-OneVision)**: `LanceVLMDataset`

These loaders are designed for higher throughput and native object-store (S3) access while maintaining verified equivalence with the original loaders (exact labels/tokens; video within one offline H.264 re-encode).

## Key Features

- **Higher Throughput**: Up to 3.8x speedup locally and 4.4x on S3 when tuned.
- **Native S3 Support**: Uses LanceDB's native object-store integration for parallel, selective reads without FUSE or full downloads.
- **Verified Equivalence**: VLM records byte-identical, vision-SFT token-ids exact, action labels (action/pose/caption) bit-exact with video within H.264 re-encode tolerance (~1.5%).

## Performance Summary

### Combined Throughput (samples/s)
Combined 3-loader throughput, 327 DROID episodes, batch 16:

| Workers (Action/VLM/VSFT) | Base (Local) | Lance (Local) | Base (S3) | Lance (S3) |
| ------------------------- | ------------ | ------------- | --------- | ---------- |
| 4/4/4 (Default)           | 86.5         | 249.4 (2.9x)  | 67.7      | 246.9 (3.6x)|
| 18/4/18 (Tuned)           | 253.7        | 961.9 (3.8x)  | 232.0     | 1016.9 (4.4x)|

### Per-Loader Throughput (samples/s)
Each loader standalone, tuned workers (Action/VSFT 18, VLM 4):

| Loader         | Base (Local) | Lance (Local) | Base (S3) | Lance (S3)  |
| -------------- | ------------ | ------------- | --------- | ----------- |
| Action (DROID) | 154.0        | 288.7 (1.9x)  | 146.7     | 307.2 (2.1x)|
| Vision-SFT     | 119.1        | 1017.9 (8.5x) | 100.4     | 959.5 (9.6x)|
| VLM (LLaVA)    | 190.0        | — (see note)  | 179.6     | — (see note)|

Action decodes 1 composed clip vs 3 runtime views (~2x); Vision-SFT decodes a pre-resized
short-GOP clip in-process vs the base's per-sample ffmpeg resize (~8x). **VLM is not a decode
comparison**: both loaders emit *raw* records (image bytes + conversation) and decode/tokenize
downstream in the processor, so loader-level throughput mainly reflects the source (base streams
from the HF Hub; Lance is local/S3 random access). The fair VLM measure is the Combined table.

## Memory: a note on the per-frame index (not a Lance advantage)

`ActionBaseDataset.__init__` builds a per-frame index (`self._rows`, a list of row dicts) and
ships it to every DataLoader worker. `DROIDLeRobotDataset` **never reads it** — it indexes via
compact column arrays and reconstructs window rows on demand — so the Lance loader drops it
(`self._rows = None`), which we verified is **output-neutral (bit-identical batches)**.

That accounts for most of the per-worker memory gap vs the *shipped* base (~2.7× at 1.5M
frames). **It is not a fundamental Lance advantage, though** — `_rows` is a freeable redundancy
the base could drop too. Once it does, memory is at parity (a `_rows`-freed base is ~0.70 GB vs
Lance ~0.98 GB per worker at 16×; Lance carries the torchcodec decoder cache). The genuine Lance
wins are **throughput and S3**, not memory. _(We've raised this upstream to confirm `_rows` is
safe to drop for `DROIDLeRobotDataset`.)_

What `_rows` costs — per-worker spawn payload (327-episode DROID subset replicated N×):

| Dataset Size       | base keeps `_rows` | base drops `_rows` |
| ------------------ | ------------------ | ------------------ |
| 96k frames (1×)    | 37 MB              | 11 MB              |
| 1.54M frames (16×) | 552 MB             | 133 MB             |
| 3.08M frames (32×) | 1.1 GB             | 263 MB             |
| 6.16M frames (64×) | 2.2 GB             | 524 MB             |

`_rows` ≈ 270 B/frame; the remaining ~85 B/frame is the compact arrays both keep. A base that
keeps `_rows` reaches a ~12 GB resident index at 64× (OOM territory at full-DROID scale).

## How it works

Two phases. An offline **build** (`tools/lance_datagen/`) writes one LanceDB table per modality;
at train time the **loader** reads columns and decodes clips in-process. The win is moving
per-epoch work (multi-view compose, resize, subprocess decode) offline into the table, so the
hot path just does a columnar read + one in-process decode.

Shared mechanisms:
- **Permutation API** (lancedb, no pylance): columnar `take` for O(1) random access + true global
  shuffle. `take` returns rows sorted by offset, so `_read_clip_bytes` keys results by row rather
  than relying on input order.
- **Media as plain `large_binary`**: one mp4 clip (or image) per row. _TODO: move to blob-v2 for
  larger per-row payloads read in parallel._
- **torchcodec** decodes the mp4 bytes **in-process** (no ffmpeg subprocess) with
  `seek_mode="approximate"` — which is exact because clips are encoded **all-intra (`gop=1`, every
  frame a keyframe)**, making random window seeks cheap. Each worker keeps an LRU decoder cache.
- **Worker-safe**: `__getstate__` nulls the DB/decoder handles (lancedb isn't fork-safe), so each
  spawn worker reopens them lazily.

### Action — `LanceDROIDComposedDataset`
- **Current base**: decodes 3 camera views per sample from the LeRobot tree, resizes, and
  concatenates them at runtime (→ 270×320).
- **Lance**: stores that composed 270×320 clip **once per episode**, so the loader decodes a single
  stream. It **subclasses `DROIDLeRobotDataset`** — inheriting the frame indexing and action/pose
  assembly — and overrides only where the video comes from (labels stay bit-exact).
- **Schema**: `episode_index (int64)`, `video_bytes (large_binary)`.

### Vision-SFT — `LanceVisionSFTDataset`
- **Current base**: `SFTDataset` fetches each source clip, decodes at native size, and resizes it
  **per sample every epoch** via an ffmpeg subprocess.
- **Lance**: stores each clip **pre-resized to training resolution** with a short GOP, so the hot
  path decodes fewer pixels in-process with cheap seeks. Reuses the base's caption selection and
  tokenization, so `text_token_ids` are token-exact.
- **Schema**: `clip_id`, `width/height`, `start_frame/end_frame/temporal_interval`, `enc_h/enc_w`,
  `fps`, `caption_json`, `caption`, `video_bytes (large_binary)`.

### VLM — `LanceVLMDataset`
- **Current base**: LLaVA-OneVision streamed from the HuggingFace Hub (sequential shards + a bounded
  shuffle buffer).
- **Lance**: image bytes + conversation per row → O(1) random access and true global shuffle via the
  Permutation API; `LanceVLMShuffleScan` does a chunked-shuffle columnar scan for S3-friendly
  sequential reads. Records are byte-identical to the base.
- **Schema**: `sample_id (str)`, `image_bytes (large_binary)`, `conversations (JSON str)`.

## Usage

### 1. Build Tables
The conversion scripts live in [`tools/lance_datagen/`](../../../../tools/lance_datagen) (VLM uses
`convert_llava_to_lance` in [`vlm_dataset.py`](./vlm_dataset.py)):
```bash
# Action      — tools/lance_datagen/build_composed_droid.py
python tools/lance_datagen/build_composed_droid.py --root <droid_root> --uri <lance_uri> --gop 1

# Vision-SFT  — tools/lance_datagen/build_vision_sft.py
python tools/lance_datagen/build_vision_sft.py --jsonl <metadata.jsonl> --uri <lance_uri>

# VLM         — convert_llava_to_lance() in cosmos_framework/data/lance/vlm_dataset.py
python -c "from datasets import load_dataset; from cosmos_framework.data.lance.vlm_dataset import convert_llava_to_lance; \
convert_llava_to_lance(load_dataset('lmms-lab/LLaVA-OneVision-Data', name='<subset>', split='train', streaming=True), '<lance_uri>')"
```
`tools/lance_datagen/prepare_droid_subset.py` materializes a Cosmos-canonical DROID subset from the public LeRobot release.

### 2. Integration
Replace the standard datasets with their Lance counterparts in your configuration.
```python
from cosmos_framework.data.lance import LanceDROIDComposedDataset, LanceVisionSFTDataset, LanceVLMDataset
```

## Testing
Run equivalence tests to verify parity with base loaders:
```bash
pytest tests/data/lance/
```
