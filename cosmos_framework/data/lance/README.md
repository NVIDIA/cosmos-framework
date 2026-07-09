# LanceDB-powered Cosmos Dataloaders

This directory contains LanceDB-backed implementations of the three main dataloaders used in Cosmos training:
- **Action (DROID/LeRobot)**: `LanceDROIDComposedDataset`
- **Vision-SFT (Local clips)**: `LanceVisionSFTDataset`
- **VLM (LLaVA-OneVision)**: `LanceVLMDataset`

Each is a drop-in for the corresponding base loader, reading from a converted LanceDB table
instead of the original source (LeRobot tree / local clips / HuggingFace stream). Output is
equivalent to the base — VLM records byte-identical, vision-SFT token-ids exact, action labels
(action/pose/caption) bit-exact, and video within one offline H.264 re-encode (~1.5%) — and the
table can be read directly from object storage (S3) without FUSE or full downloads.

## Performance Summary

Numbers below use a **327-episode subset of the public [`lerobot/droid_1.0.1`](https://huggingface.co/datasets/lerobot/droid_1.0.1)**
dataset (LeRobot v3.0; materialized via `tools/lance_datagen/prepare_droid_subset.py`), whose camera
views are 320×180 → 270×320 composed. Production DROID uses 640×360 views → 540×640 (see Dataset Size).

### Combined Throughput (samples/s)
Combined 3-loader throughput, 327 DROID episodes, batch 16:

| Workers (Action/VLM/VSFT) | Base (Local) | Lance (Local) | Base (S3) | Lance (S3) |
| ------------------------- | ------------ | ------------- | --------- | ---------- |
| 4/4/4 (Default)           | 86.5         | 249.4 (2.9x)  | 67.7      | 246.9 (3.6x)|
| 18/4/18 (Tuned)           | 253.7        | 961.9 (3.8x)  | 232.0     | 1016.9 (4.4x)|

### Per-Loader Throughput (samples/s)
Each loader standalone, tuned workers (Action/VSFT 18, VLM 4):

| Loader         | Base (Local) | Lance (Local) | Base (S3)  | Lance (S3)  |
| -------------- | ------------ | ------------- | ---------- | ----------- |
| Action (DROID) | 154.0        | 288.7 (1.9x)  | 146.7      | 307.2 (2.1x)|
| Vision-SFT     | 119.1        | 1017.9 (8.5x) | 100.4      | 959.5 (9.6x)|
| VLM (LLaVA)    | 118.1 (hf)   | 392.3 (3.3x)  | 118.1 (hf) | 328.6 (2.8x)|

Action decodes one composed clip instead of three runtime views; Vision-SFT decodes a pre-resized
short-GOP clip in-process instead of the base's per-sample ffmpeg resize. The VLM base has no
local/S3 form — it streams from the HuggingFace Hub (marked `hf`, so the same number appears in
both columns) — and the VLM row is measured end-to-end (image decode + tokenize) to be comparable
to the video-decoding loaders.

### Dataset Size (Action)
327 DROID episodes — original three views vs the composed Lance table:

| metric   | Original (3 views)     | Composed (Lance)               |
| -------- | ---------------------- | ------------------------------ |
| encoding | AV1, long-GOP          | H.264, all-intra (`gop=1`)     |
| streams  | 3 views @ 320×180 RGB  | 1 composed view @ 270×320 RGB  |
| size     | 1.47 GB                | 0.55 GB (0.37×)                |

The `3`/`1` are the number of video **streams** (three camera views vs one composed view), not
channels — every frame is RGB. The composed 270×320 frame is the wrist view on top of the two
half-size exterior views.

The composed table is ~2.7× smaller even though all-intra `gop=1` H.264 is *less* space-efficient
per pixel than the source's AV1 long-GOP: it stores one stream at reduced resolution (the two
exterior views are downscaled to half) rather than three full views, which outweighs the codec/GOP
cost. `gop=1` is a deliberate trade — exact, cheap random-window seeks in exchange for size (a
larger GOP would shrink the table further at some seek cost).

The composed resolution is derived from the source (`1.5×h × w`), not fixed: this public subset has
320×180 views → 270×320.

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

There are two phases: an offline conversion (`tools/lance_datagen/`) writes one LanceDB table per
modality, and the training-time loader reads that table and decodes clips in-process. Tables are
read through the lancedb Permutation API, and media is stored one clip/image per row in a plain
`large_binary` column. Loaders null their DB/decoder handles in `__getstate__`, so each spawn
worker reopens them lazily (lancedb is not fork-safe).

> Note: video is currently stored as plain `large_binary`. It will move to blob encoding
> (blob-v2) once the lancedb-level blob API is available.

### Action — `LanceDROIDComposedDataset`

This loader is a **hybrid**: it takes both `root` (the LeRobot tree) and `lance_uri`, and reads
the two halves of a sample from different places:
- **action/state/task labels + frame indexing** — from the **base LeRobot parquet** (`root/data/`,
  `root/meta/`), via the inherited `DROIDLeRobotDataset` (`_window_rows` → `_build_joint_action`).
  These columns are small and correctness-critical, so they stay in the parquet and the base's exact
  assembly is reused — labels are bit-exact.
- **composed video** — from the **Lance table**. The base otherwise decodes three camera views per
  sample and resizes + concatenates them into one 270×320 frame at runtime; the table stores that
  composed frame once per episode, so the loader decodes a single mp4 stream instead.

`LanceDROIDComposedDataset` subclasses `DROIDLeRobotDataset` and overrides only the video source.
Clips are encoded all-intra (`gop=1`), so torchcodec's `seek_mode="approximate"` lands on each
window exactly; a per-worker LRU cache keeps recently used episode decoders open. `take` returns
rows sorted by offset, so the byte read keys results by row rather than the requested order.

The loader reads only `episode_index` (to locate a clip) and `video_bytes`; `ep_start`/`length` are
episode metadata written at build time. (Vision-SFT and VLM tables, below, are self-contained —
their captions/conversations live in the table, so those loaders are not hybrids.)

| column          | type           | description                              |
| --------------- | -------------- | ---------------------------------------- |
| `episode_index` | int64          | episode id (used to locate the clip)     |
| `ep_start`      | int64          | first global frame index (build metadata) |
| `length`        | int64          | number of frames (build metadata)        |
| `video_bytes`   | large_binary   | composed 270×320 mp4 for the episode     |

### Vision-SFT — `LanceVisionSFTDataset`

The base `SFTDataset` fetches each source clip, decodes it at native size, and resizes it per
sample every epoch through an ffmpeg subprocess. The Lance table stores each clip already resized
to the training resolution with a short GOP, so the loader decodes fewer pixels in-process and
seeks windows cheaply. Caption selection and tokenization reuse the base code, so `text_token_ids`
are token-exact.

| column                | type         | description                          |
| --------------------- | ------------ | ------------------------------------ |
| `clip_id`             | string       | `{uuid}_w{window}`                   |
| `width`, `height`     | int64        | original resolution                  |
| `start_frame`, `end_frame` | int64   | window bounds                        |
| `temporal_interval`   | int64        | frame stride                         |
| `enc_h`, `enc_w`      | int64        | stored (resized) resolution          |
| `fps`                 | float64      | source fps                           |
| `caption_json`        | string       | structured caption (JSON) or `""`    |
| `caption`             | string       | dense caption fallback               |
| `video_bytes`         | large_binary | pre-resized clip mp4                  |

### VLM — `LanceVLMDataset`

The base streams LLaVA-OneVision from the HuggingFace Hub (sequential shards + a bounded shuffle
buffer). The Lance table stores each sample's image bytes and conversation; the Permutation API
reads them by row, so a global shuffle is just a shuffled list of row indices. `LanceVLMShuffleScan`
instead reads contiguous row-chunks in shuffled order for S3-friendly sequential access. Records are
byte-identical to the base, so downstream image decoding and tokenization are unchanged.

| column          | type         | description                     |
| --------------- | ------------ | ------------------------------- |
| `sample_id`     | string       | sample id                       |
| `image_bytes`   | large_binary | raw image (PNG/JPEG)            |
| `conversations` | string       | conversation turns (JSON)       |

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
