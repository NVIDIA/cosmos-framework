# LanceDB-powered Cosmos Dataloaders

This directory contains LanceDB-backed implementations of the three main dataloaders used in Cosmos training:
- **Action (DROID/LeRobot)**: `LanceDROIDComposedDataset`
- **Vision-SFT (Local clips)**: `LanceVisionSFTDataset`
- **VLM (LLaVA-OneVision)**: `LanceVLMDataset`

Each is a drop-in for the corresponding base loader, reading from a converted LanceDB table
instead of the original source (LeRobot tree / local clips / HuggingFace stream). Output is
equivalent to the base — VLM records byte-identical, vision-SFT token-ids exact, action labels
(action/pose/caption) bit-exact, and video within one offline H.264 re-encode (< 2.5% pixel MAD) — and the
table can be read directly from object storage (S3) without FUSE or full downloads.

## Performance Summary

Numbers below use a **327-episode subset of the public [`lerobot/droid_1.0.1`](https://huggingface.co/datasets/lerobot/droid_1.0.1)**
dataset (LeRobot v3.0; materialized via `tools/lance_datagen/prepare_droid_subset.py` into a
version-named root the base loader's registry resolves), whose camera views are 320×180 →
270×320 composed. Production DROID uses 640×360 views → 540×640 (see Dataset Size).

### Combined Throughput (samples/s)
Combined 3-loader throughput, 327 DROID episodes, batch 16:

| Workers (Action/VLM/VSFT) | Base (Local) | Lance (Local) | Base (S3) | Lance (S3) |
| ------------------------- | ------------ | ------------- | --------- | ---------- |
| 4/4/4 (Default)           | 87.4         | 225.7 (2.6x)  | 69.1      | 228.2 (3.3x)|
| 18/4/18 (Tuned)           | 251.2        | 947.5 (3.8x)  | 232.4     | 970.5 (4.2x)|

### Per-Loader Throughput (samples/s)
Each loader standalone, tuned workers (Action/VSFT 18, VLM 4):

| Loader         | Base (Local) | Lance (Local) | Base (S3)  | Lance (S3)  |
| -------------- | ------------ | ------------- | ---------- | ----------- |
| Action (DROID) | 143.2        | 269.1 (1.9x)  | 131.9      | 268.7 (2.0x)|
| Vision-SFT     | 120.7        | 1016.4 (8.4x) | 102.4      | 875.8 (8.6x)|
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

## Memory

Memory is not a differentiator in either direction. The current base loader is index-light —
lazy per-shard LeRobot readers behind an LRU, metadata-only init, near-zero spawn payload — and
the Lance loader is comparable: at 1× (87.6k samples, 8 spawn workers) per-worker PSS is
~0.60 GB (base) vs ~0.71 GB (Lance; it holds the compact label arrays in memory plus the
torchcodec decoder cache, trading a little RSS for not touching the LeRobot tree at all).
The Lance wins are **throughput and native S3**, not memory.

(Historical note: the pre-2026-07 base materialized a per-frame dict index that scaled to a
~12 GB resident index at DROID scale; the upstream rewrite to lazy LeRobot readers fixed that
wholesale, so earlier memory-scaling comparisons against it are obsolete.)

## How it works

There are two phases: an offline conversion (`tools/lance_datagen/`) writes one LanceDB table per
modality, and the training-time loader reads that table and decodes clips in-process. Tables are
read through the lancedb Permutation API, and media is stored one clip/image per row in a plain
`large_binary` column. Loaders null their DB/decoder handles in `__getstate__`, so each spawn
worker reopens them lazily (lancedb is not fork-safe).

> Note: video is currently stored as plain `large_binary`. It will move to blob encoding
> (blob-v2) once the lancedb-level blob API is available.

### Action — `LanceDROIDComposedDataset`

Fully Lance-backed: labels **and** video come from LanceDB, so the loader takes only a
`lance_uri` — no LeRobot tree at train time. The base loader queries lazy per-shard LeRobot
readers for each sample's label windows and three camera-view windows, then resizes + concatenates
the views into one composed frame at runtime; the converter stores that composed frame once per
episode, plus the per-frame labels dumped verbatim from the base's LeRobot table.
`LanceDROIDComposedDataset` subclasses `DROIDLeRobotDataset`: the train/val split and episode-span
index are built with the base's own helpers (`split_episode_ids` / `build_episode_spans`), and
`_fetch_sample` assembles the same windowed sample dict the LeRobot readers would return — so the
inherited `__getitem__` (pose math, gripper handling, action assembly for every action space) runs
unchanged. Labels are bit-exact for the same split parameters; video is within one offline H.264
re-encode plus the base's own decoder-backend difference (< 2.5% pixel MAD). A `version` parameter
selects the same per-dataset feature config the base resolves from its root name.

Clips are encoded all-intra (`gop=1`), so torchcodec's `seek_mode="approximate"` lands on each
window exactly; a per-worker LRU cache keeps recently used episode decoders open. `take` returns
rows sorted by offset, so the byte read keys results by row rather than the requested order.

Four tables (one video + three label tables, named `{table}` / `{table}_*`):

`droid_composed` — one row per episode:

| column          | type           | description                              |
| --------------- | -------------- | ---------------------------------------- |
| `episode_index` | int64          | episode id (used to locate the clip)     |
| `ep_start`      | int64          | first global frame index (build metadata) |
| `length`        | int64          | number of frames (build metadata)        |
| `video_bytes`   | large_binary   | composed 270×320 mp4 for the episode     |

`droid_composed_frames` — one row per frame (feature names store `.` as `__`):

| column | type | description |
| --- | --- | --- |
| `episode_index`, `task_index` | int64 | frame → episode/task |
| `timestamp` | float64 | frame timestamp |
| `action__joint_position` | fixed_list<float32>[7] | commanded joints |
| `action__gripper_position` | float32 | commanded gripper |
| `observation__state__joint_positions` | fixed_list<float32>[7] | observed joints |
| `observation__state__gripper_position` | float32 | observed gripper |
| `observation__state__cartesian_position` | fixed_list<float32>[6] | EE pose (ee_pose space) |

`droid_composed_tasks` (`task_index` int64, `task` string) and
`droid_composed_episodes` (`episode_index` int64, `episode_id` string — for keep-ranges
filtering) complete the label set.

### Vision-SFT — `LanceVisionSFTDataset`

The base `SFTDataset` fetches each source clip, decodes it at native size, and resizes it per
sample every epoch through an ffmpeg subprocess. The Lance table stores each clip already resized
to the training resolution with a short GOP, so the loader decodes fewer pixels in-process and
seeks windows cheaply. Caption selection, post-processing (CFG dropout, duration/resolution
conditioning suffixes), and tokenization reuse the base code, so `text_token_ids` are token-exact;
the converter uses the base's own metadata load (same duration/min-frames filters).

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
