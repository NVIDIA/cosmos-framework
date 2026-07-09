# Design: LanceDB-backed Cosmos Dataloaders

This document walks through the implementation of the three Lance loaders — what each
stores, how the converters build it, how the loaders read it, and the invariants that
keep their output equivalent to the base loaders. The [README](./README.md) covers usage
and benchmark results; this covers *how it works and why it's built this way*.

## Goals and constraints

1. **Drop-in equivalence.** Each loader must produce the same samples as the base loader
   it replaces — labels/tokens exact, video within one offline H.264 re-encode. Every
   design decision below is downstream of this: wherever possible the loaders *reuse the
   base code* rather than reimplement it, so equivalence is structural, not coincidental.
2. **Move per-epoch work offline.** The base loaders repeat work every epoch (multi-view
   compose, per-sample resize, subprocess decode). The converters do that work once at
   build time; the hot path is a columnar read + one in-process decode.
3. **Object-store native.** Tables must be readable straight from S3 (selective, parallel
   reads) without FUSE mounts or full downloads.
4. **lancedb-level APIs only.** All reads go through the lancedb `Permutation` API — no
   pylance (`lance`) dependency. Video is stored as plain `large_binary` for now and will
   move to blob encoding (blob-v2) once the lancedb-level blob API is available.

## Shared implementation notes

These apply to all three loaders.

**Permutation reads.** A `Permutation.identity(table).select_columns([...]).with_format("arrow")`
handle is the read path for everything — full-column scans at init (labels, metadata) and
point lookups in the hot path (`__getitems__` on a list of row indices). One behavioral
detail matters: `take` returns rows **sorted by offset**, not in request order. Any code
that reads a batch of rows must therefore key results by row id (`_read_clip_bytes`
returns `{row: bytes}`) rather than zipping positionally against the requested list.
Assuming request order is preserved was an actual bug during development: equivalence
tests with monotonic indices passed while shuffled training crashed.

**Worker safety.** lancedb connections and video decoders are not fork/pickle-safe.
Every loader implements `__getstate__` to null its handles (`_perm`, decoder caches,
row maps); each spawn worker lazily reopens them on first use (`_ensure_open`). This
also keeps the spawn payload small — workers receive config + label arrays, not open
connections.

**In-process video decode.** Clips are stored as short mp4s and decoded with torchcodec
(`VideoDecoder(bytes, seek_mode="approximate")`) — no ffmpeg subprocess per sample. The
converters encode all-intra (`gop=1`, every frame a keyframe), which makes "approximate"
seeking exact and random window reads cheap. A per-worker LRU cache
(`decoder_cache_size`, default 32) keeps recently used clip decoders open, evicting only
decoders not needed by the current batch. `gop` is a build-time knob: larger GOPs shrink
the table at some seek cost.

**Batched `__getitems__`.** The map-style loaders implement `__getitems__` (PyTorch's
batched fetch). The pattern is two-pass: first plan the batch (group requested frame
windows by clip, remembering which output slot owns which slice), then decode each needed
clip **once** and scatter slices to their owners. Samples in a batch that hit the same
clip cost one decode.

**Storage/compression tradeoffs.** All-intra H.264 at the source resolution costs more
bits per pixel than the source's long-GOP encoding, but the composed/pre-resized clips
store fewer pixels, so tables come out smaller in practice (see README "Dataset Size").
The re-encode is the single source of lossiness (~1–2% pixel MAD), verified to be
training-irrelevant by the real-model forward-equivalence runs.

---

## Action — `LanceDROIDComposedDataset`

### What the base does

`DROIDLeRobotDataset` (the base) reads a LeRobot-format tree: per-frame labels from
`data/*.parquet`, episode/task metadata from `meta/`, and three camera views from
concatenated mp4s under `videos/`. Per **sample** it decodes a window from all three
views, resizes the two exteriors to half size, and stacks them under the wrist view into
one `1.5·h × w` frame (`_load_concat_video`). Labels are assembled by `_window_rows` →
`_build_joint_action` / `_build_raw_action` from compact numpy arrays built at init.

### What the converter stores

`tools/lance_datagen/build_composed_droid.py` writes four tables:

- **`{table}`** — one row per episode: `episode_index`, `ep_start`, `length`,
  `video_bytes`. The video is the base's exact composition (`_load_concat_video` output,
  byte-for-byte the same pixels) re-encoded once with `gop=1`.
- **`{table}_frames`** — one row per frame: `episode_index`, `task_index`, `timestamp`,
  plus every feature column either action space reads (joint/gripper actions and states,
  cartesian state), stored as `float32` / `fixed_size_list<float32>`. These are dumped
  **verbatim from the base loader's own arrays** (`_row_*`, `_feat`), so they roundtrip
  bit-exact. Feature names store `.` as `__` (Lance treats dots as nested-field paths).
- **`{table}_tasks`** — `task_index → task` string.
- **`{table}_episodes`** — `episode_index → episode_id` (needed only by the keep-ranges
  window filter).

`--labels-only` rewrites the three label tables against an existing video table (schema
migrations without re-encoding video).

### How the loader works

The loader subclasses `DROIDLeRobotDataset` but **bypasses its parquet-reading
`__init__`** entirely: it takes only `lance_uri` (+ `storage_options` for S3), sets the
same config attributes the base would, and rebuilds the base's compact label arrays from
`{table}_frames` (a single full-column Permutation read at init — ~10 MB for 96k frames).
From that point on, the *inherited* base code runs unchanged:

- flat-index → (episode, offset) mapping via `_valid_cum` / `_ep_starts` / `_ep_vals`
  (same `np.unique` construction as the base);
- `_window_rows` reconstructs per-frame dicts from the arrays on demand;
- `_build_joint_action` / `_build_raw_action` / `_build_result` produce the labels,
  captions, idle-frame counts, and normalization exactly as the base does;
- the keep-ranges filter (`use_filter_dict`) builds the same per-segment index, using
  `{table}_episodes` for the episode ids;
- `get_shuffle_blocks` / `ActionIterableShuffleDataset` give the production
  episode-shuffle stream.

Only the video source is different: `__getitems__` groups the batch's windows by
episode, fetches the missing episodes' `video_bytes` in one batched take, and decodes a
single composed stream per episode instead of three views. Both action spaces
(`joint_pos`, `ee_pose`) are supported; labels are bit-exact against the base for both.

The base's `_rows` (a per-frame list of dicts the DROID subclass never reads — it is
built by the shared `ActionBaseDataset.__init__` for sibling datasets that do use it) is
never constructed here. Note this is a freeable redundancy in the base too, not a
structural Lance advantage; see the README's memory note.

Image augmentation (`use_image_augmentation`) is not supported: the base applies it to
the three raw views *before* composition, and the table stores the composed result.

`get_lance_action_droid_sft_dataset` mirrors `get_action_droid_sft_dataset` (the base
factory), building the same `ActionSFTDataset` + `ActionTransformPipeline` stack around
the Lance dataset — the training-recipe swap is one `_target_` change.

---

## Vision-SFT — `LanceVisionSFTDataset`

### What the base does

`SFTDataset` streams clip windows described by a `video_dataset_file.jsonl`: per sample
it fetches the source clip (S3/local), decodes it at native resolution through an ffmpeg
subprocess with a `scale_hw` resize to the training resolution, selects a frame window,
center-crops, picks a caption (structured JSON preferred), and tokenizes.

### What the converter stores

`tools/lance_datagen/build_vision_sft.py` writes one row per clip-window: the clip
decoded once and resized to the training resolution **with the base's exact resize op**
(same `scale_hw` ratio; the spatial center-crop is left to decode time so the stored clip
stays a clean rectangle), re-encoded `gop=1`, plus everything needed to reproduce the
base's window/caption logic: original `width`/`height`, `start_frame`/`end_frame`/
`temporal_interval`, stored `enc_h`/`enc_w`, `fps`, and the caption fields
(`caption_json` verbatim JSON, `caption` dense fallback).

### How the loader works

Map-style; two Permutation handles per worker — all metadata columns are read once into
`_rows` at init (they're tiny), `video_bytes` is fetched per clip through the batched
take + LRU decoder cache. Per sample it recomputes the base's window plan
(`_window_plan`: same `temporal_interval_mode` / `frame_selection_mode` /
`num_video_frames` arithmetic), decodes the frames in-process, applies the center crop
(`enc_h/enc_w` → target size from `VIDEO_RES_SIZE_INFO`, the base's `get_aspect_ratio`),
and reuses the base's caption selection (`_select_caption`) and tokenization
(`tokenize_caption` + `add_special_tokens`) — which is why `text_token_ids` are
token-exact.

`LanceVisionSFTIterable` + `get_lance_vision_sft_dataset` adapt the map-style dataset to
the training packing stack's iterable/self-sharding contract (per-(rank, worker) shard of
a seeded shuffle, `conditioning_fps` added to match the base sample dict).

---

## VLM — `LanceVLMDataset` / `LanceVLMShuffleScan`

### What the base does

The VLM base streams LLaVA-OneVision from the HuggingFace Hub (`streaming=True`):
sequential shard reads, a bounded shuffle buffer, and a filter for valid
image+conversation records. Image decode and chat tokenization happen downstream in the
processor, not in the loader.

### What the converter stores

`convert_llava_to_lance` (in `vlm_dataset.py`) writes one row per sample: `sample_id`,
`image_bytes` (the raw PNG/JPEG bytes, byte-identical to the source), and `conversations`
(the ShareGPT turns as a JSON string). Because records are byte-identical, everything
downstream (processor, tokenizer) is unchanged by construction.

### How the loaders work

- **`LanceVLMDataset`** (map-style): point lookups by row through one Permutation handle;
  a global shuffle is just a shuffled list of row indices fed by the sampler. Used for
  local training and resumable map-style recipes.
- **`LanceVLMShuffleScan`** (iterable): for S3, random point-lookups are
  latency-bound, so this reads contiguous row-chunks (`batch_size` rows per take) in
  seeded-shuffled chunk order, sharded across workers, and pushes rows through a local
  shuffle buffer — sequential I/O with decorrelated output, the same access pattern the
  HF base gets from shard streaming.

`get_lance_vlm_dataset` mirrors the base `get_llava_ov_map` factory signature so the VLM
recipe swap is also a one-line `_target_` change.

---

## Equivalence methodology

Two layers, both in-repo:

1. **Data-level tests** (`tests/data/lance/`): per-sample comparison against the genuine
   base loaders — action labels/captions bit-exact (both action spaces), vision-SFT
   token-ids exact, VLM records byte-identical; video asserted within re-encode tolerance
   (pixel MAD < 2%). Index lists are deliberately **unsorted** to cover the
   sorted-take ordering contract.
2. **Real-model forward equivalence** (`benchmarks/lance/forward_equivalence.py`): the
   same samples pushed through the real Cosmos3-Nano with fixed weights and seed
   (`lr=0` / per-sample standalone), comparing per-step loss. Base-vs-Lance matched
   within the re-encode tolerance (action ≤1.4%, vision ≤2.1%, VLM exact), while
   different samples differ by ~10× more — i.e. the metric is sensitive and the residual
   is the re-encode, not variance (verified by a base-vs-base control at 0.000%).

## Benchmark methodology

Fairness rules (see `benchmarks/lance/`): the base side is always the **genuine shipped
loader** (never a reconstruction); S3 regimes give the base a materialization standin
(download-then-run) since it has no native S3 path; the access pattern matches production
(episode-shuffle for action on both sides, not a random sampler); VLM throughput is
measured end-to-end (image decode + tokenize) because the loaders themselves emit raw
records. Memory is measured per-worker under spawn (PSS for fork/COW fairness), one side
per process.
