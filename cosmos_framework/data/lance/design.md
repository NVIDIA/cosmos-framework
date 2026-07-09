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
The re-encode is the single source of lossiness (~1–2% pixel MAD; the action gate is
2.5% because the base's decoder backend also differs), verified to be
training-irrelevant by the real-model forward-equivalence runs.

---

## Action — `LanceDROIDComposedDataset`

### What the base does

`DROIDLeRobotDataset` (built on `BaseActionLeRobotDataset`) registers LeRobot sources
metadata-only at init, derives a deterministic train/val episode split
(`split_episode_ids`) and per-episode span index (`build_episode_spans`), and keeps the
heavy per-shard `LeRobotDataset` readers lazy behind an LRU. Per **sample**,
`_fetch_sample` maps the flat index to (dataset, row, episode, offset) and asks the
LeRobot reader for the windowed label features *and* the three camera-view windows
(`delta_timestamps`); `__getitem__` then composes the views into one `1.5·h × w` frame
(`_compose_multi_view`) and assembles the action for the chosen action space (midtrain
pose deltas / joint_pos / ee_pose_delta, including per-version gripper flipping).
Per-dataset feature names and flags resolve from a version registry keyed by the root's
directory name (`droid_lerobot_dataset_config`).

### What the converter stores

`tools/lance_datagen/build_composed_droid.py` writes four tables:

- **`{table}`** — one row per episode: `episode_index`, `ep_start`, `length`,
  `video_bytes`. The video is the base's exact composition (`_compose_multi_view` over the
  full episode's views) re-encoded once with `gop=1`.
- **`{table}_frames`** — one row per frame: `episode_index`, `task_index`, `timestamp`,
  plus every feature column any action space reads (joint/gripper actions and states,
  cartesian state), stored as `float32` / `fixed_size_list<float32>`. These are dumped
  **verbatim from the base's LeRobot table**, so they roundtrip bit-exact. Feature names
  store `.` as `__` (Lance treats dots as nested-field paths).
- **`{table}_tasks`** — `task_index → task` string.
- **`{table}_episodes`** — `episode_index → episode_id` (needed only by the keep-ranges
  window filter).

`--labels-only` rewrites the three label tables against an existing video table (schema
migrations without re-encoding video).

### How the loader works

The loader subclasses `DROIDLeRobotDataset` but **bypasses its LeRobot-reading
`__init__`**: it takes only `lance_uri` (+ `storage_options` for S3) and a `version`
(same registry the base resolves from its root name), sets the same config attributes the
base would, and loads the per-frame label columns from `{table}_frames` in a single
full-column Permutation read (~10 MB for 96k frames). The split/span index is then built
with the base's **own helpers** (`split_episode_ids` + `build_episode_spans`), so
`split`, `split_seed`, `split_val_ratio`, `sample_stride`, and the keep-ranges filter
behave identically. From there the *inherited* base code runs unchanged:

- `_resolve_index` maps a flat index over the same `_episode_records` /
  `_episode_cum_ends` structures;
- our `_fetch_sample` override returns the same windowed sample dict the LeRobot readers
  would (contiguous row slices of each feature, per the base's `delta_timestamps` plan,
  plus the task string) — so the inherited `__getitem__` assembles actions, captions,
  gripper flips, idle frames, and normalization exactly as the base does;
- our `_compose_multi_view` override decodes the requested window straight from the
  stored composed clip (uint8 → the `[0,1]` float layout the base expects), instead of
  decoding and composing three views;
- `get_shuffle_blocks` / `ActionIterableShuffleDataset` give the production
  episode-shuffle stream; `__getitems__` pre-warms the decoder cache for a whole batch
  with one batched byte read.

All action spaces route through the inherited assembly; labels are bit-exact against the
base for the same split parameters (verified for `joint_pos` and `midtrain`, with and
without `use_state`). Video is
within one offline H.264 re-encode plus the base's decoder-backend difference (< 2.5%
pixel MAD). Not supported: image augmentation (applied to raw views before composition),
`max_num_history_actions` (needs pre-window history rows), the `val_temp_seg` split, and
multi-shard roots (the converter dumps one LeRobot shard into one frames table and
raises otherwise; the `*_sharded` registry versions would need per-shard tables).

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
(`caption_json` verbatim JSON, `caption` dense fallback). The metadata pass is the
base's **own loader** (`_load_sft_metadata_from_s3` + `_flatten_metadata_by_window`),
so the duration and `min_frames` window filters match the base pipeline's population
exactly; windows carrying caption keys the schema does not persist (the
`CAPTION_TYPES` styles / `qwen3_32b_rewrite-dense`) are rejected at build time rather
than silently losing captions.

### How the loader works

Map-style; two Permutation handles per worker — all metadata columns are read once into
`_rows` at init (they're tiny), `video_bytes` is fetched per clip through the batched
take + LRU decoder cache. Per sample it recomputes the base's window plan
(`_window_plan`: same `temporal_interval_mode` / `frame_selection_mode` /
`num_video_frames` arithmetic, frame indices clamped to the stored clip the way the
base's sequential decode naturally clamps), decodes the frames in-process, applies the
center crop (`enc_h/enc_w` → target size from `VIDEO_RES_SIZE_INFO`, the base's
`get_aspect_ratio`; a table built at a smaller `--resolution` than requested raises),
and reuses the base's caption pipeline: selection (`_select_caption`), the same
post-processing (`caption_suffix`, CFG dropout, duration/FPS and resolution
conditioning suffixes for non-structured captions, in the base's order), and
tokenization (`tokenize_caption` + `add_special_tokens`) — which is why
`text_token_ids` are token-exact for structured *and* dense captions. Samples the base
would skip (short window, no usable caption) come back as `None` — the same contract
as `process_one_sample` — and the iterable filters them.

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
  seeded-shuffled chunk order (reshuffled each pass), sharded per (rank, worker) —
  with a `torch.distributed` fallback when the dataloader doesn't set the shard
  attributes — and pushes rows through a local shuffle buffer: sequential I/O with
  decorrelated output, the same access pattern the HF base gets from shard streaming.

`get_lance_vlm_dataset` mirrors the base `get_llava_ov_map` factory signature so the VLM
recipe swap is also a one-line `_target_` change; `n` caps the dataset like the base's
`.select(range(n))`.

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
   (The action run predates the upstream loader rewrite; data-level bit-exactness has
   been re-verified against the rewritten base.)

## Benchmark methodology

Fairness rules (see `benchmarks/lance/`): the base side is always the **genuine shipped
loader** (never a reconstruction); S3 regimes give the base a materialization standin
(download-then-run) since it has no native S3 path; the access pattern matches production
(episode-shuffle for action on both sides, not a random sampler); VLM throughput is
measured end-to-end (image decode + tokenize) because the loaders themselves emit raw
records. Memory is measured per-worker under spawn (PSS for fork/COW fairness), one side
per process.
