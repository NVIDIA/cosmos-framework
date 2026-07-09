# Design: LanceDB-backed Cosmos Dataloaders

How the three Lance loaders work and what keeps their output equivalent to the base
loaders. Usage and benchmark results are in the [README](./README.md).

## Constraints

1. **Drop-in equivalence** — same samples as the base loader: labels/tokens exact, video
   within one offline H.264 re-encode. Wherever possible the loaders reuse the base code
   rather than reimplement it, so equivalence is structural.
2. **Per-epoch work moved offline** — the converters do the repeated work (multi-view
   compose, per-sample resize, subprocess decode) once at build time; the hot path is a
   columnar read + one in-process decode.
3. **Object-store native** — tables readable straight from S3 (selective, parallel reads);
   no FUSE, no full downloads.
4. **lancedb-level APIs only** — all reads via the `Permutation` API, no pylance. Video is
   plain `large_binary` until the lancedb-level blob API (blob-v2) lands.

## Shared mechanics

- **Permutation reads.** One `Permutation.identity(tbl).select_columns([...]).with_format("arrow")`
  handle per table: full-column scans at init, point lookups in the hot path. `take`
  returns rows **sorted by offset and deduplicated**, so batched reads key results by row
  (`_read_clip_bytes` → `{row: bytes}`), never positionally. The equivalence tests use
  unsorted/duplicate index lists to pin this contract.
- **Worker safety.** lancedb connections and decoders are not fork/pickle-safe. Every
  loader nulls its handles in `__getstate__`; spawn workers reopen lazily
  (`_ensure_open`). This also keeps the spawn payload small.
- **In-process decode.** Clips are short mp4s decoded with torchcodec
  (`VideoDecoder(bytes, seek_mode="approximate")`). Converters encode all-intra
  (`gop=1`), which makes approximate seeking exact and random window reads cheap. A
  per-worker LRU (`decoder_cache_size`, default 32) keeps recent clip decoders open;
  eviction skips decoders the current batch still needs. `gop` is a build-time size/seek
  knob.
- **Batched `__getitems__`.** Two-pass: plan the batch (group requested windows by clip,
  record which output slot owns which slice), then decode each clip once and scatter.
- **Lossiness.** The offline re-encode is the only lossy step (~1–2% pixel MAD; the
  action gate is 2.5% because the base's decoder backend also differs). Verified
  training-irrelevant by the real-model forward-equivalence runs.

---

## Action — `LanceDROIDComposedDataset`

Base behavior: `DROIDLeRobotDataset` registers LeRobot sources metadata-only, splits
episodes deterministically (`split_episode_ids`), builds a span index
(`build_episode_spans`), and keeps per-shard `LeRobotDataset` readers lazy behind an LRU.
Per sample, `_fetch_sample` returns windowed label features + three camera-view windows
(`delta_timestamps`); `__getitem__` composes the views into one `1.5·h × w` frame
(`_compose_multi_view`) and assembles the action for the chosen action space. Feature
names/flags resolve from a version registry keyed by the root's directory name.

### Tables (`tools/lance_datagen/build_composed_droid.py`)

| table | one row per | columns |
| --- | --- | --- |
| `{table}` | episode | `episode_index` int64, `ep_start` int64, `length` int64, `video_bytes` large_binary — the base's exact composition over the full episode, re-encoded `gop=1` |
| `{table}_frames` | frame | `episode_index` int64, `task_index` int64, `timestamp` float64, + every feature column any action space reads, as `float32` / `fixed_size_list<float32>` (`.` stored as `__`) — dumped verbatim from the base's LeRobot table, bit-exact roundtrip |
| `{table}_tasks` | task | `task_index` int64, `task` string |
| `{table}_episodes` | episode | `episode_index` int64, `episode_id` string (keep-ranges filter only) |

`--labels-only` rewrites the label tables against an existing video table. The converter
raises on multi-shard roots (one frames table = one shard).

### Loader

Subclasses `DROIDLeRobotDataset`, **bypassing its LeRobot-reading `__init__`**:

- Takes `lance_uri` (+ `storage_options`) and a `version` (same registry); sets the same
  config attributes the base would; loads `{table}_frames` label columns in one
  full-column read (~10 MB / 96k frames).
- Split/span index built with the base's **own helpers** (`split_episode_ids` +
  `build_episode_spans`) → `split`, `split_seed`, `split_val_ratio`, `sample_stride`,
  keep-ranges filter behave identically. An empty filter match raises.
- `_fetch_sample` override returns the same windowed sample dict the LeRobot readers
  would (contiguous row slices per the base's `delta_timestamps` plan + task string) —
  the inherited `__getitem__` then does actions, captions, gripper flips, idle frames,
  and normalization unchanged.
- `_compose_multi_view` override decodes the requested window from the stored composed
  clip (uint8 → the `[0,1]` float layout the base expects).
- `get_shuffle_blocks` + the base `ActionIterableShuffleDataset` give the production
  episode-shuffle stream; `__getitems__` pre-warms the decoder cache with one batched
  byte read.

Labels are bit-exact for `joint_pos` and `midtrain`, with and without `use_state`; video
< 2.5% pixel MAD. Not supported: image augmentation (applies to raw views before
composition), `max_num_history_actions` (needs pre-window history rows), `val_temp_seg`,
multi-shard roots. `get_lance_action_droid_sft_dataset` mirrors the base factory
(`ActionSFTDataset` + `ActionTransformPipeline`) — the recipe swap is one `_target_`
change.

---

## Vision-SFT — `LanceVisionSFTDataset`

Base behavior: `SFTDataset` fetches each source clip (S3/local), decodes at native
resolution through an ffmpeg subprocess with a `scale_hw` resize, selects a frame window,
center-crops, picks a caption, post-processes it, and tokenizes.

### Table (`tools/lance_datagen/build_vision_sft.py`)

| column | type | description |
| --- | --- | --- |
| `clip_id` | string | `{uuid}_w{window}` |
| `width`, `height` | int64 | original resolution |
| `start_frame`, `end_frame`, `temporal_interval` | int64 | window bounds / stride |
| `enc_h`, `enc_w` | int64 | stored (resized) resolution |
| `fps` | float64 | source fps |
| `caption_json` | string | structured caption (verbatim JSON) or `""` |
| `caption` | string | dense caption fallback |
| `video_bytes` | large_binary | clip resized to the training resolution, `gop=1` |

One row per clip-window; the resize is the base's exact op (same `scale_hw` ratio; the
center-crop is left to decode time). The metadata pass is the base's own loader
(`_load_sft_metadata_from_s3` + `_flatten_metadata_by_window`), so the duration and
`min_frames` filters match the base population exactly. Windows carrying caption keys
the schema does not persist (`CAPTION_TYPES` styles, `qwen3_32b_rewrite-dense`) are
rejected at build time.

### Loader

- Metadata columns read once into `_rows` at init; `video_bytes` fetched per clip via
  batched take + the LRU decoder cache.
- `_window_plan` recomputes the base's window arithmetic (`temporal_interval_mode` /
  `frame_selection_mode` / `num_video_frames`); frame indices are clamped to the stored
  clip the way the base's sequential decode clamps.
- Center crop from `enc_h/enc_w` to the `VIDEO_RES_SIZE_INFO` target; a table built at a
  smaller `--resolution` than requested raises.
- Caption pipeline is the base's: `_select_caption`, then the same post-processing order
  (`caption_suffix`, CFG dropout, duration/FPS + resolution suffixes for non-structured
  captions), then `tokenize_caption` — `text_token_ids` are token-exact for structured
  and dense captions.
- Samples the base would skip (short window, no usable caption) return `None` — the
  `process_one_sample` contract; the iterable filters them.
- `LanceVisionSFTIterable` + `get_lance_vision_sft_dataset` provide the packing stack's
  iterable contract: per-(rank, worker) shard of a seeded shuffle with a
  `torch.distributed` fallback, `conditioning_fps` added to match the base sample dict.

---

## VLM — `LanceVLMDataset` / `LanceVLMShuffleScan`

Base behavior: streams LLaVA-OneVision from the HuggingFace Hub (`streaming=True`) —
sequential shard reads, bounded shuffle buffer, image+conversation filter. Image decode
and tokenization happen downstream in the processor.

### Table (`convert_llava_to_lance` in `vlm_dataset.py`)

| column | type | description |
| --- | --- | --- |
| `sample_id` | string | sample id |
| `image_bytes` | large_binary | raw PNG/JPEG bytes, byte-identical to the source |
| `conversations` | string | ShareGPT turns as JSON |

Records are byte-identical, so everything downstream is unchanged by construction.

### Loaders

- **`LanceVLMDataset`** (map-style): row point-lookups; a global shuffle is just a
  shuffled index list from the sampler. `__getitems__` keys the sorted-take result by
  row and maps back to the requested order (duplicate-safe).
- **`LanceVLMShuffleScan`** (iterable): for S3 — contiguous row-chunks in shuffled chunk
  order (reshuffled each pass), sharded per (rank, worker) with a `torch.distributed`
  fallback, through a local shuffle buffer. Sequential I/O with decorrelated output —
  the same access pattern the HF base gets from shard streaming.
- `get_lance_vlm_dataset` mirrors `get_llava_ov_map`; `n` caps the dataset like the
  base's `.select(range(n))`.

---

## Equivalence methodology

1. **Data-level tests** (`tests/data/lance/`): per-sample comparison against the genuine
   base loaders — action labels/captions bit-exact (both action spaces, ± `use_state`),
   vision-SFT token-ids exact (structured + dense captions), VLM records byte-identical;
   video within the re-encode tolerance. Index lists are unsorted (with duplicates for
   VLM) to pin the sorted-take contract.
2. **Real-model forward equivalence** (`benchmarks/lance/forward_equivalence.py`): the
   same samples through the real Cosmos3-Nano with fixed weights and seed (`lr=0`),
   comparing per-step loss. Base-vs-Lance within the re-encode tolerance (action ≤1.4%,
   vision ≤2.1%, VLM exact 0.00%), while different samples differ ~10× more; a
   base-vs-base control gives 0.000%. (The action run predates the upstream loader
   rewrite; label bit-exactness was re-verified against the rewritten base.)

## Benchmark methodology

- The base side is always the **genuine shipped loader**, never a reconstruction.
- S3 regimes give the base a materialization standin (download-then-run) — it has no
  native S3 path.
- Access patterns match production: episode-shuffle for action on both sides (the
  shipped `ActionIterableShuffleDataset`), not a random sampler.
- VLM throughput is end-to-end (image decode + tokenize) since the loaders emit raw
  records.
- Memory is per-worker under spawn (PSS for fork/COW fairness), one side per process.
