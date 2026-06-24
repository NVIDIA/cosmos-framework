# How the LanceDB loaders achieve their speedups

A per-loader mechanism guide. For each of the three Cosmos dataloaders this explains **what the base
loader does that is slow, what the Lance port does instead, why the base structurally can't do the
same, and which measured win each mechanism produces.** Numbers are from [`BENCHMARKS.md`](BENCHMARKS.md).

There are two *kinds* of win, and they are different:

1. **Representation wins** (action, vision-SFT): do the per-epoch video transform **once, offline**, and
   store a training-optimized clip. The hot path then decodes far less. This is a property of the *stored
   data*, not of Lance per se — but it's only practical because Lance gives you an indexed, versioned,
   shuffle-sampled, object-store-native multimodal store to put those clips in.
2. **Access-layer wins** (all three, and the whole point on S3): columnar random access + true global
   shuffle + object-store-native reads, vs. the base's sequential-tar / per-sample-file / streaming models.

Both ride on a small set of shared techniques (Permutation API, plain-binary blobs, per-worker lazy
handles, batched `__getitems__`) covered at the end.

---

## 1. Action / LeRobot — `LanceDROIDComposedDataset` (`action_dataset.py`)

### What the base does (the bottleneck)
`DROIDLeRobotDataset.__getitem__` (`data/vfm/action/datasets/droid_lerobot_dataset.py`) for **every
sample, every epoch**:
1. seeks **three** camera-view mp4s (wrist + 2 exteriors),
2. decodes a window from each (torchcodec),
3. `F.interpolate`s the two exteriors to half-resolution,
4. concatenates into one `(3, T, 270, 320)` tensor (wrist on top, exteriors bottom).

~98% of per-sample time is this 3-stream decode + resize + concat. It is redone identically every epoch
because the canonical LeRobot v3 dataset only stores the raw per-view mp4s.

### What Lance does
The converter `tools/lance_datagen/build_composed_droid.py` runs the base's **exact** resize+concat op
**once, offline**, and stores **one composed `270×320` clip per episode**, re-encoded **all-intra
(`gop=1`)** as a single blob row. At train time `LanceDROIDComposedDataset`:
- decodes **one small stream** instead of three full views — no interpolate, no concat (it's baked in),
- uses `seek_mode="approximate"` — with `gop=1` every frame is a keyframe, so approximate seek is exact **and** skips the full-file index scan (cheap decoder init for the shuffled, many-clip access pattern),
- keeps a **per-worker LRU `VideoDecoder` cache** keyed by episode, so consecutive windows of the same episode reuse the decoder,
- batches the whole DataLoader batch in `__getitems__`: group the needed frames per clip → **one
  `get_frames_at` per clip** instead of one decode call per sample,
- pairs with `LanceDROIDComposedIterable` (episode-shuffle): windows of an episode stream contiguously, so
  the clip is fetched/decoded **once** and reused across all its windows (vs `RandomSampler` re-fetching).

### Why the base can't do this
It is bound to the canonical LeRobot v3 format (3 raw views) and recomputes the transform every epoch.
Pre-composing requires an indexed, versioned, per-episode multimodal store to serve the optimized clips
from — i.e. you'd be rebuilding Lance.

### Equivalence & win
Action/captions/poses are **bit-exact** (all index/pose/action logic is inherited unchanged); video
differs only by the H.264 re-encode (PSNR ~32 dB, mean|Δ|≈1.6%). A separate `LanceDROIDDataset` stores the
original mp4 bytes for **byte-exact** parity (used by the equivalence test). Measured single-modality:
**1.82× LOCAL / 2.68× S3** (18 workers). Disk is **0.35× the original** (fusing 3 views → 1 half-res clip
more than offsets the all-intra penalty) — not a blowup, and nowhere near per-frame-JPEG (1.8×, rejected).

---

## 2. WebDataset / VLM — `LanceVLMDataset` + `LanceVLMShuffleScan` (`vlm_dataset.py`)

### What the base does (the bottleneck)
The stock VLM path streams `lmms-lab/LLaVA-OneVision-Data` either as an HF `IterableDataset`
(`streaming=True`, the cosmos default) or as WebDataset tar shards: **sequential shard reads**, a
**bounded shuffle buffer** (approximate shuffle, not global), and **re-streamed/re-decoded every epoch**.
There is no random access — you cannot fetch sample *i* without walking the shard.

### What Lance does
`convert_llava_to_lance` stores each record `{sample_id, image_bytes (PLAIN large_binary), conversations}`
columnar — original encoded image bytes, **no re-encode, no disk blowup**. Two access modes:
- **`LanceVLMDataset`** — map-style **O(1) random access** via the Permutation API → **true global
  shuffle** (shuffle row indices, `take` them), not a buffer. Best on local/NVMe.
- **`LanceVLMShuffleScan`** — the right pattern for **object storage**: shuffle *fragment order* + a
  row buffer over a sequential `to_batches(..., batch_readahead=8)` columnar scan → **bandwidth-bound**
  reads (fast on S3) with shuffle quality on par with a WebDataset buffer, but columnar (much faster than
  tar streaming) and with true random access still available.

### Why the base can't do this
A tar is sequential-only; its shuffle is a local buffer. Lance gives random access, global shuffle, and
columnar/selective reads (fetch only the rows/columns a curriculum needs) the tar/stream model can't.

### Win & the honest caveat
Single-modality **4.94× LOCAL / 3.79× S3** raw access (and up to ~22× at very large batch). **But the
VLM end-to-end step is gated by the Qwen image-processor** (patchify/normalize + tokenize), which is
storage-independent — so single-node **e2e is ~1×**. The access win surfaces e2e only at scale (object
storage, many nodes, true global shuffle) or when that compute is precomputed. VLM is also never the
combined-mixer bottleneck (it's 10–400× faster than the video loaders). Report the regime; don't quote the
raw ratio as an e2e win.

---

## 3. Local vision-SFT — `LanceVisionSFTDataset` (`vision_sft_dataset.py`)

### What the base does (the bottleneck)
`SFTDataset` / `LocalSFTDataset` per **every sample, every epoch**: seek the source mp4, spawn an
**ffmpeg subprocess** to decode a window **with a `scale` filter** (resize to training resolution), then
tokenize the caption. Process spawn + full-resolution decode + on-the-fly resize, per sample.

### What Lance does
`tools/lance_datagen/build_vision_sft.py` decodes each clip once, **resizes to training resolution
offline** (the base's exact resize), re-encodes **all-intra (`gop=1`)**, and stores
`{clip_id, sizing, caption_json, caption, video_bytes}`. At train time the loader:
- decodes a clip **already at training resolution** → far fewer pixels, **no on-the-fly resize**,
- **approximate seek is exact** (gop=1) and cheap,
- uses an **in-process torchcodec** decoder + per-worker LRU cache → **no ffmpeg process spawn**,
- one batched `get_frames_at` per clip, with the same window math + center-crop + temporal truncation +
  tokenize as the base.

### Why the win holds end-to-end (unlike VLM)
The only non-video work is one chat-template tokenize (cheap), so the video savings aren't masked.
Token-ids are **exact**; video within H.264 tolerance (mean|Δ|≈1.3%). Measured single-modality
**8.23× LOCAL / 7.28× S3** (18 workers) — and it holds e2e (~6.5×). This is the largest per-loader win.

---

## 4. Cross-cutting mechanisms (apply to more than one loader)

### 4a. Plain `large_binary` + columnar `take` — the S3 read win (6.3×)
`take_blobs` (lance blob-v2) returns lazy `BlobFile` handles; reading them in a Python loop issues GETs
**one at a time** → serialized, latency-bound on S3 (~31 clips/s, 55 MB/s — *unchanged* by
`LANCE_IO_THREADS`, `io_buffer_size`, or sorted indices, because the reads are sequential in Python). For
clips <2 MB, storing the bytes as a **plain `large_binary`** column and reading via
`ds.take(indices, columns=["video_bytes"])` lets Lance parallelize the GETs across the **IO thread pool**
→ **197 clips/s, 345 MB/s (6.3×)**. Blob-v2 only pays off for multi-GB payloads. The loaders **auto-detect**
the encoding (`_is_blob` from the column's `lance-encoding:blob` metadata) and pick `take` vs `take_blobs`;
converters default to `--storage plain`. Byte-identical either way, so equivalence is preserved. Effect on
the read-bound loaders, S3: vision-SFT **178 → 376 (2.1×)**, action random **110 → 167 (1.5×)**.
(`data_storage_version` stays at the stable **2.1** — 2.2 is unstable in Lance 7.0.0.)

### 4b. Per-loader worker rebalancing
Each sub-loader is its own `DataLoader` with its own `num_workers` (cosmos defaults to a flat ~4 and does
**not** auto-balance; its "multiplex" is ratio-based modality mixing, not worker allocation). The combined
mixer is gated by the slowest loader, so moving workers off the idle VLM onto action+vsft roughly **4×'s**
the combined throughput. The ceiling is ~3× a heavy loader's per-loader peak (~18 workers on 48 cores);
oversubscribing cores past that *degrades* it. Lance scales better than base (lighter per-sample decode),
so its lead widens with worker count. Exposed via `--action-workers/--vlm-workers/--vsft-workers`.

### 4c. The Permutation-API worker-safe pattern (all three loaders)
Following `lerobot-lancedb` / the `training/object-detection` reference: the Dataset stores only
connection params; `__getstate__` nulls all live handles so it pickles cleanly to **spawn** workers
(Lance is **not fork-safe** — always `multiprocessing_context="spawn"`); each worker lazily reopens its own
`lance.dataset` / `Permutation` + decoder cache in `_ensure_open`. `__getitems__` is the hot path — the
DataLoader hands the whole batch's indices at once, so reads/decodes are batched (one `take`/`take_blobs`
+ one `get_frames_at` per file), not per-sample.

### 4d. Decode device (fairness note)
All base-vs-lance comparisons use **CPU decode on both sides** (the base can only decode on CPU). NVDEC is
*not* the win at these small robot frames (it's slower than many-core CPU per torchcodec's own perf docs);
the win is the optimized stored representation + access layer, which is why it's a fair comparison.

---

## Summary

| loader | base bottleneck | Lance mechanism | win kind | measured (single-modality) |
| ------ | --------------- | --------------- | -------- | --------------------------- |
| action / DROID | 3-view decode + resize + concat per sample/epoch | pre-composed 1-clip, all-intra, per-episode blob, decoder-cache reuse | representation + access | 1.82× LOCAL / 2.68× S3 |
| VLM / LLaVA | sequential tar / HF-stream + shuffle buffer, no random access | columnar random access + global shuffle / chunked-shuffle scan | access | 4.94× LOCAL / 3.79× S3 raw (≈1× e2e, compute-bound) |
| vision-SFT / Bridge | per-sample ffmpeg seek+decode+scale subprocess | pre-resized all-intra clip, in-process torchcodec, batched decode | representation + access | 8.23× LOCAL / 7.28× S3 (holds e2e) |
| **all, on S3** | serialized `take_blobs` GETs | **plain `large_binary` + columnar `take`** | access | 6.3× raw blob read; 2.1× vsft e2e |
| **combined** | flat per-loader workers, gated by slowest | **worker rebalancing** toward the bottleneck loaders | scheduling | ~4× the equal-worker combined |

See [`BENCHMARKS.md`](BENCHMARKS.md) for full tables, `VALIDATION.md` for the representation-preserves-data proofs, and the
equivalence tests in `tests/data/lance/`.
