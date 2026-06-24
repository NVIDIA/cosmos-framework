# LanceDB Cosmos dataloaders — results

Hardware: single node, 48 CPU + NVIDIA L40S, driver 580. **CPU decode on both sides** (the
base can only decode on CPU). All comparisons use the base's production config; for action
that is episode-shuffle on both sides. Per-loader datasets noted in each section.

## Combined 3-loader throughput (the headline)

327 DROID episodes, 1:1:1 round-robin mixer, 6 workers/loader, batch 16. The mixer aggregate is
**gated by the slowest loader** (aggregate ≈ 3×slowest), so the combined "speedup" tracks the
bottleneck loader, not a multiplicative win. Reproduce: `bench_combined_faithful.py` (run
`--trios base` and `--trios lance` in SEPARATE processes — the torchcodec/lance teardown raises a
benign SIGABRT between trios).

**LOCAL (apples-to-apples — cosmos's documented workflow is download-to-local-then-train):**

| loader (RAW) | base | lance | speedup |
| ------------ | ---- | ----- | ------- |
| action / DROID | 62.2 | 119.9 | 1.93× |
| VLM (raw access) | 21,918 | 35,728 | 1.63× |
| vision-SFT | 41.9 | 317.3 | 7.57× |
| **combined (1:1:1)** | **122.1** | **379.5** | **3.11×** |

**S3 (Lance native `s3://`, `LANCE_IO_THREADS=256`; base = stock access per loader):**

| loader (RAW) | base | lance | speedup | base S3 access |
| ------------ | ---- | ----- | ------- | -------------- |
| action / DROID | 73.8 | 126.4 | 1.71× | s3fs FUSE (no native reader) |
| VLM | 18,838 | 32,097 | 1.70× | s3fs FUSE (no native reader) |
| vision-SFT | 31.4 | 83.4 | 2.66× | boto3 download-per-sample (stock `SFTDataset`) |
| **combined (1:1:1)** | **95.5** | **251.9** | **2.64×** |

**DEFAULT-MIXED (each loader on its actual default storage — the most realistic single run):**
base → action LOCAL, vision-SFT S3 (boto3), VLM HF-Hub streaming; Lance → action LOCAL, vision-SFT S3, VLM S3.

| loader (RAW) | base | lance | speedup | notes |
| ------------ | ---- | ----- | ------- | ----- |
| action / DROID | 81.6 | 138.6 | 1.70× | both local |
| VLM | 901 | 39,428 | 43.7× | base = HF-Hub stream (PIL decode); lance = S3 byte-scan; **not the bottleneck** |
| vision-SFT | 39.1 | 98.2 | 2.51× | base boto3 S3 / lance S3 |
| **combined (1:1:1)** | **95.3** | **253.5** | **2.66×** | gated by the video loaders |

All three regimes agree: **combined ≈ 2.6–3.1×**, gated by the slowest (video) loader. The VLM's huge
raw ratio never surfaces in the combined because it's already 10–400× faster than the video loaders.

**Methodology lesson (do not repeat).** An earlier draft reported **8.49× from S3**. That was an
artifact of benchmarking the vision-SFT base through an **s3fs FUSE mount** (ffmpeg seeky reads →
11.2 samples/s). The *stock* cosmos vision-SFT loader (`SFTDataset`) downloads each video via
**boto3** (`download_from_s3`), which runs at **31.4** — ~2.8× faster than FUSE. Using the correct
stock base collapses the combined to the honest **2.64×**. Always benchmark against the loader the
base *actually ships*, and label exactly how each side accessed storage.

# LanceDB action dataloader — detail (DROID)

Data: subsets of public `lerobot/droid_1.0.1` (3 camera views, 320×180), renamed to the
Cosmos-canonical schema so the base and LanceDB loaders read identical inputs.

## Equivalence (bit-exact)
`tests/data/lance/test_action_equivalence.py` — 8/8 pass. With `decode_device="cpu"`
the LanceDB loader is byte-identical to `DROIDLeRobotDataset`:
`video max|Δ|=0`, `action max|Δ|=0`, identical captions / idle_frames / poses,
for both `joint_pos` and `ee_pose` action spaces.

## Throughput — video decode (the bottleneck), `bench_decode.py`
64 windows × 5 repeats, 3 views × 17 frames each:

| backend                      | frames/s | speedup |
| ---------------------------- | -------- | ------- |
| base (CPU torchcodec, mp4)   | 1244     | 1.00×   |
| lance-cpu (blob-v2, batched) | 1449     | 1.16×   |
| lance-gpu (blob-v2 + NVDEC)  | 5163     | 4.15×   |

LanceDB blob-v2 + NVDEC decodes the multi-view video **4.15× faster**. This is a
floor on the win: droid_1.0.1 is 320×180 and the subset is 3 fully-OS-cached
files (best case for the mp4 base path). Cosmos trains at 640×360 over thousands
of files, where decode dominates and the base path also pays file-open/seek and
page-cache misses.

## End-to-end DataLoader, `bench_action_faithful.py`
On this subset the full per-sample pipeline (index map, pose/action math) is a
large share of per-sample cost at 320×180, so end-to-end speedup is smaller than
the decode-isolated number. The GPU decode path is intentionally NOT used in any
base-vs-lance comparison (the base can only decode on CPU; comparing CPU-vs-GPU
would be invalid).

# LanceDB VLM dataloader — results (LLaVA-OneVision)

Data: `figureqa(cauldron,llava_format)` subset of `lmms-lab/LLaVA-OneVision-Data`
(99,995 image+conversation samples, ~2.1GB). Lance table stores original PNG bytes
inline (no re-encode, no disk blowup) + conversations; served via the Permutation API.

Base = HF `IterableDataset` (`streaming`-style: sequential shards + bounded shuffle
buffer, no random access). Lance = `LanceVLMDataset` map-style (Permutation random
access + true global shuffle). Both feed the SAME tokenize+image-process step.

| measurement                         | base IterableDataset | lance | speedup |
| ----------------------------------- | -------------------- | ----- | ------- |
| raw access (samples/s, no process)  | 966                  | 21635 | 22.4×   |
| end-to-end (w/ Qwen image+tokenize) | 300                  | 324   | 1.08×   |

The access layer — exactly the webdataset/IterableDataset bottleneck — is ~22× faster
**at a large batch (16384)**. This is batch-regime-dependent: at a training batch of 16 with
6 workers (the combined-table config) the raw-access advantage is **~1.6–1.7×** (local/S3), and
single-node **end-to-end is ~1×** because it's gated by per-sample processing compute (image
patchify/normalize + tokenize), which is storage-independent. The access win surfaces e2e only
when that compute is precomputed (disk cost) or the pipeline is access/IO-bound (object storage,
many nodes, global shuffle — i.e. at scale). Report the regime; don't quote 22× as an e2e win.

# S3 / object-storage findings (the scalability regime)

Same bucket (us-east-2, same region as the GPU box). LLaVA figureqa: lance table,
webdataset tar shards, and base parquet all on S3.

Raw-access samples/s reading from S3:

| access pattern                          | 4 workers | 8 workers | notes |
| --------------------------------------- | --------- | --------- | ----- |
| webdataset tar (sequential stream)      | ~9,500    | ~28,000   | bandwidth-bound |
| lance chunked-shuffle scan              | ~35,000   | ~29,000   | bandwidth-bound, beats wds at low parallelism |
| lance batched-random (Permutation)      | ~7,800    | ~12,400   | latency-bound (~80 MB/s single-call ceiling) |

Key facts:
* **Random reads on S3 are bandwidth-inefficient** — scattered ~22KB GETs can't
  coalesce, so `take`/`__getitems__` plateaus ~80 MB/s single-call (3.7k samples/s
  at batch 16384) and ~270 MB/s across 8 workers, vs ~620 MB/s sequential. This is
  object-storage physics, not a Lance bug (verified across batch 256→16384).
* At **saturation, both webdataset and lance-scan are network-bandwidth-bound and
  comparable** (~620 MB/s). Lance-scan wins at lower parallelism (3.7× at 4 workers).
* Lance's durable advantages are **capabilities**, not raw full-epoch throughput:
  true random access + global shuffle (webdataset can't do either — only a local
  shuffle buffer over sequential reads), columnar/selective + filtered reads (fetch
  only the rows/columns a curriculum needs vs streaming whole shards), and bit-exact
  drop-in parity for the action loader. The raw-throughput win is real only at
  low/moderate worker counts.

# Action loader BEATS base via pre-composed representation (the decode-bound win)

GPU/NVDEC is NOT the win at these small (270×320) frames; instead store a training-optimized
representation the base loader can't. We pre-compose each episode's 3 views (base's exact
resize+concat) into ONE 270×320 clip, re-encoded all-intra (gop=1), one per-episode blob (162M
for 100 eps vs 1.5GB raw blobs).

`LanceDROIDComposedDataset` decodes that single small clip (approximate seek, per-worker
LRU decoder cache) instead of 3 full views + F.interpolate + concat. Fair CPU-vs-CPU, shuffled,
local. **The speedup is worker-count-dependent** (the base's heavier 3-view decode parallelizes
better as workers scale), so report the config:

| config | base-random | base-episode | lance-random | lance-episode | faithful speedup |
| ------ | ----------- | ------------ | ------------ | ------------- | ---------------- |
| 4 workers / batch 8  | 43.2 | — | 108.0 | — | 2.50× |
| 8 workers / batch 16 | 92.4 | 95.4 | 195.5 | 177.4 | **1.86×** (episode) |

At a fixed config `base-random ≈ base-episode` — **shuffle mode is throughput-neutral locally**
(episode-shuffle's win is on S3, where it avoids re-fetching clips). The honest single-loader
action speedup at a realistic 8-worker config is **~1.9×**, not the 2.5× seen at 4 workers.

Equivalence: action/captions/idle bit-exact; video mean|Δ|≈4/255 (~1.6%, H.264 re-encode
loss only — the resize/concat is the base's exact op done once offline). Use the bit-exact
video-blob variant when strict parity is required; the composed variant when throughput matters.

# LanceDB vision-SFT dataloader — results (BridgeData2 synthetic captions)

Data: 200-clip subset of public `nvidia/BridgeData2-Subset-Synthetic-Captions`
(`sft_dataset_bridge/train`), at `/home/ubuntu/work/data/bridge_src` (105 MB; 97 MB of
mp4). Each clip is 256×256, 5 fps, 74–96 frames, with a structured `caption_json` + dense
`caption`. JSONL built with the repo's own `captions_to_sft_jsonl` logic (`min_frames=61`,
all 200 kept). Loader pair (the 3rd Lance dataloader):

* **base** `LocalSFTDataset` (`cosmos_framework/data/vfm/local_datasets/sft_local_dataset.py`)
  — a faithful **local map-style** stand-in for the shipped `SFTDataset` (an S3 `IterableDataset`).
  It reproduces `process_one_sample` verbatim: resolution sizing from `VIDEO_RES_SIZE_INFO`,
  `entire_chunk` window math, `ffmpeg_decode_video` decode+resize, temporal truncation to
  `4N+1`, structured-caption selection (`caption_json_to_prompt`), and `tokenize_caption`
  (Qwen2.5-7B + `add_special_tokens`). It is *not* S3/packing/sharding — that's the only
  part dropped; the per-sample compute is identical.
* **lance** `LanceVisionSFTDataset` (`cosmos_framework/data/lance/vision_sft_dataset.py`) —
  mirrors `LanceDROIDComposedDataset` exactly (worker-safe lazy lance handle, per-worker
  `torchcodec` LRU decoder cache, `seek_mode="approximate"`, batched `__getitems__`). The
  converter (`tools/lance_datagen/build_vision_sft.py`) decodes each clip once, resizes to
  training resolution (the base's exact resize), re-encodes all-intra (gop=1), and stores
  `{clip_id, sizing, caption_json, caption, video_bytes(blob-v2)}` — 110 MB. Per sample the
  loader applies the same window math + center-crop + temporal truncation + tokenize.

## Equivalence
`tests/data/lance/test_vision_sft_equivalence.py` — 7/7 pass. Over 40 clips: caption text
and **token ids exact** (40/40), video shape exact, video **mean|Δ|/255 = 0.013 (~1.3%,
H.264 re-encode loss only** — min 0.009, max 0.016). The resize is the base's exact op done
once offline; only the re-encode is lossy.

## Throughput — `bench_vision_sft.py` (CPU decode, shuffled RandomSampler, LOCAL, batch 8)

| workers | mode | base samples/s | lance samples/s | speedup |
| ------- | ---- | -------------- | --------------- | ------- |
| 4       | raw (video only)      | 33.2 | 225.3 | 6.79× |
| 8       | raw (video only)      | 63.4 | 431.4 | 6.80× |
| 4       | e2e (video+tokenize)  | 32.8 | 206.9 | 6.32× |
| 8       | e2e (video+tokenize)  | 61.9 | 401.3 | 6.49× |

The win (~6.5–6.8×) holds **end-to-end**, unlike the action loader (whose e2e collapsed to
~1× under heavy pose math): here the per-sample non-video work is just one chat-template
tokenize, which is cheap relative to video decode. Where the win comes from for single-view
video: the base seeks the source mp4 and runs a full ffmpeg decode+`scale` filter **per
sample every epoch**; the Lance loader decodes a clip that is **already at training
resolution** and **all-intra**, so it (1) decodes far fewer pixels (no on-the-fly resize),
(2) seeks cheaply (every frame a keyframe → approximate seek is exact), and (3) skips
process spawn for ffmpeg via the in-process torchcodec decoder + per-worker LRU cache, with
one batched `get_frames_at` per clip. Same encoded-video storage policy as the action
loader — no per-frame JPEG.
