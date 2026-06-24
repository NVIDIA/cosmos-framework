# LanceDB-powered Cosmos dataloaders

Drop-in LanceDB replacements for the three dataloaders Cosmos mixes during training
(LeRobot action, WebDataset VLM, local vision-SFT), built to demonstrate higher
dataloading throughput and better scalability while preserving the training signal.

All comparisons below are **fair**: same decode device (**CPU decode on both sides** — the base
can only decode on CPU), same hardware (single node, 48 CPU), and the base's **production shuffle**
(for action that is *episode-shuffle*, `iterable_shuffle=True`, not RandomSampler). RAW =
data-access + decode, no model. Nothing uses per-frame JPEG (disk blowup); the action/vision-SFT
wins come from a one-time, offline, *lossy* re-encode into a training-optimized layout.

**Two storage regimes, because they answer different questions.** Cosmos's documented workflow
**downloads datasets to local disk, then trains** (every public NVIDIA post-training guide), so
**LOCAL is the apples-to-apples comparison**. S3 is Lance's *additional* value: it reads object
storage **natively**, which the stock action/VLM loaders cannot do at all (they only read
`Path(root)`; only the vision-SFT `SFTDataset` has a boto3 reader). For the S3 row the base
accesses each dataset the way the stock loader actually would — action/VLM via an s3fs FUSE
mount (the only option), vision-SFT via boto3 download-per-sample.

## Results at a glance

327 DROID episodes, 1:1:1 round-robin mixer, 6 workers/loader, batch 16, CPU decode both sides.
Reproduce: `benchmarks/lance/bench_combined_faithful.py` (run `--trios base` and `--trios lance`
in **separate** processes). Per-loader detail: [`RESULTS.md`](RESULTS.md).

**LOCAL — apples-to-apples, cosmos's real workflow:**

| loader (RAW) | base (cosmos) | Lance | speedup |
| ------------ | ------------- | ----- | ------- |
| action / DROID (episode-shuffle both sides) | 62.2 | 119.9 | **1.93×** |
| webdataset / VLM (LLaVA-OneVision) | 21,918 | 35,728 | **1.63×** (raw; ≈1× e2e) |
| local vision-SFT (Bridge) | 41.9 | 317.3 | **7.57×** |
| **combined (1:1:1 mixer)** | **122.1** | **379.5** | **3.11×** |

**S3 — Lance native `s3://` vs stock base access (`LANCE_IO_THREADS=256`):**

| loader (RAW) | base (stock S3 access) | Lance | speedup |
| ------------ | ---------------------- | ----- | ------- |
| action / DROID (base via FUSE) | 73.8 | 126.4 | **1.71×** |
| webdataset / VLM (base via FUSE) | 18,838 | 32,097 | **1.70×** |
| vision-SFT (base via boto3) | 31.4 | 83.4 | **2.66×** |
| **combined (1:1:1 mixer)** | **95.5** | **251.9** | **2.64×** |

**DEFAULT-MIXED — each loader on its *actual* default storage** (the most realistic single number):
base → action LOCAL, vision-SFT S3 (boto3), VLM HF-Hub streaming; Lance → action LOCAL, vision-SFT S3, VLM S3.

| loader (RAW) | base (default) | Lance | speedup |
| ------------ | -------------- | ----- | ------- |
| action / DROID (both local) | 81.6 | 138.6 | **1.70×** |
| VLM (base: HF-Hub stream · Lance: S3 scan) | 901 | 39,428 | 43.7×† |
| vision-SFT (base: boto3 S3 · Lance: S3) | 39.1 | 98.2 | **2.51×** |
| **combined (1:1:1 mixer)** | **95.3** | **253.5** | **2.66×** |

†The 43.7× VLM number compares the base's HF-Hub *streaming* (decodes PIL over the network) vs Lance's
S3 columnar byte-scan — different work, and VLM is never the mixer bottleneck (it's ~10–400× faster than
the video loaders), so it doesn't move the combined number. The combined is gated by the video loaders.

**How to read the combined number.** The 1:1:1 mixer aggregate is **gated by the slowest loader**
(aggregate ≈ 3×slowest — verified: local 379≈3×120, S3 252≈3×83). So the combined "speedup" tracks
whichever loader bottlenecks each trio; it is *not* a multiplicative win across loaders. The honest
combined dataloader speedup is **~3× (local) / ~2.6× (S3)** — consistent across regimes and with the
per-loader wins. (An earlier draft reported **8.5× from S3**; that was an artifact of benchmarking the
vision-SFT base through a FUSE mount at 11.2 samples/s. The *stock* base downloads via boto3 at 31.4,
which collapses the combined to the honest 2.64×. Lesson recorded in [`RESULTS.md`](RESULTS.md).)

> **Action 2×2 (the speedup is worker-count-dependent, not shuffle-mode-dependent).** Early drafts
> cited 2.5× — that was at **4 workers / batch 8**. At a fixed 8-worker config (local, CPU decode):
> `base-random 92.4 / base-episode 95.4 / lance-random 195.5 / lance-episode 177.4`. So `base-random`
> ≈ `base-episode` — **shuffle mode is throughput-neutral locally** (episode-shuffle's win shows up on
> S3, avoiding clip re-fetch); the ratio drops from 2.5×→~1.9× because the base's heavier 3-view decode
> parallelizes better as workers scale. Reproduce: `bench_action_faithful.py --modes base-random
> base-episode lance-random lance-episode`.

Full numbers, methodology, and worker-scaling: [`RESULTS.md`](RESULTS.md).
Proof the optimized clips preserve the real training data (token-exact labels, PSNR/SSIM,
training-output equivalence): [`VALIDATION.md`](VALIDATION.md).

## Disk footprint (action loader) — the pre-composed clips are *smaller*, not bigger

A common worry: doesn't re-encoding (especially all-intra gop=1) blow up disk? Measured on
the DROID subset and extrapolated to full DROID (27.6M frames, 3 views). Fusing 3 views → 1
half-resolution clip more than offsets the all-intra penalty, so even gop=1 is **0.35× the
original** — and nowhere near the per-frame-JPEG option we rejected.

| storage | KB/frame | full-DROID est. | vs original |
| ------- | -------- | --------------- | ----------- |
| original 3-view long-GOP (320×180 ×3) | 16.3 | ~450 GB | 1.00× |
| **composed gop=1 (shipped)** | **5.7** | **~160 GB** | **0.35×** |
| composed gop=2 | 4.7 | ~131 GB | 0.29× |
| composed gop=8 | 2.8 | ~80 GB | 0.18× |
| composed gop=30 | 2.5 | ~69 GB | 0.15× |
| ~~per-frame JPEG q95~~ (rejected — disk blowup) | 29.3 | ~828 GB | 1.8× |

Concretely on the 100-episode subset: composed gop=1 = **162 MB** vs ~459 MB of equivalent
original 3-view footage. gop=1 gives the fastest random-window seek (every frame a keyframe);
gop=2–8 roughly halves disk again for a small decode cost since training windows are
contiguous runs. Derivation in [`VALIDATION.md`](VALIDATION.md).

## What changed, per loader, and how it was built

### 1. Action / LeRobot — `action_dataset.py`
- **Base bottleneck**: `DROIDLeRobotDataset.__getitem__` decodes 3 camera mp4 views,
  resizes the 2 exteriors to half, and concatenates → one (3,T,270,320) tensor, **per
  sample every epoch** (~98% of per-sample time is this video work).
- **`LanceDROIDDataset`** (bit-exact): stores the original mp4 bytes as Lance blob-v2 and
  decodes with the same torchcodec path → byte-identical frames. Used by the equivalence
  test. Modest fair speedup (decode is unchanged).
- **`LanceDROIDComposedDataset`** (the throughput win): the converter
  `tools/lance_datagen/build_composed_droid.py` does the base's *exact* resize+concat
  **once, offline**, and stores ONE 270×320 all-intra (gop=1) clip per episode as a
  blob-v2 row. The loader then decodes a single half-resolution stream (approximate seek +
  per-worker LRU decoder cache, batched `__getitems__`) instead of 3 views + resize +
  concat. Inherits all index/pose/action logic from the base, so action labels are
  **bit-exact**; video differs only by the H.264 re-encode (PSNR 32 dB).

### 2. WebDataset / VLM — `vlm_dataset.py`
- **Base**: `webdataset.WebLoader` streams tar shards sequentially with a bounded shuffle
  buffer — no random access, re-streams every epoch.
- **`LanceVLMDataset`**: Permutation-API map-style random access (one `__getitems__` =
  batched random read). Fast locally; on S3 random point reads are latency-bound.
- **`LanceVLMShuffleScan`**: chunked-shuffle scan (fragment-order shuffle + buffer) — the
  right pattern for shuffled reads from object storage; bandwidth-bound, beats sequential
  tar at low/moderate worker counts. Converter: `tools/lance_datagen/build_wds_shards.py`
  (writes the comparison tar shards) + `convert_llava_to_lance` (the Lance table; stores
  original PNG bytes inline, no re-encode). Output dict matches the base raw record, so the
  same downstream tokenizer produces identical tensors. The raw-access win is large at big
  batches (up to ~22× at batch 16384) but ~1.6–1.7× at a training batch of 16; either way the
  end-to-end VLM step is gated by the Qwen image-processor (≈1× e2e on a single node). It
  matters at object-store/multi-node scale and for true global shuffle.

### 3. Local vision-SFT — `vision_sft_dataset.py`
- **Base**: `SFTDataset` (faithful local stand-in `sft_local_dataset.py`) seeks the source
  mp4 per sample, decodes a window with an ffmpeg `scale` filter, and tokenizes the caption.
- **`LanceVisionSFTDataset`**: converter `tools/lance_datagen/build_vision_sft.py`
  re-encodes each clip to a pre-resized, all-intra per-clip blob; the loader decodes it
  (approximate seek, per-worker decoder cache) and tokenizes the same caption. Token ids
  **exact**; video PSNR ~37 dB. Win holds end-to-end (~6.5×) because the only non-video
  work is a cheap tokenize.

## Why this isn't doable/practical without LanceDB
- **Object-store-native (Lance-only)**: the stock cosmos action and VLM loaders read
  `Path(root)` / `data_root` on the **local filesystem only** — no S3 reader (verified:
  `action/datasets/base_dataset.py:65`). cosmos's docs tell you to pre-download to local
  disk. Lance reads `s3://` natively (batched `take_blobs` + concurrency), so it *enables*
  efficient object-store training the base can't do without a FUSE mount or full download.
- **Structural (Lance-only)**: true random access + global shuffle (a WebDataset tar is
  sequential-only; its shuffle is an approximate buffer), columnar/filtered reads, and
  blob-v2 byte-range reads from object storage.
- **The representation wins** (the 2–6.5× video speedups) require doing the
  transform once, offline, and serving an indexed, versioned, object-store-native,
  shuffle-sampled, multimodal store of per-episode clips — i.e. you'd be rebuilding Lance.
  The base loaders are bound to the canonical LeRobot/WebDataset formats and recompute the
  transform every epoch; Lance is the substrate that makes the offline-optimized
  representation a first-class, queryable, versioned dataset.

## Reproduce / verify independently
**→ Full step-by-step recipe (exact env, dataset downloads, conversions, S3 setup, all three
benchmark regimes, and expected numbers): [`REPRODUCE.md`](REPRODUCE.md).** Start there.

Quick orientation — Python 3.12 venv with `torch==2.10+cu128`, `torchvision==0.25+cu128`,
`torchcodec==0.10+cu128` (+ `nvidia-npp-cu12` on `LD_LIBRARY_PATH`), `lancedb`/`pylance`,
`lerobot`, `webdataset`, `transformers`, `datasets`, `boto3`, system `ffmpeg`. `source
benchmarks/lance/_env.sh` sets the `LD_LIBRARY_PATH` torchcodec needs. Datasets are public on HF
(`lerobot/droid_1.0.1`, `lmms-lab/LLaVA-OneVision-Data`, `nvidia/BridgeData2-Subset-Synthetic-Captions`).

```bash
source benchmarks/lance/_env.sh
# action: prepare a Cosmos-canonical DROID subset, build the composed table, benchmark
python tools/lance_datagen/prepare_droid_subset.py --src <droid_1.0.1> --out <out> --num-episodes 100
python tools/lance_datagen/build_composed_droid.py --root <out>/success --uri <lance> --gop 1
DROID_COSMOS_ROOT=<out>/success DROID_LANCE_URI=<videoblob_lance> \
  pytest tests/data/lance/test_action_equivalence.py        # bit-exact equivalence
# action 2x2 (random vs episode-shuffle, both sides):
python benchmarks/lance/bench_action_faithful.py --root <out>/success --uri <lance_dir> \
  --modes base-random base-episode lance-random lance-episode

# vlm / vision-sft per-loader
python benchmarks/lance/bench_vlm.py --lance-uri <lance> --wds-shards "<wds>/shard-{00000..00019}.tar" --mode raw
python benchmarks/lance/bench_vision_sft.py ...

# combined (LOCAL = apples-to-apples; run base and lance in SEPARATE processes)
python benchmarks/lance/bench_combined_faithful.py --action-root ... --action-uri ... \
  --vlm-wds ... --vlm-uri ... --vsft-jsonl ... --vsft-uri ... --trios base
python benchmarks/lance/bench_combined_faithful.py ... --trios lance
# combined (S3): add --region us-east-2, s3:// uris, and --vsft-s3-bucket/--vsft-s3-prefix
# (stock boto3 vsft base); set LANCE_IO_THREADS=256.
```

Layout: dataloaders in `cosmos_framework/data/lance/`, offline converters in
`tools/lance_datagen/`, benchmarks in `benchmarks/lance/`, equivalence tests in
`tests/data/lance/`.
