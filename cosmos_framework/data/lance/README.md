# LanceDB-powered Cosmos dataloaders

Drop-in LanceDB replacements for the three dataloaders Cosmos mixes during training
(LeRobot action, WebDataset VLM, local vision-SFT), built to demonstrate higher
dataloading throughput and better scalability while preserving the training signal.

All comparisons below are **fair** (same decode device, same shuffle, same hardware) and
**measured** on a single node with 4× NVIDIA L40S / 48 CPU, reading **shuffled** as in
real training. Nothing here uses per-frame JPEG (disk blowup) — everything stays
video-encoded; the action/vision-SFT wins come from a one-time, offline, *lossy* re-encode
into a training-optimized layout.

## Results at a glance

| dataloader | base (cosmos) | Lance | speedup | bound by |
| ---------- | ------------- | ----- | ------- | -------- |
| action / lerobot (DROID) | `DROIDLeRobotDataset` | `LanceDROIDComposedDataset` | **2.0–2.5×** e2e | video decode |
| webdataset / VLM (LLaVA-OneVision) | `webdataset.WebLoader` | `LanceVLMShuffleScan` | **3.7× raw access** (≈1× e2e) | model-side image-proc |
| local vision-SFT (Bridge) | `SFTDataset` | `LanceVisionSFTDataset` | **6.5×** e2e | video decode |
| **combined (1:1:1 mix)** | all-base trio | all-Lance trio | **2.75× raw / 2.23× e2e** | the two video loaders |

Full numbers, methodology, and worker-scaling: [`RESULTS.md`](RESULTS.md).
The decode-bound optimization roadmap (incl. why NVDEC is *not* the win at these frame
sizes): [`OPTIMIZATION_ROADMAP.md`](OPTIMIZATION_ROADMAP.md).
Why the base loaders structurally can't capture these wins: [`WHY_BASE_CANT.md`](WHY_BASE_CANT.md).
Proof the optimized clips preserve the real training data (PSNR/SSIM, visual, content):
[`VALIDATION.md`](VALIDATION.md).

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
  same downstream tokenizer produces identical tensors. The raw access win is large
  (3.7–18×) but the end-to-end VLM step is gated by the Qwen image-processor, so the
  storage win doesn't surface e2e on a single node — it matters at object-store/multi-node
  scale and for true global shuffle.

### 3. Local vision-SFT — `vision_sft_dataset.py`
- **Base**: `SFTDataset` (faithful local stand-in `sft_local_dataset.py`) seeks the source
  mp4 per sample, decodes a window with an ffmpeg `scale` filter, and tokenizes the caption.
- **`LanceVisionSFTDataset`**: converter `tools/lance_datagen/build_vision_sft.py`
  re-encodes each clip to a pre-resized, all-intra per-clip blob; the loader decodes it
  (approximate seek, per-worker decoder cache) and tokenizes the same caption. Token ids
  **exact**; video PSNR ~37 dB. Win holds end-to-end (~6.5×) because the only non-video
  work is a cheap tokenize.

## Why this isn't doable/practical without LanceDB
See [`WHY_BASE_CANT.md`](WHY_BASE_CANT.md) for the full argument. In short:
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
Environment: Python 3.12 venv with `torch==2.10+cu128`, `torchvision`, `torchcodec` (+
`nvidia-npp-cu12` on `LD_LIBRARY_PATH`), `lancedb`/`pylance`, `lerobot`, `lerobot-lancedb`,
`webdataset`, `transformers`, system `ffmpeg`. Datasets are public on HF
(`lerobot/droid_1.0.1`, `lmms-lab/LLaVA-OneVision-Data`, `nvidia/BridgeData2-Subset-Synthetic-Captions`).

```bash
# action: prepare a Cosmos-canonical DROID subset, build the composed table, benchmark
python tools/lance_datagen/prepare_droid_subset.py --src <droid_1.0.1> --out <out> --num-episodes 100
python tools/lance_datagen/build_composed_droid.py --root <out>/success --uri <lance> --gop 1
DROID_COSMOS_ROOT=<out>/success DROID_LANCE_URI=<videoblob_lance> \
  pytest tests/data/lance/test_action_equivalence.py        # bit-exact equivalence
python benchmarks/lance/bench_action.py --root <out>/success --uri <lance> --modes base lance-composed

# vlm
python tools/lance_datagen/build_wds_shards.py --out <wds>          # base tar shards
python benchmarks/lance/bench_vlm.py --lance-uri <lance> --wds-shards "<wds>/shard-{00000..00019}.tar" --mode raw

# vision-sft
python tools/lance_datagen/build_vision_sft.py ...                  # see file args
python benchmarks/lance/bench_vision_sft.py ...

# combined
python benchmarks/lance/bench_combined.py
```

Layout: dataloaders in `cosmos_framework/data/lance/`, offline converters in
`tools/lance_datagen/`, benchmarks in `benchmarks/lance/`, equivalence tests in
`tests/data/lance/`.
