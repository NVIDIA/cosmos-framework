# LanceDB-powered Cosmos dataloaders

Drop-in LanceDB replacements for the three dataloaders Cosmos mixes during training
(LeRobot **action**, WebDataset **VLM**, local **vision-SFT**), built to demonstrate higher
dataloading throughput and better scalability while preserving the training signal. Output is
verified equivalent to the base loaders, so they're a faithful swap.

- **Full numbers** (all regimes, allocations, single-modality, e2e training): [`BENCHMARKS.md`](BENCHMARKS.md)
- **How each speedup works** (per-loader mechanisms): [`HOW_IT_WORKS.md`](HOW_IT_WORKS.md)
- **Run it on H100/H200/B200** (multi-GPU + the real 8B path): [`RUN_BENCHMARKS_H100.md`](RUN_BENCHMARKS_H100.md)
- **Reproduce from scratch**: [`REPRODUCE.md`](REPRODUCE.md)

## Headline

Combined 3-loader throughput, 327 DROID episodes, CPU decode both sides, RAW. Read it under three
framings (full tables + the per-loader and end-to-end-training numbers are in [`BENCHMARKS.md`](BENCHMARKS.md)):

workers shown as action/vlm/vsft (the per-loader DataLoader `num_workers`):

| comparison | LOCAL | full S3 | MIXED (realistic default) |
| ---------- | ----- | ------- | ------------------------- |
| base 4/4/4 vs lance 4/4/4 (cosmos default) | 2.85× | 3.76× | 3.79× |
| base 18/4/18 vs lance 18/4/18 (tuned) | 4.61× | 6.48× | 5.46× |
| **base 4/4/4 (as-shipped) vs lance 18/4/18 (tuned)** | **11.7×** | **19.0×** | **16.2×** |

Two compounding wins: the **Lance dataloaders** themselves, and **per-loader worker rebalancing** (Cosmos
ships a flat ~4 workers/loader and never rebalances toward the bottleneck — its "multiplex" is ratio-based
modality mixing, not worker allocation). The bottom row is the real out-of-the-box delta a user gets.

**Correctness:** action **8/8 bit-exact**, vision-SFT **7/7** (token-ids exact), VLM **3/3** (records
byte-identical) — `tests/data/lance/`. Throughput is only meaningful because the output matches.

**End-to-end training:** on a single GPU at a realistic model size, training is **compute-bound**, so the
dataloader is hidden and base ≈ lance wall-clock; the dataloader win surfaces when the pipeline is
**data-bound** (fast GPUs / many-GPU data-parallel / remote data). Details + the GPU-scaling argument and
the H100 runbook in [`BENCHMARKS.md`](BENCHMARKS.md) §5 and [`RUN_BENCHMARKS_H100.md`](RUN_BENCHMARKS_H100.md).

## What changed, per loader (mechanisms in [`HOW_IT_WORKS.md`](HOW_IT_WORKS.md))

- **Action / LeRobot** — `action_dataset.py`. Base decodes 3 camera views + resizes + concatenates per
  sample every epoch. `LanceDROIDComposedDataset` serves **one pre-composed, pre-resized, all-intra clip
  per episode** (the base's exact transform done once, offline) + a per-worker decoder cache. Action/labels
  bit-exact; video within H.264 re-encode tolerance. A bit-exact `LanceDROIDDataset` variant stores the raw
  mp4 bytes for strict parity.
- **WebDataset / VLM** — `vlm_dataset.py`. Base streams tar shards / HF-Hub with a bounded shuffle buffer,
  no random access. `LanceVLMDataset` gives O(1) random access + true global shuffle (Permutation API);
  `LanceVLMShuffleScan` is the object-storage pattern (fragment-shuffle + buffered columnar scan). Raw
  access wins big; end-to-end is gated by the image-processor (≈1× single-node).
- **Local vision-SFT** — `vision_sft_dataset.py`. Base spawns ffmpeg per sample to decode+resize.
  `LanceVisionSFTDataset` decodes a **pre-resized, all-intra per-clip** stream in-process (torchcodec +
  per-worker cache) and tokenizes the same caption. Token-ids exact; the win holds **end-to-end** (~6.5×).

Storage: clips are stored as **plain `large_binary`** (not blob-v2) — ~6× faster columnar reads on S3 for
<2 MB payloads; loaders auto-detect, converters default to `--storage plain`. No per-frame JPEG (the
composed clips are *0.35× the original* on disk, not a blowup).

## Why this isn't practical without LanceDB
- **Object-store-native**: the stock action/VLM loaders read the local filesystem only
  (`action/datasets/base_dataset.py:65`); cosmos's docs say pre-download to disk. Lance reads `s3://`
  natively, *enabling* efficient object-store training the base can't do without a FUSE mount or full download.
- **Structural**: true random access + global shuffle (a WebDataset tar is sequential-only; its shuffle is
  an approximate buffer), plus columnar / filtered reads.
- **The representation wins** require doing the per-epoch transform once, offline, and serving an indexed,
  versioned, object-store-native, shuffle-sampled multimodal store of clips — i.e. you'd be rebuilding Lance.

## Reproduce
Full recipe (env, downloads, conversions, S3 setup, expected numbers): [`REPRODUCE.md`](REPRODUCE.md).
Quick orientation — Python 3.12, `torch==2.10+cu128` / `torchvision` / `torchcodec` matched +
`nvidia-npp-cu12` on `LD_LIBRARY_PATH`, `lancedb`/`pylance`, `lerobot`, `webdataset`, `transformers`,
`datasets`, `boto3`, system `ffmpeg`. **`source benchmarks/lance/.venv-gpu/bin/activate`** (sets the
`LD_LIBRARY_PATH` torchcodec needs; do **not** use the stale `_env.sh`). Datasets are public on HF
(`lerobot/droid_1.0.1`, `lmms-lab/LLaVA-OneVision-Data`, `nvidia/BridgeData2-Subset-Synthetic-Captions`).

```bash
# build the optimized Lance tables (plain storage)
python tools/lance_datagen/build_composed_droid.py --root <droid>/success --uri <lance> --gop 1 --storage plain
python tools/lance_datagen/build_vision_sft.py --jsonl <bridge>/.../video_dataset_file.jsonl --uri <lance> --storage plain
# equivalence, then the full matrix + sweeps
pytest tests/data/lance/                              # equivalence (set the *_LANCE_URI / *_JSONL env vars)
bash benchmarks/lance/run_matrix.sh                   # LOCAL / S3 / MIXED × {4/4/4, optimal} × {base, lance}
python benchmarks/lance/train_combined_e2e.py --trio lance --regime mixed --layers 8 …   # e2e training
```

Layout: dataloaders in `cosmos_framework/data/lance/`, offline converters in `tools/lance_datagen/`,
benchmarks in `benchmarks/lance/`, equivalence tests in `tests/data/lance/`.
